"""Generate embeddings and store in LanceDB."""

from pathlib import Path

import lancedb
from sentence_transformers import SentenceTransformer

from ..logging import get_logger
from .chunker import Chunk

logger = get_logger("embedder")

# Default embedding model - good balance of quality and speed
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embed_and_store(
    chunks: list[Chunk],
    db_path: Path,
    model_name: str = DEFAULT_MODEL,
    table_name: str = "ethereum_specs",
) -> int:
    """
    Generate embeddings for chunks and store in LanceDB.

    Args:
        chunks: List of document chunks
        db_path: Path to LanceDB database
        model_name: Sentence transformer model name
        table_name: Name of table to create/update

    Returns:
        Number of chunks embedded and stored
    """
    if not chunks:
        logger.warning("No chunks to embed")
        return 0

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Generating embeddings for %d chunks...", len(chunks))
    texts = [chunk.content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Prepare records for LanceDB
    records = []
    for chunk, embedding in zip(chunks, embeddings):
        record = {
            "content": chunk.content,
            "source": chunk.source,
            "fork": chunk.fork or "",
            "section": chunk.section or "",
            "chunk_type": chunk.chunk_type,
            "vector": embedding.tolist(),
            # Flatten metadata - specs/EIPs
            "eip": chunk.metadata.get("eip", ""),
            "title": chunk.metadata.get("title", ""),
            "function_name": chunk.metadata.get("function_name", ""),
            "h1": chunk.metadata.get("h1", ""),
            "h2": chunk.metadata.get("h2", ""),
            "h3": chunk.metadata.get("h3", ""),
            # Client code metadata
            "client": chunk.metadata.get("client", ""),
            "language": chunk.metadata.get("language", ""),
        }
        records.append(record)

    # Store in LanceDB
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_path))

    # Drop existing table if it exists
    try:
        db.drop_table(table_name)
    except Exception as e:
        logger.debug("Table %s does not exist or could not be dropped: %s", table_name, e)

    logger.info("Storing %d records in LanceDB...", len(records))
    table = db.create_table(table_name, records)

    # Create vector index for fast search
    logger.info("Creating vector index...")
    table.create_index(metric="cosine", num_partitions=16, num_sub_vectors=32)

    logger.info("Successfully stored %d chunks in %s", len(records), db_path)
    return len(records)


class EmbeddingSearcher:
    """Search interface for embedded documents."""

    def __init__(
        self,
        db_path: Path,
        model_name: str = DEFAULT_MODEL,
        table_name: str = "ethereum_specs",
    ):
        self.db = lancedb.connect(str(db_path))
        self.table = self.db.open_table(table_name)
        self.model = SentenceTransformer(model_name)

    def search(
        self,
        query: str,
        limit: int = 5,
        fork: str | None = None,
        chunk_type: str | None = None,
    ) -> list[dict]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            limit: Maximum results to return
            fork: Filter by fork name
            chunk_type: Filter by chunk type

        Returns:
            List of matching chunks with scores
        """
        query_embedding = self.model.encode(query).tolist()

        # Fetch extra results if filtering, then filter in Python (safer than SQL injection surface)
        fetch_limit = limit * 2 if (fork or chunk_type) else limit
        results = self.table.search(query_embedding).limit(fetch_limit).to_list()

        # Filter in Python - no SQL injection risk
        matches = []
        for r in results:
            if fork and r.get("fork") != fork:
                continue
            if chunk_type and r.get("chunk_type") != chunk_type:
                continue

            matches.append({
                "content": r["content"],
                "source": r["source"],
                "fork": r["fork"],
                "section": r["section"],
                "chunk_type": r["chunk_type"],
                "score": 1 - r["_distance"],  # Convert distance to similarity
                "eip": r.get("eip"),
                "function_name": r.get("function_name"),
                "client": r.get("client"),
                "language": r.get("language"),
            })

            if len(matches) >= limit:
                break

        return matches

    def search_constant(self, constant_name: str) -> list[dict]:
        """Search specifically for a constant definition."""
        return self.search(
            f"{constant_name} constant value",
            limit=10,
            chunk_type="constant",
        )

    def search_function(self, function_name: str, fork: str | None = None) -> list[dict]:
        """Search specifically for a function implementation."""
        return self.search(
            f"def {function_name}",
            limit=5,
            fork=fork,
            chunk_type="function",
        )

    def search_eip(self, query: str) -> list[dict]:
        """Search only EIPs."""
        return self.search(query, limit=10, chunk_type="eip")
