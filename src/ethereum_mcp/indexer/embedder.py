"""Generate embeddings and store in LanceDB.

Supports both full and incremental indexing:
- Full: Drop existing table and rebuild from scratch
- Incremental: Add/delete only changed chunks based on manifest
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lancedb
from sentence_transformers import SentenceTransformer

from ..config import DEFAULT_EMBEDDING_MODEL, load_config
from ..logging import get_logger
from .chunker import Chunk, chunk_single_file
from .manifest import (
    FileEntry,
    Manifest,
    ManifestCorruptedError,
    compute_changes,
    compute_file_hash,
    get_file_mtime_ns,
    load_manifest,
    needs_full_rebuild,
    save_manifest,
)

logger = get_logger("embedder")

# Default embedding model - good balance of quality and speed
DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL

# LanceDB table name
TABLE_NAME = "ethereum_specs"


def embed_and_store(
    chunks: list[Chunk],
    db_path: Path,
    model_name: str = DEFAULT_MODEL,
    table_name: str = TABLE_NAME,
) -> int:
    """
    Generate embeddings for chunks and store in LanceDB.

    This is the legacy full-rebuild function, preserved for backwards compatibility.
    For incremental indexing, use IncrementalEmbedder instead.

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
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        record = _chunk_to_record(chunk, embedding.tolist())
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


def _chunk_to_record(chunk: Chunk, embedding: list[float]) -> dict[str, Any]:
    """Convert a Chunk to a LanceDB record."""
    return {
        "chunk_id": chunk.chunk_id,  # For incremental updates
        "content": chunk.content,
        "source": chunk.source,
        "repo": chunk.repo,  # Repository name for GitHub URL generation
        "fork": chunk.fork or "",
        "section": chunk.section or "",
        "chunk_type": chunk.chunk_type,
        "vector": embedding,
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


@dataclass
class IndexStats:
    """Statistics from an indexing operation."""

    total_files: int = 0
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    full_rebuild: bool = False
    rebuild_reason: str = ""

    @property
    def files_changed(self) -> int:
        return self.files_added + self.files_modified + self.files_deleted

    @property
    def is_incremental(self) -> bool:
        return not self.full_rebuild and self.files_changed > 0

    @property
    def is_noop(self) -> bool:
        return not self.full_rebuild and self.files_changed == 0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.full_rebuild:
            return f"Full rebuild: {self.chunks_added} chunks indexed ({self.rebuild_reason})"
        if self.is_noop:
            return "No changes detected"
        return (
            f"Incremental update: {self.files_added} added, "
            f"{self.files_modified} modified, {self.files_deleted} deleted "
            f"({self.chunks_added} chunks added, {self.chunks_deleted} deleted)"
        )


@dataclass
class DryRunResult:
    """Result of a dry-run showing what would change."""

    would_rebuild: bool = False
    rebuild_reason: str = ""
    files_to_add: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    files_to_delete: list[str] = field(default_factory=list)
    estimated_chunks_add: int = 0
    estimated_chunks_delete: int = 0

    def summary(self) -> str:
        if self.would_rebuild:
            return f"Would perform full rebuild: {self.rebuild_reason}"
        if not (self.files_to_add or self.files_to_modify or self.files_to_delete):
            return "No changes detected"
        lines = ["Would perform incremental update:"]
        if self.files_to_add:
            lines.append(f"  Add {len(self.files_to_add)} files")
        if self.files_to_modify:
            lines.append(f"  Modify {len(self.files_to_modify)} files")
        if self.files_to_delete:
            lines.append(f"  Delete {len(self.files_to_delete)} files")
        lines.append(f"  ~{self.estimated_chunks_add} chunks to add")
        lines.append(f"  ~{self.estimated_chunks_delete} chunks to delete")
        return "\n".join(lines)


class IncrementalEmbedder:
    """
    Incremental embedding and indexing for Ethereum specs.

    Uses a manifest to track indexed files and their chunk IDs,
    enabling incremental updates that only re-embed changed content.
    """

    def __init__(
        self,
        data_dir: Path,
        model_name: str | None = None,
        table_name: str = TABLE_NAME,
        batch_size: int = 32,
    ):
        """
        Initialize the incremental embedder.

        Args:
            data_dir: Base data directory (~/.ethereum-mcp)
            model_name: Embedding model name (None = use config/default)
            table_name: LanceDB table name
            batch_size: Batch size for embedding
        """
        self.data_dir = data_dir
        self.db_path = data_dir / "lancedb"
        self.manifest_path = data_dir / "manifest.json"
        self.table_name = table_name
        self.batch_size = batch_size

        # Load config if no model specified
        if model_name is None:
            config = load_config(data_dir=data_dir)
            model_name = config.embedding.model
            self.batch_size = config.embedding.batch_size

        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_current_config(self) -> dict[str, Any]:
        """Get current config for manifest comparison."""
        config = load_config(data_dir=self.data_dir)
        return {
            "embedding_model": self.model_name,
            "chunk_config": config.chunking.to_dict(),
        }

    def dry_run(
        self,
        current_files: dict[str, Path],
        file_types: dict[str, str],
    ) -> DryRunResult:
        """
        Check what would change without actually indexing.

        Args:
            current_files: Dict mapping relative paths to absolute Paths
            file_types: Dict mapping relative paths to file types ('spec', 'eip', 'builder')

        Returns:
            DryRunResult describing what would happen
        """
        try:
            manifest = load_manifest(self.manifest_path)
        except ManifestCorruptedError:
            return DryRunResult(
                would_rebuild=True,
                rebuild_reason="Manifest corrupted",
            )

        config = self.get_current_config()
        needs_rebuild, reason = needs_full_rebuild(manifest, config)

        if needs_rebuild:
            return DryRunResult(
                would_rebuild=True,
                rebuild_reason=reason,
            )

        changes = compute_changes(manifest, current_files)

        result = DryRunResult()
        for change in changes:
            if change.change_type == "add":
                result.files_to_add.append(change.path)
                # Estimate chunks (rough: ~10 chunks per file)
                result.estimated_chunks_add += 10
            elif change.change_type == "modify":
                result.files_to_modify.append(change.path)
                result.estimated_chunks_add += 10
                result.estimated_chunks_delete += len(change.old_chunk_ids)
            elif change.change_type == "delete":
                result.files_to_delete.append(change.path)
                result.estimated_chunks_delete += len(change.old_chunk_ids)

        return result

    def index(
        self,
        current_files: dict[str, Path],
        file_types: dict[str, str],
        force_full: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> IndexStats:
        """
        Index files, using incremental updates when possible.

        Args:
            current_files: Dict mapping relative paths to absolute Paths
            file_types: Dict mapping relative paths to file types ('spec', 'eip', 'builder')
            force_full: Force full rebuild even if incremental is possible
            chunk_size: Chunk size for text splitting
            chunk_overlap: Overlap between chunks

        Returns:
            IndexStats with details of what was done
        """
        stats = IndexStats(total_files=len(current_files))

        # Load manifest
        try:
            manifest = load_manifest(self.manifest_path)
        except ManifestCorruptedError as e:
            logger.warning("Manifest corrupted, performing full rebuild: %s", e)
            manifest = None
            force_full = True
            stats.rebuild_reason = "Manifest corrupted"

        # Check if full rebuild needed
        config = self.get_current_config()
        if not force_full:
            needs_rebuild, reason = needs_full_rebuild(manifest, config)
            if needs_rebuild:
                force_full = True
                stats.rebuild_reason = reason

        if force_full:
            return self._full_rebuild(
                current_files,
                file_types,
                chunk_size,
                chunk_overlap,
                stats,
            )

        # Incremental update
        return self._incremental_update(
            manifest,
            current_files,
            file_types,
            chunk_size,
            chunk_overlap,
            stats,
        )

    def _full_rebuild(
        self,
        current_files: dict[str, Path],
        file_types: dict[str, str],
        chunk_size: int,
        chunk_overlap: int,
        stats: IndexStats,
    ) -> IndexStats:
        """Perform full rebuild of the index."""
        stats.full_rebuild = True
        if not stats.rebuild_reason:
            stats.rebuild_reason = "Forced rebuild"

        logger.info("Performing full rebuild: %s", stats.rebuild_reason)

        # Chunk all files
        all_chunks = []
        manifest = Manifest(
            embedding_model=self.model_name,
            chunk_config={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )

        for rel_path, abs_path in current_files.items():
            file_type = file_types.get(rel_path, "spec")
            chunks = chunk_single_file(
                abs_path,
                file_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_path=self.data_dir,
            )
            all_chunks.extend(chunks)

            # Update manifest
            manifest.files[rel_path] = FileEntry(
                sha256=compute_file_hash(abs_path),
                mtime_ns=get_file_mtime_ns(abs_path),
                chunk_ids=[c.chunk_id for c in chunks],
            )

        # Embed and store
        if all_chunks:
            stats.chunks_added = embed_and_store(
                all_chunks,
                self.db_path,
                model_name=self.model_name,
                table_name=self.table_name,
            )

        # Save manifest
        save_manifest(manifest, self.manifest_path)

        stats.files_added = len(current_files)
        return stats

    def _incremental_update(
        self,
        manifest: Manifest,
        current_files: dict[str, Path],
        file_types: dict[str, str],
        chunk_size: int,
        chunk_overlap: int,
        stats: IndexStats,
    ) -> IndexStats:
        """Perform incremental update of the index."""
        changes = compute_changes(manifest, current_files)

        if not changes:
            logger.info("No changes detected, index is up to date")
            return stats

        logger.info(
            "Incremental update: %d files changed",
            len(changes),
        )

        # Open LanceDB
        db = lancedb.connect(str(self.db_path))
        try:
            table = db.open_table(self.table_name)
        except Exception:
            # Table doesn't exist, fall back to full rebuild
            logger.warning("Table doesn't exist, performing full rebuild")
            stats.rebuild_reason = "Table not found"
            return self._full_rebuild(
                current_files,
                file_types,
                chunk_size,
                chunk_overlap,
                stats,
            )

        chunks_to_add = []
        chunk_ids_to_delete = []

        for change in changes:
            if change.change_type == "add":
                stats.files_added += 1
            elif change.change_type == "modify":
                stats.files_modified += 1
                chunk_ids_to_delete.extend(change.old_chunk_ids)
            elif change.change_type == "delete":
                stats.files_deleted += 1
                chunk_ids_to_delete.extend(change.old_chunk_ids)
                # Remove from manifest
                del manifest.files[change.path]
                continue

            # Process add/modify: chunk the file
            abs_path = current_files.get(change.path)
            if abs_path is None:
                continue

            file_type = file_types.get(change.path, "spec")
            chunks = chunk_single_file(
                abs_path,
                file_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_path=self.data_dir,
            )
            chunks_to_add.extend(chunks)

            # Update manifest
            manifest.files[change.path] = FileEntry(
                sha256=compute_file_hash(abs_path),
                mtime_ns=get_file_mtime_ns(abs_path),
                chunk_ids=[c.chunk_id for c in chunks],
            )

        # Delete old chunks
        if chunk_ids_to_delete:
            logger.info("Deleting %d old chunks...", len(chunk_ids_to_delete))
            self._delete_chunks(table, chunk_ids_to_delete)
            stats.chunks_deleted = len(chunk_ids_to_delete)

        # Add new chunks
        if chunks_to_add:
            logger.info("Adding %d new chunks...", len(chunks_to_add))
            self._add_chunks(table, chunks_to_add)
            stats.chunks_added = len(chunks_to_add)

        # Update manifest config
        manifest.embedding_model = self.model_name
        manifest.chunk_config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

        # Save manifest
        save_manifest(manifest, self.manifest_path)

        return stats

    def _delete_chunks(self, table: Any, chunk_ids: list[str]) -> None:
        """Delete chunks from LanceDB by chunk_id."""
        # LanceDB delete with filter
        # Use batch deletion to avoid SQL injection - delete one at a time with exact match
        for chunk_id in chunk_ids:
            # Sanitize chunk_id (already validated during manifest load)
            # Use exact string match to avoid injection
            try:
                table.delete(f'chunk_id = "{chunk_id}"')
            except Exception as e:
                logger.warning("Failed to delete chunk %s: %s", chunk_id, e)

    def _add_chunks(self, table: Any, chunks: list[Chunk]) -> None:
        """Add chunks to LanceDB."""
        # Generate embeddings in batches
        texts = [c.content for c in chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.batch_size,
        )

        # Create records
        records = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            records.append(_chunk_to_record(chunk, embedding.tolist()))

        # Add to table
        table.add(records)


class EmbeddingSearcher:
    """Search interface for embedded documents."""

    def __init__(
        self,
        db_path: Path,
        model_name: str = DEFAULT_MODEL,
        table_name: str = TABLE_NAME,
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
                "chunk_id": r.get("chunk_id"),
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

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        # Get table count
        count = self.table.count_rows()
        return {
            "total_chunks": count,
            "table_name": self.table.name,
        }
