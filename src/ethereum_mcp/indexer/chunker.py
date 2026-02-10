"""Chunk markdown documents and client code for embedding."""

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from .client_compiler import ExtractedConstant, ExtractedItem
    from .rust_compiler import ExtractedConstant as RustConstant
    from .rust_compiler import ExtractedItem as RustItem


def generate_chunk_id(
    project: str,
    source_type: str,
    source_file: str,
    chunk_index: int,
    content: str,
) -> str:
    """
    Generate a unique, deterministic chunk ID.

    Format: {project}_{source_type}_{path_hash}_{index:04d}_{content_hash}

    The content hash ensures that if content changes, the ID changes,
    enabling proper delta updates.

    Args:
        project: Project identifier (e.g., "eth")
        source_type: Type of source (e.g., "spec", "eip", "function")
        source_file: Relative path to source file
        chunk_index: Index of this chunk within the file
        content: Chunk content for hashing

    Returns:
        Unique chunk ID string
    """
    path_hash = hashlib.sha256(source_file.encode()).hexdigest()[:8]
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{project}_{source_type}_{path_hash}_{chunk_index:04d}_{content_hash}"


@dataclass(frozen=True)
class Chunk:
    """A document chunk with metadata."""

    content: str
    source: str  # Repo-relative file path (e.g., specs/electra/beacon-chain.md)
    fork: str | None  # Fork name if from specs
    section: str | None  # Section header
    chunk_type: str  # 'spec', 'eip', 'function', 'constant'
    metadata: dict
    repo: str = ""  # Repository name (consensus-specs, EIPs, builder-specs)
    chunk_id: str = ""  # Unique ID for incremental indexing


def chunk_documents(
    spec_files: list[Path],
    eip_files: list[Path],
    builder_spec_files: list[Path] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    generate_ids: bool = False,
    base_path: Path | None = None,
    specs_base: Path | None = None,
    eips_base: Path | None = None,
    builder_specs_base: Path | None = None,
) -> list[Chunk]:
    """
    Chunk spec and EIP documents for embedding.

    Args:
        spec_files: List of spec markdown files
        eip_files: List of EIP markdown files
        builder_spec_files: List of builder-specs markdown files
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        generate_ids: If True, generate unique chunk IDs for incremental indexing
        base_path: Base path for relative paths in chunk IDs (required if generate_ids=True)
        specs_base: Base path for consensus-specs (for relative paths)
        eips_base: Base path for EIPs (for relative paths)
        builder_specs_base: Base path for builder-specs (for relative paths)

    Returns:
        List of chunks with metadata (and chunk_id if generate_ids=True)
    """
    chunks = []
    builder_spec_files = builder_spec_files or []

    # Headers to split on for markdown
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Process spec files
    for spec_file in spec_files:
        fork = _extract_fork_from_path(spec_file)
        chunks.extend(_chunk_spec_file(
            spec_file, fork, md_splitter, text_splitter,
            base_path=specs_base, repo="consensus-specs"
        ))

    # Process EIP files
    for eip_file in eip_files:
        chunks.extend(_chunk_eip_file(
            eip_file, md_splitter, text_splitter,
            base_path=eips_base, repo="EIPs"
        ))

    # Process builder-specs files
    for builder_file in builder_spec_files:
        chunks.extend(_chunk_builder_spec_file(
            builder_file, md_splitter, text_splitter,
            base_path=builder_specs_base, repo="builder-specs"
        ))

    # Generate chunk IDs if requested
    if generate_ids:
        chunks = _assign_chunk_ids(chunks, base_path)

    return chunks


def _assign_chunk_ids(chunks: list[Chunk], base_path: Path | None = None) -> list[Chunk]:
    """
    Assign unique chunk IDs to chunks, grouped by source file.

    Args:
        chunks: List of chunks without IDs
        base_path: Base path for making source paths relative

    Returns:
        New list of chunks with chunk_id populated
    """
    # Group chunks by source file to assign sequential indices
    from collections import defaultdict

    file_chunks: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        file_chunks[chunk.source].append((idx, chunk))

    import contextlib

    # Assign IDs
    new_chunks = [None] * len(chunks)
    for source, indexed_chunks in file_chunks.items():
        # Make path relative if base_path provided
        rel_source = source
        if base_path:
            with contextlib.suppress(ValueError):
                rel_source = str(Path(source).relative_to(base_path))

        for chunk_idx, (original_idx, chunk) in enumerate(indexed_chunks):
            chunk_id = generate_chunk_id(
                project="eth",
                source_type=chunk.chunk_type,
                source_file=rel_source,
                chunk_index=chunk_idx,
                content=chunk.content,
            )
            # Create new chunk with ID (Chunk is frozen)
            new_chunk = Chunk(
                content=chunk.content,
                source=chunk.source,
                fork=chunk.fork,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata,
                chunk_id=chunk_id,
            )
            new_chunks[original_idx] = new_chunk

    return new_chunks


def chunk_single_file(
    file_path: Path,
    file_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    base_path: Path | None = None,
    repo: str | None = None,
) -> list[Chunk]:
    """
    Chunk a single file for incremental indexing.

    Args:
        file_path: Path to the file to chunk
        file_type: Type of file ('spec', 'eip', 'builder', 'lean', 'rust', 'pm')
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        base_path: Base path for relative paths in chunk IDs (data_dir)
        repo: Repository name (optional, for Rust files)

    Returns:
        List of chunks with chunk_id populated
    """
    # Handle leanSpec files separately (Python AST-based chunking)
    if file_type == "lean":
        return chunk_lean_spec_single_file(file_path, base_path)

    # Handle Rust files (leanEthereum post-quantum repos)
    # file_type can be "rust" or "rust:repo_name"
    if file_type.startswith("rust"):
        rust_repo = repo
        if ":" in file_type:
            _, rust_repo = file_type.split(":", 1)
        return chunk_rust_single_file(file_path, base_path, repo=rust_repo or "leanSig")

    # Handle PM markdown files (leanEthereum meeting notes)
    if file_type == "pm":
        # Use the markdown chunker for PM files
        pass  # Fall through to markdown handling below

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Determine repo and repo_base_path from file_type and file_path
    repo_map = {
        "spec": "consensus-specs",
        "eip": "EIPs",
        "builder": "builder-specs",
    }
    repo = repo_map.get(file_type, "")

    # Compute repo-specific base path for relative paths
    repo_base_path = None
    if base_path and repo:
        repo_base_path = base_path / repo
        if not repo_base_path.exists():
            repo_base_path = None

    chunks = []
    if file_type == "spec":
        fork = _extract_fork_from_path(file_path)
        chunks = _chunk_spec_file(
            file_path, fork, md_splitter, text_splitter,
            base_path=repo_base_path, repo=repo
        )
    elif file_type == "eip":
        chunks = _chunk_eip_file(
            file_path, md_splitter, text_splitter,
            base_path=repo_base_path, repo=repo
        )
    elif file_type == "builder":
        chunks = _chunk_builder_spec_file(
            file_path, md_splitter, text_splitter,
            base_path=repo_base_path, repo=repo
        )

    # Assign chunk IDs
    return _assign_chunk_ids(chunks, base_path)


def _extract_fork_from_path(path: Path) -> str | None:
    """Extract fork name from file path."""
    parts = path.parts
    known_forks = {"phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu", "gloas"}

    for part in parts:
        if part.lower() in known_forks:
            return part.lower()

    return None


def _chunk_spec_file(
    file_path: Path,
    fork: str | None,
    md_splitter: MarkdownHeaderTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
    base_path: Path | None = None,
    repo: str = "consensus-specs",
) -> list[Chunk]:
    """Chunk a spec markdown file."""
    chunks = []
    content = file_path.read_text()

    # Compute relative path
    if base_path:
        try:
            source = str(file_path.relative_to(base_path))
        except ValueError:
            source = str(file_path)
    else:
        source = str(file_path)

    # First split by markdown headers
    md_docs = md_splitter.split_text(content)

    for doc in md_docs:
        section = doc.metadata.get("h2") or doc.metadata.get("h1") or "unknown"

        # Detect chunk type based on content
        chunk_type = _detect_chunk_type(doc.page_content)

        # Further split if too large
        if len(doc.page_content) > 1500:
            sub_chunks = text_splitter.split_text(doc.page_content)
            for i, sub_content in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        content=sub_content,
                        source=source,
                        fork=fork,
                        section=section,
                        chunk_type=chunk_type,
                        metadata={
                            "h1": doc.metadata.get("h1"),
                            "h2": doc.metadata.get("h2"),
                            "h3": doc.metadata.get("h3"),
                            "sub_chunk": i,
                        },
                        repo=repo,
                    )
                )
        else:
            chunks.append(
                Chunk(
                    content=doc.page_content,
                    source=source,
                    fork=fork,
                    section=section,
                    chunk_type=chunk_type,
                    metadata={
                        "h1": doc.metadata.get("h1"),
                        "h2": doc.metadata.get("h2"),
                        "h3": doc.metadata.get("h3"),
                    },
                    repo=repo,
                )
            )

    # Also extract functions as separate chunks
    function_chunks = _extract_function_chunks(content, file_path, fork, base_path, repo)
    chunks.extend(function_chunks)

    return chunks


def _chunk_eip_file(
    file_path: Path,
    md_splitter: MarkdownHeaderTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
    base_path: Path | None = None,
    repo: str = "EIPs",
) -> list[Chunk]:
    """Chunk an EIP markdown file."""
    chunks = []
    content = file_path.read_text()

    # Compute relative path
    if base_path:
        try:
            source = str(file_path.relative_to(base_path))
        except ValueError:
            source = str(file_path)
    else:
        source = str(file_path)

    # Extract EIP number from filename
    eip_match = re.search(r"eip-(\d+)", file_path.name)
    eip_number = eip_match.group(1) if eip_match else "unknown"

    # Extract frontmatter metadata
    frontmatter = _extract_eip_frontmatter(content)

    # Split by headers
    md_docs = md_splitter.split_text(content)

    for doc in md_docs:
        section = doc.metadata.get("h2") or doc.metadata.get("h1") or "unknown"

        if len(doc.page_content) > 1500:
            sub_chunks = text_splitter.split_text(doc.page_content)
            for i, sub_content in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        content=sub_content,
                        source=source,
                        fork=None,
                        section=section,
                        chunk_type="eip",
                        metadata={
                            "eip": eip_number,
                            "title": frontmatter.get("title"),
                            "status": frontmatter.get("status"),
                            "category": frontmatter.get("category"),
                            "sub_chunk": i,
                            **doc.metadata,
                        },
                        repo=repo,
                    )
                )
        else:
            chunks.append(
                Chunk(
                    content=doc.page_content,
                    source=source,
                    fork=None,
                    section=section,
                    chunk_type="eip",
                    metadata={
                        "eip": eip_number,
                        "title": frontmatter.get("title"),
                        "status": frontmatter.get("status"),
                        "category": frontmatter.get("category"),
                        **doc.metadata,
                    },
                    repo=repo,
                )
            )

    return chunks


def _chunk_builder_spec_file(
    file_path: Path,
    md_splitter: MarkdownHeaderTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
    base_path: Path | None = None,
    repo: str = "builder-specs",
) -> list[Chunk]:
    """Chunk a builder-specs markdown file."""
    chunks = []
    content = file_path.read_text()

    # Compute relative path
    if base_path:
        try:
            source = str(file_path.relative_to(base_path))
        except ValueError:
            source = str(file_path)
    else:
        source = str(file_path)

    # Extract fork from path (e.g., specs/bellatrix/builder.md)
    fork = _extract_fork_from_path(file_path)

    # Split by headers
    md_docs = md_splitter.split_text(content)

    for doc in md_docs:
        section = doc.metadata.get("h2") or doc.metadata.get("h1") or "unknown"

        # Detect chunk type
        chunk_type = _detect_builder_spec_chunk_type(doc.page_content)

        if len(doc.page_content) > 1500:
            sub_chunks = text_splitter.split_text(doc.page_content)
            for i, sub_content in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        content=sub_content,
                        source=source,
                        fork=fork,
                        section=section,
                        chunk_type=chunk_type,
                        metadata={
                            "spec_type": "builder",
                            "h1": doc.metadata.get("h1"),
                            "h2": doc.metadata.get("h2"),
                            "h3": doc.metadata.get("h3"),
                            "sub_chunk": i,
                        },
                        repo=repo,
                    )
                )
        else:
            chunks.append(
                Chunk(
                    content=doc.page_content,
                    source=source,
                    fork=fork,
                    section=section,
                    chunk_type=chunk_type,
                    metadata={
                        "spec_type": "builder",
                        "h1": doc.metadata.get("h1"),
                        "h2": doc.metadata.get("h2"),
                        "h3": doc.metadata.get("h3"),
                    },
                    repo=repo,
                )
            )

    return chunks


def _detect_builder_spec_chunk_type(content: str) -> str:
    """Detect the type of content in a builder-spec chunk."""
    # Check for API endpoint definitions
    if re.search(r"(POST|GET|PUT|DELETE)\s+`?/", content):
        return "builder_api"
    # Check for SSZ container definitions
    if re.search(r"class\s+\w+\s*\(Container\)", content):
        return "builder_type"
    # Check for data structure definitions
    if re.search(r"```python\s*class", content):
        return "builder_type"
    return "builder_spec"


def _detect_chunk_type(content: str) -> str:
    """Detect the type of content in a chunk."""
    if re.search(r"^def\s+\w+\s*\(", content, re.MULTILINE):
        return "function"
    if re.search(r"\|\s*`?[A-Z][A-Z0-9_]+`?\s*\|", content):
        return "constant"
    if re.search(r"^class\s+\w+", content, re.MULTILINE):
        return "type"
    return "spec"


def _extract_function_chunks(
    content: str,
    file_path: Path,
    fork: str | None,
    base_path: Path | None = None,
    repo: str = "consensus-specs",
) -> list[Chunk]:
    """Extract complete function definitions as separate chunks."""
    chunks = []

    # Compute relative path
    if base_path:
        try:
            source = str(file_path.relative_to(base_path))
        except ValueError:
            source = str(file_path)
    else:
        source = str(file_path)

    # Find all python code blocks with function definitions
    pattern = r"```python\n(def\s+(\w+)\s*\([^`]+?)```"

    for match in re.finditer(pattern, content, re.DOTALL):
        func_source = match.group(1).strip()
        func_name = match.group(2)

        chunks.append(
            Chunk(
                content=func_source,
                source=source,
                fork=fork,
                section=func_name,
                chunk_type="function",
                metadata={
                    "function_name": func_name,
                },
                repo=repo,
            )
        )

    return chunks


def _extract_eip_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from EIP."""
    frontmatter = {}

    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if match:
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip()

    return frontmatter


def chunk_client_code(
    items: list["ExtractedItem"],
    constants: list["ExtractedConstant"],
    max_body_length: int = 2000,
) -> list[Chunk]:
    """
    Convert extracted client code items into chunks for embedding.

    Args:
        items: List of ExtractedItem (functions, structs, interfaces)
        constants: List of ExtractedConstant
        max_body_length: Truncate bodies longer than this

    Returns:
        List of chunks ready for embedding
    """
    chunks = []

    for item in items:
        # Build content with doc comment + signature + body
        content_parts = []
        if item.doc_comment:
            content_parts.append(f"// {item.doc_comment}")
        content_parts.append(item.signature)

        # Include body but truncate if too long
        body = item.body
        if len(body) > max_body_length:
            body = body[:max_body_length] + "\n// ... truncated ..."

        content_parts.append(body)
        content = "\n".join(content_parts)

        # Map item kind to chunk_type
        chunk_type_map = {
            "function": "client_function",
            "struct": "client_struct",
            "interface": "client_interface",
            "enum": "client_enum",
            "type": "client_type",
        }
        chunk_type = chunk_type_map.get(item.kind, f"client_{item.kind}")

        chunks.append(
            Chunk(
                content=content,
                source=item.file_path,
                fork=None,
                section=item.name,
                chunk_type=chunk_type,
                metadata={
                    "function_name": item.name if item.kind == "function" else "",
                    "client": item.client,
                    "language": item.language,
                    "line_number": item.line_number,
                    "visibility": item.visibility,
                },
            )
        )

    for const in constants:
        content_parts = []
        if const.doc_comment:
            content_parts.append(f"// {const.doc_comment}")
        if const.type_annotation:
            content_parts.append(f"const {const.name}: {const.type_annotation} = {const.value}")
        else:
            content_parts.append(f"const {const.name} = {const.value}")

        content = "\n".join(content_parts)

        chunks.append(
            Chunk(
                content=content,
                source=const.file_path,
                fork=None,
                section=const.name,
                chunk_type="client_constant",
                metadata={
                    "client": const.client,
                    "language": const.language,
                    "line_number": const.line_number,
                },
            )
        )

    return chunks


# =============================================================================
# leanSpec Python Chunking (AST-based)
# =============================================================================


def chunk_python_spec_file(
    file_path: Path,
    base_path: Path | None = None,
    repo: str = "leanSpec",
) -> list[Chunk]:
    """
    Chunk a Python spec file using AST parsing.

    Extracts:
    - Classes (Container/Pydantic subclasses) with docstrings
    - Functions/methods with signatures and docstrings
    - Module docstrings
    - Module-level constants (UPPER_CASE assignments)

    Args:
        file_path: Path to the Python file
        base_path: Base path for computing relative source paths
        repo: Repository name for metadata

    Returns:
        List of Chunk objects with chunk types: lean_class, lean_function,
        lean_constant, lean_doc
    """
    content = file_path.read_text()

    # Compute relative path for source field
    if base_path:
        try:
            source = str(file_path.relative_to(base_path))
        except ValueError:
            source = str(file_path)
    else:
        source = str(file_path)

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # If we can't parse the file, return empty list
        return []

    chunks = []
    lines = content.splitlines()

    # Extract module-level docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        chunks.append(
            Chunk(
                content=module_doc,
                source=source,
                fork=None,
                section=file_path.stem,
                chunk_type="lean_doc",
                metadata={
                    "doc_type": "module",
                    "file_name": file_path.name,
                },
                repo=repo,
            )
        )

    # Walk the AST
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_chunks = _extract_class_chunks(node, lines, source, repo)
            chunks.extend(class_chunks)

        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_chunk = _extract_function_chunk(node, lines, source, repo)
            if func_chunk:
                chunks.append(func_chunk)

        elif isinstance(node, ast.Assign):
            const_chunk = _extract_constant_chunk(node, lines, source, repo)
            if const_chunk:
                chunks.append(const_chunk)

        elif isinstance(node, ast.AnnAssign):
            const_chunk = _extract_annotated_constant_chunk(node, lines, source, repo)
            if const_chunk:
                chunks.append(const_chunk)

    return chunks


def _get_source_segment(lines: list[str], node: ast.AST) -> str:
    """Extract source code for an AST node."""
    start_line = node.lineno - 1
    end_line = node.end_lineno if node.end_lineno else node.lineno
    return "\n".join(lines[start_line:end_line])


def _extract_class_chunks(
    node: ast.ClassDef,
    lines: list[str],
    source: str,
    repo: str,
) -> list[Chunk]:
    """Extract chunks from a class definition."""
    chunks = []

    # Get class docstring
    class_doc = ast.get_docstring(node) or ""

    # Get base classes
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            if isinstance(base.value, ast.Name):
                bases.append(f"{base.value.id}.{base.attr}")
            else:
                bases.append(base.attr)

    # Get full class source for the main chunk
    class_source = _get_source_segment(lines, node)

    # Truncate very large classes but preserve signature and docstring
    if len(class_source) > 3000:
        # Keep the beginning with signature, docstring, and first few members
        truncated = class_source[:2500] + "\n    # ... (truncated)"
        class_source = truncated

    chunks.append(
        Chunk(
            content=class_source,
            source=source,
            fork=None,
            section=node.name,
            chunk_type="lean_class",
            metadata={
                "class_name": node.name,
                "base_classes": bases,
                "docstring": class_doc[:500] if class_doc else "",
                "line_number": node.lineno,
            },
            repo=repo,
        )
    )

    # Also extract methods as separate chunks for better search
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            method_chunk = _extract_function_chunk(
                item, lines, source, repo,
                class_name=node.name,
            )
            if method_chunk:
                chunks.append(method_chunk)

    return chunks


def _extract_function_chunk(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    source: str,
    repo: str,
    class_name: str | None = None,
) -> Chunk | None:
    """Extract a chunk from a function definition."""
    func_source = _get_source_segment(lines, node)
    func_doc = ast.get_docstring(node) or ""

    # Build function signature
    import contextlib

    args = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            with contextlib.suppress(Exception):
                arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)

    # Handle return type
    return_type = ""
    if node.returns:
        with contextlib.suppress(Exception):
            return_type = f" -> {ast.unparse(node.returns)}"

    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    signature = f"{async_prefix}def {node.name}({', '.join(args)}){return_type}:"

    # Full name includes class if it's a method
    full_name = f"{class_name}.{node.name}" if class_name else node.name

    # Truncate very long function bodies
    if len(func_source) > 2000:
        func_source = func_source[:1800] + "\n    # ... (truncated)"

    return Chunk(
        content=func_source,
        source=source,
        fork=None,
        section=full_name,
        chunk_type="lean_function",
        metadata={
            "function_name": node.name,
            "full_name": full_name,
            "signature": signature,
            "docstring": func_doc[:500] if func_doc else "",
            "line_number": node.lineno,
            "is_method": class_name is not None,
            "class_name": class_name or "",
        },
        repo=repo,
    )


def _is_constant_name(name: str) -> bool:
    """Check if a name looks like a constant (UPPER_CASE)."""
    return name.isupper() or (name[0].isupper() and "_" in name and name.replace("_", "").isupper())


def _extract_constant_chunk(
    node: ast.Assign,
    lines: list[str],
    source: str,
    repo: str,
) -> Chunk | None:
    """Extract a chunk from a constant assignment."""
    # Only process simple single-target assignments
    if len(node.targets) != 1:
        return None

    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None

    name = target.id
    if not _is_constant_name(name):
        return None

    const_source = _get_source_segment(lines, node)

    # Try to get the value as string
    try:
        value_str = ast.unparse(node.value)
    except Exception:
        value_str = ""

    return Chunk(
        content=const_source,
        source=source,
        fork=None,
        section=name,
        chunk_type="lean_constant",
        metadata={
            "constant_name": name,
            "value": value_str[:200] if value_str else "",
            "line_number": node.lineno,
        },
        repo=repo,
    )


def _extract_annotated_constant_chunk(
    node: ast.AnnAssign,
    lines: list[str],
    source: str,
    repo: str,
) -> Chunk | None:
    """Extract a chunk from an annotated constant assignment."""
    if not isinstance(node.target, ast.Name):
        return None

    name = node.target.id
    if not _is_constant_name(name):
        return None

    const_source = _get_source_segment(lines, node)

    # Try to get type annotation and value
    try:
        type_str = ast.unparse(node.annotation)
    except Exception:
        type_str = ""

    try:
        value_str = ast.unparse(node.value) if node.value else ""
    except Exception:
        value_str = ""

    return Chunk(
        content=const_source,
        source=source,
        fork=None,
        section=name,
        chunk_type="lean_constant",
        metadata={
            "constant_name": name,
            "type_annotation": type_str,
            "value": value_str[:200] if value_str else "",
            "line_number": node.lineno,
        },
        repo=repo,
    )


def chunk_lean_spec_single_file(
    file_path: Path,
    base_path: Path | None = None,
) -> list[Chunk]:
    """
    Chunk a single leanSpec file for incremental indexing.

    This is a wrapper around chunk_python_spec_file that assigns chunk IDs.

    Args:
        file_path: Path to the Python file
        base_path: Base path for relative paths (data_dir)

    Returns:
        List of chunks with chunk_id populated
    """
    # Compute repo base path for relative paths
    repo_base_path = None
    if base_path:
        repo_base_path = base_path / "leanSpec"
        if not repo_base_path.exists():
            repo_base_path = None

    chunks = chunk_python_spec_file(
        file_path,
        base_path=repo_base_path,
        repo="leanSpec",
    )

    # Assign chunk IDs
    return _assign_lean_spec_chunk_ids(chunks, base_path)


def _assign_lean_spec_chunk_ids(
    chunks: list[Chunk],
    base_path: Path | None = None,
) -> list[Chunk]:
    """Assign unique chunk IDs to leanSpec chunks."""
    import contextlib
    from collections import defaultdict

    # Group chunks by source file to assign sequential indices
    file_chunks: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        file_chunks[chunk.source].append((idx, chunk))

    # Assign IDs
    new_chunks = [None] * len(chunks)
    for source, indexed_chunks in file_chunks.items():
        # Make path relative if base_path provided
        rel_source = source
        if base_path:
            with contextlib.suppress(ValueError):
                rel_source = str(Path(source).relative_to(base_path))

        for chunk_idx, (original_idx, chunk) in enumerate(indexed_chunks):
            chunk_id = generate_chunk_id(
                project="eth",
                source_type=chunk.chunk_type,
                source_file=rel_source,
                chunk_index=chunk_idx,
                content=chunk.content,
            )
            # Create new chunk with ID (Chunk is frozen)
            new_chunk = Chunk(
                content=chunk.content,
                source=chunk.source,
                fork=chunk.fork,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata,
                repo=chunk.repo,
                chunk_id=chunk_id,
            )
            new_chunks[original_idx] = new_chunk

    return new_chunks


# =============================================================================
# Rust Chunking (for leanEthereum post-quantum repos)
# =============================================================================


def chunk_rust_file(
    file_path: Path,
    base_path: Path | None = None,
    repo: str = "leanSig",
) -> list[Chunk]:
    """
    Chunk a Rust file using the RustParser.

    Args:
        file_path: Path to the Rust source file
        base_path: Base path for computing relative source paths
        repo: Repository name for metadata

    Returns:
        List of Chunk objects with chunk types: rust_function, rust_struct,
        rust_enum, rust_const, rust_impl
    """
    from .rust_compiler import RustParser

    # Compute relative path for source field
    if base_path:
        try:
            source = str(file_path.relative_to(base_path))
        except ValueError:
            source = str(file_path)
    else:
        source = str(file_path)

    parser = RustParser()
    items, constants = parser.parse_file(file_path)

    chunks = []

    # Convert items to chunks
    for item in items:
        chunk = chunk_rust_item(item, source, repo)
        chunks.append(chunk)

    # Convert constants to chunks
    for const in constants:
        chunk = chunk_rust_constant(const, source, repo)
        chunks.append(chunk)

    return chunks


def chunk_rust_item(
    item: "RustItem",
    source: str,
    repo: str,
) -> Chunk:
    """
    Convert an extracted Rust item to a chunk.

    Args:
        item: ExtractedItem from RustParser
        source: Relative source path
        repo: Repository name

    Returns:
        Chunk object
    """
    # Build content with doc comment + signature + body
    content_parts = []
    if item.doc_comment:
        content_parts.append(f"/// {item.doc_comment}")
    if item.attributes:
        content_parts.extend(item.attributes)

    # Truncate very long bodies
    body = item.body
    if len(body) > 3000:
        body = body[:2500] + "\n// ... (truncated)"

    content_parts.append(body)
    content = "\n".join(content_parts)

    # Map item kind to chunk type
    chunk_type_map = {
        "function": "rust_function",
        "struct": "rust_struct",
        "enum": "rust_enum",
        "impl": "rust_impl",
    }
    chunk_type = chunk_type_map.get(item.kind, f"rust_{item.kind}")

    return Chunk(
        content=content,
        source=source,
        fork=None,
        section=item.name,
        chunk_type=chunk_type,
        metadata={
            "function_name": item.name if item.kind == "function" else "",
            "signature": item.signature,
            "visibility": item.visibility,
            "line_number": item.line_number,
            "attributes": item.attributes,
            "docstring": item.doc_comment[:500] if item.doc_comment else "",
        },
        repo=repo,
    )


def chunk_rust_constant(
    const: "RustConstant",
    source: str,
    repo: str,
) -> Chunk:
    """
    Convert an extracted Rust constant to a chunk.

    Args:
        const: ExtractedConstant from RustParser
        source: Relative source path
        repo: Repository name

    Returns:
        Chunk object
    """
    content_parts = []
    if const.doc_comment:
        content_parts.append(f"/// {const.doc_comment}")

    if const.type_annotation:
        content_parts.append(f"const {const.name}: {const.type_annotation} = {const.value};")
    else:
        content_parts.append(f"const {const.name} = {const.value};")

    content = "\n".join(content_parts)

    return Chunk(
        content=content,
        source=source,
        fork=None,
        section=const.name,
        chunk_type="rust_const",
        metadata={
            "constant_name": const.name,
            "type_annotation": const.type_annotation or "",
            "value": const.value[:200] if const.value else "",
            "line_number": const.line_number,
        },
        repo=repo,
    )


def chunk_rust_single_file(
    file_path: Path,
    base_path: Path | None = None,
    repo: str = "leanSig",
) -> list[Chunk]:
    """
    Chunk a single Rust file for incremental indexing.

    This is a wrapper around chunk_rust_file that assigns chunk IDs.

    Args:
        file_path: Path to the Rust file
        base_path: Base path for relative paths (data_dir)
        repo: Repository name

    Returns:
        List of chunks with chunk_id populated
    """
    # Compute repo base path for relative paths
    repo_base_path = None
    if base_path:
        repo_base_path = base_path / repo
        if not repo_base_path.exists():
            repo_base_path = None

    chunks = chunk_rust_file(
        file_path,
        base_path=repo_base_path,
        repo=repo,
    )

    # Assign chunk IDs
    return _assign_rust_chunk_ids(chunks, base_path)


def _assign_rust_chunk_ids(
    chunks: list[Chunk],
    base_path: Path | None = None,
) -> list[Chunk]:
    """Assign unique chunk IDs to Rust chunks."""
    import contextlib
    from collections import defaultdict

    # Group chunks by source file to assign sequential indices
    file_chunks: dict[str, list[tuple[int, Chunk]]] = defaultdict(list)
    for idx, chunk in enumerate(chunks):
        file_chunks[chunk.source].append((idx, chunk))

    # Assign IDs
    new_chunks = [None] * len(chunks)
    for source, indexed_chunks in file_chunks.items():
        # Make path relative if base_path provided
        rel_source = source
        if base_path:
            with contextlib.suppress(ValueError):
                rel_source = str(Path(source).relative_to(base_path))

        for chunk_idx, (original_idx, chunk) in enumerate(indexed_chunks):
            chunk_id = generate_chunk_id(
                project="eth",
                source_type=chunk.chunk_type,
                source_file=rel_source,
                chunk_index=chunk_idx,
                content=chunk.content,
            )
            # Create new chunk with ID (Chunk is frozen)
            new_chunk = Chunk(
                content=chunk.content,
                source=chunk.source,
                fork=chunk.fork,
                section=chunk.section,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata,
                repo=chunk.repo,
                chunk_id=chunk_id,
            )
            new_chunks[original_idx] = new_chunk

    return new_chunks
