"""CLI for Ethereum MCP management."""

import subprocess
from pathlib import Path

import click

from .config import get_model_info, load_config
from .indexer.chunker import chunk_client_code
from .indexer.client_compiler import compile_client, load_client_constants, load_client_items
from .indexer.compiler import compile_specs
from .indexer.downloader import (
    CLIENT_REPOS,
    LEAN_PQ_REPOS,
    download_clients,
    download_specs,
    get_builder_spec_files,
    get_eip_files,
    get_lean_pm_files,
    get_lean_rust_files,
    get_lean_snappy_files,
    get_lean_spec_files,
    get_spec_files,
    list_downloaded_clients,
)
from .indexer.embedder import IncrementalEmbedder, embed_and_store
from .indexer.manifest import ManifestCorruptedError, load_manifest

DEFAULT_DATA_DIR = Path.home() / ".ethereum-mcp"


@click.group()
def main():
    """Ethereum MCP - RAG-powered Ethereum specs search."""
    pass


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("--include-clients", is_flag=True, help="Also download client source code")
@click.option("--skip-lean-pq", is_flag=True, help="Skip leanEthereum post-quantum repos")
def download(data_dir: str, force: bool, include_clients: bool, skip_lean_pq: bool):
    """Download specs, EIPs, leanSpec, leanEthereum PQ repos, and optionally client code."""
    data_path = Path(data_dir)
    click.echo(f"Downloading to {data_path}...")

    consensus_dir, eips_dir, builder_specs_dir, lean_spec_dir = download_specs(
        data_path, force=force, include_lean_pq=not skip_lean_pq
    )

    click.echo(f"Consensus specs: {consensus_dir}")
    click.echo(f"EIPs: {eips_dir}")
    click.echo(f"Builder specs: {builder_specs_dir}")
    click.echo(f"leanSpec: {lean_spec_dir}")

    if not skip_lean_pq:
        click.echo("\nleanEthereum post-quantum repos:")
        for repo_name in LEAN_PQ_REPOS:
            repo_path = data_path / repo_name
            if repo_path.exists():
                click.echo(f"  {click.style('✓', fg='green')} {repo_name}")
            else:
                click.echo(f"  {click.style('✗', fg='red')} {repo_name}")

    if include_clients:
        click.echo("\nDownloading client source code...")
        results = download_clients(data_path, progress_callback=click.echo)
        click.echo("\nClient download results:")
        for name, success in results.items():
            status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
            click.echo(f"  {status} {name}")


@main.command("download-clients")
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--client", multiple=True, help="Specific clients to download (default: all)")
def download_clients_cmd(data_dir: str, client: tuple):
    """Download Ethereum client source code (reth, geth, lighthouse, etc.)."""
    data_path = Path(data_dir)

    clients_to_download = list(client) if client else None

    click.echo("Available clients:")
    for name, config in CLIENT_REPOS.items():
        click.echo(f"  {name}: {config['language']} ({config['layer']})")

    click.echo(f"\nDownloading to {data_path / 'clients'}...")
    results = download_clients(data_path, clients=clients_to_download, progress_callback=click.echo)

    click.echo("\nResults:")
    for name, success in results.items():
        status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
        click.echo(f"  {status} {name}")


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--include-clients", is_flag=True, help="Also compile client source code")
def compile(data_dir: str, include_clients: bool):
    """Compile specs and optionally client source code into indexed JSON."""
    data_path = Path(data_dir)
    consensus_dir = data_path / "consensus-specs"
    output_dir = data_path / "compiled"

    if not consensus_dir.exists():
        click.echo("Error: Consensus specs not found. Run 'download' first.")
        raise click.Abort()

    click.echo("Compiling specs...")
    compiled = compile_specs(consensus_dir, output_dir)
    click.echo(f"Compiled {len(compiled)} forks to {output_dir}")

    if include_clients:
        click.echo("\nCompiling client source code...")
        clients_dir = data_path / "clients"
        client_output_dir = output_dir / "clients"

        total_stats = {
            "files_processed": 0,
            "items_extracted": 0,
            "constants_extracted": 0,
            "functions": 0,
            "structs": 0,
        }

        for client_name, config in CLIENT_REPOS.items():
            client_path = clients_dir / client_name
            if not client_path.exists():
                click.echo(f"  Skipping {client_name} (not downloaded)")
                continue

            click.echo(f"  Compiling {client_name} ({config['language']})...")
            stats = compile_client(
                client_path,
                client_output_dir / client_name,
                client_name,
                config["language"],
                progress_callback=lambda msg: click.echo(f"    {msg}"),
            )

            if "error" not in stats:
                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)

        click.echo("\nClient compilation complete:")
        click.echo(f"  Files: {total_stats['files_processed']}")
        click.echo(f"  Functions: {total_stats['functions']}")
        click.echo(f"  Structs: {total_stats['structs']}")
        click.echo(f"  Constants: {total_stats['constants_extracted']}")


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--chunk-size", default=1000, help="Chunk size in characters")
@click.option("--chunk-overlap", default=200, help="Chunk overlap in characters")
@click.option("--include-clients", is_flag=True, help="Also index compiled client code")
@click.option("--full", is_flag=True, help="Force full rebuild (ignore incremental)")
@click.option("--dry-run", is_flag=True, help="Show what would change without indexing")
@click.option("--model", default=None, help="Embedding model to use")
def index(
    data_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    include_clients: bool,
    full: bool,
    dry_run: bool,
    model: str | None,
):
    """Build vector index from specs, EIPs, builder-specs, leanSpec, and optionally client code.

    By default, performs incremental indexing (only re-embeds changed files).
    Use --full to force a complete rebuild.
    """
    data_path = Path(data_dir)
    consensus_dir = data_path / "consensus-specs"
    eips_dir = data_path / "EIPs"
    builder_specs_dir = data_path / "builder-specs"
    lean_spec_dir = data_path / "leanSpec"

    if not consensus_dir.exists():
        click.echo("Error: Consensus specs not found. Run 'download' first.")
        raise click.Abort()

    click.echo("Collecting files...")

    # Collect files with their types
    current_files: dict[str, Path] = {}
    file_types: dict[str, str] = {}

    spec_files = get_spec_files(consensus_dir)
    for f in spec_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "spec"

    eip_files = get_eip_files(eips_dir) if eips_dir.exists() else []
    for f in eip_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "eip"

    if builder_specs_dir.exists():
        builder_spec_files = get_builder_spec_files(builder_specs_dir)
    else:
        builder_spec_files = []
    for f in builder_spec_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "builder"

    # Collect leanSpec Python files
    if lean_spec_dir.exists():
        try:
            lean_spec_files = get_lean_spec_files(lean_spec_dir)
        except FileNotFoundError:
            lean_spec_files = []
    else:
        lean_spec_files = []
    for f in lean_spec_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "lean"

    # Collect leanEthereum post-quantum Rust files
    lean_rust_count = 0
    for repo_name, repo_info in LEAN_PQ_REPOS.items():
        repo_dir = data_path / repo_name
        if not repo_dir.exists():
            continue

        if repo_info["language"] == "rust":
            try:
                rust_files = get_lean_rust_files(repo_dir)
                for f in rust_files:
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    # Store repo name in file_types for later use
                    file_types[rel] = f"rust:{repo_name}"
                lean_rust_count += len(rust_files)
            except FileNotFoundError:
                pass
        elif repo_info["language"] == "markdown" and repo_name == "pm":
            try:
                pm_files = get_lean_pm_files(repo_dir)
                for f in pm_files:
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = "pm"
            except FileNotFoundError:
                pass
        elif repo_info["language"] == "python" and repo_name == "leanSnappy":
            try:
                snappy_files = get_lean_snappy_files(repo_dir)
                for f in snappy_files:
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = "lean"  # Use same chunker as leanSpec
            except FileNotFoundError:
                pass

    click.echo(
        f"Found {len(spec_files)} spec files, {len(eip_files)} EIP files, "
        f"{len(builder_spec_files)} builder-spec files, {len(lean_spec_files)} leanSpec files, "
        f"{lean_rust_count} leanEthereum Rust files"
    )

    # Create incremental embedder
    embedder = IncrementalEmbedder(
        data_dir=data_path,
        model_name=model,
    )

    if dry_run:
        # Show what would change
        result = embedder.dry_run(current_files, file_types)
        click.echo("\n" + result.summary())
        if result.files_to_add:
            click.echo("\nFiles to add:")
            for f in result.files_to_add[:10]:
                click.echo(f"  + {f}")
            if len(result.files_to_add) > 10:
                click.echo(f"  ... and {len(result.files_to_add) - 10} more")
        if result.files_to_modify:
            click.echo("\nFiles to modify:")
            for f in result.files_to_modify[:10]:
                click.echo(f"  ~ {f}")
            if len(result.files_to_modify) > 10:
                click.echo(f"  ... and {len(result.files_to_modify) - 10} more")
        if result.files_to_delete:
            click.echo("\nFiles to delete:")
            for f in result.files_to_delete[:10]:
                click.echo(f"  - {f}")
            if len(result.files_to_delete) > 10:
                click.echo(f"  ... and {len(result.files_to_delete) - 10} more")
        return

    # Perform indexing
    click.echo("\nIndexing...")
    stats = embedder.index(
        current_files,
        file_types,
        force_full=full,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    click.echo(f"\n{stats.summary()}")

    # Handle client code separately (not part of incremental for now)
    if include_clients:
        click.echo("\nIndexing client code (full rebuild)...")
        _index_client_code(data_path)


def _index_client_code(data_path: Path) -> None:
    """Index compiled client code (separate from specs)."""
    client_compiled_dir = data_path / "compiled" / "clients"
    db_path = data_path / "lancedb"

    if not client_compiled_dir.exists():
        click.echo("  No compiled clients found. Run 'compile --include-clients' first.")
        return

    all_chunks = []
    for client_dir in client_compiled_dir.iterdir():
        if client_dir.is_dir():
            items = load_client_items(client_dir)
            constants = load_client_constants(client_dir)

            if items or constants:
                client_chunks = chunk_client_code(items, constants)
                all_chunks.extend(client_chunks)
                click.echo(
                    f"  {client_dir.name}: {len(items)} items, "
                    f"{len(constants)} constants -> {len(client_chunks)} chunks"
                )

    if all_chunks:
        # Note: Client code uses separate table or appends to main table
        # For now, this is a simple append
        count = embed_and_store(all_chunks, db_path, table_name="ethereum_clients")
        click.echo(f"  Indexed {count} client code chunks")


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data dir")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("--include-clients", is_flag=True, help="Download and compile client code")
@click.option("--full", is_flag=True, help="Force full index rebuild")
def build(data_dir: str, force: bool, include_clients: bool, full: bool):
    """Full build: download, compile, and index."""
    data_path = Path(data_dir)

    # Download specs
    click.echo("=== Downloading specs ===")
    consensus_dir, eips_dir, builder_specs_dir, lean_spec_dir = download_specs(
        data_path, force=force
    )

    # Download clients if requested
    if include_clients:
        click.echo("\n=== Downloading client source code ===")
        results = download_clients(data_path, progress_callback=click.echo)
        successful = sum(1 for s in results.values() if s)
        click.echo(f"Downloaded {successful}/{len(results)} clients")

    # Compile specs
    click.echo("\n=== Compiling specs ===")
    output_dir = data_path / "compiled"
    compiled = compile_specs(consensus_dir, output_dir)
    click.echo(f"Compiled {len(compiled)} forks")

    # Compile clients if requested
    if include_clients:
        click.echo("\n=== Compiling client source code ===")
        clients_dir = data_path / "clients"
        client_output_dir = output_dir / "clients"

        for client_name, config in CLIENT_REPOS.items():
            client_path = clients_dir / client_name
            if not client_path.exists():
                continue

            click.echo(f"  {client_name} ({config['language']})...")
            compile_client(
                client_path,
                client_output_dir / client_name,
                client_name,
                config["language"],
            )

    # Index (using incremental by default unless --full)
    click.echo("\n=== Building vector index ===")

    # Collect files
    current_files: dict[str, Path] = {}
    file_types: dict[str, str] = {}

    spec_files = get_spec_files(consensus_dir)
    for f in spec_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "spec"

    eip_files = get_eip_files(eips_dir) if eips_dir.exists() else []
    for f in eip_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "eip"

    if builder_specs_dir.exists():
        builder_files = get_builder_spec_files(builder_specs_dir)
        for f in builder_files:
            rel = str(f.relative_to(data_path))
            current_files[rel] = f
            file_types[rel] = "builder"

    # Collect leanSpec Python files
    if lean_spec_dir.exists():
        try:
            lean_files = get_lean_spec_files(lean_spec_dir)
        except FileNotFoundError:
            lean_files = []
    else:
        lean_files = []
    for f in lean_files:
        rel = str(f.relative_to(data_path))
        current_files[rel] = f
        file_types[rel] = "lean"

    # Collect leanEthereum post-quantum Rust files
    lean_rust_count = 0
    for repo_name, repo_info in LEAN_PQ_REPOS.items():
        repo_dir = data_path / repo_name
        if not repo_dir.exists():
            continue

        if repo_info["language"] == "rust":
            try:
                rust_files = get_lean_rust_files(repo_dir)
                for f in rust_files:
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = f"rust:{repo_name}"
                lean_rust_count += len(rust_files)
            except FileNotFoundError:
                pass
        elif repo_info["language"] == "markdown" and repo_name == "pm":
            try:
                pm_files = get_lean_pm_files(repo_dir)
                for f in pm_files:
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = "pm"
            except FileNotFoundError:
                pass
        elif repo_info["language"] == "python" and repo_name == "leanSnappy":
            try:
                snappy_files = get_lean_snappy_files(repo_dir)
                for f in snappy_files:
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = "lean"
            except FileNotFoundError:
                pass

    click.echo(
        f"Found {len(spec_files)} spec files, {len(eip_files)} EIP files, "
        f"{len(builder_files) if builder_specs_dir.exists() else 0} builder-spec files, "
        f"{len(lean_files)} leanSpec files, {lean_rust_count} leanEthereum Rust files"
    )

    # Use incremental embedder
    embedder = IncrementalEmbedder(data_dir=data_path)
    stats = embedder.index(
        current_files,
        file_types,
        force_full=full,
    )

    click.echo(f"\n{stats.summary()}")

    # Index compiled client code if clients were included
    if include_clients:
        click.echo("\nIndexing client code...")
        _index_client_code(data_path)

    click.echo("\n=== Build complete ===")


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--full", is_flag=True, help="Force full index rebuild after update")
def update(data_dir: str, full: bool):
    """Update repos (git pull) and incrementally re-index."""
    data_path = Path(data_dir)

    # Git pull for each repo
    repos = [
        ("consensus-specs", data_path / "consensus-specs"),
        ("EIPs", data_path / "EIPs"),
        ("builder-specs", data_path / "builder-specs"),
        ("leanSpec", data_path / "leanSpec"),
    ]

    # Add leanEthereum post-quantum repos
    for repo_name in LEAN_PQ_REPOS:
        repos.append((repo_name, data_path / repo_name))

    click.echo("=== Updating repositories ===")
    for name, repo_path in repos:
        if not repo_path.exists():
            click.echo(f"  {name}: not downloaded")
            continue

        click.echo(f"  {name}: ", nl=False)
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                # Check if there were changes
                if "Already up to date" in result.stdout:
                    click.echo("already up to date")
                else:
                    click.echo("updated")
            else:
                click.echo(f"error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            click.echo("timeout")
        except Exception as e:
            click.echo(f"error: {e}")

    # Re-index
    click.echo("\n=== Rebuilding index ===")

    # Collect files
    current_files: dict[str, Path] = {}
    file_types: dict[str, str] = {}

    consensus_dir = data_path / "consensus-specs"
    eips_dir = data_path / "EIPs"
    builder_specs_dir = data_path / "builder-specs"

    if consensus_dir.exists():
        for f in get_spec_files(consensus_dir):
            rel = str(f.relative_to(data_path))
            current_files[rel] = f
            file_types[rel] = "spec"

    if eips_dir.exists():
        for f in get_eip_files(eips_dir):
            rel = str(f.relative_to(data_path))
            current_files[rel] = f
            file_types[rel] = "eip"

    if builder_specs_dir.exists():
        for f in get_builder_spec_files(builder_specs_dir):
            rel = str(f.relative_to(data_path))
            current_files[rel] = f
            file_types[rel] = "builder"

    lean_spec_dir = data_path / "leanSpec"
    if lean_spec_dir.exists():
        try:
            for f in get_lean_spec_files(lean_spec_dir):
                rel = str(f.relative_to(data_path))
                current_files[rel] = f
                file_types[rel] = "lean"
        except FileNotFoundError:
            pass

    # Collect leanEthereum post-quantum Rust files
    for repo_name, repo_info in LEAN_PQ_REPOS.items():
        repo_dir = data_path / repo_name
        if not repo_dir.exists():
            continue

        if repo_info["language"] == "rust":
            try:
                for f in get_lean_rust_files(repo_dir):
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = f"rust:{repo_name}"
            except FileNotFoundError:
                pass
        elif repo_info["language"] == "markdown" and repo_name == "pm":
            try:
                for f in get_lean_pm_files(repo_dir):
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = "pm"
            except FileNotFoundError:
                pass
        elif repo_info["language"] == "python" and repo_name == "leanSnappy":
            try:
                for f in get_lean_snappy_files(repo_dir):
                    rel = str(f.relative_to(data_path))
                    current_files[rel] = f
                    file_types[rel] = "lean"
            except FileNotFoundError:
                pass

    if not current_files:
        click.echo("No files to index. Run 'download' first.")
        return

    # Use incremental embedder
    embedder = IncrementalEmbedder(data_dir=data_path)
    stats = embedder.index(
        current_files,
        file_types,
        force_full=full,
    )

    click.echo(f"\n{stats.summary()}")


@main.command()
@click.argument("query")
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--fork", default=None, help="Filter by fork")
@click.option("--limit", default=5, help="Max results")
def search(query: str, data_dir: str, fork: str, limit: int):
    """Search the indexed specs."""
    from .indexer.embedder import EmbeddingSearcher

    data_path = Path(data_dir)
    db_path = data_path / "lancedb"

    if not db_path.exists():
        click.echo("Error: Index not found. Run 'build' first.")
        raise click.Abort()

    searcher = EmbeddingSearcher(db_path)
    results = searcher.search(query, limit=limit, fork=fork)

    for i, r in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (score: {r['score']:.3f}) ---")
        click.echo(f"Fork: {r['fork']} | Section: {r['section']} | Type: {r['chunk_type']}")
        click.echo(f"Source: {r['source']}")
        click.echo(f"\n{r['content'][:500]}...")


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
def status(data_dir: str):
    """Show index status including manifest and embedding model info."""
    data_path = Path(data_dir)

    click.echo(f"Data directory: {data_path}")
    click.echo(f"  Exists: {data_path.exists()}")

    # Load config
    config = load_config(data_dir=data_path)
    click.echo("\nConfiguration:")
    click.echo(f"  Embedding model: {config.embedding.model}")
    click.echo(f"  Batch size: {config.embedding.batch_size}")
    click.echo(f"  Chunk size: {config.chunking.chunk_size}")
    click.echo(f"  Chunk overlap: {config.chunking.chunk_overlap}")

    # Manifest info
    manifest_path = data_path / "manifest.json"
    click.echo(f"\nManifest: {manifest_path}")
    if manifest_path.exists():
        try:
            manifest = load_manifest(manifest_path)
            if manifest:
                click.echo(f"  Version: {manifest.version}")
                click.echo(f"  Updated: {manifest.updated_at}")
                click.echo(f"  Embedding model: {manifest.embedding_model}")
                click.echo(f"  Files tracked: {len(manifest.files)}")
                total_chunks = sum(len(e.chunk_ids) for e in manifest.files.values())
                click.echo(f"  Total chunks: {total_chunks}")
                if manifest.repo_versions:
                    click.echo("  Repo versions:")
                    for repo, version in manifest.repo_versions.items():
                        click.echo(f"    {repo}: {version[:8]}")
        except ManifestCorruptedError as e:
            click.echo(f"  Status: CORRUPTED - {e}")
    else:
        click.echo("  Status: Not found (will be created on first index)")

    consensus_dir = data_path / "consensus-specs"
    click.echo(f"\nConsensus specs: {consensus_dir}")
    click.echo(f"  Exists: {consensus_dir.exists()}")

    eips_dir = data_path / "EIPs"
    click.echo(f"\nEIPs: {eips_dir}")
    click.echo(f"  Exists: {eips_dir.exists()}")

    builder_specs_dir = data_path / "builder-specs"
    click.echo(f"\nBuilder specs: {builder_specs_dir}")
    click.echo(f"  Exists: {builder_specs_dir.exists()}")

    lean_spec_dir = data_path / "leanSpec"
    click.echo(f"\nleanSpec: {lean_spec_dir}")
    click.echo(f"  Exists: {lean_spec_dir.exists()}")
    if lean_spec_dir.exists():
        try:
            lean_files = get_lean_spec_files(lean_spec_dir)
            click.echo(f"  Python files: {len(lean_files)}")
        except FileNotFoundError:
            click.echo("  Source directory not found")

    # leanEthereum post-quantum repos
    click.echo("\nleanEthereum post-quantum repos:")
    for repo_name, repo_info in LEAN_PQ_REPOS.items():
        repo_dir = data_path / repo_name
        if repo_dir.exists():
            status_icon = click.style("✓", fg="green")
            if repo_info["language"] == "rust":
                try:
                    rust_files = get_lean_rust_files(repo_dir)
                    lang = repo_info['language']
                    click.echo(f"  {status_icon} {repo_name} ({lang}) - {len(rust_files)} files")
                except FileNotFoundError:
                    click.echo(f"  {status_icon} {repo_name} ({repo_info['language']}) - no src")
            elif repo_info["language"] == "markdown":
                try:
                    md_files = get_lean_pm_files(repo_dir)
                    lang = repo_info['language']
                    click.echo(f"  {status_icon} {repo_name} ({lang}) - {len(md_files)} files")
                except FileNotFoundError:
                    click.echo(f"  {status_icon} {repo_name} ({repo_info['language']}) - no src")
            elif repo_info["language"] == "python":
                try:
                    py_files = get_lean_snappy_files(repo_dir)
                    lang = repo_info['language']
                    click.echo(f"  {status_icon} {repo_name} ({lang}) - {len(py_files)} files")
                except FileNotFoundError:
                    click.echo(f"  {status_icon} {repo_name} ({repo_info['language']}) - no src")
            else:
                click.echo(f"  {status_icon} {repo_name} ({repo_info['language']})")
        else:
            status_icon = click.style("✗", fg="red")
            click.echo(f"  {status_icon} {repo_name} (not downloaded)")

    compiled_dir = data_path / "compiled"
    click.echo(f"\nCompiled specs: {compiled_dir}")
    click.echo(f"  Exists: {compiled_dir.exists()}")
    if compiled_dir.exists():
        json_files = list(compiled_dir.glob("*.json"))
        click.echo(f"  Forks: {[f.stem.replace('_spec', '') for f in json_files]}")

    # Client status
    click.echo("\nClient source code:")
    client_status = list_downloaded_clients(data_path)
    for name, info in client_status.items():
        if info["exists"]:
            status_icon = click.style("✓", fg="green")
            version = info["version"] or "unknown"
            click.echo(f"  {status_icon} {name} ({info['language']}, {info['layer']}) - {version}")
        else:
            status_icon = click.style("✗", fg="red")
            click.echo(f"  {status_icon} {name} (not downloaded)")

    # Compiled clients
    client_compiled_dir = compiled_dir / "clients"
    if client_compiled_dir.exists():
        click.echo("\nCompiled clients:")
        for client_dir in client_compiled_dir.iterdir():
            if client_dir.is_dir():
                index_file = client_dir / "index.json"
                if index_file.exists():
                    import json

                    with open(index_file) as f:
                        index = json.load(f)
                    funcs = len(index.get("functions", {}))
                    structs = len(index.get("structs", {}))
                    click.echo(f"  {client_dir.name}: {funcs} functions, {structs} structs")

    db_path = data_path / "lancedb"
    click.echo(f"\nVector index: {db_path}")
    click.echo(f"  Exists: {db_path.exists()}")
    if db_path.exists():
        try:
            from .indexer.embedder import EmbeddingSearcher

            searcher = EmbeddingSearcher(db_path)
            stats = searcher.get_stats()
            click.echo(f"  Total chunks: {stats['total_chunks']}")
        except Exception as e:
            click.echo(f"  Error reading index: {e}")


@main.command("models")
def list_models():
    """List available embedding models."""
    click.echo(get_model_info())


if __name__ == "__main__":
    main()
