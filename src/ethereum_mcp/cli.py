"""CLI for Ethereum MCP management."""

from pathlib import Path

import click

from .indexer.chunker import chunk_client_code, chunk_documents
from .indexer.client_compiler import compile_client, load_client_constants, load_client_items
from .indexer.compiler import compile_specs
from .indexer.downloader import (
    CLIENT_REPOS,
    download_clients,
    download_specs,
    get_builder_spec_files,
    get_eip_files,
    get_spec_files,
    list_downloaded_clients,
)
from .indexer.embedder import embed_and_store

DEFAULT_DATA_DIR = Path.home() / ".ethereum-mcp"


@click.group()
def main():
    """Ethereum MCP - RAG-powered Ethereum specs search."""
    pass


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data directory")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("--include-clients", is_flag=True, help="Also download client source code")
def download(data_dir: str, force: bool, include_clients: bool):
    """Download Ethereum specs, EIPs, builder-specs, and optionally client source code."""
    data_path = Path(data_dir)
    click.echo(f"Downloading to {data_path}...")

    consensus_dir, eips_dir, builder_specs_dir = download_specs(data_path, force=force)

    click.echo(f"Consensus specs: {consensus_dir}")
    click.echo(f"EIPs: {eips_dir}")
    click.echo(f"Builder specs: {builder_specs_dir}")

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
def index(data_dir: str, chunk_size: int, chunk_overlap: int, include_clients: bool):
    """Build vector index from specs, EIPs, builder-specs, and optionally client code."""
    data_path = Path(data_dir)
    consensus_dir = data_path / "consensus-specs"
    eips_dir = data_path / "EIPs"
    builder_specs_dir = data_path / "builder-specs"
    db_path = data_path / "lancedb"

    if not consensus_dir.exists():
        click.echo("Error: Consensus specs not found. Run 'download' first.")
        raise click.Abort()

    click.echo("Collecting files...")
    spec_files = get_spec_files(consensus_dir)
    eip_files = get_eip_files(eips_dir) if eips_dir.exists() else []
    if builder_specs_dir.exists():
        builder_spec_files = get_builder_spec_files(builder_specs_dir)
    else:
        builder_spec_files = []

    click.echo(
        f"Found {len(spec_files)} spec files, {len(eip_files)} EIP files, "
        f"{len(builder_spec_files)} builder-spec files"
    )

    click.echo("Chunking documents...")
    chunks = chunk_documents(
        spec_files,
        eip_files,
        builder_spec_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    click.echo(f"Created {len(chunks)} document chunks")

    # Index compiled client code if requested
    if include_clients:
        click.echo("\nLoading compiled client code...")
        client_compiled_dir = data_path / "compiled" / "clients"

        if client_compiled_dir.exists():
            for client_dir in client_compiled_dir.iterdir():
                if client_dir.is_dir():
                    items = load_client_items(client_dir)
                    constants = load_client_constants(client_dir)

                    if items or constants:
                        client_chunks = chunk_client_code(items, constants)
                        chunks.extend(client_chunks)
                        click.echo(
                            f"  {client_dir.name}: {len(items)} items, "
                            f"{len(constants)} constants -> {len(client_chunks)} chunks"
                        )
        else:
            click.echo("  No compiled clients found. Run 'compile --include-clients' first.")

    click.echo(f"\nTotal chunks: {len(chunks)}")

    click.echo("Embedding and storing...")
    count = embed_and_store(chunks, db_path)

    click.echo(f"Indexed {count} chunks to {db_path}")


@main.command()
@click.option("--data-dir", type=click.Path(), default=str(DEFAULT_DATA_DIR), help="Data dir")
@click.option("--force", is_flag=True, help="Force re-download")
@click.option("--include-clients", is_flag=True, help="Download and compile client code")
def build(data_dir: str, force: bool, include_clients: bool):
    """Full build: download, compile, and index."""
    data_path = Path(data_dir)

    # Download specs
    click.echo("=== Downloading specs ===")
    consensus_dir, eips_dir, builder_specs_dir = download_specs(data_path, force=force)

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

    # Index
    click.echo("\n=== Building vector index ===")
    spec_files = get_spec_files(consensus_dir)
    eip_files = get_eip_files(eips_dir) if eips_dir.exists() else []
    if builder_specs_dir.exists():
        builder_spec_files = get_builder_spec_files(builder_specs_dir)
    else:
        builder_spec_files = []

    click.echo(
        f"Found {len(spec_files)} spec files, {len(eip_files)} EIP files, "
        f"{len(builder_spec_files)} builder-spec files"
    )

    chunks = chunk_documents(spec_files, eip_files, builder_spec_files)
    click.echo(f"Created {len(chunks)} document chunks")

    # Index compiled client code if clients were included
    if include_clients:
        click.echo("\nIndexing compiled client code...")
        client_compiled_dir = output_dir / "clients"

        if client_compiled_dir.exists():
            for client_dir in client_compiled_dir.iterdir():
                if client_dir.is_dir():
                    items = load_client_items(client_dir)
                    constants = load_client_constants(client_dir)

                    if items or constants:
                        client_chunks = chunk_client_code(items, constants)
                        chunks.extend(client_chunks)
                        click.echo(f"  {client_dir.name}: {len(client_chunks)} chunks")

    click.echo(f"\nTotal chunks: {len(chunks)}")

    db_path = data_path / "lancedb"
    count = embed_and_store(chunks, db_path)

    click.echo("\n=== Build complete ===")
    click.echo(f"Indexed {count} chunks to {db_path}")


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
    """Show index status."""
    data_path = Path(data_dir)

    click.echo(f"Data directory: {data_path}")
    click.echo(f"  Exists: {data_path.exists()}")

    consensus_dir = data_path / "consensus-specs"
    click.echo(f"\nConsensus specs: {consensus_dir}")
    click.echo(f"  Exists: {consensus_dir.exists()}")

    eips_dir = data_path / "EIPs"
    click.echo(f"\nEIPs: {eips_dir}")
    click.echo(f"  Exists: {eips_dir.exists()}")

    builder_specs_dir = data_path / "builder-specs"
    click.echo(f"\nBuilder specs: {builder_specs_dir}")
    click.echo(f"  Exists: {builder_specs_dir.exists()}")

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
            status = click.style("✓", fg="green")
            version = info["version"] or "unknown"
            click.echo(f"  {status} {name} ({info['language']}, {info['layer']}) - {version}")
        else:
            status = click.style("✗", fg="red")
            click.echo(f"  {status} {name} (not downloaded)")

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


if __name__ == "__main__":
    main()
