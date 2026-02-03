# ethereum-mcp

RAG-powered MCP server for Ethereum consensus specs and EIPs.

## Quick Reference

```bash
# Full build (download + compile + index)
ethereum-mcp build

# Individual steps
ethereum-mcp download    # Clone consensus-specs and EIPs
ethereum-mcp compile     # Extract constants/functions to JSON
ethereum-mcp index       # Build vector embeddings in LanceDB

# Incremental updates (fast)
ethereum-mcp update      # Git pull + incremental re-index
ethereum-mcp index       # Incremental by default
ethereum-mcp index --full    # Force full rebuild
ethereum-mcp index --dry-run # Preview what would change

# Search
ethereum-mcp search "slashing penalty"
ethereum-mcp status      # Check index status
ethereum-mcp models      # List available embedding models
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `eth_search` | Unified search across specs + EIPs (default: current fork) |
| `eth_get_current_fork` | Returns active fork name |
| `eth_list_forks` | All upgrades with dates/epochs |
| `eth_search_specs` | Specs-only search (no EIPs) |
| `eth_grep_constant` | Fast constant lookup |
| `eth_analyze_function` | Get Python implementation from specs |
| `eth_get_spec_version` | Index metadata |
| `eth_expert_guidance` | Curated expert interpretations |
| `eth_search_eip` | EIP-specific search |

## Project Structure

```
src/ethereum_mcp/
├── server.py           # MCP server (FastMCP)
├── cli.py              # CLI commands
├── config.py           # Configuration management
├── indexer/
│   ├── downloader.py   # Git clone specs/EIPs
│   ├── compiler.py     # Extract to JSON
│   ├── chunker.py      # Markdown chunking + chunk IDs
│   ├── embedder.py     # Embeddings + LanceDB + IncrementalEmbedder
│   └── manifest.py     # File tracking for incremental updates
└── expert/
    └── guidance.py     # Curated interpretations
```

## Data Location

```
~/.ethereum-mcp/
├── config.yaml         # Configuration (optional)
├── manifest.json       # Index state tracking
├── consensus-specs/    # Cloned repo
├── EIPs/               # Cloned repo
├── builder-specs/      # Cloned repo
├── compiled/           # JSON extracts per fork
└── lancedb/            # Vector index
```

## Configuration

Create `~/.ethereum-mcp/config.yaml` to customize:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"  # Or codesage/codesage-large
  batch_size: 32

chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

Available models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `codesage/codesage-large`, `voyage:voyage-code-3`

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## Incremental Indexing

See [docs/INCREMENTAL_INDEXING.md](docs/INCREMENTAL_INDEXING.md) for detailed documentation.

Key concepts:
- **Manifest**: Tracks file hashes and chunk IDs
- **Chunk IDs**: Deterministic IDs based on content hash
- **Change detection**: Fast mtime check, then hash verification
- **Delta updates**: Only re-embed changed content

## Adding Expert Guidance

Edit `src/ethereum_mcp/expert/guidance.py` to add entries:

```python
GUIDANCE_DB["topic"] = GuidanceEntry(
    topic="Topic Name",
    summary="Brief summary",
    key_points=["point 1", "point 2"],
    gotchas=["gotcha 1", "gotcha 2"],
    references=["specs/fork/file.md"],
)
```
