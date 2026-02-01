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

# Search
ethereum-mcp search "slashing penalty"
ethereum-mcp status      # Check index status
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
├── indexer/
│   ├── downloader.py   # Git clone specs/EIPs
│   ├── compiler.py     # Extract to JSON
│   ├── chunker.py      # Markdown chunking
│   └── embedder.py     # Embeddings + LanceDB
└── expert/
    └── guidance.py     # Curated interpretations
```

## Data Location

```
~/.ethereum-mcp/
├── consensus-specs/    # Cloned repo
├── EIPs/               # Cloned repo
├── compiled/           # JSON extracts per fork
└── lancedb/            # Vector index
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

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
