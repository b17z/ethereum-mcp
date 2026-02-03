# ethereum-mcp

RAG-powered MCP server for Ethereum consensus specs, EIPs, and client source code.

[![PyPI version](https://badge.fury.io/py/ethereum-mcp.svg)](https://badge.fury.io/py/ethereum-mcp)
[![CI](https://github.com/be-nvy/ethereum-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/be-nvy/ethereum-mcp/actions/workflows/ci.yml)

## What It Does

Indexes and searches across:
- [Consensus Specs](https://github.com/ethereum/consensus-specs) - Official beacon chain specifications
- [EIPs](https://github.com/ethereum/EIPs) - Ethereum Improvement Proposals
- [Builder Specs](https://github.com/ethereum/builder-specs) - MEV-boost and PBS specifications
- Client Source Code - All major EL and CL implementations

## Installation

```bash
# From PyPI
pip install ethereum-mcp

# From source
pip install -e .

# With client code parsing support
pip install -e ".[clients]"

# With Voyage API embeddings (best quality)
pip install -e ".[voyage]"
```

## Quick Start

```bash
# Build the index (downloads specs + creates embeddings)
ethereum-mcp build

# Search the specs
ethereum-mcp search "slashing penalty"

# Check status
ethereum-mcp status
```

## Features

### Incremental Indexing

The v0.2.0 release introduces **incremental indexing** - only re-embeds changed files instead of rebuilding the entire index. This reduces update time from minutes to seconds.

```bash
# Update repos and incrementally re-index (fast!)
ethereum-mcp update

# Incremental index (default behavior)
ethereum-mcp index

# Preview what would change without indexing
ethereum-mcp index --dry-run

# Force full rebuild
ethereum-mcp index --full
```

**How it works:**
1. Tracks file hashes and modification times in a manifest
2. Detects which files changed since last index
3. Only re-embeds the changed content
4. Updates LanceDB incrementally (add/delete operations)

### Configurable Embedding Models

Choose from multiple embedding models based on your quality/speed tradeoff:

```bash
# List available models
ethereum-mcp models

# Use a specific model
ethereum-mcp index --model codesage/codesage-large
```

| Model | Dims | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `all-MiniLM-L6-v2` | 384 | Fair | Fast | Default, good for quick searches |
| `all-mpnet-base-v2` | 768 | Good | Medium | Better quality |
| `codesage/codesage-large` | 1024 | Good | Medium | Code-specialized |
| `voyage:voyage-code-3` | 1024 | Excellent | API | Best quality, requires API key |

Configure in `~/.ethereum-mcp/config.yaml`:

```yaml
embedding:
  model: "codesage/codesage-large"
  batch_size: 32

chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

### Expert Guidance

Curated knowledge beyond what's in the specs:

```bash
# Via CLI (when running as MCP server)
eth_expert_guidance("slashing")
eth_expert_guidance("mev")
eth_expert_guidance("maxeb")
```

Topics include: `churn`, `slashing`, `maxeb`, `withdrawals`, `mev`, `pbs`, `epbs`, `mev_boost`, `flashbots`

## CLI Commands

```bash
# Full build pipeline
ethereum-mcp build                    # Specs + EIPs only
ethereum-mcp build --include-clients  # Include client source code
ethereum-mcp build --full             # Force full rebuild

# Individual steps
ethereum-mcp download                 # Clone consensus-specs, EIPs, builder-specs
ethereum-mcp download --include-clients
ethereum-mcp compile                  # Extract specs to JSON
ethereum-mcp index                    # Build vector embeddings
ethereum-mcp index --dry-run          # Preview changes
ethereum-mcp index --full             # Force full rebuild
ethereum-mcp index --model MODEL      # Use specific embedding model

# Update (git pull + incremental index)
ethereum-mcp update
ethereum-mcp update --full            # Update + force rebuild

# Search
ethereum-mcp search "slashing penalty"
ethereum-mcp search "attestation" --fork electra
ethereum-mcp search "EIP-4844" --limit 10

# Info
ethereum-mcp status                   # Index status, manifest info
ethereum-mcp models                   # List embedding models
```

## MCP Tools

When running as an MCP server:

| Tool | Purpose |
|------|---------|
| `eth_search` | Unified search across specs + EIPs |
| `eth_search_specs` | Specs-only search (no EIPs) |
| `eth_search_eip` | EIP-specific search |
| `eth_grep_constant` | Fast constant lookup |
| `eth_analyze_function` | Get Python implementation from specs |
| `eth_get_current_fork` | Current fork (Electra/Pectra) |
| `eth_list_forks` | All upgrades with dates/epochs |
| `eth_get_spec_version` | Index metadata |
| `eth_expert_guidance` | Curated expert interpretations |
| `eth_list_clients` | List all EL/CL clients |
| `eth_get_client` | Details on specific client |
| `eth_get_client_diversity` | Diversity stats and health |
| `eth_get_recommended_client_pairs` | EL+CL pairing recommendations |

## Client Source Code

### Execution Layer Clients

| Client | Language | Organization |
|--------|----------|--------------|
| [Geth](https://github.com/ethereum/go-ethereum) | Go | Ethereum Foundation |
| [Reth](https://github.com/paradigmxyz/reth) | Rust | Paradigm |
| [Nethermind](https://github.com/NethermindEth/nethermind) | C# | Nethermind |
| [Erigon](https://github.com/ledgerwatch/erigon) | Go | Erigon |

### Consensus Layer Clients

| Client | Language | Organization |
|--------|----------|--------------|
| [Prysm](https://github.com/prysmaticlabs/prysm) | Go | Offchain Labs |
| [Lighthouse](https://github.com/sigp/lighthouse) | Rust | Sigma Prime |
| [Teku](https://github.com/ConsenSys/teku) | Java | ConsenSys |
| [Nimbus](https://github.com/status-im/nimbus-eth2) | Nim | Status |

### MEV Infrastructure

| Project | Description |
|---------|-------------|
| [mev-boost](https://github.com/flashbots/mev-boost) | MEV-boost middleware |
| [Flashbots Builder](https://github.com/flashbots/builder) | Block builder reference |
| [mev-boost-relay](https://github.com/flashbots/mev-boost-relay) | Relay implementation |
| [rbuilder](https://github.com/flashbots/rbuilder) | Rust block builder |

```bash
# Download specific clients
ethereum-mcp download-clients --client reth --client lighthouse

# Download MEV infrastructure
ethereum-mcp download-clients --client mev-boost --client flashbots-builder
```

## Project Structure

```
src/ethereum_mcp/
├── server.py               # MCP server (FastMCP)
├── cli.py                  # CLI commands
├── config.py               # Configuration management
├── clients.py              # Client tracking and diversity
├── indexer/
│   ├── downloader.py       # Git clone specs + clients
│   ├── compiler.py         # Spec extraction
│   ├── client_compiler.py  # Multi-language client parsing
│   ├── chunker.py          # Document chunking + chunk IDs
│   ├── embedder.py         # Embeddings + LanceDB + incremental
│   └── manifest.py         # File tracking for incremental updates
└── expert/
    └── guidance.py         # Curated interpretations
```

## Data Location

```
~/.ethereum-mcp/
├── config.yaml             # Configuration (optional)
├── manifest.json           # Index state tracking
├── consensus-specs/        # Cloned specs repo
├── EIPs/                   # Cloned EIPs repo
├── builder-specs/          # Cloned builder-specs repo
├── clients/                # Client source code (optional)
├── compiled/               # Extracted JSON
└── lancedb/                # Vector index
```

## Fork History

| Fork | Epoch | Date | Description |
|------|-------|------|-------------|
| Phase0 | 0 | 2020-12-01 | Beacon chain genesis |
| Altair | 74240 | 2021-10-27 | Light clients, sync committees |
| Bellatrix | 144896 | 2022-09-06 | Merge preparation |
| Capella | 194048 | 2023-04-12 | Withdrawals enabled |
| Deneb | 269568 | 2024-03-13 | Proto-danksharding (EIP-4844) |
| Electra | 364032 | 2025-05-07 | MaxEB, consolidations |
| Fulu | 411392 | 2025-11-15 | PeerDAS, verkle prep |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ethereum_mcp

# Lint
ruff check src/
```

## Running as MCP Server

```bash
# Start the server
eth-mcp

# Or with uvicorn for development
uvicorn ethereum_mcp.server:mcp --reload
```

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "ethereum": {
      "command": "eth-mcp"
    }
  }
}
```

## Documentation

- [Incremental Indexing](docs/INCREMENTAL_INDEXING.md) - How the incremental update system works
- [CLAUDE.md](CLAUDE.md) - Quick reference for Claude Code

## License

MIT
