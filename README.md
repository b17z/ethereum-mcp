# ethereum-mcp

RAG-powered MCP server for Ethereum consensus specs, EIPs, and client source code.

## What It Does

Indexes and searches across:
- **Consensus Specs** - Official beacon chain specifications
- **EIPs** - Ethereum Improvement Proposals
- **Client Source Code** - All major EL and CL implementations
- **MEV Infrastructure** - Flashbots mev-boost, relays, builders, PBS specs

## Quick Start

```bash
# Install
pip install -e .

# Build specs index (fast)
ethereum-mcp build

# Build with client source code (slower, more complete)
ethereum-mcp build --include-clients
```

## CLI Commands

```bash
# Build pipeline
ethereum-mcp build                    # Specs + EIPs only
ethereum-mcp build --include-clients  # Include client source code

# Individual steps
ethereum-mcp download                 # Clone consensus-specs and EIPs
ethereum-mcp download --include-clients
ethereum-mcp download-clients         # Just download clients
ethereum-mcp download-clients --client reth --client lighthouse

ethereum-mcp compile                  # Extract specs to JSON
ethereum-mcp compile --include-clients

ethereum-mcp index                    # Build vector embeddings

# Search
ethereum-mcp search "slashing penalty"
ethereum-mcp search "attestation" --fork electra
ethereum-mcp search "EIP-4844" --limit 10

# Status
ethereum-mcp status                   # Show what's downloaded/compiled
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
| Geth | Go | Ethereum Foundation |
| Reth | Rust | Paradigm |
| Nethermind | C# | Nethermind |
| Besu | Java | Hyperledger/ConsenSys |
| Erigon | Go | Erigon |

### Consensus Layer Clients

| Client | Language | Organization |
|--------|----------|--------------|
| Prysm | Go | Offchain Labs |
| Lighthouse | Rust | Sigma Prime |
| Teku | Java | ConsenSys |
| Nimbus | Nim | Status |
| Lodestar | TypeScript | ChainSafe |

For current client diversity statistics, see [clientdiversity.org](https://clientdiversity.org).

### MEV Infrastructure (Flashbots)

| Repo | Language | Purpose |
|------|----------|---------|
| mev-boost | Go | Middleware connecting validators to builders |
| flashbots-builder | Go | Block builder (geth fork) |
| mev-boost-relay | Go | Relay implementation |
| builder-specs | Markdown | Builder API specifications |
| mev-share-node | Go | MEV-Share orderflow auctions |
| rbuilder | Rust | High-performance Rust block builder |

### Downloading Specific Clients

```bash
# Download just the Rust clients
ethereum-mcp download-clients --client reth --client lighthouse

# Download all execution layer clients
ethereum-mcp download-clients --client geth --client reth --client nethermind --client erigon

# Download MEV infrastructure
ethereum-mcp download-clients --client mev-boost --client flashbots-builder --client rbuilder
```

## Project Structure

```
src/ethereum_mcp/
├── server.py               # MCP server (FastMCP)
├── cli.py                  # CLI commands
├── clients.py              # Client tracking and diversity
├── indexer/
│   ├── downloader.py       # Git clone specs + clients
│   ├── compiler.py         # Spec extraction
│   ├── client_compiler.py  # Multi-language client parsing
│   ├── chunker.py          # Document chunking
│   └── embedder.py         # Embeddings + LanceDB
└── expert/
    └── guidance.py         # Curated interpretations
```

## Data Location

```
~/.ethereum-mcp/
├── consensus-specs/        # Cloned specs repo
├── EIPs/                   # Cloned EIPs repo
├── clients/                # Client + MEV source code
│   ├── reth/               # EL - Paradigm
│   ├── go-ethereum/        # EL - Geth
│   ├── lighthouse/         # CL - Sigma Prime
│   ├── prysm/              # CL - Offchain Labs
│   ├── mev-boost/          # MEV - Flashbots
│   ├── flashbots-builder/  # MEV - Block builder
│   ├── rbuilder/           # MEV - Rust builder
│   └── ...
├── compiled/               # Extracted JSON
│   ├── electra_spec.json
│   └── clients/
│       ├── reth/
│       ├── lighthouse/
│       └── mev-boost/
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

## Expert Guidance Topics

The `eth_expert_guidance` tool provides curated knowledge on:

**Validator Mechanics:**
- `churn` - Validator activation/exit queue mechanics
- `slashing` - Slashing penalties across forks
- `maxeb` - Maximum Effective Balance (Electra)
- `withdrawals` - Withdrawal mechanics

**MEV / PBS:**
- `mev` - MEV overview and sources
- `pbs` - Proposer-Builder Separation
- `epbs` - Enshrined PBS (future)
- `mev_boost` - MEV-boost architecture
- `flashbots` - Flashbots ecosystem

## Client Diversity

Client diversity is critical for network security. The `eth_get_client_diversity` tool provides:

- Current stake/node distribution
- Health assessment (>66% single client = risk)
- Recommendations for improving diversity

```python
# Example output
{
    "diversity_health": {
        "consensus_layer": "GOOD - No client >34%",
        "execution_layer": "MODERATE - Geth still >50%"
    },
    "supermajority_risk": "If any client >66%, a bug could cause incorrect finalization"
}
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install client parsing support
pip install -e ".[clients]"

# Run tests
pytest

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

## License

MIT
