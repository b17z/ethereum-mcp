"""Ethereum client tracking.

Tracks execution layer (EL) and consensus layer (CL) client implementations.
Ethereum has better client diversity than most chains - a key security feature.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EthereumClient:
    """An Ethereum client implementation."""

    name: str
    layer: str  # "execution", "consensus", "both"
    organization: str
    language: str
    repo: str
    description: str
    mainnet_status: str  # "production", "beta", "development"
    stake_percentage: float | None  # For CL clients
    node_percentage: float | None  # For EL clients
    key_features: list[str]
    notes: list[str]


# Execution Layer Clients
EXECUTION_CLIENTS: list[EthereumClient] = [
    EthereumClient(
        name="Geth",
        layer="execution",
        organization="Ethereum Foundation",
        language="Go",
        repo="https://github.com/ethereum/go-ethereum",
        description="Original and most widely used Ethereum execution client.",
        mainnet_status="production",
        stake_percentage=None,
        node_percentage=55.0,  # Approximate as of late 2025
        key_features=[
            "Reference implementation",
            "Most battle-tested",
            "Snap sync for fast initial sync",
            "Full and light node modes",
        ],
        notes=[
            "Historically >80% dominance, now improving",
            "Single Geth bug could halt network if >66%",
            "EF actively encouraging diversity",
        ],
    ),
    EthereumClient(
        name="Reth",
        layer="execution",
        organization="Paradigm",
        language="Rust",
        repo="https://github.com/paradigmxyz/reth",
        description="High-performance Rust implementation. Fast sync, modular architecture.",
        mainnet_status="production",
        stake_percentage=None,
        node_percentage=15.0,  # Growing rapidly
        key_features=[
            "Written in Rust for safety and performance",
            "Modular architecture (reth-* crates)",
            "Fast sync times",
            "Lower memory usage than Geth",
            "Active development by Paradigm",
        ],
        notes=[
            "Fastest growing EL client",
            "Production-ready since 2024",
            "Popular with infrastructure providers",
            "Excellent documentation",
        ],
    ),
    EthereumClient(
        name="Nethermind",
        layer="execution",
        organization="Nethermind",
        language="C#/.NET",
        repo="https://github.com/NethermindEth/nethermind",
        description="Enterprise-grade .NET implementation with extensive plugin system.",
        mainnet_status="production",
        stake_percentage=None,
        node_percentage=18.0,
        key_features=[
            ".NET ecosystem integration",
            "Extensive plugin system",
            "Good for enterprise deployments",
            "MEV-boost compatible",
        ],
        notes=[
            "Second most used EL client",
            "Popular with institutional stakers",
            "Good Windows support",
        ],
    ),
    EthereumClient(
        name="Besu",
        layer="execution",
        organization="Hyperledger / ConsenSys",
        language="Java",
        repo="https://github.com/hyperledger/besu",
        description="Enterprise Ethereum client, Apache 2.0 licensed.",
        mainnet_status="production",
        stake_percentage=None,
        node_percentage=8.0,
        key_features=[
            "Apache 2.0 license (enterprise-friendly)",
            "Privacy features (Tessera integration)",
            "Permissioning for private networks",
            "GraphQL API",
        ],
        notes=[
            "Popular for enterprise and consortium chains",
            "Hyperledger project governance",
            "Good for private Ethereum networks",
        ],
    ),
    EthereumClient(
        name="Erigon",
        layer="execution",
        organization="Erigon (formerly Turbo-Geth)",
        language="Go",
        repo="https://github.com/ledgerwatch/erigon",
        description="Efficiency-focused client optimized for archive nodes.",
        mainnet_status="production",
        stake_percentage=None,
        node_percentage=4.0,
        key_features=[
            "Extremely efficient disk usage",
            "Best for archive nodes",
            "Staged sync architecture",
            "Lower hardware requirements for full history",
        ],
        notes=[
            "Forked from Geth, heavily modified",
            "Preferred for archive node operators",
            "Different database structure (MDBX)",
        ],
    ),
]

# Consensus Layer Clients
CONSENSUS_CLIENTS: list[EthereumClient] = [
    EthereumClient(
        name="Prysm",
        layer="consensus",
        organization="Prysmatic Labs (Offchain Labs)",
        language="Go",
        repo="https://github.com/prysmaticlabs/prysm",
        description="Most popular consensus client, originally the first to mainnet.",
        mainnet_status="production",
        stake_percentage=35.0,
        node_percentage=None,
        key_features=[
            "First production CL client",
            "Slasher included",
            "Web UI for monitoring",
            "gRPC and REST APIs",
        ],
        notes=[
            "~35% of stake (down from >60% historically)",
            "Now part of Offchain Labs (Arbitrum)",
            "Diversity improving",
        ],
    ),
    EthereumClient(
        name="Lighthouse",
        layer="consensus",
        organization="Sigma Prime",
        language="Rust",
        repo="https://github.com/sigp/lighthouse",
        description="Rust implementation focused on security and performance.",
        mainnet_status="production",
        stake_percentage=33.0,
        node_percentage=None,
        key_features=[
            "Rust safety guarantees",
            "Security audit focused",
            "Efficient memory usage",
            "Good documentation",
        ],
        notes=[
            "~33% of stake - excellent diversity",
            "Security-focused development",
            "Popular with solo stakers",
        ],
    ),
    EthereumClient(
        name="Teku",
        layer="consensus",
        organization="ConsenSys",
        language="Java",
        repo="https://github.com/ConsenSys/teku",
        description="Enterprise-grade Java implementation, pairs well with Besu.",
        mainnet_status="production",
        stake_percentage=18.0,
        node_percentage=None,
        key_features=[
            "Java enterprise ecosystem",
            "Pairs naturally with Besu",
            "REST API focused",
            "Good institutional support",
        ],
        notes=[
            "~18% of stake",
            "ConsenSys enterprise offering",
            "Popular with institutions",
        ],
    ),
    EthereumClient(
        name="Nimbus",
        layer="consensus",
        organization="Status",
        language="Nim",
        repo="https://github.com/status-im/nimbus-eth2",
        description="Lightweight client designed for resource-constrained devices.",
        mainnet_status="production",
        stake_percentage=10.0,
        node_percentage=None,
        key_features=[
            "Extremely lightweight",
            "Can run on Raspberry Pi",
            "Low memory footprint",
            "Nim language (compiles to C)",
        ],
        notes=[
            "~10% of stake",
            "Best for home stakers with limited hardware",
            "Also developing EL client (nimbus-eth1)",
        ],
    ),
    EthereumClient(
        name="Lodestar",
        layer="consensus",
        organization="ChainSafe",
        language="TypeScript",
        repo="https://github.com/ChainSafe/lodestar",
        description="TypeScript implementation, great for JS/TS developers.",
        mainnet_status="production",
        stake_percentage=4.0,
        node_percentage=None,
        key_features=[
            "TypeScript/JavaScript ecosystem",
            "Good for web3 developers",
            "Light client focus",
            "Browser-compatible components",
        ],
        notes=[
            "~4% of stake",
            "Youngest production CL client",
            "Growing adoption",
        ],
    ),
]


def list_execution_clients() -> list[EthereumClient]:
    """List all execution layer clients."""
    return EXECUTION_CLIENTS


def list_consensus_clients() -> list[EthereumClient]:
    """List all consensus layer clients."""
    return CONSENSUS_CLIENTS


def list_all_clients() -> list[EthereumClient]:
    """List all Ethereum clients."""
    return EXECUTION_CLIENTS + CONSENSUS_CLIENTS


def get_client(name: str) -> EthereumClient | None:
    """Get a specific client by name."""
    name_lower = name.lower()
    for client in list_all_clients():
        if name_lower in client.name.lower():
            return client
    return None


def get_client_diversity() -> dict:
    """Get Ethereum client diversity statistics."""
    return {
        "execution_layer": {
            "Geth (Go)": "~55% - Reference implementation, still dominant",
            "Nethermind (C#)": "~18% - Enterprise-focused",
            "Reth (Rust)": "~15% - Fastest growing, Paradigm",
            "Besu (Java)": "~8% - Enterprise/Hyperledger",
            "Erigon (Go)": "~4% - Archive node specialist",
        },
        "consensus_layer": {
            "Prysm (Go)": "~35% - Prysmatic Labs/Offchain Labs",
            "Lighthouse (Rust)": "~33% - Sigma Prime",
            "Teku (Java)": "~18% - ConsenSys",
            "Nimbus (Nim)": "~10% - Status, lightweight",
            "Lodestar (TypeScript)": "~4% - ChainSafe",
        },
        "diversity_health": {
            "consensus_layer": "GOOD - No client >34% (supermajority threshold)",
            "execution_layer": "MODERATE - Geth still >50%, improving",
        },
        "recommendations": [
            "CL diversity is healthy - no client can cause finality failure alone",
            "EL still needs improvement - Geth bug could affect majority",
            "Reth growth is positive for EL diversity",
            "Run minority clients if possible to help the network",
        ],
        "supermajority_risk": (
            "If any client >66%, a bug in that client could cause "
            "incorrect finalization, requiring manual intervention to fix"
        ),
    }


def get_recommended_pairs() -> list[dict]:
    """Get recommended EL+CL client pairs."""
    return [
        {
            "pair": "Geth + Lighthouse",
            "notes": "Most common, very stable",
        },
        {
            "pair": "Reth + Lighthouse",
            "notes": "Modern stack, both Rust, good performance",
        },
        {
            "pair": "Nethermind + Prysm",
            "notes": "Enterprise-friendly combination",
        },
        {
            "pair": "Besu + Teku",
            "notes": "Both ConsenSys/Java, enterprise pairing",
        },
        {
            "pair": "Geth + Nimbus",
            "notes": "Good for resource-constrained setups",
        },
        {
            "pair": "Reth + Nimbus",
            "notes": "Minimal resource usage, modern stack",
        },
    ]


if __name__ == "__main__":
    print("Ethereum Execution Layer Clients:")
    print("=" * 60)
    for c in EXECUTION_CLIENTS:
        pct = f" ({c.node_percentage}%)" if c.node_percentage else ""
        print(f"  {c.name}{pct} - {c.language} - {c.organization}")

    print("\nEthereum Consensus Layer Clients:")
    print("=" * 60)
    for c in CONSENSUS_CLIENTS:
        pct = f" ({c.stake_percentage}%)" if c.stake_percentage else ""
        print(f"  {c.name}{pct} - {c.language} - {c.organization}")

    print("\nClient Diversity:")
    print("=" * 60)
    diversity = get_client_diversity()
    print(f"  EL: {diversity['diversity_health']['execution_layer']}")
    print(f"  CL: {diversity['diversity_health']['consensus_layer']}")
