"""MCP server for Ethereum specs search."""

import json
from pathlib import Path

from fastmcp import FastMCP
from pydantic import ValidationError

from .clients import (
    get_client,
    get_client_diversity,
    get_recommended_pairs,
    list_all_clients,
    list_consensus_clients,
    list_execution_clients,
)
from .expert.guidance import get_expert_guidance, list_guidance_topics
from .indexer.compiler import get_function_source
from .indexer.embedder import EmbeddingSearcher
from .logging import get_logger
from .models import (
    ClientListInput,
    ClientLookupInput,
    ConstantLookupInput,
    EipSearchInput,
    FunctionAnalysisInput,
    GuidanceInput,
    SearchInput,
)

logger = get_logger("server")

# Initialize MCP server
mcp = FastMCP("ethereum-mcp")

# Default paths
DEFAULT_DATA_DIR = Path.home() / ".ethereum-mcp"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "lancedb"
DEFAULT_SPECS_DIR = DEFAULT_DATA_DIR / "consensus-specs"

# Fork information
FORKS = {
    "phase0": {"epoch": 0, "date": "2020-12-01", "description": "Beacon chain genesis"},
    "altair": {"epoch": 74240, "date": "2021-10-27", "description": "Light clients, sync cttes"},
    "bellatrix": {"epoch": 144896, "date": "2022-09-06", "description": "Merge preparation"},
    "capella": {"epoch": 194048, "date": "2023-04-12", "description": "Withdrawals enabled"},
    "deneb": {"epoch": 269568, "date": "2024-03-13", "description": "Blobs (EIP-4844)"},
    "electra": {"epoch": 364032, "date": "2025-05-07", "description": "MaxEB, consolidations"},
    "fulu": {"epoch": 411392, "date": "2025-11-15", "description": "PeerDAS, verkle prep"},
}

CURRENT_FORK = "electra"  # Pectra = Prague + Electra


def get_searcher() -> EmbeddingSearcher:
    """Get or create embedding searcher."""
    return EmbeddingSearcher(DEFAULT_DB_PATH)


# GitHub URL mappings for source references
GITHUB_REPOS = {
    "consensus-specs": {
        "url": "https://github.com/ethereum/consensus-specs",
        "branch": "master",
    },
    "EIPs": {
        "url": "https://github.com/ethereum/EIPs",
        "branch": "master",
    },
    "builder-specs": {
        "url": "https://github.com/ethereum/builder-specs",
        "branch": "main",
    },
    "leanSpec": {
        "url": "https://github.com/leanEthereum/leanSpec",
        "branch": "main",
    },
    # leanEthereum post-quantum Rust repos
    "leanSig": {
        "url": "https://github.com/leanEthereum/leanSig",
        "branch": "main",
    },
    "leanMultisig": {
        "url": "https://github.com/leanEthereum/leanMultisig",
        "branch": "main",
    },
    "multilinear-toolkit": {
        "url": "https://github.com/leanEthereum/multilinear-toolkit",
        "branch": "main",
    },
    "fiat-shamir": {
        "url": "https://github.com/leanEthereum/fiat-shamir",
        "branch": "main",
    },
    "pm": {
        "url": "https://github.com/leanEthereum/pm",
        "branch": "main",
    },
    "leanSnappy": {
        "url": "https://github.com/leanEthereum/leanSnappy",
        "branch": "main",
    },
}


def _source_to_github_url(source_path: str, repo: str | None = None) -> str | None:
    """
    Convert a source path to a GitHub URL.

    Args:
        source_path: Repo-relative file path (e.g., specs/electra/beacon-chain.md)
        repo: Repository name (e.g., "consensus-specs", "EIPs", "builder-specs")

    Returns:
        GitHub blob URL or None if repo not recognized
    """
    if not source_path:
        return None

    # If repo is provided, use it directly
    if repo and repo in GITHUB_REPOS:
        repo_info = GITHUB_REPOS[repo]
        return f"{repo_info['url']}/blob/{repo_info['branch']}/{source_path}"

    # Fallback: try to extract repo from absolute path (backwards compatibility)
    path = Path(source_path)
    parts = path.parts

    for repo_name, repo_info in GITHUB_REPOS.items():
        if repo_name in parts:
            # Find the index of the repo name and extract the relative path
            idx = parts.index(repo_name)
            relative_path = "/".join(parts[idx + 1 :])
            return f"{repo_info['url']}/blob/{repo_info['branch']}/{relative_path}"

    return None


def _add_github_url(result: dict) -> dict:
    """Add github_url field to a search result if source path is recognized."""
    if "source" in result:
        repo = result.get("repo")
        github_url = _source_to_github_url(result["source"], repo=repo)
        if github_url:
            result["github_url"] = github_url
    return result


def _safe_path(base: Path, *parts: str) -> Path:
    """
    Safely construct a path, preventing directory traversal.

    Args:
        base: Base directory that must contain the result
        *parts: Path components to join

    Returns:
        Resolved path guaranteed to be under base

    Raises:
        ValueError: If the resulting path would escape base directory
    """
    result = (base / "/".join(parts)).resolve()
    base_resolved = base.resolve()
    if not str(result).startswith(str(base_resolved)):
        raise ValueError(f"Path traversal attempt: {parts}")
    return result


@mcp.tool()
def eth_health() -> dict:
    """
    Check health of the Ethereum MCP server.

    Verifies that the index exists and is queryable.
    Call this to diagnose issues with search functionality.

    Returns:
        Health status with component checks
    """
    status = {
        "healthy": True,
        "checks": {},
    }

    # Check index exists
    if DEFAULT_DB_PATH.exists():
        status["checks"]["index_exists"] = "ok"
    else:
        status["checks"]["index_exists"] = "missing"
        status["healthy"] = False

    # Check specs directory
    if DEFAULT_SPECS_DIR.exists():
        status["checks"]["specs_dir"] = "ok"
    else:
        status["checks"]["specs_dir"] = "missing"

    # Check compiled specs
    compiled_dir = DEFAULT_DATA_DIR / "compiled"
    if compiled_dir.exists() and any(compiled_dir.glob("*.json")):
        status["checks"]["compiled_specs"] = "ok"
    else:
        status["checks"]["compiled_specs"] = "missing"

    # Try a test query if index exists
    if status["checks"]["index_exists"] == "ok":
        try:
            searcher = get_searcher()
            searcher.search("test", limit=1)  # Just verify search works
            status["checks"]["search"] = "ok"
        except Exception as e:
            status["checks"]["search"] = f"error: {e}"
            status["healthy"] = False

    # Add paths for debugging
    status["paths"] = {
        "data_dir": str(DEFAULT_DATA_DIR),
        "db_path": str(DEFAULT_DB_PATH),
        "specs_dir": str(DEFAULT_SPECS_DIR),
    }

    return status


@mcp.tool()
def eth_get_current_fork() -> dict:
    """
    Get the current active Ethereum fork.

    Returns:
        Current fork name and metadata
    """
    return {
        "fork": CURRENT_FORK,
        "info": FORKS[CURRENT_FORK],
        "note": "Pectra = Prague (EL) + Electra (CL)",
    }


@mcp.tool()
def eth_list_forks() -> list[dict]:
    """
    List all major Ethereum network upgrades with dates and epochs.

    Returns:
        List of forks in chronological order
    """
    return [
        {"fork": name, **info, "current": name == CURRENT_FORK}
        for name, info in FORKS.items()
    ]


@mcp.tool()
def eth_search(
    query: str,
    fork: str | None = None,
    limit: int = 5,
    include_lean_spec: bool = False,
) -> list[dict]:
    """
    Search Ethereum specs and EIPs together.

    This is the primary search tool - it searches across all indexed
    content including consensus specs and EIPs.

    Args:
        query: Search query (e.g., "slashing penalty calculation")
        fork: Optional fork to filter by (defaults to current fork context)
        limit: Maximum results to return
        include_lean_spec: If True, also include leanSpec Python implementation results

    Returns:
        List of relevant chunks with source and similarity score
    """
    try:
        validated = SearchInput(query=query, fork=fork, limit=limit)
    except ValidationError as e:
        return [{"error": str(e)}]

    searcher = get_searcher()

    # Default to current fork context if not specified
    fork = validated.fork or CURRENT_FORK

    results = searcher.search(validated.query, limit=validated.limit, fork=fork)

    # Also search without fork filter to catch EIPs
    eip_results = searcher.search_eip(query)

    # Optionally include leanSpec results
    lean_results = []
    if include_lean_spec:
        lean_results = searcher.search_lean_spec(query, limit=validated.limit)

    # Merge and dedupe
    seen = set()
    merged = []
    for r in results + eip_results + lean_results:
        key = (r["source"], r["section"])
        if key not in seen:
            seen.add(key)
            merged.append(_add_github_url(r))

    return sorted(merged, key=lambda x: x["score"], reverse=True)[:limit]


@mcp.tool()
def eth_search_specs(query: str, fork: str | None = None, limit: int = 5) -> list[dict]:
    """
    Search consensus specs only (no EIPs).

    Use this when you specifically want spec content without EIP results.

    Args:
        query: Search query
        fork: Optional fork filter
        limit: Maximum results

    Returns:
        List of spec chunks
    """
    try:
        validated = SearchInput(query=query, fork=fork, limit=limit)
    except ValidationError as e:
        return [{"error": str(e)}]

    searcher = get_searcher()
    fork = validated.fork or CURRENT_FORK

    # Exclude EIP chunk type
    results = searcher.search(validated.query, limit=validated.limit * 2, fork=fork)
    return [_add_github_url(r) for r in results if r["chunk_type"] != "eip"][:validated.limit]


@mcp.tool()
def eth_grep_constant(constant_name: str, fork: str | None = None) -> dict | None:
    """
    Look up a specific constant value from the specs.

    Optimized for finding constants like MAX_EFFECTIVE_BALANCE,
    MIN_SLASHING_PENALTY_QUOTIENT, etc.

    Args:
        constant_name: Name of constant (e.g., "MAX_EFFECTIVE_BALANCE")
        fork: Fork to look up (defaults to current)

    Returns:
        Constant definition with value and context
    """
    try:
        validated = ConstantLookupInput(constant_name=constant_name, fork=fork)
    except ValidationError as e:
        return {"error": str(e)}

    fork = validated.fork or CURRENT_FORK

    # Try compiled specs first - use safe path construction
    try:
        spec_file = _safe_path(DEFAULT_DATA_DIR, "compiled", f"{fork}_spec.json")
    except ValueError:
        return {"error": "Invalid fork name"}

    if spec_file.exists():
        with open(spec_file) as f:
            spec_data = json.load(f)
            if validated.constant_name in spec_data.get("constants", {}):
                return {
                    "constant": validated.constant_name,
                    "value": spec_data["constants"][validated.constant_name],
                    "fork": fork,
                    "source": "compiled_spec",
                }

    # Fall back to embedding search
    searcher = get_searcher()
    results = searcher.search_constant(validated.constant_name)

    for r in results:
        if validated.constant_name.upper() in r["content"].upper():
            result = {
                "constant": validated.constant_name,
                "context": r["content"],
                "fork": r["fork"] or fork,
                "source": r["source"],
            }
            return _add_github_url(result)

    return None


@mcp.tool()
def eth_analyze_function(function_name: str, fork: str | None = None) -> dict | None:
    """
    Get the actual Python implementation of a spec function.

    Returns the complete function source code from the consensus specs.

    Args:
        function_name: Name of function (e.g., "process_slashings")
        fork: Fork version (defaults to current)

    Returns:
        Function source code and metadata
    """
    try:
        validated = FunctionAnalysisInput(function_name=function_name, fork=fork)
    except ValidationError as e:
        return {"error": str(e)}

    fork = validated.fork or CURRENT_FORK

    # Try to get from source files
    source = get_function_source(DEFAULT_SPECS_DIR, fork, validated.function_name)

    if source:
        return {
            "function": validated.function_name,
            "fork": fork,
            "source": source,
        }

    # Fall back to embedding search
    searcher = get_searcher()
    results = searcher.search_function(validated.function_name, fork=fork)

    if results:
        result = {
            "function": validated.function_name,
            "fork": results[0]["fork"] or fork,
            "source": results[0]["content"],
            "file": results[0]["source"],
        }
        github_url = _source_to_github_url(results[0]["source"])
        if github_url:
            result["github_url"] = github_url
        return result

    return None


@mcp.tool()
def eth_get_spec_version() -> dict:
    """
    Get metadata about indexed spec versions.

    Returns:
        Version info for indexed specs and EIPs
    """
    import subprocess

    version_info = {
        "indexed_forks": list(FORKS.keys()),
        "current_fork": CURRENT_FORK,
    }

    # Try to get git commit info
    if DEFAULT_SPECS_DIR.exists():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=DEFAULT_SPECS_DIR,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                version_info["consensus_specs_commit"] = result.stdout.strip()
            else:
                logger.debug("Git rev-parse failed: %s", result.stderr)
        except Exception as e:
            logger.debug("Failed to get git commit info: %s", e)

    # Check if index exists
    db_path = DEFAULT_DB_PATH
    if db_path.exists():
        version_info["index_path"] = str(db_path)
        version_info["index_exists"] = True
    else:
        version_info["index_exists"] = False

    return version_info


@mcp.tool()
def eth_expert_guidance(topic: str) -> dict | None:
    """
    Get curated expert interpretations on Ethereum topics.

    This provides nuanced guidance that goes beyond what's explicitly
    in the specs - including common gotchas, implementation notes,
    and insights from EF discussions.

    Args:
        topic: Topic to get guidance on (e.g., "slashing", "churn", "withdrawals")

    Returns:
        Expert guidance with key points and references
    """
    try:
        validated = GuidanceInput(topic=topic)
    except ValidationError as e:
        return {"error": str(e)}

    guidance = get_expert_guidance(validated.topic)

    if guidance:
        return {
            "topic": validated.topic,
            "guidance": guidance,
        }

    # List available topics if not found
    available = list_guidance_topics()
    return {
        "topic": validated.topic,
        "error": f"No guidance found for '{validated.topic}'",
        "available_topics": available,
    }


@mcp.tool()
def eth_search_eip(query: str, eip_number: str | None = None, limit: int = 5) -> list[dict]:
    """
    Search EIPs specifically.

    Args:
        query: Search query
        eip_number: Optional specific EIP number to search within
        limit: Maximum results

    Returns:
        List of relevant EIP chunks
    """
    try:
        validated = EipSearchInput(query=query, eip_number=eip_number, limit=limit)
    except ValidationError as e:
        return [{"error": str(e)}]

    searcher = get_searcher()
    results = searcher.search_eip(validated.query)

    if validated.eip_number:
        results = [r for r in results if r.get("eip") == validated.eip_number]

    return [_add_github_url(r) for r in results[:validated.limit]]


@mcp.tool()
def eth_search_lean_spec(query: str, limit: int = 5) -> list[dict]:
    """
    Search leanSpec Python consensus specification.

    leanSpec is a Python implementation of the Ethereum consensus specification
    using Pydantic/Container classes. It provides a different perspective from
    the official markdown specs, with executable Python code.

    Args:
        query: Search query (e.g., "forkchoice store", "State container")
        limit: Maximum results to return

    Returns:
        List of relevant leanSpec chunks with source and score
    """
    try:
        validated = SearchInput(query=query, limit=limit)
    except ValidationError as e:
        return [{"error": str(e)}]

    searcher = get_searcher()
    results = searcher.search_lean_spec(validated.query, limit=validated.limit)

    return [_add_github_url(r) for r in results]


@mcp.tool()
def eth_search_lean_pq(query: str, limit: int = 5, repo: str | None = None) -> list[dict]:
    """
    Search leanEthereum post-quantum Rust implementations.

    leanEthereum develops post-quantum cryptography for Ethereum including:
    - leanSig: PQ signature implementation
    - leanMultisig: XMSS aggregation zkVM for post-quantum multisig
    - multilinear-toolkit: Crypto support library
    - fiat-shamir: Fiat-Shamir crypto support

    Args:
        query: Search query (e.g., "XMSS signature", "multilinear polynomial")
        limit: Maximum results to return
        repo: Optional filter by repo (leanSig, leanMultisig, multilinear-toolkit, fiat-shamir)

    Returns:
        List of relevant Rust code chunks with source and score
    """
    try:
        validated = SearchInput(query=query, limit=limit)
    except ValidationError as e:
        return [{"error": str(e)}]

    # Validate repo if provided
    valid_repos = {"leanSig", "leanMultisig", "multilinear-toolkit", "fiat-shamir"}
    if repo and repo not in valid_repos:
        return [{
            "error": f"Invalid repo: {repo}",
            "valid_repos": list(valid_repos),
        }]

    searcher = get_searcher()
    results = searcher.search_lean_rust(validated.query, limit=validated.limit, repo=repo)

    return [_add_github_url(r) for r in results]


@mcp.tool()
def eth_list_clients(layer: str | None = None) -> list[dict]:
    """
    List Ethereum client implementations.

    Ethereum has separate execution layer (EL) and consensus layer (CL) clients
    that run together. Client diversity is critical for network security.

    Args:
        layer: Filter by "execution" or "consensus" (default: both)

    Returns:
        List of clients with their details
    """
    try:
        validated = ClientListInput(layer=layer)
    except ValidationError as e:
        return [{"error": str(e)}]

    if validated.layer == "execution":
        clients = list_execution_clients()
    elif validated.layer == "consensus":
        clients = list_consensus_clients()
    else:
        clients = list_all_clients()

    return [
        {
            "name": c.name,
            "layer": c.layer,
            "organization": c.organization,
            "language": c.language,
            "repo": c.repo,
            "status": c.mainnet_status,
            "percentage": c.node_percentage or c.stake_percentage,
        }
        for c in clients
    ]


@mcp.tool()
def eth_get_client(name: str) -> dict | None:
    """
    Get details about a specific Ethereum client.

    Args:
        name: Client name (e.g., "geth", "reth", "lighthouse", "prysm")

    Returns:
        Full client details including features and notes
    """
    try:
        validated = ClientLookupInput(name=name)
    except ValidationError as e:
        return {"error": str(e)}

    client = get_client(validated.name)
    if not client:
        all_clients = list_all_clients()
        return {
            "error": f"Client '{validated.name}' not found",
            "available": [c.name for c in all_clients],
        }

    return {
        "name": client.name,
        "layer": client.layer,
        "organization": client.organization,
        "language": client.language,
        "repo": client.repo,
        "description": client.description,
        "status": client.mainnet_status,
        "percentage": client.node_percentage or client.stake_percentage,
        "key_features": client.key_features,
        "notes": client.notes,
    }


@mcp.tool()
def eth_get_client_diversity() -> dict:
    """
    Get Ethereum client diversity statistics.

    Client diversity is critical for network security. If any client has >66% share,
    a bug in that client could cause incorrect finalization.

    Returns:
        Diversity statistics for EL and CL, health assessment, and recommendations
    """
    return get_client_diversity()


@mcp.tool()
def eth_get_recommended_client_pairs() -> list[dict]:
    """
    Get recommended execution + consensus client pairings.

    Validators need both an EL and CL client. Some combinations work better together.

    Returns:
        List of recommended EL+CL pairs with notes
    """
    return get_recommended_pairs()


def run():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
