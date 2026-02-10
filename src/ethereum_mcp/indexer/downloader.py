"""Download Ethereum specs, EIPs, client source code, and MEV infrastructure from GitHub.

Repositories:
- ethereum/consensus-specs: Consensus layer specifications
- ethereum/EIPs: Ethereum Improvement Proposals

Execution Layer Clients:
- paradigmxyz/reth: Rust execution client (Paradigm)
- ethereum/go-ethereum: Go execution client (Geth)
- NethermindEth/nethermind: C# execution client
- ledgerwatch/erigon: Go execution client (archive-focused)

Consensus Layer Clients:
- sigp/lighthouse: Rust consensus client (Sigma Prime)
- prysmaticlabs/prysm: Go consensus client
- ConsenSys/teku: Java consensus client
- status-im/nimbus-eth2: Nim consensus client

MEV Infrastructure (Flashbots):
- flashbots/mev-boost: MEV-boost middleware (connects validators to builders)
- flashbots/builder: Block builder reference implementation
- flashbots/mev-boost-relay: Relay implementation
- flashbots/flashbots-protect-rpc: RPC for private transactions
- ethereum/builder-specs: Builder API specifications
"""

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from git import Repo

from ..logging import get_logger

logger = get_logger("downloader")


@dataclass(frozen=True)
class SpecsConfig:
    """Configuration for specs download."""

    consensus_specs_url: str = "https://github.com/ethereum/consensus-specs.git"
    eips_url: str = "https://github.com/ethereum/EIPs.git"
    builder_specs_url: str = "https://github.com/ethereum/builder-specs.git"
    lean_spec_url: str = "https://github.com/leanEthereum/leanSpec.git"
    consensus_specs_branch: str = "master"
    eips_branch: str = "master"
    builder_specs_branch: str = "main"
    lean_spec_branch: str = "main"
    # leanEthereum post-quantum Rust repos
    lean_sig_url: str = "https://github.com/leanEthereum/leanSig.git"
    lean_multisig_url: str = "https://github.com/leanEthereum/leanMultisig.git"
    multilinear_toolkit_url: str = "https://github.com/leanEthereum/multilinear-toolkit.git"
    fiat_shamir_url: str = "https://github.com/leanEthereum/fiat-shamir.git"
    lean_pm_url: str = "https://github.com/leanEthereum/pm.git"
    lean_snappy_url: str = "https://github.com/leanEthereum/leanSnappy.git"
    lean_sig_branch: str = "main"
    lean_multisig_branch: str = "main"
    multilinear_toolkit_branch: str = "main"
    fiat_shamir_branch: str = "main"
    lean_pm_branch: str = "main"
    lean_snappy_branch: str = "main"


# Client repositories to index
# Using sparse checkout for large repos to only get relevant code
CLIENT_REPOS = {
    # Execution Layer Clients
    "reth": {
        "url": "https://github.com/paradigmxyz/reth.git",
        "branch": "main",
        "language": "rust",
        "layer": "execution",
        "sparse_paths": [
            "crates/consensus",
            "crates/engine",
            "crates/evm",
            "crates/execution",
            "crates/net",
            "crates/payload",
            "crates/primitives",
            "crates/rpc",
            "crates/stages",
            "crates/storage",
            "crates/transaction-pool",
            "crates/trie",
        ],
    },
    "go-ethereum": {
        "url": "https://github.com/ethereum/go-ethereum.git",
        "branch": "master",
        "language": "go",
        "layer": "execution",
        "sparse_paths": [
            "consensus",
            "core",
            "eth",
            "ethdb",
            "miner",
            "node",
            "p2p",
            "params",
            "rpc",
            "trie",
        ],
    },
    "nethermind": {
        "url": "https://github.com/NethermindEth/nethermind.git",
        "branch": "master",
        "language": "csharp",
        "layer": "execution",
        "sparse_paths": [
            "src/Nethermind/Nethermind.Consensus",
            "src/Nethermind/Nethermind.Core",
            "src/Nethermind/Nethermind.Evm",
            "src/Nethermind/Nethermind.JsonRpc",
            "src/Nethermind/Nethermind.Merge.Plugin",
            "src/Nethermind/Nethermind.Network",
            "src/Nethermind/Nethermind.State",
            "src/Nethermind/Nethermind.Trie",
        ],
    },
    "erigon": {
        "url": "https://github.com/ledgerwatch/erigon.git",
        "branch": "main",
        "language": "go",
        "layer": "execution",
        "sparse_paths": [
            "consensus",
            "core",
            "erigon-lib",
            "eth",
            "ethdb",
            "turbo",
        ],
    },
    # Consensus Layer Clients
    "lighthouse": {
        "url": "https://github.com/sigp/lighthouse.git",
        "branch": "stable",
        "language": "rust",
        "layer": "consensus",
        "sparse_paths": [
            "beacon_node/beacon_chain",
            "beacon_node/client",
            "beacon_node/execution_layer",
            "beacon_node/lighthouse_network",
            "beacon_node/network",
            "beacon_node/store",
            "consensus/fork_choice",
            "consensus/state_processing",
            "consensus/types",
            "slasher",
            "validator_client",
        ],
    },
    "prysm": {
        "url": "https://github.com/prysmaticlabs/prysm.git",
        "branch": "develop",
        "language": "go",
        "layer": "consensus",
        "sparse_paths": [
            "beacon-chain/blockchain",
            "beacon-chain/core",
            "beacon-chain/db",
            "beacon-chain/execution",
            "beacon-chain/forkchoice",
            "beacon-chain/operations",
            "beacon-chain/p2p",
            "beacon-chain/slasher",
            "beacon-chain/state",
            "beacon-chain/sync",
            "proto",
            "validator",
        ],
    },
    "teku": {
        "url": "https://github.com/ConsenSys/teku.git",
        "branch": "master",
        "language": "java",
        "layer": "consensus",
        "sparse_paths": [
            "ethereum/spec",
            "ethereum/statetransition",
            "ethereum/executionclient",
            "ethereum/executionlayer",
            "ethereum/networks",
            "ethereum/pow",
            "networking",
            "storage",
            "validator",
        ],
    },
    "nimbus-eth2": {
        "url": "https://github.com/status-im/nimbus-eth2.git",
        "branch": "stable",
        "language": "nim",
        "layer": "consensus",
        "sparse_paths": [
            "beacon_chain",
            "ncli",
            "research",
        ],
    },
    # ===================
    # MEV Infrastructure
    # ===================
    "mev-boost": {
        "url": "https://github.com/flashbots/mev-boost.git",
        "branch": "develop",
        "language": "go",
        "layer": "mev",
        "sparse_paths": None,  # Small repo, full clone
        "description": "MEV-boost middleware connecting validators to block builders",
    },
    "flashbots-builder": {
        "url": "https://github.com/flashbots/builder.git",
        "branch": "main",
        "language": "go",
        "layer": "mev",
        "sparse_paths": [
            "builder",
            "core",
            "eth",
            "miner",
            "flashbotsextra",
        ],
        "description": "Flashbots block builder (geth fork)",
    },
    "mev-boost-relay": {
        "url": "https://github.com/flashbots/mev-boost-relay.git",
        "branch": "main",
        "language": "go",
        "layer": "mev",
        "sparse_paths": None,  # Full clone
        "description": "MEV-boost relay for connecting builders to proposers",
    },
    "builder-specs": {
        "url": "https://github.com/ethereum/builder-specs.git",
        "branch": "main",
        "language": "markdown",
        "layer": "mev",
        "sparse_paths": None,  # Specs repo, full clone
        "description": "Builder API specifications for PBS",
    },
    "mev-share-node": {
        "url": "https://github.com/flashbots/mev-share-node.git",
        "branch": "main",
        "language": "go",
        "layer": "mev",
        "sparse_paths": None,
        "description": "MEV-Share node for orderflow auctions",
    },
    "rbuilder": {
        "url": "https://github.com/flashbots/rbuilder.git",
        "branch": "develop",
        "language": "rust",
        "layer": "mev",
        "sparse_paths": [
            "crates/rbuilder",
            "crates/op-rbuilder",
        ],
        "description": "Rust block builder by Flashbots (high performance)",
    },
}


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a git command."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result


def clone_client_repo(
    name: str,
    url: str,
    dest: Path,
    branch: str = "main",
    sparse_paths: list[str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> bool:
    """
    Clone a client repository with sparse checkout.

    Args:
        name: Repository name for logging
        url: Git URL to clone
        dest: Destination path
        branch: Branch to checkout
        sparse_paths: Paths to checkout (sparse checkout)
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    if dest.exists():
        log(f"  {name}: Already exists, pulling latest...")
        result = run_git(["pull", "--ff-only"], cwd=dest)
        if result.returncode != 0:
            log(f"  {name}: Pull failed, trying fetch + reset...")
            run_git(["fetch", "origin"], cwd=dest)
            run_git(["reset", "--hard", f"origin/{branch}"], cwd=dest)
        return True

    log(f"  {name}: Cloning from {url}...")

    if sparse_paths:
        # Sparse checkout for large repos
        dest.mkdir(parents=True, exist_ok=True)

        # Initialize repo
        run_git(["init"], cwd=dest)
        run_git(["remote", "add", "origin", url], cwd=dest)

        # Configure sparse checkout
        run_git(["config", "core.sparseCheckout", "true"], cwd=dest)

        # Write sparse-checkout file
        sparse_file = dest / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text("\n".join(sparse_paths) + "\n")

        # Fetch and checkout
        log(f"  {name}: Fetching (sparse checkout: {len(sparse_paths)} paths)...")
        result = run_git(["fetch", "--depth=1", "origin", branch], cwd=dest)
        if result.returncode != 0:
            log(f"  {name}: Fetch failed: {result.stderr}")
            return False

        result = run_git(["checkout", branch], cwd=dest)
        if result.returncode != 0:
            log(f"  {name}: Checkout failed: {result.stderr}")
            return False
    else:
        # Full clone for small repos
        result = run_git(["clone", "--depth=1", "--branch", branch, url, str(dest)])
        if result.returncode != 0:
            log(f"  {name}: Clone failed: {result.stderr}")
            return False

    log(f"  {name}: Done")
    return True


def download_specs(
    data_dir: Path,
    config: SpecsConfig | None = None,
    force: bool = False,
    include_lean_pq: bool = True,
) -> tuple[Path, Path, Path, Path]:
    """
    Download consensus specs, EIPs, builder-specs, leanSpec, and leanEthereum PQ repos.

    Args:
        data_dir: Directory to store downloaded repos
        config: Optional configuration override
        force: If True, re-download even if exists
        include_lean_pq: If True, also download leanEthereum post-quantum repos

    Returns:
        Tuple of (consensus_specs_path, eips_path, builder_specs_path, lean_spec_path)
    """
    config = config or SpecsConfig()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    consensus_dir = data_dir / "consensus-specs"
    eips_dir = data_dir / "EIPs"

    # Download consensus specs
    if force and consensus_dir.exists():
        shutil.rmtree(consensus_dir)

    if not consensus_dir.exists():
        logger.info("Cloning consensus-specs to %s...", consensus_dir)
        Repo.clone_from(
            config.consensus_specs_url,
            consensus_dir,
            branch=config.consensus_specs_branch,
            depth=1,
        )
    else:
        logger.info("Consensus specs already exists at %s, pulling latest...", consensus_dir)
        repo = Repo(consensus_dir)
        repo.remotes.origin.pull()

    # Download EIPs
    if force and eips_dir.exists():
        shutil.rmtree(eips_dir)

    if not eips_dir.exists():
        logger.info("Cloning EIPs to %s...", eips_dir)
        Repo.clone_from(
            config.eips_url,
            eips_dir,
            branch=config.eips_branch,
            depth=1,
        )
    else:
        logger.info("EIPs already exists at %s, pulling latest...", eips_dir)
        repo = Repo(eips_dir)
        repo.remotes.origin.pull()

    # Download builder-specs
    builder_specs_dir = data_dir / "builder-specs"
    if force and builder_specs_dir.exists():
        shutil.rmtree(builder_specs_dir)

    if not builder_specs_dir.exists():
        logger.info("Cloning builder-specs to %s...", builder_specs_dir)
        Repo.clone_from(
            config.builder_specs_url,
            builder_specs_dir,
            branch=config.builder_specs_branch,
            depth=1,
        )
    else:
        logger.info("Builder specs already exists at %s, pulling latest...", builder_specs_dir)
        repo = Repo(builder_specs_dir)
        repo.remotes.origin.pull()

    # Download leanSpec
    lean_spec_dir = data_dir / "leanSpec"
    if force and lean_spec_dir.exists():
        shutil.rmtree(lean_spec_dir)

    if not lean_spec_dir.exists():
        logger.info("Cloning leanSpec to %s...", lean_spec_dir)
        Repo.clone_from(
            config.lean_spec_url,
            lean_spec_dir,
            branch=config.lean_spec_branch,
            depth=1,
        )
    else:
        logger.info("leanSpec already exists at %s, pulling latest...", lean_spec_dir)
        repo = Repo(lean_spec_dir)
        repo.remotes.origin.pull()

    # Download leanEthereum post-quantum repos
    if include_lean_pq:
        lean_pq_repos = [
            ("leanSig", config.lean_sig_url, config.lean_sig_branch),
            ("leanMultisig", config.lean_multisig_url, config.lean_multisig_branch),
            (
                "multilinear-toolkit",
                config.multilinear_toolkit_url,
                config.multilinear_toolkit_branch,
            ),
            ("fiat-shamir", config.fiat_shamir_url, config.fiat_shamir_branch),
            ("pm", config.lean_pm_url, config.lean_pm_branch),
            ("leanSnappy", config.lean_snappy_url, config.lean_snappy_branch),
        ]

        for repo_name, repo_url, repo_branch in lean_pq_repos:
            repo_dir = data_dir / repo_name
            if force and repo_dir.exists():
                shutil.rmtree(repo_dir)

            if not repo_dir.exists():
                logger.info("Cloning %s to %s...", repo_name, repo_dir)
                try:
                    Repo.clone_from(
                        repo_url,
                        repo_dir,
                        branch=repo_branch,
                        depth=1,
                    )
                except Exception as e:
                    logger.warning("Failed to clone %s: %s", repo_name, e)
            else:
                logger.info("%s already exists at %s, pulling latest...", repo_name, repo_dir)
                try:
                    repo = Repo(repo_dir)
                    repo.remotes.origin.pull()
                except Exception as e:
                    logger.warning("Failed to pull %s: %s", repo_name, e)

    return consensus_dir, eips_dir, builder_specs_dir, lean_spec_dir


def download_clients(
    data_dir: Path,
    clients: list[str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, bool]:
    """
    Download Ethereum client source code.

    Args:
        data_dir: Directory to store downloaded repos
        clients: List of client names to download (default: all)
        progress_callback: Optional callback for progress updates

    Returns:
        Dict mapping client name to success status
    """
    data_dir = Path(data_dir)
    clients_dir = data_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)

    if clients is None:
        clients = list(CLIENT_REPOS.keys())

    results = {}
    for name in clients:
        if name not in CLIENT_REPOS:
            if progress_callback:
                progress_callback(f"  {name}: Unknown client, skipping")
            results[name] = False
            continue

        config = CLIENT_REPOS[name]
        dest = clients_dir / name

        success = clone_client_repo(
            name=name,
            url=config["url"],
            dest=dest,
            branch=config["branch"],
            sparse_paths=config.get("sparse_paths"),
            progress_callback=progress_callback,
        )
        results[name] = success

    return results


def list_downloaded_clients(data_dir: Path) -> dict[str, dict]:
    """List all downloaded client repositories with their status."""
    clients_dir = data_dir / "clients"

    status = {}
    for name, config in CLIENT_REPOS.items():
        path = clients_dir / name
        if path.exists():
            result = run_git(["rev-parse", "HEAD"], cwd=path)
            version = result.stdout.strip()[:12] if result.returncode == 0 else None
            status[name] = {
                "path": str(path),
                "version": version,
                "exists": True,
                "language": config["language"],
                "layer": config["layer"],
            }
        else:
            status[name] = {
                "path": str(path),
                "version": None,
                "exists": False,
                "language": config["language"],
                "layer": config["layer"],
            }

    return status


def get_spec_files(consensus_dir: Path) -> list[Path]:
    """Get all markdown spec files organized by fork."""
    specs_dir = consensus_dir / "specs"
    if not specs_dir.exists():
        raise FileNotFoundError(f"Specs directory not found: {specs_dir}")

    return list(specs_dir.rglob("*.md"))


def get_eip_files(eips_dir: Path) -> list[Path]:
    """Get all EIP markdown files."""
    eips_content_dir = eips_dir / "EIPS"
    if not eips_content_dir.exists():
        raise FileNotFoundError(f"EIPs directory not found: {eips_content_dir}")

    return list(eips_content_dir.glob("eip-*.md"))


def get_builder_spec_files(builder_specs_dir: Path) -> list[Path]:
    """Get all builder-specs markdown files."""
    specs_dir = builder_specs_dir / "specs"
    if not specs_dir.exists():
        raise FileNotFoundError(f"Builder specs directory not found: {specs_dir}")

    return list(specs_dir.rglob("*.md"))


def get_lean_spec_files(lean_spec_dir: Path) -> list[Path]:
    """
    Get all Python files from leanSpec src/lean_spec/.

    Args:
        lean_spec_dir: Path to the leanSpec repository root

    Returns:
        List of Python file paths

    Raises:
        FileNotFoundError: If the leanSpec src directory doesn't exist
    """
    src_dir = lean_spec_dir / "src" / "lean_spec"
    if not src_dir.exists():
        raise FileNotFoundError(f"leanSpec source directory not found: {src_dir}")

    # Get all Python files, excluding __pycache__ and test files
    files = []
    for f in src_dir.rglob("*.py"):
        if "__pycache__" not in str(f) and not f.name.startswith("test_"):
            files.append(f)

    return files


def get_lean_rust_files(repo_dir: Path) -> list[Path]:
    """
    Get Rust source files from a leanEthereum Rust repository.

    Args:
        repo_dir: Path to the repository root (e.g., leanSig, leanMultisig)

    Returns:
        List of Rust file paths

    Raises:
        FileNotFoundError: If the repository directory doesn't exist
    """
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repository directory not found: {repo_dir}")

    # Get all Rust files, excluding target directory and test files
    files = []
    for f in repo_dir.rglob("*.rs"):
        path_str = str(f)
        if "/target/" in path_str:
            continue
        if "__pycache__" in path_str:
            continue
        files.append(f)

    return files


def get_lean_pm_files(pm_dir: Path) -> list[Path]:
    """
    Get markdown files from leanEthereum/pm (meeting notes).

    Args:
        pm_dir: Path to the pm repository root

    Returns:
        List of markdown file paths

    Raises:
        FileNotFoundError: If the pm directory doesn't exist
    """
    if not pm_dir.exists():
        raise FileNotFoundError(f"PM directory not found: {pm_dir}")

    # Get all markdown files
    files = []
    for f in pm_dir.rglob("*.md"):
        files.append(f)

    return files


def get_lean_snappy_files(snappy_dir: Path) -> list[Path]:
    """
    Get Python files from leanEthereum/leanSnappy.

    Args:
        snappy_dir: Path to the leanSnappy repository root

    Returns:
        List of Python file paths

    Raises:
        FileNotFoundError: If the leanSnappy directory doesn't exist
    """
    if not snappy_dir.exists():
        raise FileNotFoundError(f"leanSnappy directory not found: {snappy_dir}")

    # Get all Python files, excluding __pycache__ and test files
    files = []
    for f in snappy_dir.rglob("*.py"):
        if "__pycache__" not in str(f) and not f.name.startswith("test_"):
            files.append(f)

    return files


# leanEthereum post-quantum Rust repositories
LEAN_PQ_REPOS = {
    "leanSig": {
        "url": "https://github.com/leanEthereum/leanSig.git",
        "branch": "main",
        "language": "rust",
        "description": "Post-quantum signature implementation",
    },
    "leanMultisig": {
        "url": "https://github.com/leanEthereum/leanMultisig.git",
        "branch": "main",
        "language": "rust",
        "description": "XMSS aggregation zkVM for post-quantum multisig",
    },
    "multilinear-toolkit": {
        "url": "https://github.com/leanEthereum/multilinear-toolkit.git",
        "branch": "main",
        "language": "rust",
        "description": "Crypto support library for multilinear operations",
    },
    "fiat-shamir": {
        "url": "https://github.com/leanEthereum/fiat-shamir.git",
        "branch": "main",
        "language": "rust",
        "description": "Fiat-Shamir crypto support library",
    },
    "pm": {
        "url": "https://github.com/leanEthereum/pm.git",
        "branch": "main",
        "language": "markdown",
        "description": "Meeting notes and agendas",
    },
    "leanSnappy": {
        "url": "https://github.com/leanEthereum/leanSnappy.git",
        "branch": "main",
        "language": "python",
        "description": "Snappy compression for Ethereum",
    },
}
