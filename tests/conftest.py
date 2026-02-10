"""Pytest configuration and shared fixtures."""

import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir(tmp_path):
    """Isolated data directory for each test."""
    data_dir = tmp_path / ".ethereum-mcp"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_spec_content():
    """Sample spec markdown content for testing."""
    return """# Beacon Chain

## Configuration

| Name | Value |
| - | - |
| `MAX_EFFECTIVE_BALANCE` | `32 * 10**9` |
| `EFFECTIVE_BALANCE_INCREMENT` | `10**9` |

## Helper Functions

### `get_current_epoch`

```python
def get_current_epoch(state: BeaconState) -> Epoch:
    return compute_epoch_at_slot(state.slot)
```

### `process_attestation`

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    data = attestation.data
    assert data.target.epoch in (get_previous_epoch(state), get_current_epoch(state))
```

## State Transition

The beacon chain state transition function processes blocks and epochs.
"""


@pytest.fixture
def sample_eip_content():
    """Sample EIP markdown content for testing."""
    return """---
eip: 4844
title: Shard Blob Transactions
status: Final
category: Core
---

# Abstract

This EIP introduces a new transaction format for blob-carrying transactions.

## Motivation

Rollups need cheaper data availability.

## Specification

### Parameters

| Constant | Value |
| - | - |
| `BLOB_TX_TYPE` | `0x03` |
| `TARGET_BLOB_GAS_PER_BLOCK` | `393216` |
"""


@pytest.fixture
def sample_manifest():
    """Valid manifest for testing."""
    return {
        "version": "1.0.0",
        "updated_at": "2026-02-03T14:30:00Z",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_config": {"chunk_size": 1000, "chunk_overlap": 200},
        "files": {
            "consensus-specs/specs/electra/beacon-chain.md": {
                "sha256": "abc123def456789012345678901234567890123456789012345678901234abcd",
                "mtime_ns": 1707000000000000000,
                "chunk_ids": ["eth_spec_a1b2c3d4_0001_ef567890", "eth_spec_a1b2c3d4_0002_12345678"],
            }
        },
        "repo_versions": {
            "consensus-specs": "abc123def456",
            "EIPs": "789xyz000111",
        },
    }


@pytest.fixture
def sample_manifest_file(temp_data_dir, sample_manifest):
    """Create a manifest file in the temp directory."""
    manifest_path = temp_data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sample_manifest, f, indent=2)
    return manifest_path


@pytest.fixture
def sample_spec_file(temp_data_dir, sample_spec_content):
    """Create a sample spec file in the temp directory."""
    spec_dir = temp_data_dir / "consensus-specs" / "specs" / "electra"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / "beacon-chain.md"
    spec_file.write_text(sample_spec_content)
    return spec_file


@pytest.fixture
def sample_eip_file(temp_data_dir, sample_eip_content):
    """Create a sample EIP file in the temp directory."""
    eip_dir = temp_data_dir / "EIPs" / "EIPS"
    eip_dir.mkdir(parents=True)
    eip_file = eip_dir / "eip-4844.md"
    eip_file.write_text(sample_eip_content)
    return eip_file


@pytest.fixture
def sample_config_yaml():
    """Sample config YAML content."""
    return """embedding:
  model: "all-MiniLM-L6-v2"
  batch_size: 32

chunking:
  chunk_size: 1000
  chunk_overlap: 200
"""


@pytest.fixture
def sample_config_file(temp_data_dir, sample_config_yaml):
    """Create a config file in the temp directory."""
    config_path = temp_data_dir / "config.yaml"
    config_path.write_text(sample_config_yaml)
    return config_path


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing (avoids loading real model)."""
    import numpy as np

    def generate_mock_embedding(text: str) -> list[float]:
        """Generate deterministic mock embedding based on text hash."""
        # Use hash to generate deterministic but varied embeddings
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        return np.random.randn(384).tolist()

    return generate_mock_embedding


@pytest.fixture
def sample_lean_spec_content():
    """Sample Python spec content for testing."""
    return '''"""State Container for Lean Ethereum.

This module defines the core state containers for the beacon chain.
"""

from dataclasses import dataclass
from typing import List

SLOTS_PER_EPOCH: int = 32
MAX_VALIDATORS: int = 2**20


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint in the chain.

    Used for finalization.
    """
    epoch: int
    root: bytes


class State:
    """The main consensus state."""
    slot: int
    validators: List["Validator"]

    def get_current_epoch(self) -> int:
        """Return current epoch from slot."""
        return self.slot // SLOTS_PER_EPOCH

    def is_finalized(self, checkpoint: Checkpoint) -> bool:
        """Check if checkpoint is finalized."""
        return checkpoint.epoch <= self.finalized_epoch
'''


@pytest.fixture
def sample_lean_spec_file(temp_data_dir, sample_lean_spec_content):
    """Create a sample leanSpec file in the temp directory."""
    spec_dir = temp_data_dir / "leanSpec" / "src" / "lean_spec" / "subspecs" / "containers"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / "state.py"
    spec_file.write_text(sample_lean_spec_content)
    return spec_file


@pytest.fixture
def sample_lean_spec_dir(temp_data_dir, sample_lean_spec_content):
    """Create a sample leanSpec directory structure."""
    lean_spec_dir = temp_data_dir / "leanSpec"
    src_dir = lean_spec_dir / "src" / "lean_spec"

    # Create multiple modules
    (src_dir / "subspecs" / "containers").mkdir(parents=True)
    (src_dir / "subspecs" / "config").mkdir(parents=True)
    (src_dir / "helpers").mkdir(parents=True)

    # State file
    (src_dir / "subspecs" / "containers" / "state.py").write_text(sample_lean_spec_content)

    # Config file
    (src_dir / "subspecs" / "config" / "constants.py").write_text('''"""Configuration constants."""

GENESIS_SLOT: int = 0
GENESIS_EPOCH: int = 0
FAR_FUTURE_EPOCH: int = 2**64 - 1
''')

    # Helper file
    (src_dir / "helpers" / "math.py").write_text('''"""Math helpers."""

def integer_squareroot(n: int) -> int:
    """Compute integer square root."""
    if n < 0:
        raise ValueError("negative input")
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x
''')

    # __init__.py files
    (src_dir / "__init__.py").write_text('')
    (src_dir / "subspecs" / "__init__.py").write_text('')
    (src_dir / "subspecs" / "containers" / "__init__.py").write_text('')
    (src_dir / "subspecs" / "config" / "__init__.py").write_text('')
    (src_dir / "helpers" / "__init__.py").write_text('')

    return lean_spec_dir


@pytest.fixture
def sample_rust_content():
    """Sample Rust code for testing post-quantum signature implementations."""
    return '''//! A signature scheme for post-quantum security.
//!
//! This module implements XMSS-based signatures.

use std::vec::Vec;

/// Maximum size of a signature in bytes.
pub const MAX_SIGNATURE_SIZE: usize = 2048;

/// Tree height for XMSS.
pub const TREE_HEIGHT: usize = 10;

/// A post-quantum signature scheme.
#[derive(Debug, Clone)]
pub struct LeanSig {
    /// The signing key.
    pub key: [u8; 32],
    /// Tree state.
    tree_state: TreeState,
}

/// Internal tree state.
struct TreeState {
    height: usize,
    index: u64,
}

/// Signature verification result.
#[derive(Debug, PartialEq)]
pub enum VerifyResult {
    /// Signature is valid.
    Valid,
    /// Signature is invalid.
    Invalid,
    /// Error during verification.
    Error(String),
}

impl LeanSig {
    /// Create a new signature scheme instance.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for key generation
    ///
    /// # Returns
    ///
    /// A new LeanSig instance.
    pub fn new(seed: &[u8; 32]) -> Self {
        Self {
            key: *seed,
            tree_state: TreeState {
                height: TREE_HEIGHT,
                index: 0,
            },
        }
    }

    /// Sign a message.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to sign
    ///
    /// # Returns
    ///
    /// The signature bytes.
    pub fn sign(&mut self, message: &[u8]) -> Vec<u8> {
        // XMSS signing logic
        let mut sig = Vec::with_capacity(MAX_SIGNATURE_SIZE);
        sig.extend_from_slice(&self.key);
        sig.extend_from_slice(message);
        self.tree_state.index += 1;
        sig
    }

    /// Verify a signature.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> VerifyResult {
        if signature.len() < 32 {
            return VerifyResult::Invalid;
        }
        // Simplified verification
        if &signature[..32] == &self.key {
            VerifyResult::Valid
        } else {
            VerifyResult::Invalid
        }
    }
}

/// Helper function for hashing.
pub fn hash_message(message: &[u8]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for (i, byte) in message.iter().take(32).enumerate() {
        result[i] = *byte;
    }
    result
}
'''


@pytest.fixture
def sample_rust_file(temp_data_dir, sample_rust_content):
    """Create a sample Rust file in the temp directory."""
    lean_sig_dir = temp_data_dir / "leanSig" / "src"
    lean_sig_dir.mkdir(parents=True)
    rust_file = lean_sig_dir / "lib.rs"
    rust_file.write_text(sample_rust_content)
    return rust_file


@pytest.fixture
def sample_lean_pq_dir(temp_data_dir, sample_rust_content):
    """Create a sample leanEthereum PQ directory structure."""
    # leanSig
    lean_sig_dir = temp_data_dir / "leanSig" / "src"
    lean_sig_dir.mkdir(parents=True)
    (lean_sig_dir / "lib.rs").write_text(sample_rust_content)
    (lean_sig_dir / "xmss.rs").write_text('''//! XMSS implementation.

/// XMSS tree node.
#[derive(Clone)]
pub struct XmssNode {
    pub hash: [u8; 32],
    pub height: usize,
}

impl XmssNode {
    /// Create a leaf node.
    pub fn leaf(data: &[u8]) -> Self {
        Self {
            hash: [0u8; 32],
            height: 0,
        }
    }
}
''')

    # leanMultisig
    lean_multisig_dir = temp_data_dir / "leanMultisig" / "src"
    lean_multisig_dir.mkdir(parents=True)
    (lean_multisig_dir / "lib.rs").write_text('''//! XMSS aggregation for multisig.

/// Aggregated signature.
#[derive(Debug)]
pub struct AggregatedSig {
    pub signatures: Vec<Vec<u8>>,
    pub threshold: usize,
}

impl AggregatedSig {
    /// Create a new aggregated signature.
    pub fn new(threshold: usize) -> Self {
        Self {
            signatures: Vec::new(),
            threshold,
        }
    }

    /// Add a signature to the aggregate.
    pub fn add_signature(&mut self, sig: Vec<u8>) {
        self.signatures.push(sig);
    }
}
''')

    # pm (meeting notes)
    pm_dir = temp_data_dir / "pm"
    pm_dir.mkdir(parents=True)
    (pm_dir / "2024-01-15.md").write_text('''# leanEthereum Meeting 2024-01-15

## Attendees
- Alice
- Bob

## Agenda
1. XMSS implementation progress
2. Multisig aggregation design

## Notes
Discussed the tree height parameter for XMSS signatures.
''')

    return temp_data_dir
