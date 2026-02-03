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
