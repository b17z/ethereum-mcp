"""Tests for the chunker module."""

from pathlib import Path

import pytest

from ethereum_mcp.indexer.chunker import (
    Chunk,
    _detect_builder_spec_chunk_type,
    _detect_chunk_type,
    _extract_eip_frontmatter,
    _extract_fork_from_path,
    _extract_function_chunks,
    chunk_documents,
    chunk_single_file,
    generate_chunk_id,
)


class TestExtractForkFromPath:
    """Tests for _extract_fork_from_path."""

    def test_extracts_phase0(self):
        path = Path("/specs/phase0/beacon-chain.md")
        assert _extract_fork_from_path(path) == "phase0"

    def test_extracts_electra(self):
        path = Path("/consensus-specs/specs/electra/beacon-chain.md")
        assert _extract_fork_from_path(path) == "electra"

    def test_extracts_deneb_case_insensitive(self):
        path = Path("/specs/Deneb/beacon-chain.md")
        assert _extract_fork_from_path(path) == "deneb"

    def test_returns_none_for_unknown(self):
        path = Path("/docs/readme.md")
        assert _extract_fork_from_path(path) is None

    def test_handles_nested_paths(self):
        path = Path("/home/user/ethereum/consensus-specs/specs/capella/beacon-chain.md")
        assert _extract_fork_from_path(path) == "capella"


class TestDetectChunkType:
    """Tests for _detect_chunk_type."""

    def test_detects_function(self):
        content = """def process_attestation(state, attestation):
    # Process attestation
    pass"""
        assert _detect_chunk_type(content) == "function"

    def test_detects_constant_table(self):
        content = """| Name | Value |
| `MAX_EFFECTIVE_BALANCE` | `32 * 10**9` |
| `EFFECTIVE_BALANCE_INCREMENT` | `10**9` |"""
        assert _detect_chunk_type(content) == "constant"

    def test_detects_class(self):
        content = """class BeaconState(Container):
    genesis_time: uint64
    slot: Slot"""
        assert _detect_chunk_type(content) == "type"

    def test_defaults_to_spec(self):
        content = """This is some general spec documentation about
how the beacon chain works."""
        assert _detect_chunk_type(content) == "spec"


class TestDetectBuilderSpecChunkType:
    """Tests for _detect_builder_spec_chunk_type."""

    def test_detects_api_endpoint(self):
        content = """POST `/eth/v1/builder/blinded_blocks`
Submit a signed blinded block."""
        assert _detect_builder_spec_chunk_type(content) == "builder_api"

    def test_detects_get_endpoint(self):
        content = """GET `/eth/v1/builder/header/{slot}/{parent_hash}/{pubkey}`"""
        assert _detect_builder_spec_chunk_type(content) == "builder_api"

    def test_detects_ssz_container(self):
        content = """class ExecutionPayloadHeader(Container):
    parent_hash: Hash32"""
        assert _detect_builder_spec_chunk_type(content) == "builder_type"

    def test_detects_python_class(self):
        content = """```python
class SignedBuilderBid(Container):
    message: BuilderBid
    signature: BLSSignature
```"""
        assert _detect_builder_spec_chunk_type(content) == "builder_type"

    def test_defaults_to_builder_spec(self):
        content = """The builder API allows validators to outsource
block building to specialized builders."""
        assert _detect_builder_spec_chunk_type(content) == "builder_spec"


class TestExtractEipFrontmatter:
    """Tests for _extract_eip_frontmatter."""

    def test_extracts_frontmatter(self):
        content = """---
eip: 4844
title: Shard Blob Transactions
status: Final
category: Core
---

# Abstract
This EIP introduces blob transactions."""

        result = _extract_eip_frontmatter(content)

        assert result["eip"] == "4844"
        assert result["title"] == "Shard Blob Transactions"
        assert result["status"] == "Final"
        assert result["category"] == "Core"

    def test_handles_missing_frontmatter(self):
        content = """# EIP-1234
Just some content without frontmatter."""

        result = _extract_eip_frontmatter(content)
        assert result == {}

    def test_handles_partial_frontmatter(self):
        content = """---
eip: 1559
title: Fee market change
---
Content here."""

        result = _extract_eip_frontmatter(content)
        assert result["eip"] == "1559"
        assert result["title"] == "Fee market change"
        assert "status" not in result


class TestExtractFunctionChunks:
    """Tests for _extract_function_chunks."""

    def test_extracts_function_from_code_block(self):
        content = """# Process Attestation

```python
def process_attestation(state: BeaconState, attestation: Attestation) -> None:
    data = attestation.data
    assert data.target.epoch in (get_previous_epoch(state), get_current_epoch(state))
```

Some more text here.
"""
        path = Path("/specs/phase0/beacon-chain.md")
        chunks = _extract_function_chunks(content, path, "phase0")

        assert len(chunks) == 1
        assert chunks[0].chunk_type == "function"
        assert chunks[0].section == "process_attestation"
        assert chunks[0].fork == "phase0"
        assert "def process_attestation" in chunks[0].content

    def test_extracts_multiple_functions(self):
        content = """
```python
def func_one():
    pass
```

Some text.

```python
def func_two(x, y):
    return x + y
```
"""
        path = Path("/specs/electra/beacon-chain.md")
        chunks = _extract_function_chunks(content, path, "electra")

        assert len(chunks) == 2
        names = [c.metadata["function_name"] for c in chunks]
        assert "func_one" in names
        assert "func_two" in names

    def test_handles_no_functions(self):
        content = """# Overview

This is just documentation with no code blocks."""
        path = Path("/docs/readme.md")
        chunks = _extract_function_chunks(content, path, None)

        assert len(chunks) == 0


class TestChunkDataclass:
    """Tests for the Chunk dataclass."""

    def test_chunk_is_frozen(self):
        chunk = Chunk(
            content="test",
            source="/path/to/file.md",
            fork="electra",
            section="test",
            chunk_type="spec",
            metadata={},
        )

        with pytest.raises(AttributeError):
            chunk.content = "modified"

    def test_chunk_equality(self):
        chunk1 = Chunk(
            content="test",
            source="/path",
            fork="electra",
            section="test",
            chunk_type="spec",
            metadata={"key": "value"},
        )
        chunk2 = Chunk(
            content="test",
            source="/path",
            fork="electra",
            section="test",
            chunk_type="spec",
            metadata={"key": "value"},
        )

        assert chunk1 == chunk2

    def test_chunk_with_id(self):
        """Chunk can have an optional chunk_id."""
        chunk = Chunk(
            content="test",
            source="/path",
            fork="electra",
            section="test",
            chunk_type="spec",
            metadata={},
            chunk_id="eth_spec_abc123_0001_def456",
        )

        assert chunk.chunk_id == "eth_spec_abc123_0001_def456"

    def test_chunk_default_empty_id(self):
        """Chunk without explicit ID should have empty string."""
        chunk = Chunk(
            content="test",
            source="/path",
            fork="electra",
            section="test",
            chunk_type="spec",
            metadata={},
        )

        assert chunk.chunk_id == ""


class TestGenerateChunkId:
    """Tests for generate_chunk_id function."""

    def test_generates_valid_format(self):
        """Generated ID should follow expected format."""
        chunk_id = generate_chunk_id(
            project="eth",
            source_type="spec",
            source_file="specs/electra/beacon-chain.md",
            chunk_index=5,
            content="test content",
        )

        parts = chunk_id.split("_")
        assert len(parts) == 5
        assert parts[0] == "eth"
        assert parts[1] == "spec"
        assert len(parts[2]) == 8  # path hash
        assert parts[3] == "0005"  # zero-padded index
        assert len(parts[4]) == 8  # content hash

    def test_deterministic_generation(self):
        """Same inputs should produce same ID."""
        id1 = generate_chunk_id("eth", "spec", "test.md", 0, "content")
        id2 = generate_chunk_id("eth", "spec", "test.md", 0, "content")

        assert id1 == id2

    def test_different_content_different_id(self):
        """Different content should produce different IDs."""
        id1 = generate_chunk_id("eth", "spec", "test.md", 0, "content A")
        id2 = generate_chunk_id("eth", "spec", "test.md", 0, "content B")

        assert id1 != id2

    def test_different_index_different_id(self):
        """Different indices should produce different IDs."""
        id1 = generate_chunk_id("eth", "spec", "test.md", 0, "content")
        id2 = generate_chunk_id("eth", "spec", "test.md", 1, "content")

        assert id1 != id2


class TestChunkDocumentsWithIds:
    """Tests for chunk_documents with ID generation."""

    def test_generates_ids_when_requested(self, tmp_path):
        """chunk_documents should generate IDs when generate_ids=True."""
        # Create a sample spec file
        spec_dir = tmp_path / "specs" / "electra"
        spec_dir.mkdir(parents=True)
        spec_file = spec_dir / "beacon-chain.md"
        spec_file.write_text("""# Test Spec

## Section One

Some content here.

## Section Two

More content here.
""")

        chunks = chunk_documents(
            spec_files=[spec_file],
            eip_files=[],
            generate_ids=True,
            base_path=tmp_path,
        )

        assert len(chunks) > 0
        assert all(c.chunk_id for c in chunks)  # All have IDs
        assert all(c.chunk_id.startswith("eth_") for c in chunks)

    def test_no_ids_by_default(self, tmp_path):
        """chunk_documents should not generate IDs by default."""
        spec_dir = tmp_path / "specs" / "electra"
        spec_dir.mkdir(parents=True)
        spec_file = spec_dir / "beacon-chain.md"
        spec_file.write_text("# Test\n\nContent here.")

        chunks = chunk_documents(
            spec_files=[spec_file],
            eip_files=[],
            generate_ids=False,
        )

        assert len(chunks) > 0
        assert all(c.chunk_id == "" for c in chunks)


class TestChunkSingleFile:
    """Tests for chunk_single_file function."""

    def test_chunks_spec_file(self, tmp_path):
        """Should chunk a spec file with IDs."""
        spec_dir = tmp_path / "specs" / "electra"
        spec_dir.mkdir(parents=True)
        spec_file = spec_dir / "beacon-chain.md"
        spec_file.write_text("""# Beacon Chain

## Configuration

| Name | Value |
| `MAX_EFFECTIVE_BALANCE` | `32 * 10**9` |

## Functions

```python
def get_current_epoch(state):
    return state.slot // 32
```
""")

        chunks = chunk_single_file(spec_file, "spec", base_path=tmp_path)

        assert len(chunks) > 0
        assert all(c.chunk_id for c in chunks)

    def test_chunks_eip_file(self, tmp_path):
        """Should chunk an EIP file with IDs."""
        eip_file = tmp_path / "eip-1234.md"
        eip_file.write_text("""---
eip: 1234
title: Test EIP
status: Draft
---

# Abstract

This is a test EIP.
""")

        chunks = chunk_single_file(eip_file, "eip", base_path=tmp_path)

        assert len(chunks) > 0
        assert all(c.chunk_type == "eip" for c in chunks)
        assert all(c.chunk_id for c in chunks)
