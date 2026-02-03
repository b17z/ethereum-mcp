"""Tests for the manifest module."""

import json
from pathlib import Path

import pytest

from ethereum_mcp.indexer.manifest import (
    FileEntry,
    Manifest,
    ManifestCorruptedError,
    ManifestValidationError,
    _sanitize_chunk_id,
    _validate_path,
    compute_changes,
    compute_file_hash,
    generate_chunk_id,
    get_file_mtime_ns,
    load_manifest,
    needs_full_rebuild,
    save_manifest,
)


class TestManifestValidation:
    """Tests for manifest validation."""

    def test_rejects_invalid_version(self):
        """Manifest with invalid version format should be rejected."""
        data = {
            "version": "invalid",
            "files": {},
        }
        with pytest.raises(ManifestValidationError, match="Invalid version format"):
            Manifest.from_dict(data)

    def test_rejects_missing_files_key(self):
        """Manifest without files key should be rejected."""
        data = {"version": "1.0.0"}
        with pytest.raises(ManifestValidationError, match="Missing required field: files"):
            Manifest.from_dict(data)

    def test_rejects_files_not_dict(self):
        """Manifest with non-dict files should be rejected."""
        data = {
            "version": "1.0.0",
            "files": ["not", "a", "dict"],
        }
        with pytest.raises(ManifestValidationError, match="'files' must be a dictionary"):
            Manifest.from_dict(data)

    def test_handles_corrupted_json(self, temp_data_dir):
        """Corrupted JSON should raise ManifestCorruptedError and create backup."""
        manifest_path = temp_data_dir / "manifest.json"
        manifest_path.write_text("{ invalid json }")

        with pytest.raises(ManifestCorruptedError, match="corrupted"):
            load_manifest(manifest_path)

        # Check backup was created
        backup_path = manifest_path.with_suffix(".json.corrupted")
        assert backup_path.exists()

    def test_creates_backup_on_corruption(self, temp_data_dir):
        """Corrupted manifest should be backed up."""
        manifest_path = temp_data_dir / "manifest.json"
        original_content = "{ broken: json }"
        manifest_path.write_text(original_content)

        try:
            load_manifest(manifest_path)
        except ManifestCorruptedError:
            pass

        backup_path = manifest_path.with_suffix(".json.corrupted")
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

    def test_validates_sha256_format(self):
        """SHA256 hashes must be valid hex strings."""
        data = {
            "version": "1.0.0",
            "files": {
                "test.md": {
                    "sha256": "not-a-valid-hash",
                    "mtime_ns": 1234567890,
                }
            },
        }
        with pytest.raises(ManifestValidationError, match="Invalid SHA256"):
            Manifest.from_dict(data)

    def test_accepts_valid_sha256(self):
        """Valid SHA256 should be accepted."""
        valid_hash = "a" * 64  # 64 hex chars
        data = {
            "version": "1.0.0",
            "files": {
                "test.md": {
                    "sha256": valid_hash,
                    "mtime_ns": 1234567890,
                    "chunk_ids": [],
                }
            },
        }
        manifest = Manifest.from_dict(data)
        assert manifest.files["test.md"].sha256 == valid_hash

    def test_rejects_missing_version(self):
        """Manifest without version should be rejected."""
        data = {"files": {}}
        with pytest.raises(ManifestValidationError, match="Missing required field: version"):
            Manifest.from_dict(data)


class TestManifestSerialization:
    """Tests for manifest serialization."""

    def test_roundtrip_serialization(self, sample_manifest):
        """Manifest should survive roundtrip serialization."""
        manifest = Manifest.from_dict(sample_manifest)
        serialized = manifest.to_dict()
        restored = Manifest.from_dict(serialized)

        assert restored.version == manifest.version
        assert restored.embedding_model == manifest.embedding_model
        assert len(restored.files) == len(manifest.files)

    def test_handles_unicode_paths(self, temp_data_dir):
        """Manifest should handle unicode in file paths."""
        manifest = Manifest(
            version="1.0.0",
            embedding_model="test",
            files={
                "specs/日本語/test.md": FileEntry(
                    sha256="a" * 64,
                    mtime_ns=1234567890,
                    chunk_ids=["test_0001"],
                )
            },
        )

        manifest_path = temp_data_dir / "manifest.json"
        save_manifest(manifest, manifest_path)

        loaded = load_manifest(manifest_path)
        assert "specs/日本語/test.md" in loaded.files

    def test_preserves_mtime_precision(self, temp_data_dir):
        """Nanosecond precision should be preserved."""
        precise_mtime = 1707000000123456789  # Has nanosecond precision
        manifest = Manifest(
            version="1.0.0",
            files={
                "test.md": FileEntry(
                    sha256="a" * 64,
                    mtime_ns=precise_mtime,
                )
            },
        )

        manifest_path = temp_data_dir / "manifest.json"
        save_manifest(manifest, manifest_path)

        loaded = load_manifest(manifest_path)
        assert loaded.files["test.md"].mtime_ns == precise_mtime


class TestPathValidation:
    """Tests for path validation security."""

    def test_rejects_path_traversal(self):
        """Path traversal attempts should be rejected."""
        with pytest.raises(ManifestValidationError, match="Path traversal"):
            _validate_path("../../../etc/passwd")

    def test_rejects_absolute_paths(self):
        """Absolute paths should be rejected."""
        with pytest.raises(ManifestValidationError, match="Absolute paths not allowed"):
            _validate_path("/etc/passwd")

    def test_rejects_windows_absolute_paths(self):
        """Windows absolute paths should be rejected."""
        with pytest.raises(ManifestValidationError, match="Absolute paths not allowed"):
            _validate_path("C:\\Windows\\System32")

    def test_rejects_null_bytes(self):
        """Null bytes in paths should be rejected."""
        with pytest.raises(ManifestValidationError, match="Null bytes"):
            _validate_path("test\x00file.md")

    def test_rejects_newlines(self):
        """Newlines in paths should be rejected."""
        with pytest.raises(ManifestValidationError, match="Invalid characters"):
            _validate_path("test\nfile.md")

    def test_accepts_valid_paths(self):
        """Valid relative paths should be accepted."""
        _validate_path("specs/electra/beacon-chain.md")
        _validate_path("EIPs/EIPS/eip-4844.md")
        _validate_path("test.md")


class TestChunkIdGeneration:
    """Tests for chunk ID generation."""

    def test_generates_valid_id(self):
        """Generated IDs should match expected format."""
        chunk_id = generate_chunk_id(
            project="eth",
            source_type="spec",
            source_file="specs/electra/beacon-chain.md",
            chunk_index=1,
            content="test content",
        )

        assert chunk_id.startswith("eth_spec_")
        assert "_0001_" in chunk_id
        # Should only contain alphanumeric and underscore
        assert all(c.isalnum() or c == "_" for c in chunk_id)

    def test_different_content_different_id(self):
        """Different content should produce different IDs."""
        id1 = generate_chunk_id("eth", "spec", "test.md", 0, "content A")
        id2 = generate_chunk_id("eth", "spec", "test.md", 0, "content B")

        # Same position but different content = different ID
        assert id1 != id2

    def test_same_content_same_id(self):
        """Same content should produce same ID (deterministic)."""
        id1 = generate_chunk_id("eth", "spec", "test.md", 0, "same content")
        id2 = generate_chunk_id("eth", "spec", "test.md", 0, "same content")

        assert id1 == id2

    def test_sanitize_chunk_id_valid(self):
        """Valid chunk IDs should pass sanitization."""
        result = _sanitize_chunk_id("eth_spec_abc12345_0001_def67890")
        assert result == "eth_spec_abc12345_0001_def67890"

    def test_sanitize_chunk_id_invalid(self):
        """Invalid chunk IDs should be rejected."""
        with pytest.raises(ManifestValidationError):
            _sanitize_chunk_id("chunk; DROP TABLE specs;--")


class TestChangeDetection:
    """Tests for change detection logic."""

    def test_detects_new_file(self, temp_data_dir, sample_spec_file):
        """New files should be detected as 'add'."""
        manifest = Manifest(version="1.0.0", files={})
        current_files = {
            "consensus-specs/specs/electra/beacon-chain.md": sample_spec_file
        }

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 1
        assert changes[0].change_type == "add"
        assert "beacon-chain.md" in changes[0].path

    def test_detects_deleted_file(self, temp_data_dir):
        """Deleted files should be detected."""
        manifest = Manifest(
            version="1.0.0",
            files={
                "deleted.md": FileEntry(
                    sha256="a" * 64,
                    mtime_ns=1234567890,
                    chunk_ids=["chunk_0001"],
                )
            },
        )
        current_files = {}  # File no longer exists

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 1
        assert changes[0].change_type == "delete"
        assert changes[0].path == "deleted.md"
        assert changes[0].old_chunk_ids == ["chunk_0001"]

    def test_skips_unchanged_file_fast_path(self, temp_data_dir, sample_spec_file):
        """Unchanged files (same mtime) should be skipped."""
        # Get actual mtime
        actual_mtime = get_file_mtime_ns(sample_spec_file)
        actual_hash = compute_file_hash(sample_spec_file)

        manifest = Manifest(
            version="1.0.0",
            files={
                "test.md": FileEntry(
                    sha256=actual_hash,
                    mtime_ns=actual_mtime,  # Same mtime
                    chunk_ids=["chunk_0001"],
                )
            },
        )
        current_files = {"test.md": sample_spec_file}

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 0  # No changes detected

    def test_detects_modified_file_by_hash(self, temp_data_dir, sample_spec_file):
        """Modified files (different hash) should be detected."""
        manifest = Manifest(
            version="1.0.0",
            files={
                "test.md": FileEntry(
                    sha256="b" * 64,  # Different hash
                    mtime_ns=0,  # Different mtime to trigger hash check
                    chunk_ids=["old_chunk_0001"],
                )
            },
        )
        current_files = {"test.md": sample_spec_file}

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 1
        assert changes[0].change_type == "modify"
        assert changes[0].old_chunk_ids == ["old_chunk_0001"]


class TestNeedsFullRebuild:
    """Tests for full rebuild detection."""

    def test_no_manifest_needs_rebuild(self):
        """Missing manifest should require full rebuild."""
        needs, reason = needs_full_rebuild(None, {"embedding_model": "test"})
        assert needs is True
        assert "No manifest" in reason

    def test_model_change_needs_rebuild(self, sample_manifest):
        """Changed embedding model should require full rebuild."""
        manifest = Manifest.from_dict(sample_manifest)
        config = {"embedding_model": "different-model", "chunk_config": {}}

        needs, reason = needs_full_rebuild(manifest, config)

        assert needs is True
        assert "Embedding model changed" in reason

    def test_chunk_config_change_needs_rebuild(self, sample_manifest):
        """Changed chunk config should require full rebuild."""
        manifest = Manifest.from_dict(sample_manifest)
        config = {
            "embedding_model": manifest.embedding_model,
            "chunk_config": {"chunk_size": 2000, "chunk_overlap": 100},  # Different
        }

        needs, reason = needs_full_rebuild(manifest, config)

        assert needs is True
        assert "Chunk config changed" in reason

    def test_same_config_no_rebuild(self, sample_manifest):
        """Same config should not require rebuild."""
        manifest = Manifest.from_dict(sample_manifest)
        config = {
            "embedding_model": manifest.embedding_model,
            "chunk_config": manifest.chunk_config,
        }

        needs, reason = needs_full_rebuild(manifest, config)

        assert needs is False
        assert reason == ""


class TestAtomicWrites:
    """Tests for atomic file operations."""

    def test_save_creates_parent_directories(self, tmp_path):
        """Save should create parent directories if needed."""
        manifest = Manifest(version="1.0.0", files={})
        manifest_path = tmp_path / "subdir" / "nested" / "manifest.json"

        save_manifest(manifest, manifest_path)

        assert manifest_path.exists()
        loaded = load_manifest(manifest_path)
        assert loaded.version == "1.0.0"

    def test_save_atomic_on_failure(self, temp_data_dir, monkeypatch):
        """Failed saves should not corrupt existing manifest."""
        # Create initial manifest
        manifest_path = temp_data_dir / "manifest.json"
        original = Manifest(version="1.0.0", files={})
        save_manifest(original, manifest_path)

        # Make save fail
        def raise_error(*args, **kwargs):
            raise IOError("Simulated write failure")

        # Try to save new manifest (will fail)
        new_manifest = Manifest(version="2.0.0", files={})
        monkeypatch.setattr("builtins.open", raise_error)

        with pytest.raises(IOError):
            save_manifest(new_manifest, manifest_path)

        # Original should still be intact
        monkeypatch.undo()
        loaded = load_manifest(manifest_path)
        assert loaded.version == "1.0.0"


class TestFileHashing:
    """Tests for file hashing utilities."""

    def test_compute_file_hash(self, sample_spec_file):
        """File hash should be valid SHA256."""
        hash_val = compute_file_hash(sample_spec_file)

        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_hash_deterministic(self, sample_spec_file):
        """Same file should produce same hash."""
        hash1 = compute_file_hash(sample_spec_file)
        hash2 = compute_file_hash(sample_spec_file)

        assert hash1 == hash2

    def test_different_content_different_hash(self, temp_data_dir):
        """Different content should produce different hashes."""
        file1 = temp_data_dir / "file1.md"
        file2 = temp_data_dir / "file2.md"
        file1.write_text("content A")
        file2.write_text("content B")

        assert compute_file_hash(file1) != compute_file_hash(file2)

    def test_get_file_mtime_ns(self, sample_spec_file):
        """Should return nanosecond precision mtime."""
        mtime = get_file_mtime_ns(sample_spec_file)

        assert isinstance(mtime, int)
        assert mtime > 0
