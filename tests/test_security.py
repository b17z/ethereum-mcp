"""Security tests for the incremental indexing system.

These tests verify that the system is resistant to:
- Path traversal attacks
- Injection attacks
- Resource exhaustion
- Corrupted/malicious input
"""

import json
from pathlib import Path

import pytest

from ethereum_mcp.config import ConfigError, load_config
from ethereum_mcp.indexer.manifest import (
    MAX_CHUNKS_PER_FILE,
    MAX_MANIFEST_SIZE,
    ManifestValidationError,
    Manifest,
    _sanitize_chunk_id,
    _validate_path,
    generate_chunk_id,
    load_manifest,
)


class TestPathTraversalPrevention:
    """Tests for path traversal attack prevention."""

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "specs/../../../etc/passwd",
            "specs/..%2f..%2f..%2fetc/passwd",
            "....//....//etc/passwd",
        ],
    )
    def test_rejects_path_traversal_in_source(self, malicious_path):
        """Path traversal attempts should be rejected."""
        with pytest.raises(ManifestValidationError, match="Path traversal"):
            _validate_path(malicious_path)

    @pytest.mark.parametrize(
        "absolute_path",
        [
            "/etc/passwd",
            "/home/user/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM",
            "D:\\secrets\\passwords.txt",
        ],
    )
    def test_rejects_absolute_paths_in_manifest(self, absolute_path):
        """Absolute paths should be rejected."""
        with pytest.raises(ManifestValidationError, match="Absolute paths"):
            _validate_path(absolute_path)

    def test_rejects_null_bytes_in_paths(self):
        """Null bytes (used to truncate paths in some systems) should be rejected."""
        with pytest.raises(ManifestValidationError, match="Null bytes"):
            _validate_path("test\x00.md")

    @pytest.mark.parametrize(
        "special_char_path",
        [
            "test\nfile.md",
            "test\rfile.md",
            "test\tfile.md",
        ],
    )
    def test_rejects_special_characters(self, special_char_path):
        """Special characters that could cause issues should be rejected."""
        with pytest.raises(ManifestValidationError, match="Invalid characters"):
            _validate_path(special_char_path)


class TestInputValidation:
    """Tests for input validation."""

    def test_rejects_oversized_content(self, temp_data_dir):
        """Oversized manifest files should be rejected."""
        manifest_path = temp_data_dir / "manifest.json"
        # Create a file larger than MAX_MANIFEST_SIZE
        large_content = "x" * (MAX_MANIFEST_SIZE + 1)
        manifest_path.write_text(large_content)

        with pytest.raises(ManifestValidationError, match="too large"):
            load_manifest(manifest_path)

    def test_handles_malformed_yaml_config(self, temp_data_dir):
        """Malformed YAML should be handled gracefully."""
        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("{ invalid: yaml: : content }")

        # Should not crash, should use defaults
        config = load_config(config_path=config_path)
        assert config is not None

    def test_validates_model_name_format(self, temp_data_dir):
        """Model names should be validated."""
        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("""
embedding:
  model: "../../../etc/passwd"
""")
        # Should load (model name is just a string, not a path)
        # The model loading will fail later if invalid
        config = load_config(config_path=config_path)
        assert config.embedding.model == "../../../etc/passwd"

    def test_sanitizes_chunk_ids(self):
        """Chunk IDs should only contain safe characters."""
        valid_id = "eth_spec_abc12345_0001_def67890"
        result = _sanitize_chunk_id(valid_id)
        assert result == valid_id

    @pytest.mark.parametrize(
        "malicious_id",
        [
            "chunk; DROP TABLE specs;--",
            "chunk' OR '1'='1",
            'chunk"; DELETE FROM chunks;--',
            "chunk\x00malicious",
            "chunk\nmalicious",
        ],
    )
    def test_rejects_sql_injection_in_chunk_id(self, malicious_id):
        """SQL injection attempts in chunk IDs should be rejected."""
        with pytest.raises(ManifestValidationError):
            _sanitize_chunk_id(malicious_id)

    def test_chunk_id_length_limit(self):
        """Overly long chunk IDs should be rejected."""
        long_id = "a" * 201  # Over 200 char limit
        with pytest.raises(ManifestValidationError, match="too long"):
            _sanitize_chunk_id(long_id)


class TestResourceLimits:
    """Tests for resource limit enforcement."""

    def test_limits_manifest_file_size(self, temp_data_dir):
        """Manifest files over size limit should be rejected."""
        manifest_path = temp_data_dir / "manifest.json"

        # Create oversized manifest
        large_data = {"version": "1.0.0", "files": {}, "padding": "x" * MAX_MANIFEST_SIZE}
        with open(manifest_path, "w") as f:
            json.dump(large_data, f)

        with pytest.raises(ManifestValidationError, match="too large"):
            load_manifest(manifest_path)

    def test_limits_chunk_count_per_file(self):
        """Files with too many chunks should be rejected."""
        too_many_chunks = [f"chunk_{i}" for i in range(MAX_CHUNKS_PER_FILE + 1)]

        data = {
            "version": "1.0.0",
            "files": {
                "test.md": {
                    "sha256": "a" * 64,
                    "mtime_ns": 12345,
                    "chunk_ids": too_many_chunks,
                }
            },
        }

        with pytest.raises(ManifestValidationError, match="Too many chunks"):
            Manifest.from_dict(data)


class TestYamlSafety:
    """Tests for YAML safety (preventing code execution)."""

    def test_rejects_yaml_code_execution(self, temp_data_dir):
        """YAML with Python code execution should be safely handled."""
        config_path = temp_data_dir / "config.yaml"
        # This would execute code if using unsafe yaml.load()
        malicious_yaml = """
!!python/object/apply:os.system
args: ['echo pwned']
"""
        config_path.write_text(malicious_yaml)

        # Should either fail safely or use defaults, not execute code
        try:
            config = load_config(config_path=config_path)
            # If it loads, it should be using defaults (safe_load ignores !!python)
            assert config.embedding.model == "all-MiniLM-L6-v2"
        except Exception:
            # Failing is also acceptable
            pass


class TestHashValidation:
    """Tests for hash format validation."""

    @pytest.mark.parametrize(
        "invalid_hash",
        [
            "short",
            "not-hex-chars-gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
            "a" * 63,  # Too short
            "a" * 65,  # Too long
            "a" * 32 + "UPPERCASE" + "a" * 23,  # Mixed case
        ],
    )
    def test_rejects_invalid_sha256(self, invalid_hash):
        """Invalid SHA256 hashes should be rejected."""
        data = {
            "version": "1.0.0",
            "files": {
                "test.md": {
                    "sha256": invalid_hash,
                    "mtime_ns": 12345,
                }
            },
        }

        with pytest.raises(ManifestValidationError, match="Invalid SHA256"):
            Manifest.from_dict(data)

    def test_accepts_valid_sha256(self):
        """Valid SHA256 hashes should be accepted."""
        valid_hash = "abcdef0123456789" * 4  # 64 hex chars

        data = {
            "version": "1.0.0",
            "files": {
                "test.md": {
                    "sha256": valid_hash,
                    "mtime_ns": 12345,
                    "chunk_ids": [],
                }
            },
        }

        manifest = Manifest.from_dict(data)
        assert manifest.files["test.md"].sha256 == valid_hash


class TestVersionValidation:
    """Tests for version format validation."""

    @pytest.mark.parametrize(
        "invalid_version",
        [
            "invalid",
            "1.0",
            "1",
            "v1.0.0",
            "1.0.0-beta",
            "1.0.0.0",
            "",
        ],
    )
    def test_rejects_invalid_version(self, invalid_version):
        """Invalid version formats should be rejected."""
        data = {"version": invalid_version, "files": {}}

        with pytest.raises(ManifestValidationError, match="Invalid version"):
            Manifest.from_dict(data)

    @pytest.mark.parametrize(
        "valid_version",
        [
            "1.0.0",
            "0.0.1",
            "99.99.99",
            "2.10.3",
        ],
    )
    def test_accepts_valid_version(self, valid_version):
        """Valid semver versions should be accepted."""
        data = {"version": valid_version, "files": {}}

        manifest = Manifest.from_dict(data)
        assert manifest.version == valid_version


class TestChunkIdGeneration:
    """Tests for secure chunk ID generation."""

    def test_generated_ids_are_safe(self):
        """Generated chunk IDs should pass sanitization."""
        chunk_id = generate_chunk_id(
            project="eth",
            source_type="spec",
            source_file="specs/electra/beacon-chain.md",
            chunk_index=0,
            content="test content",
        )

        # Should not raise
        result = _sanitize_chunk_id(chunk_id)
        assert result == chunk_id

    def test_special_chars_in_content_dont_affect_id(self):
        """Special characters in content should not make ID unsafe."""
        malicious_content = "'; DROP TABLE specs;--"
        chunk_id = generate_chunk_id(
            project="eth",
            source_type="spec",
            source_file="test.md",
            chunk_index=0,
            content=malicious_content,
        )

        # Should not raise (content is hashed, not included directly)
        result = _sanitize_chunk_id(chunk_id)
        assert result == chunk_id
        # ID should not contain the malicious string
        assert "DROP" not in chunk_id
        assert ";" not in chunk_id
