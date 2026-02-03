"""Tests for the config module."""

import os
from pathlib import Path

import pytest

from ethereum_mcp.config import (
    ChunkingConfig,
    Config,
    ConfigError,
    EmbeddingConfig,
    EMBEDDING_MODELS,
    load_config,
    save_config,
)


class TestEmbeddingConfig:
    """Tests for embedding configuration."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = EmbeddingConfig()
        assert config.model == "all-MiniLM-L6-v2"
        assert config.batch_size == 32

    def test_model_info(self):
        """Should return model info from EMBEDDING_MODELS."""
        config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        info = config.model_info
        assert info["dimensions"] == 384
        assert info["type"] == "local"

    def test_dimensions(self):
        """Should return correct dimensions for model."""
        config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        assert config.dimensions == 384

        config = EmbeddingConfig(model="codesage/codesage-large")
        assert config.dimensions == 1024

    def test_requires_api_key(self):
        """API models should indicate they need keys."""
        local_config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        assert local_config.requires_api_key is False

        api_config = EmbeddingConfig(model="voyage:voyage-code-3")
        assert api_config.requires_api_key is True

    def test_validate_batch_size_too_small(self):
        """Batch size < 1 should fail validation."""
        config = EmbeddingConfig(batch_size=0)
        with pytest.raises(ConfigError, match="batch_size must be >= 1"):
            config.validate()

    def test_validate_api_key_missing(self, monkeypatch):
        """API model without key should fail validation."""
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        config = EmbeddingConfig(model="voyage:voyage-code-3")
        with pytest.raises(ConfigError, match="requires VOYAGE_API_KEY"):
            config.validate()

    def test_validate_api_key_present(self, monkeypatch):
        """API model with key should pass validation."""
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        config = EmbeddingConfig(model="voyage:voyage-code-3")
        config.validate()  # Should not raise


class TestChunkingConfig:
    """Tests for chunking configuration."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_validate_chunk_size_too_small(self):
        """Chunk size < 100 should fail validation."""
        config = ChunkingConfig(chunk_size=50)
        with pytest.raises(ConfigError, match="chunk_size must be >= 100"):
            config.validate()

    def test_validate_chunk_size_too_large(self):
        """Chunk size > 10000 should fail validation."""
        config = ChunkingConfig(chunk_size=20000)
        with pytest.raises(ConfigError, match="chunk_size must be <= 10000"):
            config.validate()

    def test_validate_overlap_negative(self):
        """Negative overlap should fail validation."""
        config = ChunkingConfig(chunk_overlap=-1)
        with pytest.raises(ConfigError, match="chunk_overlap must be >= 0"):
            config.validate()

    def test_validate_overlap_too_large(self):
        """Overlap >= chunk_size should fail validation."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=500)
        with pytest.raises(ConfigError, match="chunk_overlap.*must be < chunk_size"):
            config.validate()

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        config = ChunkingConfig(chunk_size=800, chunk_overlap=100)
        result = config.to_dict()
        assert result == {"chunk_size": 800, "chunk_overlap": 100}


class TestConfigLoading:
    """Tests for config file loading."""

    def test_load_nonexistent_uses_defaults(self, tmp_path):
        """Missing config file should use defaults."""
        config = load_config(data_dir=tmp_path)
        assert config.embedding.model == "all-MiniLM-L6-v2"
        assert config.chunking.chunk_size == 1000

    def test_load_from_file(self, sample_config_file, temp_data_dir):
        """Should load config from YAML file."""
        config = load_config(config_path=sample_config_file)
        assert config.embedding.model == "all-MiniLM-L6-v2"
        assert config.embedding.batch_size == 32

    def test_handles_malformed_yaml(self, temp_data_dir):
        """Malformed YAML should fall back to defaults."""
        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("{ invalid: yaml: content")

        config = load_config(config_path=config_path)
        # Should use defaults
        assert config.embedding.model == "all-MiniLM-L6-v2"

    def test_validates_model_name_format(self, temp_data_dir):
        """Config should warn about unknown models."""
        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("""
embedding:
  model: "unknown-model"
  batch_size: 32
""")

        # Should load but use defaults for unknown model info
        config = load_config(config_path=config_path)
        assert config.embedding.model == "unknown-model"
        assert config.embedding.dimensions == 384  # Default fallback

    def test_rejects_oversized_config(self, temp_data_dir):
        """Config file > 1MB should fall back to defaults (logged warning)."""
        config_path = temp_data_dir / "config.yaml"
        # Create a large file (>1MB)
        config_path.write_text("x" * (1024 * 1024 + 1))

        # Should fall back to defaults (warning is logged)
        config = load_config(config_path=config_path)
        assert config.embedding.model == "all-MiniLM-L6-v2"


class TestConfigEnvOverrides:
    """Tests for environment variable overrides."""

    def test_env_overrides_model(self, monkeypatch, temp_data_dir, sample_config_file):
        """ETHEREUM_MCP_EMBEDDING_MODEL should override config file."""
        monkeypatch.setenv("ETHEREUM_MCP_EMBEDDING_MODEL", "all-mpnet-base-v2")

        config = load_config(config_path=sample_config_file)
        assert config.embedding.model == "all-mpnet-base-v2"

    def test_env_overrides_batch_size(self, monkeypatch, temp_data_dir, sample_config_file):
        """ETHEREUM_MCP_BATCH_SIZE should override config file."""
        monkeypatch.setenv("ETHEREUM_MCP_BATCH_SIZE", "64")

        config = load_config(config_path=sample_config_file)
        assert config.embedding.batch_size == 64

    def test_invalid_env_batch_size_ignored(self, monkeypatch, temp_data_dir, sample_config_file):
        """Invalid ETHEREUM_MCP_BATCH_SIZE should be ignored."""
        monkeypatch.setenv("ETHEREUM_MCP_BATCH_SIZE", "not-a-number")

        config = load_config(config_path=sample_config_file)
        # Should use value from file
        assert config.embedding.batch_size == 32


class TestConfigSaving:
    """Tests for config file saving."""

    def test_save_creates_file(self, temp_data_dir):
        """Save should create config file."""
        config = Config()
        config_path = temp_data_dir / "config.yaml"

        save_config(config, config_path)

        assert config_path.exists()

    def test_save_roundtrip(self, temp_data_dir):
        """Config should survive save/load roundtrip."""
        original = Config(
            embedding=EmbeddingConfig(model="all-mpnet-base-v2", batch_size=64),
            chunking=ChunkingConfig(chunk_size=800, chunk_overlap=100),
        )
        config_path = temp_data_dir / "config.yaml"

        save_config(original, config_path)
        loaded = load_config(config_path=config_path)

        assert loaded.embedding.model == original.embedding.model
        assert loaded.embedding.batch_size == original.embedding.batch_size
        assert loaded.chunking.chunk_size == original.chunking.chunk_size

    def test_save_creates_parent_directories(self, tmp_path):
        """Save should create parent directories."""
        config = Config()
        config_path = tmp_path / "subdir" / "nested" / "config.yaml"

        save_config(config, config_path)

        assert config_path.exists()


class TestEmbeddingModels:
    """Tests for embedding model definitions."""

    def test_all_models_have_required_fields(self):
        """All models should have dimensions, max_tokens, type, description."""
        required_fields = {"dimensions", "max_tokens", "type", "description"}

        for model_name, info in EMBEDDING_MODELS.items():
            for field in required_fields:
                assert field in info, f"Model {model_name} missing {field}"

    def test_api_models_have_env_var(self):
        """API models should specify their env var."""
        for model_name, info in EMBEDDING_MODELS.items():
            if info["type"] == "api":
                assert "env_var" in info, f"API model {model_name} missing env_var"
