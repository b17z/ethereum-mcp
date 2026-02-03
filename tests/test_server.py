"""Tests for the MCP server tools."""

from ethereum_mcp.server import (
    CURRENT_FORK,
    DEFAULT_DATA_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_SPECS_DIR,
    FORKS,
)


class TestForkConstants:
    """Tests for fork-related constants."""

    def test_forks_dict_has_expected_forks(self):
        expected = ["phase0", "altair", "bellatrix", "capella", "deneb", "electra"]
        for fork in expected:
            assert fork in FORKS, f"Missing fork: {fork}"

    def test_forks_have_required_fields(self):
        for name, info in FORKS.items():
            assert "epoch" in info, f"{name} missing epoch"
            assert "date" in info, f"{name} missing date"
            assert "description" in info, f"{name} missing description"

    def test_forks_epochs_are_ascending(self):
        epochs = [info["epoch"] for info in FORKS.values()]
        assert epochs == sorted(epochs), "Fork epochs should be in ascending order"

    def test_current_fork_exists(self):
        assert CURRENT_FORK in FORKS


class TestServerPaths:
    """Tests for server path configuration."""

    def test_default_data_dir_is_in_home(self):
        from pathlib import Path

        assert str(Path.home()) in str(DEFAULT_DATA_DIR)
        assert ".ethereum-mcp" in str(DEFAULT_DATA_DIR)

    def test_db_path_is_under_data_dir(self):
        assert str(DEFAULT_DATA_DIR) in str(DEFAULT_DB_PATH)

    def test_specs_dir_is_under_data_dir(self):
        assert str(DEFAULT_DATA_DIR) in str(DEFAULT_SPECS_DIR)
