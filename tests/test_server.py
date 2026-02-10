"""Tests for the MCP server tools."""

from ethereum_mcp.server import (
    CURRENT_FORK,
    DEFAULT_DATA_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_SPECS_DIR,
    FORKS,
    GITHUB_REPOS,
    _add_github_url,
    _source_to_github_url,
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


class TestGitHubUrlGeneration:
    """Tests for GitHub URL generation."""

    def test_github_repos_have_correct_branches(self):
        """Ensure repos use correct branches."""
        assert GITHUB_REPOS["consensus-specs"]["branch"] == "master"
        assert GITHUB_REPOS["EIPs"]["branch"] == "master"
        assert GITHUB_REPOS["builder-specs"]["branch"] == "main"

    def test_source_to_github_url_with_repo(self):
        """Test URL generation with repo parameter (new format)."""
        url = _source_to_github_url("specs/electra/beacon-chain.md", repo="consensus-specs")
        assert url == "https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md"

    def test_source_to_github_url_eips_with_repo(self):
        """Test EIP URL generation with repo parameter."""
        url = _source_to_github_url("EIPS/eip-4844.md", repo="EIPs")
        assert url == "https://github.com/ethereum/EIPs/blob/master/EIPS/eip-4844.md"

    def test_source_to_github_url_builder_specs_with_repo(self):
        """Test builder-specs URL generation with repo parameter."""
        url = _source_to_github_url("specs/bellatrix/builder.md", repo="builder-specs")
        assert url == "https://github.com/ethereum/builder-specs/blob/main/specs/bellatrix/builder.md"

    def test_source_to_github_url_fallback_absolute_path(self):
        """Test fallback for absolute paths (backwards compatibility)."""
        path = "/Users/someone/.ethereum-mcp/consensus-specs/specs/electra/beacon-chain.md"
        url = _source_to_github_url(path)
        assert url == "https://github.com/ethereum/consensus-specs/blob/master/specs/electra/beacon-chain.md"

    def test_source_to_github_url_unknown_repo(self):
        """Test that unknown repos return None."""
        url = _source_to_github_url("some/file.md", repo="unknown-repo")
        assert url is None

    def test_source_to_github_url_empty_path(self):
        """Test that empty paths return None."""
        url = _source_to_github_url("", repo="consensus-specs")
        assert url is None

    def test_add_github_url_with_repo_field(self):
        """Test _add_github_url uses repo field from result."""
        result = {
            "content": "test",
            "source": "specs/electra/beacon-chain.md",
            "repo": "consensus-specs",
        }
        updated = _add_github_url(result)
        assert "github_url" in updated
        assert "blob/master" in updated["github_url"]
        assert "specs/electra/beacon-chain.md" in updated["github_url"]

    def test_add_github_url_no_source(self):
        """Test _add_github_url handles missing source."""
        result = {"content": "test"}
        updated = _add_github_url(result)
        assert "github_url" not in updated

    def test_add_github_url_unknown_source(self):
        """Test _add_github_url handles unrecognized paths without repo."""
        result = {"content": "test", "source": "/random/path.md"}
        updated = _add_github_url(result)
        assert "github_url" not in updated
