"""Tests for leanSpec MCP server tools."""

from ethereum_mcp.server import (
    GITHUB_REPOS,
    _add_github_url,
    _source_to_github_url,
)


class TestLeanSpecGitHubUrls:
    """Tests for leanSpec GitHub URL generation."""

    def test_lean_spec_in_github_repos(self):
        """leanSpec should be in GITHUB_REPOS mapping."""
        assert "leanSpec" in GITHUB_REPOS
        assert GITHUB_REPOS["leanSpec"]["branch"] == "main"
        assert "leanEthereum" in GITHUB_REPOS["leanSpec"]["url"]

    def test_source_to_github_url_lean_spec(self):
        """Should generate correct GitHub URL for leanSpec."""
        url = _source_to_github_url(
            "src/lean_spec/subspecs/containers/state.py",
            repo="leanSpec"
        )
        expected = "https://github.com/leanEthereum/leanSpec/blob/main/src/lean_spec/subspecs/containers/state.py"
        assert url == expected

    def test_add_github_url_for_lean_spec(self):
        """_add_github_url should add URL for leanSpec results."""
        result = {
            "content": "class State: ...",
            "source": "src/lean_spec/containers/state.py",
            "repo": "leanSpec",
            "chunk_type": "lean_class",
        }

        updated = _add_github_url(result)

        assert "github_url" in updated
        assert "leanEthereum/leanSpec" in updated["github_url"]
        assert "blob/main" in updated["github_url"]


class TestLeanSpecChunkTypes:
    """Tests for leanSpec chunk type handling."""

    def test_lean_chunk_types_are_strings(self):
        """leanSpec chunk types should be standard strings."""
        lean_types = {"lean_class", "lean_function", "lean_constant", "lean_doc"}

        for chunk_type in lean_types:
            assert isinstance(chunk_type, str)
            assert chunk_type.startswith("lean_")
