"""Integration tests for leanEthereum post-quantum repos."""

from pathlib import Path

import pytest


class TestLeanPQDownloader:
    """Tests for downloading leanEthereum PQ repos."""

    def test_specs_config_has_lean_pq_urls(self):
        """Test that SpecsConfig includes leanEthereum PQ repo URLs."""
        from ethereum_mcp.indexer.downloader import SpecsConfig

        config = SpecsConfig()

        # Check all PQ repos are configured
        assert hasattr(config, "lean_sig_url")
        assert "leanSig" in config.lean_sig_url
        assert hasattr(config, "lean_multisig_url")
        assert "leanMultisig" in config.lean_multisig_url
        assert hasattr(config, "multilinear_toolkit_url")
        assert hasattr(config, "fiat_shamir_url")
        assert hasattr(config, "lean_pm_url")
        assert hasattr(config, "lean_snappy_url")

    def test_lean_pq_repos_dict(self):
        """Test the LEAN_PQ_REPOS dictionary."""
        from ethereum_mcp.indexer.downloader import LEAN_PQ_REPOS

        # Check all expected repos are present
        expected_repos = [
            "leanSig",
            "leanMultisig",
            "multilinear-toolkit",
            "fiat-shamir",
            "pm",
            "leanSnappy",
        ]

        for repo in expected_repos:
            assert repo in LEAN_PQ_REPOS
            assert "url" in LEAN_PQ_REPOS[repo]
            assert "branch" in LEAN_PQ_REPOS[repo]
            assert "language" in LEAN_PQ_REPOS[repo]

    def test_get_lean_rust_files(self, sample_lean_pq_dir):
        """Test getting Rust files from a leanEthereum repo."""
        from ethereum_mcp.indexer.downloader import get_lean_rust_files

        lean_sig_dir = sample_lean_pq_dir / "leanSig"
        rust_files = get_lean_rust_files(lean_sig_dir)

        assert len(rust_files) >= 1
        assert all(f.suffix == ".rs" for f in rust_files)

    def test_get_lean_pm_files(self, sample_lean_pq_dir):
        """Test getting markdown files from pm repo."""
        from ethereum_mcp.indexer.downloader import get_lean_pm_files

        pm_dir = sample_lean_pq_dir / "pm"
        md_files = get_lean_pm_files(pm_dir)

        assert len(md_files) >= 1
        assert all(f.suffix == ".md" for f in md_files)

    def test_get_lean_rust_files_not_found(self, temp_data_dir):
        """Test handling of non-existent repo directory."""
        from ethereum_mcp.indexer.downloader import get_lean_rust_files

        with pytest.raises(FileNotFoundError):
            get_lean_rust_files(temp_data_dir / "nonexistent")


class TestLeanPQIndexing:
    """Tests for indexing leanEthereum PQ repos."""

    def test_chunk_rust_files_from_repo(self, sample_lean_pq_dir):
        """Test chunking Rust files from a leanEthereum repo."""
        from ethereum_mcp.indexer.chunker import chunk_rust_file
        from ethereum_mcp.indexer.downloader import get_lean_rust_files

        lean_sig_dir = sample_lean_pq_dir / "leanSig"
        rust_files = get_lean_rust_files(lean_sig_dir)

        all_chunks = []
        for rust_file in rust_files:
            chunks = chunk_rust_file(
                rust_file,
                base_path=lean_sig_dir,
                repo="leanSig",
            )
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0

        # Check chunk types
        chunk_types = {c.chunk_type for c in all_chunks}
        # Should have at least one Rust-specific chunk type
        rust_types = {"rust_function", "rust_struct", "rust_enum", "rust_const", "rust_impl"}
        assert len(chunk_types & rust_types) > 0

    def test_rust_chunks_have_metadata(self, sample_lean_pq_dir):
        """Test that Rust chunks have proper metadata."""
        from ethereum_mcp.indexer.chunker import chunk_rust_file
        from ethereum_mcp.indexer.downloader import get_lean_rust_files

        lean_sig_dir = sample_lean_pq_dir / "leanSig"
        rust_files = get_lean_rust_files(lean_sig_dir)

        for rust_file in rust_files:
            chunks = chunk_rust_file(rust_file, base_path=lean_sig_dir, repo="leanSig")
            for chunk in chunks:
                # All chunks should have repo set
                assert chunk.repo == "leanSig"

                # Function chunks should have function_name
                if chunk.chunk_type == "rust_function":
                    assert "function_name" in chunk.metadata

                # Constant chunks should have constant_name
                if chunk.chunk_type == "rust_const":
                    assert "constant_name" in chunk.metadata

    def test_chunk_single_file_with_repo_type(self, sample_lean_pq_dir):
        """Test chunk_single_file with rust:repo_name type."""
        from ethereum_mcp.indexer.chunker import chunk_single_file
        from ethereum_mcp.indexer.downloader import get_lean_rust_files

        lean_sig_dir = sample_lean_pq_dir / "leanSig"
        rust_files = get_lean_rust_files(lean_sig_dir)

        if rust_files:
            chunks = chunk_single_file(
                rust_files[0],
                file_type="rust:leanSig",
                base_path=sample_lean_pq_dir,
            )

            assert len(chunks) > 0

            # All chunks should have IDs
            for chunk in chunks:
                assert chunk.chunk_id != ""


class TestLeanPQServer:
    """Tests for leanEthereum PQ server tools."""

    def test_github_repos_includes_lean_pq(self):
        """Test that GITHUB_REPOS includes leanEthereum PQ repos."""
        from ethereum_mcp.server import GITHUB_REPOS

        pq_repos = ["leanSig", "leanMultisig", "multilinear-toolkit", "fiat-shamir", "pm", "leanSnappy"]

        for repo in pq_repos:
            assert repo in GITHUB_REPOS
            assert "url" in GITHUB_REPOS[repo]
            assert "leanEthereum" in GITHUB_REPOS[repo]["url"]

    def test_source_to_github_url_for_lean_pq(self):
        """Test GitHub URL generation for leanEthereum repos."""
        from ethereum_mcp.server import _source_to_github_url

        # Test leanSig
        url = _source_to_github_url("src/lib.rs", repo="leanSig")
        assert url is not None
        assert "leanEthereum/leanSig" in url
        assert "src/lib.rs" in url

        # Test leanMultisig
        url = _source_to_github_url("src/aggregator.rs", repo="leanMultisig")
        assert url is not None
        assert "leanEthereum/leanMultisig" in url


class TestLeanPQCLI:
    """Tests for CLI commands with leanEthereum PQ repos."""

    def test_lean_pq_repos_imported(self):
        """Test that LEAN_PQ_REPOS is available in CLI."""
        from ethereum_mcp.cli import LEAN_PQ_REPOS

        assert len(LEAN_PQ_REPOS) > 0
        assert "leanSig" in LEAN_PQ_REPOS

    def test_get_lean_rust_files_imported(self):
        """Test that get_lean_rust_files is available in CLI."""
        from ethereum_mcp.cli import get_lean_rust_files

        assert callable(get_lean_rust_files)

    def test_get_lean_pm_files_imported(self):
        """Test that get_lean_pm_files is available in CLI."""
        from ethereum_mcp.cli import get_lean_pm_files

        assert callable(get_lean_pm_files)
