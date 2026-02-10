"""Integration tests for leanSpec functionality."""

import pytest

from ethereum_mcp.indexer.chunker import chunk_lean_spec_single_file, chunk_python_spec_file
from ethereum_mcp.indexer.downloader import get_lean_spec_files


class TestGetLeanSpecFiles:
    """Tests for get_lean_spec_files helper."""

    def test_lists_python_files(self, sample_lean_spec_dir):
        """Should list Python files from leanSpec src directory."""
        files = get_lean_spec_files(sample_lean_spec_dir)

        assert len(files) >= 3  # state.py, constants.py, math.py
        assert all(f.suffix == ".py" for f in files)

    def test_excludes_init_files(self, sample_lean_spec_dir):
        """Should include __init__.py files (they may have content)."""
        files = get_lean_spec_files(sample_lean_spec_dir)

        # __init__.py files are included by default
        # but filtered if they're test files
        file_names = [f.name for f in files]
        assert "state.py" in file_names
        assert "constants.py" in file_names
        assert "math.py" in file_names

    def test_excludes_test_files(self, sample_lean_spec_dir):
        """Should exclude test_ prefixed files."""
        # Create a test file
        test_file = sample_lean_spec_dir / "src" / "lean_spec" / "test_state.py"
        test_file.write_text('def test_something(): pass')

        files = get_lean_spec_files(sample_lean_spec_dir)

        file_names = [f.name for f in files]
        assert "test_state.py" not in file_names

    def test_raises_for_missing_dir(self, tmp_path):
        """Should raise FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            get_lean_spec_files(tmp_path / "nonexistent")

    def test_raises_for_missing_src_dir(self, tmp_path):
        """Should raise FileNotFoundError if src/lean_spec doesn't exist."""
        lean_dir = tmp_path / "leanSpec"
        lean_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            get_lean_spec_files(lean_dir)


class TestLeanSpecChunkingIntegration:
    """Integration tests for leanSpec chunking pipeline."""

    def test_chunks_sample_lean_spec_file(self, sample_lean_spec_file, temp_data_dir):
        """Should chunk Python spec file correctly."""
        chunks = chunk_python_spec_file(
            sample_lean_spec_file,
            base_path=temp_data_dir / "leanSpec",
        )

        # Should have multiple chunk types
        chunk_types = {c.chunk_type for c in chunks}
        assert "lean_doc" in chunk_types  # Module docstring
        assert "lean_constant" in chunk_types  # SLOTS_PER_EPOCH, MAX_VALIDATORS
        assert "lean_class" in chunk_types  # Checkpoint, State
        assert "lean_function" in chunk_types  # Methods

    def test_chunks_have_correct_metadata(self, sample_lean_spec_file, temp_data_dir):
        """Should populate metadata correctly."""
        chunks = chunk_python_spec_file(
            sample_lean_spec_file,
            base_path=temp_data_dir / "leanSpec",
        )

        # Check class metadata
        class_chunks = [c for c in chunks if c.chunk_type == "lean_class"]
        state_chunk = next((c for c in class_chunks if c.section == "State"), None)
        assert state_chunk is not None
        assert state_chunk.metadata["class_name"] == "State"
        assert "main consensus state" in state_chunk.metadata["docstring"].lower()

        # Check function metadata
        func_chunks = [c for c in chunks if c.chunk_type == "lean_function"]
        epoch_func = next(
            (c for c in func_chunks if "get_current_epoch" in c.metadata["function_name"]),
            None,
        )
        assert epoch_func is not None
        assert epoch_func.metadata["is_method"] is True
        assert epoch_func.metadata["class_name"] == "State"

        # Check constant metadata
        const_chunks = [c for c in chunks if c.chunk_type == "lean_constant"]
        slots_const = next(
            (c for c in const_chunks if c.metadata["constant_name"] == "SLOTS_PER_EPOCH"),
            None,
        )
        assert slots_const is not None
        assert slots_const.metadata["type_annotation"] == "int"

    def test_chunks_entire_directory(self, sample_lean_spec_dir, temp_data_dir):
        """Should chunk all files in leanSpec directory."""
        files = get_lean_spec_files(sample_lean_spec_dir)

        all_chunks = []
        for f in files:
            chunks = chunk_lean_spec_single_file(f, base_path=temp_data_dir)
            all_chunks.extend(chunks)

        # Should have chunks from multiple files
        assert len(all_chunks) > 5

        # Should have unique chunk IDs
        ids = [c.chunk_id for c in all_chunks if c.chunk_id]
        assert len(ids) == len(set(ids))

    def test_relative_paths_are_correct(self, sample_lean_spec_file, temp_data_dir):
        """Should use relative paths for source field."""
        chunks = chunk_python_spec_file(
            sample_lean_spec_file,
            base_path=temp_data_dir / "leanSpec",
        )

        for chunk in chunks:
            # Should be relative to leanSpec dir
            assert not chunk.source.startswith("/")
            assert "leanSpec" not in chunk.source
            assert chunk.source.startswith("src/lean_spec")


class TestLeanSpecIncrementalIndexing:
    """Tests for incremental indexing of leanSpec files."""

    def test_incremental_chunk_ids_stable(self, sample_lean_spec_file, temp_data_dir):
        """Chunk IDs should be stable across multiple runs."""
        chunks1 = chunk_lean_spec_single_file(sample_lean_spec_file, base_path=temp_data_dir)
        chunks2 = chunk_lean_spec_single_file(sample_lean_spec_file, base_path=temp_data_dir)

        ids1 = sorted([c.chunk_id for c in chunks1])
        ids2 = sorted([c.chunk_id for c in chunks2])

        assert ids1 == ids2

    def test_chunk_ids_change_on_content_change(self, sample_lean_spec_file, temp_data_dir):
        """Chunk IDs should change when content changes."""
        chunks1 = chunk_lean_spec_single_file(sample_lean_spec_file, base_path=temp_data_dir)
        original_ids = {c.chunk_id for c in chunks1}

        # Modify the file
        content = sample_lean_spec_file.read_text()
        modified_content = content.replace(
            "SLOTS_PER_EPOCH: int = 32", "SLOTS_PER_EPOCH: int = 64"
        )
        sample_lean_spec_file.write_text(modified_content)

        chunks2 = chunk_lean_spec_single_file(sample_lean_spec_file, base_path=temp_data_dir)
        new_ids = {c.chunk_id for c in chunks2}

        # The constant chunk should have a different ID
        # But other chunks should remain the same
        assert original_ids != new_ids
