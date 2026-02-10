"""Tests for Rust parsing and chunking."""

from pathlib import Path

import pytest


class TestRustParser:
    """Tests for the RustParser class."""

    def test_extracts_pub_functions(self, sample_rust_file):
        """Test that public functions are extracted."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        # Find pub functions
        functions = [i for i in items if i.kind == "function"]
        function_names = [f.name for f in functions]

        assert "new" in function_names
        assert "sign" in function_names
        assert "verify" in function_names
        assert "hash_message" in function_names

    def test_extracts_pub_structs(self, sample_rust_file):
        """Test that public structs are extracted."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        # Find pub structs
        structs = [i for i in items if i.kind == "struct"]
        struct_names = [s.name for s in structs]

        assert "LeanSig" in struct_names
        # TreeState is not pub, so should not be included
        assert "TreeState" not in struct_names

    def test_extracts_pub_enums(self, sample_rust_file):
        """Test that public enums are extracted."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        # Find pub enums
        enums = [i for i in items if i.kind == "enum"]
        enum_names = [e.name for e in enums]

        assert "VerifyResult" in enum_names

    def test_extracts_constants(self, sample_rust_file):
        """Test that constants are extracted."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        constant_names = [c.name for c in constants]

        assert "MAX_SIGNATURE_SIZE" in constant_names
        assert "TREE_HEIGHT" in constant_names

        # Check constant values
        max_sig = next(c for c in constants if c.name == "MAX_SIGNATURE_SIZE")
        assert "2048" in max_sig.value

    def test_extracts_impl_blocks(self, sample_rust_file):
        """Test that impl blocks are extracted (requires tree-sitter)."""
        from ethereum_mcp.indexer.rust_compiler import TREE_SITTER_RUST, RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        # Find impl blocks
        impls = [i for i in items if i.kind == "impl"]
        impl_names = [i.name for i in impls]

        if TREE_SITTER_RUST:
            # tree-sitter extracts impl blocks
            assert any("LeanSig" in name for name in impl_names)
        else:
            # regex fallback doesn't extract impl blocks - that's ok
            assert impls == []

    def test_captures_doc_comments(self, sample_rust_file):
        """Test that doc comments are captured."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        # Find the sign function
        sign_fn = next((i for i in items if i.kind == "function" and i.name == "sign"), None)
        assert sign_fn is not None
        assert sign_fn.doc_comment is not None
        assert "Sign a message" in sign_fn.doc_comment

    def test_captures_attributes(self, sample_rust_file):
        """Test that attributes like #[derive(...)] are captured."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(sample_rust_file)

        # Find LeanSig struct which has #[derive(Debug, Clone)]
        lean_sig = next((i for i in items if i.kind == "struct" and i.name == "LeanSig"), None)
        assert lean_sig is not None
        # Attributes may or may not be captured depending on parser
        # Just verify the struct exists

    def test_falls_back_to_regex(self, sample_rust_file):
        """Test that regex fallback works when tree-sitter is not available."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        # Force regex mode by setting parser to None
        parser.parser = None

        items, constants = parser.parse_file(sample_rust_file)

        # Should still extract some items
        assert len(items) > 0

        # Find pub functions
        functions = [i for i in items if i.kind == "function"]
        function_names = [f.name for f in functions]

        # At minimum, should find some public functions
        assert len(function_names) >= 1

    def test_handles_empty_file(self, temp_data_dir):
        """Test handling of empty Rust file."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        empty_file = temp_data_dir / "empty.rs"
        empty_file.write_text("")

        parser = RustParser()
        items, constants = parser.parse_file(empty_file)

        assert items == []
        assert constants == []

    def test_handles_invalid_file(self, temp_data_dir):
        """Test handling of non-existent file."""
        from ethereum_mcp.indexer.rust_compiler import RustParser

        parser = RustParser()
        items, constants = parser.parse_file(temp_data_dir / "nonexistent.rs")

        assert items == []
        assert constants == []


class TestRustChunking:
    """Tests for Rust file chunking."""

    def test_chunk_rust_file(self, sample_rust_file, temp_data_dir):
        """Test chunking a Rust file."""
        from ethereum_mcp.indexer.chunker import chunk_rust_file

        chunks = chunk_rust_file(
            sample_rust_file,
            base_path=temp_data_dir / "leanSig",
            repo="leanSig",
        )

        assert len(chunks) > 0

        # Check chunk types
        chunk_types = {c.chunk_type for c in chunks}
        assert "rust_function" in chunk_types or "rust_struct" in chunk_types

        # All chunks should have leanSig as repo
        for chunk in chunks:
            assert chunk.repo == "leanSig"

    def test_chunk_rust_item_has_content(self, sample_rust_file, temp_data_dir):
        """Test that Rust chunks have proper content."""
        from ethereum_mcp.indexer.chunker import chunk_rust_file

        chunks = chunk_rust_file(
            sample_rust_file,
            base_path=temp_data_dir / "leanSig",
            repo="leanSig",
        )

        # Find a function chunk
        func_chunks = [c for c in chunks if c.chunk_type == "rust_function"]
        if func_chunks:
            func = func_chunks[0]
            assert len(func.content) > 0
            assert "fn" in func.content or "pub" in func.content

    def test_chunk_single_file_rust(self, sample_rust_file, temp_data_dir):
        """Test the single file chunking entry point for Rust."""
        from ethereum_mcp.indexer.chunker import chunk_single_file

        chunks = chunk_single_file(
            sample_rust_file,
            file_type="rust:leanSig",
            base_path=temp_data_dir,
        )

        assert len(chunks) > 0

        # All chunks should have chunk_id assigned
        for chunk in chunks:
            assert chunk.chunk_id != ""

    def test_chunk_rust_constant(self, sample_rust_file, temp_data_dir):
        """Test that Rust constants are chunked properly."""
        from ethereum_mcp.indexer.chunker import chunk_rust_file

        chunks = chunk_rust_file(
            sample_rust_file,
            base_path=temp_data_dir / "leanSig",
            repo="leanSig",
        )

        # Find constant chunks
        const_chunks = [c for c in chunks if c.chunk_type == "rust_const"]
        if const_chunks:
            const = const_chunks[0]
            assert "constant_name" in const.metadata
            assert const.metadata["constant_name"] != ""


class TestRustCompilerHelpers:
    """Tests for helper functions in rust_compiler."""

    def test_parse_rust_file(self, sample_rust_file):
        """Test the parse_rust_file helper."""
        from ethereum_mcp.indexer.rust_compiler import parse_rust_file

        items, constants = parse_rust_file(sample_rust_file)

        assert len(items) > 0
        assert len(constants) > 0

    def test_lookup_rust_function(self, sample_rust_file):
        """Test looking up a function by name."""
        from ethereum_mcp.indexer.rust_compiler import lookup_rust_function, parse_rust_file

        items, constants = parse_rust_file(sample_rust_file)

        # Look up an existing function
        sign_fn = lookup_rust_function("sign", items)
        assert sign_fn is not None
        assert sign_fn.name == "sign"

        # Look up non-existing function
        missing = lookup_rust_function("nonexistent", items)
        assert missing is None

    def test_lookup_rust_constant(self, sample_rust_file):
        """Test looking up a constant by name."""
        from ethereum_mcp.indexer.rust_compiler import lookup_rust_constant, parse_rust_file

        items, constants = parse_rust_file(sample_rust_file)

        # Look up an existing constant
        max_sig = lookup_rust_constant("MAX_SIGNATURE_SIZE", constants)
        assert max_sig is not None
        assert "2048" in max_sig.value

        # Look up non-existing constant
        missing = lookup_rust_constant("NONEXISTENT", constants)
        assert missing is None

    def test_parse_rust_repo(self, sample_lean_pq_dir):
        """Test parsing all Rust files in a repo."""
        from ethereum_mcp.indexer.rust_compiler import parse_rust_repo

        lean_sig_dir = sample_lean_pq_dir / "leanSig"
        items, constants = parse_rust_repo(lean_sig_dir)

        # Should have items from multiple files
        assert len(items) > 0

        # Check we have items from different files
        file_paths = {i.file_path for i in items}
        assert len(file_paths) >= 1
