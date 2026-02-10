"""Tests for leanSpec Python chunking functionality."""

from ethereum_mcp.indexer.chunker import (
    _extract_class_chunks,
    _is_constant_name,
    chunk_lean_spec_single_file,
    chunk_python_spec_file,
)


class TestChunkPythonSpecFile:
    """Tests for chunk_python_spec_file."""

    def test_extracts_class_definitions(self, tmp_path):
        """Should extract Container classes with docstrings."""
        py_file = tmp_path / "state.py"
        py_file.write_text('''"""State Container for Lean Ethereum."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint in the chain.

    Contains epoch and root hash.
    """
    epoch: int
    root: bytes
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)

        class_chunks = [c for c in chunks if c.chunk_type == "lean_class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].section == "Checkpoint"
        assert "A checkpoint in the chain" in class_chunks[0].metadata["docstring"]

    def test_extracts_function_definitions(self, tmp_path):
        """Should extract functions with signatures."""
        py_file = tmp_path / "helpers.py"
        py_file.write_text('''def get_current_epoch(state: "BeaconState") -> int:
    """Return current epoch from slot."""
    return state.slot // SLOTS_PER_EPOCH


async def process_block(state: "BeaconState", block: "Block") -> None:
    """Process a beacon block."""
    pass
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)

        func_chunks = [c for c in chunks if c.chunk_type == "lean_function"]
        assert len(func_chunks) == 2

        names = [c.metadata["function_name"] for c in func_chunks]
        assert "get_current_epoch" in names
        assert "process_block" in names

        # Check async function is detected
        async_func = next(c for c in func_chunks if c.metadata["function_name"] == "process_block")
        assert "async def" in async_func.content

    def test_extracts_constants(self, tmp_path):
        """Should extract module-level constants."""
        py_file = tmp_path / "config.py"
        py_file.write_text('''SLOTS_PER_EPOCH: int = 32
MAX_VALIDATORS: int = 2**20
BASE_REWARD_FACTOR = 64
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)

        const_chunks = [c for c in chunks if c.chunk_type == "lean_constant"]
        assert len(const_chunks) == 3

        names = [c.metadata["constant_name"] for c in const_chunks]
        assert "SLOTS_PER_EPOCH" in names
        assert "MAX_VALIDATORS" in names
        assert "BASE_REWARD_FACTOR" in names

    def test_extracts_module_docstring(self, tmp_path):
        """Should extract module-level docstring."""
        py_file = tmp_path / "module.py"
        py_file.write_text('''"""This is the module docstring.

It spans multiple lines and provides context.
"""

def some_func():
    pass
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)

        doc_chunks = [c for c in chunks if c.chunk_type == "lean_doc"]
        assert len(doc_chunks) == 1
        assert "module docstring" in doc_chunks[0].content

    def test_handles_nested_classes(self, tmp_path):
        """Should handle classes with nested definitions."""
        py_file = tmp_path / "nested.py"
        py_file.write_text('''class State:
    """The main consensus state."""
    slot: int
    validators: list

    def get_current_epoch(self) -> int:
        """Return current epoch from slot."""
        return self.slot // 32

    def is_valid(self) -> bool:
        """Check if state is valid."""
        return self.slot >= 0
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)

        # Should have class chunk
        class_chunks = [c for c in chunks if c.chunk_type == "lean_class"]
        assert len(class_chunks) == 1
        assert class_chunks[0].section == "State"

        # Should also have method chunks
        func_chunks = [c for c in chunks if c.chunk_type == "lean_function"]
        assert len(func_chunks) == 2

        # Methods should include class name
        method_names = [c.metadata["full_name"] for c in func_chunks]
        assert "State.get_current_epoch" in method_names
        assert "State.is_valid" in method_names

    def test_preserves_type_annotations(self, tmp_path):
        """Should preserve type annotations in signatures."""
        py_file = tmp_path / "typed.py"
        py_file.write_text('''from typing import List, Optional

def process_validators(
    validators: List["Validator"],
    indices: Optional[List[int]] = None,
) -> List["Validator"]:
    """Process validators."""
    return validators
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)

        func_chunks = [c for c in chunks if c.chunk_type == "lean_function"]
        assert len(func_chunks) == 1

        sig = func_chunks[0].metadata["signature"]
        assert "List" in sig
        assert "Validator" in sig

    def test_handles_syntax_error(self, tmp_path):
        """Should return empty list for files with syntax errors."""
        py_file = tmp_path / "invalid.py"
        py_file.write_text('''def broken(
    # Missing closing paren
''')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path)
        assert chunks == []

    def test_uses_relative_source_path(self, tmp_path):
        """Should use relative path for source field."""
        spec_dir = tmp_path / "leanSpec" / "src" / "lean_spec"
        spec_dir.mkdir(parents=True)
        py_file = spec_dir / "state.py"
        py_file.write_text('CONST = 1')

        chunks = chunk_python_spec_file(py_file, base_path=tmp_path / "leanSpec")

        assert chunks[0].source == "src/lean_spec/state.py"


class TestLeanSpecChunkTypes:
    """Tests for chunk type detection."""

    def test_container_class_is_lean_class(self, tmp_path):
        """Container subclass should be lean_class."""
        py_file = tmp_path / "containers.py"
        py_file.write_text('''class BeaconState(Container):
    """Beacon state container."""
    slot: int
''')

        chunks = chunk_python_spec_file(py_file)

        class_chunks = [c for c in chunks if c.chunk_type == "lean_class"]
        assert len(class_chunks) == 1
        assert "Container" in class_chunks[0].metadata["base_classes"]

    def test_regular_function_is_lean_function(self, tmp_path):
        """Regular function should be lean_function."""
        py_file = tmp_path / "funcs.py"
        py_file.write_text('''def helper_func(x: int) -> int:
    return x * 2
''')

        chunks = chunk_python_spec_file(py_file)

        func_chunks = [c for c in chunks if c.chunk_type == "lean_function"]
        assert len(func_chunks) == 1
        assert func_chunks[0].metadata["function_name"] == "helper_func"

    def test_constant_assignment_is_lean_constant(self, tmp_path):
        """UPPER_CASE assignment should be lean_constant."""
        py_file = tmp_path / "config.py"
        py_file.write_text('''MAX_VALUE = 100
MIN_VALUE: int = 0
regular_var = "not a constant"
''')

        chunks = chunk_python_spec_file(py_file)

        const_chunks = [c for c in chunks if c.chunk_type == "lean_constant"]
        assert len(const_chunks) == 2

        names = [c.metadata["constant_name"] for c in const_chunks]
        assert "MAX_VALUE" in names
        assert "MIN_VALUE" in names
        assert "regular_var" not in names


class TestLeanSpecChunkIds:
    """Tests for chunk ID generation."""

    def test_generates_deterministic_ids(self, tmp_path):
        """Same content should produce same IDs."""
        py_file = tmp_path / "state.py"
        py_file.write_text('CONST = 42')

        chunks1 = chunk_lean_spec_single_file(py_file, base_path=tmp_path)
        chunks2 = chunk_lean_spec_single_file(py_file, base_path=tmp_path)

        assert len(chunks1) > 0
        assert all(c.chunk_id for c in chunks1)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_different_content_different_ids(self, tmp_path):
        """Changed content should produce new IDs."""
        py_file = tmp_path / "state.py"

        # First version
        py_file.write_text('CONST = 42')
        chunks1 = chunk_lean_spec_single_file(py_file, base_path=tmp_path)
        id1 = chunks1[0].chunk_id

        # Modified version
        py_file.write_text('CONST = 99')
        chunks2 = chunk_lean_spec_single_file(py_file, base_path=tmp_path)
        id2 = chunks2[0].chunk_id

        assert id1 != id2

    def test_chunk_ids_follow_format(self, tmp_path):
        """Chunk IDs should follow expected format."""
        py_file = tmp_path / "state.py"
        py_file.write_text('CONST = 42')

        chunks = chunk_lean_spec_single_file(py_file, base_path=tmp_path)

        for chunk in chunks:
            assert chunk.chunk_id.startswith("eth_")
            # Format: eth_{chunk_type}_{path_hash}_{index}_{content_hash}
            # chunk_type can contain underscores (e.g., lean_constant)
            assert "lean_" in chunk.chunk_id
            # Should end with two 8-char hashes separated by underscore-padded index
            parts = chunk.chunk_id.rsplit("_", 2)
            assert len(parts) == 3
            assert len(parts[1]) == 4  # zero-padded index
            assert len(parts[2]) == 8  # content hash


class TestIsConstantName:
    """Tests for _is_constant_name helper."""

    def test_uppercase_is_constant(self):
        assert _is_constant_name("MAX_VALUE") is True
        assert _is_constant_name("SLOTS_PER_EPOCH") is True
        assert _is_constant_name("BASE_REWARD_FACTOR") is True

    def test_lowercase_is_not_constant(self):
        assert _is_constant_name("some_variable") is False
        assert _is_constant_name("helper") is False

    def test_mixed_case_is_not_constant(self):
        assert _is_constant_name("SomeClass") is False
        assert _is_constant_name("camelCase") is False


class TestExtractClassChunks:
    """Tests for _extract_class_chunks helper."""

    def test_extracts_base_classes(self, tmp_path):
        """Should extract base class names."""
        import ast

        content = '''class MyContainer(Container, BaseClass):
    """A container class."""
    field: int
'''
        tree = ast.parse(content)
        class_node = tree.body[0]
        lines = content.splitlines()

        chunks = _extract_class_chunks(class_node, lines, "test.py", "leanSpec")

        class_chunk = chunks[0]
        assert "Container" in class_chunk.metadata["base_classes"]
        assert "BaseClass" in class_chunk.metadata["base_classes"]

    def test_truncates_large_classes(self, tmp_path):
        """Should truncate very large class bodies."""
        import ast

        # Create a large class
        methods = "\n".join([f"    def method_{i}(self): pass" for i in range(100)])
        content = f'''class BigClass:
    """A big class."""
{methods}
'''
        tree = ast.parse(content)
        class_node = tree.body[0]
        lines = content.splitlines()

        chunks = _extract_class_chunks(class_node, lines, "test.py", "leanSpec")

        class_chunk = chunks[0]
        # Should be truncated
        assert len(class_chunk.content) <= 3000


class TestChunkLeanSpecSingleFile:
    """Tests for chunk_lean_spec_single_file wrapper."""

    def test_assigns_chunk_ids(self, tmp_path):
        """Should assign unique chunk IDs."""
        py_file = tmp_path / "test.py"
        py_file.write_text('''"""Module doc."""

CONST = 1

def func():
    pass

class MyClass:
    pass
''')

        chunks = chunk_lean_spec_single_file(py_file, base_path=tmp_path)

        assert all(c.chunk_id for c in chunks)
        # IDs should be unique
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_sets_repo_to_leanspec(self, tmp_path):
        """Should set repo field to leanSpec."""
        py_file = tmp_path / "test.py"
        py_file.write_text('CONST = 1')

        chunks = chunk_lean_spec_single_file(py_file)

        assert all(c.repo == "leanSpec" for c in chunks)
