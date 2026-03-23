"""Compile consensus specs into executable Python modules."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger

logger = get_logger("compiler")


@dataclass(frozen=True)
class CompiledSpec:
    """A compiled spec module with extracted metadata."""

    fork: str
    module_path: Path
    constants: dict[str, str | int]
    functions: list[str]
    types: list[str]


def compile_specs(consensus_dir: Path, output_dir: Path) -> list[CompiledSpec]:
    """
    Compile consensus specs using the spec build process.

    The consensus-specs repo has a build system that compiles
    the markdown + python into executable modules.

    Args:
        consensus_dir: Path to cloned consensus-specs repo
        output_dir: Where to store compiled output

    Returns:
        List of compiled spec metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if pyspec can be built via generate_specs
    generate_specs = consensus_dir / "pysetup" / "generate_specs.py"
    if not generate_specs.exists():
        logger.warning("generate_specs.py not found, using fallback extraction")
        return _extract_specs_fallback(consensus_dir, output_dir)

    # Generate pyspec modules from markdown
    try:
        result = subprocess.run(
            ["python3", "-m", "pysetup.generate_specs", "--all-forks"],
            cwd=consensus_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("pyspec generation failed: %s", result.stderr)
            return _extract_specs_fallback(consensus_dir, output_dir)

        return _extract_from_pyspec(consensus_dir, output_dir)

    except subprocess.TimeoutExpired:
        logger.warning("pyspec generation timed out, using fallback")
        return _extract_specs_fallback(consensus_dir, output_dir)
    except Exception as e:
        logger.warning("pyspec generation error: %s, using fallback", e)
        return _extract_specs_fallback(consensus_dir, output_dir)


def _extract_from_pyspec(consensus_dir: Path, output_dir: Path) -> list[CompiledSpec]:
    """Extract metadata from built pyspec modules."""
    compiled = []

    # Import and inspect pyspec modules
    try:
        import importlib
        import inspect
        import sys

        # Add the pyspec path so eth_consensus_specs is importable
        pyspec_path = str(consensus_dir / "tests" / "core" / "pyspec")
        if pyspec_path not in sys.path:
            sys.path.insert(0, pyspec_path)

        forks = ["phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"]

        for fork in forks:
            try:
                # Try modern eth_consensus_specs path first, fall back to eth2spec
                try:
                    module = importlib.import_module(f"eth_consensus_specs.{fork}.mainnet")
                except ImportError:
                    module = importlib.import_module(f"eth2spec.{fork}.mainnet")

                constants = {}
                functions = []
                types = []

                for name, obj in inspect.getmembers(module):
                    if name.startswith("_"):
                        continue
                    if inspect.isfunction(obj):
                        functions.append(name)
                    elif inspect.isclass(obj):
                        types.append(name)
                    elif name.isupper():  # Constants are uppercase
                        constants[name] = obj

                # Save to JSON
                spec_data = {
                    "fork": fork,
                    "constants": {k: str(v) for k, v in constants.items()},
                    "functions": functions,
                    "types": types,
                }

                output_file = output_dir / f"{fork}_spec.json"
                with open(output_file, "w") as f:
                    json.dump(spec_data, f, indent=2)

                compiled.append(
                    CompiledSpec(
                        fork=fork,
                        module_path=output_file,
                        constants=constants,
                        functions=functions,
                        types=types,
                    )
                )
                logger.info(
                    "Extracted %s: %d constants, %d functions",
                    fork, len(constants), len(functions)
                )

            except ImportError:
                logger.warning("Could not import %s spec", fork)
                continue

    except ImportError:
        logger.warning("eth2spec not available, using fallback")
        return _extract_specs_fallback(consensus_dir, output_dir)

    return compiled


def _extract_specs_fallback(consensus_dir: Path, output_dir: Path) -> list[CompiledSpec]:
    """
    Fallback: extract specs directly from markdown/python files.

    This doesn't give us executable code but extracts constants,
    function signatures, and type definitions.
    """
    import re

    compiled = []
    specs_dir = consensus_dir / "specs"

    # Known forks in order
    forks = ["phase0", "altair", "bellatrix", "capella", "deneb", "electra", "fulu"]

    for fork in forks:
        fork_dir = specs_dir / fork
        if not fork_dir.exists():
            continue

        constants = {}
        functions = []
        types = []

        # Parse all markdown files in fork directory
        for md_file in fork_dir.glob("*.md"):
            content = md_file.read_text()

            # Extract constants from tables (| Name | Value | pattern)
            const_pattern = r"\|\s*`?([A-Z][A-Z0-9_]+)`?\s*\|\s*`?([^|`]+)`?\s*\|"
            for match in re.finditer(const_pattern, content):
                name, value = match.groups()
                constants[name.strip()] = value.strip()

            # Extract function definitions
            func_pattern = r"^def\s+([a-z_][a-z0-9_]*)\s*\("
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                functions.append(match.group(1))

            # Extract class/type definitions
            class_pattern = r"^class\s+([A-Z][a-zA-Z0-9_]*)"
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                types.append(match.group(1))

        if constants or functions or types:
            spec_data = {
                "fork": fork,
                "constants": constants,
                "functions": list(set(functions)),
                "types": list(set(types)),
            }

            output_file = output_dir / f"{fork}_spec.json"
            with open(output_file, "w") as f:
                json.dump(spec_data, f, indent=2)

            compiled.append(
                CompiledSpec(
                    fork=fork,
                    module_path=output_file,
                    constants=constants,
                    functions=list(set(functions)),
                    types=list(set(types)),
                )
            )
            logger.info(
                "Extracted %s (fallback): %d constants, %d functions",
                fork, len(constants), len(functions)
            )

    return compiled


def get_function_source(consensus_dir: Path, fork: str, function_name: str) -> str | None:
    """
    Extract the full source code of a function from spec markdown.

    Args:
        consensus_dir: Path to consensus-specs repo
        fork: Fork name (e.g., 'electra')
        function_name: Name of function to find

    Returns:
        Function source code or None if not found
    """
    import re

    # Prevent path traversal - ensure fork doesn't escape specs directory
    specs_dir = (consensus_dir / "specs" / fork).resolve()
    base_specs = (consensus_dir / "specs").resolve()
    if not str(specs_dir).startswith(str(base_specs)):
        logger.warning("Path traversal attempt blocked: %s", fork)
        return None

    if not specs_dir.exists():
        return None

    for md_file in specs_dir.glob("*.md"):
        content = md_file.read_text()

        # Find function definition and extract full body
        # Functions in specs are in ```python blocks
        # Escape function_name to prevent regex injection (defense in depth)
        safe_name = re.escape(function_name)
        pattern = rf"```python\n(def\s+{safe_name}\s*\([^)]*\)[^`]*?)```"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1).strip()

    return None
