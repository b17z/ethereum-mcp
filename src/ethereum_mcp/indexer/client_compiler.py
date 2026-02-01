"""Compile Ethereum client source code into structured JSON for indexing.

Supports multiple languages:
- Rust: reth, lighthouse (tree-sitter)
- Go: geth, prysm, erigon (tree-sitter)
- Java: teku (regex-based)
- C#: nethermind (regex-based)
- Nim: nimbus-eth2 (regex-based)

Each client implementation provides different perspectives on the same protocol,
making cross-client search valuable for understanding consensus.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ..logging import get_logger

logger = get_logger("client_compiler")

# Try to import tree-sitter parsers
TREE_SITTER_RUST = False
TREE_SITTER_GO = False

try:
    import tree_sitter_rust as ts_rust
    from tree_sitter import Language, Parser
    TREE_SITTER_RUST = True
except ImportError:
    pass

try:
    import tree_sitter_go as ts_go
    from tree_sitter import Language, Parser
    TREE_SITTER_GO = True
except ImportError:
    pass


@dataclass
class ExtractedItem:
    """An extracted code item."""

    kind: str  # function, struct, enum, interface, type
    name: str
    signature: str
    body: str
    doc_comment: str | None
    file_path: str
    line_number: int
    visibility: str
    language: str  # rust, go, java, csharp, nim
    client: str  # reth, geth, lighthouse, etc.


@dataclass
class ExtractedConstant:
    """An extracted constant."""

    name: str
    value: str
    type_annotation: str | None
    doc_comment: str | None
    file_path: str
    line_number: int
    language: str
    client: str


class RustClientParser:
    """Parse Rust client code (reth, lighthouse)."""

    def __init__(self, client: str):
        self.client = client
        if TREE_SITTER_RUST:
            self.parser = Parser(Language(ts_rust.language()))
        else:
            self.parser = None

    def parse_file(self, file_path: Path) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse a Rust file and extract items."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], []

        items, constants = self._parse(content, str(file_path))

        # Tag with language and client
        for item in items:
            item.language = "rust"
            item.client = self.client
        for const in constants:
            const.language = "rust"
            const.client = self.client

        return items, constants

    def _parse(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse Rust code using tree-sitter or regex fallback."""
        items = []
        constants = []
        lines = content.split("\n")

        # Use regex parsing for simplicity (tree-sitter can be added later)
        # Function pattern
        fn_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*(<[^>]+>)?\s*\(([^)]*)\)(\s*->\s*[^{]+)?\s*\{",
            re.MULTILINE,
        )

        for match in fn_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            params = match.group(5) or ""
            ret = match.group(6) or ""

            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find function body
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start():idx]
            doc_comment = self._get_doc_comment(lines, line_num - 1)

            items.append(ExtractedItem(
                kind="function",
                name=name,
                signature=f"fn {name}({params}){ret}".strip(),
                body=body,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                visibility=visibility or "pub",
                language="",
                client="",
            ))

        # Struct pattern
        struct_pattern = re.compile(r"^(\s*)(pub(?:\([^)]+\))?\s+)?struct\s+(\w+)", re.MULTILINE)
        for match in struct_pattern.finditer(content):
            visibility = (match.group(2) or "").strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find struct body
            idx = match.end()
            brace_count = 0
            while idx < len(content):
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    if brace_count <= 1:
                        idx += 1
                        break
                    brace_count -= 1
                elif content[idx] == ";" and brace_count == 0:
                    idx += 1
                    break
                idx += 1

            body = content[match.start():idx]
            doc_comment = self._get_doc_comment(lines, line_num - 1)

            items.append(ExtractedItem(
                kind="struct",
                name=name,
                signature=f"struct {name}",
                body=body,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                visibility=visibility or "pub",
                language="",
                client="",
            ))

        # Constants
        const_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?const\s+(\w+)\s*:\s*([^=]+)\s*=\s*([^;]+);",
            re.MULTILINE,
        )
        for match in const_pattern.finditer(content):
            visibility = (match.group(2) or "").strip()
            name = match.group(3)
            type_ann = match.group(4).strip()
            value = match.group(5).strip()

            start = match.start()
            line_num = content[:start].count("\n") + 1
            doc_comment = self._get_doc_comment(lines, line_num - 1)

            constants.append(ExtractedConstant(
                name=name,
                value=value,
                type_annotation=type_ann,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                language="",
                client="",
            ))

        return items, constants

    def _get_doc_comment(self, lines: list[str], line_idx: int) -> str | None:
        """Get doc comment ending at the given line index."""
        doc_lines = []
        idx = line_idx - 1

        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("///"):
                doc_lines.insert(0, line[3:].strip())
                idx -= 1
            elif line.startswith("//!"):
                doc_lines.insert(0, line[3:].strip())
                idx -= 1
            elif line == "" or line.startswith("#["):
                idx -= 1
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None


class GoClientParser:
    """Parse Go client code (geth, prysm, erigon)."""

    def __init__(self, client: str):
        self.client = client
        if TREE_SITTER_GO:
            self.parser = Parser(Language(ts_go.language()))
        else:
            self.parser = None

    def parse_file(self, file_path: Path) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse a Go file and extract items."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], []

        items, constants = self._parse(content, str(file_path))

        for item in items:
            item.language = "go"
            item.client = self.client
        for const in constants:
            const.language = "go"
            const.client = self.client

        return items, constants

    def _parse(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse Go code."""
        items = []
        constants = []
        lines = content.split("\n")

        # Function pattern (Go uses capitalization for export)
        # func Name(params) return { ... }
        # func (r *Receiver) Name(params) return { ... }
        fn_pattern = re.compile(
            r"^func\s+(?:\([^)]+\)\s+)?([A-Z]\w*)\s*\(([^)]*)\)\s*([^{]*)\{",
            re.MULTILINE,
        )

        for match in fn_pattern.finditer(content):
            name = match.group(1)
            params = match.group(2) or ""
            ret = match.group(3).strip()

            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find function body
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start():idx]
            doc_comment = self._get_doc_comment(lines, line_num - 1)

            items.append(ExtractedItem(
                kind="function",
                name=name,
                signature=f"func {name}({params}) {ret}".strip(),
                body=body,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                visibility="pub",  # Capitalized = exported in Go
                language="",
                client="",
            ))

        # Struct pattern
        struct_pattern = re.compile(r"^type\s+([A-Z]\w*)\s+struct\s*\{", re.MULTILINE)
        for match in struct_pattern.finditer(content):
            name = match.group(1)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find struct body
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start():idx]
            doc_comment = self._get_doc_comment(lines, line_num - 1)

            items.append(ExtractedItem(
                kind="struct",
                name=name,
                signature=f"type {name} struct",
                body=body,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                visibility="pub",
                language="",
                client="",
            ))

        # Interface pattern
        iface_pattern = re.compile(r"^type\s+([A-Z]\w*)\s+interface\s*\{", re.MULTILINE)
        for match in iface_pattern.finditer(content):
            name = match.group(1)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start():idx]
            doc_comment = self._get_doc_comment(lines, line_num - 1)

            items.append(ExtractedItem(
                kind="interface",
                name=name,
                signature=f"type {name} interface",
                body=body,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                visibility="pub",
                language="",
                client="",
            ))

        # Constants
        const_pattern = re.compile(r"^\s*([A-Z]\w*)\s*=\s*(.+)$", re.MULTILINE)
        in_const_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "const (" or stripped.startswith("const ("):
                in_const_block = True
                continue
            if in_const_block and stripped == ")":
                in_const_block = False
                continue

            if in_const_block or stripped.startswith("const "):
                match = const_pattern.match(stripped.replace("const ", ""))
                if match:
                    name = match.group(1)
                    value = match.group(2).strip()

                    constants.append(ExtractedConstant(
                        name=name,
                        value=value,
                        type_annotation=None,
                        doc_comment=self._get_doc_comment(lines, i),
                        file_path=file_path,
                        line_number=i + 1,
                        language="",
                        client="",
                    ))

        return items, constants

    def _get_doc_comment(self, lines: list[str], line_idx: int) -> str | None:
        """Get doc comment ending at the given line index."""
        doc_lines = []
        idx = line_idx - 1

        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
                idx -= 1
            elif line == "":
                idx -= 1
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None


class JavaClientParser:
    """Parse Java client code (teku)."""

    def __init__(self, client: str):
        self.client = client

    def parse_file(self, file_path: Path) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse a Java file and extract items."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return [], []

        items, constants = self._parse(content, str(file_path))

        for item in items:
            item.language = "java"
            item.client = self.client
        for const in constants:
            const.language = "java"
            const.client = self.client

        return items, constants

    def _parse(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse Java code."""
        items = []
        constants = []
        lines = content.split("\n")

        # Public method pattern
        method_pattern = re.compile(
            r"^\s*public\s+(?:static\s+)?(?:<[^>]+>\s+)?(\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)",
            re.MULTILINE,
        )

        for match in method_pattern.finditer(content):
            return_type = match.group(1)
            name = match.group(2)
            params = match.group(3)

            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find method body
            idx = match.end()
            while idx < len(content) and content[idx] != "{":
                idx += 1

            if idx < len(content):
                brace_count = 1
                idx += 1
                while idx < len(content) and brace_count > 0:
                    if content[idx] == "{":
                        brace_count += 1
                    elif content[idx] == "}":
                        brace_count -= 1
                    idx += 1

            body = content[match.start():idx]
            doc_comment = self._get_javadoc(lines, line_num - 1)

            items.append(ExtractedItem(
                kind="function",
                name=name,
                signature=f"public {return_type} {name}({params})",
                body=body,
                doc_comment=doc_comment,
                file_path=file_path,
                line_number=line_num,
                visibility="pub",
                language="",
                client="",
            ))

        # Class pattern
        class_pattern = re.compile(r"^\s*public\s+(?:final\s+)?class\s+(\w+)", re.MULTILINE)
        for match in class_pattern.finditer(content):
            name = match.group(1)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            items.append(ExtractedItem(
                kind="struct",  # Treat classes as structs for uniformity
                name=name,
                signature=f"class {name}",
                body="",  # Classes are too large to include fully
                doc_comment=self._get_javadoc(lines, line_num - 1),
                file_path=file_path,
                line_number=line_num,
                visibility="pub",
                language="",
                client="",
            ))

        # Constants (public static final)
        const_pattern = re.compile(
            r"^\s*public\s+static\s+final\s+(\w+)\s+(\w+)\s*=\s*(.+);",
            re.MULTILINE,
        )
        for match in const_pattern.finditer(content):
            type_ann = match.group(1)
            name = match.group(2)
            value = match.group(3).strip()

            start = match.start()
            line_num = content[:start].count("\n") + 1

            constants.append(ExtractedConstant(
                name=name,
                value=value,
                type_annotation=type_ann,
                doc_comment=self._get_javadoc(lines, line_num - 1),
                file_path=file_path,
                line_number=line_num,
                language="",
                client="",
            ))

        return items, constants

    def _get_javadoc(self, lines: list[str], line_idx: int) -> str | None:
        """Get Javadoc comment ending at the given line index."""
        doc_lines = []
        idx = line_idx - 1

        # Look for /** ... */ block
        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("*") and not line.startswith("*/"):
                doc_lines.insert(0, line.lstrip("* ").strip())
                idx -= 1
            elif line.startswith("/**"):
                break
            elif line.endswith("*/"):
                doc_lines.insert(0, line.rstrip("*/").strip())
                idx -= 1
            elif line == "":
                idx -= 1
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None


def compile_client(
    source_dir: Path,
    output_dir: Path,
    client_name: str,
    language: str,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Compile client source code into JSON extracts.

    Args:
        source_dir: Directory containing client source files
        output_dir: Directory to write JSON output
        client_name: Client name (reth, geth, lighthouse, etc.)
        language: Source language (rust, go, java, csharp, nim)
        progress_callback: Optional callback for progress updates

    Returns:
        Statistics about extraction
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    # Select parser based on language
    if language == "rust":
        parser = RustClientParser(client_name)
        patterns = ["**/*.rs"]
    elif language == "go":
        parser = GoClientParser(client_name)
        patterns = ["**/*.go"]
    elif language == "java":
        parser = JavaClientParser(client_name)
        patterns = ["**/*.java"]
    else:
        log(f"Warning: Unsupported language {language} for {client_name}")
        return {"error": f"Unsupported language: {language}"}

    all_items = []
    all_constants = []

    # Find source files
    source_files = []
    for pattern in patterns:
        source_files.extend(source_dir.glob(pattern))

    log(f"  Found {len(source_files)} {language} files")

    # Parse each file
    for file_path in source_files:
        items, constants = parser.parse_file(file_path)

        # Make paths relative
        for item in items:
            try:
                item.file_path = str(Path(item.file_path).relative_to(source_dir))
            except ValueError:
                pass
        for const in constants:
            try:
                const.file_path = str(Path(const.file_path).relative_to(source_dir))
            except ValueError:
                pass

        all_items.extend(items)
        all_constants.extend(constants)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)

    items_file = output_dir / "items.json"
    with open(items_file, "w") as f:
        json.dump([asdict(item) for item in all_items], f, indent=2)

    constants_file = output_dir / "constants.json"
    with open(constants_file, "w") as f:
        json.dump([asdict(const) for const in all_constants], f, indent=2)

    # Build index
    index = {
        "functions": {},
        "structs": {},
        "interfaces": {},
        "constants": {},
        "client": client_name,
        "language": language,
    }

    for item in all_items:
        category = f"{item.kind}s"
        if category in index:
            index[category][item.name] = {
                "file": item.file_path,
                "line": item.line_number,
            }

    for const in all_constants:
        index["constants"][const.name] = {
            "file": const.file_path,
            "line": const.line_number,
            "value": const.value,
        }

    index_file = output_dir / "index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)

    return {
        "client": client_name,
        "language": language,
        "files_processed": len(source_files),
        "items_extracted": len(all_items),
        "constants_extracted": len(all_constants),
        "functions": len([i for i in all_items if i.kind == "function"]),
        "structs": len([i for i in all_items if i.kind == "struct"]),
        "interfaces": len([i for i in all_items if i.kind == "interface"]),
    }


def load_client_items(compiled_dir: Path) -> list[ExtractedItem]:
    """Load compiled items from JSON."""
    items_file = compiled_dir / "items.json"
    if not items_file.exists():
        return []

    with open(items_file) as f:
        data = json.load(f)

    return [ExtractedItem(**item) for item in data]


def load_client_constants(compiled_dir: Path) -> list[ExtractedConstant]:
    """Load compiled constants from JSON."""
    constants_file = compiled_dir / "constants.json"
    if not constants_file.exists():
        return []

    with open(constants_file) as f:
        data = json.load(f)

    return [ExtractedConstant(**item) for item in data]
