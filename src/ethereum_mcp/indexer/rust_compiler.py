"""Compile/extract Rust source code into structured data for indexing.

Parses Rust source files to extract:
- pub fn (public functions)
- pub struct (public structs)
- pub enum (public enums)
- const (constants)
- impl blocks

Uses tree-sitter for robust parsing when available, with regex fallback.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# Try to import tree-sitter parser
TREE_SITTER_RUST = False

try:
    import tree_sitter_rust as ts_rust
    from tree_sitter import Language, Parser

    TREE_SITTER_RUST = True
except ImportError:
    pass


@dataclass
class ExtractedItem:
    """An extracted code item."""

    kind: str  # function, struct, enum, const, impl, type
    name: str
    signature: str  # For functions: full signature; for types: definition line
    body: str  # Full source code
    doc_comment: str | None
    file_path: str
    line_number: int
    visibility: str  # pub, pub(crate), private
    attributes: list[str]  # #[derive(...)] etc.


@dataclass
class ExtractedConstant:
    """An extracted constant."""

    name: str
    value: str
    type_annotation: str | None
    doc_comment: str | None
    file_path: str
    line_number: int


class RustParser:
    """Parse Rust source code to extract definitions."""

    def __init__(self) -> None:
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

        if self.parser:
            return self._parse_with_tree_sitter(content, str(file_path))
        else:
            return self._parse_with_regex(content, str(file_path))

    def _parse_with_tree_sitter(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Parse using tree-sitter for accurate AST."""
        tree = self.parser.parse(bytes(content, "utf-8"))
        items = []
        constants = []

        lines = content.split("\n")

        def get_text(node) -> str:
            return content[node.start_byte : node.end_byte]

        def get_doc_comment(node) -> str | None:
            """Get doc comment preceding a node."""
            # Look for comment nodes before this node
            doc_lines = []
            line = node.start_point[0] - 1

            while line >= 0:
                line_text = lines[line].strip()
                if line_text.startswith("///") or line_text.startswith("//!"):
                    doc_lines.insert(0, line_text[3:].strip())
                    line -= 1
                elif line_text == "" or line_text.startswith("#["):
                    line -= 1
                else:
                    break

            return "\n".join(doc_lines) if doc_lines else None

        def get_attributes(node) -> list[str]:
            """Get attributes preceding a node."""
            attrs = []
            line = node.start_point[0] - 1

            while line >= 0:
                line_text = lines[line].strip()
                if line_text.startswith("#["):
                    attrs.insert(0, line_text)
                    line -= 1
                elif line_text.startswith("///") or line_text.startswith("//!") or line_text == "":
                    line -= 1
                else:
                    break

            return attrs

        def get_visibility(node) -> str:
            """Determine visibility of a node."""
            for child in node.children:
                if child.type == "visibility_modifier":
                    vis_text = get_text(child)
                    if vis_text == "pub":
                        return "pub"
                    elif "crate" in vis_text:
                        return "pub(crate)"
                    else:
                        return vis_text
            return "private"

        def process_node(node):
            if node.type == "function_item":
                visibility = get_visibility(node)
                if visibility.startswith("pub"):
                    name = None
                    signature_parts = []

                    for child in node.children:
                        if child.type == "identifier":
                            name = get_text(child)
                        elif child.type == "parameters":
                            signature_parts.append(get_text(child))
                        elif child.type == "return_type":
                            signature_parts.append(f"-> {get_text(child)}")

                    if name:
                        # Build signature
                        params = signature_parts[0] if signature_parts else "()"
                        ret = signature_parts[1] if len(signature_parts) > 1 else ""
                        signature = f"fn {name}{params} {ret}".strip()

                        items.append(
                            ExtractedItem(
                                kind="function",
                                name=name,
                                signature=signature,
                                body=get_text(node),
                                doc_comment=get_doc_comment(node),
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                visibility=visibility,
                                attributes=get_attributes(node),
                            )
                        )

            elif node.type == "struct_item":
                visibility = get_visibility(node)
                if visibility.startswith("pub"):
                    name = None
                    for child in node.children:
                        if child.type == "type_identifier":
                            name = get_text(child)
                            break

                    if name:
                        items.append(
                            ExtractedItem(
                                kind="struct",
                                name=name,
                                signature=f"struct {name}",
                                body=get_text(node),
                                doc_comment=get_doc_comment(node),
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                visibility=visibility,
                                attributes=get_attributes(node),
                            )
                        )

            elif node.type == "enum_item":
                visibility = get_visibility(node)
                if visibility.startswith("pub"):
                    name = None
                    for child in node.children:
                        if child.type == "type_identifier":
                            name = get_text(child)
                            break

                    if name:
                        items.append(
                            ExtractedItem(
                                kind="enum",
                                name=name,
                                signature=f"enum {name}",
                                body=get_text(node),
                                doc_comment=get_doc_comment(node),
                                file_path=file_path,
                                line_number=node.start_point[0] + 1,
                                visibility=visibility,
                                attributes=get_attributes(node),
                            )
                        )

            elif node.type == "const_item":
                visibility = get_visibility(node)
                name = None
                type_ann = None
                value = None

                for child in node.children:
                    if child.type == "identifier":
                        name = get_text(child)
                    elif child.type == "type_identifier" or child.type.endswith("_type"):
                        type_ann = get_text(child)

                # Extract value from the full text
                full_text = get_text(node)
                if "=" in full_text:
                    value = full_text.split("=", 1)[1].strip().rstrip(";")

                if name:
                    constants.append(
                        ExtractedConstant(
                            name=name,
                            value=value or "",
                            type_annotation=type_ann,
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                        )
                    )

            elif node.type == "impl_item":
                # Extract impl blocks for types
                type_name = None
                for child in node.children:
                    if child.type == "type_identifier" or child.type == "generic_type":
                        type_name = get_text(child)
                        break

                if type_name:
                    items.append(
                        ExtractedItem(
                            kind="impl",
                            name=f"impl {type_name}",
                            signature=f"impl {type_name}",
                            body=get_text(node),
                            doc_comment=get_doc_comment(node),
                            file_path=file_path,
                            line_number=node.start_point[0] + 1,
                            visibility="pub",  # impl blocks are effectively pub if type is
                            attributes=get_attributes(node),
                        )
                    )

            # Recurse into children
            for child in node.children:
                process_node(child)

        process_node(tree.root_node)
        return items, constants

    def _parse_with_regex(
        self, content: str, file_path: str
    ) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
        """Fallback regex-based parsing when tree-sitter isn't available."""
        items = []
        constants = []
        lines = content.split("\n")

        # Patterns for extraction
        fn_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?fn\s+(\w+)\s*(<[^>]+>)?\s*\(([^)]*)\)(\s*->\s*[^{]+)?\s*\{",
            re.MULTILINE,
        )
        struct_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?struct\s+(\w+)", re.MULTILINE
        )
        enum_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?enum\s+(\w+)", re.MULTILINE
        )
        const_pattern = re.compile(
            r"^(\s*)(pub(?:\([^)]+\))?\s+)?const\s+(\w+)\s*:\s*([^=]+)\s*=\s*([^;]+);",
            re.MULTILINE,
        )

        # Extract functions
        for match in fn_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            params = match.group(5)
            ret = match.group(6) or ""

            # Find the full function body
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Simple brace matching to find end
            brace_count = 1
            idx = match.end()
            while idx < len(content) and brace_count > 0:
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    brace_count -= 1
                idx += 1

            body = content[match.start() : idx]

            # Get doc comment
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="function",
                    name=name,
                    signature=f"fn {name}({params}){ret}".strip(),
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility=visibility if visibility else "pub",
                    attributes=[],
                )
            )

        # Extract structs
        for match in struct_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find end of struct (either ; or })
            idx = match.end()
            brace_count = 0
            while idx < len(content):
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    if brace_count == 0:
                        idx += 1
                        break
                    brace_count -= 1
                elif content[idx] == ";" and brace_count == 0:
                    idx += 1
                    break
                idx += 1

            body = content[match.start() : idx]
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="struct",
                    name=name,
                    signature=f"struct {name}",
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility=visibility if visibility else "pub",
                    attributes=[],
                )
            )

        # Extract enums
        for match in enum_pattern.finditer(content):
            visibility = match.group(2) or ""
            visibility = visibility.strip()
            if not visibility.startswith("pub"):
                continue

            name = match.group(3)
            start = match.start()
            line_num = content[:start].count("\n") + 1

            # Find end of enum
            idx = match.end()
            brace_count = 0
            while idx < len(content):
                if content[idx] == "{":
                    brace_count += 1
                elif content[idx] == "}":
                    if brace_count == 1:
                        idx += 1
                        break
                    brace_count -= 1
                idx += 1

            body = content[match.start() : idx]
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            items.append(
                ExtractedItem(
                    kind="enum",
                    name=name,
                    signature=f"enum {name}",
                    body=body,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                    visibility=visibility if visibility else "pub",
                    attributes=[],
                )
            )

        # Extract constants
        for match in const_pattern.finditer(content):
            visibility = match.group(2) or ""
            if not visibility.strip().startswith("pub"):
                continue

            name = match.group(3)
            type_ann = match.group(4).strip()
            value = match.group(5).strip()

            start = match.start()
            line_num = content[:start].count("\n") + 1
            doc_comment = self._get_doc_comment_at_line(lines, line_num - 1)

            constants.append(
                ExtractedConstant(
                    name=name,
                    value=value,
                    type_annotation=type_ann,
                    doc_comment=doc_comment,
                    file_path=file_path,
                    line_number=line_num,
                )
            )

        return items, constants

    def _get_doc_comment_at_line(self, lines: list[str], line_idx: int) -> str | None:
        """Get doc comment ending at the given line index."""
        doc_lines = []
        idx = line_idx - 1

        while idx >= 0:
            line = lines[idx].strip()
            if line.startswith("///") or line.startswith("//!"):
                doc_lines.insert(0, line[3:].strip())
                idx -= 1
            elif line == "" or line.startswith("#["):
                idx -= 1
            else:
                break

        return "\n".join(doc_lines) if doc_lines else None


def parse_rust_file(file_path: Path) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
    """
    Parse a single Rust file and extract items.

    Args:
        file_path: Path to the Rust source file

    Returns:
        Tuple of (items, constants) extracted from the file
    """
    parser = RustParser()
    return parser.parse_file(file_path)


def parse_rust_repo(
    repo_dir: Path,
    file_patterns: list[str] | None = None,
) -> tuple[list[ExtractedItem], list[ExtractedConstant]]:
    """
    Parse all Rust files in a repository.

    Args:
        repo_dir: Root directory of the Rust repository
        file_patterns: Glob patterns for files to include (default: ["**/*.rs"])

    Returns:
        Tuple of (items, constants) extracted from all files
    """
    if file_patterns is None:
        file_patterns = ["**/*.rs"]

    parser = RustParser()
    all_items = []
    all_constants = []

    # Find all Rust files
    rust_files = []
    for pattern in file_patterns:
        rust_files.extend(repo_dir.glob(pattern))

    # Parse each file
    for file_path in rust_files:
        # Skip test files and build artifacts
        if "/target/" in str(file_path) or "test" in file_path.name.lower():
            continue

        items, constants = parser.parse_file(file_path)

        # Make paths relative to repo_dir
        for item in items:
            item.file_path = str(Path(item.file_path).relative_to(repo_dir))
        for const in constants:
            const.file_path = str(Path(const.file_path).relative_to(repo_dir))

        all_items.extend(items)
        all_constants.extend(constants)

    return all_items, all_constants


def lookup_rust_function(
    name: str,
    items: list[ExtractedItem],
) -> ExtractedItem | None:
    """
    Look up a function by name from extracted items.

    Args:
        name: Function name to find
        items: List of extracted items to search

    Returns:
        The ExtractedItem if found, None otherwise
    """
    for item in items:
        if item.kind == "function" and item.name == name:
            return item
    return None


def lookup_rust_constant(
    name: str,
    constants: list[ExtractedConstant],
) -> ExtractedConstant | None:
    """
    Look up a constant by name.

    Args:
        name: Constant name to find
        constants: List of extracted constants to search

    Returns:
        The ExtractedConstant if found, None otherwise
    """
    for const in constants:
        if const.name == name:
            return const
    return None
