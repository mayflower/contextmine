"""SCIP (Sourcegraph Code Intelligence Protocol) semantic snapshot provider.

SCIP provides fully resolved semantic information from language-specific
indexers. It provides accurate cross-file references, symbol definitions,
and relationships.

See: https://sourcegraph.com/docs/code-navigation/code-intelligence-protocol
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

from contextmine_core.semantic_snapshot.models import (
    FileInfo,
    Occurrence,
    OccurrenceRole,
    Range,
    Relation,
    RelationKind,
    Snapshot,
    Symbol,
    SymbolKind,
)

if TYPE_CHECKING:
    from contextmine_core.semantic_snapshot.proto import scip_pb2

logger = logging.getLogger(__name__)

# Map SCIP SymbolInformation.Kind to our SymbolKind
SCIP_KIND_TO_SYMBOL_KIND: dict[int, SymbolKind] = {
    # From scip.proto SymbolInformation.Kind enum
    7: SymbolKind.CLASS,  # Class
    17: SymbolKind.FUNCTION,  # Function
    26: SymbolKind.METHOD,  # Method
    49: SymbolKind.STRUCT,  # Struct
    21: SymbolKind.INTERFACE,  # Interface
    11: SymbolKind.ENUM,  # Enum
    53: SymbolKind.TRAIT,  # Trait
    54: SymbolKind.TYPE_ALIAS,  # Type
    55: SymbolKind.TYPE_ALIAS,  # TypeAlias
    61: SymbolKind.VARIABLE,  # Variable
    8: SymbolKind.CONSTANT,  # Constant
    29: SymbolKind.MODULE,  # Module
    30: SymbolKind.MODULE,  # Namespace
    35: SymbolKind.MODULE,  # Package
    41: SymbolKind.PROPERTY,  # Property
    15: SymbolKind.PROPERTY,  # Field
    37: SymbolKind.PARAMETER,  # Parameter
    9: SymbolKind.FUNCTION,  # Constructor
    66: SymbolKind.METHOD,  # AbstractMethod
    80: SymbolKind.METHOD,  # StaticMethod
    12: SymbolKind.CONSTANT,  # EnumMember
}

# SCIP SymbolRole bit flags
SCIP_ROLE_DEFINITION = 0x1
SCIP_ROLE_IMPORT = 0x2
SCIP_ROLE_WRITE_ACCESS = 0x4
SCIP_ROLE_READ_ACCESS = 0x8


class SCIPProvider:
    """Semantic snapshot provider using SCIP indexes.

    SCIP indexes are produced by language-specific indexers and provide
    fully resolved semantic information including cross-file references.

    This provider parses .scip files (protobuf format) and converts them
    to the Snapshot model.
    """

    def __init__(self, scip_path: Path | str) -> None:
        """Initialize the provider.

        Args:
            scip_path: Path to the .scip index file
        """
        self._scip_path = Path(scip_path)

    def is_available(self) -> bool:
        """Check if the SCIP file exists."""
        return self._scip_path.exists()

    def extract(self) -> Snapshot:
        """Extract semantic information from the SCIP index.

        Returns:
            Snapshot with fully resolved symbols and references
        """
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        index = scip_pb2.Index()
        with open(self._scip_path, "rb") as f:
            index.ParseFromString(f.read())

        return self._convert_index(index)

    def _convert_index(self, index: scip_pb2.Index) -> Snapshot:
        """Convert SCIP Index to Snapshot model.

        Args:
            index: Parsed SCIP Index protobuf

        Returns:
            Snapshot with converted data
        """
        files: list[FileInfo] = []
        symbols: list[Symbol] = []
        occurrences: list[Occurrence] = []
        relations: list[Relation] = []

        # Track symbols we've seen for deduplication
        seen_symbols: set[str] = set()

        # Process metadata
        tool_name = index.metadata.tool_info.name if index.metadata.tool_info else "unknown"
        tool_version = index.metadata.tool_info.version if index.metadata.tool_info else "unknown"

        # Process each document
        for doc in index.documents:
            file_path = doc.relative_path
            language = self._get_language_string(doc.language)

            files.append(FileInfo(path=file_path, language=language))

            # Process symbol definitions in this document
            for sym_info in doc.symbols:
                symbol_str = sym_info.symbol
                if symbol_str.startswith("local "):
                    continue
                if symbol_str in seen_symbols:
                    continue
                seen_symbols.add(symbol_str)

                # Parse symbol kind
                kind = SCIP_KIND_TO_SYMBOL_KIND.get(sym_info.kind, SymbolKind.UNKNOWN)
                inferred_kind, inferred_name = self._infer_kind_and_name_from_symbol(symbol_str)
                if kind == SymbolKind.UNKNOWN:
                    kind = inferred_kind

                # Get display name
                name = sym_info.display_name or inferred_name or self._extract_name_from_symbol(
                    symbol_str
                )

                # Keep only relevant symbols with a resolvable kind and name.
                if kind == SymbolKind.UNKNOWN or not name:
                    continue
                if kind == SymbolKind.PARAMETER:
                    continue

                # Find the definition occurrence for this symbol to get range
                def_range = self._find_definition_range(doc, symbol_str)

                if def_range:
                    symbol = Symbol(
                        def_id=symbol_str,
                        kind=kind,
                        file_path=file_path,
                        range=def_range,
                        name=name,
                        container_def_id=sym_info.enclosing_symbol or None,
                    )
                    symbols.append(symbol)

                # Process relationships
                for rel in sym_info.relationships:
                    if rel.is_implementation:
                        relations.append(
                            Relation(
                                src_def_id=symbol_str,
                                kind=RelationKind.IMPLEMENTS,
                                dst_def_id=rel.symbol,
                                resolved=True,
                            )
                        )
                    if rel.is_reference:
                        relations.append(
                            Relation(
                                src_def_id=symbol_str,
                                kind=RelationKind.REFERENCES,
                                dst_def_id=rel.symbol,
                                resolved=True,
                            )
                        )
                    if rel.is_type_definition:
                        relations.append(
                            Relation(
                                src_def_id=symbol_str,
                                kind=RelationKind.EXTENDS,
                                dst_def_id=rel.symbol,
                                resolved=True,
                            )
                        )

            # Process all occurrences in this document
            for occ in doc.occurrences:
                if not occ.symbol:
                    continue

                # Parse range
                occ_range = self._parse_range(occ.range)
                if not occ_range:
                    continue

                # Determine role from symbol_roles bitmask
                is_definition = bool(occ.symbol_roles & SCIP_ROLE_DEFINITION)
                role = OccurrenceRole.DEFINITION if is_definition else OccurrenceRole.REFERENCE

                occurrences.append(
                    Occurrence(
                        file_path=file_path,
                        range=occ_range,
                        role=role,
                        def_id=occ.symbol,
                    )
                )

        # Process external symbols (symbols defined in external packages)
        for ext_sym in index.external_symbols:
            symbol_str = ext_sym.symbol
            if symbol_str.startswith("local "):
                continue
            if symbol_str in seen_symbols:
                continue
            seen_symbols.add(symbol_str)

            kind = SCIP_KIND_TO_SYMBOL_KIND.get(ext_sym.kind, SymbolKind.UNKNOWN)
            inferred_kind, inferred_name = self._infer_kind_and_name_from_symbol(symbol_str)
            if kind == SymbolKind.UNKNOWN:
                kind = inferred_kind
            name = ext_sym.display_name or inferred_name or self._extract_name_from_symbol(symbol_str)

            if kind == SymbolKind.UNKNOWN or not name:
                continue
            if kind == SymbolKind.PARAMETER:
                continue

            # External symbols don't have a file path or range in this index
            # We still track them for reference resolution
            symbols.append(
                Symbol(
                    def_id=symbol_str,
                    kind=kind,
                    file_path="<external>",
                    range=Range(start_line=0, start_col=0, end_line=0, end_col=0),
                    name=name,
                    container_def_id=ext_sym.enclosing_symbol or None,
                )
            )

        return Snapshot(
            files=files,
            symbols=symbols,
            occurrences=occurrences,
            relations=relations,
            meta={
                "provider": "scip",
                "tool_name": tool_name,
                "tool_version": tool_version,
                "project_root": index.metadata.project_root,
            },
        )

    def _find_definition_range(self, doc: scip_pb2.Document, symbol_str: str) -> Range | None:
        """Find the definition occurrence range for a symbol in a document."""
        for occ in doc.occurrences:
            if occ.symbol == symbol_str and (occ.symbol_roles & SCIP_ROLE_DEFINITION):
                return self._parse_range(occ.range)
        return None

    def _parse_range(self, range_list: list[int]) -> Range | None:
        """Parse SCIP range format to Range model.

        SCIP ranges are:
        - 4 elements: [startLine, startChar, endLine, endChar]
        - 3 elements: [startLine, startChar, endChar] (same line)

        SCIP uses 0-based line numbers, we use 1-based.
        """
        if len(range_list) == 4:
            return Range(
                start_line=range_list[0] + 1,  # Convert to 1-based
                start_col=range_list[1],
                end_line=range_list[2] + 1,  # Convert to 1-based
                end_col=range_list[3],
            )
        elif len(range_list) == 3:
            return Range(
                start_line=range_list[0] + 1,  # Convert to 1-based
                start_col=range_list[1],
                end_line=range_list[0] + 1,  # Same line
                end_col=range_list[2],
            )
        return None

    def _get_language_string(self, language: str) -> str | None:
        """Convert SCIP language string to lowercase."""
        if not language:
            return None
        return language.lower()

    def _extract_name_from_symbol(self, symbol_str: str) -> str | None:
        """Extract a human-readable name from SCIP symbol string.

        SCIP symbol format examples:
        - scip-python python mypackage 0.1.0 mymodule/MyClass#method().
        - local 42
        """
        if not symbol_str:
            return None

        # Handle local symbols
        if symbol_str.startswith("local "):
            return f"local_{symbol_str[6:]}"

        # Extract the last descriptor from the symbol
        # Descriptors end with: / # . : ! (). [] ()
        parts = symbol_str.split()
        if len(parts) < 4:
            return None

        # The descriptor part is everything after scheme, manager, package, version
        descriptors = " ".join(parts[4:]) if len(parts) > 4 else ""

        # Match name followed by descriptor suffix.
        matches = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)[/#.\[\]():!]", descriptors)
        if matches:
            return matches[-1]

        return None

    def _infer_kind_and_name_from_symbol(self, symbol_str: str) -> tuple[SymbolKind, str | None]:
        """Infer kind and human-readable name from SCIP symbol descriptors.

        This is primarily used when scip-python emits UnspecifiedKind and no
        display_name for document symbols.
        """
        if not symbol_str or symbol_str.startswith("local "):
            return SymbolKind.UNKNOWN, None

        descriptor_tail = self._descriptor_tail(symbol_str)
        if not descriptor_tail:
            return SymbolKind.UNKNOWN, None

        # Parameter descriptor: "...().(param)"
        parameter_match = re.search(r"\(([^()]+)\)$", descriptor_tail)
        if parameter_match:
            return SymbolKind.PARAMETER, parameter_match.group(1)

        # Method/function descriptor: "...name()."
        if descriptor_tail.endswith("()."):
            name = self._last_identifier(descriptor_tail.removesuffix("()."))
            if "#" in descriptor_tail:
                return SymbolKind.METHOD, name
            return SymbolKind.FUNCTION, name

        # Type/class descriptor: "...Type#"
        if descriptor_tail.endswith("#"):
            return SymbolKind.CLASS, self._last_identifier(descriptor_tail.removesuffix("#"))

        # Term/property descriptor: "...field."
        if descriptor_tail.endswith("."):
            name = self._last_identifier(descriptor_tail.removesuffix("."))
            if "#" in descriptor_tail:
                return SymbolKind.PROPERTY, name
            return SymbolKind.FUNCTION, name

        # Namespace/module descriptors.
        if descriptor_tail.endswith("/") or descriptor_tail.endswith(":"):
            return SymbolKind.MODULE, self._last_identifier(descriptor_tail[:-1])

        # Type parameter descriptor: "...[T]"
        if descriptor_tail.endswith("]"):
            m = re.search(r"\[([^\]]+)\]$", descriptor_tail)
            return SymbolKind.TYPE_ALIAS, m.group(1) if m else None

        # Macro/meta descriptors.
        if descriptor_tail.endswith("!"):
            return SymbolKind.FUNCTION, self._last_identifier(descriptor_tail.removesuffix("!"))

        return SymbolKind.UNKNOWN, self._last_identifier(descriptor_tail)

    def _descriptor_tail(self, symbol_str: str) -> str:
        """Return the descriptor section of a SCIP symbol string."""
        parts = symbol_str.split(maxsplit=4)
        if len(parts) < 5:
            return ""
        return parts[4]

    def _last_identifier(self, raw: str) -> str | None:
        """Extract the trailing identifier token from descriptor text."""
        # Strip backticks used by scip-python module names.
        cleaned = raw.replace("`", "")
        matches = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", cleaned)
        if not matches:
            return None
        return matches[-1]


def build_snapshot_scip(scip_path: Path | str) -> Snapshot:
    """Build a semantic snapshot from a SCIP index file.

    Args:
        scip_path: Path to the .scip index file

    Returns:
        Snapshot with fully resolved semantic information

    Raises:
        FileNotFoundError: If SCIP file doesn't exist
    """
    provider = SCIPProvider(scip_path)

    if not provider.is_available():
        raise FileNotFoundError(f"SCIP index not found: {scip_path}")

    return provider.extract()
