"""LSIF semantic snapshot provider.

Parses LSIF JSON lines and maps it into the unified Snapshot model with
file/symbol/occurrence/relation entities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)


class LSIFProvider:
    """Semantic snapshot provider for LSIF JSONL indexes."""

    def __init__(self, lsif_path: Path | str) -> None:
        self._lsif_path = Path(lsif_path)

    def is_available(self) -> bool:
        return self._lsif_path.exists()

    def extract(self) -> Snapshot:
        vertices: dict[int, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        with self._lsif_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "vertex":
                    vertices[int(obj["id"])] = obj
                elif obj.get("type") == "edge":
                    edges.append(obj)

        return _convert(vertices, edges)


def _convert(vertices: dict[int, dict[str, Any]], edges: list[dict[str, Any]]) -> Snapshot:
    documents = {vid: v for vid, v in vertices.items() if v.get("label") == "document"}
    ranges = {vid: v for vid, v in vertices.items() if v.get("label") == "range"}
    monikers = {vid: v for vid, v in vertices.items() if v.get("label") == "moniker"}

    files: list[FileInfo] = []
    symbols: dict[str, Symbol] = {}
    occurrences: list[Occurrence] = []
    relations: list[Relation] = []

    doc_uri_to_path: dict[int, str] = {}
    for doc_id, doc in documents.items():
        path = _normalize_uri(doc.get("uri", ""))
        doc_uri_to_path[doc_id] = path
        files.append(FileInfo(path=path, language=None))

    contains_edges = [e for e in edges if e.get("label") == "contains"]
    next_edges = [e for e in edges if e.get("label") == "next"]
    moniker_edges = [e for e in edges if e.get("label") == "moniker"]
    definition_edges = [e for e in edges if e.get("label") == "textDocument/definition"]
    reference_edges = [e for e in edges if e.get("label") == "textDocument/references"]
    item_edges = [e for e in edges if e.get("label") == "item"]

    range_to_doc: dict[int, int] = {}
    for edge in contains_edges:
        out_v = int(edge["outV"])
        in_vs = [int(v) for v in edge.get("inVs", [])]
        if out_v in documents:
            for in_v in in_vs:
                if in_v in ranges:
                    range_to_doc[in_v] = out_v

    range_to_result: dict[int, int] = {}
    for edge in next_edges:
        out_v = int(edge["outV"])
        in_v = int(edge["inV"])
        if out_v in ranges:
            range_to_result[out_v] = in_v

    result_to_moniker: dict[int, str] = {}
    for edge in moniker_edges:
        out_v = int(edge["outV"])
        in_v = int(edge["inV"])
        moniker = monikers.get(in_v)
        if moniker:
            identifier = str(moniker.get("identifier") or "")
            if identifier:
                result_to_moniker[out_v] = identifier

    definition_result_to_ranges: dict[int, list[int]] = {}
    reference_result_to_ranges: dict[int, list[int]] = {}
    for edge in item_edges:
        out_v = int(edge["outV"])
        in_vs = [int(v) for v in edge.get("inVs", [])]
        prop = str(edge.get("property", "references"))
        target = (
            definition_result_to_ranges if prop == "definitions" else reference_result_to_ranges
        )
        target.setdefault(out_v, []).extend([v for v in in_vs if v in ranges])

    range_to_def_result: dict[int, int] = {}
    for edge in definition_edges:
        out_v = int(edge["outV"])
        in_v = int(edge["inV"])
        if out_v in ranges:
            range_to_def_result[out_v] = in_v

    range_to_ref_result: dict[int, int] = {}
    for edge in reference_edges:
        out_v = int(edge["outV"])
        in_v = int(edge["inV"])
        if out_v in ranges:
            range_to_ref_result[out_v] = in_v

    # Build symbols from definition ranges.
    for _source_range_id, def_result_id in range_to_def_result.items():
        def_ranges = definition_result_to_ranges.get(def_result_id, [])
        if not def_ranges:
            continue
        canonical_range_id = def_ranges[0]
        symbol_id = _symbol_id(
            canonical_range_id,
            range_to_doc,
            doc_uri_to_path,
            ranges,
            result_to_moniker,
            range_to_result,
        )
        if not symbol_id:
            continue
        if symbol_id in symbols:
            continue

        rng = _range_obj(ranges[canonical_range_id])
        file_path = _file_of(canonical_range_id, range_to_doc, doc_uri_to_path)
        symbols[symbol_id] = Symbol(
            def_id=symbol_id,
            kind=SymbolKind.UNKNOWN,
            file_path=file_path,
            range=rng,
            name=_symbol_name(symbol_id),
            container_def_id=None,
        )

    # Occurrences and relationships.
    for source_range_id, def_result_id in range_to_def_result.items():
        def_ranges = definition_result_to_ranges.get(def_result_id, [])
        if not def_ranges:
            continue
        symbol_id = _symbol_id(
            def_ranges[0], range_to_doc, doc_uri_to_path, ranges, result_to_moniker, range_to_result
        )
        if not symbol_id:
            continue

        src_occ = Occurrence(
            file_path=_file_of(source_range_id, range_to_doc, doc_uri_to_path),
            range=_range_obj(ranges[source_range_id]),
            role=OccurrenceRole.REFERENCE,
            def_id=symbol_id,
        )
        occurrences.append(src_occ)

        for def_range in def_ranges:
            occurrences.append(
                Occurrence(
                    file_path=_file_of(def_range, range_to_doc, doc_uri_to_path),
                    range=_range_obj(ranges[def_range]),
                    role=OccurrenceRole.DEFINITION,
                    def_id=symbol_id,
                )
            )

        ref_result_id = range_to_ref_result.get(source_range_id)
        if ref_result_id is not None:
            for ref_range in reference_result_to_ranges.get(ref_result_id, []):
                occurrences.append(
                    Occurrence(
                        file_path=_file_of(ref_range, range_to_doc, doc_uri_to_path),
                        range=_range_obj(ranges[ref_range]),
                        role=OccurrenceRole.REFERENCE,
                        def_id=symbol_id,
                    )
                )
                relations.append(
                    Relation(
                        src_def_id=f"range:{ref_range}",
                        kind=RelationKind.REFERENCES,
                        dst_def_id=symbol_id,
                        resolved=True,
                    )
                )

    # Containment relationships
    for edge in contains_edges:
        out_v = int(edge["outV"])
        if out_v not in ranges:
            continue
        parent_symbol_id = _symbol_id(
            out_v, range_to_doc, doc_uri_to_path, ranges, result_to_moniker, range_to_result
        )
        if not parent_symbol_id:
            continue
        for in_v in [int(v) for v in edge.get("inVs", [])]:
            if in_v not in ranges:
                continue
            child_symbol_id = _symbol_id(
                in_v, range_to_doc, doc_uri_to_path, ranges, result_to_moniker, range_to_result
            )
            if not child_symbol_id or child_symbol_id == parent_symbol_id:
                continue
            relations.append(
                Relation(
                    src_def_id=parent_symbol_id,
                    kind=RelationKind.CONTAINS,
                    dst_def_id=child_symbol_id,
                    resolved=True,
                )
            )

    return Snapshot(
        files=files,
        symbols=list(symbols.values()),
        occurrences=occurrences,
        relations=relations,
        meta={"provider": "lsif"},
    )


def build_snapshot_lsif(lsif_path: Path | str) -> Snapshot:
    provider = LSIFProvider(lsif_path)
    if not provider.is_available():
        raise FileNotFoundError(f"LSIF index not found: {lsif_path}")
    return provider.extract()


def _normalize_uri(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def _range_obj(range_vertex: dict[str, Any]) -> Range:
    start = range_vertex.get("start") or {"line": 0, "character": 0}
    end = range_vertex.get("end") or {"line": 0, "character": 0}
    return Range(
        start_line=int(start.get("line", 0)) + 1,
        start_col=int(start.get("character", 0)),
        end_line=int(end.get("line", 0)) + 1,
        end_col=int(end.get("character", 0)),
    )


def _file_of(range_id: int, range_to_doc: dict[int, int], doc_uri_to_path: dict[int, str]) -> str:
    doc_id = range_to_doc.get(range_id)
    if doc_id is None:
        return "<unknown>"
    return doc_uri_to_path.get(doc_id, "<unknown>")


def _symbol_id(
    range_id: int,
    range_to_doc: dict[int, int],
    doc_uri_to_path: dict[int, str],
    ranges: dict[int, dict[str, Any]],
    result_to_moniker: dict[int, str],
    range_to_result: dict[int, int],
) -> str | None:
    result_set_id = range_to_result.get(range_id)
    if result_set_id is not None:
        moniker = result_to_moniker.get(result_set_id)
        if moniker:
            return f"lsif:{moniker}"

    range_vertex = ranges.get(range_id)
    if not range_vertex:
        return None
    rng = _range_obj(range_vertex)
    file_path = _file_of(range_id, range_to_doc, doc_uri_to_path)
    return f"lsif:{file_path}:{rng.start_line}:{rng.start_col}:{rng.end_line}:{rng.end_col}"


def _symbol_name(symbol_id: str) -> str:
    parts = symbol_id.split(":")
    return parts[-1] if parts else symbol_id
