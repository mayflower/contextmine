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


def _classify_edges(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition edges by label into a dict."""
    result: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        label = edge.get("label", "")
        result.setdefault(label, []).append(edge)
    return result


def _build_range_to_doc(
    contains_edges: list[dict[str, Any]],
    documents: dict[int, dict[str, Any]],
    ranges: dict[int, dict[str, Any]],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for edge in contains_edges:
        out_v = int(edge["outV"])
        if out_v not in documents:
            continue
        for in_v in (int(v) for v in edge.get("inVs", [])):
            if in_v in ranges:
                mapping[in_v] = out_v
    return mapping


def _build_range_to_result(
    next_edges: list[dict[str, Any]],
    ranges: dict[int, dict[str, Any]],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for edge in next_edges:
        out_v = int(edge["outV"])
        if out_v in ranges:
            mapping[out_v] = int(edge["inV"])
    return mapping


def _build_result_to_moniker(
    moniker_edges: list[dict[str, Any]],
    monikers: dict[int, dict[str, Any]],
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for edge in moniker_edges:
        moniker = monikers.get(int(edge["inV"]))
        if not moniker:
            continue
        identifier = str(moniker.get("identifier") or "")
        if identifier:
            mapping[int(edge["outV"])] = identifier
    return mapping


def _build_item_result_maps(
    item_edges: list[dict[str, Any]],
    ranges: dict[int, dict[str, Any]],
    def_result_to_ranges: dict[int, list[int]],
    ref_result_to_ranges: dict[int, list[int]],
) -> None:
    for edge in item_edges:
        out_v = int(edge["outV"])
        in_vs = [int(v) for v in edge.get("inVs", []) if int(v) in ranges]
        prop = str(edge.get("property", "references"))
        target = def_result_to_ranges if prop == "definitions" else ref_result_to_ranges
        target.setdefault(out_v, []).extend(in_vs)


def _build_range_to_target(
    target_edges: list[dict[str, Any]],
    ranges: dict[int, dict[str, Any]],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for edge in target_edges:
        out_v = int(edge["outV"])
        if out_v in ranges:
            mapping[out_v] = int(edge["inV"])
    return mapping


_LsifCtx = tuple[
    dict[int, int], dict[int, str], dict[int, dict[str, Any]], dict[int, str], dict[int, int]
]


def _resolve_symbol_id(range_id: int, ctx: _LsifCtx) -> str | None:
    range_to_doc, doc_uri_to_path, ranges, result_to_moniker, range_to_result = ctx
    return _symbol_id(
        range_id, range_to_doc, doc_uri_to_path, ranges, result_to_moniker, range_to_result
    )


def _build_symbols_from_definitions(
    range_to_def_result: dict[int, int],
    definition_result_to_ranges: dict[int, list[int]],
    ctx: _LsifCtx,
    symbols: dict[str, Symbol],
) -> None:
    """Build Symbol objects from definition ranges."""
    range_to_doc, doc_uri_to_path, ranges, _, _ = ctx
    for _source_range_id, def_result_id in range_to_def_result.items():
        def_ranges = definition_result_to_ranges.get(def_result_id, [])
        if not def_ranges:
            continue
        canonical_range_id = def_ranges[0]
        symbol_id = _resolve_symbol_id(canonical_range_id, ctx)
        if not symbol_id or symbol_id in symbols:
            continue
        symbols[symbol_id] = Symbol(
            def_id=symbol_id,
            kind=SymbolKind.UNKNOWN,
            file_path=_file_of(canonical_range_id, range_to_doc, doc_uri_to_path),
            range=_range_obj(ranges[canonical_range_id]),
            name=_symbol_name(symbol_id),
            container_def_id=None,
        )


def _build_occurrences_and_relations(
    range_to_def_result: dict[int, int],
    definition_result_to_ranges: dict[int, list[int]],
    reference_result_to_ranges: dict[int, list[int]],
    range_to_ref_result: dict[int, int],
    ctx: _LsifCtx,
    occurrences: list[Occurrence],
    relations: list[Relation],
) -> None:
    """Build occurrence and reference relation entries."""
    range_to_doc, doc_uri_to_path, ranges, _, _ = ctx
    for source_range_id, def_result_id in range_to_def_result.items():
        def_ranges = definition_result_to_ranges.get(def_result_id, [])
        if not def_ranges:
            continue
        symbol_id = _resolve_symbol_id(def_ranges[0], ctx)
        if not symbol_id:
            continue
        occurrences.append(
            Occurrence(
                file_path=_file_of(source_range_id, range_to_doc, doc_uri_to_path),
                range=_range_obj(ranges[source_range_id]),
                role=OccurrenceRole.REFERENCE,
                def_id=symbol_id,
            )
        )
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
        if ref_result_id is None:
            continue
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


def _build_containment_relations(
    contains_edges: list[dict[str, Any]],
    ctx: _LsifCtx,
    relations: list[Relation],
) -> None:
    """Build containment relations from contains edges."""
    _, _, ranges, _, _ = ctx
    for edge in contains_edges:
        out_v = int(edge["outV"])
        if out_v not in ranges:
            continue
        parent_symbol_id = _resolve_symbol_id(out_v, ctx)
        if not parent_symbol_id:
            continue
        for in_v in (int(v) for v in edge.get("inVs", [])):
            if in_v not in ranges:
                continue
            child_symbol_id = _resolve_symbol_id(in_v, ctx)
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

    edges_by_label = _classify_edges(edges)
    contains_edges = edges_by_label.get("contains", [])

    range_to_doc = _build_range_to_doc(contains_edges, documents, ranges)
    range_to_result = _build_range_to_result(edges_by_label.get("next", []), ranges)
    result_to_moniker = _build_result_to_moniker(edges_by_label.get("moniker", []), monikers)

    definition_result_to_ranges: dict[int, list[int]] = {}
    reference_result_to_ranges: dict[int, list[int]] = {}
    _build_item_result_maps(
        edges_by_label.get("item", []),
        ranges,
        definition_result_to_ranges,
        reference_result_to_ranges,
    )

    range_to_def_result = _build_range_to_target(
        edges_by_label.get("textDocument/definition", []),
        ranges,
    )
    range_to_ref_result = _build_range_to_target(
        edges_by_label.get("textDocument/references", []),
        ranges,
    )

    _lsif_ctx = (range_to_doc, doc_uri_to_path, ranges, result_to_moniker, range_to_result)

    _build_symbols_from_definitions(
        range_to_def_result,
        definition_result_to_ranges,
        _lsif_ctx,
        symbols,
    )

    _build_occurrences_and_relations(
        range_to_def_result,
        definition_result_to_ranges,
        reference_result_to_ranges,
        range_to_ref_result,
        _lsif_ctx,
        occurrences,
        relations,
    )

    _build_containment_relations(contains_edges, _lsif_ctx, relations)

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
