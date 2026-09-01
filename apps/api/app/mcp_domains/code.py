"""Code navigation and analysis MCP tools."""

import json
import uuid
from typing import Annotated, Any, Literal

from contextmine_core import (
    Collection,
    Document,
    Symbol,
    SymbolEdge,
    get_settings,
    user_can_access_collection,
)
from contextmine_core import get_session as get_db_session
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from app.mcp_auth import get_current_user_id


class CodeTextResult(BaseModel):
    """Bounded code-navigation result."""

    kind: Literal["outline", "symbol", "definition", "references"]
    file_path: str
    content: str
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    count: int = 0
    truncated: bool = False


class CodeGraphResult(BaseModel):
    """Bounded code relationship graph."""

    seeds: list[str]
    depth: int
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    truncated: bool


class AnalysisResult(BaseModel):
    """Structured result returned by existing multi-engine analysis functions."""

    data: dict[str, Any]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _optional_uuid(value: str | None, field_name: str) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ToolError(f"Invalid {field_name}: {value}") from exc


async def _require_collection(db, collection_id: str) -> Collection:
    try:
        collection_uuid = uuid.UUID(collection_id)
    except ValueError as exc:
        raise ToolError(f"Invalid collection_id: {collection_id}") from exc
    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_uuid))
    ).scalar_one_or_none()
    if collection is None:
        raise ToolError("Collection not found.")
    if not await user_can_access_collection(db, collection, get_current_user_id()):
        raise ToolError("Access denied to this collection.")
    return collection


async def _lookup_document(session, file_path: str) -> Document | None:
    result = await session.execute(
        select(Document)
        .where(or_(Document.uri == file_path, Document.uri.contains(file_path)))
        .limit(1)
    )
    return result.scalar_one_or_none()


def _format_outline(file_path: str, symbols: list, include_children: bool) -> str:
    lines = [f"# Outline: {file_path}"]
    top_level = [symbol for symbol in symbols if symbol.parent_name is None]
    children: dict[str, list[Symbol]] = {}
    for symbol in symbols:
        if symbol.parent_name:
            children.setdefault(symbol.parent_name, []).append(symbol)
    for symbol in top_level:
        lines.append(
            f"## {symbol.kind.value} `{symbol.name}` (L{symbol.start_line}-{symbol.end_line})"
        )
        if symbol.signature:
            lines.append(f"```\n{symbol.signature}\n```")
        if include_children:
            for child in children.get(symbol.qualified_name, []):
                lines.append(
                    f"- {child.kind.value} `{child.name}` (L{child.start_line}-{child.end_line})"
                )
    return "\n".join(lines)


async def outline(
    file_path: Annotated[str, Field(description="Indexed document URI", min_length=1)],
    include_children: Annotated[bool, Field(description="Include class methods")] = True,
) -> CodeTextResult:
    """List indexed functions, classes, and methods for a file."""
    try:
        async with get_db_session() as session:
            document = await _lookup_document(session, file_path)
            if document is None:
                raise ToolError(f"No indexed document matches: {file_path}")
            symbols = list(
                (
                    await session.execute(
                        select(Symbol)
                        .where(Symbol.document_id == document.id)
                        .order_by(Symbol.start_line)
                    )
                ).scalars()
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to read code outline: {exc}") from exc
    return CodeTextResult(
        kind="outline",
        file_path=file_path,
        content=_format_outline(file_path, symbols, include_children),
        count=len(symbols),
    )


async def find_symbol(
    file_path: Annotated[str, Field(description="Indexed document URI", min_length=1)],
    name: Annotated[str, Field(description="Function, class, or method name", min_length=1)],
) -> CodeTextResult:
    """Return indexed source for a named symbol."""
    try:
        async with get_db_session() as session:
            document = await _lookup_document(session, file_path)
            if document is None:
                raise ToolError(f"No indexed document matches: {file_path}")
            symbol = (
                await session.execute(
                    select(Symbol).where(
                        Symbol.document_id == document.id,
                        Symbol.name == name,
                    )
                )
            ).scalar_one_or_none()
            if symbol is None:
                raise ToolError(f"No symbol named {name} in indexed document: {file_path}")
            lines = document.content_markdown.splitlines()
            content = "\n".join(lines[max(0, symbol.start_line - 1) : symbol.end_line])
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to find symbol: {exc}") from exc
    return CodeTextResult(
        kind="symbol",
        file_path=file_path,
        symbol=symbol.qualified_name,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        content=content,
        count=1,
    )


async def _symbol_at_location(db, file_path: str, line: int):  # noqa: ANN202
    escaped_path = _escape_like(file_path)
    document = (
        await db.execute(
            select(Document).where(Document.uri.ilike(f"%{escaped_path}%", escape="\\"))
        )
    ).scalar_one_or_none()
    if document is None:
        raise ToolError(f"No indexed document matches: {file_path}")
    symbol = (
        (
            await db.execute(
                select(Symbol)
                .where(
                    Symbol.document_id == document.id,
                    Symbol.start_line <= line,
                    Symbol.end_line >= line,
                )
                .order_by(Symbol.end_line - Symbol.start_line)
            )
        )
        .scalars()
        .first()
    )
    if symbol is None:
        raise ToolError(f"No indexed symbol at {file_path}:{line}")
    return document, symbol


async def definition(
    file_path: Annotated[str, Field(description="Indexed document URI", min_length=1)],
    line: Annotated[int, Field(description="One-indexed line", ge=1)],
    column: Annotated[int, Field(description="Zero-indexed column", ge=0)],
) -> CodeTextResult:
    """Return the smallest indexed symbol containing a source location."""
    del column
    async with get_db_session() as db:
        document, symbol = await _symbol_at_location(db, file_path, line)
    lines = (document.content_markdown or "").splitlines()
    content = "\n".join(lines[max(0, symbol.start_line - 1) : symbol.end_line])
    return CodeTextResult(
        kind="definition",
        file_path=document.uri,
        symbol=symbol.qualified_name,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        content=content,
        count=1,
    )


async def references(
    file_path: Annotated[str, Field(description="Indexed document URI", min_length=1)],
    line: Annotated[int, Field(description="One-indexed line", ge=1)],
    column: Annotated[int, Field(description="Zero-indexed column", ge=0)],
    limit: Annotated[int, Field(description="Maximum references", ge=1, le=200)] = 20,
) -> CodeTextResult:
    """Return bounded incoming references to the symbol at a source location."""
    del column
    async with get_db_session() as db:
        _document, symbol = await _symbol_at_location(db, file_path, line)
        edges = list(
            (
                await db.execute(
                    select(SymbolEdge)
                    .options(selectinload(SymbolEdge.source_symbol).selectinload(Symbol.document))
                    .where(SymbolEdge.target_symbol_id == symbol.id)
                    .limit(limit + 1)
                )
            ).scalars()
        )
    lines = []
    for edge in edges[:limit]:
        source = edge.source_symbol
        if source and source.document:
            lines.append(
                f"- {source.qualified_name} ({edge.edge_type.value}) "
                f"at {source.document.uri}:{edge.source_line or source.start_line}"
            )
    return CodeTextResult(
        kind="references",
        file_path=file_path,
        symbol=symbol.qualified_name,
        content="\n".join(lines),
        count=min(len(edges), limit),
        truncated=len(edges) > limit,
    )


def _parse_edge_types(edge_types: list[str] | None, edge_type_cls: type) -> list | None:
    if not edge_types:
        return None
    parsed = []
    for edge_type in edge_types:
        try:
            parsed.append(edge_type_cls(edge_type.lower()))
        except ValueError as exc:
            raise ToolError(f"Unknown edge type: {edge_type}") from exc
    return parsed


def _resolve_seeds(seeds: list[str], node_ids: set[str]) -> list[str]:
    exact = [seed for seed in seeds if seed in node_ids]
    if exact:
        return exact
    resolved = []
    for seed in seeds:
        symbol_name = seed.split("::")[-1]
        match = next((node_id for node_id in node_ids if symbol_name in node_id), None)
        if match:
            resolved.append(match)
    return resolved


async def expand(
    seeds: Annotated[list[str], Field(description="file_uri::symbol seeds", min_length=1)],
    depth: Annotated[int, Field(description="Traversal depth", ge=1, le=3)] = 2,
    edge_types: Annotated[list[str] | None, Field(description="Optional edge types")] = None,
    limit: Annotated[int, Field(description="Maximum nodes", ge=1, le=200)] = 30,
) -> CodeGraphResult:
    """Build and return a bounded code relationship graph."""
    try:
        from contextmine_core.graph import EdgeType, expand_graph, get_graph_builder

        file_paths = []
        for seed in seeds:
            if "::" not in seed:
                raise ToolError(f"Invalid seed format: {seed}")
            file_paths.append(seed.split("::", 1)[0])
        builder = get_graph_builder()
        if not builder.has_treesitter:
            raise ToolError("Tree-sitter is required for graph expansion.")
        graph = builder.build_multi_file_graph(sorted(set(file_paths)))
        node_ids = {node.id for node in graph.get_all_nodes()}
        valid_seeds = _resolve_seeds(seeds, node_ids)
        if not valid_seeds:
            raise ToolError("The provided seed symbols were not found.")
        subgraph = expand_graph(
            graph=graph,
            seeds=valid_seeds,
            depth=depth,
            edge_types=_parse_edge_types(edge_types, EdgeType),
        )
        all_nodes = list(subgraph.get_all_nodes())
        all_edges = list(subgraph.get_all_edges())
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Graph expansion failed: {exc}") from exc
    nodes = [
        {
            "id": node.id,
            "name": node.name,
            "kind": str(node.kind),
            "file_path": node.file_path,
            "start_line": node.start_line,
        }
        for node in all_nodes[:limit]
    ]
    edges = [
        {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "edge_type": edge.edge_type.value,
        }
        for edge in all_edges[:200]
    ]
    return CodeGraphResult(
        seeds=valid_seeds,
        depth=depth,
        nodes=nodes,
        edges=edges,
        truncated=len(all_nodes) > limit or len(all_edges) > 200,
    )


async def get_codebase_summary(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import get_codebase_summary_multi

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await get_codebase_summary_multi(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch codebase summary: {exc}") from exc


async def list_methods(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    query: Annotated[str | None, Field(description="Optional method-name filter")] = None,
    page: Annotated[int, Field(description="Zero-indexed page", ge=0)] = 0,
    limit: Annotated[int, Field(description="Items per page", ge=1, le=200)] = 50,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import list_methods_multi, sanitize_regex_query

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await list_methods_multi(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                query=sanitize_regex_query(query),
                page=page,
                limit=limit,
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to list methods: {exc}") from exc


async def list_calls(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    page: Annotated[int, Field(description="Zero-indexed page", ge=0)] = 0,
    limit: Annotated[int, Field(description="Items per page", ge=1, le=200)] = 50,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import list_calls_multi

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await list_calls_multi(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                page=page,
                limit=limit,
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to list calls: {exc}") from exc


async def get_cfg(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    node_ref: Annotated[str, Field(description="Method or node reference", min_length=1)],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    depth: Annotated[int, Field(description="Traversal depth", ge=1, le=8)] = 2,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import get_cfg_multi

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await get_cfg_multi(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                node_ref=node_ref,
                depth=depth,
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch CFG: {exc}") from exc


async def get_variable_flow(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    node_ref: Annotated[str, Field(description="Method or node reference", min_length=1)],
    variable: Annotated[str | None, Field(description="Optional variable name")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    max_hops: Annotated[int, Field(description="Maximum traversal hops", ge=1, le=20)] = 6,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import get_variable_flow_multi

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await get_variable_flow_multi(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                node_ref=node_ref,
                variable=variable,
                max_hops=max_hops,
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to fetch variable flow: {exc}") from exc


async def _taint_candidates(
    *,
    collection_id: str,
    scenario_id: str | None,
    language: str | None,
    limit: int,
    engines: str | None,
    sink: bool,
) -> AnalysisResult:
    from contextmine_core.twin import find_taint_sinks_multi, find_taint_sources_multi

    operation = find_taint_sinks_multi if sink else find_taint_sources_multi
    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await operation(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                language=language,
                limit=limit,
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        label = "sinks" if sink else "sources"
        raise ToolError(f"Failed to find taint {label}: {exc}") from exc


async def find_taint_sources(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    language: Annotated[str | None, Field(description="Language pattern profile")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    limit: Annotated[int, Field(description="Maximum candidates", ge=1, le=300)] = 50,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    return await _taint_candidates(
        collection_id=collection_id,
        scenario_id=scenario_id,
        language=language,
        limit=limit,
        engines=engines,
        sink=False,
    )


async def find_taint_sinks(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    language: Annotated[str | None, Field(description="Language pattern profile")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    limit: Annotated[int, Field(description="Maximum candidates", ge=1, le=300)] = 50,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    return await _taint_candidates(
        collection_id=collection_id,
        scenario_id=scenario_id,
        language=language,
        limit=limit,
        engines=engines,
        sink=True,
    )


async def find_taint_flows(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    language: Annotated[str | None, Field(description="Language pattern profile")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    max_hops: Annotated[int, Field(description="Maximum traversal hops", ge=1, le=20)] = 6,
    max_results: Annotated[int, Field(description="Maximum flows", ge=1, le=200)] = 50,
    engines: Annotated[str | None, Field(description="Comma-separated analysis engines")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import find_taint_flows_multi

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await find_taint_flows_multi(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                language=language,
                max_hops=max_hops,
                max_results=max_results,
                cache_ttl_seconds=get_settings().twin_analysis_cache_ttl_seconds,
                engines=_parse_csv(engines),
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to find taint flows: {exc}") from exc


async def store_findings(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    findings_json: Annotated[str, Field(description="JSON array of normalized findings")],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import store_findings as persist_findings

    try:
        findings = json.loads(findings_json)
    except json.JSONDecodeError as exc:
        raise ToolError("findings_json must be a valid JSON array.") from exc
    if not isinstance(findings, list):
        raise ToolError("findings_json must be a JSON array.")
    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await persist_findings(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                findings=findings,
            )
            await db.commit()
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to store findings: {exc}") from exc


async def export_sarif(
    collection_id: Annotated[str, Field(description="Collection UUID")],
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    status: Annotated[str | None, Field(description="Optional finding status")] = None,
    min_severity: Annotated[str | None, Field(description="Optional severity threshold")] = None,
) -> AnalysisResult:
    from contextmine_core.twin import export_findings_sarif

    try:
        async with get_db_session() as db:
            collection = await _require_collection(db, collection_id)
            payload = await export_findings_sarif(
                db,
                collection_id=collection.id,
                scenario_id=_optional_uuid(scenario_id, "scenario_id"),
                status=status,
                min_severity=min_severity,
            )
        return AnalysisResult(data=payload)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to export SARIF: {exc}") from exc


def register_code_tools(mcp: FastMCP) -> None:
    """Register all code-domain tools on the shared server."""
    for name, function in (
        ("outline", outline),
        ("find_symbol", find_symbol),
        ("definition", definition),
        ("references", references),
        ("expand", expand),
        ("get_codebase_summary", get_codebase_summary),
        ("list_methods", list_methods),
        ("list_calls", list_calls),
        ("get_cfg", get_cfg),
        ("get_variable_flow", get_variable_flow),
        ("find_taint_sources", find_taint_sources),
        ("find_taint_sinks", find_taint_sinks),
        ("find_taint_flows", find_taint_flows),
        ("store_findings", store_findings),
        ("export_sarif", export_sarif),
    ):
        mcp.tool(name=name)(function)
