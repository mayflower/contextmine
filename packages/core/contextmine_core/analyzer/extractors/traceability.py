"""Symbol-level traceability helpers for behavioral extraction.

This module resolves AST call sites to canonical symbol nodes using a
multi-engine strategy:
1. SCIP graph (via twin symbol call edges)
2. LSP go-to-definition (when repo checkout is available)
3. Joern call-site method resolution

Endpoint linkage is symbol-based through endpoint operation metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from contextmine_core.joern import JoernClient, parse_joern_output
from contextmine_core.lsp import get_lsp_manager
from contextmine_core.models import (
    KnowledgeNode,
    KnowledgeNodeKind,
    TwinEdge,
    TwinNode,
    TwinScenario,
    TwinSourceVersion,
)
from contextmine_core.settings import get_settings
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert


def _canonical_path(path: str) -> str:
    value = (path or "").replace("\\", "/").split("?", 1)[0].strip()
    if value.startswith("./"):
        value = value[2:]
    return value


def _normalize_symbol_token(value: str) -> str:
    token = (value or "").strip()
    if not token:
        return ""
    token = token.rsplit("::", 1)[-1]
    token = token.rsplit(".", 1)[-1]
    token = token.rsplit("#", 1)[-1]
    token = token.rsplit("/", 1)[-1]
    token = token.rsplit(":", 1)[-1]
    return token.strip().lower()


def _extract_token_variants(value: str) -> set[str]:
    raw = (value or "").strip()
    if not raw:
        return set()
    base = _normalize_symbol_token(raw)
    variants: set[str] = set()
    if base:
        variants.add(base)
    compact = re.sub(r"[^a-zA-Z0-9_]", "", raw).lower()
    if compact:
        variants.add(compact)
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw).lower()
    snake = re.sub(r"[^a-z0-9_]", "_", snake).strip("_")
    if snake:
        variants.add(snake)
    return {item for item in variants if item}


def symbol_token_variants(value: str) -> set[str]:
    """Public helper for consistent symbol token normalization."""
    return _extract_token_variants(value)


def _parse_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class CallSite:
    """One call site extracted from source."""

    file_path: str
    line: int
    column: int
    callee: str


@dataclass(frozen=True)
class ResolvedSymbolRef:
    """Resolved symbol reference with provenance for traceability."""

    symbol_node_id: UUID
    symbol_name: str
    engine: str
    confidence: float
    natural_key: str


@dataclass
class _KnowledgeSymbolRow:
    node_id: UUID
    name: str
    natural_key: str
    file_path: str
    start_line: int
    end_line: int
    def_id: str | None


@dataclass
class _TwinSymbolRow:
    node_id: UUID
    natural_key: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    def_id: str | None
    kind: str


class SymbolTraceResolver:
    """Resolve call sites to symbol nodes using SCIP/LSP/Joern."""

    def __init__(
        self,
        *,
        session: Any,
        collection_id: UUID,
        source_id: UUID | None = None,
    ) -> None:
        self._session = session
        self._collection_id = collection_id
        self._source_id = source_id

        self._kg_loaded = False
        self._kg_by_name: dict[str, list[_KnowledgeSymbolRow]] = {}
        self._kg_by_file: dict[str, list[_KnowledgeSymbolRow]] = {}
        self._kg_by_def_id: dict[str, _KnowledgeSymbolRow] = {}
        self._kg_by_node_id: dict[UUID, _KnowledgeSymbolRow] = {}

        self._scip_loaded = False
        self._twin_by_id: dict[UUID, _TwinSymbolRow] = {}
        self._twin_by_file: dict[str, list[_TwinSymbolRow]] = {}
        self._twin_calls: dict[UUID, list[UUID]] = {}
        self._twin_refs: dict[UUID, list[UUID]] = {}

        self._repo_root: Path | None = None
        self._lsp_client: Any | None = None

        self._joern_checked = False
        self._joern_client: JoernClient | None = None
        self._joern_ready = False

    async def resolve_many(
        self,
        *,
        call_sites: list[CallSite],
        fallback_symbol_hints: list[str],
    ) -> list[ResolvedSymbolRef]:
        await self._ensure_knowledge_symbols()
        resolved_by_node: dict[UUID, ResolvedSymbolRef] = {}

        unresolved: list[CallSite] = []
        for call_site in call_sites:
            refs = await self._resolve_with_scip(call_site)
            if not refs:
                unresolved.append(call_site)
                continue
            for ref in refs:
                self._remember_best(resolved_by_node, ref)

        still_unresolved: list[CallSite] = []
        for call_site in unresolved:
            refs = await self._resolve_with_lsp(call_site)
            if not refs:
                still_unresolved.append(call_site)
                continue
            for ref in refs:
                self._remember_best(resolved_by_node, ref)

        # Joern is expensive: only attempt for a bounded number of unresolved callsites.
        for call_site in still_unresolved[:20]:
            refs = await self._resolve_with_joern(call_site)
            for ref in refs:
                self._remember_best(resolved_by_node, ref)

        if not resolved_by_node:
            for hint in fallback_symbol_hints:
                key = _normalize_symbol_token(hint)
                if not key:
                    continue
                for row in self._kg_by_name.get(key, []):
                    self._remember_best(
                        resolved_by_node,
                        ResolvedSymbolRef(
                            symbol_node_id=row.node_id,
                            symbol_name=row.name,
                            engine="name_fallback",
                            confidence=0.45,
                            natural_key=row.natural_key,
                        ),
                    )

        return sorted(
            resolved_by_node.values(),
            key=lambda item: (-item.confidence, item.symbol_name.lower(), item.natural_key),
        )

    def _remember_best(
        self,
        resolved_by_node: dict[UUID, ResolvedSymbolRef],
        candidate: ResolvedSymbolRef,
    ) -> None:
        current = resolved_by_node.get(candidate.symbol_node_id)
        if current is None or candidate.confidence > current.confidence:
            resolved_by_node[candidate.symbol_node_id] = candidate

    async def _resolve_with_scip(self, call_site: CallSite) -> list[ResolvedSymbolRef]:
        await self._ensure_scip_graph()
        if not self._twin_by_file:
            return []

        file_key = _canonical_path(call_site.file_path)
        candidates = list(self._twin_by_file.get(file_key, []))
        if not candidates:
            return []

        callers = [
            item
            for item in candidates
            if item.start_line <= call_site.line <= item.end_line
            and (item.end_col <= 0 or item.start_col <= call_site.column <= item.end_col)
        ]
        if not callers:
            callers = [
                item for item in candidates if item.start_line <= call_site.line <= item.end_line
            ]
        if not callers:
            return []
        callers.sort(key=lambda item: (item.end_line - item.start_line, item.start_col))

        callee_token = _normalize_symbol_token(call_site.callee)
        seen_targets: set[UUID] = set()
        refs: list[ResolvedSymbolRef] = []

        for caller in callers[:3]:
            called_ids = self._twin_calls.get(caller.node_id, [])
            referenced_ids = self._twin_refs.get(caller.node_id, [])
            for target_id in [*called_ids, *referenced_ids]:
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                target = self._twin_by_id.get(target_id)
                if target is None:
                    continue
                if callee_token and callee_token not in _extract_token_variants(target.name):
                    continue
                kg_row = await self._ensure_knowledge_symbol_from_twin(target)
                engine = "scip.calls" if target_id in called_ids else "scip.refs"
                confidence = 0.94 if engine == "scip.calls" else 0.81
                refs.append(
                    ResolvedSymbolRef(
                        symbol_node_id=kg_row.node_id,
                        symbol_name=kg_row.name,
                        engine=engine,
                        confidence=confidence,
                        natural_key=kg_row.natural_key,
                    )
                )
        return refs

    async def _resolve_with_lsp(self, call_site: CallSite) -> list[ResolvedSymbolRef]:
        await self._ensure_lsp_client()
        if self._lsp_client is None or self._repo_root is None:
            return []

        file_path = (self._repo_root / call_site.file_path).resolve()
        if not file_path.exists():
            return []

        refs: list[ResolvedSymbolRef] = []
        try:
            defs = await self._lsp_client.get_definition(
                str(file_path),
                max(1, int(call_site.line)),
                max(0, int(call_site.column)),
            )
        except Exception:
            return []

        for definition in defs:
            row = await self._find_knowledge_symbol_by_location(
                file_path=definition.file_path,
                line=definition.start_line,
            )
            if row is None:
                row = await self._find_or_create_knowledge_symbol_from_twin_location(
                    file_path=definition.file_path,
                    line=definition.start_line,
                )
            if row is None:
                continue
            refs.append(
                ResolvedSymbolRef(
                    symbol_node_id=row.node_id,
                    symbol_name=row.name,
                    engine="lsp.definition",
                    confidence=0.82,
                    natural_key=row.natural_key,
                )
            )
        return refs

    async def _resolve_with_joern(self, call_site: CallSite) -> list[ResolvedSymbolRef]:
        await self._ensure_joern_client()
        if not self._joern_ready or self._joern_client is None:
            return []

        file_needle = _canonical_path(call_site.file_path)
        if not file_needle:
            return []
        escaped = file_needle.replace("\\", "\\\\").replace('"', '\\"')
        line = max(1, int(call_site.line))
        query = (
            "val rows = cpg.call.where(c => "
            f'c.lineNumber.l.contains({line}) && c.file.name.l.exists(_.endsWith("{escaped}")))'
            ".map(_.methodFullName).toList.distinct; "
            'println("<contextmine_result>" + rows.mkString("||") + "</contextmine_result>")'
        )

        try:
            response = await self._joern_client.execute_query(query)
        except Exception:
            return []
        if not response.success:
            return []

        parsed = parse_joern_output(response.stdout)
        rows: list[str] = []
        if isinstance(parsed, str):
            rows = [item.strip() for item in parsed.split("||") if item.strip()]
        elif isinstance(parsed, list):
            rows = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            return []

        refs: list[ResolvedSymbolRef] = []
        for method_full_name in rows:
            token = _normalize_symbol_token(method_full_name)
            if not token:
                continue
            for row in self._kg_by_name.get(token, []):
                refs.append(
                    ResolvedSymbolRef(
                        symbol_node_id=row.node_id,
                        symbol_name=row.name,
                        engine="joern.call",
                        confidence=0.68,
                        natural_key=row.natural_key,
                    )
                )
        return refs

    async def _ensure_knowledge_symbols(self) -> None:
        if self._kg_loaded:
            return

        rows = (
            (
                await self._session.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.collection_id == self._collection_id,
                        KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            meta = row.meta or {}
            file_path = _canonical_path(str(meta.get("file_path") or ""))
            start_line = int(meta.get("start_line") or 0)
            end_line = int(meta.get("end_line") or 0)
            def_id = str(meta.get("def_id") or "").strip() or None
            symbol = _KnowledgeSymbolRow(
                node_id=row.id,
                name=row.name,
                natural_key=row.natural_key,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                def_id=def_id,
            )
            self._kg_by_node_id[row.id] = symbol
            self._kg_by_name.setdefault(_normalize_symbol_token(row.name), []).append(symbol)
            if file_path:
                self._kg_by_file.setdefault(file_path, []).append(symbol)
            if def_id:
                self._kg_by_def_id[def_id] = symbol
        self._kg_loaded = True

    async def _ensure_scip_graph(self) -> None:
        if self._scip_loaded:
            return
        scenario = (
            await self._session.execute(
                select(TwinScenario)
                .where(
                    TwinScenario.collection_id == self._collection_id,
                    TwinScenario.is_as_is.is_(True),
                )
                .order_by(TwinScenario.updated_at.desc(), TwinScenario.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if scenario is None:
            self._scip_loaded = True
            return

        twin_rows = (
            (
                await self._session.execute(
                    select(TwinNode).where(
                        TwinNode.scenario_id == scenario.id,
                        TwinNode.natural_key.like("symbol:%"),
                        TwinNode.is_active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in twin_rows:
            meta = row.meta or {}
            file_path = _canonical_path(str(meta.get("file_path") or ""))
            range_obj = meta.get("range") if isinstance(meta.get("range"), dict) else {}
            start_line = int((range_obj or {}).get("start_line") or 0)
            end_line = int((range_obj or {}).get("end_line") or 0)
            start_col = int((range_obj or {}).get("start_col") or 0)
            end_col = int((range_obj or {}).get("end_col") or 0)
            twin_symbol = _TwinSymbolRow(
                node_id=row.id,
                natural_key=row.natural_key,
                name=row.name,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                start_col=start_col,
                end_col=end_col,
                def_id=str(meta.get("def_id") or "").strip() or None,
                kind=str(meta.get("symbol_kind") or row.kind),
            )
            self._twin_by_id[row.id] = twin_symbol
            if file_path:
                self._twin_by_file.setdefault(file_path, []).append(twin_symbol)

        edge_rows = (
            (
                await self._session.execute(
                    select(TwinEdge).where(
                        TwinEdge.scenario_id == scenario.id,
                        TwinEdge.kind.in_(["symbol_calls_symbol", "symbol_references_symbol"]),
                        TwinEdge.is_active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        for edge in edge_rows:
            if edge.kind == "symbol_calls_symbol":
                self._twin_calls.setdefault(edge.source_node_id, []).append(edge.target_node_id)
            else:
                self._twin_refs.setdefault(edge.source_node_id, []).append(edge.target_node_id)
        self._scip_loaded = True

    async def _ensure_lsp_client(self) -> None:
        if self._lsp_client is not None:
            return
        if self._source_id is None:
            return

        settings = get_settings()
        repo_root = Path(settings.repos_root) / str(self._source_id)
        if not repo_root.exists():
            return
        self._repo_root = repo_root

        try:
            # Select any likely code file to initialize the client.
            sample = next(
                (
                    path
                    for path in repo_root.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".java", ".php"}
                ),
                None,
            )
            if sample is None:
                return
            self._lsp_client = await get_lsp_manager().get_client(sample, project_root=repo_root)
        except Exception:
            self._lsp_client = None

    async def _ensure_joern_client(self) -> None:
        if self._joern_checked:
            return
        self._joern_checked = True

        settings = get_settings()
        query = select(TwinSourceVersion).where(
            TwinSourceVersion.collection_id == self._collection_id,
            TwinSourceVersion.status == "ready",
            TwinSourceVersion.joern_cpg_path.is_not(None),
        )
        if self._source_id is not None:
            query = query.where(TwinSourceVersion.source_id == self._source_id)
        row = (
            await self._session.execute(
                query.order_by(
                    TwinSourceVersion.finished_at.desc(), TwinSourceVersion.created_at.desc()
                ).limit(1)
            )
        ).scalar_one_or_none()
        if row is None or not row.joern_cpg_path:
            return

        base_url = row.joern_server_url or settings.joern_server_url
        client = JoernClient(base_url, timeout_seconds=settings.joern_query_timeout_seconds)
        try:
            if not await client.check_health():
                return
            load = await client.load_cpg(
                row.joern_cpg_path,
                timeout_seconds=max(settings.joern_query_timeout_seconds, 300),
            )
            if not load.success:
                return
        except Exception:
            return

        self._joern_client = client
        self._joern_ready = True

    async def _find_knowledge_symbol_by_location(
        self,
        *,
        file_path: str,
        line: int,
    ) -> _KnowledgeSymbolRow | None:
        key = _canonical_path(file_path)
        if key.startswith("/") and self._repo_root is not None:
            # Try to map absolute repo path to relative path.
            root = str(self._repo_root.resolve()).replace("\\", "/")
            if key.startswith(root + "/"):
                key = key[len(root) + 1 :]

        candidates = self._kg_by_file.get(key, [])
        matches = [
            row
            for row in candidates
            if row.start_line > 0
            and row.end_line >= row.start_line
            and row.start_line <= line <= row.end_line
        ]
        if not matches:
            return None
        matches.sort(key=lambda row: (row.end_line - row.start_line, row.start_line))
        return matches[0]

    async def _find_or_create_knowledge_symbol_from_twin_location(
        self,
        *,
        file_path: str,
        line: int,
    ) -> _KnowledgeSymbolRow | None:
        await self._ensure_scip_graph()
        key = _canonical_path(file_path)
        if key.startswith("/") and self._repo_root is not None:
            root = str(self._repo_root.resolve()).replace("\\", "/")
            if key.startswith(root + "/"):
                key = key[len(root) + 1 :]

        candidates = [
            row for row in self._twin_by_file.get(key, []) if row.start_line <= line <= row.end_line
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda row: (row.end_line - row.start_line, row.start_line))
        return await self._ensure_knowledge_symbol_from_twin(candidates[0])

    async def _ensure_knowledge_symbol_from_twin(
        self,
        twin_symbol: _TwinSymbolRow,
    ) -> _KnowledgeSymbolRow:
        await self._ensure_knowledge_symbols()
        if twin_symbol.def_id and twin_symbol.def_id in self._kg_by_def_id:
            return self._kg_by_def_id[twin_symbol.def_id]

        stmt = pg_insert(KnowledgeNode).values(
            collection_id=self._collection_id,
            kind=KnowledgeNodeKind.SYMBOL,
            natural_key=f"symbol:{twin_symbol.def_id or twin_symbol.natural_key}",
            name=twin_symbol.name,
            meta={
                "file_path": twin_symbol.file_path,
                "start_line": twin_symbol.start_line,
                "end_line": twin_symbol.end_line,
                "def_id": twin_symbol.def_id,
                "symbol_kind": twin_symbol.kind,
                "source": "twin_snapshot",
                "resolver": "scip",
            },
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_knowledge_node_natural",
            set_={
                "name": stmt.excluded.name,
                "meta": stmt.excluded.meta,
            },
        ).returning(
            KnowledgeNode.id, KnowledgeNode.natural_key, KnowledgeNode.name, KnowledgeNode.meta
        )
        row = (await self._session.execute(stmt)).one()

        meta = row.meta or {}
        symbol = _KnowledgeSymbolRow(
            node_id=row.id,
            name=row.name,
            natural_key=row.natural_key,
            file_path=_canonical_path(str(meta.get("file_path") or twin_symbol.file_path)),
            start_line=int(meta.get("start_line") or twin_symbol.start_line),
            end_line=int(meta.get("end_line") or twin_symbol.end_line),
            def_id=str(meta.get("def_id") or twin_symbol.def_id or "").strip() or None,
        )
        self._kg_by_node_id[symbol.node_id] = symbol
        self._kg_by_name.setdefault(_normalize_symbol_token(symbol.name), []).append(symbol)
        if symbol.file_path:
            self._kg_by_file.setdefault(symbol.file_path, []).append(symbol)
        if symbol.def_id:
            self._kg_by_def_id[symbol.def_id] = symbol
        return symbol


async def resolve_symbol_refs_for_calls(
    *,
    session: Any,
    collection_id: UUID,
    source_id: UUID | None,
    file_path: str,
    call_sites: list[dict[str, Any]],
    fallback_symbol_hints: list[str],
) -> list[ResolvedSymbolRef]:
    """Resolve extracted call sites to symbol nodes."""
    resolver = SymbolTraceResolver(
        session=session, collection_id=collection_id, source_id=source_id
    )
    normalized_sites: list[CallSite] = []
    for item in call_sites:
        line = int(item.get("line") or 0)
        if line <= 0:
            continue
        normalized_sites.append(
            CallSite(
                file_path=file_path,
                line=line,
                column=int(item.get("column") or 0),
                callee=str(item.get("callee") or ""),
            )
        )
    return await resolver.resolve_many(
        call_sites=normalized_sites,
        fallback_symbol_hints=fallback_symbol_hints,
    )


async def build_endpoint_symbol_index(
    *,
    session: Any,
    collection_id: UUID,
) -> dict[str, set[UUID]]:
    """Build symbol-token -> endpoint-node-id index."""
    endpoint_rows = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
                )
            )
        )
        .scalars()
        .all()
    )
    index: dict[str, set[UUID]] = {}
    symbol_node_ids: set[UUID] = set()
    for endpoint in endpoint_rows:
        meta = endpoint.meta or {}
        raw_ids = meta.get("handler_symbol_node_ids")
        if not isinstance(raw_ids, list):
            continue
        for raw in raw_ids:
            parsed = _parse_uuid(raw)
            if parsed is not None:
                symbol_node_ids.add(parsed)

    symbol_rows: dict[UUID, KnowledgeNode] = {}
    if symbol_node_ids:
        loaded_rows = (
            (
                await session.execute(
                    select(KnowledgeNode).where(
                        KnowledgeNode.collection_id == collection_id,
                        KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
                        KnowledgeNode.id.in_(symbol_node_ids),
                    )
                )
            )
            .scalars()
            .all()
        )
        symbol_rows = {row.id: row for row in loaded_rows}

    for endpoint in endpoint_rows:
        meta = endpoint.meta or {}
        tokens: set[str] = set()

        operation_id = str(meta.get("operation_id") or "").strip()
        tokens.update(_extract_token_variants(operation_id))

        for item in (
            meta.get("handler_symbol_names", [])
            if isinstance(meta.get("handler_symbol_names"), list)
            else []
        ):
            tokens.update(_extract_token_variants(str(item)))

        for item in (
            meta.get("handler_symbols", []) if isinstance(meta.get("handler_symbols"), list) else []
        ):
            tokens.update(_extract_token_variants(str(item)))

        for raw in (
            meta.get("handler_symbol_node_ids", [])
            if isinstance(meta.get("handler_symbol_node_ids"), list)
            else []
        ):
            symbol_id = _parse_uuid(raw)
            if symbol_id is None:
                continue
            symbol = symbol_rows.get(symbol_id)
            if symbol is None:
                continue
            tokens.update(_extract_token_variants(symbol.name))
            tokens.update(_extract_token_variants(symbol.natural_key))

        for token in tokens:
            index.setdefault(token, set()).add(endpoint.id)
    return index
