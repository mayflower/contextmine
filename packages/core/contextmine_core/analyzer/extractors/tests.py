"""Deterministic test extractor and graph materializer."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.analyzer.extractors.ast_utils import (
    first_child,
    is_pascal_case,
    line_number,
    node_text,
    unquote,
    walk,
)
from contextmine_core.analyzer.extractors.traceability import resolve_symbol_refs_for_calls
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
)
from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import get_treesitter_manager
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]  # noqa: S324 - stable key only


def _provenance(
    *,
    mode: str,
    extractor: str,
    confidence: float,
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "provenance": {
            "mode": mode,
            "extractor": extractor,
            "confidence": round(max(0.0, min(confidence, 1.0)), 4),
            "evidence_ids": list(dict.fromkeys(evidence_ids or [])),
        }
    }


TEST_FILE_PATTERNS: tuple[str, ...] = (
    "test_",
    "_test.",
    ".spec.",
    ".test.",
    "__tests__",
    "/tests/",
)

JS_SUITE_CALLS = {"describe", "context", "suite"}
JS_CASE_CALLS = {"test", "it", "specify"}
JS_FIXTURE_CALLS = {"beforeEach", "beforeAll", "afterEach", "afterAll"}
HTTP_METHOD_NAMES = {"get", "post", "put", "patch", "delete"}
HTTP_CLIENT_NAMES = {"axios", "client", "api", "http", "request", "agent", "supertest"}
SYMBOL_STOP_WORDS = {
    "assert",
    "expect",
    "test",
    "it",
    "describe",
    "beforeEach",
    "beforeAll",
    "afterEach",
    "afterAll",
    "fetch",
    "get",
    "post",
    "put",
    "patch",
    "delete",
}


@dataclass
class TestFixtureDef:
    """Extracted test fixture."""

    name: str
    file_path: str
    line: int


@dataclass
class TestCaseDef:
    """Extracted test case."""

    name: str
    file_path: str
    line: int
    suite_name: str | None = None
    fixture_names: list[str] = field(default_factory=list)
    symbol_hints: list[str] = field(default_factory=list)
    endpoint_hints: list[str] = field(default_factory=list)
    call_sites: list[dict[str, Any]] = field(default_factory=list)
    raw_assertions: list[str] = field(default_factory=list)
    natural_key: str = ""


@dataclass
class TestSuiteDef:
    """Extracted test suite."""

    name: str
    file_path: str
    line: int
    natural_key: str = ""


@dataclass
class TestsExtraction:
    """Result of extracting test semantics from one file."""

    file_path: str
    framework: str
    suites: list[TestSuiteDef] = field(default_factory=list)
    cases: list[TestCaseDef] = field(default_factory=list)
    fixtures: list[TestFixtureDef] = field(default_factory=list)


def looks_like_test_file(file_path: str) -> bool:
    """Return True when a file path resembles a test file."""
    lower = file_path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".svg", ".lock", ".sum", ".map")):
        return False
    return any(marker in lower for marker in TEST_FILE_PATTERNS)


def detect_test_framework(file_path: str, content: str) -> str:
    """Best-effort framework detection."""
    lower = content.lower()
    if "pytest" in lower:
        return "pytest"
    if "unittest" in lower:
        return "unittest"
    if "jest" in lower:
        return "jest"
    if "@test" in lower or "junit" in lower:
        return "junit"
    if "vitest" in lower:
        return "vitest"
    if "cypress" in lower:
        return "cypress"
    if "playwright" in lower:
        return "playwright"
    if file_path.endswith((".spec.ts", ".spec.js", ".test.ts", ".test.js")):
        return "js_test"
    return "unknown"


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v.strip() for v in values if v and v.strip()))


def extract_tests_from_file(file_path: str, content: str) -> TestsExtraction:
    """Extract test suites/cases/fixtures and behavioral hints from one file."""
    framework = detect_test_framework(file_path, content)
    extraction = TestsExtraction(file_path=file_path, framework=framework)
    language = detect_language(file_path)
    if language is None:
        return extraction

    manager = get_treesitter_manager()
    try:
        tree = manager.parse(file_path, content)
    except Exception as exc:
        logger.debug("test extraction skipped for %s: %s", file_path, exc)
        return extraction

    root = tree.root_node
    if language == TreeSitterLanguage.PYTHON:
        _extract_python_tests(file_path, content, root, extraction)
    elif language in {
        TreeSitterLanguage.JAVASCRIPT,
        TreeSitterLanguage.TYPESCRIPT,
        TreeSitterLanguage.TSX,
    }:
        _extract_js_tests(file_path, content, root, extraction)

    extraction.suites = _dedupe_suites(extraction.suites)
    extraction.fixtures = _dedupe_fixtures(extraction.fixtures)
    extraction.cases = _dedupe_cases(extraction.cases)

    for case in extraction.cases:
        if not case.natural_key:
            case.natural_key = (
                f"test_case:{file_path}:{case.name}:{_hash(f'{file_path}:{case.name}:{case.line}')}"
            )
    return extraction


def extract_tests_from_files(files: list[tuple[str, str]]) -> list[TestsExtraction]:
    """Extract tests semantics from a list of files."""
    extractions: list[TestsExtraction] = []
    for file_path, content in files:
        if not looks_like_test_file(file_path):
            continue
        if not content.strip():
            continue
        extraction = extract_tests_from_file(file_path, content)
        if extraction.suites or extraction.cases or extraction.fixtures:
            extractions.append(extraction)
    return extractions


def _extract_python_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    for node in walk(root):
        if node.type == "class_definition":
            name = node_text(content, first_child(node, "identifier")).strip()
            if name and ("test" in name.lower() or is_pascal_case(name)):
                extraction.suites.append(
                    TestSuiteDef(
                        name=name,
                        file_path=file_path,
                        line=line_number(node),
                        natural_key=f"test_suite:{file_path}:{name}",
                    )
                )

        if node.type not in {"function_definition", "decorated_definition"}:
            continue

        decorators = _python_decorator_names(content, node)
        function_node = _python_function_node(node)
        if function_node is None:
            continue
        fn_name = node_text(content, first_child(function_node, "identifier")).strip()
        if not fn_name:
            continue

        if any(name.endswith("fixture") for name in decorators):
            extraction.fixtures.append(
                TestFixtureDef(
                    name=fn_name,
                    file_path=file_path,
                    line=line_number(function_node),
                )
            )

        if not fn_name.startswith("test_"):
            continue

        fixture_names = [
            name
            for name in _python_param_names(content, function_node)
            if name not in {"self", "cls"}
        ]
        symbol_hints, endpoint_hints, call_sites, raw_assertions = _collect_python_case_signals(
            content, function_node
        )
        suite_name = _python_parent_suite(content, function_node)
        case = TestCaseDef(
            name=fn_name,
            file_path=file_path,
            line=line_number(function_node),
            suite_name=suite_name,
            fixture_names=fixture_names[:8],
            symbol_hints=symbol_hints[:12],
            endpoint_hints=endpoint_hints[:12],
            call_sites=call_sites[:80],
            raw_assertions=raw_assertions[:8],
        )
        case.natural_key = (
            f"test_case:{file_path}:{fn_name}:{_hash(f'{file_path}:{fn_name}:{case.line}')}"
        )
        extraction.cases.append(case)


def _python_function_node(node: Any) -> Any | None:
    if node.type == "function_definition":
        return node
    if node.type == "decorated_definition":
        return first_child(node, "function_definition")
    return None


def _python_decorator_names(content: str, node: Any) -> list[str]:
    if node.type != "decorated_definition":
        return []
    names: list[str] = []
    for decorator in [child for child in node.children if child.type == "decorator"]:
        for child in decorator.children:
            if child.type in {"identifier", "attribute", "call"}:
                names.append(node_text(content, child).strip())
                break
    return [name.lower() for name in names if name]


def _python_param_names(content: str, function_node: Any) -> list[str]:
    parameters = first_child(function_node, "parameters")
    if parameters is None:
        return []
    values: list[str] = []
    for node in parameters.children:
        if node.type == "identifier":
            values.append(node_text(content, node).strip())
        elif node.type == "default_parameter":
            ident = first_child(node, "identifier")
            if ident is not None:
                values.append(node_text(content, ident).strip())
    return _dedupe(values)


def _python_parent_suite(content: str, node: Any) -> str | None:
    parent = node.parent
    while parent is not None:
        if parent.type == "class_definition":
            name = node_text(content, first_child(parent, "identifier")).strip()
            if name:
                return name
            break
        parent = parent.parent
    return None


def _collect_python_case_signals(
    content: str, function_node: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(function_node):
        if node.type == "assert_statement":
            assertions.append(node_text(content, node).strip())
            continue
        if node.type != "call":
            continue
        full_name, base_name, method_name = _python_call_name(content, node)
        callee = method_name or base_name or full_name
        if callee:
            call_sites.append(
                {
                    "line": line_number(node),
                    "column": int(node.start_point[1]),
                    "callee": callee,
                }
            )
        endpoint = _endpoint_from_call(content, node, base_name=base_name, method_name=method_name)
        if endpoint:
            endpoints.append(endpoint)

        symbol_token = method_name or base_name
        if symbol_token and len(symbol_token) >= 3 and symbol_token not in SYMBOL_STOP_WORDS:
            symbols.append(symbol_token)
        call_text = node_text(content, node).strip()
        if "expect(" in call_text or base_name in {"assert", "expect"}:
            assertions.append(call_text)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


def _python_call_name(content: str, call_node: Any) -> tuple[str, str, str]:
    function = call_node.child_by_field_name("function")
    if function is None:
        return "", "", ""
    if function.type == "identifier":
        base = node_text(content, function).strip()
        return base, base, ""
    if function.type == "attribute":
        obj = node_text(content, function.child_by_field_name("object")).strip()
        attr = node_text(content, function.child_by_field_name("attribute")).strip()
        full = f"{obj}.{attr}" if obj and attr else obj or attr
        return full, obj, attr
    token = node_text(content, function).strip()
    return token, token, ""


def _extract_js_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    fixtures_by_scope: dict[str, list[str]] = {}

    def visit(node: Any, suite_stack: list[str]) -> None:
        if node.type != "call_expression":
            for child in node.children:
                visit(child, suite_stack)
            return

        full_name, base_name, method_name = _js_call_name(content, node)
        call_name = method_name or base_name
        lower_call = call_name.lower()

        if lower_call in {name.lower() for name in JS_SUITE_CALLS}:
            suite_name = _first_string_argument(content, node) or f"suite@{line_number(node)}"
            extraction.suites.append(
                TestSuiteDef(
                    name=suite_name,
                    file_path=file_path,
                    line=line_number(node),
                    natural_key=f"test_suite:{file_path}:{suite_name}",
                )
            )
            callback = _js_callback(node)
            if callback is not None:
                visit(callback, suite_stack + [suite_name])
                return

        if lower_call in {name.lower() for name in JS_FIXTURE_CALLS}:
            fixture_name = call_name
            scope_key = suite_stack[-1] if suite_stack else "__global__"
            fixtures_by_scope.setdefault(scope_key, []).append(fixture_name)
            extraction.fixtures.append(
                TestFixtureDef(name=fixture_name, file_path=file_path, line=line_number(node))
            )

        if lower_call in {name.lower() for name in JS_CASE_CALLS}:
            case_name = _first_string_argument(content, node) or f"{call_name}@{line_number(node)}"
            callback = _js_callback(node)
            signal_root = callback or node
            symbol_hints, endpoint_hints, call_sites, raw_assertions = _collect_js_case_signals(
                content, signal_root
            )
            suite_name = suite_stack[-1] if suite_stack else None
            fixture_names = list(fixtures_by_scope.get("__global__", []))
            if suite_name:
                fixture_names.extend(fixtures_by_scope.get(suite_name, []))
            case = TestCaseDef(
                name=case_name,
                file_path=file_path,
                line=line_number(node),
                suite_name=suite_name,
                fixture_names=_dedupe(fixture_names)[:8],
                symbol_hints=symbol_hints[:12],
                endpoint_hints=endpoint_hints[:12],
                call_sites=call_sites[:80],
                raw_assertions=raw_assertions[:8],
            )
            case.natural_key = (
                f"test_case:{file_path}:{case_name}:{_hash(f'{file_path}:{case_name}:{case.line}')}"
            )
            extraction.cases.append(case)

        for child in node.children:
            visit(child, suite_stack)

    visit(root, [])


def _collect_js_case_signals(
    content: str,
    root: Any,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(root):
        if node.type != "call_expression":
            continue
        full_name, base_name, method_name = _js_call_name(content, node)
        callee = method_name or base_name or full_name
        if callee:
            call_sites.append(
                {
                    "line": line_number(node),
                    "column": int(node.start_point[1]),
                    "callee": callee,
                }
            )
        endpoint = _endpoint_from_call(content, node, base_name=base_name, method_name=method_name)
        if endpoint:
            endpoints.append(endpoint)

        symbol_token = method_name or base_name
        if symbol_token and len(symbol_token) >= 3 and symbol_token not in SYMBOL_STOP_WORDS:
            symbols.append(symbol_token)

        call_text = node_text(content, node).strip()
        if "expect(" in call_text or base_name in {"assert", "expect"}:
            assertions.append(call_text)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


def _js_call_name(content: str, call_node: Any) -> tuple[str, str, str]:
    function = call_node.child_by_field_name("function")
    if function is None:
        return "", "", ""
    if function.type == "identifier":
        base = node_text(content, function).strip()
        return base, base, ""
    if function.type == "member_expression":
        obj = node_text(content, function.child_by_field_name("object")).strip()
        prop = node_text(content, function.child_by_field_name("property")).strip()
        full = f"{obj}.{prop}" if obj and prop else obj or prop
        return full, obj, prop
    token = node_text(content, function).strip()
    return token, token, ""


def _js_callback(call_node: Any) -> Any | None:
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    callback: Any | None = None
    for child in args.children:
        if child.type in {"arrow_function", "function_expression"}:
            callback = child
    return callback


def _first_string_argument(content: str, call_node: Any) -> str | None:
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        value = _string_literal(content, child)
        if value:
            return value
    return None


def _endpoint_from_call(
    content: str,
    call_node: Any,
    *,
    base_name: str,
    method_name: str,
) -> str | None:
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None

    first_literal: str | None = None
    for child in args.children:
        literal = _string_literal(content, child)
        if literal:
            first_literal = literal
            break
    if not first_literal:
        return None

    lower_base = base_name.lower()
    lower_method = method_name.lower()
    if lower_base == "fetch":
        return first_literal
    if lower_method in HTTP_METHOD_NAMES and (
        lower_base in HTTP_CLIENT_NAMES or "." in lower_base or lower_base.endswith("client")
    ):
        return first_literal
    return None


def _string_literal(content: str, node: Any) -> str | None:
    if node.type == "string":
        return unquote(node_text(content, node))
    if node.type == "template_string":
        raw = node_text(content, node).strip()
        if "${" in raw:
            return None
        return unquote(raw)
    return None


def _dedupe_suites(values: list[TestSuiteDef]) -> list[TestSuiteDef]:
    seen: set[str] = set()
    deduped: list[TestSuiteDef] = []
    for value in values:
        if value.natural_key in seen:
            continue
        seen.add(value.natural_key)
        deduped.append(value)
    return deduped


def _dedupe_fixtures(values: list[TestFixtureDef]) -> list[TestFixtureDef]:
    seen: set[str] = set()
    deduped: list[TestFixtureDef] = []
    for value in values:
        key = f"{value.file_path}:{value.name}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _dedupe_cases(values: list[TestCaseDef]) -> list[TestCaseDef]:
    seen: set[str] = set()
    deduped: list[TestCaseDef] = []
    for value in values:
        if value.natural_key in seen:
            continue
        seen.add(value.natural_key)
        deduped.append(value)
    return deduped


async def _create_node_evidence(
    session: AsyncSession,
    *,
    node_id: UUID,
    file_path: str,
    start_line: int,
    end_line: int,
    snippet: str | None = None,
) -> str:
    from contextmine_core.models import KnowledgeEvidence

    evidence = KnowledgeEvidence(
        file_path=file_path,
        start_line=max(1, start_line),
        end_line=max(1, end_line),
        snippet=snippet,
    )
    session.add(evidence)
    await session.flush()
    session.add(KnowledgeNodeEvidence(node_id=node_id, evidence_id=evidence.id))
    return str(evidence.id)


async def _upsert_node(
    session: AsyncSession,
    *,
    collection_id: UUID,
    kind: KnowledgeNodeKind,
    natural_key: str,
    name: str,
    meta: dict[str, Any],
) -> UUID:
    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=kind,
        natural_key=natural_key,
        name=name,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={"name": stmt.excluded.name, "meta": stmt.excluded.meta},
    ).returning(KnowledgeNode.id)
    return (await session.execute(stmt)).scalar_one()


async def _upsert_edge(
    session: AsyncSession,
    *,
    collection_id: UUID,
    source_node_id: UUID,
    target_node_id: UUID,
    kind: KnowledgeEdgeKind,
    meta: dict[str, Any],
) -> UUID:
    stmt = pg_insert(KnowledgeEdge).values(
        collection_id=collection_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        meta=meta,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_edge_unique",
        set_={"meta": stmt.excluded.meta},
    ).returning(KnowledgeEdge.id)
    return (await session.execute(stmt)).scalar_one()


async def build_tests_graph(
    session: AsyncSession,
    collection_id: UUID,
    extractions: list[TestsExtraction],
    *,
    source_id: UUID | None = None,
) -> dict[str, int]:
    """Persist extracted test semantics into the knowledge graph."""
    stats = {
        "test_suites": 0,
        "test_cases": 0,
        "test_fixtures": 0,
        "test_edges": 0,
        "test_links_to_symbols": 0,
        "test_links_to_rules": 0,
    }
    if not extractions:
        return stats

    rule_rows = (
        (
            await session.execute(
                select(KnowledgeNode).where(
                    KnowledgeNode.collection_id == collection_id,
                    KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
                )
            )
        )
        .scalars()
        .all()
    )
    rules_by_token: dict[str, UUID] = {}
    for row in rule_rows:
        rules_by_token[row.name.lower()] = row.id
        natural = str((row.meta or {}).get("natural_language") or "").lower()
        if natural:
            rules_by_token[natural] = row.id

    for extraction in extractions:
        suite_ids: dict[str, UUID] = {}
        fixture_ids: dict[str, UUID] = {}

        for suite in extraction.suites:
            node_meta = {
                "file_path": suite.file_path,
                "framework": extraction.framework,
                **_provenance(
                    mode="deterministic",
                    extractor="tests.v1",
                    confidence=0.99,
                ),
            }
            suite_id = await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_SUITE,
                natural_key=suite.natural_key,
                name=suite.name,
                meta=node_meta,
            )
            evidence_id = await _create_node_evidence(
                session,
                node_id=suite_id,
                file_path=suite.file_path,
                start_line=suite.line,
                end_line=suite.line,
                snippet=f"suite {suite.name}",
            )
            suite_ids[suite.name] = suite_id
            stats["test_suites"] += 1

            node_meta["provenance"]["evidence_ids"] = [evidence_id]  # type: ignore[index]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_SUITE,
                natural_key=suite.natural_key,
                name=suite.name,
                meta=node_meta,
            )

        for fixture in extraction.fixtures:
            natural_key = f"test_fixture:{fixture.file_path}:{fixture.name}"
            node_meta = {
                "file_path": fixture.file_path,
                "framework": extraction.framework,
                **_provenance(mode="deterministic", extractor="tests.v1", confidence=0.96),
            }
            fixture_id = await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_FIXTURE,
                natural_key=natural_key,
                name=fixture.name,
                meta=node_meta,
            )
            evidence_id = await _create_node_evidence(
                session,
                node_id=fixture_id,
                file_path=fixture.file_path,
                start_line=fixture.line,
                end_line=fixture.line,
                snippet=f"fixture {fixture.name}",
            )
            fixture_ids[fixture.name] = fixture_id
            stats["test_fixtures"] += 1

            node_meta["provenance"]["evidence_ids"] = [evidence_id]  # type: ignore[index]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_FIXTURE,
                natural_key=natural_key,
                name=fixture.name,
                meta=node_meta,
            )

        for case in extraction.cases:
            inferred_suite = case.suite_name not in suite_ids if case.suite_name else True
            node_meta = {
                "file_path": case.file_path,
                "framework": extraction.framework,
                "suite_name": case.suite_name,
                "fixture_names": case.fixture_names,
                "symbol_hints": case.symbol_hints,
                "endpoint_hints": case.endpoint_hints,
                "call_sites": case.call_sites,
                "assertions": case.raw_assertions,
                **_provenance(
                    mode="inferred" if inferred_suite else "deterministic",
                    extractor="tests.v1",
                    confidence=0.87 if inferred_suite else 0.95,
                ),
            }
            case_id = await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_CASE,
                natural_key=case.natural_key,
                name=case.name,
                meta=node_meta,
            )
            evidence_id = await _create_node_evidence(
                session,
                node_id=case_id,
                file_path=case.file_path,
                start_line=case.line,
                end_line=case.line,
                snippet=f"test {case.name}",
            )
            stats["test_cases"] += 1

            node_meta["provenance"]["evidence_ids"] = [evidence_id]  # type: ignore[index]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_CASE,
                natural_key=case.natural_key,
                name=case.name,
                meta=node_meta,
            )

            for fixture_name in case.fixture_names:
                fixture_id = fixture_ids.get(fixture_name)
                if not fixture_id:
                    continue
                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=case_id,
                    target_node_id=fixture_id,
                    kind=KnowledgeEdgeKind.TEST_USES_FIXTURE,
                    meta=_provenance(
                        mode="deterministic",
                        extractor="tests.v1",
                        confidence=0.93,
                        evidence_ids=[evidence_id],
                    ),
                )
                stats["test_edges"] += 1

            resolved_symbol_refs = await resolve_symbol_refs_for_calls(
                session=session,
                collection_id=collection_id,
                source_id=source_id,
                file_path=case.file_path,
                call_sites=case.call_sites,
                fallback_symbol_hints=case.symbol_hints,
            )
            node_meta["resolved_symbol_refs"] = [
                {
                    "symbol_node_id": str(ref.symbol_node_id),
                    "symbol_name": ref.symbol_name,
                    "engine": ref.engine,
                    "confidence": ref.confidence,
                }
                for ref in resolved_symbol_refs
            ]
            await _upsert_node(
                session,
                collection_id=collection_id,
                kind=KnowledgeNodeKind.TEST_CASE,
                natural_key=case.natural_key,
                name=case.name,
                meta=node_meta,
            )

            for ref in resolved_symbol_refs:
                await _upsert_edge(
                    session,
                    collection_id=collection_id,
                    source_node_id=case_id,
                    target_node_id=ref.symbol_node_id,
                    kind=KnowledgeEdgeKind.TEST_CASE_COVERS_SYMBOL,
                    meta=_provenance(
                        mode="deterministic" if ref.engine.startswith("scip") else "inferred",
                        extractor=f"tests.v1.{ref.engine}",
                        confidence=ref.confidence,
                        evidence_ids=[evidence_id],
                    ),
                )
                stats["test_links_to_symbols"] += 1

            joined = " ".join(case.raw_assertions).lower()
            for token, rule_id in rules_by_token.items():
                if token and token in joined:
                    await _upsert_edge(
                        session,
                        collection_id=collection_id,
                        source_node_id=case_id,
                        target_node_id=rule_id,
                        kind=KnowledgeEdgeKind.TEST_CASE_VALIDATES_RULE,
                        meta=_provenance(
                            mode="inferred",
                            extractor="tests.v1",
                            confidence=0.71,
                            evidence_ids=[evidence_id],
                        ),
                    )
                    stats["test_links_to_rules"] += 1
                    break

    return stats
