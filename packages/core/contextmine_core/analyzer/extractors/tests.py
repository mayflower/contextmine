"""Deterministic test extractor and graph materializer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.analyzer.extractors.ast_utils import (
    csharp_attribute_names,
    endpoint_from_call,
    find_enclosing_class_name,
    first_child,
    first_string_argument,
    is_pascal_case,
    java_annotation_names,
    js_call_name,
    line_number,
    node_text,
    ruby_first_string_arg,
    string_literal,
    walk,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    content_hash as _hash,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    create_node_evidence as _create_node_evidence,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    dedupe_strings as _dedupe,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    provenance as _provenance,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    upsert_edge as _upsert_edge,
)
from contextmine_core.analyzer.extractors.graph_helpers import (
    upsert_node as _upsert_node,
)
from contextmine_core.analyzer.extractors.traceability import resolve_symbol_refs_for_calls
from contextmine_core.models import (
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import get_treesitter_manager
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


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
JS_SUITE_CALLS_LOWER = {name.lower() for name in JS_SUITE_CALLS}
JS_CASE_CALLS_LOWER = {name.lower() for name in JS_CASE_CALLS}
JS_FIXTURE_CALLS_LOWER = {name.lower() for name in JS_FIXTURE_CALLS}
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
    if "vitest" in lower:
        return "vitest"
    if "cypress" in lower:
        return "cypress"
    if "playwright" in lower:
        return "playwright"
    # Java
    if "@test" in lower and file_path.endswith(".java"):
        return "junit"
    if "junit" in lower:
        return "junit"
    # Go
    if file_path.endswith("_test.go"):
        return "go_testing"
    # PHP
    if "phpunit" in lower or ("extends testcase" in lower and file_path.endswith(".php")):
        return "phpunit"
    # Ruby
    if "rspec" in lower or ("describe " in lower and file_path.endswith((".rb", "_spec.rb"))):
        return "rspec"
    if "minitest" in lower or "test::unit" in lower:
        return "minitest"
    # C#
    if "[test]" in lower or "[fact]" in lower or "[testmethod]" in lower:
        return "nunit_or_xunit"
    if "nunit" in lower:
        return "nunit"
    if "xunit" in lower:
        return "xunit"
    if file_path.endswith((".spec.ts", ".spec.js", ".test.ts", ".test.js")):
        return "js_test"
    return "unknown"


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
    elif language == TreeSitterLanguage.JAVA:
        _extract_java_tests(file_path, content, root, extraction)
    elif language == TreeSitterLanguage.GO:
        _extract_go_tests(file_path, content, root, extraction)
    elif language == TreeSitterLanguage.PHP:
        _extract_php_tests(file_path, content, root, extraction)
    elif language == TreeSitterLanguage.RUBY:
        _extract_ruby_tests(file_path, content, root, extraction)
    elif language == TreeSitterLanguage.CSHARP:
        _extract_csharp_tests(file_path, content, root, extraction)

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


def _maybe_extract_python_suite(
    file_path: str,
    content: str,
    node: Any,
    extraction: TestsExtraction,
) -> None:
    """Extract a test suite from a class definition node if it looks like a test class."""
    if node.type != "class_definition":
        return
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


def _extract_python_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    for node in walk(root):
        _maybe_extract_python_suite(file_path, content, node, extraction)

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
                TestFixtureDef(name=fn_name, file_path=file_path, line=line_number(function_node))
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


def _process_call_node(
    content: str,
    node: Any,
    symbols: list[str],
    endpoints: list[str],
    call_sites: list[dict[str, Any]],
    assertions: list[str],
) -> None:
    """Process a single call node, extracting signals for test case analysis."""
    full_name, base_name, method_name = _python_call_name(content, node)
    callee = method_name or base_name or full_name
    if callee:
        call_sites.append(
            {"line": line_number(node), "column": int(node.start_point[1]), "callee": callee}
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
        elif node.type == "call":
            _process_call_node(content, node, symbols, endpoints, call_sites, assertions)
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

        _, base_name, method_name = _js_call_name(content, node)
        call_name = method_name or base_name
        lower_call = call_name.lower()

        if _handle_js_suite(
            content,
            file_path,
            node,
            lower_call,
            suite_stack,
            extraction,
            visit,
        ):
            return

        _handle_js_fixture(
            file_path,
            node,
            call_name,
            lower_call,
            suite_stack,
            fixtures_by_scope,
            extraction,
        )
        _handle_js_test_case(
            content,
            file_path,
            node,
            call_name,
            lower_call,
            suite_stack,
            fixtures_by_scope,
            extraction,
        )

        for child in node.children:
            visit(child, suite_stack)

    visit(root, [])


def _handle_js_suite(
    content: str,
    file_path: str,
    node: Any,
    lower_call: str,
    suite_stack: list[str],
    extraction: TestsExtraction,
    visit_fn: Any,
) -> bool:
    """Handle a JS suite call. Returns True if children were visited via callback."""
    if lower_call not in JS_SUITE_CALLS_LOWER:
        return False
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
        visit_fn(callback, suite_stack + [suite_name])
        return True
    return False


def _handle_js_fixture(
    file_path: str,
    node: Any,
    call_name: str,
    lower_call: str,
    suite_stack: list[str],
    fixtures_by_scope: dict[str, list[str]],
    extraction: TestsExtraction,
) -> None:
    """Handle a JS fixture call."""
    if lower_call not in JS_FIXTURE_CALLS_LOWER:
        return
    scope_key = suite_stack[-1] if suite_stack else "__global__"
    fixtures_by_scope.setdefault(scope_key, []).append(call_name)
    extraction.fixtures.append(
        TestFixtureDef(name=call_name, file_path=file_path, line=line_number(node))
    )


def _handle_js_test_case(
    content: str,
    file_path: str,
    node: Any,
    call_name: str,
    lower_call: str,
    suite_stack: list[str],
    fixtures_by_scope: dict[str, list[str]],
    extraction: TestsExtraction,
) -> None:
    """Handle a JS test case call."""
    if lower_call not in JS_CASE_CALLS_LOWER:
        return
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


# _js_call_name, _first_string_argument, _string_literal, _endpoint_from_call
# imported from ast_utils at the top of this module.
_js_call_name = js_call_name
_string_literal = string_literal
_first_string_argument = first_string_argument
_endpoint_from_call = endpoint_from_call


def _js_callback(call_node: Any) -> Any | None:
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    callback: Any | None = None
    for child in args.children:
        if child.type in {"arrow_function", "function_expression"}:
            callback = child
    return callback


# ---------------------------------------------------------------------------
# Java (JUnit 4/5) extraction
# ---------------------------------------------------------------------------

_JAVA_TEST_ANNOTATIONS = {"test", "parameterizedtest", "repeatedtest"}
_JAVA_LIFECYCLE_ANNOTATIONS = {
    "beforeeach",
    "aftereach",
    "beforeall",
    "afterall",
    "before",
    "after",
}


def _extract_java_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    """Extract JUnit 4/5 test classes, methods, and lifecycle hooks."""
    for node in walk(root):
        # Test classes
        if node.type == "class_declaration":
            name_node = first_child(node, "identifier")
            name = node_text(content, name_node).strip() if name_node else ""
            if name and "test" in name.lower():
                extraction.suites.append(
                    TestSuiteDef(
                        name=name,
                        file_path=file_path,
                        line=line_number(node),
                        natural_key=f"test_suite:{file_path}:{name}",
                    )
                )

        # Test methods and lifecycle hooks
        if node.type != "method_declaration":
            continue
        annotations = java_annotation_names(content, node)
        fn_name_node = first_child(node, "identifier")
        fn_name = node_text(content, fn_name_node).strip() if fn_name_node else ""
        if not fn_name:
            continue

        if any(ann in _JAVA_LIFECYCLE_ANNOTATIONS for ann in annotations):
            extraction.fixtures.append(
                TestFixtureDef(name=fn_name, file_path=file_path, line=line_number(node))
            )

        is_test = any(ann in _JAVA_TEST_ANNOTATIONS for ann in annotations)
        if not is_test and fn_name.startswith("test"):
            is_test = True  # JUnit 3 convention
        if not is_test:
            continue

        suite_name = _java_parent_class(content, node)
        symbol_hints, endpoint_hints, call_sites, raw_assertions = _collect_java_case_signals(
            content, node
        )
        case = TestCaseDef(
            name=fn_name,
            file_path=file_path,
            line=line_number(node),
            suite_name=suite_name,
            symbol_hints=symbol_hints[:12],
            endpoint_hints=endpoint_hints[:12],
            call_sites=call_sites[:80],
            raw_assertions=raw_assertions[:8],
        )
        case.natural_key = (
            f"test_case:{file_path}:{fn_name}:{_hash(f'{file_path}:{fn_name}:{case.line}')}"
        )
        extraction.cases.append(case)


def _java_parent_class(content: str, node: Any) -> str | None:
    return find_enclosing_class_name(content, node)


def _collect_java_case_signals(
    content: str, method_node: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(method_node):
        if node.type == "method_invocation":
            name_node = first_child(node, "identifier")
            callee = node_text(content, name_node).strip() if name_node else ""
            if callee:
                call_sites.append(
                    {
                        "line": line_number(node),
                        "column": int(node.start_point[1]),
                        "callee": callee,
                    }
                )
                if callee.startswith("assert") or callee == "assertEquals":
                    assertions.append(node_text(content, node).strip())
                elif len(callee) >= 3 and callee not in SYMBOL_STOP_WORDS:
                    symbols.append(callee)
            endpoint = endpoint_from_call(content, node, base_name=callee, method_name="")
            if endpoint:
                endpoints.append(endpoint)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


# ---------------------------------------------------------------------------
# Go (testing package) extraction
# ---------------------------------------------------------------------------


def _extract_go_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    """Extract Go test/benchmark/fuzz functions and TestMain."""
    for node in walk(root):
        if node.type != "function_declaration":
            continue
        fn_name_node = first_child(node, "identifier")
        fn_name = node_text(content, fn_name_node).strip() if fn_name_node else ""
        if not fn_name:
            continue

        if fn_name == "TestMain":
            extraction.fixtures.append(
                TestFixtureDef(name=fn_name, file_path=file_path, line=line_number(node))
            )
            continue

        is_test = fn_name.startswith("Test") and len(fn_name) > 4 and fn_name[4].isupper()
        is_bench = fn_name.startswith("Benchmark") and len(fn_name) > 9
        is_fuzz = fn_name.startswith("Fuzz") and len(fn_name) > 4 and fn_name[4].isupper()
        if not (is_test or is_bench or is_fuzz):
            continue

        # Check for t.Run subtests
        _extract_go_subtests(file_path, content, node, fn_name, extraction)

        symbol_hints, endpoint_hints, call_sites, raw_assertions = _collect_go_case_signals(
            content, node
        )
        case = TestCaseDef(
            name=fn_name,
            file_path=file_path,
            line=line_number(node),
            suite_name=None,
            symbol_hints=symbol_hints[:12],
            endpoint_hints=endpoint_hints[:12],
            call_sites=call_sites[:80],
            raw_assertions=raw_assertions[:8],
        )
        case.natural_key = (
            f"test_case:{file_path}:{fn_name}:{_hash(f'{file_path}:{fn_name}:{case.line}')}"
        )
        extraction.cases.append(case)


def _extract_go_subtests(
    file_path: str,
    content: str,
    func_node: Any,
    parent_name: str,
    extraction: TestsExtraction,
) -> None:
    """Extract t.Run("name", ...) subtests."""
    for node in walk(func_node):
        if node.type != "call_expression":
            continue
        func = node.child_by_field_name("function")
        if func is None or func.type != "selector_expression":
            continue
        field = func.child_by_field_name("field")
        if field is None or node_text(content, field).strip() != "Run":
            continue
        args = node.child_by_field_name("arguments")
        if args is None:
            continue
        subtest_name = None
        for child in args.children:
            if child.type == "interpreted_string_literal":
                subtest_name = node_text(content, child).strip().strip('"')
                break
        if subtest_name:
            full_name = f"{parent_name}/{subtest_name}"
            case = TestCaseDef(
                name=full_name,
                file_path=file_path,
                line=line_number(node),
                suite_name=parent_name,
            )
            case.natural_key = (
                f"test_case:{file_path}:{full_name}:{_hash(f'{file_path}:{full_name}:{case.line}')}"
            )
            extraction.cases.append(case)


def _collect_go_case_signals(
    content: str, func_node: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(func_node):
        if node.type == "call_expression":
            func = node.child_by_field_name("function")
            if func is None:
                continue
            callee = node_text(content, func).strip()
            short = callee.rsplit(".", 1)[-1] if "." in callee else callee
            call_sites.append(
                {"line": line_number(node), "column": int(node.start_point[1]), "callee": short}
            )
            if short in {
                "Equal",
                "NotEqual",
                "True",
                "False",
                "Nil",
                "NotNil",
                "Error",
                "NoError",
                "Contains",
                "Errorf",
                "Fatalf",
                "Assert",
                "Require",
            }:
                assertions.append(node_text(content, node).strip())
            elif len(short) >= 3 and short not in SYMBOL_STOP_WORDS:
                symbols.append(short)
            endpoint = endpoint_from_call(content, node, base_name=short, method_name="")
            if endpoint:
                endpoints.append(endpoint)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


# ---------------------------------------------------------------------------
# PHP (PHPUnit) extraction
# ---------------------------------------------------------------------------

_PHP_TEST_ANNOTATIONS_RE = ("@test",)


def _extract_php_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    """Extract PHPUnit test classes, methods, setUp/tearDown."""
    for node in walk(root):
        # Test classes (extends TestCase)
        if node.type == "class_declaration":
            name_node = first_child(node, "name")
            if name_node is None:
                name_node = first_child(node, "identifier")
            name = node_text(content, name_node).strip() if name_node else ""
            if name and ("test" in name.lower() or _php_extends_testcase(content, node)):
                extraction.suites.append(
                    TestSuiteDef(
                        name=name,
                        file_path=file_path,
                        line=line_number(node),
                        natural_key=f"test_suite:{file_path}:{name}",
                    )
                )

        if node.type != "method_declaration":
            continue
        fn_name_node = first_child(node, "name")
        if fn_name_node is None:
            fn_name_node = first_child(node, "identifier")
        fn_name = node_text(content, fn_name_node).strip() if fn_name_node else ""
        if not fn_name:
            continue

        # Lifecycle hooks
        if fn_name in {"setUp", "tearDown", "setUpBeforeClass", "tearDownAfterClass"}:
            extraction.fixtures.append(
                TestFixtureDef(name=fn_name, file_path=file_path, line=line_number(node))
            )
            continue

        # Test methods: prefixed with test or annotated with @test
        is_test = fn_name.startswith("test")
        if not is_test:
            # Check doc comment for @test annotation
            prev = node.prev_sibling
            if prev and prev.type == "comment":
                comment_text = node_text(content, prev).strip()
                is_test = "@test" in comment_text
        if not is_test:
            continue

        suite_name = _php_parent_class(content, node)
        symbol_hints, endpoint_hints, call_sites, raw_assertions = _collect_php_case_signals(
            content, node
        )
        case = TestCaseDef(
            name=fn_name,
            file_path=file_path,
            line=line_number(node),
            suite_name=suite_name,
            symbol_hints=symbol_hints[:12],
            endpoint_hints=endpoint_hints[:12],
            call_sites=call_sites[:80],
            raw_assertions=raw_assertions[:8],
        )
        case.natural_key = (
            f"test_case:{file_path}:{fn_name}:{_hash(f'{file_path}:{fn_name}:{case.line}')}"
        )
        extraction.cases.append(case)


def _php_extends_testcase(content: str, class_node: Any) -> bool:
    for child in class_node.children:
        if child.type == "base_clause" or child.type == "class_interface_clause":
            text = node_text(content, child).lower()
            if "testcase" in text or "phpunit" in text:
                return True
    return False


def _php_parent_class(content: str, node: Any) -> str | None:
    return find_enclosing_class_name(content, node, name_fields=("name", "identifier"))


_PHP_ASSERT_PREFIXES = ("assert", "expect")


def _collect_php_case_signals(
    content: str, method_node: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    """Collect call-site signals from a PHPUnit test method."""
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(method_node):
        # PHP method calls: $this->method(), $obj->method()
        if node.type == "member_call_expression":
            name_node = node.child_by_field_name("name")
            callee = node_text(content, name_node).strip() if name_node else ""
            if callee:
                call_sites.append(
                    {
                        "line": line_number(node),
                        "column": int(node.start_point[1]),
                        "callee": callee,
                    }
                )
                if callee.lower().startswith(_PHP_ASSERT_PREFIXES):
                    assertions.append(node_text(content, node).strip())
                elif len(callee) >= 3 and callee not in SYMBOL_STOP_WORDS:
                    symbols.append(callee)
        # PHP static calls: ClassName::method()
        elif node.type == "scoped_call_expression":
            name_node = node.child_by_field_name("name")
            callee = node_text(content, name_node).strip() if name_node else ""
            if callee:
                call_sites.append(
                    {
                        "line": line_number(node),
                        "column": int(node.start_point[1]),
                        "callee": callee,
                    }
                )
                if len(callee) >= 3 and callee not in SYMBOL_STOP_WORDS:
                    symbols.append(callee)
        # PHP function calls: function_name()
        elif node.type == "function_call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node is None:
                for child in node.children:
                    if child.type in {"name", "identifier"}:
                        fn_node = child
                        break
            callee = node_text(content, fn_node).strip() if fn_node else ""
            if callee:
                call_sites.append(
                    {
                        "line": line_number(node),
                        "column": int(node.start_point[1]),
                        "callee": callee,
                    }
                )
                if len(callee) >= 3 and callee not in SYMBOL_STOP_WORDS:
                    symbols.append(callee)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


# ---------------------------------------------------------------------------
# Ruby (RSpec / Minitest) extraction
# ---------------------------------------------------------------------------


def _extract_ruby_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    """Extract RSpec (describe/it) and Minitest (test_*) patterns in a single traversal."""
    for node in walk(root):
        # RSpec call patterns
        if node.type == "call":
            method_node = first_child(node, "identifier")
            if method_node is None:
                method_node = node.child_by_field_name("method")
            method_name = node_text(content, method_node).strip() if method_node else ""

            if method_name in {"describe", "context"}:
                desc = _ruby_first_string_arg(content, node) or f"suite@{line_number(node)}"
                extraction.suites.append(
                    TestSuiteDef(
                        name=desc,
                        file_path=file_path,
                        line=line_number(node),
                        natural_key=f"test_suite:{file_path}:{desc}",
                    )
                )
            elif method_name in {"it", "specify", "example"}:
                desc = _ruby_first_string_arg(content, node) or f"test@{line_number(node)}"
                symbol_hints, endpoint_hints, call_sites, raw_assertions = (
                    _collect_ruby_case_signals(content, node)
                )
                case = TestCaseDef(
                    name=desc,
                    file_path=file_path,
                    line=line_number(node),
                    symbol_hints=symbol_hints[:12],
                    endpoint_hints=endpoint_hints[:12],
                    call_sites=call_sites[:80],
                    raw_assertions=raw_assertions[:8],
                )
                case.natural_key = (
                    f"test_case:{file_path}:{desc}:{_hash(f'{file_path}:{desc}:{case.line}')}"
                )
                extraction.cases.append(case)
            elif method_name in {"before", "after", "let", "let!", "subject"}:
                extraction.fixtures.append(
                    TestFixtureDef(name=method_name, file_path=file_path, line=line_number(node))
                )

        # Minitest class patterns
        elif node.type == "class" and _ruby_extends_minitest(content, node):
            name_node = node.child_by_field_name("name") or first_child(node, "constant")
            name = node_text(content, name_node).strip() if name_node else ""
            if name:
                extraction.suites.append(
                    TestSuiteDef(
                        name=name,
                        file_path=file_path,
                        line=line_number(node),
                        natural_key=f"test_suite:{file_path}:{name}",
                    )
                )

        # Minitest method patterns
        elif node.type in {"method", "singleton_method"}:
            fn_name_node = node.child_by_field_name("name") or first_child(node, "identifier")
            fn_name = node_text(content, fn_name_node).strip() if fn_name_node else ""
            if fn_name.startswith("test_"):
                symbol_hints, endpoint_hints, call_sites, raw_assertions = (
                    _collect_ruby_case_signals(content, node)
                )
                case = TestCaseDef(
                    name=fn_name,
                    file_path=file_path,
                    line=line_number(node),
                    symbol_hints=symbol_hints[:12],
                    endpoint_hints=endpoint_hints[:12],
                    call_sites=call_sites[:80],
                    raw_assertions=raw_assertions[:8],
                )
                case.natural_key = (
                    f"test_case:{file_path}:{fn_name}:{_hash(f'{file_path}:{fn_name}:{case.line}')}"
                )
                extraction.cases.append(case)
            elif fn_name in {"setup", "teardown"}:
                extraction.fixtures.append(
                    TestFixtureDef(name=fn_name, file_path=file_path, line=line_number(node))
                )


def _ruby_extends_minitest(content: str, class_node: Any) -> bool:
    superclass = class_node.child_by_field_name("superclass")
    if superclass:
        text = node_text(content, superclass)
        if "Minitest" in text or "Test::Unit" in text:
            return True
    return False


def _ruby_first_string_arg(content: str, call_node: Any) -> str | None:
    return ruby_first_string_arg(content, call_node)


def _collect_ruby_case_signals(
    content: str, block_node: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    """Collect call-site signals from an RSpec it/specify block or Minitest test method."""
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(block_node):
        if node.type != "call":
            continue
        method_node = first_child(node, "identifier")
        if method_node is None:
            method_node = node.child_by_field_name("method")
        callee = node_text(content, method_node).strip() if method_node else ""
        if not callee:
            continue
        call_sites.append(
            {"line": line_number(node), "column": int(node.start_point[1]), "callee": callee}
        )
        callee_lower = callee.lower()
        # RSpec: expect(...).to, assert_equal, assert_raises, etc.
        if callee_lower.startswith("assert") or callee in {"expect", "should", "must"}:
            assertions.append(node_text(content, node).strip())
        elif len(callee) >= 3 and callee not in SYMBOL_STOP_WORDS:
            symbols.append(callee)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


# ---------------------------------------------------------------------------
# C# (NUnit / xUnit / MSTest) extraction
# ---------------------------------------------------------------------------

_CSHARP_TEST_ATTRS = {"test", "testmethod", "fact", "theory", "testcase"}
_CSHARP_LIFECYCLE_ATTRS = {
    "setup",
    "teardown",
    "onetimesetup",
    "onetimeteardown",
    "testinitialize",
    "testcleanup",
    "classinitialize",
    "classcleanup",
}


def _extract_csharp_tests(
    file_path: str,
    content: str,
    root: Any,
    extraction: TestsExtraction,
) -> None:
    """Extract NUnit/xUnit/MSTest test classes, methods, and lifecycle hooks."""
    for node in walk(root):
        # Test classes
        if node.type == "class_declaration":
            name_node = first_child(node, "identifier")
            name = node_text(content, name_node).strip() if name_node else ""
            attrs = _csharp_attributes(content, node)
            if name and ("test" in name.lower() or "testfixture" in attrs or "testclass" in attrs):
                extraction.suites.append(
                    TestSuiteDef(
                        name=name,
                        file_path=file_path,
                        line=line_number(node),
                        natural_key=f"test_suite:{file_path}:{name}",
                    )
                )

        if node.type != "method_declaration":
            continue
        fn_name_node = first_child(node, "identifier")
        fn_name = node_text(content, fn_name_node).strip() if fn_name_node else ""
        if not fn_name:
            continue

        attrs = _csharp_attributes(content, node)

        if attrs & _CSHARP_LIFECYCLE_ATTRS:
            extraction.fixtures.append(
                TestFixtureDef(name=fn_name, file_path=file_path, line=line_number(node))
            )
            continue

        if attrs & _CSHARP_TEST_ATTRS:
            suite_name = _csharp_parent_class(content, node)
            symbol_hints, endpoint_hints, call_sites, raw_assertions = _collect_csharp_case_signals(
                content, node
            )
            case = TestCaseDef(
                name=fn_name,
                file_path=file_path,
                line=line_number(node),
                suite_name=suite_name,
                symbol_hints=symbol_hints[:12],
                endpoint_hints=endpoint_hints[:12],
                call_sites=call_sites[:80],
                raw_assertions=raw_assertions[:8],
            )
            case.natural_key = (
                f"test_case:{file_path}:{fn_name}:{_hash(f'{file_path}:{fn_name}:{case.line}')}"
            )
            extraction.cases.append(case)


def _csharp_attributes(content: str, node: Any) -> set[str]:
    return csharp_attribute_names(content, node)


def _collect_csharp_case_signals(
    content: str, method_node: Any
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    """Collect call-site signals from an NUnit/xUnit/MSTest test method."""
    symbols: list[str] = []
    endpoints: list[str] = []
    call_sites: list[dict[str, Any]] = []
    assertions: list[str] = []
    for node in walk(method_node):
        if node.type != "invocation_expression":
            continue
        func = node.child_by_field_name("function")
        if func is None:
            continue
        # member_access_expression: obj.Method() or Assert.Equal()
        if func.type == "member_access_expression":
            name_node = func.child_by_field_name("name")
            callee = node_text(content, name_node).strip() if name_node else ""
            obj_node = func.child_by_field_name("expression")
            obj_name = node_text(content, obj_node).strip() if obj_node else ""
        elif func.type == "identifier":
            callee = node_text(content, func).strip()
            obj_name = ""
        else:
            callee = node_text(content, func).strip().rsplit(".", 1)[-1]
            obj_name = ""
        if not callee:
            continue
        call_sites.append(
            {"line": line_number(node), "column": int(node.start_point[1]), "callee": callee}
        )
        # Assert.Equal, Assert.True, Assert.Throws, ClassicAssert.*, etc.
        if (
            obj_name in {"Assert", "ClassicAssert", "CollectionAssert", "StringAssert"}
            or callee.startswith("Assert")
            or callee.startswith("Verify")
        ):
            assertions.append(node_text(content, node).strip())
        elif len(callee) >= 3 and callee not in SYMBOL_STOP_WORDS:
            symbols.append(callee)
    return _dedupe(symbols), _dedupe(endpoints), call_sites, _dedupe(assertions)


def _csharp_parent_class(content: str, node: Any) -> str | None:
    return find_enclosing_class_name(content, node)


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
        suite_ids = await _persist_suites(session, collection_id, extraction, stats)
        fixture_ids = await _persist_fixtures(session, collection_id, extraction, stats)
        await _persist_cases(
            session,
            collection_id,
            extraction,
            source_id,
            suite_ids,
            fixture_ids,
            rules_by_token,
            stats,
        )

    return stats


async def _persist_suites(
    session: AsyncSession,
    collection_id: UUID,
    extraction: TestsExtraction,
    stats: dict[str, int],
) -> dict[str, UUID]:
    """Persist test suite nodes. Returns a mapping of suite name to node ID."""
    suite_ids: dict[str, UUID] = {}

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

        node_meta["provenance"]["evidence_ids"] = [evidence_id]
        await _upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.TEST_SUITE,
            natural_key=suite.natural_key,
            name=suite.name,
            meta=node_meta,
        )

    return suite_ids


async def _persist_fixtures(
    session: AsyncSession,
    collection_id: UUID,
    extraction: TestsExtraction,
    stats: dict[str, int],
) -> dict[str, UUID]:
    """Persist test fixture nodes. Returns a mapping of fixture name to node ID."""
    fixture_ids: dict[str, UUID] = {}

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

        node_meta["provenance"]["evidence_ids"] = [evidence_id]
        await _upsert_node(
            session,
            collection_id=collection_id,
            kind=KnowledgeNodeKind.TEST_FIXTURE,
            natural_key=natural_key,
            name=fixture.name,
            meta=node_meta,
        )

    return fixture_ids


async def _persist_cases(
    session: AsyncSession,
    collection_id: UUID,
    extraction: TestsExtraction,
    source_id: UUID | None,
    suite_ids: dict[str, UUID],
    fixture_ids: dict[str, UUID],
    rules_by_token: dict[str, UUID],
    stats: dict[str, int],
) -> None:
    """Persist test case nodes, fixture edges, symbol resolution, and rule matching."""
    for case in extraction.cases:
        case_id, evidence_id, node_meta = await _persist_single_case(
            session,
            collection_id,
            extraction,
            case,
            suite_ids,
            stats,
        )
        await _link_case_fixtures(
            session,
            collection_id,
            case_id,
            case,
            fixture_ids,
            evidence_id,
            stats,
        )
        await _resolve_and_link_symbols(
            session,
            collection_id,
            source_id,
            case_id,
            case,
            node_meta,
            evidence_id,
            stats,
        )
        await _persist_cases_rule_edges(
            session,
            collection_id,
            case_id,
            case,
            rules_by_token,
            evidence_id,
            stats,
        )


async def _persist_single_case(
    session: AsyncSession,
    collection_id: UUID,
    extraction: TestsExtraction,
    case: TestCaseDef,
    suite_ids: dict[str, UUID],
    stats: dict[str, int],
) -> tuple[UUID, str, dict[str, Any]]:
    """Create a test case node with evidence and return (case_id, evidence_id, node_meta)."""
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
    node_meta["provenance"]["evidence_ids"] = [evidence_id]
    await _upsert_node(
        session,
        collection_id=collection_id,
        kind=KnowledgeNodeKind.TEST_CASE,
        natural_key=case.natural_key,
        name=case.name,
        meta=node_meta,
    )
    return case_id, evidence_id, node_meta


async def _link_case_fixtures(
    session: AsyncSession,
    collection_id: UUID,
    case_id: UUID,
    case: TestCaseDef,
    fixture_ids: dict[str, UUID],
    evidence_id: str,
    stats: dict[str, int],
) -> None:
    """Link test case to its fixtures."""
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


async def _resolve_and_link_symbols(
    session: AsyncSession,
    collection_id: UUID,
    source_id: UUID | None,
    case_id: UUID,
    case: TestCaseDef,
    node_meta: dict[str, Any],
    evidence_id: str,
    stats: dict[str, int],
) -> None:
    """Resolve symbol references and create edges."""
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


async def _persist_cases_rule_edges(
    session: AsyncSession,
    collection_id: UUID,
    case_id: UUID,
    case: TestCaseDef,
    rules_by_token: dict[str, UUID],
    evidence_id: str,
    stats: dict[str, int],
) -> None:
    """Match assertions against business rules and create edges."""
    joined = " ".join(case.raw_assertions).lower()
    for token, rule_id in rules_by_token.items():
        if not token or token not in joined:
            continue
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
