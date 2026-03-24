"""Scatter coverage tests targeting many files with small uncovered line ranges.

Files covered:
- twin/evolution.py: pure helpers (_safe_float, _safe_int, _safe_ratio,
  _coverage_value, _normalize_min_max, _file_path_from_natural_key,
  derive_arch_group, build_entity_key, _display_label, _bus_factor, _percentile)
- context.py: FakeLLM.generate, LLM base class generate_stream
- exports/mermaid_c4.py: _safe_id, _safe_text, _limit_nodes_by_degree,
  _build_relation_lines, _normalize_c4_view, _normalize_scope, _kind_value
- analyzer/extractors/surface.py: SurfaceCatalogExtractor._load_openapi_document,
  _is_job_file
- analyzer/extractors/tests.py: looks_like_test_file, detect_test_framework,
  extract_tests_from_file for Python and JS
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# twin/evolution.py pure helpers
# ---------------------------------------------------------------------------


class TestEvolutionHelpers:
    def test_safe_float_none(self) -> None:
        from contextmine_core.twin.evolution import _safe_float

        assert _safe_float(None) == 0.0

    def test_safe_float_string(self) -> None:
        from contextmine_core.twin.evolution import _safe_float

        assert _safe_float("not_a_number") == 0.0

    def test_safe_float_valid(self) -> None:
        from contextmine_core.twin.evolution import _safe_float

        assert _safe_float(3.14) == 3.14

    def test_safe_float_int(self) -> None:
        from contextmine_core.twin.evolution import _safe_float

        assert _safe_float(42) == 42.0

    def test_safe_int_none(self) -> None:
        from contextmine_core.twin.evolution import _safe_int

        assert _safe_int(None) == 0

    def test_safe_int_string(self) -> None:
        from contextmine_core.twin.evolution import _safe_int

        assert _safe_int("not_a_number") == 0

    def test_safe_int_valid(self) -> None:
        from contextmine_core.twin.evolution import _safe_int

        assert _safe_int(42) == 42

    def test_safe_ratio_zero_denominator(self) -> None:
        from contextmine_core.twin.evolution import _safe_ratio

        assert _safe_ratio(10.0, 0.0) == 0.0

    def test_safe_ratio_negative_denominator(self) -> None:
        from contextmine_core.twin.evolution import _safe_ratio

        assert _safe_ratio(10.0, -5.0) == 0.0

    def test_safe_ratio_valid(self) -> None:
        from contextmine_core.twin.evolution import _safe_ratio

        assert _safe_ratio(10.0, 5.0) == 2.0

    def test_coverage_value_none(self) -> None:
        from contextmine_core.twin.evolution import _coverage_value

        assert _coverage_value(None) is None

    def test_coverage_value_clamped_high(self) -> None:
        from contextmine_core.twin.evolution import _coverage_value

        assert _coverage_value(150.0) == 100.0

    def test_coverage_value_clamped_low(self) -> None:
        from contextmine_core.twin.evolution import _coverage_value

        assert _coverage_value(-10.0) == 0.0

    def test_coverage_value_normal(self) -> None:
        from contextmine_core.twin.evolution import _coverage_value

        assert _coverage_value(75.0) == 75.0

    def test_normalize_min_max_equal_range_positive(self) -> None:
        from contextmine_core.twin.evolution import _normalize_min_max

        assert _normalize_min_max(5.0, 3.0, 3.0) == 1.0

    def test_normalize_min_max_equal_range_zero(self) -> None:
        from contextmine_core.twin.evolution import _normalize_min_max

        assert _normalize_min_max(0.0, 3.0, 3.0) == 0.0

    def test_normalize_min_max_normal(self) -> None:
        from contextmine_core.twin.evolution import _normalize_min_max

        assert _normalize_min_max(5.0, 0.0, 10.0) == 0.5

    def test_file_path_from_natural_key_file(self) -> None:
        from contextmine_core.twin.evolution import _file_path_from_natural_key

        result = _file_path_from_natural_key("file:src/main.py")
        assert result is not None
        assert "main.py" in result

    def test_file_path_from_natural_key_not_file(self) -> None:
        from contextmine_core.twin.evolution import _file_path_from_natural_key

        assert _file_path_from_natural_key("symbol:MyClass") is None

    def test_file_path_from_natural_key_empty(self) -> None:
        from contextmine_core.twin.evolution import _file_path_from_natural_key

        assert _file_path_from_natural_key("file:") is None

    def test_build_entity_key_file(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key("src/main.py", None, "file")
        assert result is not None
        assert result.startswith("file:")

    def test_build_entity_key_container(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key("packages/core/contextmine_core/main.py", {}, "container")
        # May or may not resolve depending on heuristic, but should not crash
        assert result is None or result.startswith("container:")

    def test_build_entity_key_component(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key("packages/core/contextmine_core/models.py", {}, "component")
        assert result is None or result.startswith("component:")

    def test_build_entity_key_unknown_level(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key("src/main.py", {}, "unknown")
        assert result is None

    def test_display_label_with_colon(self) -> None:
        from contextmine_core.twin.evolution import _display_label

        assert _display_label("container:myapp/backend") == "myapp/backend"

    def test_display_label_without_colon(self) -> None:
        from contextmine_core.twin.evolution import _display_label

        assert _display_label("simple") == "simple"

    def test_bus_factor_empty(self) -> None:
        from contextmine_core.twin.evolution import _bus_factor

        assert _bus_factor({}) == 0

    def test_bus_factor_single(self) -> None:
        from contextmine_core.twin.evolution import _bus_factor

        assert _bus_factor({"alice": 100.0}) == 1

    def test_bus_factor_even_distribution(self) -> None:
        from contextmine_core.twin.evolution import _bus_factor

        result = _bus_factor({"alice": 50.0, "bob": 50.0})
        assert result == 2

    def test_bus_factor_dominated(self) -> None:
        from contextmine_core.twin.evolution import _bus_factor

        result = _bus_factor({"alice": 90.0, "bob": 10.0})
        assert result == 1

    def test_bus_factor_zero_total(self) -> None:
        from contextmine_core.twin.evolution import _bus_factor

        assert _bus_factor({"alice": 0.0, "bob": 0.0}) == 0

    def test_percentile_empty(self) -> None:
        from contextmine_core.twin.evolution import _percentile

        assert _percentile([], 0.5) == 0.0

    def test_percentile_single(self) -> None:
        from contextmine_core.twin.evolution import _percentile

        assert _percentile([42.0], 0.5) == 42.0

    def test_percentile_multiple(self) -> None:
        from contextmine_core.twin.evolution import _percentile

        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert 2.0 <= result <= 4.0

    def test_percentile_boundaries(self) -> None:
        from contextmine_core.twin.evolution import _percentile

        assert _percentile([1.0, 10.0], 0.0) == 1.0
        assert _percentile([1.0, 10.0], 1.0) == 10.0

    def test_derive_arch_group_none_path(self) -> None:
        from contextmine_core.twin.evolution import derive_arch_group

        assert derive_arch_group(None) is None

    def test_derive_arch_group_valid(self) -> None:
        from contextmine_core.twin.evolution import derive_arch_group

        result = derive_arch_group("packages/core/contextmine_core/models.py")
        # May return None or EntityGroup depending on heuristics
        if result is not None:
            assert result.domain is not None
            assert result.container is not None
            assert result.component is not None


# ---------------------------------------------------------------------------
# context.py: FakeLLM and LLM base
# ---------------------------------------------------------------------------


class TestFakeLLM:
    @pytest.mark.anyio
    async def test_generate_with_query(self) -> None:
        from contextmine_core.context import FakeLLM

        llm = FakeLLM()
        prompt = "## Query\nWhat is testing?\n\n### Chunk 1\nTesting content\n---\n"
        result = await llm.generate("system", prompt, 1000)
        assert "What is testing?" in result
        assert "Sources" in result

    @pytest.mark.anyio
    async def test_generate_without_query(self) -> None:
        from contextmine_core.context import FakeLLM

        llm = FakeLLM()
        result = await llm.generate("system", "No query header here", 1000)
        assert "Response to:" in result

    @pytest.mark.anyio
    async def test_generate_with_chunks(self) -> None:
        from contextmine_core.context import FakeLLM

        llm = FakeLLM()
        prompt = (
            "## Query\nTest\n\n### Chunk 1\nFirst chunk content\n---\n### Chunk 2\nSecond chunk\n"
        )
        result = await llm.generate("system", prompt, 1000)
        assert "Relevant Content" in result
        assert "2 retrieved chunks" in result

    @pytest.mark.anyio
    async def test_generate_with_uris(self) -> None:
        from contextmine_core.context import FakeLLM

        llm = FakeLLM()
        prompt = "## Query\nTest\n\n### Chunk 1 (from: git://repo/file.py)\ncontent\n---\n"
        result = await llm.generate("system", prompt, 1000)
        assert "git://repo/file.py" in result

    @pytest.mark.anyio
    async def test_generate_stream_default(self) -> None:
        from contextmine_core.context import FakeLLM

        llm = FakeLLM()
        chunks = []
        async for chunk in llm.generate_stream("system", "## Query\nTest\n", 1000):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "Test" in chunks[0]


class TestContextDataClasses:
    def test_context_request(self) -> None:
        from contextmine_core.context import ContextRequest, LLMProvider

        req = ContextRequest(query="test", max_chunks=5)
        assert req.query == "test"
        assert req.max_chunks == 5
        assert req.provider == LLMProvider.OPENAI

    def test_context_response(self) -> None:
        from contextmine_core.context import ContextResponse

        resp = ContextResponse(markdown="# Test", query="test", chunks_used=3, sources=[])
        assert resp.markdown == "# Test"
        assert resp.chunks_used == 3


# ---------------------------------------------------------------------------
# exports/mermaid_c4.py: pure helpers
# ---------------------------------------------------------------------------


class TestMermaidC4Helpers:
    def test_safe_id(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _safe_id

        assert _safe_id("my-node:path/to") == "my_node_path_to"

    def test_safe_text_empty(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _safe_text

        assert _safe_text(None, "default") == "default"

    def test_safe_text_quotes(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _safe_text

        assert '"' not in _safe_text('has "quotes"', "")

    def test_safe_text_newlines(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _safe_text

        assert "\n" not in _safe_text("line1\nline2", "")

    def test_safe_text_empty_value(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _safe_text

        assert _safe_text("", "fallback") == "fallback"

    def test_limit_nodes_under_limit(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _limit_nodes_by_degree

        nodes = [{"id": "1"}, {"id": "2"}]
        edges = [{"source_node_id": "1", "target_node_id": "2"}]
        result_nodes, result_edges, truncated = _limit_nodes_by_degree(nodes, edges, 10)
        assert len(result_nodes) == 2
        assert truncated is False

    def test_limit_nodes_truncated(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _limit_nodes_by_degree

        nodes = [{"id": str(i), "name": f"n{i}"} for i in range(10)]
        edges = [{"source_node_id": "0", "target_node_id": str(i)} for i in range(1, 10)]
        result_nodes, result_edges, truncated = _limit_nodes_by_degree(nodes, edges, 3)
        assert len(result_nodes) == 3
        assert truncated is True
        # Node 0 should be kept (highest degree)
        node_ids = {n["id"] for n in result_nodes}
        assert "0" in node_ids

    def test_limit_nodes_zero_max(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _limit_nodes_by_degree

        nodes = [{"id": "1"}]
        edges = []
        result_nodes, _, truncated = _limit_nodes_by_degree(nodes, edges, 0)
        assert len(result_nodes) == 1
        assert truncated is False

    def test_build_relation_lines(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _build_relation_lines

        edges = [
            {"source_node_id": "a", "target_node_id": "b", "kind": "calls"},
            {
                "source_node_id": "c",
                "target_node_id": "d",
                "kind": "imports",
                "meta": {"weight": 5},
            },
        ]
        lines = _build_relation_lines(edges)
        assert len(lines) == 2
        assert "calls" in lines[0]
        assert "w=5" in lines[1]

    def test_build_relation_lines_skips_empty(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _build_relation_lines

        edges = [{"source_node_id": "", "target_node_id": "b", "kind": "calls"}]
        lines = _build_relation_lines(edges)
        assert len(lines) == 0

    def test_normalize_c4_view_valid(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _normalize_c4_view

        assert _normalize_c4_view("container") == "container"
        assert _normalize_c4_view("COMPONENT") == "component"
        assert _normalize_c4_view("code") == "code"

    def test_normalize_c4_view_invalid(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _normalize_c4_view

        assert _normalize_c4_view("invalid") == "container"
        assert _normalize_c4_view(None) == "container"

    def test_normalize_scope(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _normalize_scope

        assert _normalize_scope(None) is None
        assert _normalize_scope("") is None
        assert _normalize_scope("  ") is None
        assert _normalize_scope("backend") == "backend"

    def test_kind_value_enum(self) -> None:
        from types import SimpleNamespace

        from contextmine_core.exports.mermaid_c4 import _kind_value

        assert _kind_value(SimpleNamespace(value="api_endpoint")) == "api_endpoint"

    def test_kind_value_string(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _kind_value

        assert _kind_value("simple_string") == "simple_string"


# ---------------------------------------------------------------------------
# analyzer/extractors/surface.py: extractor methods
# ---------------------------------------------------------------------------


class TestSurfaceCatalogExtractor:
    def test_load_openapi_invalid_yaml(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        result = extractor._load_openapi_document("spec.yaml", "{{invalid yaml")
        assert result is None

    def test_load_openapi_not_dict(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        result = extractor._load_openapi_document("spec.yaml", "- list item\n- another")
        assert result is None

    def test_load_openapi_no_spec_version(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        result = extractor._load_openapi_document("spec.yaml", "paths:\n  /api: {}")
        assert result is None

    def test_load_openapi_valid(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        content = """
openapi: 3.0.0
paths:
  /api/users:
    get:
      summary: List users
"""
        result = extractor._load_openapi_document("spec.yaml", content)
        assert result is not None
        assert "openapi" in result

    def test_load_openapi_non_yaml_extension(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        result = extractor._load_openapi_document("spec.txt", "openapi: 3.0.0\npaths: {}")
        assert result is None

    def test_is_job_file_github_actions(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        assert extractor._is_job_file(".github/workflows/ci.yaml") is True
        assert extractor._is_job_file(".github/workflows/deploy.yml") is True

    def test_is_job_file_k8s_cronjob(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        assert extractor._is_job_file("k8s/cronjob.yaml") is True
        assert extractor._is_job_file("deploy/cronjob.yml") is True

    def test_is_job_file_prefect(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        assert extractor._is_job_file("flows/prefect-deploy.yaml") is True

    def test_is_job_file_not_job(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        assert extractor._is_job_file("src/config.yaml") is False

    def test_add_file_graphql(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        content = """
type Query {
  user(id: ID!): User
}

type User {
  id: ID!
  name: String!
}
"""
        added = extractor.add_file("schema.graphql", content)
        assert added is True
        assert len(extractor.catalog.graphql_schemas) >= 1

    def test_add_file_unrecognized(self) -> None:
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        added = extractor.add_file("readme.md", "# Hello")
        assert added is False


# ---------------------------------------------------------------------------
# analyzer/extractors/tests.py: test extraction
# ---------------------------------------------------------------------------


class TestTestExtractor:
    def test_looks_like_test_file(self) -> None:
        from contextmine_core.analyzer.extractors.tests import looks_like_test_file

        assert looks_like_test_file("test_main.py") is True
        assert looks_like_test_file("main_test.py") is True
        assert looks_like_test_file("Button.spec.tsx") is True
        assert looks_like_test_file("src/__tests__/util.ts") is True
        assert looks_like_test_file("src/main.py") is False

    def test_looks_like_test_file_media_excluded(self) -> None:
        from contextmine_core.analyzer.extractors.tests import looks_like_test_file

        assert looks_like_test_file("test_image.png") is False
        assert looks_like_test_file("test_styles.svg") is False

    def test_detect_framework_pytest(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("test_main.py", "import pytest") == "pytest"

    def test_detect_framework_jest(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("test.js", "// jest test") == "jest"

    def test_detect_framework_vitest(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("test.ts", "import { describe } from 'vitest'") == "vitest"

    def test_detect_framework_cypress(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("login.cy.ts", "// cypress") == "cypress"

    def test_detect_framework_playwright(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("test.ts", "import playwright") == "playwright"

    def test_detect_framework_junit(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("Test.java", "@Test public void test()") == "junit"

    def test_detect_framework_unittest(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("test_main.py", "import unittest") == "unittest"

    def test_detect_framework_unknown_spec_ts(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("component.spec.ts", "plain test code") == "js_test"

    def test_detect_framework_unknown(self) -> None:
        from contextmine_core.analyzer.extractors.tests import detect_test_framework

        assert detect_test_framework("main.py", "plain code") == "unknown"

    def test_extract_python_tests(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import pytest

class TestUser:
    def test_create_user(self):
        user = create_user("alice")
        assert user.name == "alice"

    def test_delete_user(self):
        delete_user("alice")
        assert True

@pytest.fixture
def db_session():
    return mock_session()
"""
        extraction = extract_tests_from_file("tests/test_user.py", code)
        assert extraction.framework == "pytest"
        assert len(extraction.suites) >= 1
        assert any(s.name == "TestUser" for s in extraction.suites)
        assert len(extraction.cases) >= 2
        assert len(extraction.fixtures) >= 1

    def test_extract_js_tests(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import { describe, it, expect } from 'jest';

describe('UserService', () => {
  it('should create user', () => {
    const user = createUser('alice');
    expect(user.name).toBe('alice');
  });

  it('should delete user', () => {
    deleteUser('alice');
    expect(true).toBe(true);
  });
});
"""
        extraction = extract_tests_from_file("tests/user.test.ts", code)
        assert extraction.framework == "jest"
        assert len(extraction.suites) >= 1
        assert len(extraction.cases) >= 2

    def test_extract_tests_from_files(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_files

        files = [
            ("tests/test_a.py", "import pytest\ndef test_one(): assert True\n"),
            ("src/main.py", "def main(): pass\n"),  # Not a test file
            ("tests/test_b.py", ""),  # Empty file
        ]
        results = extract_tests_from_files(files)
        assert len(results) >= 1
        assert all(r.file_path.startswith("tests/test_") for r in results)

    def test_extract_tests_unsupported_language(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        extraction = extract_tests_from_file("test_main.xyz", "some code")
        assert extraction.suites == []
        assert extraction.cases == []
