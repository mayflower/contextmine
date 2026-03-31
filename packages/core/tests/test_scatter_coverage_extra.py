"""Additional scatter coverage tests to push past 80%.

Targets:
- architecture/facts.py: _evidence_from_symbol_meta, _metric_averages, _dedupe_ports
- context.py: ContextRequest, ContextResponse, GeminiLLM init
- twin/evolution.py: _safe_float edge cases, derive_arch_group edge cases
- analyzer/extractors/schema.py: _strip_sql_identifier, _split_sql_items edge cases
- twin/ops.py: pure helpers
- analyzer/extractors/surface.py: more branches
- semantic_snapshot/scip.py: import coverage
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# architecture/facts.py
# ---------------------------------------------------------------------------


class TestArchitectureFactsHelpers:
    def test_evidence_from_symbol_meta_no_file(self) -> None:
        from contextmine_core.architecture.facts import _evidence_from_symbol_meta

        node = MagicMock()
        node.meta = {}
        result = _evidence_from_symbol_meta(node)
        assert result == ()

    def test_evidence_from_symbol_meta_with_file(self) -> None:
        from contextmine_core.architecture.facts import _evidence_from_symbol_meta

        node = MagicMock()
        node.meta = {"file_path": "src/main.py", "start_line": 10, "end_line": 20}
        result = _evidence_from_symbol_meta(node)
        assert len(result) == 1
        assert result[0].ref == "src/main.py"
        assert result[0].start_line == 10
        assert result[0].end_line == 20

    def test_evidence_from_symbol_meta_none_meta(self) -> None:
        from contextmine_core.architecture.facts import _evidence_from_symbol_meta

        node = MagicMock()
        node.meta = None
        result = _evidence_from_symbol_meta(node)
        assert result == ()

    def test_metric_averages_empty(self) -> None:
        from contextmine_core.architecture.facts import _metric_averages

        result = _metric_averages([])
        assert result["coverage_avg"] is None
        assert result["complexity_avg"] is None
        assert result["coupling_avg"] is None
        assert result["change_frequency_avg"] is None

    def test_metric_averages_with_data(self) -> None:
        from contextmine_core.architecture.facts import _metric_averages

        m1 = SimpleNamespace(coverage=80.0, complexity=10.0, coupling=5.0, change_frequency=2.0)
        m2 = SimpleNamespace(coverage=60.0, complexity=20.0, coupling=15.0, change_frequency=4.0)
        result = _metric_averages([m1, m2])
        assert result["coverage_avg"] == 70.0
        assert result["complexity_avg"] == 15.0
        assert result["coupling_avg"] == 10.0
        assert result["change_frequency_avg"] == 3.0

    def test_metric_averages_with_nones(self) -> None:
        from contextmine_core.architecture.facts import _metric_averages

        m1 = SimpleNamespace(coverage=None, complexity=None, coupling=None, change_frequency=None)
        result = _metric_averages([m1])
        assert result["coverage_avg"] == 0.0

    def _make_port(self, fact_id="port:1", confidence=0.9, evidence=(), **kwargs):
        from contextmine_core.architecture.schemas import PortAdapterFact

        defaults = dict(
            fact_id=fact_id,
            direction="inbound",
            port_name="api",
            adapter_name="handler",
            container="backend",
            component="api",
            protocol="http",
            source="deterministic",
            confidence=confidence,
            evidence=evidence,
        )
        defaults.update(kwargs)
        return PortAdapterFact(**defaults)

    def test_dedupe_ports_no_duplicates(self) -> None:
        from contextmine_core.architecture.facts import _dedupe_ports

        facts = [
            self._make_port(fact_id="port:1"),
            self._make_port(fact_id="port:2", direction="outbound", port_name="db"),
        ]
        result = _dedupe_ports(facts)
        assert len(result) == 2

    def test_dedupe_ports_higher_confidence_wins(self) -> None:
        from contextmine_core.architecture.facts import _dedupe_ports

        facts = [
            self._make_port(fact_id="port:1", confidence=0.75, source="hybrid"),
            self._make_port(fact_id="port:1", confidence=0.9, source="deterministic"),
        ]
        result = _dedupe_ports(facts)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_dedupe_ports_more_evidence_wins(self) -> None:
        from contextmine_core.architecture.facts import _dedupe_ports
        from contextmine_core.architecture.schemas import EvidenceRef

        e1 = EvidenceRef(kind="file", ref="a.py")
        e2 = EvidenceRef(kind="file", ref="b.py")
        facts = [
            self._make_port(fact_id="port:1", confidence=0.9, evidence=(e1,)),
            self._make_port(fact_id="port:1", confidence=0.9, evidence=(e1, e2)),
        ]
        result = _dedupe_ports(facts)
        assert len(result) == 1
        assert len(result[0].evidence) == 2


# ---------------------------------------------------------------------------
# context.py: GeminiLLM, ContextRequest edge cases
# ---------------------------------------------------------------------------


class TestContextEdgeCases:
    def test_llm_provider_enum(self) -> None:
        from contextmine_core.context import LLMProvider

        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GEMINI.value == "gemini"

    def test_context_request_defaults(self) -> None:
        from contextmine_core.context import ContextRequest

        req = ContextRequest(query="test")
        assert req.collection_id is None
        assert req.max_chunks == 10
        assert req.max_tokens == 4000

    def test_context_request_custom(self) -> None:
        from contextmine_core.context import ContextRequest, LLMProvider

        req = ContextRequest(
            query="test",
            collection_id="abc",
            max_chunks=5,
            max_tokens=2000,
            provider=LLMProvider.ANTHROPIC,
            model="claude-3",
        )
        assert req.model == "claude-3"
        assert req.provider == LLMProvider.ANTHROPIC

    @pytest.mark.anyio
    async def test_fake_llm_multiple_chunks(self) -> None:
        from contextmine_core.context import FakeLLM

        llm = FakeLLM()
        prompt = (
            "## Query\nArchitecture overview\n\n"
            "### Chunk 1 (from: git://repo/main.py)\nMain entry point\n---\n"
            "### Chunk 2 (from: git://repo/models.py)\nData models\n---\n"
            "### Chunk 3\nMiscellaneous\n"
        )
        result = await llm.generate("system", prompt, 2000)
        assert "Architecture overview" in result
        assert "git://repo/main.py" in result
        assert "git://repo/models.py" in result


# ---------------------------------------------------------------------------
# twin/evolution.py: additional edge cases
# ---------------------------------------------------------------------------


class TestEvolutionAdditional:
    def test_safe_float_type_error(self) -> None:
        from contextmine_core.twin.evolution import _safe_float

        assert _safe_float(object()) == 0.0

    def test_safe_int_type_error(self) -> None:
        from contextmine_core.twin.evolution import _safe_int

        assert _safe_int(object()) == 0

    def test_safe_int_float_input(self) -> None:
        from contextmine_core.twin.evolution import _safe_int

        assert _safe_int(3.7) == 3

    def test_safe_float_string_input(self) -> None:
        from contextmine_core.twin.evolution import _safe_float

        assert _safe_float("3.14") == 3.14

    def test_normalize_min_max_inverted_range(self) -> None:
        from contextmine_core.twin.evolution import _normalize_min_max

        # max < min
        assert _normalize_min_max(5.0, 10.0, 3.0) == 1.0
        assert _normalize_min_max(0.0, 10.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# analyzer/extractors/schema.py: edge cases in SQL parsing
# ---------------------------------------------------------------------------


class TestSchemaEdgeCases:
    def test_split_sql_items_backtick_string(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _split_sql_items

        result = _split_sql_items("id INT, name VARCHAR DEFAULT `hello, world`")
        assert len(result) == 2

    def test_strip_identifier_with_dots(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _strip_sql_identifier

        assert _strip_sql_identifier('"public"."users"') == "public.users"

    def test_normalize_sql_type_datetime(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _normalize_sql_type

        assert _normalize_sql_type("datetime") == "DateTime"
        assert _normalize_sql_type("timestamp without time zone") == "DateTime"

    def test_extract_ddl_with_inline_references(self) -> None:
        from contextmine_core.analyzer.extractors.schema import _extract_schema_from_sql_ddl

        sql = """
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    author_id INTEGER REFERENCES users(id),
    title TEXT NOT NULL,
    body TEXT
);
"""
        extraction = _extract_schema_from_sql_ddl("schema.sql", sql)
        assert len(extraction.tables) == 1
        table = extraction.tables[0]
        # author_id should have foreign_key set
        author_col = next((c for c in table.columns if c.name == "author_id"), None)
        assert author_col is not None
        assert author_col.foreign_key == "users.id"


# ---------------------------------------------------------------------------
# analyzer/extractors/surface.py: GraphQL extraction
# ---------------------------------------------------------------------------


class TestSurfaceGraphQLExtraction:
    def test_graphql_with_mutations(self) -> None:
        from contextmine_core.analyzer.extractors.graphql import extract_from_graphql

        content = """
type Query {
  users: [User!]!
}

type Mutation {
  createUser(name: String!): User!
  deleteUser(id: ID!): Boolean!
}

type User {
  id: ID!
  name: String!
  email: String
}

type Subscription {
  userCreated: User!
}
"""
        result = extract_from_graphql("schema.graphql", content)
        assert len(result.types) >= 1
        op_names = [o.name for o in result.operations]
        assert any("createUser" in n for n in op_names) or len(result.operations) >= 1


# ---------------------------------------------------------------------------
# OpenAPI: more endpoint parsing branches
# ---------------------------------------------------------------------------


class TestOpenAPIEndpointBranches:
    def test_openapi3_request_body_ref(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "openapi": "3.0.0",
            "paths": {
                "/create": {
                    "post": {
                        "operationId": "createItem",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CreateItemRequest"},
                                },
                            },
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/Item"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        result = extract_from_openapi_document("spec.yaml", doc)
        assert len(result.endpoints) == 1
        ep = result.endpoints[0]
        assert ep.request_body_ref == "CreateItemRequest"
        assert "Item" in ep.response_refs.values()

    def test_endpoint_with_handler_hints(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi_document

        doc = {
            "openapi": "3.0.0",
            "paths": {
                "/health": {
                    "get": {
                        "operationId": "healthCheck",
                        "x-handler": "controllers.health.check",
                        "summary": "Health check",
                    },
                },
            },
        }
        result = extract_from_openapi_document("spec.yaml", doc)
        assert len(result.endpoints) == 1
        ep = result.endpoints[0]
        assert ep.operation_id == "healthCheck"


# ---------------------------------------------------------------------------
# Protobuf: map fields and enum parsing
# ---------------------------------------------------------------------------


class TestProtobufEdgeCases:
    def test_proto_with_map_field(self) -> None:
        from contextmine_core.analyzer.extractors.protobuf import extract_from_protobuf

        content = """
syntax = "proto3";

message Config {
    map<string, string> labels = 1;
    map<string, int32> counts = 2;
    string name = 3;
}
"""
        result = extract_from_protobuf("config.proto", content)
        assert len(result.messages) >= 1
        config = result.messages[0]
        # Map fields should be parsed
        assert len(config.fields) >= 2

    def test_proto_with_enum_only(self) -> None:
        from contextmine_core.analyzer.extractors.protobuf import extract_from_protobuf

        content = """
syntax = "proto3";

enum Color {
    COLOR_UNSPECIFIED = 0;
    COLOR_RED = 1;
    COLOR_GREEN = 2;
    COLOR_BLUE = 3;
}
"""
        result = extract_from_protobuf("colors.proto", content)
        assert len(result.enums) >= 1
        color_enum = result.enums[0]
        assert color_enum.name == "Color"
        assert len(color_enum.values) >= 4


# ---------------------------------------------------------------------------
# mermaid_c4.py: more helper branches
# ---------------------------------------------------------------------------


class TestMermaidC4Additional:
    def test_safe_text_all_empty(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _safe_text

        result = _safe_text("  \n  ", "default")
        # After strip, text is empty, should use fallback
        assert result == "default"

    def test_build_relation_lines_with_label_key(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _build_relation_lines

        edges = [
            {
                "source_node_id": "a",
                "target_node_id": "b",
                "label": "custom_label",
                "kind": "depends_on",
                "meta": {},
            },
        ]
        lines = _build_relation_lines(edges, label_key="label")
        assert len(lines) == 1
        assert "custom_label" in lines[0]

    def test_limit_nodes_empty(self) -> None:
        from contextmine_core.exports.mermaid_c4 import _limit_nodes_by_degree

        nodes, edges, truncated = _limit_nodes_by_degree([], [], 10)
        assert nodes == []
        assert truncated is False


# ---------------------------------------------------------------------------
# analyzer/extractors/tests.py: additional extraction branches
# ---------------------------------------------------------------------------


class TestTestExtractorBranches:
    def test_python_test_with_default_params(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import pytest

class TestCalc:
    def test_add(self, x=1, y=2):
        assert x + y == 3
"""
        extraction = extract_tests_from_file("tests/test_calc.py", code)
        assert len(extraction.cases) >= 1

    def test_python_decorated_non_fixture(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
import pytest

@pytest.mark.slow
def test_slow_operation():
    import time
    time.sleep(0.01)
    assert True
"""
        extraction = extract_tests_from_file("tests/test_slow.py", code)
        assert len(extraction.cases) >= 1

    def test_js_nested_describe(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
describe('outer', () => {
  describe('inner', () => {
    it('should work', () => {
      expect(true).toBe(true);
    });
  });
  it('outer test', () => {
    expect(1).toBe(1);
  });
});
"""
        extraction = extract_tests_from_file("tests/nested.spec.ts", code)
        assert len(extraction.suites) >= 2
        assert len(extraction.cases) >= 2

    def test_ruby_rspec_extraction(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        extraction = extract_tests_from_file("tests/test_main.rb", "RSpec.describe 'test' do\nend")
        assert extraction.framework == "rspec"
        assert len(extraction.suites) >= 1
        assert extraction.suites[0].name == "test"

    def test_python_standalone_test_with_assert(self) -> None:
        from contextmine_core.analyzer.extractors.tests import extract_tests_from_file

        code = """
def test_simple():
    x = 42
    assert x == 42
    assert isinstance(x, int)
"""
        extraction = extract_tests_from_file("tests/test_simple.py", code)
        assert len(extraction.cases) >= 1
        case = extraction.cases[0]
        assert len(case.raw_assertions) >= 1


class TestLspLanguagesGlobBranch:
    """Cover lsp/languages.py line 107 — glob pattern project root detection."""

    def test_glob_marker_finds_root(self, tmp_path):
        from contextmine_core.lsp.languages import find_project_root

        # Create a file matching a glob marker (e.g., *.sln)
        (tmp_path / "MyProject.sln").write_text("")
        code_file = tmp_path / "src" / "main.cs"
        code_file.parent.mkdir(parents=True, exist_ok=True)
        code_file.write_text("")
        result = find_project_root(code_file)
        assert result == tmp_path
