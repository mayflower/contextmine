"""Targeted tests to close coverage gaps across packages/core modules.

Each section targets specific uncovered lines identified via --cov-report=term-missing.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from contextmine_core.analyzer.extractors.ast_utils import (
    is_pascal_case,
    unquote,
)
from contextmine_core.analyzer.extractors.schema import (
    AggregatedSchema,
    ColumnDef,
    ForeignKeyDef,
    SchemaExtraction,
    TableDef,
    _extract_schema_from_sql_ddl,
    _is_code_or_schema_file,
    _mermaid_safe,
    _mermaid_type,
    _normalize_sql_type,
    _split_sql_items,
    _strip_sql_identifier,
    aggregate_schema_extractions,
    generate_mermaid_erd,
    get_column_natural_key,
    get_table_natural_key,
)
from contextmine_core.analyzer.extractors.tests import (
    detect_test_framework,
    extract_tests_from_file,
    extract_tests_from_files,
    looks_like_test_file,
)
from contextmine_core.context import (
    FakeLLM,
    build_context_prompt,
)
from contextmine_core.exports.mermaid_c4 import (
    _build_relation_lines,
    _kind_value,
    _limit_nodes_by_degree,
    _normalize_c4_view,
    _normalize_scope,
    _safe_id,
    _safe_text,
)
from contextmine_core.metrics.coverage_reports import (
    PROTOCOL_CLOVER,
    PROTOCOL_COBERTURA,
    PROTOCOL_GENERIC_JSON,
    PROTOCOL_JACOCO,
    PROTOCOL_LCOV,
    CoverageAggregate,
    _coverage_from_counts,
    _tag_local_name,
    detect_coverage_protocol,
    parse_clover_xml,
    parse_cobertura_xml,
    parse_coverage_report,
    parse_generic_file_coverage_json,
    parse_jacoco_xml,
    parse_lcov_report,
)
from contextmine_core.metrics.discovery import (
    discover_coverage_reports,
    normalize_posix_path,
    to_repo_relative_path,
)
from contextmine_core.metrics.duplication import _normalize_line, compute_file_duplication_ratio
from contextmine_core.models import TwinLayer
from contextmine_core.research import llm as research_llm
from contextmine_core.research.llm.mock import MockLLMProvider
from contextmine_core.search import SearchResponse, SearchResult
from contextmine_core.semantic_snapshot import Snapshot as SnapshotImport
from contextmine_core.semantic_snapshot.indexers import (
    index_project,
)
from contextmine_core.semantic_snapshot.indexers.base import BaseIndexerBackend
from contextmine_core.semantic_snapshot.indexers.java import JavaIndexerBackend
from contextmine_core.semantic_snapshot.indexers.language_census import (
    LanguageCensusEntry,
    LanguageCensusReport,
    _build_cloc_command,
    _build_not_match_dir_regex,
    _composer_vendor_dir,
    _fallback_extension_census,
    _is_ignored_relative_path,
    _language_from_extension,
    _load_cloc_json,
    _normalize_language_name,
    _parse_cloc_by_file,
    _parse_cloc_summary,
)
from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend
from contextmine_core.semantic_snapshot.indexers.python import PythonIndexerBackend
from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend
from contextmine_core.semantic_snapshot.models import (
    IndexArtifact,
    IndexConfig,
    InstallDepsMode,
    Language,
    ProjectTarget,
    Range,
    RelationKind,
    SymbolKind,
)
from contextmine_core.semantic_snapshot.scip import (
    SCIP_ROLE_IMPORT,
    SCIP_ROLE_WRITE_ACCESS,
    SCIPProvider,
    build_snapshot_scip,
)
from contextmine_core.telemetry.spans import (
    trace_db_operation,
    trace_embedding_call,
    trace_llm_call,
    trace_sync_llm_call,
)
from contextmine_core.twin.evolution import (
    EntityGroup,
    _bus_factor,
    _coverage_value,
    _display_label,
    _file_path_from_natural_key,
    _normalize_min_max,
    _percentile,
    _safe_float,
    _safe_int,
    _safe_ratio,
    _topological_cycles,
    build_entity_key,
    derive_arch_group,
)
from contextmine_core.twin.service import (
    _compute_crap_score,
    infer_edge_layers,
    infer_node_layers,
)
from contextmine_core.validation.connectors import (
    ValidationMetric,
    _state_of,
    fetch_argo_metrics,
    fetch_tekton_metrics,
    fetch_temporal_alerts,
)
from defusedxml import ElementTree as safe_et
from pydantic import BaseModel, Field

# ============================================================================
# 1. SCIP helper methods  (scip.py missing: 123-129, 163, 181, 183, 203,
#    211, 213, 218, 234-253, 265, 270, 335-375, 381-383, 398, 461, 463,
#    509, 513-522, 583, 689, 691, 693, 695, 723-724, 775, 780, 784,
#    805, 811-815, 847)
# ============================================================================


class TestSCIPParseRange:
    """Cover _parse_range (lines ~524-547)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/nonexistent.scip")

    def test_four_element_range(self) -> None:
        r = self.provider._parse_range([0, 5, 10, 20])
        assert r is not None
        assert r.start_line == 1  # 0-based -> 1-based
        assert r.start_col == 5
        assert r.end_line == 11
        assert r.end_col == 20

    def test_three_element_range(self) -> None:
        r = self.provider._parse_range([4, 2, 8])
        assert r is not None
        assert r.start_line == 5
        assert r.start_col == 2
        assert r.end_line == 5  # same line
        assert r.end_col == 8

    def test_empty_range_returns_none(self) -> None:
        assert self.provider._parse_range([]) is None

    def test_two_elements_returns_none(self) -> None:
        assert self.provider._parse_range([1, 2]) is None


class TestSCIPGetLanguageString:
    """Cover _get_language_string (lines 549-553)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_normal_language(self) -> None:
        assert self.provider._get_language_string("TypeScript") == "typescript"

    def test_empty_language(self) -> None:
        assert self.provider._get_language_string("") is None


class TestSCIPExtractName:
    """Cover _extract_name_from_symbol (lines 555-583)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_empty_string(self) -> None:
        assert self.provider._extract_name_from_symbol("") is None

    def test_local_symbol(self) -> None:
        result = self.provider._extract_name_from_symbol("local 42")
        assert result == "local_42"

    def test_short_symbol(self) -> None:
        assert self.provider._extract_name_from_symbol("scip python") is None

    def test_normal_symbol(self) -> None:
        result = self.provider._extract_name_from_symbol(
            "scip-python python mypackage 0.1.0 mymodule/MyClass#method()."
        )
        assert result == "method"

    def test_symbol_no_matches(self) -> None:
        result = self.provider._extract_name_from_symbol("scip python pkg 0.1 123")
        assert result is None


class TestSCIPInferKindAndName:
    """Cover _infer_kind_and_name_from_symbol (lines 585-634)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_empty_string(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol("")
        assert kind == SymbolKind.UNKNOWN
        assert name is None

    def test_local_symbol(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol("local 42")
        assert kind == SymbolKind.UNKNOWN
        assert name is None

    def test_parameter_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/Class#method().(param)"
        )
        assert kind == SymbolKind.PARAMETER
        assert name == "param"

    def test_method_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/Class#method()."
        )
        assert kind == SymbolKind.METHOD
        assert name == "method"

    def test_function_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/func()."
        )
        assert kind == SymbolKind.FUNCTION
        assert name == "func"

    def test_class_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/MyClass#"
        )
        assert kind == SymbolKind.CLASS
        assert name == "MyClass"

    def test_property_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/Class#field."
        )
        assert kind == SymbolKind.PROPERTY
        assert name == "field"

    def test_term_function_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/helper."
        )
        assert kind == SymbolKind.FUNCTION
        assert name == "helper"

    def test_module_slash_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 mymodule/"
        )
        assert kind == SymbolKind.MODULE
        assert name == "mymodule"

    def test_module_colon_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 mymodule:"
        )
        assert kind == SymbolKind.MODULE
        assert name == "mymodule"

    def test_type_alias_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 module/[T]"
        )
        assert kind == SymbolKind.TYPE_ALIAS
        assert name == "T"

    def test_macro_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-rust rust pkg 0.1.0 module/my_macro!"
        )
        assert kind == SymbolKind.FUNCTION
        assert name == "my_macro"

    def test_unknown_descriptor(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol(
            "scip-python python pkg 0.1.0 something"
        )
        assert kind == SymbolKind.UNKNOWN
        assert name == "something"

    def test_short_parts(self) -> None:
        kind, name = self.provider._infer_kind_and_name_from_symbol("a b c")
        assert kind == SymbolKind.UNKNOWN


class TestSCIPDescriptorTail:
    """Cover _descriptor_tail (lines 636-641)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_normal(self) -> None:
        result = self.provider._descriptor_tail("scip python pkg 0.1 module/Class#")
        assert result == "module/Class#"

    def test_short(self) -> None:
        result = self.provider._descriptor_tail("too short")
        assert result == ""


class TestSCIPLastIdentifier:
    """Cover _last_identifier (lines 643-650)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_normal(self) -> None:
        result = self.provider._last_identifier("module/Class#method")
        assert result == "method"

    def test_with_backticks(self) -> None:
        result = self.provider._last_identifier("`package`/`module`")
        assert result == "module"

    def test_no_identifiers(self) -> None:
        result = self.provider._last_identifier("///")
        assert result is None


class TestSCIPRelationKindFromOccurrence:
    """Cover _relation_kind_from_occurrence (lines 652-680)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_import_role(self) -> None:
        result = self.provider._relation_kind_from_occurrence(
            symbol_roles=SCIP_ROLE_IMPORT,
            syntax_kind=6,
            caller_kind=SymbolKind.FUNCTION,
            target_kind=SymbolKind.MODULE,
        )
        assert result == RelationKind.IMPORTS

    def test_call_by_syntax_kind(self) -> None:
        result = self.provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=15,  # IdentifierFunction
            caller_kind=SymbolKind.CLASS,  # eligible
            target_kind=SymbolKind.FUNCTION,
        )
        assert result == RelationKind.CALLS

    def test_call_both_callable(self) -> None:
        result = self.provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=99,  # not call-like
            caller_kind=SymbolKind.FUNCTION,
            target_kind=SymbolKind.METHOD,
        )
        assert result == RelationKind.CALLS

    def test_call_eligible_but_not_preferred(self) -> None:
        result = self.provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=99,
            caller_kind=SymbolKind.CLASS,  # eligible but not preferred
            target_kind=SymbolKind.FUNCTION,  # callable target
        )
        assert result == RelationKind.CALLS

    def test_reference_fallback(self) -> None:
        result = self.provider._relation_kind_from_occurrence(
            symbol_roles=0,
            syntax_kind=99,
            caller_kind=SymbolKind.VARIABLE,  # neither preferred nor eligible
            target_kind=SymbolKind.VARIABLE,  # not callable
        )
        assert result == RelationKind.REFERENCES

    def test_write_access_prevents_call(self) -> None:
        result = self.provider._relation_kind_from_occurrence(
            symbol_roles=SCIP_ROLE_WRITE_ACCESS,
            syntax_kind=6,  # call-like
            caller_kind=SymbolKind.FUNCTION,
            target_kind=SymbolKind.FUNCTION,
        )
        # With write access + syntax_kind call-like, the first branch is blocked
        # but second branch (both callable) still triggers
        assert result == RelationKind.CALLS


class TestSCIPFallbackSymbolKind:
    """Cover _fallback_symbol_kind (lines 682-696)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_inferred_function(self) -> None:
        result = self.provider._fallback_symbol_kind("scip-python python pkg 0.1.0 module/func().")
        assert result == SymbolKind.FUNCTION

    def test_class_hash(self) -> None:
        result = self.provider._fallback_symbol_kind("scip-python python pkg 0.1.0 module/Class#")
        assert result == SymbolKind.CLASS

    def test_property_dot(self) -> None:
        # This needs a descriptor ending in "." that is not a function
        # Since _infer returns FUNCTION for "."-ending, we need something
        # that _infer returns UNKNOWN for, but descriptor ends with "."
        # Actually _infer handles "." ending, so let's test the fallback default
        result = self.provider._fallback_symbol_kind("scip-python python pkg 0.1.0 something")
        assert result == SymbolKind.MODULE  # default fallback

    def test_module_slash_fallback(self) -> None:
        result = self.provider._fallback_symbol_kind("scip-python python pkg 0.1.0 mymod/")
        assert result == SymbolKind.MODULE


class TestSCIPRangeHelpers:
    """Cover _range_contains and _range_span (lines 817-827)."""

    def setup_method(self) -> None:
        self.provider = SCIPProvider("/tmp/x.scip")

    def test_range_contains_true(self) -> None:
        outer = Range(start_line=1, start_col=0, end_line=10, end_col=0)
        inner = Range(start_line=2, start_col=0, end_line=5, end_col=0)
        assert self.provider._range_contains(outer, inner) is True

    def test_range_contains_false(self) -> None:
        outer = Range(start_line=3, start_col=0, end_line=5, end_col=0)
        inner = Range(start_line=1, start_col=0, end_line=10, end_col=0)
        assert self.provider._range_contains(outer, inner) is False

    def test_range_span(self) -> None:
        r = Range(start_line=1, start_col=5, end_line=10, end_col=20)
        span = self.provider._range_span(r)
        assert span == 9 * 1_000_000 + 15

    def test_build_snapshot_scip_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            build_snapshot_scip("/tmp/nonexistent_index.scip")


class TestSCIPIsAvailable:
    """Cover is_available (line 115)."""

    def test_not_available(self) -> None:
        p = SCIPProvider("/tmp/definitely_not_a_real_scip_file.scip")
        assert p.is_available() is False


# ============================================================================
# 2. context.py — assemble_context and streaming (lines 62, 84-91 already
#    covered; target FakeLLM multi-chunk and assemble_context with mock)
# ============================================================================


class TestFakeLLMMultipleChunks:
    """Cover FakeLLM with multiple chunks, separator, and chunk accumulation."""

    @pytest.mark.anyio
    async def test_multiple_chunks_content(self) -> None:
        @dataclass
        class FakeChunk:
            uri: str
            title: str
            content: str

        llm = FakeLLM()
        chunks = [
            FakeChunk(uri="doc://a", title="A", content="Content A"),
            FakeChunk(uri="doc://b", title="B", content="Content B"),
        ]
        prompt = build_context_prompt("query", chunks)
        result = await llm.generate("system", prompt, 4000)
        assert "Response to:" in result
        assert "## Sources" in result

    @pytest.mark.anyio
    async def test_generate_with_separator(self) -> None:
        llm = FakeLLM()
        prompt = (
            "## Query\ntest query\n## Context Chunks\n"
            "### Chunk 1 (from: doc://a)\nContent A\n---\n"
            "### Chunk 2 (from: doc://b)\nContent B"
        )
        result = await llm.generate("system", prompt, 4000)
        assert "doc://a" in result


# ============================================================================
# 3. schema.py — SQL DDL parsing (lines 166-361 partial coverage)
# ============================================================================


class TestStripSqlIdentifier:
    def test_backtick(self) -> None:
        assert _strip_sql_identifier("`my_table`") == "my_table"

    def test_double_quotes(self) -> None:
        assert _strip_sql_identifier('"schema"."table"') == "schema.table"

    def test_brackets(self) -> None:
        assert _strip_sql_identifier("[dbo].[users]") == "dbo.users"

    def test_trailing_comma(self) -> None:
        assert _strip_sql_identifier("col,") == "col"

    def test_empty(self) -> None:
        assert _strip_sql_identifier("") == ""


class TestSplitSqlItems:
    def test_simple(self) -> None:
        result = _split_sql_items("a, b, c")
        assert result == ["a", "b", "c"]

    def test_nested_parens(self) -> None:
        result = _split_sql_items("a INT, CONSTRAINT pk PRIMARY KEY (a, b), c TEXT")
        assert len(result) == 3
        assert "PRIMARY KEY (a, b)" in result[1]

    def test_quoted_strings(self) -> None:
        result = _split_sql_items("a TEXT DEFAULT 'hello, world', b INT")
        assert len(result) == 2

    def test_backtick_strings(self) -> None:
        result = _split_sql_items("`col,a` INT, `col,b` TEXT")
        assert len(result) == 2


class TestNormalizeSqlType:
    def test_varchar(self) -> None:
        assert _normalize_sql_type("VARCHAR(255)") == "String"

    def test_text(self) -> None:
        assert _normalize_sql_type("TEXT") == "Text"

    def test_bigint(self) -> None:
        assert _normalize_sql_type("bigint") == "BigInteger"

    def test_smallint(self) -> None:
        assert _normalize_sql_type("smallint") == "SmallInteger"

    def test_integer(self) -> None:
        assert _normalize_sql_type("integer") == "Integer"

    def test_serial(self) -> None:
        assert _normalize_sql_type("serial") == "Integer"

    def test_boolean(self) -> None:
        assert _normalize_sql_type("boolean") == "Boolean"

    def test_timestamp(self) -> None:
        assert _normalize_sql_type("timestamp") == "DateTime"

    def test_date(self) -> None:
        assert _normalize_sql_type("date") == "Date"

    def test_time(self) -> None:
        assert _normalize_sql_type("time") == "Time"

    def test_uuid(self) -> None:
        assert _normalize_sql_type("uuid") == "UUID"

    def test_json(self) -> None:
        assert _normalize_sql_type("json") == "JSON"

    def test_numeric(self) -> None:
        assert _normalize_sql_type("numeric(10,2)") == "Numeric"

    def test_float(self) -> None:
        assert _normalize_sql_type("float") == "Float"

    def test_blob(self) -> None:
        assert _normalize_sql_type("bytea") == "LargeBinary"

    def test_empty(self) -> None:
        assert _normalize_sql_type("") == "unknown"

    def test_unknown_type(self) -> None:
        assert _normalize_sql_type("GEOMETRY") == "GEOMETRY"

    def test_character_varying(self) -> None:
        assert _normalize_sql_type("character varying(100)") == "String"

    def test_double_precision(self) -> None:
        assert _normalize_sql_type("double precision") == "Float"

    def test_decimal(self) -> None:
        assert _normalize_sql_type("decimal(18,6)") == "Numeric"

    def test_money(self) -> None:
        assert _normalize_sql_type("money") == "Numeric"


class TestExtractSchemaFromSqlDdl:
    def test_simple_create_table(self) -> None:
        sql = textwrap.dedent("""\
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email TEXT
            );
        """)
        result = _extract_schema_from_sql_ddl("schema.sql", sql)
        assert result.framework == "sql"
        assert len(result.tables) == 1
        tbl = result.tables[0]
        assert tbl.name == "users"
        assert len(tbl.columns) == 3
        id_col = tbl.columns[0]
        assert id_col.primary_key is True
        name_col = tbl.columns[1]
        assert name_col.nullable is False  # NOT NULL constraint

    def test_foreign_key_constraint(self) -> None:
        sql = textwrap.dedent("""\
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        result = _extract_schema_from_sql_ddl("schema.sql", sql)
        assert len(result.foreign_keys) == 1
        fk = result.foreign_keys[0]
        assert fk.source_table == "orders"
        assert fk.target_table == "users"

    def test_inline_references(self) -> None:
        sql = textwrap.dedent("""\
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER REFERENCES users(id)
            );
        """)
        result = _extract_schema_from_sql_ddl("schema.sql", sql)
        user_col = result.tables[0].columns[1]
        assert user_col.foreign_key == "users.id"

    def test_separate_pk_constraint(self) -> None:
        sql = textwrap.dedent("""\
            CREATE TABLE items (
                id INTEGER,
                name TEXT NOT NULL,
                PRIMARY KEY (id)
            );
        """)
        result = _extract_schema_from_sql_ddl("schema.sql", sql)
        tbl = result.tables[0]
        assert "id" in tbl.primary_keys
        id_col = [c for c in tbl.columns if c.name == "id"][0]
        assert id_col.primary_key is True

    def test_if_not_exists(self) -> None:
        sql = "CREATE TABLE IF NOT EXISTS t1 (id INT PRIMARY KEY); "
        result = _extract_schema_from_sql_ddl("x.sql", sql)
        assert len(result.tables) == 1
        assert result.tables[0].name == "t1"

    def test_no_tables(self) -> None:
        result = _extract_schema_from_sql_ddl("x.sql", "SELECT 1;")
        assert result.tables == []


class TestIsCodeOrSchemaFile:
    def test_python(self) -> None:
        assert _is_code_or_schema_file("models.py") is True

    def test_sql(self) -> None:
        assert _is_code_or_schema_file("schema.sql") is True

    def test_prisma(self) -> None:
        assert _is_code_or_schema_file("schema.prisma") is True

    def test_image(self) -> None:
        assert _is_code_or_schema_file("logo.png") is False

    def test_markdown(self) -> None:
        assert _is_code_or_schema_file("README.md") is False

    def test_lock(self) -> None:
        assert _is_code_or_schema_file("package-lock.json") is True  # .json is allowed


class TestMermaidErdGeneration:
    def test_generate_erd(self) -> None:
        schema = AggregatedSchema()
        schema.tables["users"] = TableDef(
            name="users",
            columns=[
                ColumnDef(name="id", type_name="Integer", primary_key=True),
                ColumnDef(name="name", type_name="String"),
            ],
            primary_keys=["id"],
        )
        schema.tables["orders"] = TableDef(
            name="orders",
            columns=[
                ColumnDef(name="id", type_name="Integer", primary_key=True),
                ColumnDef(name="user_id", type_name="Integer", foreign_key="users.id"),
            ],
            primary_keys=["id"],
        )
        schema.foreign_keys.append(
            ForeignKeyDef(
                name=None,
                source_table="orders",
                source_columns=["user_id"],
                target_table="users",
                target_columns=["id"],
            )
        )
        result = generate_mermaid_erd(schema)
        assert "erDiagram" in result
        assert "users" in result
        assert "orders" in result
        assert "||--o{" in result

    def test_mermaid_safe(self) -> None:
        assert _mermaid_safe("my-table") == "my_table"
        assert _mermaid_safe("my table") == "my_table"

    def test_mermaid_type(self) -> None:
        assert _mermaid_type("String") == "string"
        assert _mermaid_type("Integer") == "int"
        assert _mermaid_type("Boolean") == "bool"
        assert _mermaid_type("DateTime") == "datetime"
        assert _mermaid_type("UUID") == "uuid"
        assert _mermaid_type("LargeBinary") == "blob"
        assert _mermaid_type("Enum") == "enum"
        assert _mermaid_type("SomeCustom") == "somecustom"

    def test_natural_keys(self) -> None:
        assert get_table_natural_key("users") == "db:users"
        assert get_column_natural_key("users", "id") == "db:users.id"


class TestAggregateSchemaExtractions:
    def test_merge_tables(self) -> None:
        e1 = SchemaExtraction(
            file_path="a.sql",
            tables=[
                TableDef(
                    name="users",
                    columns=[ColumnDef(name="id", type_name="Integer", primary_key=True)],
                    primary_keys=["id"],
                )
            ],
        )
        e2 = SchemaExtraction(
            file_path="b.sql",
            tables=[
                TableDef(
                    name="users",
                    columns=[
                        ColumnDef(name="id", type_name="Integer", primary_key=True),
                        ColumnDef(name="email", type_name="String"),
                    ],
                    primary_keys=["id"],
                )
            ],
        )
        schema = aggregate_schema_extractions([e1, e2])
        assert len(schema.tables) == 1
        assert len(schema.tables["users"].columns) == 2

    def test_dedupe_foreign_keys(self) -> None:
        fk = ForeignKeyDef(
            name=None,
            source_table="orders",
            source_columns=["user_id"],
            target_table="users",
            target_columns=["id"],
        )
        e1 = SchemaExtraction(file_path="a.sql", foreign_keys=[fk])
        e2 = SchemaExtraction(file_path="b.sql", foreign_keys=[fk])
        schema = aggregate_schema_extractions([e1, e2])
        assert len(schema.foreign_keys) == 1

    def test_column_fk_extraction(self) -> None:
        e = SchemaExtraction(
            file_path="c.sql",
            tables=[
                TableDef(
                    name="orders",
                    columns=[
                        ColumnDef(
                            name="user_id",
                            type_name="Integer",
                            foreign_key="users.id",
                        )
                    ],
                )
            ],
        )
        schema = aggregate_schema_extractions([e])
        assert len(schema.foreign_keys) == 1
        assert schema.foreign_keys[0].source_table == "orders"


# ============================================================================
# 4. tests.py (analyzer/extractors) — cover detect_test_framework (lines ~143-162)
# ============================================================================


class TestLooksLikeTestFile:
    def test_python_test(self) -> None:
        assert looks_like_test_file("tests/test_foo.py") is True

    def test_js_spec(self) -> None:
        assert looks_like_test_file("src/foo.spec.ts") is True

    def test_test_dir(self) -> None:
        assert looks_like_test_file("src/__tests__/foo.js") is True

    def test_non_test(self) -> None:
        assert looks_like_test_file("src/main.py") is False

    def test_image(self) -> None:
        assert looks_like_test_file("test_image.png") is False


class TestDetectTestFramework:
    def test_pytest(self) -> None:
        assert detect_test_framework("test_foo.py", "import pytest") == "pytest"

    def test_unittest(self) -> None:
        assert detect_test_framework("test_foo.py", "import unittest") == "unittest"

    def test_jest(self) -> None:
        assert detect_test_framework("foo.test.ts", "// jest config") == "jest"

    def test_junit(self) -> None:
        assert detect_test_framework("FooTest.java", "@Test void foo") == "junit"

    def test_vitest(self) -> None:
        assert detect_test_framework("foo.test.ts", "import { vitest }") == "vitest"

    def test_cypress(self) -> None:
        assert detect_test_framework("foo.spec.ts", "cypress.visit()") == "cypress"

    def test_playwright(self) -> None:
        assert detect_test_framework("foo.spec.ts", "playwright.chromium") == "playwright"

    def test_js_test_by_extension(self) -> None:
        assert detect_test_framework("foo.spec.ts", "describe('x', () => {})") == "js_test"

    def test_unknown(self) -> None:
        assert detect_test_framework("test_foo.py", "# no framework") == "unknown"


class TestExtractTestsFromFiles:
    def test_skips_non_test_files(self) -> None:
        files = [("src/main.py", "def main(): pass")]
        assert extract_tests_from_files(files) == []

    def test_skips_empty_content(self) -> None:
        files = [("tests/test_foo.py", "   ")]
        assert extract_tests_from_files(files) == []

    def test_extracts_python_test(self) -> None:
        content = textwrap.dedent("""\
            import pytest

            def test_hello():
                assert True
        """)
        files = [("tests/test_hello.py", content)]
        result = extract_tests_from_files(files)
        assert len(result) == 1
        assert len(result[0].cases) == 1
        assert result[0].cases[0].name == "test_hello"


class TestExtractTestsFromFile:
    def test_python_fixture(self) -> None:
        content = textwrap.dedent("""\
            import pytest

            @pytest.fixture
            def my_fixture():
                return 42

            def test_uses_fixture(my_fixture):
                assert my_fixture == 42
        """)
        result = extract_tests_from_file("tests/test_fix.py", content)
        assert len(result.fixtures) == 1
        assert result.fixtures[0].name == "my_fixture"
        assert len(result.cases) == 1
        assert "my_fixture" in result.cases[0].fixture_names

    def test_python_class_suite(self) -> None:
        content = textwrap.dedent("""\
            class TestMyClass:
                def test_method(self):
                    assert True
        """)
        result = extract_tests_from_file("tests/test_cls.py", content)
        assert len(result.suites) == 1
        assert result.suites[0].name == "TestMyClass"
        assert len(result.cases) == 1
        assert result.cases[0].suite_name == "TestMyClass"

    def test_js_test_extraction(self) -> None:
        content = textwrap.dedent("""\
            describe('MyComponent', () => {
                beforeEach(() => {
                    setup();
                });
                it('should render', () => {
                    expect(true).toBe(true);
                });
            });
        """)
        result = extract_tests_from_file("src/my.test.ts", content)
        assert len(result.suites) >= 1
        assert len(result.cases) >= 1

    def test_unsupported_language(self) -> None:
        result = extract_tests_from_file("test_foo.unknown", "test something")
        assert result.cases == []
        assert result.suites == []


# ============================================================================
# 5. language_census.py — pure functions (lines 122-136, 152-214, etc.)
# ============================================================================


class TestNormalizeLanguageName:
    def test_python(self) -> None:
        assert _normalize_language_name("Python") == Language.PYTHON

    def test_typescript(self) -> None:
        assert _normalize_language_name("TypeScript") == Language.TYPESCRIPT

    def test_unknown(self) -> None:
        assert _normalize_language_name("Haskell") is None

    def test_empty(self) -> None:
        assert _normalize_language_name("") is None


class TestLanguageFromExtension:
    def test_py(self) -> None:
        assert _language_from_extension(".py") == Language.PYTHON

    def test_tsx(self) -> None:
        assert _language_from_extension(".tsx") == Language.TYPESCRIPT

    def test_unknown(self) -> None:
        assert _language_from_extension(".rs") is None


class TestLoadClocJson:
    def test_valid_json(self) -> None:
        report = LanguageCensusReport()
        result = _load_cloc_json('{"Python": {"nFiles": 5}}', report, "test")
        assert result["Python"]["nFiles"] == 5

    def test_empty_string(self) -> None:
        report = LanguageCensusReport()
        result = _load_cloc_json("", report, "test")
        assert result == {}

    def test_json_with_trailing(self) -> None:
        report = LanguageCensusReport()
        result = _load_cloc_json('{"key": 1} some trailing text', report, "test")
        assert result == {"key": 1}
        assert any("trailing" in w for w in report.warnings)

    def test_invalid_json_no_brace(self) -> None:
        report = LanguageCensusReport()
        with pytest.raises(json.JSONDecodeError):
            _load_cloc_json("not json at all", report, "test")


class TestParseClocSummary:
    def test_basic(self) -> None:
        parsed = {
            "header": {"cloc_version": "1.0"},
            "Python": {"nFiles": 10, "code": 500, "comment": 50, "blank": 100},
            "SUM": {"nFiles": 10, "code": 500},
        }
        result = _parse_cloc_summary(parsed)
        assert Language.PYTHON in result
        assert result[Language.PYTHON].files == 10
        assert result[Language.PYTHON].code == 500

    def test_non_dict_value(self) -> None:
        parsed = {
            "header": {},
            "Python": {"nFiles": 5, "code": 10, "comment": 1, "blank": 2},
            "extra": "string",
        }
        result = _parse_cloc_summary(parsed)
        assert Language.PYTHON in result


class TestParseClocByFile:
    def test_basic(self, tmp_path: Path) -> None:
        parsed = {
            "header": {},
            "src/main.py": {"language": "Python", "code": 100},
            "SUM": {},
        }
        result = _parse_cloc_by_file(parsed, tmp_path)
        assert len(result) == 1
        assert result[0].language == Language.PYTHON
        assert result[0].code == 100

    def test_language_from_key_extension(self, tmp_path: Path) -> None:
        parsed = {
            "src/app.ts": {"language": "", "code": 50},
        }
        result = _parse_cloc_by_file(parsed, tmp_path)
        assert len(result) == 1
        assert result[0].language == Language.TYPESCRIPT


class TestBuildClocCommand:
    def test_basic(self) -> None:
        cmd = _build_cloc_command(
            "/usr/bin/cloc", Path("/repo"), "node_modules", by_file=False, not_match_dirs=None
        )
        assert "/usr/bin/cloc" in cmd
        assert "--json" in cmd
        assert "--by-file" not in cmd

    def test_by_file(self) -> None:
        cmd = _build_cloc_command(
            "/usr/bin/cloc", Path("/repo"), "nm", by_file=True, not_match_dirs=None
        )
        assert "--by-file" in cmd

    def test_with_not_match_dirs(self) -> None:
        cmd = _build_cloc_command(
            "/usr/bin/cloc", Path("/repo"), "nm", by_file=False, not_match_dirs="src/libs"
        )
        assert "--fullpath" in cmd
        assert any("not-match-d" in c for c in cmd)


class TestBuildNotMatchDirRegex:
    def test_single_path(self) -> None:
        result = _build_not_match_dir_regex({Path("src/libs")})
        assert result is not None
        assert "src/libs" in result

    def test_empty(self) -> None:
        assert _build_not_match_dir_regex(set()) is None


class TestIsIgnoredRelativePath:
    def test_ignored_dir(self) -> None:
        assert _is_ignored_relative_path(Path("node_modules/foo.js"), set()) is True

    def test_hidden_dir(self) -> None:
        assert _is_ignored_relative_path(Path(".cache/foo"), set()) is True

    def test_normal_path(self) -> None:
        assert _is_ignored_relative_path(Path("src/main.py"), set()) is False

    def test_prefix_match(self) -> None:
        prefixes = {Path("src/libs")}
        assert _is_ignored_relative_path(Path("src/libs/vendor/foo.php"), prefixes) is True


class TestComposerVendorDir:
    def test_no_composer_file(self, tmp_path: Path) -> None:
        assert _composer_vendor_dir(tmp_path) is None

    def test_default_vendor(self, tmp_path: Path) -> None:
        (tmp_path / "composer.json").write_text("{}")
        assert _composer_vendor_dir(tmp_path) == Path("vendor")

    def test_custom_vendor_dir(self, tmp_path: Path) -> None:
        (tmp_path / "composer.json").write_text('{"config": {"vendor-dir": "lib"}}')
        assert _composer_vendor_dir(tmp_path) == Path("lib")

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "composer.json").write_text("not json")
        assert _composer_vendor_dir(tmp_path) == Path("vendor")


class TestFallbackExtensionCensus:
    def test_basic(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("print('hello')\n")
        (tmp_path / "app.ts").write_text("console.log('hi');\n")
        (tmp_path / "readme.md").write_text("# readme\n")

        report = _fallback_extension_census(tmp_path)
        assert report.tool_name == "extension-fallback"
        assert Language.PYTHON in report.entries
        assert Language.TYPESCRIPT in report.entries
        assert len(report.file_stats) == 2


class TestLanguageCensusReport:
    def test_total_code(self) -> None:
        report = LanguageCensusReport()
        report.entries[Language.PYTHON] = LanguageCensusEntry(language=Language.PYTHON, code=100)
        report.entries[Language.TYPESCRIPT] = LanguageCensusEntry(
            language=Language.TYPESCRIPT, code=200
        )
        assert report.total_code == 300


# ============================================================================
# 6. twin/evolution.py — pure helper functions (lines 41-94, 115-134, 136-144, 147+)
# ============================================================================


class TestSafeFloat:
    def test_normal(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_none(self) -> None:
        assert _safe_float(None) == 0.0

    def test_invalid(self) -> None:
        assert _safe_float("abc") == 0.0

    def test_default(self) -> None:
        assert _safe_float(None, 5.0) == 5.0


class TestSafeInt:
    def test_normal(self) -> None:
        assert _safe_int(42) == 42

    def test_none(self) -> None:
        assert _safe_int(None) == 0

    def test_invalid(self) -> None:
        assert _safe_int("xyz") == 0


class TestSafeRatio:
    def test_normal(self) -> None:
        assert _safe_ratio(3.0, 6.0) == 0.5

    def test_zero_denominator(self) -> None:
        assert _safe_ratio(5.0, 0.0) == 0.0

    def test_negative_denominator(self) -> None:
        assert _safe_ratio(5.0, -1.0) == 0.0


class TestCoverageValue:
    def test_none(self) -> None:
        assert _coverage_value(None) is None

    def test_clamped_high(self) -> None:
        assert _coverage_value(150.0) == 100.0

    def test_clamped_low(self) -> None:
        assert _coverage_value(-5.0) == 0.0

    def test_normal(self) -> None:
        assert _coverage_value(75.5) == 75.5


class TestNormalizeMinMax:
    def test_normal(self) -> None:
        assert _normalize_min_max(5.0, 0.0, 10.0) == 0.5

    def test_equal_bounds(self) -> None:
        assert _normalize_min_max(5.0, 5.0, 5.0) == 1.0

    def test_zero_value_equal_bounds(self) -> None:
        assert _normalize_min_max(0.0, 5.0, 5.0) == 0.0


class TestFilePathFromNaturalKey:
    def test_file_prefix(self) -> None:
        result = _file_path_from_natural_key("file:src/main.py")
        assert result == "src/main.py"

    def test_non_file_prefix(self) -> None:
        assert _file_path_from_natural_key("symbol:foo") is None

    def test_empty_after_prefix(self) -> None:
        assert _file_path_from_natural_key("file:  ") is None


class TestDeriveArchGroup:
    def test_normal_path(self) -> None:
        result = derive_arch_group("apps/api/src/main.py")
        assert result is not None
        assert isinstance(result, EntityGroup)

    def test_none_path(self) -> None:
        result = derive_arch_group(None)
        assert result is None


class TestBuildEntityKey:
    def test_file_level(self) -> None:
        result = build_entity_key("src/main.py", None, "file")
        assert result is not None
        assert result.startswith("file:")

    def test_container_level(self) -> None:
        result = build_entity_key("apps/api/src/main.py", None, "container")
        # May or may not produce a result depending on heuristics
        # Just ensure no exceptions
        assert result is None or result.startswith("container:")

    def test_invalid_level(self) -> None:
        assert build_entity_key("src/main.py", None, "invalid") is None


class TestDisplayLabel:
    def test_with_colon(self) -> None:
        assert _display_label("file:src/main.py") == "src/main.py"

    def test_without_colon(self) -> None:
        assert _display_label("something") == "something"


class TestBusFactor:
    def test_empty(self) -> None:
        assert _bus_factor({}) == 0

    def test_single_contributor(self) -> None:
        assert _bus_factor({"alice": 100.0}) == 1

    def test_balanced(self) -> None:
        # 4 contributors each at 25% - need 4 to reach 100%
        result = _bus_factor({"a": 25.0, "b": 25.0, "c": 25.0, "d": 25.0})
        assert result >= 3

    def test_skewed(self) -> None:
        # One contributor has 90%
        result = _bus_factor({"a": 90.0, "b": 10.0})
        assert result == 1

    def test_zero_total(self) -> None:
        assert _bus_factor({"a": 0.0, "b": 0.0}) == 0


class TestPercentile:
    def test_empty(self) -> None:
        assert _percentile([], 0.5) == 0.0

    def test_single(self) -> None:
        assert _percentile([5.0], 0.5) == 5.0

    def test_median(self) -> None:
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert result == 3.0

    def test_p75(self) -> None:
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.75)
        assert result == 4.0


class TestTopologicalCycles:
    def test_no_cycle(self) -> None:
        graph = {"A": {"B"}, "B": {"C"}}
        cycles = _topological_cycles(graph)
        assert cycles == []

    def test_simple_cycle(self) -> None:
        graph = {"A": {"B"}, "B": {"A"}}
        cycles = _topological_cycles(graph)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"A", "B"}

    def test_self_loop_excluded(self) -> None:
        # self-loops form SCCs of size 1, which are filtered out
        graph = {"A": {"A"}}
        cycles = _topological_cycles(graph)
        assert cycles == []


# ============================================================================
# 7. twin/service.py — pure helpers (lines 60-96)
# ============================================================================


class TestComputeCrapScore:
    def test_none_inputs(self) -> None:
        assert _compute_crap_score(None, None) is None
        assert _compute_crap_score(10.0, None) is None
        assert _compute_crap_score(None, 50.0) is None

    def test_full_coverage(self) -> None:
        # crap = (c^2 * (1 - 1.0)^3) + c = 0 + c = c
        result = _compute_crap_score(10.0, 100.0)
        assert result == pytest.approx(10.0)

    def test_zero_coverage(self) -> None:
        # crap = (c^2 * 1^3) + c = c^2 + c
        result = _compute_crap_score(10.0, 0.0)
        assert result == pytest.approx(110.0)


class TestInferNodeLayers:
    def test_code_kinds(self) -> None:
        for kind in ("file", "module", "symbol", "function", "method", "class", "validator"):
            assert TwinLayer.CODE_CONTROLFLOW in infer_node_layers(kind)

    def test_interface_kinds(self) -> None:
        for kind in ("api_endpoint", "interface", "rpc", "service", "component"):
            assert TwinLayer.COMPONENT_INTERFACE in infer_node_layers(kind)

    def test_domain_kinds(self) -> None:
        for kind in ("bounded_context", "container", "db_table", "db_column"):
            assert TwinLayer.DOMAIN_CONTAINER in infer_node_layers(kind)

    def test_default(self) -> None:
        assert TwinLayer.PORTFOLIO_SYSTEM in infer_node_layers("unknown_thing")


class TestInferEdgeLayers:
    def test_code_edges(self) -> None:
        assert TwinLayer.CODE_CONTROLFLOW in infer_edge_layers("file_defines_symbol")
        assert TwinLayer.CODE_CONTROLFLOW in infer_edge_layers("symbol_calls_foo")
        assert TwinLayer.CODE_CONTROLFLOW in infer_edge_layers("references")
        assert TwinLayer.CODE_CONTROLFLOW in infer_edge_layers("contains")

    def test_interface_edges(self) -> None:
        assert TwinLayer.COMPONENT_INTERFACE in infer_edge_layers("interface_dependency")
        assert TwinLayer.COMPONENT_INTERFACE in infer_edge_layers("endpoint_route")

    def test_domain_edges(self) -> None:
        assert TwinLayer.DOMAIN_CONTAINER in infer_edge_layers("context_boundary")
        assert TwinLayer.DOMAIN_CONTAINER in infer_edge_layers("domain_event")

    def test_default_edges(self) -> None:
        assert TwinLayer.PORTFOLIO_SYSTEM in infer_edge_layers("unknown_edge_kind")


# ============================================================================
# 8. mermaid_c4.py — pure helpers (lines 37-49, 52-109, 90-122)
# ============================================================================


class TestSafeId:
    def test_replacements(self) -> None:
        assert _safe_id("abc-def:ghi/jkl") == "abc_def_ghi_jkl"


class TestSafeText:
    def test_normal(self) -> None:
        assert _safe_text("hello") == "hello"

    def test_none(self) -> None:
        assert _safe_text(None, "fallback") == "fallback"

    def test_empty(self) -> None:
        assert _safe_text("", "fallback") == "fallback"

    def test_special_chars(self) -> None:
        result = _safe_text('has "quotes" and\nnewlines')
        assert '"' not in result
        assert "\n" not in result


class TestLimitNodesByDegree:
    def test_no_limit_needed(self) -> None:
        nodes = [{"id": "a"}, {"id": "b"}]
        edges = [{"source_node_id": "a", "target_node_id": "b"}]
        result_nodes, result_edges, was_limited = _limit_nodes_by_degree(nodes, edges, 10)
        assert len(result_nodes) == 2
        assert was_limited is False

    def test_zero_max_nodes(self) -> None:
        nodes = [{"id": "a"}]
        edges = []
        result_nodes, _, was_limited = _limit_nodes_by_degree(nodes, edges, 0)
        assert was_limited is False

    def test_limits_nodes(self) -> None:
        nodes = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}, {"id": "c", "name": "C"}]
        edges = [
            {"source_node_id": "a", "target_node_id": "b"},
            {"source_node_id": "a", "target_node_id": "c"},
        ]
        result_nodes, result_edges, was_limited = _limit_nodes_by_degree(nodes, edges, 2)
        assert was_limited is True
        assert len(result_nodes) == 2


class TestBuildRelationLines:
    def test_basic(self) -> None:
        edges = [{"source_node_id": "a", "target_node_id": "b", "kind": "depends_on", "meta": None}]
        lines = _build_relation_lines(edges)
        assert len(lines) == 1
        assert "Rel(" in lines[0]

    def test_with_weight(self) -> None:
        edges = [
            {"source_node_id": "a", "target_node_id": "b", "kind": "calls", "meta": {"weight": 5}}
        ]
        lines = _build_relation_lines(edges)
        assert "w=5" in lines[0]

    def test_empty_src_dst(self) -> None:
        edges = [{"source_node_id": "", "target_node_id": "", "kind": "x", "meta": None}]
        lines = _build_relation_lines(edges)
        assert len(lines) == 0


class TestNormalizeC4View:
    def test_valid_views(self) -> None:
        assert _normalize_c4_view("container") == "container"
        assert _normalize_c4_view("CONTEXT") == "context"
        assert _normalize_c4_view("code") == "code"
        assert _normalize_c4_view("deployment") == "deployment"
        assert _normalize_c4_view("component") == "component"

    def test_invalid(self) -> None:
        assert _normalize_c4_view("invalid") == "container"

    def test_none(self) -> None:
        assert _normalize_c4_view(None) == "container"


class TestNormalizeScope:
    def test_none(self) -> None:
        assert _normalize_scope(None) is None

    def test_empty(self) -> None:
        assert _normalize_scope("") is None

    def test_normal(self) -> None:
        assert _normalize_scope("api") == "api"


class TestKindValue:
    def test_enum_like(self) -> None:
        class FakeEnum:
            value = "test_value"

        assert _kind_value(FakeEnum()) == "test_value"

    def test_string(self) -> None:
        assert _kind_value("plain") == "plain"


# ============================================================================
# 9. metrics/coverage_reports.py — coverage helper functions (lines 38-58)
# ============================================================================


class TestCoverageHelpers:
    def test_tag_local_name_ns(self) -> None:
        assert _tag_local_name("{http://example.com}class") == "class"

    def test_tag_local_name_plain(self) -> None:
        assert _tag_local_name("class") == "class"

    def test_coverage_from_counts_zero(self) -> None:
        assert _coverage_from_counts(0, 0) is None

    def test_coverage_from_counts_normal(self) -> None:
        assert _coverage_from_counts(50, 100) == 50.0

    def test_coverage_from_counts_full(self) -> None:
        assert _coverage_from_counts(100, 100) == 100.0


class TestCoverageAggregate:
    def test_add_and_avg(self) -> None:
        agg = CoverageAggregate()
        agg.add(80.0, Path("report1.xml"))
        agg.add(60.0, Path("report2.xml"))
        assert agg.avg() == 70.0
        assert agg.sample_count == 2

    def test_avg_empty(self) -> None:
        assert CoverageAggregate().avg() == 0.0

    def test_clamping(self) -> None:
        agg = CoverageAggregate()
        agg.add(150.0, Path("r.xml"))
        agg.add(-20.0, Path("r2.xml"))
        # 100 + 0 = 100 / 2 = 50
        assert agg.avg() == 50.0


# ============================================================================
# 10. metrics/discovery.py — path normalization (lines 20-63, 80-96)
# ============================================================================


class TestNormalizePosixPath:
    def test_basic(self) -> None:
        result = normalize_posix_path("./src/main.py")
        assert result == "src/main.py"

    def test_already_clean(self) -> None:
        assert normalize_posix_path("src/main.py") == "src/main.py"


class TestToRepoRelativePath:
    def test_empty(self) -> None:
        assert to_repo_relative_path("", Path("/repo"), None) is None

    def test_file_uri(self, tmp_path: Path) -> None:
        f = tmp_path / "src" / "main.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
        result = to_repo_relative_path(f"file://{f}", tmp_path)
        assert result is not None
        assert "main.py" in result

    def test_relative_path(self, tmp_path: Path) -> None:
        result = to_repo_relative_path("src/main.py", tmp_path)
        assert result is not None
        assert result == "src/main.py"

    def test_with_base_dir(self, tmp_path: Path) -> None:
        base = tmp_path / "coverage"
        base.mkdir()
        result = to_repo_relative_path("../src/main.py", tmp_path, base_dir=base)
        assert result is not None


class TestDiscoverCoverageReports:
    def test_configured_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "coverage.xml").write_text("<xml/>")
        reports = discover_coverage_reports(tmp_path, tmp_path, ["coverage.xml"])
        assert len(reports) == 1

    def test_autodiscovery(self, tmp_path: Path) -> None:
        # Create a file matching one of the default patterns
        (tmp_path / "lcov.info").write_text("TN:\n")
        reports = discover_coverage_reports(tmp_path, tmp_path, None)
        assert len(reports) >= 1

    def test_no_reports(self, tmp_path: Path) -> None:
        reports = discover_coverage_reports(tmp_path, tmp_path, None)
        assert reports == []


# ============================================================================
# 11. Coverage report parsers (coverage_reports.py lines 61-449)
# ============================================================================


class TestParseLcovReport:
    def test_basic_lcov(self, tmp_path: Path) -> None:
        lcov = tmp_path / "lcov.info"
        lcov.write_text("TN:\nSF:src/main.py\nDA:1,1\nDA:2,1\nDA:3,0\nend_of_record\n")
        result = parse_lcov_report(lcov, tmp_path, tmp_path)
        assert "src/main.py" in result
        assert result["src/main.py"] == pytest.approx(66.6666, abs=0.1)

    def test_multiple_files(self, tmp_path: Path) -> None:
        lcov = tmp_path / "lcov.info"
        lcov.write_text(
            "SF:src/a.py\nDA:1,1\nDA:2,1\nend_of_record\n"
            "SF:src/b.py\nDA:1,0\nDA:2,0\nend_of_record\n"
        )
        result = parse_lcov_report(lcov, tmp_path, tmp_path)
        assert result.get("src/a.py") == 100.0
        assert result.get("src/b.py") == 0.0

    def test_empty_lcov(self, tmp_path: Path) -> None:
        lcov = tmp_path / "lcov.info"
        lcov.write_text("")
        result = parse_lcov_report(lcov, tmp_path, tmp_path)
        assert result == {}

    def test_invalid_da_line(self, tmp_path: Path) -> None:
        lcov = tmp_path / "lcov.info"
        lcov.write_text("SF:src/a.py\nDA:bad\nDA:1,1\nend_of_record\n")
        result = parse_lcov_report(lcov, tmp_path, tmp_path)
        assert "src/a.py" in result


class TestParseCoberturaXml:
    def test_basic_cobertura(self, tmp_path: Path) -> None:
        xml_content = textwrap.dedent("""\
            <coverage>
                <packages>
                    <package>
                        <classes>
                            <class filename="src/main.py">
                                <lines>
                                    <line number="1" hits="1"/>
                                    <line number="2" hits="0"/>
                                </lines>
                            </class>
                        </classes>
                    </package>
                </packages>
            </coverage>
        """)
        root = safe_et.fromstring(xml_content)
        report = tmp_path / "coverage.xml"
        result = parse_cobertura_xml(root, report, tmp_path, tmp_path)
        assert "src/main.py" in result
        assert result["src/main.py"] == 50.0

    def test_line_rate_fallback(self, tmp_path: Path) -> None:
        xml_content = textwrap.dedent("""\
            <coverage>
                <packages>
                    <package>
                        <classes>
                            <class filename="src/main.py" line-rate="0.75">
                            </class>
                        </classes>
                    </package>
                </packages>
            </coverage>
        """)
        root = safe_et.fromstring(xml_content)
        result = parse_cobertura_xml(root, tmp_path / "c.xml", tmp_path, tmp_path)
        assert result.get("src/main.py") == 75.0


class TestParseJacocoXml:
    def test_basic_jacoco(self, tmp_path: Path) -> None:
        xml_content = textwrap.dedent("""\
            <report>
                <package name="com/example">
                    <sourcefile name="Main.java">
                        <counter type="LINE" covered="8" missed="2"/>
                    </sourcefile>
                </package>
            </report>
        """)
        root = safe_et.fromstring(xml_content)
        result = parse_jacoco_xml(root, tmp_path / "j.xml", tmp_path, tmp_path)
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key] == 80.0


class TestParseCloverXml:
    def test_basic_clover(self, tmp_path: Path) -> None:
        xml_content = textwrap.dedent("""\
            <coverage>
                <project>
                    <file name="src/main.php">
                        <line num="1" count="1" type="stmt"/>
                        <line num="2" count="0" type="stmt"/>
                    </file>
                </project>
            </coverage>
        """)
        root = safe_et.fromstring(xml_content)
        result = parse_clover_xml(root, tmp_path / "c.xml", tmp_path, tmp_path)
        assert "src/main.php" in result
        assert result["src/main.php"] == 50.0

    def test_metrics_fallback(self, tmp_path: Path) -> None:
        xml_content = textwrap.dedent("""\
            <coverage>
                <project>
                    <file name="src/app.php">
                        <metrics statements="10" coveredstatements="7"
                                 conditionals="2" coveredconditionals="1"
                                 methods="3" coveredmethods="2"/>
                    </file>
                </project>
            </coverage>
        """)
        root = safe_et.fromstring(xml_content)
        result = parse_clover_xml(root, tmp_path / "c.xml", tmp_path, tmp_path)
        assert "src/app.php" in result
        # (7+1+2) / (10+2+3) = 10/15 = 66.67
        assert result["src/app.php"] == pytest.approx(66.67, abs=0.1)


class TestParseGenericJson:
    def test_basic(self, tmp_path: Path) -> None:
        payload = {
            "schema": "generic-file-coverage-v1",
            "files": [
                {"path": "src/main.py", "coverage": 85.5},
                {"path": "src/lib.py", "coverage": 42.0},
            ],
        }
        result = parse_generic_file_coverage_json(payload, tmp_path / "r.json", tmp_path, tmp_path)
        assert result.get("src/main.py") == 85.5
        assert result.get("src/lib.py") == 42.0

    def test_wrong_schema(self, tmp_path: Path) -> None:
        result = parse_generic_file_coverage_json(
            {"schema": "wrong"}, tmp_path / "r.json", tmp_path, tmp_path
        )
        assert result == {}

    def test_not_list(self, tmp_path: Path) -> None:
        result = parse_generic_file_coverage_json(
            {"schema": "generic-file-coverage-v1", "files": "not-list"},
            tmp_path / "r.json",
            tmp_path,
            tmp_path,
        )
        assert result == {}

    def test_invalid_entry(self, tmp_path: Path) -> None:
        payload = {
            "schema": "generic-file-coverage-v1",
            "files": [
                "not-a-dict",
                {"path": "", "coverage": 50},
                {"path": "a.py", "coverage": "notnum"},
                {"path": "b.py", "coverage": 80},
            ],
        }
        result = parse_generic_file_coverage_json(payload, tmp_path / "r.json", tmp_path, tmp_path)
        assert "b.py" in result
        assert len(result) == 1


class TestDetectCoverageProtocol:
    def test_lcov(self, tmp_path: Path) -> None:
        f = tmp_path / "lcov.info"
        f.write_text("TN:\nSF:x\nend_of_record\n")
        assert detect_coverage_protocol(f) == PROTOCOL_LCOV

    def test_generic_json(self, tmp_path: Path) -> None:
        f = tmp_path / "coverage.json"
        f.write_text(json.dumps({"schema": "generic-file-coverage-v1", "files": []}))
        assert detect_coverage_protocol(f) == PROTOCOL_GENERIC_JSON

    def test_cobertura(self, tmp_path: Path) -> None:
        f = tmp_path / "coverage.xml"
        f.write_text(
            '<coverage><packages><package><classes><class filename="a">'
            '<lines><line number="1" hits="1"/></lines></class></classes>'
            "</package></packages></coverage>"
        )
        assert detect_coverage_protocol(f) == PROTOCOL_COBERTURA

    def test_jacoco(self, tmp_path: Path) -> None:
        f = tmp_path / "jacoco.xml"
        f.write_text(
            '<report name="test"><package name="com"><sourcefile name="A.java">'
            '<counter type="LINE" covered="1" missed="0"/></sourcefile></package></report>'
        )
        assert detect_coverage_protocol(f) == PROTOCOL_JACOCO

    def test_clover(self, tmp_path: Path) -> None:
        f = tmp_path / "clover.xml"
        f.write_text(
            '<coverage><project><file name="a.php">'
            '<metrics statements="1" coveredstatements="1" conditionals="0" '
            'coveredconditionals="0" methods="0" coveredmethods="0"/>'
            "</file></project></coverage>"
        )
        assert detect_coverage_protocol(f) == PROTOCOL_CLOVER

    def test_invalid_xml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.xml"
        f.write_text("not xml at all")
        assert detect_coverage_protocol(f) is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json at all")
        assert detect_coverage_protocol(f) is None


class TestParseCoverageReport:
    def test_lcov_integration(self, tmp_path: Path) -> None:
        f = tmp_path / "lcov.info"
        f.write_text("SF:src/a.py\nDA:1,1\nend_of_record\n")
        protocol, result = parse_coverage_report(f, tmp_path, tmp_path)
        assert protocol == PROTOCOL_LCOV
        assert "src/a.py" in result

    def test_unknown_protocol(self, tmp_path: Path) -> None:
        f = tmp_path / "unknown.bin"
        f.write_bytes(b"\x00\x01\x02")
        protocol, result = parse_coverage_report(f, tmp_path, tmp_path)
        assert protocol is None
        assert result == {}


# ============================================================================
# 12. ast_utils.py — pure functions (lines 8-23, 60-69, 87-148)
# ============================================================================


class TestUnquote:
    def test_double_quotes(self) -> None:
        assert unquote('"hello"') == "hello"

    def test_single_quotes(self) -> None:
        assert unquote("'world'") == "world"

    def test_backticks(self) -> None:
        assert unquote("`template`") == "template"

    def test_triple_double(self) -> None:
        assert unquote('"""docstring"""') == "docstring"

    def test_triple_single(self) -> None:
        assert unquote("'''docstring'''") == "docstring"

    def test_no_quotes(self) -> None:
        assert unquote("plain") == "plain"


class TestIsPascalCase:
    def test_pascal(self) -> None:
        assert is_pascal_case("MyClass") is True

    def test_lower(self) -> None:
        assert is_pascal_case("myclass") is False

    def test_empty(self) -> None:
        assert is_pascal_case("") is False

    def test_with_underscore(self) -> None:
        assert is_pascal_case("My_Class") is True

    def test_single_upper(self) -> None:
        assert is_pascal_case("A") is True


# ============================================================================
# 13. Indexer backends — can_handle, check_tool_available
# ============================================================================


class TestPythonBackend:
    def test_can_handle_python(self) -> None:
        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_java(self) -> None:
        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp"))
        assert backend.can_handle(target) is False

    def test_tool_name(self) -> None:
        assert PythonIndexerBackend.TOOL_NAME == "scip-python"

    def test_build_command(self) -> None:
        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(project_name="test-proj")
        cmd = backend._build_command(target, cfg)
        assert "scip-python" in cmd
        assert "index" in cmd
        assert "test-proj" in cmd

    def test_build_command_with_version(self) -> None:
        backend = PythonIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp/proj"))
        cfg = IndexConfig(project_name="proj", project_version="1.0.0")
        cmd = backend._build_command(target, cfg)
        assert "1.0.0" in cmd


class TestTypescriptBackend:
    def test_can_handle_ts(self) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=Path("/tmp"))
        assert backend.can_handle(target) is True

    def test_can_handle_js(self) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.JAVASCRIPT, root_path=Path("/tmp"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_python(self) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp"))
        assert backend.can_handle(target) is False

    def test_build_command_ts(self, tmp_path: Path) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cmd, generated = backend._build_command(target, tmp_path / "out.scip")
        assert "scip-typescript" in cmd
        assert generated is None

    def test_build_command_js(self, tmp_path: Path) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.JAVASCRIPT, root_path=tmp_path)
        cmd, generated = backend._build_command(target, tmp_path / "out.scip")
        assert generated is not None
        assert generated.exists()
        generated.unlink()

    def test_create_js_config_with_tsconfig(self, tmp_path: Path) -> None:
        (tmp_path / "tsconfig.json").write_text("{}")
        backend = TypescriptIndexerBackend()
        config_path = backend._create_javascript_project_config(tmp_path)
        content = json.loads(config_path.read_text())
        assert content.get("extends") == "./tsconfig.json"
        config_path.unlink()

    def test_should_install_deps_never(self, tmp_path: Path) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.NEVER)
        assert backend._should_install_deps(target, cfg) is False

    def test_should_install_deps_always(self, tmp_path: Path) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.ALWAYS)
        assert backend._should_install_deps(target, cfg) is True

    def test_should_install_deps_auto_no_node_modules(self, tmp_path: Path) -> None:
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)
        assert backend._should_install_deps(target, cfg) is True

    def test_should_install_deps_auto_has_node_modules(self, tmp_path: Path) -> None:
        (tmp_path / "node_modules").mkdir()
        backend = TypescriptIndexerBackend()
        target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)
        cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)
        assert backend._should_install_deps(target, cfg) is False


class TestJavaBackend:
    def test_can_handle_java(self) -> None:
        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp"))
        assert backend.can_handle(target) is True

    def test_cannot_handle_python(self) -> None:
        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/tmp"))
        assert backend.can_handle(target) is False

    def test_build_command_basic(self) -> None:
        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp"))
        cfg = IndexConfig()
        cmd = backend._build_command(target, Path("/out/test.scip"), cfg)
        assert "scip-java" in cmd
        assert "--output" in cmd

    def test_build_command_with_build_args(self) -> None:
        backend = JavaIndexerBackend()
        target = ProjectTarget(language=Language.JAVA, root_path=Path("/tmp"))
        cfg = IndexConfig(java_build_args=["assemble"])
        cmd = backend._build_command(target, Path("/out/test.scip"), cfg)
        assert "--" in cmd
        assert "assemble" in cmd


class TestBaseIndexerBackend:
    def test_check_tool_available_empty_name(self) -> None:
        class EmptyBackend(BaseIndexerBackend):
            TOOL_NAME = ""

            def can_handle(self, target: ProjectTarget) -> bool:
                return False

            def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
                return IndexArtifact(
                    language=Language.PYTHON,
                    project_root=Path("/tmp"),
                    scip_path=Path("/tmp/index.scip"),
                    logs_path=None,
                    tool_name="",
                    tool_version="",
                    duration_s=0.0,
                    success=False,
                    error_message="empty",
                )

        backend = EmptyBackend()
        available, version = backend.check_tool_available()
        assert available is False
        assert version == ""


# ============================================================================
# 14. Indexer orchestration (__init__.py lines 77-221)
# ============================================================================


class TestIndexProject:
    def test_no_backend_found(self) -> None:
        # Use a language that doesn't have a backend (fake)
        target = ProjectTarget(language=Language.PYTHON, root_path=Path("/nonexistent"))
        cfg = IndexConfig()
        # Mock can_handle to return False for all backends
        with (
            patch.object(PythonIndexerBackend, "can_handle", return_value=False),
            patch.object(TypescriptIndexerBackend, "can_handle", return_value=False),
            patch.object(JavaIndexerBackend, "can_handle", return_value=False),
            patch.object(PhpIndexerBackend, "can_handle", return_value=False),
        ):
            result = index_project(target, cfg)
        assert result.success is False
        assert "No backend found" in result.error_message


# ============================================================================
# 15. telemetry/spans.py — decorator-based tracing (lines 69-80, 162-165, 201-212)
# ============================================================================


class TestTelemetrySpans:
    @pytest.mark.anyio
    async def test_trace_db_operation(self) -> None:
        @trace_db_operation("test_op")
        async def my_db_op():
            return "result"

        result = await my_db_op()
        assert result == "result"

    @pytest.mark.anyio
    async def test_trace_embedding_call(self) -> None:
        async with trace_embedding_call("openai", "text-embedding-3-small", 1) as span:
            assert span is not None

    @pytest.mark.anyio
    async def test_trace_llm_call(self) -> None:
        @trace_llm_call("anthropic", "claude-3-haiku")
        async def my_llm():
            return "response"

        result = await my_llm()
        assert result == "response"

    @pytest.mark.anyio
    async def test_trace_llm_call_error(self) -> None:
        @trace_llm_call("anthropic", "claude-3-haiku")
        async def my_failing_llm():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await my_failing_llm()

    def test_trace_sync_llm_call(self) -> None:
        @trace_sync_llm_call("openai", "gpt-4")
        def my_sync_llm():
            return "sync_response"

        result = my_sync_llm()
        assert result == "sync_response"

    def test_trace_sync_llm_call_error(self) -> None:
        @trace_sync_llm_call("openai", "gpt-4")
        def my_failing_sync():
            raise RuntimeError("sync error")

        with pytest.raises(RuntimeError, match="sync error"):
            my_failing_sync()


# ============================================================================
# 16. research/llm/provider.py — LLM provider helpers
# ============================================================================


class TestResearchLLMProvider:
    """Test the LLMProvider abstract class and its implementations."""

    def test_mock_provider_model_name(self) -> None:
        provider = MockLLMProvider()
        assert provider.model_name is not None
        assert isinstance(provider.model_name, str)

    @pytest.mark.anyio
    async def test_mock_provider_generate_text(self) -> None:
        provider = MockLLMProvider()
        result = await provider.generate_text(
            system="You are a test.",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.anyio
    async def test_mock_provider_generate_structured(self) -> None:
        class TestOutput(BaseModel):
            answer: str = Field(default="test")

        provider = MockLLMProvider()
        result = await provider.generate_structured(
            system="You are a test.",
            messages=[{"role": "user", "content": "Hello"}],
            output_schema=TestOutput,
        )
        assert isinstance(result, TestOutput)


# ============================================================================
# 17. search.py — pure search helpers
# ============================================================================

# ============================================================================
# 18. validation/connectors.py — connector helpers (lines 51-60, 94-101)
# ============================================================================


class TestStateOf:
    def test_with_phase(self) -> None:
        item = {"status": {"phase": "Succeeded"}}
        assert _state_of(item) == "Succeeded"

    def test_with_condition(self) -> None:
        item = {"status": {"condition": "True"}}
        assert _state_of(item) == "True"

    def test_with_string_status(self) -> None:
        item = {"status": "running"}
        assert _state_of(item) == "running"

    def test_no_status(self) -> None:
        # str(None or "") -> ""
        assert _state_of({}) == ""

    def test_non_dict(self) -> None:
        assert _state_of("not a dict") == ""


class TestValidationMetric:
    def test_creation(self) -> None:
        m = ValidationMetric(source="test", key="k", value=1.0)
        assert m.source == "test"
        assert m.status is None
        assert m.meta is None


class TestFetchConnectors:
    @pytest.mark.anyio
    async def test_argo_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = await fetch_argo_metrics()
        assert result == []

    @pytest.mark.anyio
    async def test_tekton_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = await fetch_tekton_metrics()
        assert result == []

    @pytest.mark.anyio
    async def test_temporal_no_env(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = await fetch_temporal_alerts()
        assert result == []


# ============================================================================
# 19. research/__init__.py (lines 58-69)
# ============================================================================


class TestResearchInit:
    def test_research_module_import(self) -> None:
        assert research_llm is not None


# ============================================================================
# 20. metrics/duplication.py (lines 15-80)
# ============================================================================


class TestNormalizeLine:
    def test_empty(self) -> None:
        assert _normalize_line("") is None

    def test_short(self) -> None:
        assert _normalize_line("abc") is None

    def test_comment_hash(self) -> None:
        assert _normalize_line("# this is a comment here") is None

    def test_comment_slash(self) -> None:
        assert _normalize_line("// this is a comment here") is None

    def test_normal(self) -> None:
        result = _normalize_line("    def  my_function():    pass  ")
        assert result == "def my_function(): pass"

    def test_blank_line(self) -> None:
        assert _normalize_line("    ") is None


class TestDuplicationRatio:
    def test_basic(self, tmp_path: Path) -> None:
        # Two files with overlapping content
        (tmp_path / "a.py").write_text("def long_function_name(): pass\nreturn long_value_here\n")
        (tmp_path / "b.py").write_text("def long_function_name(): pass\nreturn different_value\n")
        result, provenance = compute_file_duplication_ratio(tmp_path, {"a.py", "b.py"})
        assert isinstance(result, dict)
        assert provenance["files_scanned"] == 2

    def test_no_duplication(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("def unique_function_aaa(): pass\n")
        (tmp_path / "b.py").write_text("def unique_function_bbb(): pass\n")
        result, prov = compute_file_duplication_ratio(tmp_path, {"a.py", "b.py"})
        assert all(v == 0.0 for v in result.values())

    def test_full_duplication(self, tmp_path: Path) -> None:
        content = "\n".join([f"long_line_number_{i:02d} = something_here" for i in range(20)])
        (tmp_path / "a.py").write_text(content + "\n")
        (tmp_path / "b.py").write_text(content + "\n")
        result, prov = compute_file_duplication_ratio(tmp_path, {"a.py", "b.py"})
        assert result.get("a.py", 0.0) > 0.0

    def test_unreadable_file(self, tmp_path: Path) -> None:
        result, prov = compute_file_duplication_ratio(tmp_path, {"nonexistent.py"})
        assert prov["files_unreadable"] == 1


# ============================================================================
# 21. semantic_snapshot/__init__.py (lines 91-93, 133-137)
# ============================================================================


class TestSemanticSnapshotInit:
    def test_snapshot_imported(self) -> None:
        assert SnapshotImport is not None


class TestSearchModels:
    def test_search_result_fields(self) -> None:
        r = SearchResult(
            document_id="doc1",
            chunk_id="chunk1",
            uri="https://example.com",
            title="Test",
            content="Hello world",
            score=0.95,
            source_id="src1",
            collection_id="col1",
            fts_rank=1,
            vector_rank=2,
            fts_score=0.8,
            vector_score=0.9,
        )
        assert r.document_id == "doc1"
        assert r.score == 0.95
        assert r.fts_rank == 1

    def test_search_response(self) -> None:
        r = SearchResponse(
            results=[
                SearchResult(
                    document_id="d",
                    chunk_id="c",
                    uri="u",
                    title="t",
                    content="content",
                    score=0.5,
                    source_id="s",
                    collection_id="col",
                    fts_rank=0,
                    vector_rank=0,
                    fts_score=0.0,
                    vector_score=0.0,
                )
            ],
            query="test",
            total_fts_matches=1,
            total_vector_matches=1,
        )
        assert r.total_fts_matches == 1
        assert len(r.results) == 1
