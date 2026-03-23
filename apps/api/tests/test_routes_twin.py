"""Tests for twin route helper functions and route-level validation.

Focuses on pure utility/helper functions that can be tested without a database,
plus route-level input validation and error paths.
"""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Pure helper function tests (no DB, no HTTP)
# ---------------------------------------------------------------------------


class TestParseCollectionId:
    """Tests for _parse_collection_id helper."""

    def test_valid_uuid(self) -> None:
        from app.routes.twin import _parse_collection_id

        test_id = uuid.uuid4()
        result = _parse_collection_id(str(test_id))
        assert result == test_id

    def test_invalid_uuid_raises_400(self) -> None:
        from app.routes.twin import _parse_collection_id
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_collection_id("not-a-uuid")
        assert exc_info.value.status_code == 400
        assert "Invalid collection_id" in exc_info.value.detail


class TestParseOptionalScenarioId:
    """Tests for _parse_optional_scenario_id helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_optional_scenario_id

        assert _parse_optional_scenario_id(None) is None

    def test_empty_string_returns_none(self) -> None:
        from app.routes.twin import _parse_optional_scenario_id

        assert _parse_optional_scenario_id("") is None

    def test_valid_uuid(self) -> None:
        from app.routes.twin import _parse_optional_scenario_id

        test_id = uuid.uuid4()
        result = _parse_optional_scenario_id(str(test_id))
        assert result == test_id

    def test_invalid_uuid_raises_400(self) -> None:
        from app.routes.twin import _parse_optional_scenario_id
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_optional_scenario_id("bad-id")
        assert exc_info.value.status_code == 400
        assert "Invalid scenario_id" in exc_info.value.detail


class TestParseEnginesQuery:
    """Tests for _parse_engines_query helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_engines_query

        assert _parse_engines_query(None) is None

    def test_empty_string_returns_none(self) -> None:
        from app.routes.twin import _parse_engines_query

        assert _parse_engines_query("") is None

    def test_single_engine(self) -> None:
        from app.routes.twin import _parse_engines_query

        result = _parse_engines_query("tree-sitter")
        assert result == ["tree-sitter"]

    def test_multiple_engines(self) -> None:
        from app.routes.twin import _parse_engines_query

        result = _parse_engines_query("tree-sitter,semgrep,regex")
        assert result == ["tree-sitter", "semgrep", "regex"]

    def test_whitespace_stripped(self) -> None:
        from app.routes.twin import _parse_engines_query

        result = _parse_engines_query(" tree-sitter , semgrep ")
        assert result == ["tree-sitter", "semgrep"]

    def test_empty_items_skipped(self) -> None:
        from app.routes.twin import _parse_engines_query

        result = _parse_engines_query("tree-sitter,,semgrep,")
        assert result == ["tree-sitter", "semgrep"]


class TestParseLayer:
    """Tests for _parse_layer helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_layer

        assert _parse_layer(None) is None

    def test_empty_string_returns_none(self) -> None:
        from app.routes.twin import _parse_layer

        assert _parse_layer("") is None

    def test_valid_layer(self) -> None:
        from app.routes.twin import _parse_layer
        from contextmine_core.models import TwinLayer

        result = _parse_layer("portfolio_system")
        assert result == TwinLayer.PORTFOLIO_SYSTEM

    def test_invalid_layer_raises_400(self) -> None:
        from app.routes.twin import _parse_layer
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_layer("invalid_layer")
        assert exc_info.value.status_code == 400
        assert "Invalid layer" in exc_info.value.detail


class TestParseProjection:
    """Tests for _parse_projection helper."""

    def test_none_returns_code_symbol(self) -> None:
        from app.routes.twin import _parse_projection
        from contextmine_core.twin import GraphProjection

        result = _parse_projection(None)
        assert result == GraphProjection.CODE_SYMBOL

    def test_empty_returns_code_symbol(self) -> None:
        from app.routes.twin import _parse_projection
        from contextmine_core.twin import GraphProjection

        result = _parse_projection("")
        assert result == GraphProjection.CODE_SYMBOL

    def test_invalid_raises_400(self) -> None:
        from app.routes.twin import _parse_projection
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_projection("invalid_projection")
        assert exc_info.value.status_code == 400
        assert "Invalid projection" in exc_info.value.detail


class TestParseC4View:
    """Tests for _parse_c4_view helper."""

    def test_none_returns_container(self) -> None:
        from app.routes.twin import _parse_c4_view

        assert _parse_c4_view(None) == "container"

    def test_valid_views(self) -> None:
        from app.routes.twin import _parse_c4_view

        for view in ("context", "container", "component", "code", "deployment"):
            assert _parse_c4_view(view) == view

    def test_case_insensitive(self) -> None:
        from app.routes.twin import _parse_c4_view

        assert _parse_c4_view("CONTAINER") == "container"
        assert _parse_c4_view("Context") == "context"

    def test_whitespace_stripped(self) -> None:
        from app.routes.twin import _parse_c4_view

        assert _parse_c4_view("  container  ") == "container"

    def test_invalid_view_raises_400(self) -> None:
        from app.routes.twin import _parse_c4_view
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_c4_view("invalid_view")
        assert exc_info.value.status_code == 400
        assert "Invalid c4_view" in exc_info.value.detail


class TestParseC4Scope:
    """Tests for _parse_c4_scope helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_c4_scope

        assert _parse_c4_scope(None) is None

    def test_empty_returns_none(self) -> None:
        from app.routes.twin import _parse_c4_scope

        assert _parse_c4_scope("") is None
        assert _parse_c4_scope("   ") is None

    def test_valid_scope(self) -> None:
        from app.routes.twin import _parse_c4_scope

        assert _parse_c4_scope("billing") == "billing"

    def test_control_characters_raise_400(self) -> None:
        from app.routes.twin import _parse_c4_scope
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_c4_scope("bad\x00scope")
        assert exc_info.value.status_code == 400
        assert "Invalid c4_scope" in exc_info.value.detail

    def test_too_long_raises_400(self) -> None:
        from app.routes.twin import _parse_c4_scope
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_c4_scope("x" * 256)
        assert exc_info.value.status_code == 400
        assert "Invalid c4_scope" in exc_info.value.detail


class TestParseKindFilter:
    """Tests for _parse_kind_filter helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_kind_filter

        assert _parse_kind_filter(None) is None

    def test_empty_returns_none(self) -> None:
        from app.routes.twin import _parse_kind_filter

        assert _parse_kind_filter("") is None

    def test_single_kind(self) -> None:
        from app.routes.twin import _parse_kind_filter

        result = _parse_kind_filter("file")
        assert result == {"file"}

    def test_multiple_kinds(self) -> None:
        from app.routes.twin import _parse_kind_filter

        result = _parse_kind_filter("file,symbol,db_table")
        assert result == {"file", "symbol", "db_table"}

    def test_lowercase_normalization(self) -> None:
        from app.routes.twin import _parse_kind_filter

        result = _parse_kind_filter("FILE,Symbol")
        assert result == {"file", "symbol"}


class TestParseKnowledgeNodeKinds:
    """Tests for _parse_knowledge_node_kinds helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_knowledge_node_kinds

        assert _parse_knowledge_node_kinds(None) is None

    def test_valid_kinds(self) -> None:
        from app.routes.twin import _parse_knowledge_node_kinds
        from contextmine_core.models import KnowledgeNodeKind

        result = _parse_knowledge_node_kinds("file,symbol")
        assert KnowledgeNodeKind.FILE in result
        assert KnowledgeNodeKind.SYMBOL in result

    def test_invalid_kind_raises_400(self) -> None:
        from app.routes.twin import _parse_knowledge_node_kinds
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_knowledge_node_kinds("file,totally_bogus")
        assert exc_info.value.status_code == 400
        assert "Invalid knowledge node kind" in exc_info.value.detail


class TestParseKnowledgeEdgeKinds:
    """Tests for _parse_knowledge_edge_kinds helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_knowledge_edge_kinds

        assert _parse_knowledge_edge_kinds(None) is None

    def test_invalid_kind_raises_400(self) -> None:
        from app.routes.twin import _parse_knowledge_edge_kinds
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_knowledge_edge_kinds("made_up_edge")
        assert exc_info.value.status_code == 400
        assert "Invalid knowledge edge kind" in exc_info.value.detail


class TestParseGraphragCommunityMode:
    """Tests for _parse_graphrag_community_mode helper."""

    def test_none_returns_color(self) -> None:
        from app.routes.twin import _parse_graphrag_community_mode

        assert _parse_graphrag_community_mode(None) == "color"

    def test_valid_modes(self) -> None:
        from app.routes.twin import _parse_graphrag_community_mode

        assert _parse_graphrag_community_mode("none") == "none"
        assert _parse_graphrag_community_mode("color") == "color"
        assert _parse_graphrag_community_mode("focus") == "focus"

    def test_case_insensitive(self) -> None:
        from app.routes.twin import _parse_graphrag_community_mode

        assert _parse_graphrag_community_mode("COLOR") == "color"

    def test_invalid_raises_400(self) -> None:
        from app.routes.twin import _parse_graphrag_community_mode
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_graphrag_community_mode("invalid")
        assert exc_info.value.status_code == 400
        assert "Invalid community_mode" in exc_info.value.detail


class TestParseSemanticMapMode:
    """Tests for _parse_semantic_map_mode helper."""

    def test_none_returns_code_structure(self) -> None:
        from app.routes.twin import _parse_semantic_map_mode

        assert _parse_semantic_map_mode(None) == "code_structure"

    def test_valid_modes(self) -> None:
        from app.routes.twin import _parse_semantic_map_mode

        assert _parse_semantic_map_mode("code_structure") == "code_structure"
        assert _parse_semantic_map_mode("semantic") == "semantic"

    def test_invalid_raises_400(self) -> None:
        from app.routes.twin import _parse_semantic_map_mode
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            _parse_semantic_map_mode("invalid")
        assert exc_info.value.status_code == 400
        assert "Invalid map_mode" in exc_info.value.detail


class TestParsePgvectorText:
    """Tests for _parse_pgvector_text helper."""

    def test_none_returns_empty(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        assert _parse_pgvector_text(None) == []

    def test_empty_string_returns_empty(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        assert _parse_pgvector_text("") == []
        assert _parse_pgvector_text("[]") == []

    def test_bracket_notation(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        result = _parse_pgvector_text("[1.0,2.0,3.0]")
        assert result == [1.0, 2.0, 3.0]

    def test_comma_separated(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        result = _parse_pgvector_text("1.0,2.0,3.0")
        assert result == [1.0, 2.0, 3.0]

    def test_invalid_values_skipped(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        result = _parse_pgvector_text("1.0,bad,3.0")
        assert result == [1.0, 3.0]


class TestExtractDomainToken:
    """Tests for _extract_domain_token helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _extract_domain_token

        assert _extract_domain_token(None) is None

    def test_empty_returns_none(self) -> None:
        from app.routes.twin import _extract_domain_token

        assert _extract_domain_token("") is None

    def test_file_path_extracts_directory(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("file:src/billing/invoice.py")
        assert result == "billing"

    def test_symbol_path_extracts_directory(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("symbol:src/billing/invoice.py:create_invoice")
        # The function processes "src/billing/invoice.py:create_invoice" by removing prefix
        # and extracting the second-to-last meaningful path segment
        assert result is not None

    def test_ignores_common_prefixes(self) -> None:
        from app.routes.twin import _extract_domain_token

        # src, lib, app etc. are in the ignored set
        result = _extract_domain_token("src/billing/main.py")
        assert result == "billing"

    def test_url_extracts_path_segment(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("https://example.com/docs/billing/api.html")
        assert result == "billing"


class TestExtractDocumentLines:
    """Tests for _extract_document_lines helper."""

    def test_none_content(self) -> None:
        from app.routes.twin import _extract_document_lines

        assert _extract_document_lines(None, 1, 5) == ""

    def test_empty_content(self) -> None:
        from app.routes.twin import _extract_document_lines

        assert _extract_document_lines("", 1, 5) == ""

    def test_extracts_lines(self) -> None:
        from app.routes.twin import _extract_document_lines

        content = "line1\nline2\nline3\nline4\nline5"
        result = _extract_document_lines(content, 2, 4)
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        assert "line1" not in result
        assert "line5" not in result

    def test_start_line_clamped_to_1(self) -> None:
        from app.routes.twin import _extract_document_lines

        content = "line1\nline2\nline3"
        result = _extract_document_lines(content, 0, 2)
        assert "line1" in result

    def test_start_beyond_content_returns_empty(self) -> None:
        from app.routes.twin import _extract_document_lines

        content = "line1\nline2"
        result = _extract_document_lines(content, 100, 200)
        assert result == ""

    def test_truncation(self) -> None:
        from app.routes.twin import _extract_document_lines

        content = "\n".join(f"line{i}" for i in range(100))
        result = _extract_document_lines(content, 1, 100, max_chars=50)
        assert len(result) <= 53  # 50 chars + "..."
        assert result.endswith("...")


class TestSerializeScenario:
    """Tests for _serialize_scenario helper."""

    def test_serialize_with_base_scenario(self) -> None:
        from app.routes.twin import _serialize_scenario

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "Test Scenario"
        scenario.version = 2
        scenario.is_as_is = False
        scenario.base_scenario_id = uuid.uuid4()

        result = _serialize_scenario(scenario)
        assert result["id"] == str(scenario.id)
        assert result["collection_id"] == str(scenario.collection_id)
        assert result["name"] == "Test Scenario"
        assert result["version"] == 2
        assert result["is_as_is"] is False
        assert result["base_scenario_id"] == str(scenario.base_scenario_id)

    def test_serialize_without_base_scenario(self) -> None:
        from app.routes.twin import _serialize_scenario

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "AS-IS"
        scenario.version = 1
        scenario.is_as_is = True
        scenario.base_scenario_id = None

        result = _serialize_scenario(scenario)
        assert result["base_scenario_id"] is None
        assert result["is_as_is"] is True


class TestSafeMeta:
    """Tests for _safe_meta helper."""

    def test_dict_returned_as_copy(self) -> None:
        from app.routes.twin import _safe_meta

        original = {"key": "value"}
        result = _safe_meta(original)
        assert result == {"key": "value"}
        # Should be a copy, not the same object
        result["new_key"] = "new_value"
        assert "new_key" not in original

    def test_non_dict_returns_empty_dict(self) -> None:
        from app.routes.twin import _safe_meta

        assert _safe_meta(None) == {}
        assert _safe_meta("string") == {}
        assert _safe_meta(42) == {}
        assert _safe_meta([]) == {}


class TestTruncateText:
    """Tests for _truncate_text helper."""

    def test_short_text_unchanged(self) -> None:
        from app.routes.twin import _truncate_text

        assert _truncate_text("hello", max_chars=100) == "hello"

    def test_long_text_truncated(self) -> None:
        from app.routes.twin import _truncate_text

        result = _truncate_text("a" * 100, max_chars=50)
        assert result.endswith("...")
        # Truncated to 50 chars + "..."
        assert len(result) <= 53


class TestEscapeLikePattern:
    """Tests for _escape_like_pattern helper."""

    def test_no_special_chars(self) -> None:
        from app.routes.twin import _escape_like_pattern

        assert _escape_like_pattern("hello") == "hello"

    def test_backslash_escaped(self) -> None:
        from app.routes.twin import _escape_like_pattern

        assert _escape_like_pattern("a\\b") == "a\\\\b"

    def test_percent_escaped(self) -> None:
        from app.routes.twin import _escape_like_pattern

        assert _escape_like_pattern("100%") == "100\\%"

    def test_underscore_escaped(self) -> None:
        from app.routes.twin import _escape_like_pattern

        assert _escape_like_pattern("a_b") == "a\\_b"


class TestSha256Text:
    """Tests for _sha256_text helper."""

    def test_deterministic(self) -> None:
        from app.routes.twin import _sha256_text

        assert _sha256_text("hello") == _sha256_text("hello")

    def test_different_inputs_different_hashes(self) -> None:
        from app.routes.twin import _sha256_text

        assert _sha256_text("hello") != _sha256_text("world")

    def test_returns_hex_string(self) -> None:
        from app.routes.twin import _sha256_text

        result = _sha256_text("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestArc42ArtifactName:
    """Tests for _arc42_artifact_name helper."""

    def test_format(self) -> None:
        from app.routes.twin import _arc42_artifact_name

        test_id = uuid.uuid4()
        result = _arc42_artifact_name(test_id)
        assert result == f"{test_id}.arc42.md"


class TestCosineeSimilarity:
    """Tests for _cosine_similarity helper."""

    def test_identical_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-6

    def test_empty_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        assert _cosine_similarity([], []) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0
        assert _cosine_similarity([], [1.0]) == 0.0


class TestJaccard:
    """Tests for _jaccard helper."""

    def test_identical_sets(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_both_empty(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard({"a"}, set()) == 0.0

    def test_partial_overlap(self) -> None:
        from app.routes.twin import _jaccard

        result = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 0.5) < 1e-6  # 2/4 = 0.5


class TestNormalizeXY:
    """Tests for _normalize_xy helper."""

    def test_empty_points(self) -> None:
        from app.routes.twin import _normalize_xy

        points: list[dict[str, Any]] = []
        _normalize_xy(points)
        assert points == []

    def test_single_point(self) -> None:
        from app.routes.twin import _normalize_xy

        points = [{"x": 5.0, "y": 3.0}]
        _normalize_xy(points)
        # Single point normalizes to [-1, -1] due to (val - min) / span * 2 - 1
        # with span = 1e-9 (minimum), result is 0*2 - 1 = -1
        assert points[0]["x"] == -1.0
        assert points[0]["y"] == -1.0

    def test_multiple_points_normalized_to_range(self) -> None:
        from app.routes.twin import _normalize_xy

        points = [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 10.0},
            {"x": 5.0, "y": 5.0},
        ]
        _normalize_xy(points)
        # First point: min, should be -1
        assert points[0]["x"] == -1.0
        assert points[0]["y"] == -1.0
        # Last point: max, should be 1
        assert points[1]["x"] == 1.0
        assert points[1]["y"] == 1.0
        # Middle point: should be 0
        assert points[2]["x"] == 0.0
        assert points[2]["y"] == 0.0


class TestLayoutCodeStructurePoints:
    """Tests for _layout_code_structure_points helper."""

    def test_empty(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        assert _layout_code_structure_points([], {}) == {}

    def test_single_point(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        result = _layout_code_structure_points(["a"], {})
        assert result == {"a": (0.0, 0.0)}

    def test_two_points_no_edges(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        result = _layout_code_structure_points(["a", "b"], {})
        assert "a" in result
        assert "b" in result
        # Each position is a tuple of two floats
        assert len(result["a"]) == 2
        assert len(result["b"]) == 2


class TestDetectIsolatedPointIds:
    """Tests for _detect_isolated_point_ids helper."""

    def test_empty(self) -> None:
        from app.routes.twin import _detect_isolated_point_ids

        isolated, distances = _detect_isolated_point_ids([])
        assert isolated == set()
        assert distances == {}

    def test_single_point(self) -> None:
        from app.routes.twin import _detect_isolated_point_ids

        points = [{"id": "a", "x": 0.0, "y": 0.0}]
        isolated, distances = _detect_isolated_point_ids(points)
        assert "a" in isolated

    def test_cluster_with_outlier(self) -> None:
        from app.routes.twin import _detect_isolated_point_ids

        # Three close points and one far away
        points = [
            {"id": "a", "x": 0.0, "y": 0.0},
            {"id": "b", "x": 0.1, "y": 0.0},
            {"id": "c", "x": 0.0, "y": 0.1},
            {"id": "d", "x": 100.0, "y": 100.0},
        ]
        isolated, distances = _detect_isolated_point_ids(points)
        assert "d" in isolated
        # The close cluster points should not be isolated
        assert "a" not in isolated


class TestParseDbTableFromNaturalKey:
    """Tests for _parse_db_table_from_natural_key helper."""

    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key(None) is None

    def test_empty_returns_none(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("") is None

    def test_non_db_prefix_returns_none(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("file:src/main.py") is None

    def test_simple_table(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:users") == "users"

    def test_table_with_column(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:users.id") == "users"

    def test_db_prefix_only_returns_none(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:") is None


class TestTopologyEntityLevel:
    """Tests for _topology_entity_level helper."""

    def test_explicit_overrides(self) -> None:
        from app.routes.twin import _topology_entity_level

        assert _topology_entity_level(None, "file") == "file"

    def test_portfolio_system_layer(self) -> None:
        from app.routes.twin import _topology_entity_level
        from contextmine_core.models import TwinLayer

        assert _topology_entity_level(TwinLayer.PORTFOLIO_SYSTEM, None) == "domain"

    def test_component_interface_layer(self) -> None:
        from app.routes.twin import _topology_entity_level
        from contextmine_core.models import TwinLayer

        assert _topology_entity_level(TwinLayer.COMPONENT_INTERFACE, None) == "component"

    def test_default(self) -> None:
        from app.routes.twin import _topology_entity_level

        assert _topology_entity_level(None, None) == "container"


class TestExtractNeighborhood:
    """Tests for _extract_neighborhood helper."""

    def test_empty_graph(self) -> None:
        from app.routes.twin import _extract_neighborhood

        nodes, edges = _extract_neighborhood([], [], "root", 2, 100)
        assert nodes == []
        assert edges == []

    def test_root_not_found(self) -> None:
        from app.routes.twin import _extract_neighborhood

        nodes = [{"id": "a"}]
        result_nodes, result_edges = _extract_neighborhood(nodes, [], "missing", 2, 100)
        assert result_nodes == []
        assert result_edges == []

    def test_one_hop(self) -> None:
        from app.routes.twin import _extract_neighborhood

        nodes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        edges = [
            {"source_node_id": "a", "target_node_id": "b"},
            {"source_node_id": "b", "target_node_id": "c"},
        ]
        result_nodes, result_edges = _extract_neighborhood(nodes, edges, "a", 1, 100)
        result_ids = {n["id"] for n in result_nodes}
        assert "a" in result_ids
        assert "b" in result_ids
        # c is 2 hops away
        assert "c" not in result_ids

    def test_limit_caps_results(self) -> None:
        from app.routes.twin import _extract_neighborhood

        nodes = [{"id": str(i)} for i in range(10)]
        edges = [{"source_node_id": "0", "target_node_id": str(i)} for i in range(1, 10)]
        result_nodes, _ = _extract_neighborhood(nodes, edges, "0", 1, 3)
        assert len(result_nodes) <= 3


class TestPaginateGraph:
    """Tests for _paginate_graph helper."""

    def test_empty(self) -> None:
        from app.routes.twin import _paginate_graph

        result = _paginate_graph([], [], page=0, limit=10)
        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["total_nodes"] == 0

    def test_pagination(self) -> None:
        from app.routes.twin import _paginate_graph

        nodes = [
            {"id": str(i), "kind": "file", "name": f"file{i}", "natural_key": f"key{i}"}
            for i in range(5)
        ]
        edges = [{"source_node_id": "0", "target_node_id": "1"}]

        result = _paginate_graph(nodes, edges, page=0, limit=2)
        assert len(result["nodes"]) == 2
        assert result["total_nodes"] == 5
        assert result["page"] == 0
        assert result["limit"] == 2


class TestBuildProjectionCoeffs:
    """Tests for _build_projection_coeffs helper."""

    def test_deterministic(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        a = _build_projection_coeffs(10, seed=42)
        b = _build_projection_coeffs(10, seed=42)
        assert a == b

    def test_different_seeds(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        a = _build_projection_coeffs(10, seed=42)
        b = _build_projection_coeffs(10, seed=99)
        assert a != b

    def test_correct_dimension(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        result = _build_projection_coeffs(5, seed=1)
        assert len(result) == 5


class TestBuildSemanticMapSignals:
    """Tests for _build_semantic_map_signals helper."""

    def test_empty_points(self) -> None:
        from app.routes.twin import _build_semantic_map_signals

        result = _build_semantic_map_signals([], mode="code_structure")
        assert result["mixed_clusters"] == []
        assert result["isolated_points"] == []
        assert result["semantic_duplication"] == []
        assert result["misplaced_code"] == []

    def test_mixed_cluster_detected(self) -> None:
        from app.routes.twin import _build_semantic_map_signals

        points = [
            {
                "id": "comm_1",
                "label": "Mixed",
                "x": 0.0,
                "y": 0.0,
                "member_count": 10,
                "dominant_ratio": 0.4,
                "domain_counts": [
                    {"domain": "billing", "count": 4},
                    {"domain": "auth", "count": 6},
                ],
                "anchor_node_id": "node_1",
                "name_tokens": [],
                "source_refs": [],
            },
        ]
        result = _build_semantic_map_signals(points, mode="code_structure")
        assert len(result["mixed_clusters"]) == 1
        assert result["mixed_clusters"][0]["community_id"] == "comm_1"


# ---------------------------------------------------------------------------
# Route-level tests (HTTP validation / auth / error paths)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestTwinScenarioRoutes:
    """Tests for scenario CRUD route authentication and validation."""

    async def test_create_scenario_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/twin/scenarios",
            json={"collection_id": str(uuid.uuid4()), "name": "Test"},
        )
        assert response.status_code == 401

    async def test_list_scenarios_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/scenarios")
        assert response.status_code == 401

    async def test_get_scenario_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/scenarios/{uuid.uuid4()}")
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_create_scenario_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            "/api/twin/scenarios",
            json={"collection_id": "not-a-uuid", "name": "Test"},
        )
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_create_scenario_name_too_long(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            "/api/twin/scenarios",
            json={"collection_id": str(uuid.uuid4()), "name": "x" * 256},
        )
        assert response.status_code == 422  # Pydantic validation

    @patch("app.routes.twin.get_session")
    async def test_create_scenario_empty_name_rejected(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            "/api/twin/scenarios",
            json={"collection_id": str(uuid.uuid4()), "name": ""},
        )
        assert response.status_code == 422

    @patch("app.routes.twin.get_session")
    async def test_get_scenario_invalid_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/scenarios/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid scenario_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_list_scenarios_invalid_collection_filter(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/scenarios?collection_id=not-a-uuid")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinIntentRoutes:
    """Tests for intent route authentication and validation."""

    async def test_create_intent_requires_auth(self, client: AsyncClient) -> None:
        scenario_id = str(uuid.uuid4())
        response = await client.post(
            f"/api/twin/scenarios/{scenario_id}/intents",
            json={
                "intent_version": "1.0",
                "scenario_id": scenario_id,
                "action": "EXTRACT_DOMAIN",
                "target": {"type": "node", "id": "test"},
                "expected_scenario_version": 1,
            },
        )
        assert response.status_code == 401

    async def test_approve_intent_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/twin/scenarios/{uuid.uuid4()}/intents/{uuid.uuid4()}/approve",
        )
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_approve_intent_invalid_scenario_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            f"/api/twin/scenarios/bad-id/intents/{uuid.uuid4()}/approve",
        )
        assert response.status_code == 400
        assert "Invalid scenario_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_approve_intent_invalid_intent_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            f"/api/twin/scenarios/{uuid.uuid4()}/intents/bad-id/approve",
        )
        assert response.status_code == 400
        assert "Invalid intent_id" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinPatchesRoute:
    """Tests for patches route authentication."""

    async def test_get_patches_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/scenarios/{uuid.uuid4()}/patches")
        assert response.status_code == 401


@pytest.mark.anyio
class TestTwinGraphRoutes:
    """Tests for scenario graph route authentication and validation."""

    async def test_get_graph_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/scenarios/{uuid.uuid4()}/graph")
        assert response.status_code == 401

    async def test_get_neighborhood_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            f"/api/twin/scenarios/{uuid.uuid4()}/graph/neighborhood?node_id=test-node"
        )
        assert response.status_code == 401


@pytest.mark.anyio
class TestTwinCollectionRoutes:
    """Tests for collection-level twin route auth and validation."""

    async def test_twin_status_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/status")
        assert response.status_code == 401

    async def test_twin_timeline_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/timeline")
        assert response.status_code == 401

    async def test_twin_refresh_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/twin/collections/{uuid.uuid4()}/refresh",
            json={},
        )
        assert response.status_code == 401

    async def test_twin_diff_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            f"/api/twin/collections/{uuid.uuid4()}/views/diff?from_version=1&to_version=2"
        )
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_twin_status_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/status")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_twin_timeline_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/timeline")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_twin_refresh_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            "/api/twin/collections/not-a-uuid/refresh",
            json={},
        )
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_twin_timeline_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/timeline?source_id=bad-id"
        )
        assert response.status_code == 400
        assert "Invalid source_id" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinAnalysisRoutes:
    """Tests for analysis route authentication and validation."""

    async def test_summary_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/summary")
        assert response.status_code == 401

    async def test_methods_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/methods")
        assert response.status_code == 401

    async def test_calls_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/calls")
        assert response.status_code == 401

    async def test_cfg_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            f"/api/twin/collections/{uuid.uuid4()}/analysis/cfg?node_ref=test"
        )
        assert response.status_code == 401

    async def test_variable_flow_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            f"/api/twin/collections/{uuid.uuid4()}/analysis/variable-flow?node_ref=test"
        )
        assert response.status_code == 401

    async def test_taint_sources_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/taint/sources")
        assert response.status_code == 401

    async def test_taint_sinks_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/taint/sinks")
        assert response.status_code == 401

    async def test_taint_flows_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/twin/collections/{uuid.uuid4()}/analysis/taint/flows",
            json={},
        )
        assert response.status_code == 401

    async def test_findings_store_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/twin/collections/{uuid.uuid4()}/analysis/findings/store",
            json={"findings": []},
        )
        assert response.status_code == 401

    async def test_findings_list_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/findings")
        assert response.status_code == 401

    async def test_findings_sarif_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/analysis/findings/sarif")
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_summary_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-uuid/analysis/summary")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinViewRoutes:
    """Tests for view route auth and validation (arc42, topology, deep-dive, mermaid)."""

    async def test_arc42_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/views/arc42")
        assert response.status_code == 401

    async def test_arc42_drift_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/views/arc42/drift")
        assert response.status_code == 401

    async def test_topology_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/views/topology")
        assert response.status_code == 401

    async def test_deep_dive_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/views/deep-dive")
        assert response.status_code == 401

    async def test_mermaid_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/collections/{uuid.uuid4()}/views/mermaid")
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_arc42_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/views/arc42")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_topology_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/views/topology")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_deep_dive_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/views/deep-dive")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_mermaid_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/twin/collections/not-a-uuid/views/mermaid")
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_mermaid_invalid_c4_view(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        collection_id = str(uuid.uuid4())
        response = await client.get(
            f"/api/twin/collections/{collection_id}/views/mermaid?c4_view=invalid"
        )
        assert response.status_code == 400
        assert "Invalid c4_view" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinExportRoutes:
    """Tests for export route authentication and validation."""

    async def test_create_export_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/twin/scenarios/{uuid.uuid4()}/exports",
            json={"format": "lpg_jsonl"},
        )
        assert response.status_code == 401

    async def test_get_export_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/twin/scenarios/{uuid.uuid4()}/exports/{uuid.uuid4()}")
        assert response.status_code == 401

    async def test_get_export_raw_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(
            f"/api/twin/scenarios/{uuid.uuid4()}/exports/{uuid.uuid4()}/raw"
        )
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_get_export_invalid_export_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        scenario_id = str(uuid.uuid4())
        response = await client.get(f"/api/twin/scenarios/{scenario_id}/exports/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid export_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_get_export_raw_invalid_export_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        scenario_id = str(uuid.uuid4())
        response = await client.get(f"/api/twin/scenarios/{scenario_id}/exports/not-a-uuid/raw")
        assert response.status_code == 400
        assert "Invalid export_id" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinCypherRoute:
    """Tests for cypher route authentication and validation."""

    async def test_cypher_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/twin/scenarios/{uuid.uuid4()}/cypher",
            json={"query": "MATCH (n) RETURN n LIMIT 10"},
        )
        assert response.status_code == 401

    @patch("app.routes.twin.get_session")
    async def test_cypher_invalid_scenario_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            "/api/twin/scenarios/not-a-uuid/cypher",
            json={"query": "MATCH (n) RETURN n LIMIT 10"},
        )
        assert response.status_code == 400
        assert "Invalid scenario_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_cypher_empty_query_rejected(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post(
            f"/api/twin/scenarios/{uuid.uuid4()}/cypher",
            json={"query": ""},
        )
        assert response.status_code == 422  # Pydantic Field(min_length=1)


@pytest.mark.anyio
class TestTwinEvolutionRoutes:
    """Tests for evolution view route auth (complementing existing tests)."""

    @patch("app.routes.twin.get_session")
    async def test_investment_utilization_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get(
            "/api/twin/collections/not-a-uuid/views/evolution/investment-utilization"
        )
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_knowledge_islands_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get(
            "/api/twin/collections/not-a-uuid/views/evolution/knowledge-islands"
        )
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_temporal_coupling_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get(
            "/api/twin/collections/not-a-uuid/views/evolution/temporal-coupling"
        )
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]

    @patch("app.routes.twin.get_session")
    async def test_fitness_functions_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get(
            "/api/twin/collections/not-a-uuid/views/evolution/fitness-functions"
        )
        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]


@pytest.mark.anyio
class TestTwinScenarioNotFound:
    """Tests for scenario-not-found paths with mocked DB."""

    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_get_scenario_not_found(
        self,
        mock_get_session: Any,
        mock_get_db_session: Any,
        client: AsyncClient,
    ) -> None:
        from contextlib import asynccontextmanager

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        scenario_id = str(uuid.uuid4())
        response = await client.get(f"/api/twin/scenarios/{scenario_id}")
        assert response.status_code == 404
        assert "Scenario not found" in response.json()["detail"]

    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_get_export_scenario_not_found(
        self,
        mock_get_session: Any,
        mock_get_db_session: Any,
        client: AsyncClient,
    ) -> None:
        from contextlib import asynccontextmanager

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        scenario_id = str(uuid.uuid4())
        export_id = str(uuid.uuid4())
        response = await client.get(f"/api/twin/scenarios/{scenario_id}/exports/{export_id}")
        assert response.status_code == 404
        assert "Scenario not found" in response.json()["detail"]
