"""Extended tests for twin route helper functions, validators, and edge paths.

Covers additional pure helper functions, serialization helpers, validators,
and parsing functions not yet tested in test_routes_twin.py.
"""

from __future__ import annotations

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# _parse_db_table_from_natural_key
# ---------------------------------------------------------------------------


class TestParseDbTableFromNaturalKey:
    def test_none_returns_none(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key(None) is None

    def test_empty_returns_none(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("") is None

    def test_non_db_prefix(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("file:something") is None

    def test_db_prefix_no_body(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:") is None

    def test_simple_table(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:users") == "users"

    def test_dotted_path(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:users.id") == "users"

    def test_dotted_path_empty_table(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        assert _parse_db_table_from_natural_key("db:.id") is None


# ---------------------------------------------------------------------------
# _serialize_erm_column
# ---------------------------------------------------------------------------


class TestSerializeErmColumn:
    def test_basic_column(self) -> None:
        from app.routes.twin import _serialize_erm_column

        node = MagicMock()
        node.id = uuid.uuid4()
        node.natural_key = "db:users.name"
        node.name = "name"
        node.meta = {
            "table": "users",
            "type": "varchar",
            "nullable": False,
            "primary_key": True,
            "foreign_key": "other.id",
        }

        result = _serialize_erm_column(node)
        assert result["name"] == "name"
        assert result["table"] == "users"
        assert result["type"] == "varchar"
        assert result["nullable"] is False
        assert result["primary_key"] is True
        assert result["foreign_key"] == "other.id"

    def test_column_with_no_meta(self) -> None:
        from app.routes.twin import _serialize_erm_column

        node = MagicMock()
        node.id = uuid.uuid4()
        node.natural_key = "db:users.name"
        node.name = "name"
        node.meta = None

        result = _serialize_erm_column(node)
        assert result["nullable"] is True
        assert result["primary_key"] is False
        assert result["foreign_key"] is None


# ---------------------------------------------------------------------------
# _serialize_erm_table
# ---------------------------------------------------------------------------


class TestSerializeErmTable:
    def test_basic_table(self) -> None:
        from app.routes.twin import _serialize_erm_table

        node = MagicMock()
        node.id = uuid.uuid4()
        node.natural_key = "db:users"
        node.name = "users"
        node.description = "User accounts"
        node.meta = {"column_count": 5, "primary_keys": ["id"]}

        columns = [{"name": "id"}, {"name": "email"}]
        result = _serialize_erm_table(node, columns)

        assert result["name"] == "users"
        assert result["description"] == "User accounts"
        assert result["column_count"] == 5
        assert result["primary_keys"] == ["id"]
        assert len(result["columns"]) == 2


# ---------------------------------------------------------------------------
# _serialize_scenario
# ---------------------------------------------------------------------------


class TestSerializeScenario:
    def test_basic_scenario(self) -> None:
        from app.routes.twin import _serialize_scenario

        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "AS-IS"
        scenario.version = 3
        scenario.is_as_is = True
        scenario.base_scenario_id = None

        result = _serialize_scenario(scenario)
        assert result["id"] == str(scenario.id)
        assert result["name"] == "AS-IS"
        assert result["version"] == 3
        assert result["is_as_is"] is True
        assert result["base_scenario_id"] is None

    def test_scenario_with_base(self) -> None:
        from app.routes.twin import _serialize_scenario

        base_id = uuid.uuid4()
        scenario = MagicMock()
        scenario.id = uuid.uuid4()
        scenario.collection_id = uuid.uuid4()
        scenario.name = "TO-BE"
        scenario.version = 1
        scenario.is_as_is = False
        scenario.base_scenario_id = base_id

        result = _serialize_scenario(scenario)
        assert result["base_scenario_id"] == str(base_id)


# ---------------------------------------------------------------------------
# _sha256_text
# ---------------------------------------------------------------------------


class TestSha256Text:
    def test_deterministic(self) -> None:
        from app.routes.twin import _sha256_text

        assert _sha256_text("hello") == _sha256_text("hello")

    def test_correct_value(self) -> None:
        from app.routes.twin import _sha256_text

        expected = hashlib.sha256(b"test").hexdigest()
        assert _sha256_text("test") == expected


# ---------------------------------------------------------------------------
# _arc42_artifact_name
# ---------------------------------------------------------------------------


class TestArc42ArtifactName:
    def test_format(self) -> None:
        from app.routes.twin import _arc42_artifact_name

        sid = uuid.uuid4()
        result = _arc42_artifact_name(sid)
        assert result == f"{sid}.arc42.md"


# ---------------------------------------------------------------------------
# _select_arc42_section
# ---------------------------------------------------------------------------


class TestSelectArc42Section:
    def test_none_section_returns_original(self) -> None:
        from app.routes.twin import _select_arc42_section

        doc = MagicMock()
        result = _select_arc42_section(doc, None)
        assert result is doc

    def test_with_section_key(self) -> None:
        from types import SimpleNamespace

        from app.routes.twin import _select_arc42_section

        section_key = "1_introduction_and_goals"
        doc = SimpleNamespace(
            collection_id=uuid.uuid4(),
            scenario_id=uuid.uuid4(),
            scenario_name="test",
            title="Arc42 Doc",
            generated_at="2024-01-01",
            sections={section_key: "Introduction content here"},
            markdown="# Full doc",
            warnings=[],
            confidence_summary={},
            section_coverage={section_key: True},
        )

        result = _select_arc42_section(doc, section_key)
        assert "Introduction content here" in result.markdown
        assert section_key in result.sections


# ---------------------------------------------------------------------------
# _parse_graphrag_community_mode
# ---------------------------------------------------------------------------


class TestParseGraphragCommunityMode:
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

        with pytest.raises(HTTPException) as exc_info:
            _parse_graphrag_community_mode("invalid")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _parse_semantic_map_mode
# ---------------------------------------------------------------------------


class TestParseSemanticMapMode:
    def test_none_returns_code_structure(self) -> None:
        from app.routes.twin import _parse_semantic_map_mode

        assert _parse_semantic_map_mode(None) == "code_structure"

    def test_valid_modes(self) -> None:
        from app.routes.twin import _parse_semantic_map_mode

        assert _parse_semantic_map_mode("code_structure") == "code_structure"
        assert _parse_semantic_map_mode("semantic") == "semantic"

    def test_invalid_raises_400(self) -> None:
        from app.routes.twin import _parse_semantic_map_mode

        with pytest.raises(HTTPException) as exc_info:
            _parse_semantic_map_mode("invalid_mode")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _parse_pgvector_text
# ---------------------------------------------------------------------------


class TestParsePgvectorText:
    def test_none_returns_empty(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        assert _parse_pgvector_text(None) == []

    def test_empty_string_returns_empty(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        assert _parse_pgvector_text("") == []

    def test_empty_brackets_returns_empty(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        assert _parse_pgvector_text("[]") == []

    def test_valid_vector(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        result = _parse_pgvector_text("[1.0, 2.5, 3.14]")
        assert result == [1.0, 2.5, 3.14]

    def test_without_brackets(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        result = _parse_pgvector_text("1.0, 2.5")
        assert result == [1.0, 2.5]

    def test_invalid_values_skipped(self) -> None:
        from app.routes.twin import _parse_pgvector_text

        result = _parse_pgvector_text("[1.0, not_a_number, 3.0]")
        assert result == [1.0, 3.0]


# ---------------------------------------------------------------------------
# _extract_domain_token
# ---------------------------------------------------------------------------


class TestExtractDomainToken:
    def test_none_returns_none(self) -> None:
        from app.routes.twin import _extract_domain_token

        assert _extract_domain_token(None) is None

    def test_empty_returns_none(self) -> None:
        from app.routes.twin import _extract_domain_token

        assert _extract_domain_token("") is None

    def test_file_prefix_stripped(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("file:services/billing/api/views.py")
        assert result is not None
        assert result == "billing"

    def test_url_style_path(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("git://github.com/org/repo/src/billing/main.py")
        assert result is not None

    def test_simple_path(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("services/billing/handler.py")
        assert result is not None
        assert result == "billing"

    def test_ignored_prefix(self) -> None:
        from app.routes.twin import _extract_domain_token

        result = _extract_domain_token("src/main.py")
        assert result is None or result == "src"


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_empty_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        assert _cosine_similarity([], []) == 0.0

    def test_identical_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-6

    def test_opposite_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(result - (-1.0)) < 1e-6

    def test_different_lengths(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0, 0.0], [1.0])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_empty_sets(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard(set(), set()) == 1.0

    def test_one_empty(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard({"a"}, set()) == 0.0

    def test_identical_sets(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_partial_overlap(self) -> None:
        from app.routes.twin import _jaccard

        result = _jaccard({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 1e-6

    def test_disjoint(self) -> None:
        from app.routes.twin import _jaccard

        assert _jaccard({"a"}, {"b"}) == 0.0


# ---------------------------------------------------------------------------
# _build_projection_coeffs
# ---------------------------------------------------------------------------


class TestBuildProjectionCoeffs:
    def test_returns_correct_length(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        coeffs = _build_projection_coeffs(10, seed=42)
        assert len(coeffs) == 10

    def test_deterministic(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        c1 = _build_projection_coeffs(5, seed=17)
        c2 = _build_projection_coeffs(5, seed=17)
        assert c1 == c2

    def test_different_seeds(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        c1 = _build_projection_coeffs(5, seed=17)
        c2 = _build_projection_coeffs(5, seed=89)
        assert c1 != c2

    def test_values_in_range(self) -> None:
        from app.routes.twin import _build_projection_coeffs

        coeffs = _build_projection_coeffs(100, seed=42)
        for c in coeffs:
            assert -1.1 <= c <= 1.1


# ---------------------------------------------------------------------------
# _normalize_xy
# ---------------------------------------------------------------------------


class TestNormalizeXY:
    def test_empty_list(self) -> None:
        from app.routes.twin import _normalize_xy

        points: list[dict] = []
        _normalize_xy(points)
        assert points == []

    def test_single_point(self) -> None:
        from app.routes.twin import _normalize_xy

        points = [{"x": 5.0, "y": 3.0}]
        _normalize_xy(points)
        assert points[0]["x"] == -1.0 or abs(points[0]["x"]) <= 1.0

    def test_two_points_normalized(self) -> None:
        from app.routes.twin import _normalize_xy

        points = [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0}]
        _normalize_xy(points)
        assert points[0]["x"] == -1.0
        assert points[1]["x"] == 1.0
        assert points[0]["y"] == -1.0
        assert points[1]["y"] == 1.0


# ---------------------------------------------------------------------------
# _layout_code_structure_points
# ---------------------------------------------------------------------------


class TestLayoutCodeStructurePoints:
    def test_empty_ids(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        result = _layout_code_structure_points([], {})
        assert result == {}

    def test_single_id(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        result = _layout_code_structure_points(["a"], {})
        assert "a" in result
        assert result["a"] == (0.0, 0.0)

    def test_multiple_ids(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        ids = ["a", "b", "c"]
        result = _layout_code_structure_points(ids, {})
        assert len(result) == 3
        for pid in ids:
            assert pid in result
            x, y = result[pid]
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_large_count_uses_scatter(self) -> None:
        from app.routes.twin import _layout_code_structure_points

        ids = [f"node_{i}" for i in range(250)]
        result = _layout_code_structure_points(ids, {})
        assert len(result) == 250


# ---------------------------------------------------------------------------
# _detect_isolated_point_ids
# ---------------------------------------------------------------------------


class TestDetectIsolatedPointIds:
    def test_single_point(self) -> None:
        from app.routes.twin import _detect_isolated_point_ids

        points = [{"id": "a", "x": 0.0, "y": 0.0}]
        isolated, distances = _detect_isolated_point_ids(points)
        assert "a" in isolated

    def test_two_close_points(self) -> None:
        from app.routes.twin import _detect_isolated_point_ids

        points = [
            {"id": "a", "x": 0.0, "y": 0.0},
            {"id": "b", "x": 0.1, "y": 0.0},
        ]
        isolated, distances = _detect_isolated_point_ids(points)
        assert len(isolated) == 0 or len(isolated) == 2

    def test_outlier_detected(self) -> None:
        from app.routes.twin import _detect_isolated_point_ids

        points = [
            {"id": "a", "x": 0.0, "y": 0.0},
            {"id": "b", "x": 0.1, "y": 0.0},
            {"id": "c", "x": 0.05, "y": 0.0},
            {"id": "outlier", "x": 10.0, "y": 10.0},
        ]
        isolated, distances = _detect_isolated_point_ids(points)
        assert "outlier" in isolated


# ---------------------------------------------------------------------------
# _resolve_arch_llm_provider
# ---------------------------------------------------------------------------


class TestResolveArchLlmProvider:
    def test_not_enabled(self) -> None:
        from app.routes.twin import _resolve_arch_llm_provider

        settings = MagicMock()
        settings.arch_docs_llm_enrich = False
        result = _resolve_arch_llm_provider(settings)
        assert result is None

    def test_no_provider(self) -> None:
        from app.routes.twin import _resolve_arch_llm_provider

        settings = MagicMock()
        settings.arch_docs_llm_enrich = True
        settings.default_llm_provider = None
        result = _resolve_arch_llm_provider(settings)
        assert result is None

    def test_import_error_returns_none(self) -> None:
        from app.routes.twin import _resolve_arch_llm_provider

        settings = MagicMock()
        settings.arch_docs_llm_enrich = True
        settings.default_llm_provider = "openai"
        with patch(
            "contextmine_core.research.llm.get_llm_provider",
            side_effect=ImportError("no module"),
        ):
            result = _resolve_arch_llm_provider(settings)
        assert result is None


# ---------------------------------------------------------------------------
# _ensure_evolution_enabled
# ---------------------------------------------------------------------------


class TestEnsureEvolutionEnabled:
    def test_disabled_raises_404(self) -> None:
        from app.routes.twin import _ensure_evolution_enabled

        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.twin_evolution_view_enabled = False
            with pytest.raises(HTTPException) as exc_info:
                _ensure_evolution_enabled()
            assert exc_info.value.status_code == 404

    def test_enabled_does_not_raise(self) -> None:
        from app.routes.twin import _ensure_evolution_enabled

        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.twin_evolution_view_enabled = True
            _ensure_evolution_enabled()


# ---------------------------------------------------------------------------
# _build_semantic_map_signals
# ---------------------------------------------------------------------------


class TestBuildSemanticMapSignals:
    def test_empty_points(self) -> None:
        from app.routes.twin import _build_semantic_map_signals

        result = _build_semantic_map_signals([], mode="code_structure")
        assert isinstance(result, dict)
        assert result.get("mixed_clusters") == [] or "mixed_clusters" in result

    def test_isolated_point_detected(self) -> None:
        from app.routes.twin import _build_semantic_map_signals

        points = [
            {
                "id": "a",
                "x": 0.0,
                "y": 0.0,
                "member_count": 2,
                "dominant_ratio": 0.9,
                "domain_counts": [{"domain": "billing", "count": 2}],
                "label": "a",
                "name_tokens": [],
                "source_refs": [],
            },
            {
                "id": "b",
                "x": 0.1,
                "y": 0.0,
                "member_count": 2,
                "dominant_ratio": 0.9,
                "domain_counts": [{"domain": "auth", "count": 2}],
                "label": "b",
                "name_tokens": [],
                "source_refs": [],
            },
            {
                "id": "c",
                "x": 0.05,
                "y": 0.0,
                "member_count": 2,
                "dominant_ratio": 0.9,
                "domain_counts": [{"domain": "core", "count": 2}],
                "label": "c",
                "name_tokens": [],
                "source_refs": [],
            },
            {
                "id": "outlier",
                "x": 100.0,
                "y": 100.0,
                "member_count": 2,
                "dominant_ratio": 0.9,
                "domain_counts": [{"domain": "other", "count": 2}],
                "label": "outlier",
                "name_tokens": [],
                "source_refs": [],
            },
        ]
        result = _build_semantic_map_signals(points, mode="code_structure")
        isolated = result.get("isolated_points", [])
        assert any(p["community_id"] == "outlier" for p in isolated)

    def test_mixed_cluster_detected(self) -> None:
        from app.routes.twin import _build_semantic_map_signals

        points = [
            {
                "id": "mixed",
                "x": 0.0,
                "y": 0.0,
                "member_count": 10,
                "dominant_ratio": 0.4,
                "domain_counts": [
                    {"domain": "billing", "count": 4},
                    {"domain": "auth", "count": 3},
                    {"domain": "core", "count": 3},
                ],
                "label": "mixed cluster",
                "name_tokens": [],
                "source_refs": [],
            },
        ]
        result = _build_semantic_map_signals(points, mode="code_structure")
        mixed = result.get("mixed_clusters", [])
        assert len(mixed) == 1
        assert mixed[0]["community_id"] == "mixed"
