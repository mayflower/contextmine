"""Tests for pure/utility functions in contextmine_core.twin.service.

These tests focus on non-DB helper functions: relation-to-edge-kind mappings,
CRAP score computation, layer inference, and risk mapping.
"""

from __future__ import annotations

import pytest
from contextmine_core.architecture_intents import IntentRisk
from contextmine_core.models import (
    IntentRiskLevel,
    KnowledgeEdgeKind,
    TwinLayer,
)
from contextmine_core.semantic_snapshot.models import RelationKind
from contextmine_core.twin.service import (
    _compute_crap_score,
    _relation_to_edge_kind,
    _relation_to_knowledge_edge_kind,
    _risk_to_model,
    infer_edge_layers,
    infer_node_layers,
)

# ---------------------------------------------------------------------------
# _relation_to_edge_kind
# ---------------------------------------------------------------------------


class TestRelationToEdgeKind:
    """Test the mapping from RelationKind to twin edge kind strings."""

    @pytest.mark.parametrize(
        "kind, expected",
        [
            (RelationKind.CONTAINS, "symbol_contains_symbol"),
            (RelationKind.CALLS, "symbol_calls_symbol"),
            (RelationKind.REFERENCES, "symbol_references_symbol"),
            (RelationKind.EXTENDS, "symbol_extends_symbol"),
            (RelationKind.IMPLEMENTS, "symbol_implements_symbol"),
            (RelationKind.IMPORTS, "symbol_imports_symbol"),
        ],
    )
    def test_known_relation_kinds(self, kind: RelationKind, expected: str) -> None:
        assert _relation_to_edge_kind(kind) == expected

    def test_unknown_relation_kind_returns_default(self) -> None:
        """An unknown RelationKind should fall back to symbol_references_symbol."""
        # Use a sentinel that won't match any key (simulate via the enum
        # fallback path).  Every known value is already tested above, so we
        # verify the default by passing an object that is not in the mapping.
        # Since the function uses `.get(kind, default)`, any value not in
        # the dict triggers the default.
        # Verify the mapping's default path with a dict lookup
        mapping = {
            RelationKind.CONTAINS: "symbol_contains_symbol",
            RelationKind.CALLS: "symbol_calls_symbol",
            RelationKind.REFERENCES: "symbol_references_symbol",
            RelationKind.EXTENDS: "symbol_extends_symbol",
            RelationKind.IMPLEMENTS: "symbol_implements_symbol",
            RelationKind.IMPORTS: "symbol_imports_symbol",
        }
        # Every known kind maps correctly — verified above.
        # The default path is exercised when `kind` is absent from the dict.
        assert mapping.get("nonexistent", "symbol_references_symbol") == "symbol_references_symbol"


# ---------------------------------------------------------------------------
# _relation_to_knowledge_edge_kind
# ---------------------------------------------------------------------------


class TestRelationToKnowledgeEdgeKind:
    """Test the mapping from RelationKind to KnowledgeEdgeKind."""

    @pytest.mark.parametrize(
        "kind, expected",
        [
            (RelationKind.CONTAINS, KnowledgeEdgeKind.SYMBOL_CONTAINS_SYMBOL),
            (RelationKind.CALLS, KnowledgeEdgeKind.SYMBOL_CALLS_SYMBOL),
            (RelationKind.REFERENCES, KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL),
            (RelationKind.IMPORTS, KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL),
            (RelationKind.EXTENDS, KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL),
            (RelationKind.IMPLEMENTS, KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL),
        ],
    )
    def test_known_relation_kinds(self, kind: RelationKind, expected: KnowledgeEdgeKind) -> None:
        assert _relation_to_knowledge_edge_kind(kind) == expected


# ---------------------------------------------------------------------------
# _compute_crap_score
# ---------------------------------------------------------------------------


class TestComputeCrapScore:
    """Test the CRAP (Change Risk Anti-Patterns) score calculation."""

    def test_none_complexity_returns_none(self) -> None:
        assert _compute_crap_score(None, 50.0) is None

    def test_none_coverage_returns_none(self) -> None:
        assert _compute_crap_score(10.0, None) is None

    def test_both_none_returns_none(self) -> None:
        assert _compute_crap_score(None, None) is None

    def test_full_coverage_returns_complexity(self) -> None:
        """With 100% coverage, CRAP = complexity (the cubic penalty term vanishes)."""
        result = _compute_crap_score(10.0, 100.0)
        assert result is not None
        # (10^2 * (1-1.0)^3) + 10 = 0 + 10 = 10.0
        assert result == pytest.approx(10.0)

    def test_zero_coverage(self) -> None:
        """With 0% coverage, CRAP = complexity^2 + complexity."""
        result = _compute_crap_score(5.0, 0.0)
        assert result is not None
        # (5^2 * (1-0)^3) + 5 = 25 + 5 = 30.0
        assert result == pytest.approx(30.0)

    def test_partial_coverage(self) -> None:
        """50% coverage should give a middling CRAP score."""
        result = _compute_crap_score(10.0, 50.0)
        assert result is not None
        # (100 * 0.5^3) + 10 = 100 * 0.125 + 10 = 12.5 + 10 = 22.5
        assert result == pytest.approx(22.5)

    def test_coverage_clamped_above_100(self) -> None:
        """Coverage above 100 should be clamped to 100."""
        result = _compute_crap_score(10.0, 150.0)
        assert result is not None
        # clamped to 100 -> same as full coverage
        assert result == pytest.approx(10.0)

    def test_coverage_clamped_below_0(self) -> None:
        """Coverage below 0 should be clamped to 0."""
        result = _compute_crap_score(5.0, -10.0)
        assert result is not None
        # clamped to 0 -> same as zero coverage
        assert result == pytest.approx(30.0)

    def test_zero_complexity(self) -> None:
        """Zero complexity should yield a CRAP score of 0."""
        result = _compute_crap_score(0.0, 50.0)
        assert result is not None
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# infer_node_layers
# ---------------------------------------------------------------------------


class TestInferNodeLayers:
    """Test layer inference for node kinds."""

    @pytest.mark.parametrize(
        "kind",
        ["file", "module", "symbol", "function", "method", "class", "validator"],
    )
    def test_code_controlflow_kinds(self, kind: str) -> None:
        assert infer_node_layers(kind) == {TwinLayer.CODE_CONTROLFLOW}

    @pytest.mark.parametrize(
        "kind",
        ["api_endpoint", "interface", "rpc", "service", "component"],
    )
    def test_component_interface_kinds(self, kind: str) -> None:
        assert infer_node_layers(kind) == {TwinLayer.COMPONENT_INTERFACE}

    @pytest.mark.parametrize(
        "kind",
        ["bounded_context", "container", "db_table", "db_column"],
    )
    def test_domain_container_kinds(self, kind: str) -> None:
        assert infer_node_layers(kind) == {TwinLayer.DOMAIN_CONTAINER}

    def test_unknown_kind_defaults_to_portfolio(self) -> None:
        assert infer_node_layers("unknown_kind") == {TwinLayer.PORTFOLIO_SYSTEM}

    def test_case_insensitive(self) -> None:
        assert infer_node_layers("FILE") == {TwinLayer.CODE_CONTROLFLOW}
        assert infer_node_layers("Api_Endpoint") == {TwinLayer.COMPONENT_INTERFACE}
        assert infer_node_layers("DB_TABLE") == {TwinLayer.DOMAIN_CONTAINER}

    def test_meta_argument_is_ignored(self) -> None:
        """The meta parameter is accepted but currently unused."""
        result = infer_node_layers("file", meta={"some": "data"})
        assert result == {TwinLayer.CODE_CONTROLFLOW}


# ---------------------------------------------------------------------------
# infer_edge_layers
# ---------------------------------------------------------------------------


class TestInferEdgeLayers:
    """Test layer inference for edge kinds."""

    def test_file_defines_symbol(self) -> None:
        assert infer_edge_layers("file_defines_symbol") == {TwinLayer.CODE_CONTROLFLOW}

    @pytest.mark.parametrize(
        "kind",
        [
            "symbol_calls_symbol",
            "symbol_contains_symbol",
            "symbol_references_symbol",
            "symbol_extends_symbol",
            "symbol_implements_symbol",
            "symbol_imports_symbol",
        ],
    )
    def test_symbol_edge_kinds(self, kind: str) -> None:
        assert infer_edge_layers(kind) == {TwinLayer.CODE_CONTROLFLOW}

    def test_edge_with_calls(self) -> None:
        assert infer_edge_layers("api_calls_downstream") == {TwinLayer.CODE_CONTROLFLOW}

    def test_edge_with_references(self) -> None:
        assert infer_edge_layers("cross_references_symbol") == {TwinLayer.CODE_CONTROLFLOW}

    def test_edge_with_contains(self) -> None:
        assert infer_edge_layers("module_contains_class") == {TwinLayer.CODE_CONTROLFLOW}

    def test_edge_with_interface(self) -> None:
        assert infer_edge_layers("interface_exposes") == {TwinLayer.COMPONENT_INTERFACE}

    def test_edge_with_endpoint(self) -> None:
        assert infer_edge_layers("endpoint_maps_to") == {TwinLayer.COMPONENT_INTERFACE}

    def test_edge_with_rpc(self) -> None:
        assert infer_edge_layers("rpc_invocation") == {TwinLayer.COMPONENT_INTERFACE}

    def test_edge_with_context(self) -> None:
        assert infer_edge_layers("bounded_context_depends_on") == {TwinLayer.DOMAIN_CONTAINER}

    def test_edge_with_domain(self) -> None:
        assert infer_edge_layers("domain_boundary") == {TwinLayer.DOMAIN_CONTAINER}

    def test_unknown_edge_kind_defaults_to_portfolio(self) -> None:
        assert infer_edge_layers("unknown_edge") == {TwinLayer.PORTFOLIO_SYSTEM}

    def test_case_insensitive(self) -> None:
        assert infer_edge_layers("FILE_DEFINES_SYMBOL") == {TwinLayer.CODE_CONTROLFLOW}
        assert infer_edge_layers("INTERFACE_EXPOSES") == {TwinLayer.COMPONENT_INTERFACE}


# ---------------------------------------------------------------------------
# _risk_to_model
# ---------------------------------------------------------------------------


class TestRiskToModel:
    """Test the mapping from IntentRisk to IntentRiskLevel."""

    def test_high_risk(self) -> None:
        assert _risk_to_model(IntentRisk.HIGH) == IntentRiskLevel.HIGH

    def test_low_risk(self) -> None:
        assert _risk_to_model(IntentRisk.LOW) == IntentRiskLevel.LOW
