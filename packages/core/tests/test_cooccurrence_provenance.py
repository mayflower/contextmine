"""CO_OCCURS edges must be distinguishable from LLM-extracted semantic relationships.

Both share the SEMANTIC_RELATIONSHIP edge kind, so the edge meta must record how the
relationship was derived (a co-occurrence guess vs a real extraction).
"""

from __future__ import annotations

from contextmine_core.knowledge.extraction import (
    SemanticRelationship,
    _relationship_edge_meta,
)


def test_co_occurrence_relationship_is_marked_inferred() -> None:
    rel = SemanticRelationship(
        source_entity="User",
        target_entity="Order",
        relationship_type="CO_OCCURS",
        description="Co-occur in 3 document(s)",
        strength=0.6,
    )
    meta = _relationship_edge_meta(rel)
    assert meta["method"] == "co_occurrence"
    assert meta["inferred"] is True


def test_llm_relationship_is_not_marked_inferred() -> None:
    rel = SemanticRelationship(
        source_entity="OrderService",
        target_entity="PaymentGateway",
        relationship_type="DEPENDS_ON",
        description="OrderService calls the payment gateway",
        strength=0.9,
    )
    meta = _relationship_edge_meta(rel)
    assert meta["method"] == "llm_extracted"
    assert meta["inferred"] is False
    assert meta["relationship_type"] == "DEPENDS_ON"
