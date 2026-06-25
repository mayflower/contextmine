"""arc42 must not fabricate decisions / architecture patterns from counts and tags.

Regression guard for the audit finding that arc42 invented "REST API strategy"
decisions from the mere presence of an api_endpoint fact, asserted "microservices"
from a container count, and asserted "Security" from an auth tag.
"""

from __future__ import annotations

from uuid import uuid4

from contextmine_core.architecture.arc42 import (
    _render_crosscutting,
    _render_decisions,
    _render_strategy,
)
from contextmine_core.architecture.schemas import ArchitectureFact, ArchitectureFactsBundle


def _fact(fact_type: str, *, tags: tuple[str, ...] = (), title: str = "x") -> ArchitectureFact:
    return ArchitectureFact(
        fact_id=f"{fact_type}:{title}",
        fact_type=fact_type,
        title=title,
        description="d",
        source="deterministic",
        confidence=0.9,
        tags=tags,
        attributes={},
    )


def _bundle(facts: list[ArchitectureFact]) -> ArchitectureFactsBundle:
    return ArchitectureFactsBundle(
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        facts=facts,
        ports_adapters=[],
    )


def test_decisions_not_invented_from_api_surface() -> None:
    bundle = _bundle([_fact("api_endpoint"), _fact("graphql_operation")])
    rendered = _render_decisions(bundle)
    # No fabricated "decisions" from the presence of API surfaces.
    assert "REST API strategy" not in rendered
    assert "Dual API strategy" not in rendered
    assert "GraphQL-first" not in rendered
    assert "No architecture decisions were recovered" in rendered


def test_real_decision_facts_are_rendered() -> None:
    decision = ArchitectureFact(
        fact_id="adr:001",
        fact_type="architecture_decision",
        title="Use async workers",
        description="Embedding work runs on Prefect workers.",
        source="deterministic",
        confidence=0.9,
        tags=("architecture_decision",),
        attributes={},
    )
    rendered = _render_decisions(_bundle([decision]))
    assert "Use async workers" in rendered
    assert "No architecture decisions were recovered" not in rendered


def test_strategy_does_not_assert_microservices_from_count() -> None:
    bundle = _bundle([_fact("container", title=f"c{i}") for i in range(5)])
    rendered = _render_strategy(bundle)
    assert "microservices" not in rendered.lower()
    # The count is reported as an observation, and any pattern is a hedged heuristic.
    assert "5 container" in rendered
    assert "heuristic" in rendered.lower()


def test_crosscutting_security_is_a_labelled_count_not_an_assessment() -> None:
    bundle = _bundle([_fact("api_endpoint", tags=("auth",))])
    rendered = _render_crosscutting(bundle)
    assert "not a security assessment" in rendered
