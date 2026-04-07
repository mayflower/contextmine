"""Guardrail ratchets for the prompt-pack v2 recovery roadmap.

These tests ratify the main weaknesses that still exist today:

1. Recovery doc loading is still gated on persisted ``Document`` rows linked from
   scenario ``FILE`` nodes instead of a repo-wide artifact inventory.
2. Architecture-doc detection is still largely heuristic and does not recognize
   typed parser/artifact hints unless the path/title/text also happens to match.
3. Component recovery still defaults to emitting 1:1 symbol aliases instead of
   clustering symbols into evidence-backed components.
4. LLM adjudication packets still send ref strings as ``snippet`` values rather
   than real evidence excerpts.
5. ``Arc42Document`` still has no dedicated claim-to-evidence traceability
   surface.
6. A single path heuristic can still yield a selected container assignment.
7. Low-evidence recovery should keep surfacing uncertainty explicitly.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.architecture.arc42 import generate_arc42_from_facts
from contextmine_core.architecture.recovery import recover_architecture_model
from contextmine_core.architecture.recovery_docs import (
    _looks_like_architecture_doc,
    load_recovery_docs,
)
from contextmine_core.architecture.recovery_llm import build_adjudication_packet
from contextmine_core.architecture.schemas import ArchitectureFactsBundle, EvidenceRef
from contextmine_core.models import Document, KnowledgeNode, KnowledgeNodeKind

from .models.architecture_recovery_fixture import build_architecture_recovery_fixture


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalars(self):
        return self

    def all(self):
        return self._value


class _FakeDocumentSession:
    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents or []

    async def execute(self, stmt):  # noqa: ANN001
        statement = str(stmt)
        if "FROM documents" not in statement:
            raise AssertionError(f"Unexpected statement: {statement}")
        return _ScalarResult(self.documents)


def _symbol_node(
    *,
    natural_key: str,
    name: str,
    file_path: str | None = None,
    architecture: dict[str, str] | None = None,
) -> dict[str, object]:
    meta: dict[str, object] = {}
    if file_path is not None:
        meta["file_path"] = file_path
    if architecture is not None:
        meta["architecture"] = architecture
    return {
        "id": natural_key,
        "kind": "symbol",
        "name": name,
        "natural_key": natural_key,
        "meta": meta,
    }


@pytest.mark.anyio
async def test_repo_artifacts_should_not_require_document_rows_to_be_visible() -> None:
    file_node = KnowledgeNode(
        id=uuid4(),
        collection_id=uuid4(),
        kind=KnowledgeNodeKind.FILE,
        natural_key="docs/records/async-worker-rollout.md",
        name="Async worker rollout",
        meta={
            "uri": "docs/records/async-worker-rollout.md",
            "file_path": "docs/records/async-worker-rollout.md",
            "content_markdown": (
                "---\n"
                "artifact_kind: architecture_decision\n"
                "affected_entity_ids:\n"
                "  - container:api\n"
                "  - container:worker\n"
                "---\n"
                "Embeddings execute in the worker runtime."
            ),
        },
    )

    docs = await load_recovery_docs(_FakeDocumentSession(documents=[]), [file_node])

    assert docs, "Repo inventory should surface ADR-like artifacts without a Document row."


def test_architecture_doc_detection_should_accept_typed_artifact_hints_without_token_matches() -> (
    None
):
    doc = SimpleNamespace(
        uri="docs/records/0007.md",
        title="Record 7",
        content_markdown="Implementation note for background processing.",
        meta={
            "file_path": "docs/records/0007.md",
            "artifact_kind": "architecture_decision",
            "parser": "adr_frontmatter_v1",
        },
    )

    assert _looks_like_architecture_doc(doc) is True


def test_component_recovery_should_cluster_related_symbols_instead_of_aliasing_each_symbol() -> (
    None
):
    nodes = [
        _symbol_node(
            natural_key="symbol:create_session",
            name="Create Session",
            file_path="services/contextmine/api/session_service.py",
            architecture={"domain": "contextmine", "container": "api"},
        ),
        _symbol_node(
            natural_key="symbol:load_session",
            name="Load Session",
            file_path="services/contextmine/api/session_service.py",
            architecture={"domain": "contextmine", "container": "api"},
        ),
    ]
    edges = [
        {
            "source_node_id": "symbol:create_session",
            "target_node_id": "symbol:load_session",
            "kind": "symbol_calls_symbol",
            "meta": {},
        }
    ]

    model = recover_architecture_model(nodes, edges, docs=[])

    components = [entity for entity in model.entities if entity.kind == "component"]
    assert len(components) == 1


def test_adjudication_packets_should_contain_real_snippets_not_only_refs() -> None:
    fixture = build_architecture_recovery_fixture()
    model = recover_architecture_model(
        fixture["nodes"],
        fixture["edges"],
        docs=fixture["docs"],
    )
    hypothesis = next(
        row for row in model.hypotheses if row.subject_ref == "symbol:session_manager"
    )

    packet = build_adjudication_packet(model=model, hypothesis=hypothesis)

    assert any(
        item["snippet"].strip() and item["snippet"] != item["ref"] for item in packet["evidence"]
    ), "LLM packets need evidence excerpts, not ref-string echoes."


def test_arc42_document_should_expose_claim_traceability() -> None:
    collection_id = uuid4()
    scenario_id = uuid4()
    bundle = ArchitectureFactsBundle(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name="AS-IS",
    )
    scenario = SimpleNamespace(name="AS-IS")

    document = generate_arc42_from_facts(bundle, scenario)
    field_names = {field.name for field in fields(type(document))}

    assert "claim_traceability" in field_names


def test_path_only_membership_should_not_be_selected_as_a_confirmed_container() -> None:
    model = recover_architecture_model(
        [
            _symbol_node(
                natural_key="symbol:event_publisher",
                name="Event Publisher",
                file_path="services/contextmine/api/events.py",
            )
        ],
        [],
        docs=[],
    )

    memberships = model.memberships_for("symbol:event_publisher")
    hypotheses = [row for row in model.hypotheses if row.subject_ref == "symbol:event_publisher"]

    assert not memberships
    assert hypotheses
    assert hypotheses[0].status in {"ambiguous", "unresolved"}
    assert not hypotheses[0].selected_entity_ids


def test_insufficient_evidence_keeps_uncertainty_visible() -> None:
    model = recover_architecture_model(
        [
            {
                "id": "symbol:orphan_helper",
                "kind": "symbol",
                "name": "Orphan Helper",
                "natural_key": "symbol:orphan_helper",
                "meta": {},
            }
        ],
        [],
        docs=[],
    )

    hypothesis = next(row for row in model.hypotheses if row.subject_ref == "symbol:orphan_helper")
    assert hypothesis.status == "unresolved"
    assert hypothesis.selected_entity_ids == ()
    assert hypothesis.evidence == (EvidenceRef(kind="node", ref="symbol:orphan_helper"),)
