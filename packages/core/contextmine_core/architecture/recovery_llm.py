"""Constrained LLM adjudication helpers for architecture recovery."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from .recovery_model import RecoveredArchitectureHypothesis, RecoveredArchitectureModel
from .schemas import EvidenceRef


class RecoveryAdjudicator(Protocol):
    """Minimal protocol for adjudicating one hypothesis packet."""

    def adjudicate(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Return an adjudication payload for one hypothesis packet."""


def _append_warning(model: RecoveredArchitectureModel, warning: str) -> RecoveredArchitectureModel:
    return replace(model, warnings=tuple(list(model.warnings) + [warning]))


def _evidence_packet_items(evidence: tuple[EvidenceRef, ...]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for index, ref in enumerate(evidence, start=1):
        items.append(
            {
                "evidence_id": f"ev-{index}",
                "kind": ref.kind,
                "ref": ref.ref,
                "snippet": ref.ref,
            }
        )
    return items


def build_adjudication_packet(
    *,
    model: RecoveredArchitectureModel,
    hypothesis: RecoveredArchitectureHypothesis,
) -> dict[str, Any]:
    """Build a packet containing only packet-local evidence and candidate IDs."""

    evidence = list(hypothesis.evidence)
    for membership in model.memberships_for(hypothesis.subject_ref):
        evidence.extend(membership.evidence)

    deduped_evidence: list[EvidenceRef] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for ref in evidence:
        key = (ref.kind, ref.ref, ref.start_line, ref.end_line)
        if key in seen:
            continue
        seen.add(key)
        deduped_evidence.append(ref)

    return {
        "subject_ref": hypothesis.subject_ref,
        "status": hypothesis.status,
        "candidate_entity_ids": list(hypothesis.candidate_entity_ids),
        "selected_entity_ids": list(hypothesis.selected_entity_ids),
        "evidence": _evidence_packet_items(tuple(deduped_evidence)),
    }


def apply_adjudication(
    *,
    model: RecoveredArchitectureModel,
    hypothesis: RecoveredArchitectureHypothesis,
    packet: dict[str, Any],
    adjudication: Any,
) -> RecoveredArchitectureModel:
    """Apply a validated adjudication without inventing new entities or evidence."""

    if not isinstance(adjudication, dict):
        return _append_warning(model, f"Malformed adjudication for {hypothesis.subject_ref}.")

    selected_entity_ids = adjudication.get("selected_entity_ids")
    rationale = adjudication.get("rationale")
    evidence_ids = adjudication.get("evidence_ids")

    if not isinstance(selected_entity_ids, list) or not all(
        isinstance(item, str) for item in selected_entity_ids
    ):
        return _append_warning(model, f"Malformed adjudication for {hypothesis.subject_ref}.")
    if not isinstance(rationale, str) or not rationale.strip():
        return _append_warning(model, f"Malformed adjudication for {hypothesis.subject_ref}.")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) for item in evidence_ids
    ):
        return _append_warning(model, f"Malformed adjudication for {hypothesis.subject_ref}.")

    packet_candidates = set(packet["candidate_entity_ids"])
    packet_evidence_ids = {item["evidence_id"] for item in packet["evidence"]}
    if any(entity_id not in packet_candidates for entity_id in selected_entity_ids):
        return _append_warning(
            model,
            f"Rejected adjudication for {hypothesis.subject_ref}: candidate selection outside packet candidates.",
        )
    if any(evidence_id not in packet_evidence_ids for evidence_id in evidence_ids):
        return _append_warning(
            model,
            f"Rejected adjudication for {hypothesis.subject_ref}: unknown evidence IDs referenced.",
        )

    updated_status = "unresolved"
    if len(selected_entity_ids) == 1:
        updated_status = "selected"
    elif len(selected_entity_ids) > 1:
        updated_status = "ambiguous"

    updated_hypothesis = replace(
        hypothesis,
        selected_entity_ids=tuple(selected_entity_ids),
        status=updated_status,
        rationale=f"{hypothesis.rationale}\nLLM adjudication: {rationale.strip()}",
    )

    updated_hypotheses = tuple(
        updated_hypothesis if row.subject_ref == hypothesis.subject_ref else row
        for row in model.hypotheses
    )
    allowed_selected = set(selected_entity_ids)
    updated_memberships = tuple(
        membership
        for membership in model.memberships
        if membership.subject_ref != hypothesis.subject_ref
        or membership.entity_id in allowed_selected
    )

    return replace(
        model,
        memberships=updated_memberships,
        hypotheses=updated_hypotheses,
    )
