"""Constrained LLM adjudication helpers for architecture recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any, Protocol

from .recovery_model import (
    RecoveredArchitectureDecision,
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
)
from .schemas import EvidenceRef

_ALLOWED_OPERATIONS = [
    "select_existing_candidates",
    "suggest_merge",
    "suggest_split",
    "suggest_rename",
]


class RecoveryAdjudicator(Protocol):
    """Minimal protocol for adjudicating one hypothesis packet."""

    def adjudicate(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Return an adjudication payload for one hypothesis packet."""


def _append_warning(model: RecoveredArchitectureModel, warning: str) -> RecoveredArchitectureModel:
    return replace(model, warnings=tuple(list(model.warnings) + [warning]))


def _dedupe_evidence(evidence: list[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    deduped: list[EvidenceRef] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for ref in evidence:
        key = (ref.kind, ref.ref, ref.start_line, ref.end_line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _relevant_decisions(
    model: RecoveredArchitectureModel,
    candidate_entity_ids: set[str],
) -> tuple[RecoveredArchitectureDecision, ...]:
    return tuple(
        decision
        for decision in model.decisions
        if candidate_entity_ids.intersection(decision.affected_entity_ids)
    )


def _snippet_from_decision(decision: RecoveredArchitectureDecision) -> str:
    title = decision.title.strip() or "Architecture decision"
    summary = decision.summary.strip()
    if summary:
        return f"{title}: {summary}"
    return title


def _snippet_from_entity(entity: RecoveredArchitectureEntity) -> str:
    return f"{entity.kind} {entity.name} ({entity.entity_id})"


def _snippet_from_membership(membership: RecoveredArchitectureMembership) -> str:
    signal = str(membership.attributes.get("signal") or "local evidence").replace("_", " ")
    return (
        f"{membership.subject_ref} {membership.relationship_kind} {membership.entity_id} "
        f"from {signal}"
    )


def _snippet_from_ref(
    *,
    ref: EvidenceRef,
    hypothesis: RecoveredArchitectureHypothesis,
    decision_by_ref: dict[tuple[str, str], RecoveredArchitectureDecision],
    entity_by_ref: dict[tuple[str, str], RecoveredArchitectureEntity],
    membership_by_ref: dict[tuple[str, str], RecoveredArchitectureMembership],
) -> str:
    key = (ref.kind, ref.ref)
    if key in decision_by_ref:
        return _snippet_from_decision(decision_by_ref[key])
    if key in membership_by_ref:
        return _snippet_from_membership(membership_by_ref[key])
    if key in entity_by_ref:
        return _snippet_from_entity(entity_by_ref[key])
    if ref.kind == "section":
        anchor, _, section = ref.ref.partition("#")
        label = section or "section"
        return f"Structured excerpt from {PurePosixPath(anchor).name} ({label})"
    if ref.kind == "file":
        return f"Code or doc excerpt from {PurePosixPath(ref.ref).name} at {ref.ref}"
    if ref.kind == "edge":
        parts = ref.ref.split(":")
        if len(parts) >= 3:
            source = parts[0]
            kind = parts[-2]
            target = parts[-1]
            return f"Relationship evidence {source} -> {target} ({kind})"
    if ref.kind == "node":
        if ref.ref == hypothesis.subject_ref:
            return f"Subject node {hypothesis.subject_ref}: {hypothesis.rationale}"
        return f"Node evidence for {ref.ref}"
    return f"{ref.kind.title()} evidence for {ref.ref}"


def build_snippet_bundle(
    *,
    model: RecoveredArchitectureModel,
    hypothesis: RecoveredArchitectureHypothesis,
) -> dict[str, Any]:
    """Build deterministic evidence packets with real local snippets."""

    candidate_entity_ids = set(hypothesis.candidate_entity_ids)
    memberships = tuple(model.memberships_for(hypothesis.subject_ref))
    candidate_entities = tuple(
        entity for entity in model.entities if entity.entity_id in candidate_entity_ids
    )
    decisions = _relevant_decisions(model, candidate_entity_ids)

    supporting_refs = _dedupe_evidence(
        list(hypothesis.evidence)
        + [ref for membership in memberships for ref in membership.evidence]
        + [ref for entity in candidate_entities for ref in entity.evidence]
    )
    counter_refs = _dedupe_evidence([ref for decision in decisions for ref in decision.evidence])
    all_refs = _dedupe_evidence(list(supporting_refs) + list(counter_refs))

    decision_by_ref = {
        (ref.kind, ref.ref): decision
        for decision in decisions
        for ref in decision.evidence
    }
    entity_by_ref = {
        (ref.kind, ref.ref): entity
        for entity in candidate_entities
        for ref in entity.evidence
    }
    membership_by_ref = {
        (ref.kind, ref.ref): membership
        for membership in memberships
        for ref in membership.evidence
    }

    items: list[dict[str, str]] = []
    item_by_key: dict[tuple[str, str, int | None, int | None], dict[str, str]] = {}
    for index, ref in enumerate(all_refs, start=1):
        item = {
            "evidence_id": f"ev-{index}",
            "kind": ref.kind,
            "ref": ref.ref,
            "snippet": _snippet_from_ref(
                ref=ref,
                hypothesis=hypothesis,
                decision_by_ref=decision_by_ref,
                entity_by_ref=entity_by_ref,
                membership_by_ref=membership_by_ref,
            ),
        }
        items.append(item)
        item_by_key[(ref.kind, ref.ref, ref.start_line, ref.end_line)] = item

    supporting_items = [
        item_by_key[(ref.kind, ref.ref, ref.start_line, ref.end_line)] for ref in supporting_refs
    ]
    counter_items = [
        item_by_key[(ref.kind, ref.ref, ref.start_line, ref.end_line)] for ref in counter_refs
    ]
    warnings: list[str] = []
    if len(items) <= 1 and not counter_items:
        warnings.append(
            f"Sparse evidence bundle for {hypothesis.subject_ref}: only {len(items)} local evidence item"
            f"{'' if len(items) == 1 else 's'} and no counter-evidence snippets."
        )

    return {
        "evidence": items,
        "supporting_evidence": supporting_items,
        "counter_evidence": counter_items,
        "warnings": warnings,
    }


def build_adjudication_packet(
    *,
    model: RecoveredArchitectureModel,
    hypothesis: RecoveredArchitectureHypothesis,
) -> dict[str, Any]:
    """Build a packet containing only packet-local evidence and candidate IDs."""

    snippet_bundle = build_snippet_bundle(model=model, hypothesis=hypothesis)
    candidates = [
        {
            "entity_id": entity.entity_id,
            "kind": entity.kind,
            "name": entity.name,
            "confidence": entity.confidence,
        }
        for entity in sorted(
            (entity for entity in model.entities if entity.entity_id in hypothesis.candidate_entity_ids),
            key=lambda row: row.entity_id,
        )
    ]

    return {
        "subject_ref": hypothesis.subject_ref,
        "status": hypothesis.status,
        "candidate_entity_ids": list(hypothesis.candidate_entity_ids),
        "selected_entity_ids": list(hypothesis.selected_entity_ids),
        "hypothesis": {
            "subject_ref": hypothesis.subject_ref,
            "status": hypothesis.status,
            "rationale": hypothesis.rationale,
            "confidence": hypothesis.confidence,
            "candidate_entity_ids": list(hypothesis.candidate_entity_ids),
            "selected_entity_ids": list(hypothesis.selected_entity_ids),
        },
        "candidates": candidates,
        "evidence": snippet_bundle["evidence"],
        "supporting_evidence": snippet_bundle["supporting_evidence"],
        "counter_evidence": snippet_bundle["counter_evidence"],
        "allowed_evidence_ids": [item["evidence_id"] for item in snippet_bundle["evidence"]],
        "allowed_operations": list(_ALLOWED_OPERATIONS),
        "warnings": list(snippet_bundle["warnings"]),
    }


def validate_adjudication(
    *,
    hypothesis: RecoveredArchitectureHypothesis,
    packet: dict[str, Any],
    adjudication: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate packet-local adjudication output."""

    if not isinstance(adjudication, dict):
        return None, f"Malformed adjudication for {hypothesis.subject_ref}."

    selected_entity_ids = adjudication.get("selected_entity_ids")
    rationale = adjudication.get("rationale")
    evidence_ids = adjudication.get("evidence_ids")

    if not isinstance(selected_entity_ids, list) or not all(
        isinstance(item, str) for item in selected_entity_ids
    ):
        return None, f"Malformed adjudication for {hypothesis.subject_ref}."
    if not isinstance(rationale, str) or not rationale.strip():
        return None, f"Malformed adjudication for {hypothesis.subject_ref}."
    if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
        return None, f"Malformed adjudication for {hypothesis.subject_ref}."

    packet_candidates = set(packet["candidate_entity_ids"])
    packet_evidence_ids = set(packet["allowed_evidence_ids"])
    if any(entity_id not in packet_candidates for entity_id in selected_entity_ids):
        return (
            None,
            f"Rejected adjudication for {hypothesis.subject_ref}: candidate selection outside packet candidates.",
        )
    if any(evidence_id not in packet_evidence_ids for evidence_id in evidence_ids):
        return (
            None,
            f"Rejected adjudication for {hypothesis.subject_ref}: unknown evidence IDs referenced.",
        )

    for field_name in ("merge_entity_ids", "split_entity_ids"):
        values = adjudication.get(field_name)
        if values is None:
            continue
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            return None, f"Malformed adjudication for {hypothesis.subject_ref}."
        if any(entity_id not in packet_candidates for entity_id in values):
            return (
                None,
                f"Rejected adjudication for {hypothesis.subject_ref}: {field_name} outside packet candidates.",
            )

    rename_suggestions = adjudication.get("rename_suggestions")
    if rename_suggestions is not None:
        if not isinstance(rename_suggestions, dict):
            return None, f"Malformed adjudication for {hypothesis.subject_ref}."
        for entity_id, value in rename_suggestions.items():
            if entity_id not in packet_candidates or not isinstance(value, str) or not value.strip():
                return (
                    None,
                    f"Rejected adjudication for {hypothesis.subject_ref}: rename suggestions outside packet candidates.",
                )

    return adjudication, None


def _suggestion_suffix(adjudication: dict[str, Any]) -> str:
    suffixes: list[str] = []
    merge_entity_ids = adjudication.get("merge_entity_ids")
    if isinstance(merge_entity_ids, list) and merge_entity_ids:
        suffixes.append(f"Merge suggestion: {', '.join(merge_entity_ids)}.")
    split_entity_ids = adjudication.get("split_entity_ids")
    if isinstance(split_entity_ids, list) and split_entity_ids:
        suffixes.append(f"Split suggestion: {', '.join(split_entity_ids)}.")
    rename_suggestions = adjudication.get("rename_suggestions")
    if isinstance(rename_suggestions, dict) and rename_suggestions:
        suffixes.append(
            "Rename suggestion: "
            + ", ".join(f"{entity_id} -> {name}" for entity_id, name in sorted(rename_suggestions.items()))
            + "."
        )
    return " ".join(suffixes)


def apply_adjudication(
    *,
    model: RecoveredArchitectureModel,
    hypothesis: RecoveredArchitectureHypothesis,
    packet: dict[str, Any],
    adjudication: Any,
) -> RecoveredArchitectureModel:
    """Apply a validated adjudication without inventing new entities or evidence."""

    validated, warning = validate_adjudication(
        hypothesis=hypothesis,
        packet=packet,
        adjudication=adjudication,
    )
    if warning is not None:
        return _append_warning(model, warning)
    assert validated is not None

    selected_entity_ids = validated["selected_entity_ids"]
    rationale = validated["rationale"]

    updated_status = "unresolved"
    if len(selected_entity_ids) == 1:
        updated_status = "selected"
    elif len(selected_entity_ids) > 1:
        updated_status = "ambiguous"

    rationale_parts = [f"{hypothesis.rationale}\nLLM adjudication: {rationale.strip()}"]
    suggestion_suffix = _suggestion_suffix(validated)
    if suggestion_suffix:
        rationale_parts.append(suggestion_suffix)

    updated_hypothesis = replace(
        hypothesis,
        selected_entity_ids=tuple(selected_entity_ids),
        status=updated_status,
        rationale="\n".join(rationale_parts),
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
