"""Deterministic design-decision recovery from ADR-like documents."""

from __future__ import annotations

from typing import Any

from .recovery_model import RecoveredArchitectureDecision, RecoveredArchitectureEntity
from .schemas import EvidenceRef


def _decision_evidence(doc: dict[str, Any]) -> tuple[EvidenceRef, ...]:
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    file_path = str(meta.get("file_path") or "").strip()
    evidence: list[EvidenceRef] = []
    if file_path:
        evidence.append(EvidenceRef(kind="file", ref=file_path))
    doc_id = str(doc.get("id") or "").strip()
    if doc_id:
        evidence.append(EvidenceRef(kind="artifact", ref=doc_id))
    return tuple(evidence)


def _entity_aliases(entity: RecoveredArchitectureEntity) -> set[str]:
    aliases = {
        entity.name.lower(),
        entity.entity_id.lower(),
        entity.entity_id.split(":", 1)[1].replace("-", " ").lower(),
    }
    container = str(entity.attributes.get("container") or "").strip().lower()
    if container:
        aliases.add(container)
        aliases.add(f"{container} runtime")
    return {alias for alias in aliases if alias}


def _infer_affected_entity_ids(
    doc: dict[str, Any],
    entities: tuple[RecoveredArchitectureEntity, ...],
) -> tuple[str, ...]:
    text = " ".join(
        str(part or "").lower()
        for part in (
            doc.get("title"),
            doc.get("summary"),
            doc.get("text"),
        )
    )
    matched: list[str] = []
    for entity in entities:
        if any(alias in text for alias in _entity_aliases(entity)):
            matched.append(entity.entity_id)
    return tuple(sorted(set(matched)))


def recover_architecture_decisions(
    docs: list[dict[str, Any]] | None,
    entities: tuple[RecoveredArchitectureEntity, ...],
) -> tuple[RecoveredArchitectureDecision, ...]:
    """Recover design decisions from explicit ADR metadata or indirect text matches."""
    if not docs:
        return ()

    entity_ids = {entity.entity_id for entity in entities}
    decisions: list[RecoveredArchitectureDecision] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
        explicit_affected = tuple(
            entity_id
            for entity_id in meta.get("affected_entity_ids") or []
            if isinstance(entity_id, str) and entity_id in entity_ids
        )
        affected_entity_ids = explicit_affected or _infer_affected_entity_ids(doc, entities)
        if not affected_entity_ids:
            continue
        title = str(doc.get("title") or doc.get("name") or doc.get("id") or "Architecture Decision")
        summary = str(doc.get("summary") or doc.get("text") or title).strip()
        decisions.append(
            RecoveredArchitectureDecision(
                title=title,
                summary=summary,
                status="confirmed" if explicit_affected else "hypothesis",
                affected_entity_ids=affected_entity_ids,
                confidence=0.92 if explicit_affected else 0.66,
                evidence=_decision_evidence(doc),
                attributes={"source_doc_id": str(doc.get("id") or "")},
            )
        )
    return tuple(decisions)
