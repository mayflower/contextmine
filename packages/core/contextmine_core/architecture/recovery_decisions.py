"""Deterministic design-decision recovery from ADR-like documents."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from .recovery_model import RecoveredArchitectureDecision, RecoveredArchitectureEntity
from .schemas import EvidenceRef

_SECTION_KEYS = ("context", "decision", "consequences", "alternatives")


def _normalize(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _contains_text(text: str, phrase: str) -> bool:
    candidate = _normalize(phrase)
    if not candidate:
        return False
    if any(char in candidate for char in "/:._-"):
        return candidate in text
    return re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text) is not None


def _structured_data(doc: dict[str, Any]) -> dict[str, Any]:
    payload = doc.get("structured_data")
    return payload if isinstance(payload, dict) else {}


def _decision_evidence(doc: dict[str, Any]) -> tuple[EvidenceRef, ...]:
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    structured = _structured_data(doc)
    file_path = str(meta.get("file_path") or "").strip()
    doc_id = str(doc.get("id") or "").strip()
    evidence: list[EvidenceRef] = []
    if file_path:
        evidence.append(EvidenceRef(kind="file", ref=file_path))
    if doc_id:
        evidence.append(EvidenceRef(kind="artifact", ref=doc_id))
    for key in _SECTION_KEYS:
        section_text = str(structured.get(key) or "").strip()
        if not section_text:
            continue
        anchor = file_path or doc_id
        if not anchor:
            continue
        evidence.append(EvidenceRef(kind="section", ref=f"{anchor}#{key}"))
    unique: list[EvidenceRef] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for ref in evidence:
        token = (ref.kind, ref.ref, ref.start_line, ref.end_line)
        if token in seen:
            continue
        seen.add(token)
        unique.append(ref)
    return tuple(unique)


def _entity_aliases(entity: RecoveredArchitectureEntity) -> set[str]:
    aliases = {
        _normalize(entity.name),
        _normalize(entity.entity_id),
        _normalize(entity.entity_id.split(":", 1)[1].replace("-", " ")),
    }
    container = _normalize(str(entity.attributes.get("container") or ""))
    if container:
        aliases.add(container)
        aliases.add(f"{container} runtime")

    extra_aliases = entity.attributes.get("aliases")
    if isinstance(extra_aliases, (list, tuple, set)):
        for value in extra_aliases:
            alias = _normalize(str(value or ""))
            if alias:
                aliases.add(alias)

    for ref in entity.evidence:
        if ref.kind != "file":
            continue
        repo_path = _normalize(ref.ref)
        if repo_path:
            aliases.add(repo_path)
            aliases.add(_normalize(PurePosixPath(ref.ref).name))
    return {alias for alias in aliases if alias}


def _entity_file_paths(entity: RecoveredArchitectureEntity) -> set[str]:
    return {
        _normalize(ref.ref)
        for ref in entity.evidence
        if ref.kind == "file" and _normalize(ref.ref)
    }


def _explicit_entity_ids(
    doc: dict[str, Any],
    entity_ids: set[str],
) -> tuple[str, ...]:
    structured = _structured_data(doc)
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    values: list[str] = []
    for source in (
        structured.get("affected_entity_ids"),
        meta.get("affected_entity_ids"),
    ):
        if not isinstance(source, list):
            continue
        for value in source:
            if isinstance(value, str) and value in entity_ids:
                values.append(value)
    return tuple(sorted(set(values)))


def _explicit_entity_refs(doc: dict[str, Any]) -> tuple[str, ...]:
    structured = _structured_data(doc)
    refs = structured.get("affected_entity_refs")
    if not isinstance(refs, list):
        return ()
    values = [str(value).strip() for value in refs if str(value or "").strip()]
    return tuple(values)


def _resolve_explicit_refs(
    refs: tuple[str, ...],
    entities: tuple[RecoveredArchitectureEntity, ...],
) -> tuple[tuple[str, ...], bool, str]:
    if not refs:
        return (), False, ""

    resolved_ids: set[str] = set()
    parts: list[str] = []
    all_strong = True
    entity_index = {entity.entity_id: entity for entity in entities}
    aliases = {entity.entity_id: _entity_aliases(entity) for entity in entities}
    file_paths = {entity.entity_id: _entity_file_paths(entity) for entity in entities}

    for raw_ref in refs:
        token = _normalize(raw_ref)
        matches: set[str] = set()
        match_kind = ""
        if raw_ref in entity_index:
            matches.add(raw_ref)
            match_kind = "entity_id"
        for entity in entities:
            if token in aliases[entity.entity_id]:
                matches.add(entity.entity_id)
                if not match_kind:
                    match_kind = "alias"
            if token and token in file_paths[entity.entity_id]:
                matches.add(entity.entity_id)
                match_kind = "repo_path"
        if not matches:
            all_strong = False
            parts.append(f"'{raw_ref}' unresolved")
            continue
        if len(matches) > 1:
            all_strong = False
            parts.append(f"'{raw_ref}' ambiguous across {sorted(matches)}")
        else:
            parts.append(f"'{raw_ref}' matched {next(iter(matches))} via {match_kind}")
        resolved_ids.update(matches)

    return tuple(sorted(resolved_ids)), all_strong and bool(resolved_ids), "; ".join(parts)


def _infer_affected_entity_ids(
    doc: dict[str, Any],
    entities: tuple[RecoveredArchitectureEntity, ...],
) -> tuple[tuple[str, ...], str]:
    text = _normalize(
        " ".join(
            str(part or "")
            for part in (
                doc.get("title"),
                doc.get("summary"),
                doc.get("text"),
                _structured_data(doc).get("decision"),
                _structured_data(doc).get("consequences"),
            )
        )
    )
    matched: list[str] = []
    reasons: list[str] = []
    for entity in entities:
        entity_matches = [
            alias for alias in sorted(_entity_aliases(entity)) if _contains_text(text, alias)
        ]
        if not entity_matches:
            continue
        matched.append(entity.entity_id)
        reasons.append(f"{entity.entity_id} via {entity_matches[0]!r}")
    return tuple(sorted(set(matched))), "; ".join(reasons)


def recover_architecture_decisions(
    docs: list[dict[str, Any]] | None,
    entities: tuple[RecoveredArchitectureEntity, ...],
) -> tuple[RecoveredArchitectureDecision, ...]:
    """Recover design decisions from structured ADRs and conservative text linking."""
    if not docs:
        return ()

    entity_ids = {entity.entity_id for entity in entities}
    decisions: list[RecoveredArchitectureDecision] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        structured = _structured_data(doc)
        explicit_ids = _explicit_entity_ids(doc, entity_ids)
        explicit_refs = _explicit_entity_refs(doc)
        linked_ids, strong_ref_links, ref_reason = _resolve_explicit_refs(explicit_refs, entities)
        inferred_ids, inferred_reason = _infer_affected_entity_ids(doc, entities)

        affected_entity_ids = tuple(sorted(set(explicit_ids) | set(linked_ids)))
        status = "confirmed" if explicit_ids or strong_ref_links else "hypothesis"
        confidence = 0.94 if explicit_ids or strong_ref_links else 0.66
        linking_reason = ref_reason

        if not affected_entity_ids:
            affected_entity_ids = inferred_ids
            linking_reason = inferred_reason
        elif status != "confirmed" and inferred_ids:
            affected_entity_ids = tuple(sorted(set(affected_entity_ids) | set(inferred_ids)))
            if inferred_reason:
                linking_reason = "; ".join(part for part in (ref_reason, inferred_reason) if part)

        if not affected_entity_ids:
            continue

        title = str(
            structured.get("title") or doc.get("title") or doc.get("name") or doc.get("id") or "Architecture Decision"
        ).strip()
        summary = str(
            structured.get("decision")
            or doc.get("summary")
            or doc.get("text")
            or title
        ).strip()
        evidence = _decision_evidence(doc)
        if not evidence:
            continue

        attributes: dict[str, Any] = {"source_doc_id": str(doc.get("id") or "")}
        for key in ("supersedes", "replaces", "status"):
            value = structured.get(key)
            if value is not None and str(value).strip():
                attributes[key] = str(value).strip()
        if linking_reason:
            attributes["linking_reason"] = linking_reason
        if status == "hypothesis" and len(affected_entity_ids) > 1:
            attributes["counter_evidence_entity_ids"] = list(affected_entity_ids)

        decisions.append(
            RecoveredArchitectureDecision(
                title=title,
                summary=summary,
                status=status,
                affected_entity_ids=affected_entity_ids,
                confidence=confidence,
                evidence=evidence,
                attributes=attributes,
            )
        )
    return tuple(decisions)
