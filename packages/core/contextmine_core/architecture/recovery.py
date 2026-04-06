"""Deterministic in-memory architecture recovery from local evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from contextmine_core.twin.grouping import derive_arch_group

from .recovery_decisions import recover_architecture_decisions
from .recovery_llm import apply_adjudication, build_adjudication_packet
from .recovery_model import (
    RecoveredArchitectureEntity,
    RecoveredArchitectureHypothesis,
    RecoveredArchitectureMembership,
    RecoveredArchitectureModel,
    RecoveredArchitectureRelationship,
)
from .schemas import EvidenceRef

EXPLICIT_METADATA_SCORE = 0.98
STRUCTURAL_EDGE_BASE_SCORE = 0.78
STRUCTURAL_EDGE_BONUS = 0.04
STRUCTURAL_EDGE_CAP = 0.86
PATH_HEURISTIC_SCORE = 0.62
SELECTION_THRESHOLD = 0.6
CLOSE_ENOUGH_DELTA = 0.06

_ENTITY_CONFIDENCE = 0.9
_RELATIONSHIP_CONFIDENCE = 0.86

_NODE_KIND_TO_ENTITY_KIND = {
    "db_table": "data_store",
    "message_schema": "message_channel",
    "external_system": "external_system",
}


def _slug(value: str) -> str:
    text = str(value or "").strip().lower()
    return "-".join(part for part in text.replace("_", "-").split() if part)


def _title_from_slug(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split("-") if part)


def _container_display_name(container: str) -> str:
    if container == "api":
        return "API Runtime"
    if container == "worker":
        return "Worker Runtime"
    return f"{_title_from_slug(container)} Runtime"


def _node_meta(node: dict[str, Any]) -> dict[str, Any]:
    meta = node.get("meta")
    return meta if isinstance(meta, dict) else {}


def _node_kind(node: dict[str, Any]) -> str:
    kind = node.get("kind")
    return str(kind.value if hasattr(kind, "value") else kind or "").lower()


def _node_name(node: dict[str, Any]) -> str:
    return str(node.get("name") or node.get("natural_key") or node.get("id") or "").strip()


def _node_file_path(node: dict[str, Any]) -> str | None:
    file_path = _node_meta(node).get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None


def _node_evidence(node: dict[str, Any]) -> tuple[EvidenceRef, ...]:
    refs: list[EvidenceRef] = []
    file_path = _node_file_path(node)
    if file_path:
        refs.append(EvidenceRef(kind="file", ref=file_path))
    node_id = str(node.get("id") or "").strip()
    if node_id:
        refs.append(EvidenceRef(kind="node", ref=node_id))
    return tuple(refs)


def _dedupe_evidence(
    evidence: list[EvidenceRef] | tuple[EvidenceRef, ...],
) -> tuple[EvidenceRef, ...]:
    seen: set[tuple[str, str, int | None, int | None]] = set()
    deduped: list[EvidenceRef] = []
    for ref in evidence:
        key = (ref.kind, ref.ref, ref.start_line, ref.end_line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return tuple(deduped)


def _edge_ref(edge: dict[str, Any]) -> str:
    source = str(edge.get("source_node_id") or "")
    target = str(edge.get("target_node_id") or "")
    kind = str(edge.get("kind") or "edge")
    return f"{source}:{kind}:{target}"


def _edge_evidence(
    edge: dict[str, Any],
    source_node: dict[str, Any] | None,
    target_node: dict[str, Any] | None,
) -> tuple[EvidenceRef, ...]:
    refs = [EvidenceRef(kind="edge", ref=_edge_ref(edge))]
    if source_node is not None:
        refs.extend(_node_evidence(source_node))
    if target_node is not None:
        refs.extend(_node_evidence(target_node))
    return _dedupe_evidence(refs)


def _has_explicit_architecture(node: dict[str, Any]) -> bool:
    architecture_meta = _node_meta(node).get("architecture")
    if not isinstance(architecture_meta, dict):
        return False
    return bool(
        str(architecture_meta.get("domain") or "").strip()
        and str(architecture_meta.get("container") or "").strip()
    )


def _explicit_container_from_node(node: dict[str, Any]) -> str | None:
    if not _has_explicit_architecture(node):
        return None
    architecture_meta = _node_meta(node).get("architecture") or {}
    container = str(architecture_meta.get("container") or "").strip()
    return container or None


def _path_container_from_node(node: dict[str, Any]) -> str | None:
    file_path = _node_file_path(node)
    if not file_path or not (file_path.startswith("services/") or file_path.startswith("apps/")):
        return None
    group = derive_arch_group(file_path, {})
    if group is None:
        return None
    return group[1]


def _eligible_for_membership(node: dict[str, Any] | None) -> bool:
    if node is None:
        return False
    return _node_kind(node) in {"symbol", "file", "api_endpoint", "job"}


def _entity_id_for_node(node: dict[str, Any]) -> str | None:
    kind = _node_kind(node)
    if kind == "db_table":
        return f"data_store:{_node_name(node)}"
    if kind == "message_schema":
        return f"message_channel:{_slug(_node_name(node))}"
    if kind == "external_system":
        return f"external_system:{_slug(_node_name(node))}"
    if kind == "symbol":
        return f"component:{_slug(_node_name(node))}"
    return None


def _entity_kind_for_node(node: dict[str, Any]) -> str | None:
    kind = _node_kind(node)
    if kind == "symbol":
        return "component"
    return _NODE_KIND_TO_ENTITY_KIND.get(kind)


def _direct_candidate_map(
    nodes: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for node in nodes:
        if not _eligible_for_membership(node):
            continue

        node_id = str(node.get("id") or "")
        evidence = list(_node_evidence(node))
        explicit_container = _explicit_container_from_node(node)
        if explicit_container is not None:
            candidates[node_id][explicit_container] = {
                "score": EXPLICIT_METADATA_SCORE,
                "evidence": evidence,
                "signal": "explicit",
            }

        path_container = _path_container_from_node(node)
        if path_container is None:
            continue
        existing = candidates[node_id].get(path_container)
        if existing is None or float(existing["score"]) < PATH_HEURISTIC_SCORE:
            candidates[node_id][path_container] = {
                "score": PATH_HEURISTIC_SCORE,
                "evidence": evidence,
                "signal": "path",
            }
    return candidates


def _select_candidates(
    candidate_map: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, tuple[tuple[str, ...], str]]:
    selections: dict[str, tuple[tuple[str, ...], str]] = {}
    for subject_ref, candidates in candidate_map.items():
        if not candidates:
            selections[subject_ref] = ((), "unresolved")
            continue

        ranked = sorted(
            candidates.items(),
            key=lambda item: (-float(item[1]["score"]), item[0]),
        )
        top_score = float(ranked[0][1]["score"])
        selected = tuple(
            container
            for container, payload in ranked
            if float(payload["score"]) >= SELECTION_THRESHOLD
            and (top_score - float(payload["score"])) <= CLOSE_ENOUGH_DELTA
        )

        if not selected:
            selections[subject_ref] = ((), "unresolved")
        elif len(selected) == 1:
            selections[subject_ref] = (selected, "selected")
        else:
            selections[subject_ref] = (selected, "ambiguous")
    return selections


def _dedupe_memberships(
    memberships: list[RecoveredArchitectureMembership],
) -> list[RecoveredArchitectureMembership]:
    deduped: dict[tuple[str, str, str], RecoveredArchitectureMembership] = {}
    for membership in memberships:
        key = (membership.subject_ref, membership.entity_id, membership.relationship_kind)
        existing = deduped.get(key)
        if existing is None or membership.confidence > existing.confidence:
            deduped[key] = membership
        elif membership.confidence == existing.confidence:
            deduped[key] = replace(
                existing,
                evidence=_dedupe_evidence(list(existing.evidence) + list(membership.evidence)),
            )
    return [deduped[key] for key in sorted(deduped)]


def _infer_memberships(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[RecoveredArchitectureMembership], list[RecoveredArchitectureHypothesis]]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    candidate_map = _direct_candidate_map(nodes)
    initial_selections = _select_candidates(candidate_map)

    anchor_containers_by_subject = {
        subject_ref: selected
        for subject_ref, (selected, _status) in initial_selections.items()
        if selected
    }
    structural_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    structural_evidence: dict[str, dict[str, list[EvidenceRef]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for edge in edges:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        evidence = list(_edge_evidence(edge, source_node, target_node))

        for container in anchor_containers_by_subject.get(source_id, ()):
            if _eligible_for_membership(
                target_node
            ) and container not in anchor_containers_by_subject.get(target_id, ()):
                structural_counts[target_id][container] += 1
                structural_evidence[target_id][container].extend(evidence)

        for container in anchor_containers_by_subject.get(target_id, ()):
            if _eligible_for_membership(
                source_node
            ) and container not in anchor_containers_by_subject.get(source_id, ()):
                structural_counts[source_id][container] += 1
                structural_evidence[source_id][container].extend(evidence)

    for subject_ref, counts_by_container in structural_counts.items():
        subject_candidates = candidate_map.setdefault(subject_ref, {})
        for container, count in counts_by_container.items():
            score = min(
                STRUCTURAL_EDGE_BASE_SCORE + STRUCTURAL_EDGE_BONUS * max(count - 1, 0),
                STRUCTURAL_EDGE_CAP,
            )
            existing = subject_candidates.get(container)
            evidence = structural_evidence[subject_ref][container]
            if existing is None or score > float(existing["score"]):
                subject_candidates[container] = {
                    "score": score,
                    "evidence": evidence,
                    "signal": "structural",
                }
            elif score == float(existing["score"]):
                existing["evidence"] = list(existing["evidence"]) + evidence

    memberships: list[RecoveredArchitectureMembership] = []
    hypotheses: list[RecoveredArchitectureHypothesis] = []
    selections = _select_candidates(candidate_map)

    for node in nodes:
        subject_ref = str(node.get("id") or "")
        if not _eligible_for_membership(node):
            continue

        candidates = candidate_map.get(subject_ref, {})
        selected, status = selections.get(subject_ref, ((), "unresolved"))

        for container in selected:
            payload = candidates[container]
            memberships.append(
                RecoveredArchitectureMembership(
                    subject_ref=subject_ref,
                    entity_id=f"container:{container}",
                    relationship_kind="contained_in",
                    confidence=float(payload["score"]),
                    evidence=_dedupe_evidence(list(payload["evidence"])),
                    attributes={"signal": str(payload["signal"])},
                )
            )

        if status == "selected" and len(candidates) <= 1:
            continue

        if status == "selected":
            rationale = "Explicit architecture metadata outranks weaker heuristic evidence."
        elif status == "ambiguous":
            rationale = (
                "Multiple runtime candidates remain plausible after applying the scoring policy."
            )
        else:
            rationale = "No candidate cleared the selection threshold from local evidence."

        hypothesis_evidence = _dedupe_evidence(
            [ref for payload in candidates.values() for ref in list(payload["evidence"])]
        )
        hypotheses.append(
            RecoveredArchitectureHypothesis(
                subject_ref=subject_ref,
                candidate_entity_ids=tuple(
                    f"container:{container}" for container in sorted(candidates)
                ),
                selected_entity_ids=tuple(f"container:{container}" for container in selected),
                rationale=rationale,
                status=status,
                confidence=max(
                    (float(payload["score"]) for payload in candidates.values()), default=0.0
                ),
                evidence=hypothesis_evidence or _node_evidence(node),
            )
        )

    return _dedupe_memberships(memberships), hypotheses


def _collect_container_entities(
    nodes: list[dict[str, Any]],
    memberships: list[RecoveredArchitectureMembership],
) -> list[RecoveredArchitectureEntity]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    evidence_by_container: dict[str, list[EvidenceRef]] = defaultdict(list)
    for membership in memberships:
        if not membership.entity_id.startswith("container:"):
            continue
        container = membership.entity_id.split(":", 1)[1]
        evidence_by_container[container].extend(membership.evidence)
        node = node_by_id.get(membership.subject_ref)
        if node is not None:
            evidence_by_container[container].extend(_node_evidence(node))

    entities: list[RecoveredArchitectureEntity] = []
    for container, evidence in sorted(evidence_by_container.items()):
        entities.append(
            RecoveredArchitectureEntity(
                entity_id=f"container:{container}",
                kind="container",
                name=_container_display_name(container),
                confidence=_ENTITY_CONFIDENCE,
                evidence=_dedupe_evidence(evidence),
                attributes={"container": container},
            )
        )
    return entities


def _collect_direct_entities(
    nodes: list[dict[str, Any]],
    component_subject_ids: set[str],
) -> list[RecoveredArchitectureEntity]:
    entities: dict[str, RecoveredArchitectureEntity] = {}
    for node in nodes:
        if _node_kind(node) == "symbol" and str(node.get("id") or "") not in component_subject_ids:
            continue
        entity_id = _entity_id_for_node(node)
        entity_kind = _entity_kind_for_node(node)
        if entity_id is None or entity_kind is None:
            continue
        entities[entity_id] = RecoveredArchitectureEntity(
            entity_id=entity_id,
            kind=entity_kind,
            name=_node_name(node),
            confidence=_ENTITY_CONFIDENCE,
            evidence=_node_evidence(node),
            attributes={"source_node_id": str(node.get("id") or "")},
        )
    return [entities[key] for key in sorted(entities)]


def _relationship_sources_for_node(
    node: dict[str, Any],
    memberships_by_subject: dict[str, list[RecoveredArchitectureMembership]],
) -> list[str]:
    node_id = str(node.get("id") or "")
    direct_entity_id = _entity_id_for_node(node)
    if direct_entity_id is not None and _entity_kind_for_node(node) != "component":
        return [direct_entity_id]
    memberships = memberships_by_subject.get(node_id, [])
    if memberships:
        return [membership.entity_id for membership in memberships]
    if direct_entity_id is not None:
        return [direct_entity_id]
    return []


def _lift_relationships(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    memberships: list[RecoveredArchitectureMembership],
) -> list[RecoveredArchitectureRelationship]:
    node_by_id = {str(node.get("id")): node for node in nodes}
    memberships_by_subject: dict[str, list[RecoveredArchitectureMembership]] = defaultdict(list)
    for membership in memberships:
        memberships_by_subject[membership.subject_ref].append(membership)

    relationships: dict[tuple[str, str, str], RecoveredArchitectureRelationship] = {}
    for edge in edges:
        source_id = str(edge.get("source_node_id") or "")
        target_id = str(edge.get("target_node_id") or "")
        source_node = node_by_id.get(source_id)
        target_node = node_by_id.get(target_id)
        if source_node is None or target_node is None:
            continue

        source_entity_ids = _relationship_sources_for_node(source_node, memberships_by_subject)
        target_entity_ids = _relationship_sources_for_node(target_node, memberships_by_subject)
        if not source_entity_ids or not target_entity_ids:
            continue

        relation_kind = str(edge.get("kind") or "depends_on")
        evidence = _edge_evidence(edge, source_node, target_node)
        for source_entity_id in source_entity_ids:
            for target_entity_id in target_entity_ids:
                if source_entity_id == target_entity_id:
                    continue
                key = (source_entity_id, target_entity_id, relation_kind)
                existing = relationships.get(key)
                if existing is None:
                    relationships[key] = RecoveredArchitectureRelationship(
                        source_entity_id=source_entity_id,
                        target_entity_id=target_entity_id,
                        kind=relation_kind,
                        confidence=_RELATIONSHIP_CONFIDENCE,
                        evidence=evidence,
                    )
                    continue
                relationships[key] = replace(
                    existing,
                    evidence=_dedupe_evidence(list(existing.evidence) + list(evidence)),
                    confidence=max(existing.confidence, _RELATIONSHIP_CONFIDENCE),
                )

    return [relationships[key] for key in sorted(relationships)]


def _run_adjudicator(llm_adjudicator: Any, packet: dict[str, Any]) -> Any:
    if hasattr(llm_adjudicator, "adjudicate"):
        return llm_adjudicator.adjudicate(packet)
    if callable(llm_adjudicator):
        return llm_adjudicator(packet)
    raise TypeError("llm_adjudicator must be callable or expose adjudicate(packet).")


def recover_architecture_model(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    docs: list[dict[str, Any]] | None = None,
    llm_adjudicator: Any | None = None,
) -> RecoveredArchitectureModel:
    """Recover a deterministic architecture model from local graph facts only."""

    memberships, hypotheses = _infer_memberships(nodes, edges)
    component_subject_ids = {membership.subject_ref for membership in memberships}
    container_entities = _collect_container_entities(nodes, memberships)
    direct_entities = _collect_direct_entities(nodes, component_subject_ids)
    relationships = _lift_relationships(nodes, edges, memberships)

    entity_by_id: dict[str, RecoveredArchitectureEntity] = {}
    for entity in container_entities + direct_entities:
        entity_by_id[entity.entity_id] = entity
    decisions = recover_architecture_decisions(
        docs,
        tuple(entity_by_id[key] for key in sorted(entity_by_id)),
    )

    model = RecoveredArchitectureModel(
        entities=tuple(entity_by_id[key] for key in sorted(entity_by_id)),
        relationships=tuple(relationships),
        memberships=tuple(memberships),
        hypotheses=tuple(hypotheses),
        decisions=decisions,
    )
    if llm_adjudicator is None:
        return model

    for hypothesis in tuple(model.hypotheses):
        if hypothesis.status not in {"ambiguous", "unresolved"}:
            continue
        packet = build_adjudication_packet(model=model, hypothesis=hypothesis)
        try:
            adjudication = _run_adjudicator(llm_adjudicator, packet)
        except Exception as exc:  # noqa: BLE001
            model = replace(
                model,
                warnings=tuple(
                    list(model.warnings)
                    + [f"Malformed adjudication for {hypothesis.subject_ref}: {exc}"]
                ),
            )
            continue
        model = apply_adjudication(
            model=model,
            hypothesis=hypothesis,
            packet=packet,
            adjudication=adjudication,
        )
    return model
