"""LLM-based rule candidate labeling service.

Converts deterministic RULE_CANDIDATE nodes into normalized BUSINESS_RULE nodes
using LLM labeling with structured output validation.

Key principles:
- LLM only labels existing candidates, never discovers new rules
- Strict JSON schema validation via Pydantic
- Temperature 0 for deterministic output
- Idempotent: content hash prevents relabeling unchanged candidates
- Citations: references back to evidence and candidate IDs
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.models import KnowledgeNode
    from contextmine_core.research.llm.provider import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Pydantic schemas for structured LLM output
class BusinessRuleOutput(BaseModel):
    """Structured output from LLM for a business rule."""

    name: str = Field(description="Short descriptive name for the rule (e.g., 'Age Validation')")
    description: str = Field(description="Clear description of what the rule enforces")
    category: str = Field(
        description="Category: validation, authorization, invariant, constraint, other"
    )
    severity: str = Field(description="Severity: error, warning, info")
    natural_language: str = Field(
        description="The rule expressed in natural language (e.g., 'Users must be at least 18 years old')"
    )
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0.0, le=1.0)
    is_valid_rule: bool = Field(
        description="True if this is a genuine business rule, False if it's just error handling"
    )
    reasoning: str = Field(description="Brief explanation of the classification")


class LabelingResult(BaseModel):
    """Result of labeling a single candidate."""

    candidate_id: str
    rule: BusinessRuleOutput
    raw_response: str | None = None  # For audit


@dataclass
class LabelingStats:
    """Statistics from a labeling run."""

    candidates_processed: int = 0
    rules_created: int = 0
    skipped_unchanged: int = 0
    skipped_invalid: int = 0
    errors: int = 0


LABELING_SYSTEM_PROMPT = """You are a business rule analyst. Your task is to analyze code snippets
that contain potential business rules and determine if they represent actual business rules.

A business rule is a constraint or validation that enforces business logic, policy, or domain requirements.
Examples:
- Age validation: "Users must be at least 18 years old"
- Authorization: "Only admins can delete records"
- Invariant: "Order total must equal sum of line items"

NOT business rules:
- Generic error handling (null checks for defensive coding)
- Technical validation (type checks, format validation)
- Infrastructure concerns (connection checks, timeout handling)

For each candidate, provide:
1. A short descriptive name
2. A clear description
3. The category (validation, authorization, invariant, constraint, other)
4. Severity (error, warning, info)
5. The rule in natural language
6. Your confidence (0.0-1.0)
7. Whether this is a valid business rule
8. Brief reasoning

Be conservative - only mark as valid if it's clearly a business rule."""


async def label_rule_candidates(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
    batch_size: int = 10,
    force_relabel: bool = False,
) -> LabelingStats:
    """Label rule candidates using LLM.

    Fetches unlabeled RULE_CANDIDATE nodes, sends them to LLM for classification,
    and creates BUSINESS_RULE nodes for valid rules.

    Args:
        session: Database session
        collection_id: Collection UUID
        provider: LLM provider instance
        batch_size: Number of candidates to process per batch
        force_relabel: If True, relabel all candidates (ignore content hash)

    Returns:
        LabelingStats with processing statistics
    """
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    stats = LabelingStats()

    # Fetch unlabeled candidates
    stmt = select(KnowledgeNode).where(
        KnowledgeNode.collection_id == collection_id,
        KnowledgeNode.kind == KnowledgeNodeKind.RULE_CANDIDATE,
    )

    result = await session.execute(stmt)
    candidates = result.scalars().all()

    logger.info("Found %d rule candidates to process", len(candidates))

    for candidate in candidates:
        try:
            labeled = await _label_single_candidate(
                session=session,
                collection_id=collection_id,
                candidate=candidate,
                provider=provider,
                force_relabel=force_relabel,
            )

            stats.candidates_processed += 1

            if labeled is None:
                stats.skipped_unchanged += 1
            elif labeled.rule.is_valid_rule:
                await _create_business_rule(
                    session=session,
                    collection_id=collection_id,
                    candidate=candidate,
                    labeling=labeled,
                )
                stats.rules_created += 1
            else:
                stats.skipped_invalid += 1

        except Exception:
            logger.exception("Error labeling candidate %s", candidate.id)
            stats.errors += 1

    return stats


async def _label_single_candidate(
    session: AsyncSession,
    collection_id: UUID,
    candidate: KnowledgeNode,
    provider: LLMProvider,
    force_relabel: bool,
) -> LabelingResult | None:
    """Label a single candidate.

    Returns None if candidate was already labeled and content unchanged.
    """
    from contextmine_core.models import (
        KnowledgeEdge,
        KnowledgeEdgeKind,
        KnowledgeNode,
    )
    from sqlalchemy import select

    # Compute content hash for idempotency
    meta = candidate.meta or {}
    content_hash = _compute_content_hash(meta)

    # Check if already labeled with same content
    if not force_relabel:
        existing_edge = await session.execute(
            select(KnowledgeEdge.id).where(
                KnowledgeEdge.collection_id == collection_id,
                KnowledgeEdge.source_node_id == candidate.id,
                KnowledgeEdge.kind == KnowledgeEdgeKind.RULE_DERIVED_FROM_CANDIDATE,
            )
        )
        if existing_edge.scalar_one_or_none():
            # Check if content hash matches
            existing_rule = await session.execute(
                select(KnowledgeNode)
                .join(
                    KnowledgeEdge,
                    KnowledgeEdge.target_node_id == KnowledgeNode.id,
                )
                .where(
                    KnowledgeEdge.source_node_id == candidate.id,
                    KnowledgeEdge.kind == KnowledgeEdgeKind.RULE_DERIVED_FROM_CANDIDATE,
                )
            )
            rule_node = existing_rule.scalar_one_or_none()
            if rule_node and rule_node.meta and rule_node.meta.get("source_hash") == content_hash:
                logger.debug("Skipping unchanged candidate %s", candidate.id)
                return None

    # Build evidence bundle for LLM
    predicate = meta.get("predicate", "")
    failure = meta.get("failure", "")
    language = meta.get("language", "unknown")
    container = meta.get("container_name", "")

    prompt = f"""Analyze this code snippet that may contain a business rule:

Language: {language}
Container function: {container or "N/A"}

Condition being checked:
```
{predicate}
```

Failure action:
```
{failure}
```

Determine if this represents a business rule and provide structured analysis."""

    # Call LLM with structured output
    result = await provider.generate_structured(
        system=LABELING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_schema=BusinessRuleOutput,
        max_tokens=1024,
        temperature=0.0,
    )

    return LabelingResult(
        candidate_id=str(candidate.id),
        rule=result,
        raw_response=result.model_dump_json(),
    )


async def _create_business_rule(
    session: AsyncSession,
    collection_id: UUID,
    candidate: KnowledgeNode,
    labeling: LabelingResult,
) -> UUID:
    """Create a BUSINESS_RULE node and link to candidate."""
    from contextmine_core.models import (
        KnowledgeEdge,
        KnowledgeEdgeKind,
        KnowledgeEvidence,
        KnowledgeNode,
        KnowledgeNodeEvidence,
        KnowledgeNodeKind,
    )
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    rule = labeling.rule
    source_hash = _compute_content_hash(candidate.meta or {})

    # Generate natural key for the business rule
    natural_key = f"rule:labeled:{candidate.id}:{source_hash[:8]}"

    # Upsert the business rule node
    stmt = pg_insert(KnowledgeNode).values(
        collection_id=collection_id,
        kind=KnowledgeNodeKind.BUSINESS_RULE,
        natural_key=natural_key,
        name=rule.name,
        meta={
            "description": rule.description,
            "category": rule.category,
            "severity": rule.severity,
            "natural_language": rule.natural_language,
            "confidence": rule.confidence,
            "reasoning": rule.reasoning,
            "source_candidate_id": str(candidate.id),
            "source_hash": source_hash,
            "raw_response": labeling.raw_response,
        },
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_node_natural",
        set_={
            "name": stmt.excluded.name,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeNode.id)

    result = await session.execute(stmt)
    rule_node_id = result.scalar_one()

    # Create edge: BUSINESS_RULE <-- RULE_DERIVED_FROM_CANDIDATE -- CANDIDATE
    edge_exists = await session.execute(
        select(KnowledgeEdge.id).where(
            KnowledgeEdge.collection_id == collection_id,
            KnowledgeEdge.source_node_id == candidate.id,
            KnowledgeEdge.target_node_id == rule_node_id,
            KnowledgeEdge.kind == KnowledgeEdgeKind.RULE_DERIVED_FROM_CANDIDATE,
        )
    )
    if not edge_exists.scalar_one_or_none():
        session.add(
            KnowledgeEdge(
                collection_id=collection_id,
                source_node_id=candidate.id,
                target_node_id=rule_node_id,
                kind=KnowledgeEdgeKind.RULE_DERIVED_FROM_CANDIDATE,
                meta={"labeling_model": "llm"},
            )
        )

    # Link to same evidence as candidate
    candidate_evidence = await session.execute(
        select(KnowledgeNodeEvidence.evidence_id).where(
            KnowledgeNodeEvidence.node_id == candidate.id
        )
    )
    for (evidence_id,) in candidate_evidence:
        existing = await session.execute(
            select(KnowledgeNodeEvidence.evidence_id).where(
                KnowledgeNodeEvidence.node_id == rule_node_id,
                KnowledgeNodeEvidence.evidence_id == evidence_id,
            )
        )
        if not existing.scalar_one_or_none():
            session.add(
                KnowledgeNodeEvidence(
                    node_id=rule_node_id,
                    evidence_id=evidence_id,
                )
            )

    # Also create RULE_EVIDENCED_BY edge
    evidence_nodes = await session.execute(
        select(KnowledgeEvidence.id)
        .join(
            KnowledgeNodeEvidence,
            KnowledgeNodeEvidence.evidence_id == KnowledgeEvidence.id,
        )
        .where(KnowledgeNodeEvidence.node_id == candidate.id)
    )
    for (evidence_id,) in evidence_nodes:
        edge_exists = await session.execute(
            select(KnowledgeEdge.id).where(
                KnowledgeEdge.collection_id == collection_id,
                KnowledgeEdge.source_node_id == rule_node_id,
                KnowledgeEdge.kind == KnowledgeEdgeKind.RULE_EVIDENCED_BY,
            )
        )
        if not edge_exists.scalar_one_or_none():
            # Create a special evidence edge - source is rule, target doesn't exist
            # We'll use meta to store evidence_id since evidence isn't a node
            session.add(
                KnowledgeEdge(
                    collection_id=collection_id,
                    source_node_id=rule_node_id,
                    target_node_id=candidate.id,  # Link back to candidate as proxy
                    kind=KnowledgeEdgeKind.RULE_EVIDENCED_BY,
                    meta={"evidence_id": str(evidence_id)},
                )
            )

    return rule_node_id


def _compute_content_hash(meta: dict) -> str:
    """Compute a content hash for idempotency checks."""
    # Use relevant fields for hashing
    content = json.dumps(
        {
            "predicate": meta.get("predicate", ""),
            "failure": meta.get("failure", ""),
            "failure_kind": meta.get("failure_kind", ""),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]
