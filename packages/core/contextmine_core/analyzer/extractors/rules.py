"""Business Rule extraction using LLM.

Extracts business rules from code by:
1. Sending the entire file to LLM for semantic analysis (one call per file)
2. LLM identifies business rules regardless of pattern or language

This approach handles:
- Any programming language (all 12 Tree-sitter languages)
- Any human language (German, Spanish, Dutch method names/comments)
- Any pattern (decorators, annotations, schemas, guards, assertions, etc.)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from contextmine_core.treesitter.languages import detect_language
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class BusinessRuleDef(BaseModel):
    """A business rule identified by LLM."""

    name: str = Field(description="Short name for the rule")
    description: str = Field(description="What the rule enforces")
    category: str = Field(
        description="Category: validation, authorization, invariant, constraint, rate_limit, business_logic"
    )
    severity: str = Field(description="Severity: error, warning, info")
    natural_language: str = Field(
        description="The rule in plain language (e.g., 'Users must be at least 18 years old')"
    )
    evidence_snippet: str = Field(description="The specific code that implements this rule")
    start_line: int = Field(description="Starting line number of the evidence")
    end_line: int = Field(description="Ending line number of the evidence")


class ExtractionOutput(BaseModel):
    """Structured output from LLM for business rule extraction."""

    rules: list[BusinessRuleDef] = Field(
        default_factory=list,
        description="List of business rules found in this code. Empty if none found.",
    )
    reasoning: str = Field(description="Brief explanation of what was analyzed")


@dataclass
class ExtractedRule:
    """A business rule extracted from code."""

    name: str
    description: str
    category: str
    severity: str
    natural_language: str
    file_path: str
    start_line: int
    end_line: int
    evidence_snippet: str
    container_name: str
    language: str


@dataclass
class RulesExtraction:
    """Result of business rule extraction."""

    file_path: str
    rules: list[ExtractedRule] = field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are a business rule analyst. Analyze code to identify business rules.

A business rule is any constraint, validation, or policy that enforces business logic. Examples:
- Age restrictions: "Users must be 18+"
- Authorization: "Only admins can delete"
- Invariants: "Order total must equal sum of items"
- Rate limits: "Maximum 100 requests per minute"
- Data validation: "Email must be valid format"
- Business constraints: "Cannot withdraw more than balance"

Business rules can appear as:
- If statements with exceptions/errors
- Decorators or annotations (@validate, @Min, @requires_auth)
- Schema definitions (Pydantic, JSON Schema, DB constraints)
- Guard clauses
- Assertion statements
- Switch/match patterns
- Validation function calls
- Any other pattern that enforces a constraint

DO NOT include:
- Pure error handling (null checks for defensive coding)
- Technical infrastructure (connection handling, logging)
- Type conversions without business meaning

For each rule found, extract:
1. A short descriptive name
2. What it enforces (description)
3. Category (validation, authorization, invariant, constraint, rate_limit, business_logic)
4. Severity (error, warning, info)
5. Natural language statement
6. The exact code snippet that implements it
7. Line numbers

If no business rules are found, return an empty list. Be thorough but precise."""


async def extract_rules_from_file(
    file_path: str,
    content: str,
    provider: LLMProvider,
) -> RulesExtraction:
    """Extract business rules from a source file using LLM.

    Makes ONE LLM call per file to extract all business rules at once.

    Args:
        file_path: Path to the source file
        content: Source code content
        provider: LLM provider for analysis

    Returns:
        RulesExtraction with identified business rules
    """
    result = RulesExtraction(file_path=file_path)

    language = detect_language(file_path)
    if language is None:
        logger.debug("Unsupported language for %s, skipping", file_path)
        return result

    # Truncate very large files to fit in context
    lines = content.split("\n")
    max_lines = 500  # ~15k tokens typically
    if len(lines) > max_lines:
        # Take first part, it usually has most business logic
        content = "\n".join(lines[:max_lines])
        content += f"\n... (truncated, {len(lines) - max_lines} more lines)"

    # Analyze entire file in one LLM call
    try:
        rules = await _analyze_file(file_path, content, language.value, provider)
        result.rules.extend(rules)
    except Exception as e:
        logger.warning("Failed to analyze %s: %s", file_path, e)

    return result


async def _analyze_file(
    file_path: str,
    content: str,
    language: str,
    provider: LLMProvider,
) -> list[ExtractedRule]:
    """Analyze an entire file for business rules using LLM (one call per file)."""
    prompt = f"""Analyze this {language} source file for ALL business rules:

File: {file_path}

```{language}
{content}
```

Find and list ALL business rules in this file. Line numbers should be relative to the file (starting from 1).
Return an empty list if no business rules are found."""

    result = await provider.generate_structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_schema=ExtractionOutput,
        max_tokens=4096,  # Allow more tokens for full file analysis
        temperature=0.0,
    )

    extracted: list[ExtractedRule] = []
    for rule in result.rules:
        extracted.append(
            ExtractedRule(
                name=rule.name,
                description=rule.description,
                category=rule.category,
                severity=rule.severity,
                natural_language=rule.natural_language,
                file_path=file_path,
                start_line=rule.start_line,
                end_line=rule.end_line,
                evidence_snippet=rule.evidence_snippet,
                container_name="<file>",  # No longer tracking per-function
                language=language,
            )
        )

    return extracted


def get_rule_natural_key(rule: ExtractedRule) -> str:
    """Generate a stable natural key for a business rule."""
    content = f"{rule.file_path}:{rule.natural_language}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"rule:{rule.file_path}:{rule.start_line}:{content_hash}"


async def build_rules_graph(
    session: AsyncSession,
    collection_id: UUID,
    extractions: list[RulesExtraction],
) -> dict:
    """Build knowledge graph nodes for extracted business rules.

    Args:
        session: Database session
        collection_id: Collection UUID
        extractions: List of rule extractions

    Returns:
        Stats dict
    """
    from contextmine_core.models import (
        KnowledgeEvidence,
        KnowledgeNode,
        KnowledgeNodeEvidence,
        KnowledgeNodeKind,
    )
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stats = {"rules_created": 0, "evidence_created": 0}

    for extraction in extractions:
        for rule in extraction.rules:
            natural_key = get_rule_natural_key(rule)

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
                    "container_name": rule.container_name,
                    "language": rule.language,
                    "evidence_snippet": rule.evidence_snippet,
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
            node_id = result.scalar_one()
            stats["rules_created"] += 1

            # Create evidence
            evidence = KnowledgeEvidence(
                file_path=rule.file_path,
                start_line=rule.start_line,
                end_line=rule.end_line,
            )
            session.add(evidence)
            await session.flush()

            link = KnowledgeNodeEvidence(
                node_id=node_id,
                evidence_id=evidence.id,
            )
            session.add(link)
            stats["evidence_created"] += 1

    return stats
