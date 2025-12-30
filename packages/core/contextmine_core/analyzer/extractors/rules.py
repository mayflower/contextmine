"""Business Rule Candidate extractor.

Detects rule candidates by finding conditional branches that lead to
failure actions (raise/throw/return error) using Tree-sitter AST.

This is a deterministic extraction - no LLM involvement. We capture:
- The predicate (condition being checked)
- The failure action (raise/throw/return)
- Evidence (file path, line range)
- Container symbol (function/method containing the rule)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import get_treesitter_manager

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class FailureKind(str, Enum):
    """Type of failure action detected."""

    RAISE_EXCEPTION = "raise_exception"
    THROW_ERROR = "throw_error"
    RETURN_ERROR = "return_error"
    ASSERT_FAIL = "assert_fail"


@dataclass
class RuleCandidateDef:
    """Extracted rule candidate definition."""

    file_path: str
    start_line: int
    end_line: int
    predicate_text: str  # The condition being checked
    failure_text: str  # The failure action
    failure_kind: FailureKind
    language: str
    container_name: str | None = None  # Enclosing function/method name
    container_start: int | None = None
    confidence: float = 0.5  # Heuristic confidence score


@dataclass
class RulesExtraction:
    """Result of rule candidate extraction."""

    file_path: str
    candidates: list[RuleCandidateDef] = field(default_factory=list)


def extract_rule_candidates(file_path: str, content: str) -> RulesExtraction:
    """Extract rule candidates from source code using Tree-sitter.

    Detects patterns like:
    - Python: `if condition: raise Exception(...)`
    - Python: `if not condition: raise ValueError(...)`
    - JS/TS: `if (condition) { throw new Error(...) }`
    - JS/TS: `if (!condition) throw Error(...)`

    Args:
        file_path: Path to the source file
        content: Source code content

    Returns:
        RulesExtraction with detected rule candidates
    """
    result = RulesExtraction(file_path=file_path)

    language = detect_language(file_path)
    if language is None:
        return result

    manager = get_treesitter_manager()
    if not manager.is_available():
        logger.warning("Tree-sitter not available, skipping rule extraction")
        return result

    try:
        tree = manager.parse(file_path, content)
        root = tree.root_node

        if language == TreeSitterLanguage.PYTHON:
            candidates = _extract_python_rules(root, content, file_path)
        elif language in (TreeSitterLanguage.JAVASCRIPT, TreeSitterLanguage.TYPESCRIPT):
            candidates = _extract_js_rules(root, content, file_path, language.value)
        else:
            # Unsupported language for rule extraction
            candidates = []

        result.candidates = candidates

    except (ImportError, ValueError) as e:
        logger.warning("Failed to parse %s for rule extraction: %s", file_path, e)

    return result


def _extract_python_rules(root: Any, content: str, file_path: str) -> list[RuleCandidateDef]:
    """Extract rule candidates from Python code."""
    candidates: list[RuleCandidateDef] = []
    lines = content.split("\n")

    # Find all if statements
    for node in _traverse(root):
        if node.type == "if_statement":
            candidate = _check_python_if_for_rule(node, lines, file_path)
            if candidate:
                candidates.append(candidate)

        # Also check assert statements as rule candidates
        elif node.type == "assert_statement":
            candidate = _extract_python_assert(node, lines, file_path)
            if candidate:
                candidates.append(candidate)

    return candidates


def _check_python_if_for_rule(
    node: Any, lines: list[str], file_path: str
) -> RuleCandidateDef | None:
    """Check if a Python if statement represents a rule candidate."""
    # Get the condition
    condition_node = None
    consequence_node = None

    for child in node.children:
        if (
            child.type == "comparison_operator"
            or child.type == "boolean_operator"
            or child.type == "not_operator"
            or child.type == "identifier"
            or child.type == "attribute"
            or child.type == "call"
        ):
            condition_node = child
        elif child.type == "block":
            consequence_node = child
        elif child.type == ":":
            # Skip colon
            pass
        elif child.type in ("if", "elif"):
            # Skip keyword
            pass

    # If no condition found, try looking for it differently
    if condition_node is None:
        for child in node.children:
            if child.type not in ("if", "elif", ":", "block", "else_clause", "comment"):
                condition_node = child
                break

    if condition_node is None or consequence_node is None:
        return None

    # Check if the consequence contains a raise statement
    failure_node = None
    failure_kind = None

    for child in _traverse(consequence_node):
        if child.type == "raise_statement":
            failure_node = child
            failure_kind = FailureKind.RAISE_EXCEPTION
            break
        elif child.type == "return_statement":
            # Check if it's returning an error-like value
            text = _get_node_text(child, lines).lower()
            if any(err in text for err in ["error", "err", "fail", "none", "false"]):
                failure_node = child
                failure_kind = FailureKind.RETURN_ERROR
                break

    if failure_node is None or failure_kind is None:
        return None

    # Find container function/method
    container_name, container_start = _find_python_container(node)

    # Calculate confidence based on heuristics
    confidence = _calculate_confidence(condition_node, failure_node, lines)

    return RuleCandidateDef(
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        predicate_text=_get_node_text(condition_node, lines),
        failure_text=_get_node_text(failure_node, lines),
        failure_kind=failure_kind,
        language="python",
        container_name=container_name,
        container_start=container_start,
        confidence=confidence,
    )


def _extract_python_assert(node: Any, lines: list[str], file_path: str) -> RuleCandidateDef | None:
    """Extract rule candidate from Python assert statement."""
    # Assert has the form: assert condition, message
    condition_node = None

    for child in node.children:
        if child.type not in ("assert", ",") and condition_node is None:
            condition_node = child
            break

    if condition_node is None:
        return None

    container_name, container_start = _find_python_container(node)

    return RuleCandidateDef(
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        predicate_text=_get_node_text(condition_node, lines),
        failure_text=_get_node_text(node, lines),
        failure_kind=FailureKind.ASSERT_FAIL,
        language="python",
        container_name=container_name,
        container_start=container_start,
        confidence=0.6,  # Asserts are often explicit rules
    )


def _extract_js_rules(
    root: Any, content: str, file_path: str, language: str
) -> list[RuleCandidateDef]:
    """Extract rule candidates from JavaScript/TypeScript code."""
    candidates: list[RuleCandidateDef] = []
    lines = content.split("\n")

    for node in _traverse(root):
        if node.type == "if_statement":
            candidate = _check_js_if_for_rule(node, lines, file_path, language)
            if candidate:
                candidates.append(candidate)

    return candidates


def _check_js_if_for_rule(
    node: Any, lines: list[str], file_path: str, language: str
) -> RuleCandidateDef | None:
    """Check if a JS/TS if statement represents a rule candidate."""
    condition_node = None
    consequence_node = None

    for child in node.children:
        if child.type == "parenthesized_expression":
            condition_node = child
        elif child.type == "statement_block":
            consequence_node = child
        elif child.type == "throw_statement":
            # Shorthand: if (x) throw ...
            consequence_node = child

    if condition_node is None:
        return None

    # Check if consequence contains throw
    failure_node = None
    failure_kind = None

    if consequence_node is not None:
        for child in _traverse(consequence_node):
            if child.type == "throw_statement":
                failure_node = child
                failure_kind = FailureKind.THROW_ERROR
                break
            elif child.type == "return_statement":
                text = _get_node_text(child, lines).lower()
                if any(
                    err in text for err in ["error", "err", "fail", "null", "undefined", "false"]
                ):
                    failure_node = child
                    failure_kind = FailureKind.RETURN_ERROR
                    break

    if failure_node is None or failure_kind is None:
        return None

    container_name, container_start = _find_js_container(node)
    confidence = _calculate_confidence(condition_node, failure_node, lines)

    return RuleCandidateDef(
        file_path=file_path,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        predicate_text=_get_node_text(condition_node, lines),
        failure_text=_get_node_text(failure_node, lines),
        failure_kind=failure_kind,
        language=language,
        container_name=container_name,
        container_start=container_start,
        confidence=confidence,
    )


def _find_python_container(node: Any) -> tuple[str | None, int | None]:
    """Find the enclosing function/method for a Python node."""
    parent = node.parent
    while parent is not None:
        if parent.type in ("function_definition", "async_function_definition"):
            for child in parent.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8"), parent.start_point[0] + 1
        elif parent.type == "class_definition":
            # Inside a class but not in a method
            break
        parent = parent.parent
    return None, None


def _find_js_container(node: Any) -> tuple[str | None, int | None]:
    """Find the enclosing function/method for a JS/TS node."""
    parent = node.parent
    while parent is not None:
        if parent.type in (
            "function_declaration",
            "arrow_function",
            "method_definition",
            "function",
        ):
            # Try to find the name
            for child in parent.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8"), parent.start_point[0] + 1
                if child.type == "property_identifier":
                    return child.text.decode("utf-8"), parent.start_point[0] + 1
            # Arrow function might not have a name
            return None, parent.start_point[0] + 1
        parent = parent.parent
    return None, None


def _traverse(node: Any):
    """Traverse all nodes in the AST."""
    yield node
    for child in node.children:
        yield from _traverse(child)


def _get_node_text(node: Any, lines: list[str]) -> str:
    """Get the source text for a node."""
    start_line = node.start_point[0]
    end_line = node.end_point[0]
    start_col = node.start_point[1]
    end_col = node.end_point[1]

    if start_line == end_line:
        return lines[start_line][start_col:end_col]

    result = [lines[start_line][start_col:]]
    for i in range(start_line + 1, end_line):
        result.append(lines[i])
    result.append(lines[end_line][:end_col])

    return "\n".join(result)


def _calculate_confidence(condition_node: Any, failure_node: Any, lines: list[str]) -> float:
    """Calculate a heuristic confidence score for a rule candidate."""
    confidence = 0.5

    # Higher confidence for validation-like patterns
    condition_text = _get_node_text(condition_node, lines).lower()
    failure_text = _get_node_text(failure_node, lines).lower()

    # Keywords suggesting validation
    validation_keywords = [
        "valid",
        "invalid",
        "check",
        "verify",
        "require",
        "must",
        "should",
        "cannot",
        "allowed",
        "permit",
        "forbidden",
        "auth",
        "permission",
        "access",
    ]

    for keyword in validation_keywords:
        if keyword in condition_text or keyword in failure_text:
            confidence += 0.1

    # Cap at 0.95
    confidence = min(confidence, 0.95)

    return round(confidence, 2)


def get_candidate_natural_key(candidate: RuleCandidateDef) -> str:
    """Generate a stable natural key for a rule candidate."""
    # Use file path + line range + content hash
    content = f"{candidate.predicate_text}:{candidate.failure_text}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"rule:{candidate.file_path}:{candidate.start_line}:{content_hash}"


async def build_rule_candidates_graph(
    session: AsyncSession,
    collection_id: UUID,
    extractions: list[RulesExtraction],
) -> dict:
    """Build knowledge graph nodes for rule candidates.

    Creates:
    - RULE_CANDIDATE nodes with metadata

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
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stats = {"candidates_created": 0, "evidence_created": 0}

    for extraction in extractions:
        for candidate in extraction.candidates:
            natural_key = get_candidate_natural_key(candidate)

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.RULE_CANDIDATE,
                natural_key=natural_key,
                name=f"Rule: {candidate.predicate_text[:50]}...",
                meta={
                    "predicate": candidate.predicate_text,
                    "failure": candidate.failure_text,
                    "failure_kind": candidate.failure_kind.value,
                    "language": candidate.language,
                    "container_name": candidate.container_name,
                    "container_start": candidate.container_start,
                    "confidence": candidate.confidence,
                    "start_line": candidate.start_line,
                    "end_line": candidate.end_line,
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
            stats["candidates_created"] += 1

            # Create evidence
            existing = await session.execute(
                select(KnowledgeNodeEvidence.evidence_id).where(
                    KnowledgeNodeEvidence.node_id == node_id
                )
            )
            if not existing.scalar_one_or_none():
                evidence = KnowledgeEvidence(
                    file_path=candidate.file_path,
                    start_line=candidate.start_line,
                    end_line=candidate.end_line,
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
