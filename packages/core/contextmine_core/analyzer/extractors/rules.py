"""Hybrid AST+LLM business rule extraction.

Extracts business rules from code by:
1. Using tree-sitter to split files into code units (functions/methods/classes)
2. Sending each code unit to the LLM for semantic analysis
3. Falling back to whole-file analysis for files without parseable structure

The AST provides structure (function boundaries, container names, accurate line numbers).
The LLM provides understanding (what is a business rule vs defensive coding).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language
from contextmine_core.treesitter.manager import get_treesitter_manager
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


@dataclass
class _CodeUnit:
    """A function, method, or class extracted by tree-sitter."""

    name: str
    kind: str  # "function", "method", "class"
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent_name: str | None = None


# ============================================================================
# Tree-sitter node types per language for function/method/class detection
# ============================================================================

_FUNCTION_NODE_TYPES: dict[TreeSitterLanguage, set[str]] = {
    TreeSitterLanguage.PYTHON: {"function_definition"},
    TreeSitterLanguage.TYPESCRIPT: {"function_declaration", "method_definition", "arrow_function"},
    TreeSitterLanguage.TSX: {"function_declaration", "method_definition", "arrow_function"},
    TreeSitterLanguage.JAVASCRIPT: {"function_declaration", "method_definition", "arrow_function"},
    TreeSitterLanguage.JAVA: {"method_declaration", "constructor_declaration"},
    TreeSitterLanguage.GO: {"function_declaration", "method_declaration"},
    TreeSitterLanguage.RUST: {"function_item"},
    TreeSitterLanguage.PHP: {"function_definition", "method_declaration"},
    TreeSitterLanguage.RUBY: {"method", "singleton_method"},
    TreeSitterLanguage.CSHARP: {"method_declaration", "constructor_declaration"},
    TreeSitterLanguage.C: {"function_definition"},
    TreeSitterLanguage.CPP: {"function_definition"},
}

_CLASS_NODE_TYPES: dict[TreeSitterLanguage, set[str]] = {
    TreeSitterLanguage.PYTHON: {"class_definition"},
    TreeSitterLanguage.TYPESCRIPT: {"class_declaration"},
    TreeSitterLanguage.TSX: {"class_declaration"},
    TreeSitterLanguage.JAVASCRIPT: {"class_declaration"},
    TreeSitterLanguage.JAVA: {"class_declaration", "interface_declaration", "enum_declaration"},
    TreeSitterLanguage.GO: {"type_declaration"},
    TreeSitterLanguage.RUST: {"impl_item", "struct_item"},
    TreeSitterLanguage.PHP: {"class_declaration"},
    TreeSitterLanguage.RUBY: {"class", "module"},
    TreeSitterLanguage.CSHARP: {"class_declaration", "interface_declaration"},
    TreeSitterLanguage.C: set(),
    TreeSitterLanguage.CPP: {"class_specifier"},
}


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
7. Line numbers (IMPORTANT: use the ORIGINAL file line numbers, not relative to the snippet)

If no business rules are found, return an empty list. Be thorough but precise."""


# ============================================================================
# AST-based code unit splitting
# ============================================================================


def _extract_code_units(
    file_path: str,
    content: str,
    language: TreeSitterLanguage,
) -> list[_CodeUnit]:
    """Split a file into code units (functions/methods/classes) using tree-sitter."""
    manager = get_treesitter_manager()
    try:
        tree = manager.parse(file_path, content)
    except Exception:
        return []

    root = tree.root_node
    func_types = _FUNCTION_NODE_TYPES.get(language, set())
    class_types = _CLASS_NODE_TYPES.get(language, set())

    units: list[_CodeUnit] = []
    _traverse_for_units(content, root, func_types, class_types, units, parent_name=None)
    return units


def _get_node_name(content: str, node: Any) -> str:
    """Extract the name identifier from a function/method/class node."""
    # Try common field names first
    for field_name in ("name", "declarator"):
        child = node.child_by_field_name(field_name)
        if child is not None:
            if child.type == "function_declarator":
                # C/C++: dig into declarator
                inner = child.child_by_field_name("declarator")
                if inner:
                    return content[inner.start_byte : inner.end_byte].strip()
            return content[child.start_byte : child.end_byte].strip()

    # Fallback: first identifier child
    for child in node.children:
        if child.type == "identifier":
            return content[child.start_byte : child.end_byte].strip()
    return ""


def _traverse_for_units(
    content: str,
    node: Any,
    func_types: set[str],
    class_types: set[str],
    units: list[_CodeUnit],
    parent_name: str | None,
) -> None:
    """Walk the AST and collect code units."""
    if node.type in class_types:
        name = _get_node_name(content, node)
        if name:
            units.append(
                _CodeUnit(
                    name=name,
                    kind="class",
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent_name=parent_name,
                )
            )
            # Recurse into class body to find methods
            for child in node.children:
                _traverse_for_units(
                    content, child, func_types, class_types, units, parent_name=name
                )
            return

    if node.type in func_types:
        name = _get_node_name(content, node)
        if name:
            # Include preceding decorators/annotations
            actual_start = _include_decorators(content, node)
            actual_start_line = content[:actual_start].count("\n") + 1
            units.append(
                _CodeUnit(
                    name=name,
                    kind="method" if parent_name else "function",
                    start_byte=actual_start,
                    end_byte=node.end_byte,
                    start_line=actual_start_line,
                    end_line=node.end_point[0] + 1,
                    parent_name=parent_name,
                )
            )
            return  # Don't recurse into nested functions for separate units

    for child in node.children:
        _traverse_for_units(content, child, func_types, class_types, units, parent_name)


def _include_decorators(content: str, node: Any) -> int:
    """Return the byte offset including any decorators/annotations before the node."""
    start = node.start_byte
    prev = node.prev_sibling
    # Python: decorated_definition wraps function_definition
    if node.parent and node.parent.type == "decorated_definition":
        return node.parent.start_byte
    # Java/C#: check for preceding annotation/attribute nodes
    while prev is not None:
        if prev.type in {
            "decorator",
            "marker_annotation",
            "annotation",
            "attribute_list",
            "comment",
            "block_comment",
            "line_comment",
        }:
            start = prev.start_byte
            prev = prev.prev_sibling
        else:
            break
    return start


# ============================================================================
# LLM analysis
# ============================================================================


async def extract_rules_from_file(
    file_path: str,
    content: str,
    provider: LLMProvider,
) -> RulesExtraction:
    """Extract business rules using hybrid AST+LLM approach.

    1. Tree-sitter splits the file into code units (functions/methods/classes)
    2. Each unit is sent to the LLM with its container context
    3. For files without parseable structure, falls back to whole-file analysis
    """
    result = RulesExtraction(file_path=file_path)

    language = detect_language(file_path)
    if language is None:
        logger.debug("Unsupported language for %s, skipping", file_path)
        return result

    # Try AST-based splitting
    units = _extract_code_units(file_path, content, language)

    if units:
        # Hybrid mode: analyze each code unit with LLM concurrently
        async def _analyze_one(unit: _CodeUnit) -> list[ExtractedRule]:
            container = f"{unit.parent_name}.{unit.name}" if unit.parent_name else unit.name
            try:
                return await _analyze_code_unit(
                    file_path, content, unit, language.value, provider, container
                )
            except Exception as e:
                logger.warning("Failed to analyze %s::%s: %s", file_path, unit.name, e)
                return []

        batch_results = await asyncio.gather(*[_analyze_one(u) for u in units])
        for rules in batch_results:
            result.rules.extend(rules)
    else:
        # Fallback: whole-file analysis (for config files, scripts without functions, etc.)
        try:
            rules = await _analyze_whole_file(file_path, content, language.value, provider)
            result.rules.extend(rules)
        except Exception as e:
            logger.warning("Failed to analyze %s: %s", file_path, e)

    return result


async def _analyze_code_unit(
    file_path: str,
    file_content: str,
    unit: _CodeUnit,
    language: str,
    provider: LLMProvider,
    container_name: str,
) -> list[ExtractedRule]:
    """Analyze a single code unit (function/method/class) with the LLM."""
    unit_content = file_content[unit.start_byte : unit.end_byte]
    prompt = f"""Analyze this {language} {unit.kind} for business rules:

File: {file_path}
{unit.kind.title()}: {container_name}
Lines: {unit.start_line}-{unit.end_line}

```{language}
{unit_content}
```

Line numbers in your response MUST be absolute file line numbers (starting from {unit.start_line}).
Return an empty list if no business rules are found."""

    llm_result = await provider.generate_structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_schema=ExtractionOutput,
        max_tokens=2048,
        temperature=0.0,
    )

    extracted: list[ExtractedRule] = []
    for rule in llm_result.rules:
        # Clamp line numbers to the unit's range
        start = max(rule.start_line, unit.start_line)
        end = min(rule.end_line, unit.end_line)
        extracted.append(
            ExtractedRule(
                name=rule.name,
                description=rule.description,
                category=rule.category,
                severity=rule.severity,
                natural_language=rule.natural_language,
                file_path=file_path,
                start_line=start,
                end_line=end,
                evidence_snippet=rule.evidence_snippet,
                container_name=container_name,
                language=language,
            )
        )

    return extracted


async def _analyze_whole_file(
    file_path: str,
    content: str,
    language: str,
    provider: LLMProvider,
) -> list[ExtractedRule]:
    """Analyze an entire file for business rules (fallback when AST splitting not possible)."""
    lines = content.split("\n")
    total_lines = len(lines)
    # For large files, analyze in chunks rather than truncating
    max_lines_per_chunk = 500
    if total_lines <= max_lines_per_chunk:
        return await _analyze_file_chunk(file_path, content, language, provider, 1, "<file>")

    all_rules: list[ExtractedRule] = []
    for chunk_start in range(0, total_lines, max_lines_per_chunk):
        chunk_end = min(chunk_start + max_lines_per_chunk, total_lines)
        chunk = "\n".join(lines[chunk_start:chunk_end])
        start_line = chunk_start + 1
        try:
            rules = await _analyze_file_chunk(
                file_path, chunk, language, provider, start_line, "<file>"
            )
            all_rules.extend(rules)
        except Exception as e:
            logger.warning(
                "Failed to analyze %s lines %d-%d: %s", file_path, start_line, chunk_end, e
            )
    return all_rules


async def _analyze_file_chunk(
    file_path: str,
    content: str,
    language: str,
    provider: LLMProvider,
    start_line_offset: int,
    container_name: str,
) -> list[ExtractedRule]:
    """Analyze a chunk of a file."""
    prompt = f"""Analyze this {language} source file for ALL business rules:

File: {file_path}
Starting at line: {start_line_offset}

```{language}
{content}
```

Find and list ALL business rules. Line numbers should be absolute (starting from {start_line_offset}).
Return an empty list if no business rules are found."""

    llm_result = await provider.generate_structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_schema=ExtractionOutput,
        max_tokens=4096,
        temperature=0.0,
    )

    extracted: list[ExtractedRule] = []
    for rule in llm_result.rules:
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
                container_name=container_name,
                language=language,
            )
        )
    return extracted


# ============================================================================
# Natural key and graph building
# ============================================================================


def get_rule_natural_key(rule: ExtractedRule) -> str:
    """Generate a stable natural key for a business rule.

    Uses source location (file + line + snippet hash) rather than LLM output
    so that identical code produces the same key across LLM runs.
    """
    content = f"{rule.file_path}:{rule.start_line}:{rule.evidence_snippet}"
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
