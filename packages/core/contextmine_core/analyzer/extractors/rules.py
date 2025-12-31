"""Business Rule extraction using LLM.

Extracts business rules from code by:
1. Using Tree-sitter to parse code into functions/methods/classes
2. Sending each code unit to LLM for semantic analysis
3. LLM identifies business rules regardless of pattern or language

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
class CodeUnit:
    """A unit of code to analyze (function, method, class)."""

    name: str
    kind: str  # function, method, class
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str


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

    # Parse into code units using Tree-sitter
    code_units = _parse_code_units(file_path, content, language)

    if not code_units:
        # If no functions/classes found, analyze the whole file
        code_units = [
            CodeUnit(
                name="<module>",
                kind="module",
                file_path=file_path,
                start_line=1,
                end_line=len(content.split("\n")),
                content=content[:8000],  # Limit size
                language=language.value,
            )
        ]

    # Analyze each code unit
    for unit in code_units:
        try:
            rules = await _analyze_code_unit(unit, provider)
            result.rules.extend(rules)
        except Exception as e:
            logger.warning("Failed to analyze %s in %s: %s", unit.name, file_path, e)

    return result


def _parse_code_units(
    file_path: str,
    content: str,
    language: TreeSitterLanguage,
) -> list[CodeUnit]:
    """Parse source code into analyzable units using Tree-sitter."""
    units: list[CodeUnit] = []

    manager = get_treesitter_manager()
    if not manager.is_available():
        return units

    try:
        tree = manager.parse(file_path, content)
        root = tree.root_node
        lines = content.split("\n")

        # Get node types for this language
        function_types = _get_function_node_types(language)
        class_types = _get_class_node_types(language)

        for node in _traverse(root):
            if node.type in function_types:
                unit = _extract_unit(node, lines, file_path, language, "function")
                if unit:
                    units.append(unit)
            elif node.type in class_types:
                unit = _extract_unit(node, lines, file_path, language, "class")
                if unit:
                    units.append(unit)

    except Exception as e:
        logger.warning("Failed to parse %s: %s", file_path, e)

    return units


def _get_function_node_types(language: TreeSitterLanguage) -> set[str]:
    """Get AST node types that represent functions/methods for a language."""
    mapping = {
        TreeSitterLanguage.PYTHON: {"function_definition", "async_function_definition"},
        TreeSitterLanguage.TYPESCRIPT: {
            "function_declaration",
            "method_definition",
            "arrow_function",
        },
        TreeSitterLanguage.TSX: {"function_declaration", "method_definition", "arrow_function"},
        TreeSitterLanguage.JAVASCRIPT: {
            "function_declaration",
            "method_definition",
            "arrow_function",
        },
        TreeSitterLanguage.JAVA: {"method_declaration", "constructor_declaration"},
        TreeSitterLanguage.CSHARP: {"method_declaration", "constructor_declaration"},
        TreeSitterLanguage.GO: {"function_declaration", "method_declaration"},
        TreeSitterLanguage.RUST: {"function_item"},
        TreeSitterLanguage.RUBY: {"method", "singleton_method"},
        TreeSitterLanguage.PHP: {"function_definition", "method_declaration"},
        TreeSitterLanguage.C: {"function_definition"},
        TreeSitterLanguage.CPP: {"function_definition"},
    }
    return mapping.get(language, {"function_definition", "function_declaration"})


def _get_class_node_types(language: TreeSitterLanguage) -> set[str]:
    """Get AST node types that represent classes for a language."""
    mapping = {
        TreeSitterLanguage.PYTHON: {"class_definition"},
        TreeSitterLanguage.TYPESCRIPT: {"class_declaration"},
        TreeSitterLanguage.TSX: {"class_declaration"},
        TreeSitterLanguage.JAVASCRIPT: {"class_declaration"},
        TreeSitterLanguage.JAVA: {"class_declaration", "interface_declaration"},
        TreeSitterLanguage.CSHARP: {"class_declaration", "interface_declaration"},
        TreeSitterLanguage.GO: {"type_declaration"},
        TreeSitterLanguage.RUST: {"struct_item", "impl_item"},
        TreeSitterLanguage.RUBY: {"class", "module"},
        TreeSitterLanguage.PHP: {"class_declaration"},
        TreeSitterLanguage.C: {"struct_specifier"},
        TreeSitterLanguage.CPP: {"class_specifier", "struct_specifier"},
    }
    return mapping.get(language, {"class_definition", "class_declaration"})


def _extract_unit(
    node,
    lines: list[str],
    file_path: str,
    language: TreeSitterLanguage,
    kind: str,
) -> CodeUnit | None:
    """Extract a code unit from an AST node."""
    start_line = node.start_point[0]
    end_line = node.end_point[0]

    # Get the name
    name = _get_node_name(node, language)
    if not name:
        name = f"<anonymous_{kind}>"

    # Extract content
    content_lines = lines[start_line : end_line + 1]
    content = "\n".join(content_lines)

    # Skip very small units (likely not interesting)
    if len(content_lines) < 3:
        return None

    # Limit content size for LLM
    if len(content) > 8000:
        content = content[:8000] + "\n... (truncated)"

    return CodeUnit(
        name=name,
        kind=kind,
        file_path=file_path,
        start_line=start_line + 1,  # 1-indexed
        end_line=end_line + 1,
        content=content,
        language=language.value,
    )


def _get_node_name(node, language: TreeSitterLanguage) -> str | None:
    """Extract the name from a function/class AST node."""
    for child in node.children:
        if child.type in (
            "identifier",
            "name",
            "property_identifier",
            "type_identifier",
            "constant",
        ):
            return child.text.decode("utf-8")
    return None


def _traverse(node):
    """Traverse all nodes in the AST."""
    yield node
    for child in node.children:
        yield from _traverse(child)


async def _analyze_code_unit(
    unit: CodeUnit,
    provider: LLMProvider,
) -> list[ExtractedRule]:
    """Analyze a code unit for business rules using LLM."""
    prompt = f"""Analyze this {unit.language} code for business rules:

File: {unit.file_path}
{unit.kind.title()}: {unit.name}
Lines: {unit.start_line}-{unit.end_line}

```{unit.language}
{unit.content}
```

Identify all business rules in this code. Return an empty list if none found."""

    result = await provider.generate_structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_schema=ExtractionOutput,
        max_tokens=2048,
        temperature=0.0,
    )

    extracted: list[ExtractedRule] = []
    for rule in result.rules:
        # Adjust line numbers relative to unit
        actual_start = unit.start_line + rule.start_line - 1
        actual_end = unit.start_line + rule.end_line - 1

        extracted.append(
            ExtractedRule(
                name=rule.name,
                description=rule.description,
                category=rule.category,
                severity=rule.severity,
                natural_language=rule.natural_language,
                file_path=unit.file_path,
                start_line=actual_start,
                end_line=actual_end,
                evidence_snippet=rule.evidence_snippet,
                container_name=unit.name,
                language=unit.language,
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
