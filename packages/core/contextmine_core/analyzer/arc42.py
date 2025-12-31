"""arc42 Architecture Twin generator.

Generates arc42-style architecture documentation from extracted knowledge graph facts.
Uses LLM synthesis to produce coherent narrative prose with evidence citations.
Every statement is evidence-backed or explicitly marked as inferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Pydantic Models for LLM-generated sections
# -----------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    """Reference to source evidence."""

    file_path: str = Field(description="Path to the source file")
    line_start: int | None = Field(default=None, description="Start line number")
    line_end: int | None = Field(default=None, description="End line number")
    snippet: str | None = Field(default=None, description="Code/text snippet")


class Arc42SubSection(BaseModel):
    """A subsection within an arc42 section."""

    title: str = Field(description="Subsection heading")
    content: str = Field(description="Prose content for this subsection")
    evidence_refs: list[EvidenceRef] = Field(
        default_factory=list, description="References backing this content"
    )


class Arc42SectionOutput(BaseModel):
    """LLM-generated arc42 section content."""

    narrative: str = Field(description="Opening paragraph summarizing this section")
    subsections: list[Arc42SubSection] = Field(
        default_factory=list, description="Detailed subsections"
    )
    confidence: float = Field(ge=0, le=1, description="0.0=fully inferred, 1.0=fully evidenced")
    needs_human_input: list[str] = Field(
        default_factory=list, description="Questions/info that would improve this section"
    )


# -----------------------------------------------------------------------------
# Data Classes for Document Structure
# -----------------------------------------------------------------------------


@dataclass
class Arc42Section:
    """A single arc42 section."""

    id: str  # e.g., "context", "building-blocks"
    title: str
    content: str
    evidence_count: int = 0
    confidence: float = 1.0
    needs_human_input: list[str] = field(default_factory=list)


@dataclass
class Arc42Document:
    """Complete arc42 architecture document."""

    sections: list[Arc42Section] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    collection_id: UUID | None = None

    def to_markdown(self) -> str:
        """Render as full markdown document with evidence and confidence."""
        lines = ["# Architecture Documentation (arc42)\n"]
        lines.append(f"*Generated: {self.generated_at.isoformat()}*\n")
        lines.append("---\n")

        for section in self.sections:
            lines.append(f"## {section.title}\n")
            lines.append(section.content)
            lines.append("")

            # Show confidence warning if low
            if section.confidence < 0.7:
                lines.append(f"\n> ⚠️ Confidence: {section.confidence:.0%}\n")

            # Show evidence count if available
            if section.evidence_count > 0:
                lines.append(f"*Based on {section.evidence_count} extracted facts*\n")

            # Show prompts for human input
            if section.needs_human_input:
                lines.append("\n**To improve this section, please provide:**")
                for question in section.needs_human_input:
                    lines.append(f"- {question}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    def get_section(self, section_id: str) -> Arc42Section | None:
        """Get a specific section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None


@dataclass
class DriftItem:
    """A single drift finding."""

    kind: str  # "added", "removed", "changed"
    category: str  # "endpoint", "table", "job", "rule", etc.
    name: str
    details: str


@dataclass
class DriftReport:
    """Report comparing stored vs current architecture."""

    items: list[DriftItem] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_markdown(self) -> str:
        """Render as markdown."""
        if not self.items:
            return "# Architecture Drift Report\n\n*No changes detected.*"

        lines = ["# Architecture Drift Report\n"]
        lines.append(f"*Generated: {self.generated_at.isoformat()}*\n")

        # Group by kind
        added = [i for i in self.items if i.kind == "added"]
        removed = [i for i in self.items if i.kind == "removed"]
        changed = [i for i in self.items if i.kind == "changed"]

        if added:
            lines.append("## Added\n")
            for item in added:
                lines.append(f"- **{item.category}**: {item.name}")
                if item.details:
                    lines.append(f"  - {item.details}")
            lines.append("")

        if removed:
            lines.append("## Removed\n")
            for item in removed:
                lines.append(f"- **{item.category}**: {item.name}")
                if item.details:
                    lines.append(f"  - {item.details}")
            lines.append("")

        if changed:
            lines.append("## Changed\n")
            for item in changed:
                lines.append(f"- **{item.category}**: {item.name}")
                if item.details:
                    lines.append(f"  - {item.details}")
            lines.append("")

        lines.append(
            f"\n*Total: {len(added)} added, {len(removed)} removed, {len(changed)} changed*"
        )
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# LLM Prompts for each section
# -----------------------------------------------------------------------------

CONTEXT_SECTION_PROMPT = """You are an architecture documentation expert generating arc42 Section 3: Context & Scope.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Write in clear, professional prose (minimize bullet lists, prefer paragraphs)
2. Explain the SYSTEM BOUNDARY - what's inside vs outside the system
3. Describe EXTERNAL INTERFACES - how other systems/users interact
4. Identify EXTERNAL DEPENDENCIES - what the system relies on
5. Every statement should be backed by the provided evidence
6. If key information is missing, note it in needs_human_input

You MUST include subsections for:
- "System Boundary" - what's inside/outside the system
- "External Interfaces" - APIs, protocols, entry points
- "External Dependencies" - services, databases, third-party systems"""

BUILDING_BLOCKS_PROMPT = """You are an architecture documentation expert generating arc42 Section 5: Building Blocks.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Write in clear, professional prose explaining the component structure
2. Describe the main COMPONENTS and their responsibilities
3. Explain RELATIONSHIPS between components
4. Include the DATABASE SCHEMA as a key building block if available
5. Reference specific files/modules as evidence
6. If key information is missing, note it in needs_human_input

You MUST include subsections for:
- "Component Overview" - main modules/packages and their purposes
- "Key Abstractions" - important classes, interfaces, or patterns
- "Data Model" - database tables and their relationships (if any)"""

RUNTIME_PROMPT = """You are an architecture documentation expert generating arc42 Section 6: Runtime View.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Write in clear, professional prose describing how the system behaves at runtime
2. Describe KEY EXECUTION FLOWS - typical request handling, data processing
3. Identify ENTRY POINTS - where requests/events enter the system
4. Explain ASYNC OPERATIONS - background jobs, scheduled tasks
5. Reference specific endpoints, jobs, or functions as evidence
6. If key information is missing, note it in needs_human_input

You MUST include subsections for:
- "Entry Points" - API endpoints, event handlers, CLI commands
- "Key Scenarios" - typical request flows described in prose
- "Background Processing" - jobs, schedulers, async tasks (if any)"""

DEPLOYMENT_PROMPT = """You are an architecture documentation expert generating arc42 Section 7: Deployment View.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Write in clear, professional prose describing deployment infrastructure
2. Describe JOBS AND WORKFLOWS - CI/CD, scheduled tasks, automation
3. Identify INFRASTRUCTURE COMPONENTS from job definitions
4. Note TRIGGERS AND SCHEDULES for automated processes
5. Reference specific job configurations as evidence
6. If key information is missing, note it in needs_human_input

You MUST include subsections for:
- "Infrastructure Overview" - what infrastructure components are used
- "Jobs and Workflows" - CI/CD, scheduled jobs, automation
- "Deployment Process" - how the system gets deployed (if evident)"""

CROSSCUTTING_PROMPT = """You are an architecture documentation expert generating arc42 Section 8: Crosscutting Concepts.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Write in clear, professional prose describing patterns and practices
2. Describe VALIDATION PATTERNS from business rules
3. Identify ERROR HANDLING approaches from rule candidates
4. Note SECURITY CONSIDERATIONS visible in the code
5. Reference specific rules or patterns as evidence
6. If key information is missing, note it in needs_human_input

You MUST include subsections for:
- "Validation & Business Rules" - how input/data is validated
- "Error Handling" - patterns for handling errors and edge cases
- "Security Considerations" - authentication, authorization patterns (if visible)"""

RISKS_PROMPT = """You are an architecture documentation expert generating arc42 Section 11: Risks & Technical Debt.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Write in clear, professional prose assessing potential risks
2. Identify UNREVIEWED CODE from unlabeled rule candidates
3. Note POTENTIAL TECH DEBT indicators
4. Suggest areas for MANUAL REVIEW
5. Be constructive, not alarmist
6. If key information is missing, note it in needs_human_input

You MUST include subsections for:
- "Technical Debt Indicators" - code that may need attention
- "Recommended Reviews" - areas that should be manually examined
- "Risk Assessment" - potential architectural risks (if evident)"""

GLOSSARY_PROMPT = """You are an architecture documentation expert generating arc42 Section 12: Glossary.

Synthesize the provided extracted facts into coherent architecture documentation.

Requirements:
1. Define DOMAIN TERMS from database tables and business rules
2. Explain TECHNICAL TERMS specific to this codebase
3. Convert technical names (snake_case, camelCase) to readable definitions
4. Group related terms if appropriate
5. Keep definitions concise but clear

Output format:
- Each term should have a clear, 1-2 sentence definition
- Include the source (table name, rule name) in parentheses"""


# -----------------------------------------------------------------------------
# Helper functions for querying knowledge graph
# -----------------------------------------------------------------------------


async def _get_api_endpoints(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get API endpoints from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
        )
        .limit(50)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_graphql_types(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get GraphQL types from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.GRAPHQL_TYPE,
        )
        .limit(30)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_rpc_services(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get RPC service methods from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SERVICE_RPC,
        )
        .limit(30)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_db_tables(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get database tables from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
        )
        .limit(50)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_jobs(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get jobs from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.JOB,
        )
        .limit(30)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_business_rules(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get business rules from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
        )
        .limit(30)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_rule_candidates(
    session: AsyncSession, collection_id: UUID
) -> list[tuple[str, dict]]:
    """Get rule candidates from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.RULE_CANDIDATE,
        )
        .limit(30)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_symbols(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get symbols from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
        )
        .limit(50)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_files(session: AsyncSession, collection_id: UUID) -> list[tuple[str, dict]]:
    """Get files from knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.FILE,
        )
        .limit(100)
    )
    return [(row[0], row[1] or {}) for row in result.all()]


async def _get_edge_count(session: AsyncSession, collection_id: UUID) -> int:
    """Get total edge count in knowledge graph."""
    from contextmine_core.models import KnowledgeEdge
    from sqlalchemy import func, select

    result = await session.scalar(
        select(func.count(KnowledgeEdge.id)).where(
            KnowledgeEdge.collection_id == collection_id,
        )
    )
    return result or 0


async def _get_readme_content(session: AsyncSession, collection_id: UUID) -> str | None:
    """Try to find README content from indexed documents."""
    from contextmine_core.models import Document, Source
    from sqlalchemy import or_, select

    result = await session.execute(
        select(Document.content_markdown)
        .join(Source, Document.source_id == Source.id)
        .where(
            Source.collection_id == collection_id,
            or_(
                Document.uri.ilike("%readme%"),
                Document.title.ilike("%readme%"),
            ),
        )
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return row[:5000] if row else None  # Limit content size


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def _format_endpoints(endpoints: list[tuple[str, dict]]) -> str:
    """Format endpoints for LLM context."""
    if not endpoints:
        return "*No API endpoints found*"

    lines = []
    for name, meta in endpoints[:20]:
        method = meta.get("method", "GET")
        summary = meta.get("summary", "")
        tags = meta.get("tags", [])
        line = f"- `{method} {name}`"
        if summary:
            line += f" - {summary}"
        if tags:
            line += f" (tags: {', '.join(tags)})"
        lines.append(line)

    if len(endpoints) > 20:
        lines.append(f"... and {len(endpoints) - 20} more endpoints")

    return "\n".join(lines)


def _format_graphql_types(types: list[tuple[str, dict]]) -> str:
    """Format GraphQL types for LLM context."""
    if not types:
        return "*No GraphQL types found*"

    lines = []
    for name, meta in types[:15]:
        kind = meta.get("kind", "type")
        lines.append(f"- `{name}` ({kind})")

    if len(types) > 15:
        lines.append(f"... and {len(types) - 15} more types")

    return "\n".join(lines)


def _format_rpc_services(services: list[tuple[str, dict]]) -> str:
    """Format RPC services for LLM context."""
    if not services:
        return "*No RPC services found*"

    lines = []
    for name, meta in services[:15]:
        service = meta.get("service", "")
        lines.append(f"- `{service}.{name}`" if service else f"- `{name}`")

    return "\n".join(lines)


def _format_tables(tables: list[tuple[str, dict]]) -> str:
    """Format database tables for LLM context."""
    if not tables:
        return "*No database tables found*"

    lines = []
    for name, meta in tables[:20]:
        columns = meta.get("columns", [])
        col_info = f" ({len(columns)} columns)" if columns else ""
        lines.append(f"- `{name}`{col_info}")

        # Add column details for important tables
        if columns and len(lines) < 30:
            for col in columns[:5]:
                col_name = col.get("name", "")
                col_type = col.get("type", "")
                if col_name:
                    lines.append(f"  - {col_name}: {col_type}")
            if len(columns) > 5:
                lines.append(f"  - ... and {len(columns) - 5} more columns")

    if len(tables) > 20:
        lines.append(f"... and {len(tables) - 20} more tables")

    return "\n".join(lines)


def _format_jobs(jobs: list[tuple[str, dict]]) -> str:
    """Format jobs for LLM context."""
    if not jobs:
        return "*No jobs or workflows found*"

    lines = []
    for name, meta in jobs:
        framework = meta.get("framework", "unknown")
        schedule = meta.get("schedule", "")
        triggers = meta.get("triggers", [])

        line = f"- **{name}** ({framework})"
        if schedule:
            line += f"\n  - Schedule: `{schedule}`"
        if triggers:
            line += f"\n  - Triggers: {', '.join(triggers)}"

        lines.append(line)

    return "\n".join(lines)


def _format_rules(rules: list[tuple[str, dict]]) -> str:
    """Format business rules for LLM context."""
    if not rules:
        return "*No business rules found*"

    lines = []
    for name, meta in rules[:20]:
        category = meta.get("category", "")
        description = meta.get("description", "")
        file_path = meta.get("file_path", "")

        line = f"- **{name}**"
        if category:
            line += f" [{category}]"
        if description:
            line += f"\n  {description}"
        if file_path:
            line += f"\n  (in `{file_path}`)"

        lines.append(line)

    if len(rules) > 20:
        lines.append(f"... and {len(rules) - 20} more rules")

    return "\n".join(lines)


def _format_symbols(symbols: list[tuple[str, dict]]) -> str:
    """Format symbols for LLM context."""
    if not symbols:
        return "*No symbols found*"

    lines = []
    for name, meta in symbols[:30]:
        kind = meta.get("kind", "symbol")
        file_path = meta.get("file_path", "")

        line = f"- `{name}` ({kind})"
        if file_path:
            line += f" in `{file_path}`"

        lines.append(line)

    if len(symbols) > 30:
        lines.append(f"... and {len(symbols) - 30} more symbols")

    return "\n".join(lines)


def _format_files(files: list[tuple[str, dict]]) -> str:
    """Format files for LLM context - grouped by directory."""
    if not files:
        return "*No files found*"

    # Group by top-level directory
    dirs: dict[str, list[str]] = {}
    for name, _ in files:
        parts = name.split("/")
        if len(parts) > 1:
            top_dir = parts[0]
            dirs.setdefault(top_dir, []).append(name)
        else:
            dirs.setdefault("(root)", []).append(name)

    lines = []
    for dir_name, dir_files in sorted(dirs.items())[:10]:
        lines.append(f"- `{dir_name}/` ({len(dir_files)} files)")

    if len(dirs) > 10:
        lines.append(f"... and {len(dirs) - 10} more directories")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LLM-based section generators
# -----------------------------------------------------------------------------


async def _generate_context_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Context section using LLM synthesis."""
    # Gather data
    endpoints = await _get_api_endpoints(session, collection_id)
    graphql_types = await _get_graphql_types(session, collection_id)
    rpc_services = await _get_rpc_services(session, collection_id)
    readme = await _get_readme_content(session, collection_id)

    evidence_count = len(endpoints) + len(graphql_types) + len(rpc_services)

    # Build context for LLM
    context_parts = []

    if readme:
        context_parts.append(f"## README Excerpt\n{readme[:2000]}")

    context_parts.append(f"## API Endpoints ({len(endpoints)} total)")
    context_parts.append(_format_endpoints(endpoints))

    if graphql_types:
        context_parts.append(f"\n## GraphQL Types ({len(graphql_types)} total)")
        context_parts.append(_format_graphql_types(graphql_types))

    if rpc_services:
        context_parts.append(f"\n## RPC Services ({len(rpc_services)} total)")
        context_parts.append(_format_rpc_services(rpc_services))

    context = "\n".join(context_parts)

    try:
        result = await provider.generate_structured(
            system=CONTEXT_SECTION_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            if subsection.evidence_refs:
                content_lines.append("\n*Evidence:*")
                for ref in subsection.evidence_refs[:3]:
                    if ref.line_start:
                        content_lines.append(f"- `{ref.file_path}:{ref.line_start}`")
                    else:
                        content_lines.append(f"- `{ref.file_path}`")
            content_lines.append("")

        return Arc42Section(
            id="context",
            title="3. Context & Scope",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM context generation failed: %s", e)
        # Fallback to basic
        return await _generate_context_section_basic(session, collection_id)


async def _generate_building_blocks_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Building Blocks section using LLM synthesis."""
    # Gather data
    files = await _get_files(session, collection_id)
    symbols = await _get_symbols(session, collection_id)
    tables = await _get_db_tables(session, collection_id)

    evidence_count = len(files) + len(symbols) + len(tables)

    # Build context for LLM
    context_parts = [
        f"## File Structure ({len(files)} files)",
        _format_files(files),
        f"\n## Key Symbols ({len(symbols)} total)",
        _format_symbols(symbols),
    ]

    if tables:
        context_parts.append(f"\n## Database Tables ({len(tables)} total)")
        context_parts.append(_format_tables(tables))

    context = "\n".join(context_parts)

    try:
        result = await provider.generate_structured(
            system=BUILDING_BLOCKS_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            if subsection.evidence_refs:
                content_lines.append("\n*Evidence:*")
                for ref in subsection.evidence_refs[:3]:
                    if ref.line_start:
                        content_lines.append(f"- `{ref.file_path}:{ref.line_start}`")
                    else:
                        content_lines.append(f"- `{ref.file_path}`")
            content_lines.append("")

        return Arc42Section(
            id="building-blocks",
            title="5. Building Blocks",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM building blocks generation failed: %s", e)
        return await _generate_building_blocks_section_basic(session, collection_id)


async def _generate_runtime_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Runtime View section using LLM synthesis."""
    # Gather data
    endpoints = await _get_api_endpoints(session, collection_id)
    jobs = await _get_jobs(session, collection_id)
    edge_count = await _get_edge_count(session, collection_id)

    evidence_count = len(endpoints) + len(jobs) + (1 if edge_count else 0)

    # Build context for LLM
    context_parts = [
        f"## API Endpoints ({len(endpoints)} entry points)",
        _format_endpoints(endpoints),
    ]

    if jobs:
        context_parts.append(f"\n## Jobs and Background Tasks ({len(jobs)} total)")
        context_parts.append(_format_jobs(jobs))

    context_parts.append("\n## Graph Connectivity")
    context_parts.append(
        f"The knowledge graph contains {edge_count} relationships between entities."
    )

    context = "\n".join(context_parts)

    try:
        result = await provider.generate_structured(
            system=RUNTIME_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            if subsection.evidence_refs:
                content_lines.append("\n*Evidence:*")
                for ref in subsection.evidence_refs[:3]:
                    if ref.line_start:
                        content_lines.append(f"- `{ref.file_path}:{ref.line_start}`")
                    else:
                        content_lines.append(f"- `{ref.file_path}`")
            content_lines.append("")

        return Arc42Section(
            id="runtime",
            title="6. Runtime View",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM runtime generation failed: %s", e)
        return await _generate_runtime_section_basic(session, collection_id)


async def _generate_deployment_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Deployment View section using LLM synthesis."""
    # Gather data
    jobs = await _get_jobs(session, collection_id)

    evidence_count = len(jobs)

    # Build context for LLM
    context_parts = [
        f"## Jobs and Workflows ({len(jobs)} total)",
        _format_jobs(jobs) if jobs else "*No jobs or workflows detected*",
    ]

    context = "\n".join(context_parts)

    try:
        result = await provider.generate_structured(
            system=DEPLOYMENT_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            if subsection.evidence_refs:
                content_lines.append("\n*Evidence:*")
                for ref in subsection.evidence_refs[:3]:
                    if ref.line_start:
                        content_lines.append(f"- `{ref.file_path}:{ref.line_start}`")
                    else:
                        content_lines.append(f"- `{ref.file_path}`")
            content_lines.append("")

        return Arc42Section(
            id="deployment",
            title="7. Deployment View",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM deployment generation failed: %s", e)
        return await _generate_deployment_section_basic(session, collection_id)


async def _generate_crosscutting_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Crosscutting Concepts section using LLM synthesis."""
    # Gather data
    rules = await _get_business_rules(session, collection_id)
    candidates = await _get_rule_candidates(session, collection_id)

    evidence_count = len(rules) + len(candidates)

    # Build context for LLM
    context_parts = []

    if rules:
        context_parts.append(f"## Business Rules ({len(rules)} total)")
        context_parts.append(_format_rules(rules))
    else:
        context_parts.append("## Business Rules\n*No labeled business rules found*")

    if candidates:
        context_parts.append(f"\n## Rule Candidates / Error Handlers ({len(candidates)} total)")
        context_parts.append(_format_rules(candidates))

    context = "\n".join(context_parts)

    try:
        result = await provider.generate_structured(
            system=CROSSCUTTING_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            if subsection.evidence_refs:
                content_lines.append("\n*Evidence:*")
                for ref in subsection.evidence_refs[:3]:
                    if ref.line_start:
                        content_lines.append(f"- `{ref.file_path}:{ref.line_start}`")
                    else:
                        content_lines.append(f"- `{ref.file_path}`")
            content_lines.append("")

        return Arc42Section(
            id="crosscutting",
            title="8. Crosscutting Concepts",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM crosscutting generation failed: %s", e)
        return await _generate_crosscutting_section_basic(session, collection_id)


async def _generate_risks_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Risks section using LLM synthesis."""
    # Gather data
    rules = await _get_business_rules(session, collection_id)
    candidates = await _get_rule_candidates(session, collection_id)

    unreviewed = len(candidates) - len(rules) if len(candidates) > len(rules) else 0
    evidence_count = unreviewed if unreviewed > 0 else 0

    # Build context for LLM
    context_parts = [
        "## Code Analysis Summary",
        f"- Business rules identified and labeled: {len(rules)}",
        f"- Rule candidates (unlabeled): {len(candidates)}",
        f"- Unreviewed candidates: {unreviewed}",
    ]

    if candidates:
        context_parts.append("\n## Sample Unlabeled Candidates")
        context_parts.append(_format_rules(candidates[:10]))

    context = "\n".join(context_parts)

    try:
        result = await provider.generate_structured(
            system=RISKS_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            if subsection.evidence_refs:
                content_lines.append("\n*Evidence:*")
                for ref in subsection.evidence_refs[:3]:
                    if ref.line_start:
                        content_lines.append(f"- `{ref.file_path}:{ref.line_start}`")
                    else:
                        content_lines.append(f"- `{ref.file_path}`")
            content_lines.append("")

        return Arc42Section(
            id="risks",
            title="11. Risks & Technical Debt",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM risks generation failed: %s", e)
        return await _generate_risks_section_basic(session, collection_id)


async def _generate_glossary_section_llm(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider,
) -> Arc42Section:
    """Generate Glossary section using LLM synthesis."""
    # Gather data
    tables = await _get_db_tables(session, collection_id)
    rules = await _get_business_rules(session, collection_id)

    evidence_count = len(tables) + len(rules)

    # Build context for LLM
    context_parts = []

    if tables:
        context_parts.append(f"## Database Tables ({len(tables)} terms)")
        for name, _ in tables[:30]:
            context_parts.append(f"- {name}")

    if rules:
        context_parts.append(f"\n## Business Rule Names ({len(rules)} terms)")
        for name, _ in rules[:20]:
            context_parts.append(f"- {name}")

    context = "\n".join(context_parts) if context_parts else "No domain terms found."

    try:
        result = await provider.generate_structured(
            system=GLOSSARY_PROMPT,
            messages=[{"role": "user", "content": context}],
            output_schema=Arc42SectionOutput,
            temperature=0.0,
        )

        # Convert to markdown content
        content_lines = [result.narrative, ""]
        for subsection in result.subsections:
            content_lines.append(f"### {subsection.title}\n")
            content_lines.append(subsection.content)
            content_lines.append("")

        return Arc42Section(
            id="glossary",
            title="12. Glossary",
            content="\n".join(content_lines),
            evidence_count=evidence_count,
            confidence=result.confidence,
            needs_human_input=result.needs_human_input,
        )

    except Exception as e:
        logger.warning("LLM glossary generation failed: %s", e)
        return await _generate_glossary_section_basic(session, collection_id)


# -----------------------------------------------------------------------------
# Placeholder sections (require human input)
# -----------------------------------------------------------------------------


def _generate_placeholder_section(section_id: str) -> Arc42Section:
    """Generate a placeholder section that requires human input."""
    sections = {
        "introduction": {
            "id": "introduction",
            "title": "1. Introduction & Goals",
            "questions": [
                "What is the primary business problem this system solves?",
                "Who are the main stakeholders and what are their expectations?",
                "What are the top 3-5 quality goals (e.g., performance, security)?",
            ],
        },
        "constraints": {
            "id": "constraints",
            "title": "2. Constraints",
            "questions": [
                "What technical constraints exist (legacy systems, mandated technologies)?",
                "What organizational constraints apply (budget, timeline, team size)?",
                "Are there regulatory or compliance requirements?",
            ],
        },
        "solution-strategy": {
            "id": "solution-strategy",
            "title": "4. Solution Strategy",
            "questions": [
                "What were the key technology choices and why?",
                "What architectural patterns were adopted (microservices, monolith, etc.)?",
                "How does the architecture address the quality goals?",
            ],
        },
        "decisions": {
            "id": "decisions",
            "title": "9. Architecture Decisions",
            "questions": [
                "What were the key architectural decisions and why were they made?",
                "What alternatives were considered and rejected?",
                "Are there ADR (Architecture Decision Record) files in the repository?",
            ],
        },
        "quality": {
            "id": "quality",
            "title": "10. Quality Requirements",
            "questions": [
                "What are the performance requirements (latency, throughput)?",
                "What are the availability/reliability requirements?",
                "What security requirements must be met?",
            ],
        },
    }

    config = sections.get(
        section_id,
        {
            "id": section_id,
            "title": section_id.replace("-", " ").title(),
            "questions": ["This section requires human input."],
        },
    )

    content = "This section requires human input to complete accurately.\n\n"
    content += "The system has extracted technical artifacts, but this section needs context "
    content += "that can only come from stakeholders, documentation, or design decisions.\n"

    return Arc42Section(
        id=config["id"],
        title=config["title"],
        content=content,
        evidence_count=0,
        confidence=0.0,
        needs_human_input=config["questions"],
    )


# -----------------------------------------------------------------------------
# Basic (non-LLM) section generators - fallback implementations
# -----------------------------------------------------------------------------


async def _generate_context_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Context section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### System Boundary\n"]

    # Count API endpoints as external interface
    endpoint_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
        )
    )

    # Count GraphQL types
    graphql_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.GRAPHQL_TYPE,
        )
    )

    # Count RPC services
    rpc_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SERVICE_RPC,
        )
    )

    evidence_count = 0

    if endpoint_count:
        lines.append(f"- **REST API**: {endpoint_count} endpoints exposed")
        evidence_count += endpoint_count

    if graphql_count:
        lines.append(f"- **GraphQL**: {graphql_count} types defined")
        evidence_count += graphql_count

    if rpc_count:
        lines.append(f"- **RPC Services**: {rpc_count} service methods")
        evidence_count += rpc_count

    if not evidence_count:
        lines.append(
            "*No external interfaces detected. This section may require manual enrichment.*"
        )

    lines.append("\n### External Dependencies\n")
    lines.append("*Inferred: Dependencies are detected from imports and package manifests.*")

    return Arc42Section(
        id="context",
        title="3. Context & Scope",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_building_blocks_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Building Blocks section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Component Overview\n"]

    file_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.FILE,
        )
    )

    symbol_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
        )
    )

    table_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
        )
    )

    evidence_count = 0
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")

    if file_count:
        lines.append(f"| Source Files | {file_count} |")
        evidence_count += file_count

    if symbol_count:
        lines.append(f"| Symbols (functions/classes) | {symbol_count} |")
        evidence_count += symbol_count

    if table_count:
        lines.append(f"| Database Tables | {table_count} |")
        evidence_count += table_count

    if not evidence_count:
        lines.append("| No components detected | - |")
        lines.append(
            "\n*No building blocks detected. Index the codebase to populate this section.*"
        )

    if table_count:
        lines.append("\n### Database Schema\n")
        result = await session.execute(
            select(KnowledgeNode.name)
            .where(
                KnowledgeNode.collection_id == collection_id,
                KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
            )
            .limit(10)
        )
        tables = result.scalars().all()
        for table in tables:
            lines.append(f"- `{table}`")
        if table_count > 10:
            lines.append(f"- *... and {table_count - 10} more*")

    return Arc42Section(
        id="building-blocks",
        title="5. Building Blocks",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_runtime_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Runtime View section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeEdge, KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Key Execution Flows\n"]

    edge_count = await session.scalar(
        select(func.count(KnowledgeEdge.id)).where(
            KnowledgeEdge.collection_id == collection_id,
        )
    )

    evidence_count = 0

    if edge_count:
        lines.append(f"*{edge_count} relationships detected in the knowledge graph.*\n")
        evidence_count += edge_count

        result = await session.execute(
            select(KnowledgeNode.name, KnowledgeNode.meta)
            .where(
                KnowledgeNode.collection_id == collection_id,
                KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
            )
            .limit(5)
        )
        endpoints = result.all()

        if endpoints:
            lines.append("### Entry Points\n")
            for name, meta in endpoints:
                meta = meta or {}
                method = meta.get("method", "GET")
                lines.append(f"- `{method} {name}`")
            lines.append("\n*Use `trace_path` to explore execution flows from these entry points.*")
    else:
        lines.append(
            "*No execution flows detected. Graph relationships will be populated during indexing.*"
        )

    return Arc42Section(
        id="runtime",
        title="6. Runtime View",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_deployment_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Deployment View section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    lines = ["### Deployment Infrastructure\n"]
    evidence_count = 0

    result = await session.execute(
        select(KnowledgeNode.name, KnowledgeNode.meta)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.JOB,
        )
        .limit(20)
    )
    jobs = result.all()

    if jobs:
        lines.append("### Jobs and Workflows\n")
        for name, meta in jobs:
            meta = meta or {}
            job_type = meta.get("framework", meta.get("job_type", "unknown"))
            schedule = meta.get("schedule", "")
            lines.append(f"- **{name}** ({job_type})")
            if schedule:
                lines.append(f"  - Schedule: `{schedule}`")
        evidence_count += len(jobs)
    else:
        lines.append("*No deployment manifests or job definitions detected.*")
        lines.append(
            "*Inferred: Check for Kubernetes manifests, Docker Compose, or CI/CD configs.*"
        )

    return Arc42Section(
        id="deployment",
        title="7. Deployment View",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_crosscutting_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Crosscutting Concepts section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Detected Patterns\n"]
    evidence_count = 0

    rule_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
        )
    )

    if rule_count:
        lines.append("#### Validation & Business Rules\n")
        lines.append(f"*{rule_count} business rules detected and labeled.*\n")
        lines.append("Use `list_business_rules` to explore validation patterns.\n")
        evidence_count += rule_count

    candidate_count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.RULE_CANDIDATE,
        )
    )

    if candidate_count:
        lines.append("#### Error Handling\n")
        lines.append(f"*{candidate_count} conditional error handlers detected.*\n")
        evidence_count += candidate_count

    if not evidence_count:
        lines.append("*No specific patterns detected. Manual analysis recommended.*")

    lines.append("\n### Security Considerations\n")
    lines.append("*Inferred: Security patterns should be manually reviewed.*")
    lines.append("- Authentication: Check API endpoints for auth middleware")
    lines.append("- Authorization: Review business rules for permission checks")
    lines.append("- Input validation: Validate rule candidates for injection prevention")

    return Arc42Section(
        id="crosscutting",
        title="8. Crosscutting Concepts",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_risks_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Risks & Technical Debt section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Technical Debt Indicators\n"]
    evidence_count = 0

    unlabeled_candidates = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.RULE_CANDIDATE,
        )
    )

    labeled_rules = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
        )
    )

    if unlabeled_candidates:
        unreviewed = unlabeled_candidates - (labeled_rules or 0)
        if unreviewed > 0:
            lines.append(f"- **{unreviewed} unreviewed rule candidates**")
            lines.append("  - These may contain undocumented business rules or tech debt")
            evidence_count += unreviewed

    lines.append("\n### Recommended Review\n")
    lines.append("*The following require manual review:*")
    lines.append("- TODO/FIXME comments in codebase")
    lines.append("- Dependency vulnerabilities (check package manifests)")
    lines.append("- Outdated documentation")

    if not evidence_count:
        lines.append("\n*No automated debt indicators detected.*")

    return Arc42Section(
        id="risks",
        title="11. Risks & Technical Debt",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_glossary_section_basic(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Glossary section - basic fallback without LLM."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    lines = ["### Domain Terms\n"]
    evidence_count = 0

    result = await session.execute(
        select(KnowledgeNode.name)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
        )
        .limit(20)
    )
    tables = result.scalars().all()

    if tables:
        lines.append("*Terms derived from database schema:*\n")
        for table in tables:
            term = table.replace("_", " ").title()
            lines.append(f"- **{term}**: Database entity `{table}`")
        evidence_count += len(tables)
    else:
        lines.append("*No domain terms detected. Consider adding a custom glossary.*")

    return Arc42Section(
        id="glossary",
        title="12. Glossary",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


# -----------------------------------------------------------------------------
# Main Generator
# -----------------------------------------------------------------------------


async def generate_arc42(
    session: AsyncSession,
    collection_id: UUID,
    provider: LLMProvider | None = None,
) -> Arc42Document:
    """Generate arc42 architecture document from extracted facts.

    If an LLM provider is given, uses LLM synthesis for richer documentation.
    Otherwise, falls back to basic aggregation.

    Args:
        session: Database session
        collection_id: Collection to generate documentation for
        provider: Optional LLM provider for synthesis

    Returns:
        Arc42Document with all sections
    """
    doc = Arc42Document(collection_id=collection_id)

    if provider:
        # LLM-powered generation with rich synthesis
        logger.info("Generating arc42 with LLM synthesis for collection %s", collection_id)

        doc.sections = [
            # Placeholder sections requiring human input
            _generate_placeholder_section("introduction"),
            _generate_placeholder_section("constraints"),
            # LLM-generated sections
            await _generate_context_section_llm(session, collection_id, provider),
            _generate_placeholder_section("solution-strategy"),
            await _generate_building_blocks_section_llm(session, collection_id, provider),
            await _generate_runtime_section_llm(session, collection_id, provider),
            await _generate_deployment_section_llm(session, collection_id, provider),
            await _generate_crosscutting_section_llm(session, collection_id, provider),
            _generate_placeholder_section("decisions"),
            _generate_placeholder_section("quality"),
            await _generate_risks_section_llm(session, collection_id, provider),
            await _generate_glossary_section_llm(session, collection_id, provider),
        ]
    else:
        # Basic fallback - just counts and lists
        logger.info("Generating basic arc42 (no LLM) for collection %s", collection_id)

        doc.sections = [
            _generate_placeholder_section("introduction"),
            _generate_placeholder_section("constraints"),
            await _generate_context_section_basic(session, collection_id),
            _generate_placeholder_section("solution-strategy"),
            await _generate_building_blocks_section_basic(session, collection_id),
            await _generate_runtime_section_basic(session, collection_id),
            await _generate_deployment_section_basic(session, collection_id),
            await _generate_crosscutting_section_basic(session, collection_id),
            _generate_placeholder_section("decisions"),
            _generate_placeholder_section("quality"),
            await _generate_risks_section_basic(session, collection_id),
            await _generate_glossary_section_basic(session, collection_id),
        ]

    return doc


# -----------------------------------------------------------------------------
# Artifact Storage
# -----------------------------------------------------------------------------


async def save_arc42_artifact(
    session: AsyncSession,
    collection_id: UUID,
    document: Arc42Document,
) -> UUID:
    """Save arc42 document as a KnowledgeArtifact."""
    from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = pg_insert(KnowledgeArtifact).values(
        collection_id=collection_id,
        kind=KnowledgeArtifactKind.ARC42,
        name="arc42 Architecture Documentation",
        content=document.to_markdown(),
        meta={
            "generated_at": document.generated_at.isoformat(),
            "section_count": len(document.sections),
            "sections": [s.id for s in document.sections],
            "has_llm_synthesis": any(s.confidence > 0 for s in document.sections),
        },
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_knowledge_artifact_name",
        set_={
            "content": stmt.excluded.content,
            "meta": stmt.excluded.meta,
        },
    ).returning(KnowledgeArtifact.id)

    result = await session.execute(stmt)
    return result.scalar_one()


# -----------------------------------------------------------------------------
# Drift Detection
# -----------------------------------------------------------------------------


async def compute_drift_report(
    session: AsyncSession,
    collection_id: UUID,
    stored_artifact_id: UUID | None = None,
) -> DriftReport:
    """Compare stored arc42 artifact with current extracted facts.

    If no stored artifact, returns empty report.
    """
    from contextmine_core.models import (
        KnowledgeArtifact,
        KnowledgeArtifactKind,
        KnowledgeNode,
        KnowledgeNodeKind,
    )
    from sqlalchemy import select

    report = DriftReport()

    # Get stored artifact
    if stored_artifact_id:
        result = await session.execute(
            select(KnowledgeArtifact).where(KnowledgeArtifact.id == stored_artifact_id)
        )
        stored = result.scalar_one_or_none()
    else:
        result = await session.execute(
            select(KnowledgeArtifact)
            .where(
                KnowledgeArtifact.collection_id == collection_id,
                KnowledgeArtifact.kind == KnowledgeArtifactKind.ARC42,
            )
            .order_by(KnowledgeArtifact.created_at.desc())
            .limit(1)
        )
        stored = result.scalar_one_or_none()

    if not stored:
        return report  # No baseline to compare

    stored_content = stored.content or ""

    # Simple drift detection: check if key counts changed
    # This is a simplified version - could be enhanced with semantic diff

    # Get current counts
    endpoint_count = await session.scalar(
        select(KnowledgeNode.id)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
        )
        .limit(1)
    )

    table_count = await session.scalar(
        select(KnowledgeNode.id)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
        )
        .limit(1)
    )

    job_count = await session.scalar(
        select(KnowledgeNode.id)
        .where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.JOB,
        )
        .limit(1)
    )

    # Check for missing mentions in stored content
    if endpoint_count and "endpoint" not in stored_content.lower():
        report.items.append(
            DriftItem(
                kind="added",
                category="API",
                name="REST Endpoints",
                details="New API endpoints detected since last generation",
            )
        )

    if table_count and "database" not in stored_content.lower():
        report.items.append(
            DriftItem(
                kind="added",
                category="Database",
                name="Database Tables",
                details="New database tables detected since last generation",
            )
        )

    if job_count and "job" not in stored_content.lower():
        report.items.append(
            DriftItem(
                kind="added",
                category="Deployment",
                name="Jobs/Workflows",
                details="New jobs or workflows detected since last generation",
            )
        )

    return report
