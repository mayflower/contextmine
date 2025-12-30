"""arc42 Architecture Twin generator.

Generates arc42-style architecture documentation from extracted knowledge graph facts.
Every statement is evidence-backed or explicitly marked as inferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class Arc42Section:
    """A single arc42 section."""

    id: str  # e.g., "context", "building-blocks"
    title: str
    content: str
    evidence_count: int = 0


@dataclass
class Arc42Document:
    """Complete arc42 architecture document."""

    sections: list[Arc42Section] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    collection_id: UUID | None = None

    def to_markdown(self) -> str:
        """Render as full markdown document."""
        lines = ["# Architecture Documentation (arc42)\n"]
        lines.append(f"*Generated: {self.generated_at.isoformat()}*\n")
        lines.append("---\n")

        for section in self.sections:
            lines.append(f"## {section.title}\n")
            lines.append(section.content)
            lines.append("")
            if section.evidence_count > 0:
                lines.append(f"*Based on {section.evidence_count} extracted facts*\n")
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


async def generate_arc42(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Document:
    """Generate arc42 architecture document from extracted facts.

    Produces sections:
    1. Context - System boundary and external dependencies
    2. Building Blocks - Components and their relationships
    3. Runtime View - Key execution flows
    4. Deployment View - Infrastructure and deployment
    5. Crosscutting Concepts - Security, observability patterns
    6. Risks & Technical Debt - TODOs, FIXMEs, hotspots
    7. Glossary - Domain terms
    """
    doc = Arc42Document(collection_id=collection_id)

    # Generate each section
    doc.sections.append(await _generate_context_section(session, collection_id))
    doc.sections.append(await _generate_building_blocks_section(session, collection_id))
    doc.sections.append(await _generate_runtime_section(session, collection_id))
    doc.sections.append(await _generate_deployment_section(session, collection_id))
    doc.sections.append(await _generate_crosscutting_section(session, collection_id))
    doc.sections.append(await _generate_risks_section(session, collection_id))
    doc.sections.append(await _generate_glossary_section(session, collection_id))

    return doc


async def _generate_context_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Context section - system boundary and external interfaces."""
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
        title="1. Context",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_building_blocks_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Building Blocks section - components and relationships."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Component Overview\n"]

    # Count files by type/role
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

    # Add top tables
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
        title="2. Building Blocks",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_runtime_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Runtime View section - key execution flows."""
    from contextmine_core.models import KnowledgeEdge, KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Key Execution Flows\n"]

    # Count edges to understand connectivity
    edge_count = await session.scalar(
        select(func.count(KnowledgeEdge.id)).where(
            KnowledgeEdge.collection_id == collection_id,
        )
    )

    evidence_count = 0

    if edge_count:
        lines.append(f"*{edge_count} relationships detected in the knowledge graph.*\n")
        evidence_count += edge_count

        # Get sample entry points (API endpoints)
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
        title="3. Runtime View",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_deployment_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Deployment View section - infrastructure and deployment config."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    lines = ["### Deployment Infrastructure\n"]
    evidence_count = 0

    # Get jobs (K8s, GitHub Actions, Prefect)
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
            job_type = meta.get("job_type", "unknown")
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
        title="4. Deployment View",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_crosscutting_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Crosscutting Concepts section - patterns and practices."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Detected Patterns\n"]
    evidence_count = 0

    # Business rules suggest validation patterns
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

    # Rule candidates suggest defensive coding
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
        title="5. Crosscutting Concepts",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_risks_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Risks & Technical Debt section."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import func, select

    lines = ["### Technical Debt Indicators\n"]
    evidence_count = 0

    # Count unlabeled rule candidates as potential debt
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

    # Note about TODO/FIXME detection
    lines.append("\n### Recommended Review\n")
    lines.append("*The following require manual review:*")
    lines.append("- TODO/FIXME comments in codebase")
    lines.append("- Dependency vulnerabilities (check package manifests)")
    lines.append("- Outdated documentation")

    if not evidence_count:
        lines.append("\n*No automated debt indicators detected.*")

    return Arc42Section(
        id="risks",
        title="6. Risks & Technical Debt",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


async def _generate_glossary_section(
    session: AsyncSession,
    collection_id: UUID,
) -> Arc42Section:
    """Generate Glossary section - domain terms."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
    from sqlalchemy import select

    lines = ["### Domain Terms\n"]
    evidence_count = 0

    # Extract terms from table names and business rules
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
            # Convert snake_case to human readable
            term = table.replace("_", " ").title()
            lines.append(f"- **{term}**: Database entity `{table}`")
        evidence_count += len(tables)
    else:
        lines.append("*No domain terms detected. Consider adding a custom glossary.*")

    return Arc42Section(
        id="glossary",
        title="7. Glossary",
        content="\n".join(lines),
        evidence_count=evidence_count,
    )


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
