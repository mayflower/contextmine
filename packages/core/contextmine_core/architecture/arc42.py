"""arc42 document generation from extracted architecture facts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .schemas import Arc42Document, ArchitectureFact, ArchitectureFactsBundle, summarize_confidence

SECTION_TITLES: dict[str, str] = {
    "1_introduction_and_goals": "1. Introduction and Goals",
    "2_constraints": "2. Architecture Constraints",
    "3_system_scope_and_context": "3. System Scope and Context",
    "4_solution_strategy": "4. Solution Strategy",
    "5_building_block_view": "5. Building Block View",
    "6_runtime_view": "6. Runtime View",
    "7_deployment_view": "7. Deployment View",
    "8_crosscutting_concepts": "8. Crosscutting Concepts",
    "9_architecture_decisions": "9. Architecture Decisions",
    "10_quality_requirements": "10. Quality Requirements",
    "11_risks_and_technical_debt": "11. Risks and Technical Debt",
    "12_glossary": "12. Glossary",
}

_SECTION_ALIASES: dict[str, str] = {
    "1": "1_introduction_and_goals",
    "introduction": "1_introduction_and_goals",
    "goals": "1_introduction_and_goals",
    "2": "2_constraints",
    "constraints": "2_constraints",
    "3": "3_system_scope_and_context",
    "context": "3_system_scope_and_context",
    "scope": "3_system_scope_and_context",
    "4": "4_solution_strategy",
    "strategy": "4_solution_strategy",
    "5": "5_building_block_view",
    "building_block": "5_building_block_view",
    "building_blocks": "5_building_block_view",
    "6": "6_runtime_view",
    "runtime": "6_runtime_view",
    "7": "7_deployment_view",
    "deployment": "7_deployment_view",
    "8": "8_crosscutting_concepts",
    "crosscutting": "8_crosscutting_concepts",
    "9": "9_architecture_decisions",
    "decisions": "9_architecture_decisions",
    "10": "10_quality_requirements",
    "quality": "10_quality_requirements",
    "11": "11_risks_and_technical_debt",
    "risks": "11_risks_and_technical_debt",
    "risk": "11_risks_and_technical_debt",
    "12": "12_glossary",
    "glossary": "12_glossary",
}


def normalize_arc42_section_key(section: str | None) -> str | None:
    """Normalize user-facing arc42 section names into canonical section keys."""

    if not section:
        return None
    normalized = section.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in SECTION_TITLES:
        return normalized
    return _SECTION_ALIASES.get(normalized)


def _facts_by_type(bundle: ArchitectureFactsBundle, fact_type: str) -> list[ArchitectureFact]:
    return [fact for fact in bundle.facts if fact.fact_type == fact_type]


def _facts_with_tag(bundle: ArchitectureFactsBundle, tag: str) -> list[ArchitectureFact]:
    return [fact for fact in bundle.facts if tag in fact.tags]


def _render_system_context(bundle: ArchitectureFactsBundle) -> str:
    inbound = [fact for fact in bundle.ports_adapters if fact.direction == "inbound"]
    outbound = [fact for fact in bundle.ports_adapters if fact.direction == "outbound"]

    lines = [
        f"System scenario: `{bundle.scenario_name}`.",
        f"Inbound integration points: {len(inbound)}.",
        f"Outbound integration points: {len(outbound)}.",
    ]

    context_facts = _facts_by_type(bundle, "c4_context")
    if context_facts:
        warnings = context_facts[0].attributes.get("warnings") or []
        if warnings:
            lines.append("Context rendering warnings: " + "; ".join(str(item) for item in warnings))
    return "\n".join(lines)


def _render_building_blocks(bundle: ArchitectureFactsBundle) -> str:
    containers = _facts_by_type(bundle, "container")
    components = _facts_by_type(bundle, "component")
    tables = _facts_by_type(bundle, "erm_table")

    lines = [
        f"Containers: {len(containers)}.",
        f"Components: {len(components)}.",
        f"Data entities (ERM tables): {len(tables)}.",
    ]

    top_containers = sorted(
        (
            (
                str(fact.attributes.get("container") or fact.title),
                int(fact.attributes.get("member_count") or 0),
            )
            for fact in containers
        ),
        key=lambda row: (row[1], row[0]),
        reverse=True,
    )
    if top_containers:
        rendered = ", ".join(f"{name} ({count})" for name, count in top_containers[:5])
        lines.append("Largest containers by member_count: " + rendered)

    return "\n".join(lines)


def _render_runtime_view(bundle: ArchitectureFactsBundle) -> str:
    deps = _facts_by_type(bundle, "component_dependency")
    outbound = [fact for fact in bundle.ports_adapters if fact.direction == "outbound"]

    lines = [
        f"Observed component dependencies: {len(deps)}.",
        f"Outbound adapters: {len(outbound)}.",
    ]

    if outbound:
        sample = ", ".join(
            sorted(
                {
                    f"{fact.component or fact.adapter_name or 'adapter'} -> {fact.port_name}"
                    for fact in outbound[:8]
                }
            )
        )
        lines.append("Sample runtime interactions: " + sample)

    return "\n".join(lines)


def _render_deployment(bundle: ArchitectureFactsBundle) -> str:
    deployment_facts = _facts_by_type(bundle, "c4_deployment")
    if not deployment_facts:
        return "No deployment-specific facts available."

    warnings = deployment_facts[0].attributes.get("warnings") or []
    lines = ["Deployment view generated from C4 deployment projection."]
    if warnings:
        lines.append("Deployment warnings: " + "; ".join(str(item) for item in warnings))
    return "\n".join(lines)


def _render_quality(bundle: ArchitectureFactsBundle) -> str:
    quality = _facts_by_type(bundle, "quality_summary")
    if not quality:
        return "No quality summary metrics available."

    attrs = quality[0].attributes
    return "\n".join(
        [
            f"Metric nodes: {attrs.get('metric_nodes', 0)}.",
            f"Average test coverage: {attrs.get('coverage_avg')}.",
            f"Average complexity: {attrs.get('complexity_avg')}.",
            f"Average coupling: {attrs.get('coupling_avg')}.",
            f"Average change frequency: {attrs.get('change_frequency_avg')}.",
        ]
    )


def _render_risks(bundle: ArchitectureFactsBundle) -> str:
    lines: list[str] = []
    if bundle.warnings:
        lines.append("Warnings from extraction pipeline:")
        lines.extend(f"- {warning}" for warning in bundle.warnings[:10])

    unresolved_ports = [
        fact
        for fact in bundle.ports_adapters
        if not fact.container or not fact.component or fact.source == "llm"
    ]
    if unresolved_ports:
        lines.append(
            "Potential architecture drift risk: "
            f"{len(unresolved_ports)} ports/adapters still have incomplete ownership mapping."
        )

    if not lines:
        lines.append("No critical architecture risks detected by deterministic checks.")

    return "\n".join(lines)


def _render_glossary(bundle: ArchitectureFactsBundle) -> str:
    terms: dict[str, str] = {}

    for fact in _facts_by_type(bundle, "container"):
        container = str(fact.attributes.get("container") or fact.title)
        terms.setdefault(container, "C4 container extracted from architecture projection")

    for fact in bundle.ports_adapters:
        terms.setdefault(fact.port_name, f"{fact.direction} port")

    if not terms:
        return "No glossary terms available."

    lines = [f"- **{term}**: {description}" for term, description in sorted(terms.items())[:40]]
    return "\n".join(lines)


def _section_content(bundle: ArchitectureFactsBundle) -> dict[str, str]:
    containers = _facts_by_type(bundle, "container")
    components = _facts_by_type(bundle, "component")
    rules = _facts_by_type(bundle, "business_rules")

    section_map = {
        "1_introduction_and_goals": (
            "This document describes architecture facts inferred from the digital twin, "
            "knowledge graph, surfaces, and metrics."
        ),
        "2_constraints": (
            "Constraints are derived from available repository evidence, extracted surfaces, and "
            "known data contracts."
        ),
        "3_system_scope_and_context": _render_system_context(bundle),
        "4_solution_strategy": (
            "Primary strategy: explicit container/component decomposition with ports/adapters "
            "classification and evidence-backed confidence scoring."
        ),
        "5_building_block_view": _render_building_blocks(bundle),
        "6_runtime_view": _render_runtime_view(bundle),
        "7_deployment_view": _render_deployment(bundle),
        "8_crosscutting_concepts": (
            f"Business rules extracted: {sum(int(f.attributes.get('count', 0)) for f in rules)}.\n"
            f"Containers tracked: {len(containers)}.\n"
            f"Components tracked: {len(components)}."
        ),
        "9_architecture_decisions": (
            "V1 uses advisory governance for drift and confidence. CI blocking is intentionally "
            "disabled for architecture deltas."
        ),
        "10_quality_requirements": _render_quality(bundle),
        "11_risks_and_technical_debt": _render_risks(bundle),
        "12_glossary": _render_glossary(bundle),
    }
    return section_map


def generate_arc42_from_facts(
    facts: ArchitectureFactsBundle,
    scenario: Any,
    options: dict[str, Any] | None = None,
) -> Arc42Document:
    """Generate an arc42 document from architecture facts."""

    options = options or {}
    requested_section = normalize_arc42_section_key(options.get("section"))

    all_sections = _section_content(facts)
    selected_keys = [requested_section] if requested_section else list(SECTION_TITLES.keys())

    sections: dict[str, str] = {}
    for key in selected_keys:
        if key not in all_sections:
            continue
        sections[key] = all_sections[key]

    section_coverage = {
        key: bool((sections.get(key) or "").strip()) for key in SECTION_TITLES if key in sections
    }

    title = f"arc42 - {facts.scenario_name}"
    markdown_lines = [f"# {title}", ""]
    for key in selected_keys:
        if key not in sections:
            continue
        markdown_lines.append(f"## {SECTION_TITLES[key]}")
        markdown_lines.append(sections[key])
        markdown_lines.append("")

    confidence_summary = summarize_confidence(facts.facts, facts.ports_adapters)

    scenario_name = getattr(scenario, "name", facts.scenario_name)
    return Arc42Document(
        collection_id=facts.collection_id,
        scenario_id=facts.scenario_id,
        scenario_name=scenario_name,
        title=title,
        generated_at=datetime.now(UTC),
        sections=sections,
        markdown="\n".join(markdown_lines).strip() + "\n",
        warnings=list(facts.warnings),
        confidence_summary=confidence_summary,
        section_coverage=section_coverage,
    )
