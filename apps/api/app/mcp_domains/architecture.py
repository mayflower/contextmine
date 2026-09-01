"""Architecture intent and arc42 MCP tools."""

import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

from contextmine_core import (
    Collection,
    Source,
    SourceType,
    accessible_collections_clause,
    get_settings,
)
from contextmine_core import get_session as get_db_session
from contextmine_core.architecture import (
    ClaudeAgentSdkUnavailableError,
    generate_arc42_with_claude_sdk,
)
from contextmine_core.architecture_intents import IntentAction
from contextmine_core.model_policy import ModelCallsDisabledError
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from app.mcp_auth import get_current_user_id


class ArchitectureIntentResult(BaseModel):
    """Persisted architecture intent state."""

    intent_id: str
    scenario_id: str
    status: str
    scenario_version: int
    risk: str | None = None
    requires_approval: bool | None = None


class Arc42Result(BaseModel):
    """Bounded arc42 artifact response generated with the Claude Agent SDK."""

    collection_id: str
    scenario_id: str
    artifact_id: str | None = None
    section: str | None = None
    generated: bool
    cached: bool
    facts_hash: str | None = None
    warnings: list[str] = Field(default_factory=list)
    sections: dict[str, str] = Field(default_factory=dict)
    markdown: str = ""
    truncated: bool = False


class ArchitectureDataResult(BaseModel):
    """Structured architecture analysis result."""

    collection_id: str
    scenario_id: str
    data: dict[str, Any]
    truncated: bool = False


def _uuid(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ToolError(f"Invalid {field_name}: {value}") from exc


async def _resolve_collection(db, collection_id: str | None) -> Collection:
    user_id = get_current_user_id()
    query = select(Collection).where(accessible_collections_clause(user_id))
    if collection_id:
        query = query.where(Collection.id == _uuid(collection_id, "collection_id"))
    else:
        query = query.order_by(Collection.created_at.desc()).limit(1)
    collection = (await db.execute(query)).scalar_one_or_none()
    if collection is None:
        raise ToolError("Collection not found or access denied.")
    return collection


async def _require_owned_scenario(db, scenario_id: str):  # noqa: ANN202
    from contextmine_core.models import TwinScenario

    user_id = get_current_user_id()
    if user_id is None:
        raise ToolError("Authentication required.")
    scenario_uuid = _uuid(scenario_id, "scenario_id")
    scenario = (
        await db.execute(select(TwinScenario).where(TwinScenario.id == scenario_uuid))
    ).scalar_one_or_none()
    if scenario is None:
        raise ToolError("Scenario not found.")
    collection = (
        await db.execute(select(Collection).where(Collection.id == scenario.collection_id))
    ).scalar_one_or_none()
    if collection is None or collection.owner_user_id != user_id:
        raise ToolError("Only the collection owner can execute architecture intents.")
    return scenario, user_id


async def create_architecture_intent(
    scenario_id: Annotated[str, Field(description="Scenario UUID")],
    action: Annotated[IntentAction, Field(description="Architecture intent action")],
    target_type: Annotated[
        Literal["node", "edge", "context", "service"],
        Field(description="Intent target type"),
    ],
    target_id: Annotated[
        str, Field(description="Intent target identifier", min_length=1, max_length=2048)
    ],
    expected_scenario_version: Annotated[
        int, Field(description="Optimistic-lock scenario version", ge=1)
    ],
    params_json: Annotated[
        str | None, Field(description="Optional JSON action parameters", max_length=20000)
    ] = None,
) -> ArchitectureIntentResult:
    """Submit an architecture intent through the existing twin implementation."""
    from contextmine_core.architecture_intents import ArchitectureIntentV1
    from contextmine_core.twin import submit_intent

    try:
        params = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError as exc:
        raise ToolError("params_json must be a valid JSON object.") from exc
    if not isinstance(params, dict):
        raise ToolError("params_json must be a JSON object.")

    async with get_db_session() as db:
        scenario, user_id = await _require_owned_scenario(db, scenario_id)
        try:
            payload = ArchitectureIntentV1(
                intent_version="1.0",
                scenario_id=scenario.id,
                action=action,
                target={"type": target_type, "id": target_id},
                params=params,
                expected_scenario_version=expected_scenario_version,
                requested_by=user_id,
            )
            intent = await submit_intent(
                session=db,
                scenario=scenario,
                intent=payload,
                requested_by=user_id,
                auto_execute=True,
            )
            await db.commit()
        except ValidationError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise ToolError(f"Failed to create intent: {exc}") from exc
    return ArchitectureIntentResult(
        intent_id=str(intent.id),
        scenario_id=str(scenario.id),
        status=intent.status.value,
        risk=intent.risk_level.value,
        requires_approval=intent.requires_approval,
        scenario_version=scenario.version,
    )


async def approve_architecture_intent(
    scenario_id: Annotated[str, Field(description="Scenario UUID")],
    intent_id: Annotated[str, Field(description="Blocked intent UUID")],
) -> ArchitectureIntentResult:
    """Approve and execute an existing blocked architecture intent."""
    from contextmine_core.twin import approve_and_execute_intent

    intent_uuid = _uuid(intent_id, "intent_id")
    async with get_db_session() as db:
        scenario, _user_id = await _require_owned_scenario(db, scenario_id)
        try:
            intent = await approve_and_execute_intent(db, scenario, intent_uuid)
            await db.commit()
        except Exception as exc:
            raise ToolError(f"Failed to approve intent: {exc}") from exc
    return ArchitectureIntentResult(
        intent_id=str(intent.id),
        scenario_id=str(scenario.id),
        status=intent.status.value,
        scenario_version=scenario.version,
    )


async def _resolve_scenario(db, collection: Collection, scenario_id: str | None):  # noqa: ANN202
    from contextmine_core.models import TwinScenario
    from contextmine_core.twin import get_or_create_as_is_scenario

    if scenario_id:
        scenario = (
            await db.execute(
                select(TwinScenario).where(
                    TwinScenario.id == _uuid(scenario_id, "scenario_id"),
                    TwinScenario.collection_id == collection.id,
                )
            )
        ).scalar_one_or_none()
        if scenario is None:
            raise ToolError("Scenario not found in collection.")
        return scenario
    scenario = (
        await db.execute(
            select(TwinScenario)
            .where(
                TwinScenario.collection_id == collection.id,
                TwinScenario.is_as_is.is_(True),
            )
            .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if scenario is None:
        scenario = await get_or_create_as_is_scenario(
            db, collection.id, user_id=get_current_user_id()
        )
    return scenario


async def _resolve_repo_checkout(db, collection_id: uuid.UUID) -> Path:
    source = (
        await db.execute(
            select(Source)
            .where(
                Source.collection_id == collection_id,
                Source.type == SourceType.GITHUB,
                Source.enabled.is_(True),
            )
            .order_by(Source.last_run_at.desc(), Source.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if source is None:
        raise ToolError("No enabled GitHub source found for arc42 generation.")
    repo_path = Path(get_settings().repos_root) / str(source.id)
    if not repo_path.is_dir():
        raise ToolError("Local repository checkout missing. Run a sync first.")
    return repo_path


async def _fetch_arc42_artifact(db, collection_id: uuid.UUID, scenario_id: uuid.UUID):  # noqa: ANN202
    from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind

    return (
        await db.execute(
            select(KnowledgeArtifact).where(
                KnowledgeArtifact.collection_id == collection_id,
                KnowledgeArtifact.kind == KnowledgeArtifactKind.ARC42,
                KnowledgeArtifact.name == f"{scenario_id}.arc42.md",
            )
        )
    ).scalar_one_or_none()


def _bounded_sections(
    sections: dict[str, str], section_key: str | None
) -> tuple[dict[str, str], bool]:
    selected = {section_key: sections.get(section_key, "")} if section_key else dict(sections)
    truncated = False
    bounded: dict[str, str] = {}
    for key, value in selected.items():
        if len(value) > 12000:
            truncated = True
        bounded[key] = value[:12000]
    return bounded, truncated


def _arc42_result(existing, collection, scenario, section_key: str | None) -> Arc42Result:  # noqa: ANN001
    from contextmine_core.architecture import SECTION_TITLES

    meta = existing.meta or {}
    sections, sections_truncated = _bounded_sections(meta.get("sections") or {}, section_key)
    if section_key:
        markdown = (
            f"# arc42 - {scenario.name}\n\n"
            f"## {SECTION_TITLES.get(section_key, section_key)}\n"
            f"{sections.get(section_key, '')}\n"
        )
    else:
        markdown = existing.content
    return Arc42Result(
        collection_id=str(collection.id),
        scenario_id=str(scenario.id),
        artifact_id=str(existing.id),
        section=section_key,
        generated=True,
        cached=True,
        facts_hash=meta.get("facts_hash"),
        warnings=list(meta.get("warnings") or []),
        sections=sections,
        markdown=markdown[:50000],
        truncated=sections_truncated or len(markdown) > 50000,
    )


async def _regenerate_arc42(
    db,
    collection,
    scenario,
    existing,
    section_key: str | None,
    settings,  # noqa: ANN001
) -> Arc42Result:
    from contextmine_core.architecture import SECTION_TITLES
    from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind

    repo_path = await _resolve_repo_checkout(db, collection.id)
    try:
        document, sdk_meta = await generate_arc42_with_claude_sdk(
            collection_id=collection.id,
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            repo_path=repo_path,
            section=section_key,
            model=settings.arch_docs_agent_sdk_model,
            max_turns=int(settings.arch_docs_agent_sdk_max_turns),
        )
    except (ClaudeAgentSdkUnavailableError, ModelCallsDisabledError) as exc:
        raise ToolError(str(exc)) from exc
    content_hash = hashlib.sha256(document.markdown.encode()).hexdigest()
    meta = {
        "scenario_id": str(scenario.id),
        "generated_at": document.generated_at.isoformat(),
        "facts_hash": content_hash,
        "confidence_summary": document.confidence_summary,
        "section_coverage": document.section_coverage,
        "warnings": document.warnings,
        "sections": document.sections,
        "generation_engine": "claude_agent_sdk",
        "sdk": sdk_meta,
    }
    if existing is None:
        existing = KnowledgeArtifact(
            collection_id=collection.id,
            kind=KnowledgeArtifactKind.ARC42,
            name=f"{scenario.id}.arc42.md",
            content=document.markdown,
            meta=meta,
        )
        db.add(existing)
    else:
        existing.content = document.markdown
        existing.meta = meta
    await db.commit()
    sections, sections_truncated = _bounded_sections(document.sections, section_key)
    if section_key:
        markdown = (
            f"# {document.title}\n\n"
            f"## {SECTION_TITLES.get(section_key, section_key)}\n"
            f"{sections.get(section_key, '')}\n"
        )
    else:
        markdown = document.markdown
    return Arc42Result(
        collection_id=str(collection.id),
        scenario_id=str(scenario.id),
        artifact_id=str(existing.id),
        section=section_key,
        generated=True,
        cached=False,
        facts_hash=content_hash,
        warnings=list(document.warnings),
        sections=sections,
        markdown=markdown[:50000],
        truncated=sections_truncated or len(markdown) > 50000,
    )


async def get_arc42(
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    section: Annotated[
        str | None, Field(description="Optional arc42 section", max_length=100)
    ] = None,
    regenerate: Annotated[bool, Field(description="Regenerate with the Claude Agent SDK")] = False,
) -> Arc42Result:
    """Read or generate an arc42 artifact with the configured Claude Agent SDK."""
    from contextmine_core.architecture import normalize_arc42_section_key

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise ToolError("Architecture docs are disabled.")
    section_key = normalize_arc42_section_key(section)
    if section and section_key is None:
        raise ToolError("Invalid arc42 section.")
    async with get_db_session() as db:
        collection = await _resolve_collection(db, collection_id)
        scenario = await _resolve_scenario(db, collection, scenario_id)
        existing = await _fetch_arc42_artifact(db, collection.id, scenario.id)
        if existing is not None and not regenerate:
            return _arc42_result(existing, collection, scenario, section_key)
        if not regenerate:
            return Arc42Result(
                collection_id=str(collection.id),
                scenario_id=str(scenario.id),
                section=section_key,
                generated=False,
                cached=False,
            )
        try:
            return await _regenerate_arc42(
                db, collection, scenario, existing, section_key, settings
            )
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Failed to get arc42: {exc}") from exc


def _llm_provider(settings):  # noqa: ANN001, ANN202
    if not (settings.arch_docs_llm_enrich and settings.default_llm_provider):
        return None
    try:
        from contextmine_core.research.llm import get_llm_provider

        return get_llm_provider(settings.default_llm_provider)
    except Exception:
        return None


async def _resolve_baseline(db, collection_id: uuid.UUID, scenario, baseline_id: str | None):  # noqa: ANN001, ANN202
    from contextmine_core.models import TwinScenario

    if baseline_id:
        baseline = (
            await db.execute(
                select(TwinScenario).where(
                    TwinScenario.id == _uuid(baseline_id, "baseline_scenario_id"),
                    TwinScenario.collection_id == collection_id,
                )
            )
        ).scalar_one_or_none()
        if baseline is None:
            raise ToolError("Baseline scenario not found in collection.")
        return baseline
    if scenario.base_scenario_id:
        baseline = (
            await db.execute(
                select(TwinScenario).where(
                    TwinScenario.id == scenario.base_scenario_id,
                    TwinScenario.collection_id == collection_id,
                )
            )
        ).scalar_one_or_none()
        if baseline is not None:
            return baseline
    return (
        await db.execute(
            select(TwinScenario)
            .where(
                TwinScenario.collection_id == collection_id,
                TwinScenario.id != scenario.id,
            )
            .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def arc42_drift_report(
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    baseline_scenario_id: Annotated[
        str | None, Field(description="Optional baseline scenario UUID")
    ] = None,
    limit: Annotated[int, Field(description="Maximum deltas", ge=1, le=500)] = 200,
) -> ArchitectureDataResult:
    """Compute a bounded advisory arc42 drift report."""
    from contextmine_core.architecture import build_architecture_facts, compute_arc42_drift

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise ToolError("Architecture docs are disabled.")
    async with get_db_session() as db:
        collection = await _resolve_collection(db, collection_id)
        scenario = await _resolve_scenario(db, collection, scenario_id)
        baseline = await _resolve_baseline(db, collection.id, scenario, baseline_scenario_id)
        provider = _llm_provider(settings)
        try:
            current = await build_architecture_facts(
                db,
                collection_id=collection.id,
                scenario_id=scenario.id,
                enable_llm_enrich=settings.arch_docs_llm_enrich,
                llm_provider=provider,
                llm_hypothesis_limit=settings.arch_docs_llm_max_hypotheses,
            )
            baseline_bundle = (
                await build_architecture_facts(
                    db,
                    collection_id=collection.id,
                    scenario_id=baseline.id,
                    enable_llm_enrich=settings.arch_docs_llm_enrich,
                    llm_provider=provider,
                    llm_hypothesis_limit=settings.arch_docs_llm_max_hypotheses,
                )
                if baseline
                else None
            )
            report = compute_arc42_drift(
                current,
                baseline_bundle,
                baseline_scenario_id=baseline.id if baseline else None,
            )
        except Exception as exc:
            raise ToolError(f"Failed to compute arc42 drift report: {exc}") from exc
    all_deltas = [asdict(delta) for delta in report.deltas]
    by_type: dict[str, int] = {}
    for delta in all_deltas:
        key = str(delta.get("delta_type"))
        by_type[key] = by_type.get(key, 0) + 1
    data = {
        "baseline_scenario_id": str(baseline.id) if baseline else None,
        "generated_at": report.generated_at.isoformat(),
        "current_hash": report.current_hash,
        "baseline_hash": report.baseline_hash,
        "summary": {"total": len(all_deltas), "by_type": by_type},
        "deltas": all_deltas[:limit],
        "warnings": report.warnings,
    }
    return ArchitectureDataResult(
        collection_id=str(collection.id),
        scenario_id=str(scenario.id),
        data=data,
        truncated=len(all_deltas) > limit,
    )


async def list_ports_adapters(
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
    scenario_id: Annotated[str | None, Field(description="Optional scenario UUID")] = None,
    direction: Annotated[
        Literal["inbound", "outbound"] | None,
        Field(description="Optional direction filter"),
    ] = None,
    container: Annotated[
        str | None, Field(description="Optional container filter", max_length=500)
    ] = None,
    limit: Annotated[int, Field(description="Maximum mappings", ge=1, le=500)] = 200,
) -> ArchitectureDataResult:
    """List a bounded Ports-and-Adapters mapping from existing architecture facts."""
    from contextmine_core.architecture import build_architecture_facts

    settings = get_settings()
    if not settings.arch_docs_enabled:
        raise ToolError("Architecture docs are disabled.")
    async with get_db_session() as db:
        collection = await _resolve_collection(db, collection_id)
        scenario = await _resolve_scenario(db, collection, scenario_id)
        try:
            bundle = await build_architecture_facts(
                db,
                collection_id=collection.id,
                scenario_id=scenario.id,
                enable_llm_enrich=settings.arch_docs_llm_enrich,
                llm_provider=_llm_provider(settings),
                llm_hypothesis_limit=settings.arch_docs_llm_max_hypotheses,
            )
        except Exception as exc:
            raise ToolError(f"Failed to list ports/adapters: {exc}") from exc
    rows = bundle.ports_adapters
    if direction:
        rows = [row for row in rows if row.direction == direction]
    if container:
        target = container.strip().lower()
        rows = [row for row in rows if (row.container or "").strip().lower() == target]
    data = {
        "summary": {
            "total": len(rows),
            "inbound": sum(row.direction == "inbound" for row in rows),
            "outbound": sum(row.direction == "outbound" for row in rows),
        },
        "filters": {"direction": direction, "container": container},
        "warnings": bundle.warnings,
        "items": [asdict(row) for row in rows[:limit]],
    }
    return ArchitectureDataResult(
        collection_id=str(collection.id),
        scenario_id=str(scenario.id),
        data=data,
        truncated=len(rows) > limit,
    )


def register_architecture_tools(mcp: FastMCP) -> None:
    """Register architecture intent and documentation tools."""
    mcp.tool(name="create_architecture_intent")(create_architecture_intent)
    mcp.tool(name="approve_architecture_intent")(approve_architecture_intent)
    mcp.tool(name="get_arc42")(get_arc42)
    mcp.tool(name="arc42_drift_report")(arc42_drift_report)
    mcp.tool(name="list_ports_adapters")(list_ports_adapters)
