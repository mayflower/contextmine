"""Lifecycle, status, timeline, diff, analysis, and findings helpers for the twin subsystem."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from contextmine_core.joern import JoernClient, parse_joern_output
from contextmine_core.lsp import get_lsp_manager
from contextmine_core.models import (
    Document,
    Source,
    SourceType,
    TwinAnalysisCache,
    TwinEdge,
    TwinEvent,
    TwinFinding,
    TwinNode,
    TwinScenario,
    TwinSourceVersion,
)
from contextmine_core.settings import get_settings
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_EXTRACTOR_VERSION = "scip-kg-v1"

SOURCE_PATTERNS: dict[str, list[str]] = {
    "python": ["input", "request", "args", "form", "json", "cookie", "header", "argv", "getenv"],
    "javascript": ["req", "query", "params", "body", "cookie", "header", "process.env"],
    "typescript": ["req", "query", "params", "body", "cookie", "header", "process.env"],
    "java": ["getparameter", "getheader", "getquerystring", "cookie", "inputstream"],
    "go": ["formvalue", "request", "header", "body", "os.args", "os.getenv"],
    "php": ["_get", "_post", "_cookie", "_request", "_server"],
}

SINK_PATTERNS: dict[str, list[str]] = {
    "python": ["eval", "exec", "system", "popen", "subprocess", "execute"],
    "javascript": ["eval", "exec", "spawn", "innerhtml", "query", "execute"],
    "typescript": ["eval", "exec", "spawn", "innerhtml", "query", "execute"],
    "java": ["runtime.exec", "executequery", "executeupdate", "sendredirect"],
    "go": ["exec.command", "query", "writer.write", "template.execute"],
    "php": ["eval", "exec", "shell_exec", "system", "query"],
}

SANITIZER_PATTERNS: dict[str, list[str]] = {
    "python": ["escape", "sanitize", "quote", "int", "float"],
    "javascript": ["escape", "sanitize", "encodeuri", "encodeuricomponent", "parseint"],
    "typescript": ["escape", "sanitize", "encodeuri", "encodeuricomponent", "parseint"],
    "java": ["escape", "sanitize", "parseint", "preparedstatement"],
    "go": ["queryescape", "htmlescape", "atoi", "parseint"],
    "php": ["htmlspecialchars", "htmlentities", "intval", "filter_var"],
}

SEVERITY_ORDER: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_pattern_token(value: str) -> str:
    trimmed = value.strip().lower()
    if "." in trimmed:
        return trimmed.rsplit(".", 1)[-1]
    return trimmed


async def compute_revision_key_for_source(session: AsyncSession, source: Source) -> str:
    """Compute deterministic source revision key for one source."""
    if source.type == SourceType.GITHUB and source.cursor:
        return str(source.cursor).strip()

    docs = (
        await session.execute(
            select(Document.uri, Document.content_hash)
            .where(Document.source_id == source.id)
            .order_by(Document.uri.asc())
        )
    ).all()
    if not docs:
        return f"source:{source.id}:empty"

    digest_input = "\n".join(f"{row.uri}:{row.content_hash}" for row in docs)
    return f"web:{_sha(digest_input)}"


def compute_analysis_context_key(
    *,
    source_id: UUID,
    revision_key: str,
    extractor_version: str,
    projection_profile: str,
) -> str:
    """Compute deterministic analysis cache key context."""
    return _sha(f"{source_id}:{revision_key}:{extractor_version}:{projection_profile}")


async def get_or_create_source_version(
    session: AsyncSession,
    *,
    collection_id: UUID,
    source_id: UUID,
    revision_key: str,
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
    language_profile: str | None = None,
    status: str = "queued",
) -> TwinSourceVersion:
    """Find or create a source-version row."""
    existing = (
        await session.execute(
            select(TwinSourceVersion).where(
                TwinSourceVersion.source_id == source_id,
                TwinSourceVersion.revision_key == revision_key,
                TwinSourceVersion.extractor_version == extractor_version,
            )
        )
    ).scalar_one_or_none()
    if existing:
        return existing

    source_version = TwinSourceVersion(
        id=uuid.uuid4(),
        collection_id=collection_id,
        source_id=source_id,
        revision_key=revision_key,
        extractor_version=extractor_version,
        language_profile=language_profile,
        status=status,
        stats={},
    )
    session.add(source_version)
    await session.flush()
    return source_version


async def set_source_version_status(
    session: AsyncSession,
    *,
    source_version_id: UUID,
    status: str,
    stats: dict[str, Any] | None = None,
    started: bool = False,
    finished: bool = False,
) -> TwinSourceVersion:
    """Transition source-version status and update timestamps/stats."""
    source_version = (
        await session.execute(
            select(TwinSourceVersion).where(TwinSourceVersion.id == source_version_id)
        )
    ).scalar_one()

    now = datetime.now(UTC)
    source_version.status = status
    if stats:
        merged = dict(source_version.stats or {})
        merged.update(stats)
        source_version.stats = merged
    if started and source_version.started_at is None:
        source_version.started_at = now
    if finished:
        source_version.finished_at = now
    source_version.updated_at = now
    return source_version


async def mark_previous_source_versions_stale(
    session: AsyncSession,
    *,
    source_id: UUID,
    keep_source_version_id: UUID,
) -> int:
    """Mark old ready source versions as stale for one source."""
    rows = (
        (
            await session.execute(
                select(TwinSourceVersion).where(
                    TwinSourceVersion.source_id == source_id,
                    TwinSourceVersion.id != keep_source_version_id,
                    TwinSourceVersion.status == "ready",
                )
            )
        )
        .scalars()
        .all()
    )
    now = datetime.now(UTC)
    for row in rows:
        row.status = "stale"
        row.updated_at = now
    return len(rows)


async def record_twin_event(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    source_id: UUID | None,
    source_version_id: UUID | None,
    event_type: str,
    status: str,
    payload: dict[str, Any] | None,
    idempotency_key: str,
    error: str | None = None,
) -> TwinEvent:
    """Record a twin lifecycle event with idempotency guard."""
    existing = (
        await session.execute(select(TwinEvent).where(TwinEvent.idempotency_key == idempotency_key))
    ).scalar_one_or_none()
    if existing:
        return existing

    event = TwinEvent(
        id=uuid.uuid4(),
        collection_id=collection_id,
        scenario_id=scenario_id,
        source_id=source_id,
        source_version_id=source_version_id,
        event_type=event_type,
        status=status,
        payload=payload or {},
        idempotency_key=idempotency_key,
        error=error,
    )
    session.add(event)
    await session.flush()
    return event


async def invalidate_analysis_cache_for_scenario(
    session: AsyncSession,
    *,
    scenario_id: UUID,
) -> int:
    """Delete cached analysis payloads for a scenario."""
    result = await session.execute(
        delete(TwinAnalysisCache).where(TwinAnalysisCache.scenario_id == scenario_id)
    )
    rowcount = getattr(result, "rowcount", None)
    return int(rowcount or 0)


async def _resolve_status_scenario(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
) -> TwinScenario | None:
    if scenario_id is not None:
        return (
            await session.execute(
                select(TwinScenario).where(
                    TwinScenario.id == scenario_id,
                    TwinScenario.collection_id == collection_id,
                )
            )
        ).scalar_one_or_none()
    return (
        await session.execute(
            select(TwinScenario).where(
                TwinScenario.collection_id == collection_id,
                TwinScenario.is_as_is.is_(True),
            )
        )
    ).scalar_one_or_none()


async def get_collection_twin_status(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None = None,
) -> dict[str, Any]:
    """Return source-level twin materialization status for one collection."""
    scenario = await _resolve_status_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    sources = (
        (await session.execute(select(Source).where(Source.collection_id == collection_id)))
        .scalars()
        .all()
    )

    now = datetime.now(UTC)
    source_rows: list[dict[str, Any]] = []
    ready_count = 0
    failed_count = 0
    in_progress_count = 0
    materialized_at: datetime | None = None

    stage_latest: dict[str, UUID] = {}
    stage_timings: dict[str, float] = defaultdict(float)
    behavioral_status_values: list[str] = []
    behavioral_materialized_at: datetime | None = None
    deep_warnings: list[str] = []

    for source in sources:
        latest = (
            await session.execute(
                select(TwinSourceVersion)
                .where(TwinSourceVersion.source_id == source.id)
                .order_by(TwinSourceVersion.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        status = latest.status if latest else "queued"
        revision_key = latest.revision_key if latest else None
        finished_at = latest.finished_at if latest else None
        started_at = latest.started_at if latest else None
        lag_seconds: int | None = None
        if finished_at:
            lag_seconds = int((now - finished_at).total_seconds())
        elif started_at:
            lag_seconds = int((now - started_at).total_seconds())

        if status == "ready":
            ready_count += 1
            if finished_at and (materialized_at is None or finished_at > materialized_at):
                materialized_at = finished_at
        elif status in {"failed"}:
            failed_count += 1
        elif status in {"queued", "materializing", "loading", "generating"}:
            in_progress_count += 1

        last_error_event = (
            await session.execute(
                select(TwinEvent)
                .where(
                    TwinEvent.collection_id == collection_id,
                    TwinEvent.source_id == source.id,
                    TwinEvent.status == "failed",
                )
                .order_by(TwinEvent.event_ts.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        source_rows.append(
            {
                "source_id": str(source.id),
                "revision_key": revision_key,
                "status": status,
                "lag_seconds": lag_seconds,
                "last_error": last_error_event.error if last_error_event else None,
                "source_version_id": str(latest.id) if latest else None,
            }
        )
        if latest:
            latest_stats = dict(latest.stats or {})
            behavioral_status = str(latest_stats.get("behavioral_layers_status") or "").strip()
            if behavioral_status:
                behavioral_status_values.append(behavioral_status)

            behavioral_ts_raw = latest_stats.get("last_behavioral_materialized_at")
            if isinstance(behavioral_ts_raw, str) and behavioral_ts_raw.strip():
                try:
                    parsed = datetime.fromisoformat(behavioral_ts_raw.replace("Z", "+00:00"))
                    if behavioral_materialized_at is None or parsed > behavioral_materialized_at:
                        behavioral_materialized_at = parsed
                except ValueError:
                    pass

            warnings = latest_stats.get("deep_warnings")
            if isinstance(warnings, list):
                deep_warnings.extend(str(item) for item in warnings if str(item).strip())

    events = (
        (
            await session.execute(
                select(TwinEvent)
                .where(TwinEvent.collection_id == collection_id)
                .order_by(TwinEvent.event_ts.desc())
                .limit(200)
            )
        )
        .scalars()
        .all()
    )
    for event in events:
        stage_latest.setdefault(event.event_type, event.id)
        payload = event.payload or {}
        duration = payload.get("duration_seconds")
        if isinstance(duration, (int, float)):
            stage_timings[event.event_type] += float(duration)

    if failed_count > 0:
        freshness = "degraded"
    elif in_progress_count > 0:
        freshness = "materializing"
    elif sources and ready_count == len(sources):
        freshness = "ready"
    elif not sources:
        freshness = "failed"
    else:
        freshness = "materializing"

    if not behavioral_status_values:
        behavioral_layers_status = "pending"
    elif any(status == "failed" for status in behavioral_status_values):
        behavioral_layers_status = "failed"
    elif any(status in {"materializing", "queued"} for status in behavioral_status_values):
        behavioral_layers_status = "materializing"
    elif all(status == "ready" for status in behavioral_status_values):
        behavioral_layers_status = "ready"
    elif all(status == "disabled" for status in behavioral_status_values):
        behavioral_layers_status = "disabled"
    else:
        behavioral_layers_status = "pending"

    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id) if scenario else None,
        "scenario_version": int(scenario.version) if scenario else None,
        "materialized_at": materialized_at.isoformat() if materialized_at else None,
        "freshness": freshness,
        "behavioral_layers_status": behavioral_layers_status,
        "last_behavioral_materialized_at": (
            behavioral_materialized_at.isoformat() if behavioral_materialized_at else None
        ),
        "deep_warnings": sorted(set(deep_warnings))[:100],
        "sources": source_rows,
        "pipeline": {
            "stage_timings": {key: round(value, 3) for key, value in stage_timings.items()},
            "last_successful_event_ids": {key: str(value) for key, value in stage_latest.items()},
        },
    }


async def list_collection_twin_events(
    session: AsyncSession,
    *,
    collection_id: UUID,
    page: int = 0,
    limit: int = 50,
    source_id: UUID | None = None,
    event_type: str | None = None,
    status: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
) -> dict[str, Any]:
    """Return paginated timeline rows for one collection."""
    stmt = select(TwinEvent).where(TwinEvent.collection_id == collection_id)
    if source_id:
        stmt = stmt.where(TwinEvent.source_id == source_id)
    if event_type:
        stmt = stmt.where(TwinEvent.event_type == event_type)
    if status:
        stmt = stmt.where(TwinEvent.status == status)
    if from_ts:
        stmt = stmt.where(TwinEvent.event_ts >= from_ts)
    if to_ts:
        stmt = stmt.where(TwinEvent.event_ts <= to_ts)

    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()

    rows = (
        (
            await session.execute(
                stmt.order_by(TwinEvent.event_ts.desc()).offset(page * limit).limit(limit)
            )
        )
        .scalars()
        .all()
    )

    return {
        "collection_id": str(collection_id),
        "items": [
            {
                "id": str(row.id),
                "scenario_id": str(row.scenario_id) if row.scenario_id else None,
                "source_id": str(row.source_id) if row.source_id else None,
                "source_version_id": str(row.source_version_id) if row.source_version_id else None,
                "event_type": row.event_type,
                "status": row.status,
                "payload": row.payload or {},
                "event_ts": row.event_ts.isoformat(),
                "idempotency_key": row.idempotency_key,
                "error": row.error,
            }
            for row in rows
        ],
        "page": page,
        "limit": limit,
        "total": int(total or 0),
    }


async def trigger_collection_refresh(
    session: AsyncSession,
    *,
    collection_id: UUID,
    source_ids: list[UUID] | None = None,
    force: bool = False,
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
) -> dict[str, Any]:
    """Create queued source-version rows and refresh events for one collection."""
    query = select(Source).where(Source.collection_id == collection_id)
    if source_ids:
        query = query.where(Source.id.in_(source_ids))
    sources = (await session.execute(query)).scalars().all()

    created = 0
    skipped = 0
    items: list[dict[str, Any]] = []
    for source in sources:
        revision_key = await compute_revision_key_for_source(session, source)
        latest = (
            await session.execute(
                select(TwinSourceVersion)
                .where(TwinSourceVersion.source_id == source.id)
                .order_by(TwinSourceVersion.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if (
            not force
            and latest
            and latest.revision_key == revision_key
            and latest.extractor_version == extractor_version
            and latest.status in {"ready", "queued", "materializing"}
        ):
            skipped += 1
            items.append(
                {
                    "source_id": str(source.id),
                    "source_version_id": str(latest.id),
                    "status": latest.status,
                    "revision_key": latest.revision_key,
                    "queued": False,
                }
            )
            continue

        source_version = await get_or_create_source_version(
            session,
            collection_id=collection_id,
            source_id=source.id,
            revision_key=revision_key,
            extractor_version=extractor_version,
            status="queued",
        )
        await record_twin_event(
            session,
            collection_id=collection_id,
            scenario_id=None,
            source_id=source.id,
            source_version_id=source_version.id,
            event_type="refresh_requested",
            status="queued",
            payload={"force": force, "revision_key": revision_key},
            idempotency_key=f"refresh:{source.id}:{source_version.id}",
        )
        created += 1
        items.append(
            {
                "source_id": str(source.id),
                "source_version_id": str(source_version.id),
                "status": source_version.status,
                "revision_key": source_version.revision_key,
                "queued": True,
            }
        )

    return {
        "collection_id": str(collection_id),
        "created": created,
        "skipped": skipped,
        "items": items,
    }


async def get_collection_twin_diff(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    from_version: int,
    to_version: int,
) -> dict[str, Any]:
    """Return aggregated delta between two scenario versions."""
    scenario = await _resolve_status_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    if not scenario:
        raise ValueError("Scenario not found")
    if from_version > to_version:
        raise ValueError("from_version must be <= to_version")

    events = (
        (
            await session.execute(
                select(TwinEvent).where(
                    TwinEvent.collection_id == collection_id,
                    TwinEvent.scenario_id == scenario.id,
                    TwinEvent.event_type.in_(
                        [
                            "snapshot_ingested",
                            "kg_refreshed",
                            "metrics_applied",
                            "materialization_complete",
                        ]
                    ),
                )
            )
        )
        .scalars()
        .all()
    )
    nodes_added = 0
    nodes_removed = 0
    edges_added = 0
    edges_removed = 0
    changed_keys: set[str] = set()
    for event in events:
        payload = event.payload or {}
        version = payload.get("scenario_version")
        if isinstance(version, int) and (version < from_version or version > to_version):
            continue
        nodes_added += int(payload.get("nodes_upserted", 0) or 0)
        edges_added += int(payload.get("edges_upserted", 0) or 0)
        nodes_removed += int(payload.get("nodes_deactivated", 0) or 0)
        edges_removed += int(payload.get("edges_deactivated", 0) or 0)
        for key in payload.get("sample_node_keys", [])[:10]:
            if isinstance(key, str):
                changed_keys.add(key)

    sample_nodes = (
        (
            await session.execute(
                select(TwinNode)
                .where(
                    TwinNode.scenario_id == scenario.id,
                    TwinNode.natural_key.in_(list(changed_keys)[:100]) if changed_keys else True,
                )
                .order_by(TwinNode.updated_at.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )

    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "from_version": from_version,
        "to_version": to_version,
        "delta": {
            "nodes_added": nodes_added,
            "nodes_removed": nodes_removed,
            "edges_added": edges_added,
            "edges_removed": edges_removed,
        },
        "sample_nodes": [
            {
                "id": str(node.id),
                "natural_key": node.natural_key,
                "name": node.name,
                "kind": node.kind,
                "is_active": bool(node.is_active),
            }
            for node in sample_nodes
        ],
    }


async def _current_analysis_cache_key(
    session: AsyncSession,
    *,
    collection_id: UUID,
    projection_profile: str,
) -> str:
    ready_versions = (
        (
            await session.execute(
                select(TwinSourceVersion)
                .where(
                    TwinSourceVersion.collection_id == collection_id,
                    TwinSourceVersion.status == "ready",
                )
                .order_by(TwinSourceVersion.source_id.asc(), TwinSourceVersion.revision_key.asc())
            )
        )
        .scalars()
        .all()
    )
    if not ready_versions:
        return _sha(f"{collection_id}:empty:{projection_profile}")
    raw = "|".join(
        f"{row.source_id}:{row.revision_key}:{row.extractor_version}" for row in ready_versions
    )
    return _sha(f"{raw}:{projection_profile}")


def _hash_params(params: dict[str, Any]) -> str:
    return _sha(json.dumps(params, sort_keys=True, separators=(",", ":")))


async def _read_cached_analysis(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    engine: str = "graphrag",
    tool_name: str,
    cache_key: str,
    params_hash: str,
) -> dict[str, Any] | None:
    now = datetime.now(UTC)
    row = (
        await session.execute(
            select(TwinAnalysisCache).where(
                TwinAnalysisCache.scenario_id == scenario_id,
                TwinAnalysisCache.engine == engine,
                TwinAnalysisCache.tool_name == tool_name,
                TwinAnalysisCache.cache_key == cache_key,
                TwinAnalysisCache.params_hash == params_hash,
                TwinAnalysisCache.expires_at > now,
            )
        )
    ).scalar_one_or_none()
    return dict(row.payload or {}) if row else None


async def _write_cached_analysis(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    engine: str = "graphrag",
    tool_name: str,
    cache_key: str,
    params_hash: str,
    payload: dict[str, Any],
    ttl_seconds: int,
) -> None:
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=max(ttl_seconds, 1))
    row = (
        await session.execute(
            select(TwinAnalysisCache).where(
                TwinAnalysisCache.scenario_id == scenario_id,
                TwinAnalysisCache.engine == engine,
                TwinAnalysisCache.tool_name == tool_name,
                TwinAnalysisCache.cache_key == cache_key,
                TwinAnalysisCache.params_hash == params_hash,
            )
        )
    ).scalar_one_or_none()
    if row:
        row.payload = payload
        row.created_at = now
        row.expires_at = expires_at
        return
    session.add(
        TwinAnalysisCache(
            id=uuid.uuid4(),
            cache_key=cache_key,
            scenario_id=scenario_id,
            engine=engine,
            tool_name=tool_name,
            params_hash=params_hash,
            payload=payload,
            expires_at=expires_at,
        )
    )


async def _resolve_analysis_scenario(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
) -> TwinScenario:
    scenario = await _resolve_status_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    if not scenario:
        raise ValueError("Scenario not found")
    return scenario


async def _resolve_node_by_ref(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    node_ref: str,
) -> TwinNode | None:
    node_uuid: UUID | None = None
    try:
        node_uuid = UUID(node_ref)
    except ValueError:
        node_uuid = None
    if node_uuid:
        node = (
            await session.execute(
                select(TwinNode).where(
                    TwinNode.scenario_id == scenario_id,
                    TwinNode.id == node_uuid,
                    TwinNode.is_active.is_(True),
                )
            )
        ).scalar_one_or_none()
        if node:
            return node
    return (
        (
            await session.execute(
                select(TwinNode)
                .where(
                    TwinNode.scenario_id == scenario_id,
                    TwinNode.is_active.is_(True),
                    or_(
                        TwinNode.natural_key == node_ref,
                        TwinNode.name == node_ref,
                    ),
                )
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def get_codebase_summary(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    params: dict[str, Any] = {}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile="summary"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="summary",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    total_nodes = (
        await session.execute(
            select(func.count())
            .select_from(TwinNode)
            .where(
                TwinNode.scenario_id == scenario.id,
                TwinNode.is_active.is_(True),
            )
        )
    ).scalar_one()
    total_edges = (
        await session.execute(
            select(func.count())
            .select_from(TwinEdge)
            .where(
                TwinEdge.scenario_id == scenario.id,
                TwinEdge.is_active.is_(True),
            )
        )
    ).scalar_one()
    file_count = (
        await session.execute(
            select(func.count())
            .select_from(TwinNode)
            .where(
                TwinNode.scenario_id == scenario.id,
                TwinNode.is_active.is_(True),
                TwinNode.kind == "file",
            )
        )
    ).scalar_one()
    method_like_count = (
        await session.execute(
            select(func.count())
            .select_from(TwinNode)
            .where(
                TwinNode.scenario_id == scenario.id,
                TwinNode.is_active.is_(True),
                TwinNode.kind.in_(["function", "method"]),
            )
        )
    ).scalar_one()
    call_count = (
        await session.execute(
            select(func.count())
            .select_from(TwinEdge)
            .where(
                TwinEdge.scenario_id == scenario.id,
                TwinEdge.is_active.is_(True),
                TwinEdge.kind == "symbol_calls_symbol",
            )
        )
    ).scalar_one()

    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "scenario_version": int(scenario.version),
        "summary": {
            "nodes": int(total_nodes or 0),
            "edges": int(total_edges or 0),
            "files": int(file_count or 0),
            "methods": int(method_like_count or 0),
            "calls": int(call_count or 0),
        },
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="summary",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def list_methods(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    query: str | None,
    page: int,
    limit: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    params = {"query": query or "", "page": page, "limit": limit}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile="methods"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="methods",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    stmt = select(TwinNode).where(
        TwinNode.scenario_id == scenario.id,
        TwinNode.is_active.is_(True),
        TwinNode.kind.in_(["function", "method", "class"]),
    )
    if query:
        escaped = f"%{query.strip()}%"
        stmt = stmt.where(
            or_(
                TwinNode.name.ilike(escaped),
                TwinNode.natural_key.ilike(escaped),
            )
        )
    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (
        (
            await session.execute(
                stmt.order_by(TwinNode.name.asc()).offset(page * limit).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "items": [
            {
                "id": str(row.id),
                "name": row.name,
                "kind": row.kind,
                "natural_key": row.natural_key,
                "file_path": (row.meta or {}).get("file_path"),
                "meta": row.meta or {},
            }
            for row in rows
        ],
        "page": page,
        "limit": limit,
        "total": int(total or 0),
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="methods",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def list_calls(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    page: int,
    limit: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    params = {"page": page, "limit": limit}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile="calls"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="calls",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    node_rows = (
        await session.execute(
            select(TwinNode.id, TwinNode.name, TwinNode.natural_key).where(
                TwinNode.scenario_id == scenario.id,
                TwinNode.is_active.is_(True),
            )
        )
    ).all()
    node_by_id = {row.id: row for row in node_rows}
    edge_stmt = select(TwinEdge).where(
        TwinEdge.scenario_id == scenario.id,
        TwinEdge.is_active.is_(True),
        TwinEdge.kind == "symbol_calls_symbol",
    )
    total = (
        await session.execute(select(func.count()).select_from(edge_stmt.subquery()))
    ).scalar_one()
    edges = (
        (
            await session.execute(
                edge_stmt.order_by(TwinEdge.created_at.desc()).offset(page * limit).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    items = []
    for edge in edges:
        src = node_by_id.get(edge.source_node_id)
        dst = node_by_id.get(edge.target_node_id)
        items.append(
            {
                "id": str(edge.id),
                "caller_id": str(edge.source_node_id),
                "callee_id": str(edge.target_node_id),
                "caller": src.name if src else str(edge.source_node_id),
                "callee": dst.name if dst else str(edge.target_node_id),
                "caller_natural_key": src.natural_key if src else None,
                "callee_natural_key": dst.natural_key if dst else None,
                "meta": edge.meta or {},
            }
        )
    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "items": items,
        "page": page,
        "limit": limit,
        "total": int(total or 0),
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="calls",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def get_cfg(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    depth: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    params = {"node_ref": node_ref, "depth": depth}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile="cfg"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="cfg",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    root = await _resolve_node_by_ref(session, scenario_id=scenario.id, node_ref=node_ref)
    if not root:
        raise ValueError("Node not found")

    queue: deque[tuple[UUID, int]] = deque([(root.id, 0)])
    seen: set[UUID] = {root.id}
    adjacency: list[TwinEdge] = []
    while queue:
        node_id, hop = queue.popleft()
        if hop >= max(depth, 1):
            continue
        edges = (
            (
                await session.execute(
                    select(TwinEdge).where(
                        TwinEdge.scenario_id == scenario.id,
                        TwinEdge.is_active.is_(True),
                        TwinEdge.kind.in_(["symbol_calls_symbol", "symbol_contains_symbol"]),
                        or_(
                            TwinEdge.source_node_id == node_id,
                            TwinEdge.target_node_id == node_id,
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )
        for edge in edges:
            adjacency.append(edge)
            next_id = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
            if next_id in seen:
                continue
            seen.add(next_id)
            queue.append((next_id, hop + 1))

    nodes = (
        (
            await session.execute(
                select(TwinNode).where(
                    TwinNode.scenario_id == scenario.id,
                    TwinNode.id.in_(list(seen)),
                )
            )
        )
        .scalars()
        .all()
    )
    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "root_node_id": str(root.id),
        "approximation": True,
        "nodes": [
            {
                "id": str(node.id),
                "name": node.name,
                "kind": node.kind,
                "natural_key": node.natural_key,
            }
            for node in nodes
        ],
        "edges": [
            {
                "id": str(edge.id),
                "source_node_id": str(edge.source_node_id),
                "target_node_id": str(edge.target_node_id),
                "kind": edge.kind,
                "meta": edge.meta or {},
            }
            for edge in adjacency
        ],
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="cfg",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def get_variable_flow(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    variable: str | None,
    max_hops: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    params = {
        "node_ref": node_ref,
        "variable": variable or "",
        "max_hops": max_hops,
    }
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile="variable_flow"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="variable_flow",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    root = await _resolve_node_by_ref(session, scenario_id=scenario.id, node_ref=node_ref)
    if not root:
        raise ValueError("Node not found")

    edge_kinds = ["symbol_references_symbol", "symbol_calls_symbol", "symbol_contains_symbol"]
    queue: deque[list[UUID]] = deque([[root.id]])
    paths: list[list[UUID]] = []
    while queue and len(paths) < 100:
        path = queue.popleft()
        current = path[-1]
        if len(path) - 1 >= max(max_hops, 1):
            paths.append(path)
            continue
        edges = (
            (
                await session.execute(
                    select(TwinEdge).where(
                        TwinEdge.scenario_id == scenario.id,
                        TwinEdge.is_active.is_(True),
                        TwinEdge.kind.in_(edge_kinds),
                        TwinEdge.source_node_id == current,
                    )
                )
            )
            .scalars()
            .all()
        )
        if not edges:
            paths.append(path)
            continue
        for edge in edges[:6]:
            if edge.target_node_id in path:
                continue
            queue.append([*path, edge.target_node_id])

    node_ids = {node_id for path in paths for node_id in path}
    nodes = (
        (
            await session.execute(
                select(TwinNode).where(
                    TwinNode.scenario_id == scenario.id,
                    TwinNode.id.in_(list(node_ids)),
                )
            )
        )
        .scalars()
        .all()
    )
    node_map = {node.id: node for node in nodes}
    rendered_paths = [
        [
            {
                "node_id": str(node_id),
                "name": node_map[node_id].name if node_id in node_map else str(node_id),
                "natural_key": node_map[node_id].natural_key if node_id in node_map else None,
            }
            for node_id in path
        ]
        for path in paths[:50]
    ]
    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "variable": variable,
        "approximation": True,
        "root_node_id": str(root.id),
        "paths": rendered_paths,
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="variable_flow",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def _find_pattern_nodes(
    session: AsyncSession,
    *,
    scenario_id: UUID,
    patterns: list[str],
    limit: int,
) -> list[TwinNode]:
    nodes = (
        (
            await session.execute(
                select(TwinNode).where(
                    TwinNode.scenario_id == scenario_id,
                    TwinNode.is_active.is_(True),
                    TwinNode.kind.in_(["function", "method", "api_endpoint"]),
                )
            )
        )
        .scalars()
        .all()
    )
    prepared = [_normalize_pattern_token(item) for item in patterns]
    result: list[TwinNode] = []
    for node in nodes:
        lower_name = node.name.lower()
        if any(token in lower_name for token in prepared):
            result.append(node)
            if len(result) >= limit:
                break
    return result


async def find_taint_sources(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    lang = (language or "python").lower()
    patterns = SOURCE_PATTERNS.get(lang, SOURCE_PATTERNS["python"])
    params = {"language": lang, "limit": limit}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile=f"taint_sources:{lang}"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="taint_sources",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    rows = await _find_pattern_nodes(
        session, scenario_id=scenario.id, patterns=patterns, limit=limit
    )
    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "language": lang,
        "patterns": patterns,
        "items": [
            {
                "id": str(row.id),
                "name": row.name,
                "natural_key": row.natural_key,
                "kind": row.kind,
                "meta": row.meta or {},
            }
            for row in rows
        ],
        "total": len(rows),
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="taint_sources",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def find_taint_sinks(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    lang = (language or "python").lower()
    patterns = SINK_PATTERNS.get(lang, SINK_PATTERNS["python"])
    params = {"language": lang, "limit": limit}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile=f"taint_sinks:{lang}"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="taint_sinks",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    rows = await _find_pattern_nodes(
        session, scenario_id=scenario.id, patterns=patterns, limit=limit
    )
    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "language": lang,
        "patterns": patterns,
        "items": [
            {
                "id": str(row.id),
                "name": row.name,
                "natural_key": row.natural_key,
                "kind": row.kind,
                "meta": row.meta or {},
            }
            for row in rows
        ],
        "total": len(rows),
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="taint_sinks",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def find_taint_flows(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    max_hops: int,
    max_results: int,
    cache_ttl_seconds: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    lang = (language or "python").lower()
    params = {"language": lang, "max_hops": max_hops, "max_results": max_results}
    cache_key = await _current_analysis_cache_key(
        session, collection_id=collection_id, projection_profile=f"taint_flows:{lang}"
    )
    params_hash = _hash_params(params)
    cached = await _read_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="taint_flows",
        cache_key=cache_key,
        params_hash=params_hash,
    )
    if cached is not None:
        return cached

    sources_payload = await find_taint_sources(
        session,
        collection_id=collection_id,
        scenario_id=scenario.id,
        language=lang,
        limit=120,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    sinks_payload = await find_taint_sinks(
        session,
        collection_id=collection_id,
        scenario_id=scenario.id,
        language=lang,
        limit=120,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    source_ids = [UUID(item["id"]) for item in sources_payload["items"]]
    sink_ids = {UUID(item["id"]) for item in sinks_payload["items"]}
    if not source_ids or not sink_ids:
        payload = {
            "collection_id": str(collection_id),
            "scenario_id": str(scenario.id),
            "language": lang,
            "flows": [],
            "total": 0,
        }
        await _write_cached_analysis(
            session,
            scenario_id=scenario.id,
            tool_name="taint_flows",
            cache_key=cache_key,
            params_hash=params_hash,
            payload=payload,
            ttl_seconds=cache_ttl_seconds,
        )
        return payload

    edges = (
        (
            await session.execute(
                select(TwinEdge).where(
                    TwinEdge.scenario_id == scenario.id,
                    TwinEdge.is_active.is_(True),
                    TwinEdge.kind == "symbol_calls_symbol",
                )
            )
        )
        .scalars()
        .all()
    )
    outgoing: dict[UUID, list[UUID]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source_node_id].append(edge.target_node_id)

    node_rows = (
        await session.execute(
            select(TwinNode.id, TwinNode.name, TwinNode.natural_key).where(
                TwinNode.scenario_id == scenario.id,
                TwinNode.is_active.is_(True),
            )
        )
    ).all()
    node_map = {row.id: row for row in node_rows}

    sanitizer_tokens = [_normalize_pattern_token(x) for x in SANITIZER_PATTERNS.get(lang, [])]

    flows: list[dict[str, Any]] = []
    for source_id in source_ids:
        queue: deque[list[UUID]] = deque([[source_id]])
        while queue and len(flows) < max_results:
            path = queue.popleft()
            current = path[-1]
            if current in sink_ids and len(path) > 1:
                rendered = []
                for node_id in path:
                    node = node_map.get(node_id)
                    rendered.append(
                        {
                            "id": str(node_id),
                            "name": node.name if node else str(node_id),
                            "natural_key": node.natural_key if node else None,
                        }
                    )
                flows.append(
                    {
                        "source_id": str(source_id),
                        "sink_id": str(current),
                        "path": rendered,
                        "hops": len(path) - 1,
                        "confidence": "high" if len(path) <= 4 else "medium",
                    }
                )
                continue
            if len(path) - 1 >= max_hops:
                continue
            for target in outgoing.get(current, [])[:20]:
                if target in path:
                    continue
                target_node = node_map.get(target)
                if target_node and any(
                    token in target_node.name.lower() for token in sanitizer_tokens
                ):
                    continue
                queue.append([*path, target])

    payload = {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "language": lang,
        "flows": flows,
        "total": len(flows),
    }
    await _write_cached_analysis(
        session,
        scenario_id=scenario.id,
        tool_name="taint_flows",
        cache_key=cache_key,
        params_hash=params_hash,
        payload=payload,
        ttl_seconds=cache_ttl_seconds,
    )
    return payload


async def store_findings(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist normalized findings in twin_findings."""
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )

    def _finding_fingerprint(payload: dict[str, Any]) -> str:
        canonical = {
            "source_version_id": str(payload.get("source_version_id") or ""),
            "finding_type": str(payload.get("finding_type", "unknown")).strip().lower(),
            "severity": str(payload.get("severity", "medium")).strip().lower(),
            "confidence": str(payload.get("confidence", "medium")).strip().lower(),
            "status": str(payload.get("status", "open")).strip().lower(),
            "filename": str(payload.get("filename", "")).strip(),
            "line_number": int(payload.get("line_number", 1) or 1),
            "message": str(payload.get("message", "Finding")).strip(),
            "flow_data": dict(payload.get("flow_data") or {}),
            "meta": dict(payload.get("meta") or {}),
        }
        return _sha(json.dumps(canonical, sort_keys=True, separators=(",", ":")))

    created_ids: list[str] = []
    for raw in findings:
        fingerprint = _finding_fingerprint(raw)
        existing = (
            await session.execute(
                select(TwinFinding).where(
                    TwinFinding.scenario_id == scenario.id,
                    TwinFinding.fingerprint == fingerprint,
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.severity = str(raw.get("severity", existing.severity)).lower()
            existing.confidence = str(raw.get("confidence", existing.confidence)).lower()
            existing.status = str(raw.get("status", existing.status)).lower()
            existing.message = str(raw.get("message", existing.message))
            existing.flow_data = dict(raw.get("flow_data") or existing.flow_data or {})
            existing.meta = dict(raw.get("meta") or existing.meta or {})
            created_ids.append(str(existing.id))
            continue

        finding = TwinFinding(
            id=uuid.uuid4(),
            scenario_id=scenario.id,
            source_version_id=UUID(raw["source_version_id"])
            if raw.get("source_version_id")
            else None,
            fingerprint=fingerprint,
            finding_type=str(raw.get("finding_type", "unknown")),
            severity=str(raw.get("severity", "medium")).lower(),
            confidence=str(raw.get("confidence", "medium")).lower(),
            status=str(raw.get("status", "open")).lower(),
            filename=str(raw.get("filename", ""))[:2048],
            line_number=int(raw.get("line_number", 1) or 1),
            message=str(raw.get("message", "Finding")),
            flow_data=dict(raw.get("flow_data") or {}),
            meta=dict(raw.get("meta") or {}),
        )
        session.add(finding)
        created_ids.append(str(finding.id))
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "created": len(created_ids),
        "ids": created_ids,
    }


async def list_findings(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    status: str | None = None,
    min_severity: str | None = None,
    limit: int = 100,
    page: int = 0,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    stmt = select(TwinFinding).where(TwinFinding.scenario_id == scenario.id)
    if status:
        stmt = stmt.where(TwinFinding.status == status.lower())
    rows = (await session.execute(stmt.order_by(TwinFinding.created_at.desc()))).scalars().all()

    if min_severity:
        threshold = SEVERITY_ORDER.get(min_severity.lower(), 1)
        rows = [row for row in rows if SEVERITY_ORDER.get(row.severity.lower(), 0) >= threshold]
    total = len(rows)
    paged = rows[page * limit : page * limit + limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "items": [
            {
                "id": str(row.id),
                "finding_type": row.finding_type,
                "severity": row.severity,
                "confidence": row.confidence,
                "status": row.status,
                "filename": row.filename,
                "line_number": row.line_number,
                "message": row.message,
                "flow_data": row.flow_data or {},
                "meta": row.meta or {},
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in paged
        ],
        "page": page,
        "limit": limit,
        "total": total,
    }


def findings_to_sarif(
    *,
    collection_id: UUID,
    scenario_id: UUID,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    rules_map: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for item in findings:
        rule_id = str(item.get("finding_type") or "contextmine.rule")
        if rule_id not in rules_map:
            rules_map[rule_id] = {
                "id": rule_id,
                "name": rule_id,
                "shortDescription": {"text": rule_id.replace("_", " ").title()},
                "properties": {"tags": ["security", "digital-twin"]},
            }
        results.append(
            {
                "ruleId": rule_id,
                "level": (
                    "error"
                    if item.get("severity") in {"critical", "high"}
                    else "warning"
                    if item.get("severity") == "medium"
                    else "note"
                ),
                "message": {"text": str(item.get("message") or "")},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": str(item.get("filename") or "")},
                            "region": {"startLine": int(item.get("line_number") or 1)},
                        }
                    }
                ],
            }
        )

    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ContextMine Twin Analysis",
                        "informationUri": "https://github.com/mayflower/contextmine",
                        "rules": list(rules_map.values()),
                    }
                },
                "automationDetails": {
                    "id": f"collection:{collection_id}:scenario:{scenario_id}",
                },
                "results": results,
            }
        ],
    }


async def export_findings_sarif(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    status: str | None = None,
    min_severity: str | None = None,
) -> dict[str, Any]:
    listed = await list_findings(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        status=status,
        min_severity=min_severity,
        limit=10000,
        page=0,
    )
    scenario_uuid = UUID(listed["scenario_id"])
    sarif = findings_to_sarif(
        collection_id=collection_id,
        scenario_id=scenario_uuid,
        findings=listed["items"],
    )
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario_uuid),
        "sarif": sarif,
        "finding_count": len(listed["items"]),
    }


def parse_timestamp_value(value: str | None) -> datetime | None:
    """Parse timestamp from query params with UTC fallback."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def coerce_source_ids(values: list[str] | None) -> list[UUID]:
    """Parse optional source id list."""
    if not values:
        return []
    parsed: list[UUID] = []
    for value in values:
        parsed.append(UUID(value))
    return parsed


def sanitize_regex_query(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > 128:
        raise ValueError("query too long")
    if re.search(r"[^\w\s\.\-\*:/]", cleaned):
        raise ValueError("query contains unsupported characters")
    return cleaned


ANALYSIS_ENGINES = ("graphrag", "lsp", "joern")
LSP_METHOD_KINDS = {5, 6, 12}


def normalize_analysis_engines(engines: list[str] | None) -> list[str]:
    """Validate and normalize selected analysis engines."""
    if not engines:
        return list(ANALYSIS_ENGINES)

    normalized: list[str] = []
    seen: set[str] = set()
    for engine in engines:
        candidate = engine.strip().lower()
        if not candidate:
            continue
        if candidate not in ANALYSIS_ENGINES:
            raise ValueError(
                f"Unsupported analysis engine '{candidate}'. "
                f"Allowed values: {', '.join(ANALYSIS_ENGINES)}"
            )
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    if not normalized:
        raise ValueError("At least one analysis engine must be selected")
    return normalized


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = re.findall(r"-?\d+", value)
        if digits:
            return int(digits[0])
    return 0


def _line_split(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _tsv_rows(value: Any, columns: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in _line_split(value):
        parts = line.split("\t")
        if len(parts) < columns:
            parts.extend([""] * (columns - len(parts)))
        rows.append(parts[:columns])
    return rows


def _scala_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


async def _latest_ready_joern_source_version(
    session: AsyncSession,
    *,
    collection_id: UUID,
) -> TwinSourceVersion:
    row = (
        await session.execute(
            select(TwinSourceVersion)
            .where(
                TwinSourceVersion.collection_id == collection_id,
                TwinSourceVersion.status == "ready",
                TwinSourceVersion.joern_cpg_path.is_not(None),
            )
            .order_by(TwinSourceVersion.finished_at.desc(), TwinSourceVersion.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if not row:
        raise ValueError(
            "No ready source version with Joern CPG available for this collection. "
            "Run a refresh that generates Joern CPG artifacts first."
        )
    if not row.joern_cpg_path:
        raise ValueError("Latest ready source version has no Joern CPG path")
    return row


async def _execute_joern_query(
    session: AsyncSession,
    *,
    collection_id: UUID,
    query: str,
    timeout_seconds: int | None = None,
) -> tuple[Any, TwinSourceVersion]:
    settings = get_settings()
    source_version = await _latest_ready_joern_source_version(session, collection_id=collection_id)
    base_url = source_version.joern_server_url or settings.joern_server_url
    client = JoernClient(base_url, timeout_seconds=settings.joern_query_timeout_seconds)

    if not await client.check_health():
        raise RuntimeError(f"Joern server is not reachable at {base_url}")

    load_response = await client.load_cpg(
        source_version.joern_cpg_path,
        timeout_seconds=max(settings.joern_query_timeout_seconds, 300),
    )
    if not load_response.success:
        raise RuntimeError(f"Failed to load Joern CPG: {load_response.stderr or 'unknown error'}")

    response = await client.execute_query(
        query,
        timeout_seconds=timeout_seconds or settings.joern_query_timeout_seconds,
    )
    if not response.success:
        raise RuntimeError(f"Joern query failed: {response.stderr or 'unknown error'}")
    return parse_joern_output(response.stdout), source_version


async def joern_get_codebase_summary(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    method_count_raw, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query='println("<contextmine_result>" + cpg.method.size + "</contextmine_result>")',
    )
    call_count_raw, _ = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query='println("<contextmine_result>" + cpg.call.size + "</contextmine_result>")',
    )
    type_count_raw, _ = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query='println("<contextmine_result>" + cpg.typeDecl.size + "</contextmine_result>")',
    )
    file_count_raw, _ = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query='println("<contextmine_result>" + cpg.file.size + "</contextmine_result>")',
    )
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "scenario_version": int(scenario.version),
        "source_version_id": str(source_version.id),
        "source_id": str(source_version.source_id),
        "summary": {
            "files": _as_int(file_count_raw),
            "types": _as_int(type_count_raw),
            "methods": _as_int(method_count_raw),
            "calls": _as_int(call_count_raw),
        },
    }


async def joern_list_methods(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    query: str | None,
    page: int,
    limit: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    sample_size = max(limit * (page + 3), 50)
    output, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query=(
            'println("<contextmine_result>" + '
            f"cpg.method.take({sample_size}).map(m => "
            's"${m.name}\\t${m.fullName}\\t${m.filename}\\t${m.lineNumber.getOrElse(-1)}")'
            '.mkString("\\n") + "</contextmine_result>")'
        ),
    )
    rows = _tsv_rows(output, columns=4)
    selected_query = (query or "").strip().lower()
    if selected_query:
        rows = [
            row
            for row in rows
            if selected_query in row[0].lower() or selected_query in row[1].lower()
        ]
    total = len(rows)
    paged = rows[page * limit : page * limit + limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "source_version_id": str(source_version.id),
        "items": [
            {
                "name": row[0],
                "full_name": row[1],
                "file_path": row[2],
                "line_number": _as_int(row[3]),
            }
            for row in paged
        ],
        "page": page,
        "limit": limit,
        "total": total,
    }


async def joern_list_calls(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    page: int,
    limit: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    sample_size = max(limit * (page + 3), 80)
    output, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query=(
            'println("<contextmine_result>" + '
            f"cpg.call.take({sample_size}).map(c => "
            's"${c.name}\\t${c.code}\\t${c.filename}\\t${c.lineNumber.getOrElse(-1)}")'
            '.mkString("\\n") + "</contextmine_result>")'
        ),
    )
    rows = _tsv_rows(output, columns=4)
    total = len(rows)
    paged = rows[page * limit : page * limit + limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "source_version_id": str(source_version.id),
        "items": [
            {
                "callee": row[0],
                "code": row[1],
                "file_path": row[2],
                "line_number": _as_int(row[3]),
            }
            for row in paged
        ],
        "page": page,
        "limit": limit,
        "total": total,
    }


async def joern_get_cfg(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    depth: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    escaped = _scala_escape(node_ref)
    sample = max(40, depth * 80)
    output, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query=(
            'println("<contextmine_result>" + '
            f'cpg.method.name(".*{escaped}.*").cfgNode.code.take({sample}).l.mkString("\\n") '
            '+ "</contextmine_result>")'
        ),
    )
    nodes = _line_split(output)
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "source_version_id": str(source_version.id),
        "root": node_ref,
        "depth": depth,
        "nodes": [{"code": item} for item in nodes],
        "edges": [
            {
                "source_idx": idx,
                "target_idx": idx + 1,
                "kind": "cfg_next",
            }
            for idx in range(max(len(nodes) - 1, 0))
        ],
    }


async def joern_get_variable_flow(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    variable: str | None,
    max_hops: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    selected = variable or node_ref
    escaped = _scala_escape(selected)
    sample = max(50, max_hops * 120)
    output, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query=(
            'println("<contextmine_result>" + '
            f'cpg.identifier.name(".*{escaped}.*").take({sample}).map(i => '
            's"${i.name}\\t${i.code}\\t${i.filename}\\t${i.lineNumber.getOrElse(-1)}")'
            '.mkString("\\n") + "</contextmine_result>")'
        ),
    )
    rows = _tsv_rows(output, columns=4)
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "source_version_id": str(source_version.id),
        "variable": selected,
        "max_hops": max_hops,
        "occurrences": [
            {
                "name": row[0],
                "code": row[1],
                "file_path": row[2],
                "line_number": _as_int(row[3]),
            }
            for row in rows
        ],
    }


def _language_patterns(language: str | None, default: dict[str, list[str]]) -> list[str]:
    lang = (language or "python").lower()
    return default.get(lang, default["python"])


async def joern_find_taint_sources(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    patterns = [
        _normalize_pattern_token(item) for item in _language_patterns(language, SOURCE_PATTERNS)
    ]
    output, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query=(
            'println("<contextmine_result>" + '
            f"cpg.call.take({max(limit * 30, 120)}).map(c => "
            's"${c.name}\\t${c.code}\\t${c.filename}\\t${c.lineNumber.getOrElse(-1)}")'
            '.mkString("\\n") + "</contextmine_result>")'
        ),
    )
    rows = _tsv_rows(output, columns=4)
    matched = [row for row in rows if any(token in row[0].lower() for token in patterns)][:limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "source_version_id": str(source_version.id),
        "language": (language or "python").lower(),
        "patterns": patterns,
        "items": [
            {
                "name": row[0],
                "code": row[1],
                "file_path": row[2],
                "line_number": _as_int(row[3]),
            }
            for row in matched
        ],
        "total": len(matched),
    }


async def joern_find_taint_sinks(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    patterns = [
        _normalize_pattern_token(item) for item in _language_patterns(language, SINK_PATTERNS)
    ]
    output, source_version = await _execute_joern_query(
        session,
        collection_id=collection_id,
        query=(
            'println("<contextmine_result>" + '
            f"cpg.call.take({max(limit * 30, 120)}).map(c => "
            's"${c.name}\\t${c.code}\\t${c.filename}\\t${c.lineNumber.getOrElse(-1)}")'
            '.mkString("\\n") + "</contextmine_result>")'
        ),
    )
    rows = _tsv_rows(output, columns=4)
    matched = [row for row in rows if any(token in row[0].lower() for token in patterns)][:limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "source_version_id": str(source_version.id),
        "language": (language or "python").lower(),
        "patterns": patterns,
        "items": [
            {
                "name": row[0],
                "code": row[1],
                "file_path": row[2],
                "line_number": _as_int(row[3]),
            }
            for row in matched
        ],
        "total": len(matched),
    }


async def joern_find_taint_flows(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    max_hops: int,
    max_results: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    sources = await joern_find_taint_sources(
        session,
        collection_id=collection_id,
        scenario_id=scenario.id,
        language=language,
        limit=max_results * 2,
    )
    sinks = await joern_find_taint_sinks(
        session,
        collection_id=collection_id,
        scenario_id=scenario.id,
        language=language,
        limit=max_results * 2,
    )
    sink_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sink in sinks["items"]:
        sink_by_file[str(sink.get("file_path") or "")].append(sink)

    flows: list[dict[str, Any]] = []
    for source in sources["items"]:
        source_file = str(source.get("file_path") or "")
        candidates = sink_by_file.get(source_file, [])
        for sink in candidates:
            if len(flows) >= max_results:
                break
            flows.append(
                {
                    "source": source,
                    "sink": sink,
                    "hops": min(max_hops, 1),
                    "confidence": "medium",
                }
            )
        if len(flows) >= max_results:
            break

    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "language": (language or "python").lower(),
        "flows": flows,
        "total": len(flows),
    }


def _symbol_kind_to_name(kind: Any) -> str:
    try:
        value = int(kind)
    except (TypeError, ValueError):
        return str(kind or "unknown")
    mapping = {
        1: "file",
        2: "module",
        3: "namespace",
        4: "package",
        5: "class",
        6: "method",
        7: "property",
        8: "field",
        9: "constructor",
        10: "enum",
        11: "interface",
        12: "function",
        13: "variable",
    }
    return mapping.get(value, f"kind_{value}")


def _flatten_lsp_symbols(
    *,
    file_path: str,
    symbols: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stack = list(symbols)
    while stack:
        item = stack.pop()
        if not isinstance(item, dict):
            continue
        symbol_range = item.get("selectionRange") or item.get("range") or {}
        start = symbol_range.get("start", {}) if isinstance(symbol_range, dict) else {}
        rows.append(
            {
                "name": str(item.get("name") or ""),
                "kind_id": _as_int(item.get("kind")),
                "kind": _symbol_kind_to_name(item.get("kind")),
                "file_path": file_path,
                "line_number": _as_int(start.get("line")) + 1,
                "column": _as_int(start.get("character")),
            }
        )
        children = item.get("children")
        if isinstance(children, list):
            stack.extend(children)
    return rows


def _read_node_source_id(node: TwinNode) -> UUID | None:
    if node.source_id:
        return node.source_id
    meta = dict(node.meta or {})
    candidate = meta.get("source_id")
    if not candidate:
        return None
    try:
        return UUID(str(candidate))
    except ValueError:
        return None


def _read_node_file_path(node: TwinNode) -> str | None:
    meta = dict(node.meta or {})
    candidate = meta.get("file_path")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if node.natural_key.startswith("file:"):
        path = node.natural_key.removeprefix("file:").strip()
        return path or None
    return None


async def _collect_lsp_symbols(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    max_files: int,
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    nodes = (
        (
            await session.execute(
                select(TwinNode)
                .where(
                    TwinNode.scenario_id == scenario.id,
                    TwinNode.is_active.is_(True),
                    TwinNode.kind == "file",
                )
                .order_by(TwinNode.updated_at.desc())
                .limit(max(max_files * 20, 200))
            )
        )
        .scalars()
        .all()
    )
    settings = get_settings()
    repos_root = Path(getattr(settings, "repos_root", "/data/repos"))
    manager = get_lsp_manager()
    symbols: list[dict[str, Any]] = []
    files_scanned = 0
    errors: list[str] = []

    for node in nodes:
        source_id = _read_node_source_id(node)
        rel_path = _read_node_file_path(node)
        if source_id is None or not rel_path:
            continue

        repo_root = repos_root / str(source_id)
        if not repo_root.exists():
            continue

        candidate = Path(rel_path)
        file_path = candidate if candidate.is_absolute() else repo_root / candidate
        if not file_path.exists() or not file_path.is_file():
            continue

        try:
            client = await manager.get_client(file_path=file_path, project_root=repo_root)
            raw_symbols = await client.get_document_symbols(str(file_path))
            symbols.extend(
                _flatten_lsp_symbols(
                    file_path=str(file_path),
                    symbols=raw_symbols,
                )
            )
            files_scanned += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        if files_scanned >= max_files:
            break

    if files_scanned == 0:
        raise RuntimeError("No accessible checked-out repository files for LSP analysis")

    return {
        "scenario_id": str(scenario.id),
        "scenario_version": int(scenario.version),
        "files_scanned": files_scanned,
        "errors": errors,
        "symbols": symbols,
    }


async def lsp_get_codebase_summary(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
) -> dict[str, Any]:
    symbol_data = await _collect_lsp_symbols(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        max_files=20,
    )
    symbols = symbol_data["symbols"]
    method_count = sum(1 for row in symbols if row.get("kind_id") in LSP_METHOD_KINDS)
    class_count = sum(1 for row in symbols if row.get("kind") == "class")
    return {
        "collection_id": str(collection_id),
        "scenario_id": symbol_data["scenario_id"],
        "scenario_version": symbol_data["scenario_version"],
        "summary": {
            "symbols": len(symbols),
            "methods": method_count,
            "classes": class_count,
            "files_scanned": symbol_data["files_scanned"],
        },
        "errors": symbol_data["errors"],
    }


async def lsp_list_methods(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    query: str | None,
    page: int,
    limit: int,
) -> dict[str, Any]:
    symbol_data = await _collect_lsp_symbols(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        max_files=30,
    )
    methods = [row for row in symbol_data["symbols"] if row.get("kind_id") in LSP_METHOD_KINDS]
    selected_query = (query or "").strip().lower()
    if selected_query:
        methods = [row for row in methods if selected_query in str(row.get("name", "")).lower()]
    total = len(methods)
    paged = methods[page * limit : page * limit + limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": symbol_data["scenario_id"],
        "items": paged,
        "page": page,
        "limit": limit,
        "total": total,
        "errors": symbol_data["errors"],
    }


async def lsp_list_calls(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    page: int,
    limit: int,
) -> dict[str, Any]:
    symbol_data = await _collect_lsp_symbols(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        max_files=30,
    )
    methods = [row for row in symbol_data["symbols"] if row.get("kind_id") in LSP_METHOD_KINDS]
    # LSP does not expose a direct project-wide call graph in a uniform way; we expose
    # method symbol adjacency within each file as a deterministic baseline.
    methods_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in methods:
        methods_by_file[str(row.get("file_path") or "")].append(row)

    calls: list[dict[str, Any]] = []
    for file_path, entries in methods_by_file.items():
        ordered = sorted(entries, key=lambda x: (_as_int(x.get("line_number")), str(x.get("name"))))
        for idx in range(len(ordered) - 1):
            calls.append(
                {
                    "caller": ordered[idx].get("name"),
                    "callee": ordered[idx + 1].get("name"),
                    "file_path": file_path,
                    "kind": "lsp_symbol_adjacency",
                }
            )

    total = len(calls)
    paged = calls[page * limit : page * limit + limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": symbol_data["scenario_id"],
        "items": paged,
        "page": page,
        "limit": limit,
        "total": total,
        "errors": symbol_data["errors"],
    }


async def lsp_get_cfg(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    depth: int,
) -> dict[str, Any]:
    methods_payload = await lsp_list_methods(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        query=node_ref,
        page=0,
        limit=max(20, depth * 20),
    )
    methods = methods_payload["items"]
    if not methods:
        raise ValueError("No LSP symbols matched the requested node_ref")
    return {
        "collection_id": str(collection_id),
        "scenario_id": methods_payload["scenario_id"],
        "root": node_ref,
        "depth": depth,
        "nodes": methods,
        "edges": [
            {"source_idx": idx, "target_idx": idx + 1, "kind": "lsp_order"}
            for idx in range(max(len(methods) - 1, 0))
        ],
        "errors": methods_payload["errors"],
    }


async def lsp_get_variable_flow(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    variable: str | None,
    max_hops: int,
) -> dict[str, Any]:
    selected = (variable or node_ref).strip()
    if not selected:
        raise ValueError("variable or node_ref is required")
    symbol_data = await _collect_lsp_symbols(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        max_files=max(20, max_hops * 10),
    )
    needle = selected.lower()
    occurrences = [
        row for row in symbol_data["symbols"] if needle in str(row.get("name", "")).lower()
    ]
    return {
        "collection_id": str(collection_id),
        "scenario_id": symbol_data["scenario_id"],
        "variable": selected,
        "max_hops": max_hops,
        "occurrences": occurrences[: max(max_hops * 30, 50)],
        "errors": symbol_data["errors"],
    }


async def lsp_find_taint_sources(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
) -> dict[str, Any]:
    patterns = _language_patterns(language, SOURCE_PATTERNS)
    symbol_data = await _collect_lsp_symbols(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        max_files=30,
    )
    methods = [row for row in symbol_data["symbols"] if row.get("kind_id") in LSP_METHOD_KINDS]
    matched = [
        row
        for row in methods
        if any(
            _normalize_pattern_token(token) in str(row.get("name", "")).lower()
            for token in patterns
        )
    ][:limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": symbol_data["scenario_id"],
        "language": (language or "python").lower(),
        "patterns": patterns,
        "items": matched,
        "total": len(matched),
        "errors": symbol_data["errors"],
    }


async def lsp_find_taint_sinks(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
) -> dict[str, Any]:
    patterns = _language_patterns(language, SINK_PATTERNS)
    symbol_data = await _collect_lsp_symbols(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        max_files=30,
    )
    methods = [row for row in symbol_data["symbols"] if row.get("kind_id") in LSP_METHOD_KINDS]
    matched = [
        row
        for row in methods
        if any(
            _normalize_pattern_token(token) in str(row.get("name", "")).lower()
            for token in patterns
        )
    ][:limit]
    return {
        "collection_id": str(collection_id),
        "scenario_id": symbol_data["scenario_id"],
        "language": (language or "python").lower(),
        "patterns": patterns,
        "items": matched,
        "total": len(matched),
        "errors": symbol_data["errors"],
    }


async def lsp_find_taint_flows(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    max_hops: int,
    max_results: int,
) -> dict[str, Any]:
    sources = await lsp_find_taint_sources(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        language=language,
        limit=max_results * 2,
    )
    sinks = await lsp_find_taint_sinks(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        language=language,
        limit=max_results * 2,
    )
    sink_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sink in sinks["items"]:
        sink_by_file[str(sink.get("file_path") or "")].append(sink)

    flows: list[dict[str, Any]] = []
    for source in sources["items"]:
        source_file = str(source.get("file_path") or "")
        for sink in sink_by_file.get(source_file, []):
            if len(flows) >= max_results:
                break
            flows.append(
                {
                    "source": source,
                    "sink": sink,
                    "hops": min(max_hops, 1),
                    "confidence": "low",
                }
            )
        if len(flows) >= max_results:
            break

    return {
        "collection_id": str(collection_id),
        "scenario_id": sources["scenario_id"],
        "language": (language or "python").lower(),
        "flows": flows,
        "total": len(flows),
        "errors": sources["errors"],
    }


async def _run_multi_engine(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    projection_profile: str,
    tool_name: str,
    params: dict[str, Any],
    cache_ttl_seconds: int,
    engines: list[str] | None,
    runners: dict[str, Any],
) -> dict[str, Any]:
    scenario = await _resolve_analysis_scenario(
        session, collection_id=collection_id, scenario_id=scenario_id
    )
    selected_engines = normalize_analysis_engines(engines)
    cache_key = await _current_analysis_cache_key(
        session,
        collection_id=collection_id,
        projection_profile=f"{projection_profile}:multi",
    )
    params_hash = _hash_params(params)
    results: dict[str, Any] = {}

    for engine in selected_engines:
        cached = await _read_cached_analysis(
            session,
            scenario_id=scenario.id,
            engine=engine,
            tool_name=tool_name,
            cache_key=cache_key,
            params_hash=params_hash,
        )
        if cached is not None:
            results[engine] = cached
            continue

        runner = runners.get(engine)
        if runner is None:
            payload = {
                "engine": engine,
                "status": "failed",
                "error": f"Engine runner not configured for {tool_name}",
            }
        else:
            try:
                data = await runner()
                payload = {"engine": engine, "status": "ready", "data": data}
            except Exception as exc:  # noqa: BLE001
                payload = {"engine": engine, "status": "failed", "error": str(exc)}

        await _write_cached_analysis(
            session,
            scenario_id=scenario.id,
            engine=engine,
            tool_name=tool_name,
            cache_key=cache_key,
            params_hash=params_hash,
            payload=payload,
            ttl_seconds=cache_ttl_seconds,
        )
        results[engine] = payload

    return {
        "collection_id": str(collection_id),
        "scenario_id": str(scenario.id),
        "scenario_version": int(scenario.version),
        "engines": results,
    }


async def get_codebase_summary_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="summary",
        tool_name="summary",
        params={},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: get_codebase_summary(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_get_codebase_summary(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
            ),
            "joern": lambda: joern_get_codebase_summary(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
            ),
        },
    )


async def list_methods_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    query: str | None,
    page: int,
    limit: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="methods",
        tool_name="methods",
        params={"query": query or "", "page": page, "limit": limit},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: list_methods(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                query=query,
                page=page,
                limit=limit,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_list_methods(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                query=query,
                page=page,
                limit=limit,
            ),
            "joern": lambda: joern_list_methods(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                query=query,
                page=page,
                limit=limit,
            ),
        },
    )


async def list_calls_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    page: int,
    limit: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="calls",
        tool_name="calls",
        params={"page": page, "limit": limit},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: list_calls(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                page=page,
                limit=limit,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_list_calls(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                page=page,
                limit=limit,
            ),
            "joern": lambda: joern_list_calls(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                page=page,
                limit=limit,
            ),
        },
    )


async def get_cfg_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    depth: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="cfg",
        tool_name="cfg",
        params={"node_ref": node_ref, "depth": depth},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: get_cfg(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                node_ref=node_ref,
                depth=depth,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_get_cfg(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                node_ref=node_ref,
                depth=depth,
            ),
            "joern": lambda: joern_get_cfg(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                node_ref=node_ref,
                depth=depth,
            ),
        },
    )


async def get_variable_flow_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    node_ref: str,
    variable: str | None,
    max_hops: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="variable_flow",
        tool_name="variable_flow",
        params={"node_ref": node_ref, "variable": variable or "", "max_hops": max_hops},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: get_variable_flow(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                node_ref=node_ref,
                variable=variable,
                max_hops=max_hops,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_get_variable_flow(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                node_ref=node_ref,
                variable=variable,
                max_hops=max_hops,
            ),
            "joern": lambda: joern_get_variable_flow(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                node_ref=node_ref,
                variable=variable,
                max_hops=max_hops,
            ),
        },
    )


async def find_taint_sources_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="taint_sources",
        tool_name="taint_sources",
        params={"language": language or "python", "limit": limit},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: find_taint_sources(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                limit=limit,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_find_taint_sources(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                limit=limit,
            ),
            "joern": lambda: joern_find_taint_sources(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                limit=limit,
            ),
        },
    )


async def find_taint_sinks_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    limit: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="taint_sinks",
        tool_name="taint_sinks",
        params={"language": language or "python", "limit": limit},
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: find_taint_sinks(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                limit=limit,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_find_taint_sinks(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                limit=limit,
            ),
            "joern": lambda: joern_find_taint_sinks(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                limit=limit,
            ),
        },
    )


async def find_taint_flows_multi(
    session: AsyncSession,
    *,
    collection_id: UUID,
    scenario_id: UUID | None,
    language: str | None,
    max_hops: int,
    max_results: int,
    cache_ttl_seconds: int,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    return await _run_multi_engine(
        session,
        collection_id=collection_id,
        scenario_id=scenario_id,
        projection_profile="taint_flows",
        tool_name="taint_flows",
        params={
            "language": language or "python",
            "max_hops": max_hops,
            "max_results": max_results,
        },
        cache_ttl_seconds=cache_ttl_seconds,
        engines=engines,
        runners={
            "graphrag": lambda: find_taint_flows(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                max_hops=max_hops,
                max_results=max_results,
                cache_ttl_seconds=cache_ttl_seconds,
            ),
            "lsp": lambda: lsp_find_taint_flows(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                max_hops=max_hops,
                max_results=max_results,
            ),
            "joern": lambda: joern_find_taint_flows(
                session,
                collection_id=collection_id,
                scenario_id=scenario_id,
                language=language,
                max_hops=max_hops,
                max_results=max_results,
            ),
        },
    )
