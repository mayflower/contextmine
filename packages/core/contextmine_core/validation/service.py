"""Persistence service for validation snapshots."""

from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from contextmine_core.models import ValidationSnapshot, ValidationSourceKind
from contextmine_core.validation.connectors import (
    ValidationMetric,
    fetch_argo_metrics,
    fetch_tekton_metrics,
    fetch_temporal_alerts,
)
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession


def _to_kind(source: str) -> ValidationSourceKind:
    return ValidationSourceKind(source)


async def refresh_validation_snapshots(
    session: AsyncSession,
    collection_id: UUID | None,
) -> int:
    """Fetch current external validation metrics and persist them."""
    metrics: list[ValidationMetric] = []
    metrics.extend(await fetch_argo_metrics())
    metrics.extend(await fetch_tekton_metrics())
    metrics.extend(await fetch_temporal_alerts())

    for metric in metrics:
        session.add(
            ValidationSnapshot(
                id=uuid.uuid4(),
                collection_id=collection_id,
                source_kind=_to_kind(metric.source),
                metric_key=metric.key,
                metric_value=metric.value,
                status=metric.status,
                meta=metric.meta or {},
            )
        )

    return len(metrics)


async def get_latest_validation_status(
    session: AsyncSession,
    collection_id: UUID | None,
) -> dict[str, Any]:
    """Get latest snapshots grouped by source."""
    stmt = select(ValidationSnapshot).order_by(desc(ValidationSnapshot.captured_at)).limit(200)
    if collection_id is not None:
        stmt = stmt.where(ValidationSnapshot.collection_id == collection_id)

    rows = (await session.execute(stmt)).scalars().all()
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = row.source_kind.value
        if key not in grouped:
            grouped[key] = {"source": key, "metrics": {}}
        metric_key = row.metric_key
        if metric_key not in grouped[key]["metrics"]:
            grouped[key]["metrics"][metric_key] = {
                "value": row.metric_value,
                "status": row.status,
                "captured_at": row.captured_at.isoformat(),
                "meta": row.meta,
            }

    return {"sources": list(grouped.values())}
