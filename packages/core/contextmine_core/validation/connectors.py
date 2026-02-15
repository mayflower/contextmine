"""Connectors for Tekton/Argo/Temporal validation signals."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ValidationMetric:
    source: str
    key: str
    value: float
    status: str | None = None
    meta: dict[str, Any] | None = None


async def fetch_argo_metrics() -> list[ValidationMetric]:
    """Fetch pass-rate style metrics from Argo API."""
    url = os.getenv("ARGO_API_URL")
    if not url:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    workflows = payload.get("items", []) if isinstance(payload, dict) else []
    total = len(workflows)
    succeeded = sum(1 for w in workflows if _state_of(w) in {"Succeeded", "succeeded"})
    rate = float(succeeded / total) if total else 0.0
    return [
        ValidationMetric(
            source="argo",
            key="pass_rate",
            value=rate,
            status="ok" if rate >= 0.8 else "degraded",
            meta={"total": total, "succeeded": succeeded},
        )
    ]


async def fetch_tekton_metrics() -> list[ValidationMetric]:
    """Fetch pass-rate style metrics from Tekton API."""
    url = os.getenv("TEKTON_API_URL")
    if not url:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    runs = payload.get("items", []) if isinstance(payload, dict) else []
    total = len(runs)
    succeeded = sum(1 for run in runs if _state_of(run) in {"True", "Succeeded", "success"})
    rate = float(succeeded / total) if total else 0.0
    return [
        ValidationMetric(
            source="tekton",
            key="pass_rate",
            value=rate,
            status="ok" if rate >= 0.8 else "degraded",
            meta={"total": total, "succeeded": succeeded},
        )
    ]


async def fetch_temporal_alerts() -> list[ValidationMetric]:
    """Fetch alert-like signals from Temporal API."""
    url = os.getenv("TEMPORAL_API_URL")
    if not url:
        return []
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    alerts = payload.get("alerts", []) if isinstance(payload, dict) else []
    open_alerts = [a for a in alerts if str(a.get("state", "")).lower() != "resolved"]
    return [
        ValidationMetric(
            source="temporal",
            key="open_alerts",
            value=float(len(open_alerts)),
            status="alert" if open_alerts else "ok",
            meta={"alerts": open_alerts[:50]},
        )
    ]


def _state_of(item: dict[str, Any]) -> str:
    status = item.get("status") if isinstance(item, dict) else None
    if isinstance(status, dict):
        for key in ("phase", "condition", "state"):
            value = status.get(key)
            if value:
                return str(value)
    return str(status or "")
