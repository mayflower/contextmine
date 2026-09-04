"""Prefect orchestration routes backed by the official async client."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from contextmine_core import SyncRun, SyncRunStatus, get_session, get_settings
from fastapi import APIRouter, HTTPException
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.filters import TaskRunFilter, TaskRunFilterFlowRunId
from prefect.client.schemas.sorting import FlowRunSort
from prefect.deployments import run_deployment
from prefect.exceptions import ObjectNotFound
from prefect.states import Cancelling
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prefect"])


class FlowRunProgressResponse(BaseModel):
    total: int
    completed: int
    failed: int
    running: int
    pending: int
    current_task: str | None = None
    percent: int


class FlowRunParametersResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_id: str | None = None
    source_url: str | None = None
    sync_run_id: str | None = None


class FlowRunResponse(BaseModel):
    id: str
    name: str
    flow_id: str
    flow_name: str
    state_type: str
    state_name: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    parameters: FlowRunParametersResponse
    total_run_time: float
    progress: FlowRunProgressResponse | None = None


class FlowRunsResponse(BaseModel):
    active: list[FlowRunResponse] = Field(default_factory=list)
    recent: list[FlowRunResponse] = Field(default_factory=list)
    error: str | None = None


async def start_source_sync(source_id: str, source_url: str, sync_run_id: str) -> str:
    """Start the configured single-source deployment idempotently."""
    deployment = get_settings().prefect_sync_deployment
    try:
        flow_run = await run_deployment(
            deployment,
            parameters={
                "source_id": source_id,
                "source_url": source_url,
                "sync_run_id": sync_run_id,
            },
            timeout=0,
            as_subflow=False,
            idempotency_key=sync_run_id,
        )
    except Exception as exc:
        logger.exception("Prefect scheduling failed for deployment %s", deployment)
        if isinstance(exc, ObjectNotFound):
            raise RuntimeError(
                f"Prefect deployment '{deployment}' not found. "
                "Check PREFECT_API_URL and that the worker registered its deployments."
            ) from exc
        raise
    return str(flow_run.id)


# Prefect states from which a flow run never leaves. A sync run still marked
# active behind one of these is stale and must not keep blocking the source.
_TERMINAL_FLOW_RUN_STATES = frozenset({"COMPLETED", "FAILED", "CRASHED", "CANCELLED"})

# Reported when Prefect no longer knows the flow run at all, e.g. after its
# database was reset. The sync run behind it can never complete either.
FLOW_RUN_MISSING = "MISSING"


async def terminal_flow_run_state(flow_run_id: str) -> str | None:
    """Return the flow run's state when Prefect considers it finished.

    Returns None while the run may still be in flight - including when Prefect
    cannot be reached - so callers keep waiting rather than releasing a sync
    that is genuinely still running.
    """
    try:
        run_uuid = uuid.UUID(flow_run_id)
    except ValueError:
        return FLOW_RUN_MISSING

    try:
        async with get_client() as client:
            flow_run = await client.read_flow_run(run_uuid)
    except ObjectNotFound:
        return FLOW_RUN_MISSING
    except Exception:
        logger.warning("Could not read flow run %s from Prefect", flow_run_id, exc_info=True)
        return None

    state_type = getattr(flow_run.state_type, "value", None) or str(flow_run.state_type or "")
    return state_type if state_type in _TERMINAL_FLOW_RUN_STATES else None


async def _get_flow_run_progress(
    client: PrefectClient,
    flow_run_id: uuid.UUID,
) -> dict[str, Any]:
    task_runs = await client.read_task_runs(
        task_run_filter=TaskRunFilter(flow_run_id=TaskRunFilterFlowRunId(any_=[flow_run_id]))
    )
    state_types = [str(task.state_type.value) if task.state_type else "" for task in task_runs]
    completed = state_types.count("COMPLETED")
    running = state_types.count("RUNNING")
    failed = state_types.count("FAILED")
    pending = state_types.count("PENDING") + state_types.count("SCHEDULED")
    current_task = next(
        (task.name for task in task_runs if task.state_type and task.state_type.value == "RUNNING"),
        None,
    )
    total = len(task_runs)
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "running": running,
        "pending": pending,
        "current_task": current_task,
        "percent": round((completed / total) * 100) if total else 0,
    }


@router.get("/prefect/flow-runs", response_model=FlowRunsResponse)
async def get_flow_runs() -> FlowRunsResponse:
    """Return active and recent Prefect runs using supported client models."""
    try:
        async with get_client() as client:
            flow_runs = await client.read_flow_runs(
                sort=FlowRunSort.START_TIME_DESC,
                limit=50,
            )
            flow_names: dict[uuid.UUID, str] = {}
            for flow_id in {run.flow_id for run in flow_runs}:
                flow_names[flow_id] = (await client.read_flow(flow_id)).name

            active: list[dict[str, Any]] = []
            recent: list[dict[str, Any]] = []
            for run in flow_runs:
                state_type = run.state_type.value if run.state_type else ""
                item: dict[str, Any] = {
                    "id": str(run.id),
                    "name": run.name,
                    "flow_id": str(run.flow_id),
                    "flow_name": flow_names.get(run.flow_id, ""),
                    "state_type": state_type,
                    "state_name": run.state_name,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "parameters": run.parameters,
                    "total_run_time": (
                        run.total_run_time.total_seconds() if run.total_run_time else 0
                    ),
                }
                if state_type in {"RUNNING", "PENDING", "SCHEDULED"}:
                    item["progress"] = await _get_flow_run_progress(client, run.id)
                    active.append(item)
                else:
                    recent.append(item)
            return FlowRunsResponse(active=active, recent=recent[:20])
    except Exception as e:
        return FlowRunsResponse(error=str(e))


@router.post("/prefect/flow-runs/{flow_run_id}/cancel")
async def cancel_flow_run(flow_run_id: uuid.UUID) -> dict[str, str]:
    """Request cancellation and mirror the business run's terminal state."""
    try:
        async with get_client() as client:
            await client.set_flow_run_state(flow_run_id, Cancelling())
    except ObjectNotFound as e:
        raise HTTPException(status_code=404, detail="Prefect flow run not found") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail="Prefect cancellation failed") from e

    async with get_session() as session:
        sync_run = (
            await session.execute(select(SyncRun).where(SyncRun.flow_run_id == str(flow_run_id)))
        ).scalar_one_or_none()
        if sync_run is not None and sync_run.status in {
            SyncRunStatus.SCHEDULED,
            SyncRunStatus.RUNNING,
        }:
            from datetime import UTC, datetime

            sync_run.status = SyncRunStatus.CANCELLED
            sync_run.finished_at = datetime.now(UTC)
            sync_run.error = "Cancellation requested through Prefect"
            await session.commit()
    return {"flow_run_id": str(flow_run_id), "status": "cancelling"}


@router.get("/prefect/health")
async def prefect_health() -> dict[str, str]:
    """Check Prefect server connectivity through the supported client."""
    try:
        async with get_client() as client:
            error = await client.api_healthcheck()
            if error is not None:
                raise error
        return {"prefect": "ok"}
    except Exception as e:
        return {"prefect": "error", "detail": str(e) or type(e).__name__}
