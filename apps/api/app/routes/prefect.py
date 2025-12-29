"""Prefect API proxy routes for flow run status and progress."""

import httpx
from contextmine_core import get_settings
from fastapi import APIRouter

router = APIRouter(tags=["prefect"])


@router.get("/prefect/flow-runs")
async def get_flow_runs() -> dict:
    """Get flow runs from Prefect with their status and progress.

    Only shows RUNNING/PENDING runs as "active" (not SCHEDULED, which are
    just pre-created by Prefect's scheduler).

    Filters out sync_due_sources scheduler runs (empty polling runs).
    """
    settings = get_settings()
    prefect_url = settings.prefect_api_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # First, get flow IDs to filter (we want to exclude sync_due_sources)
            flow_id_to_name = await _get_flow_names(client, prefect_url)

            # Get recently started/completed runs (not scheduled)
            response = await client.post(
                f"{prefect_url}/flow_runs/filter",
                json={
                    "limit": 50,
                    "sort": "START_TIME_DESC",
                    "flow_runs": {
                        "state": {
                            "type": {
                                "any_": [
                                    "RUNNING",
                                    "PENDING",
                                    "COMPLETED",
                                    "FAILED",
                                    "CANCELLED",
                                ]
                            }
                        }
                    },
                },
            )
            response.raise_for_status()
            all_runs = response.json()

            # Separate active and recent runs
            active_runs = []
            recent_runs = []

            for run in all_runs:
                flow_id = run.get("flow_id")
                flow_name = flow_id_to_name.get(flow_id, "")

                # Skip sync_due_sources scheduler runs (polling runs with no actual work)
                if flow_name == "sync_due_sources":
                    continue

                state_type = run.get("state_type", "")
                run_data = {
                    "id": run.get("id"),
                    "name": run.get("name"),
                    "flow_id": flow_id,
                    "flow_name": flow_name,
                    "state_type": state_type,
                    "state_name": run.get("state_name"),
                    "start_time": run.get("start_time"),
                    "end_time": run.get("end_time"),
                    "parameters": run.get("parameters", {}),
                    "total_run_time": run.get("total_run_time"),
                }

                # Only RUNNING and PENDING are truly "active"
                if state_type in ("RUNNING", "PENDING"):
                    active_runs.append(run_data)
                else:
                    recent_runs.append(run_data)

            # For active runs, get task run progress
            for run in active_runs:
                run["progress"] = await _get_flow_run_progress(client, prefect_url, run["id"])

            return {
                "active": active_runs,
                "recent": recent_runs[:20],  # Limit recent to 20
            }

        except httpx.HTTPError as e:
            return {"error": f"Failed to connect to Prefect: {e}", "active": [], "recent": []}
        except Exception as e:
            return {"error": str(e), "active": [], "recent": []}


async def _get_flow_names(client: httpx.AsyncClient, prefect_url: str) -> dict[str, str]:
    """Get mapping of flow_id to flow_name."""
    try:
        response = await client.post(
            f"{prefect_url}/flows/filter",
            json={"limit": 100},
        )
        response.raise_for_status()
        flows = response.json()
        return {flow["id"]: flow["name"] for flow in flows}
    except Exception:
        return {}


async def _get_flow_run_progress(
    client: httpx.AsyncClient, prefect_url: str, flow_run_id: str
) -> dict:
    """Get task run progress for a flow run."""
    try:
        response = await client.post(
            f"{prefect_url}/task_runs/filter",
            json={
                "flow_runs": {"id": {"any_": [flow_run_id]}},
                "limit": 100,
            },
        )
        response.raise_for_status()
        task_runs = response.json()

        total = len(task_runs)
        completed = sum(1 for t in task_runs if t.get("state_type") == "COMPLETED")
        failed = sum(1 for t in task_runs if t.get("state_type") == "FAILED")
        running = sum(1 for t in task_runs if t.get("state_type") == "RUNNING")
        pending = sum(1 for t in task_runs if t.get("state_type") in ("PENDING", "SCHEDULED"))

        # Get current task if any
        current_task = None
        for t in task_runs:
            if t.get("state_type") == "RUNNING":
                current_task = t.get("name")
                break

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "current_task": current_task,
            "percent": round((completed / total) * 100) if total > 0 else 0,
        }
    except Exception:
        return {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "current_task": None,
            "percent": 0,
        }


@router.get("/prefect/health")
async def prefect_health() -> dict:
    """Check Prefect server health."""
    settings = get_settings()
    prefect_url = settings.prefect_api_url

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get(f"{prefect_url}/health")
            response.raise_for_status()
            return {"prefect": "ok"}
        except Exception as e:
            return {"prefect": "error", "detail": str(e)}
