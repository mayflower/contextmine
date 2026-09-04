"""Tests for miscellaneous API routes: prefect, db, validation endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from app.routes.db import db_health_check, get_stats
from app.routes.prefect import (
    _get_flow_run_progress,
    get_flow_runs,
    prefect_health,
    start_source_sync,
)
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas.objects import StateType
from prefect.deployments import arun_deployment
from prefect.exceptions import ObjectNotFound

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Prefect routes
# ---------------------------------------------------------------------------


class TestGetFlowRunProgress:
    @pytest.mark.anyio
    async def test_returns_progress_dict(self) -> None:
        client = AsyncMock()
        client.read_task_runs.return_value = [
            SimpleNamespace(state_type=StateType.COMPLETED, name="task1"),
            SimpleNamespace(state_type=StateType.RUNNING, name="task2"),
            SimpleNamespace(state_type=StateType.PENDING, name="task3"),
        ]

        result = await _get_flow_run_progress(client, uuid.uuid4())
        assert result["total"] == 3
        assert result["completed"] == 1
        assert result["running"] == 1
        assert result["pending"] == 1
        assert result["current_task"] == "task2"
        assert result["percent"] == 33


class TestGetFlowRuns:
    @pytest.mark.anyio
    async def test_returns_active_and_recent(self) -> None:
        flow_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_client = AsyncMock()
        mock_client.read_flow_runs.return_value = [
            SimpleNamespace(
                id=uuid.uuid4(),
                name="run-1",
                flow_id=flow_id,
                state_type=StateType.RUNNING,
                state_name="Running",
                start_time=now,
                end_time=None,
                parameters={},
                total_run_time=timedelta(seconds=10),
            ),
            SimpleNamespace(
                id=uuid.uuid4(),
                name="run-2",
                flow_id=flow_id,
                state_type=StateType.COMPLETED,
                state_name="Completed",
                start_time=now,
                end_time=now,
                parameters={},
                total_run_time=timedelta(seconds=60),
            ),
        ]
        mock_client.read_flow.return_value = SimpleNamespace(name="sync_single_source")
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=mock_client)
        client_context.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.routes.prefect.get_client", return_value=client_context),
            patch(
                "app.routes.prefect._get_flow_run_progress",
                new=AsyncMock(
                    return_value={
                        "total": 5,
                        "completed": 2,
                        "failed": 0,
                        "running": 1,
                        "pending": 2,
                        "current_task": "extract",
                        "percent": 40,
                    }
                ),
            ),
        ):
            result = await get_flow_runs()

        assert len(result.active) == 1
        assert len(result.recent) == 1
        assert result.active[0].flow_name == "sync_single_source"


class TestPrefectHealth:
    @pytest.mark.anyio
    async def test_healthy(self) -> None:
        mock_client = AsyncMock()
        mock_client.api_healthcheck.return_value = None
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=mock_client)
        client_context.__aexit__ = AsyncMock(return_value=False)
        with patch("app.routes.prefect.get_client", return_value=client_context):
            result = await prefect_health()
        assert result == {"prefect": "ok"}

    @pytest.mark.anyio
    async def test_unhealthy(self) -> None:
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(side_effect=RuntimeError("refused"))
        client_context.__aexit__ = AsyncMock(return_value=False)
        with patch("app.routes.prefect.get_client", return_value=client_context):
            result = await prefect_health()
        assert result["prefect"] == "error"

    @pytest.mark.parametrize("error", [ConnectionError("refused"), TimeoutError()])
    async def test_returned_error_is_unhealthy(self, error: Exception) -> None:
        mock_client = AsyncMock()
        mock_client.api_healthcheck.return_value = error
        mock_client.__aenter__.return_value = mock_client
        with patch("app.routes.prefect.get_client", return_value=mock_client):
            result = await prefect_health()
        assert result == {"prefect": "error", "detail": str(error) or type(error).__name__}


async def test_missing_deployment_reports_name_and_logs_http_cause(caplog) -> None:
    def missing(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/deployments/name/sync_single_source/default"
        return httpx.Response(404, json={"detail": "Deployment not found"})

    async with PrefectClient(
        "http://prefect.test/api",
        httpx_settings={"transport": httpx.MockTransport(missing)},
    ) as client:
        with (
            patch("app.routes.prefect.run_deployment", partial(arun_deployment, client=client)),
            patch(
                "app.routes.prefect.get_settings",
                return_value=SimpleNamespace(prefect_sync_deployment="sync_single_source/default"),
            ),
            pytest.raises(
                RuntimeError, match="deployment 'sync_single_source/default' not found"
            ) as raised,
        ):
            await start_source_sync("source-id", "https://example.test/repo", "sync-run-id")

    assert "PREFECT_API_URL" in str(raised.value)
    assert isinstance(raised.value.__cause__, ObjectNotFound)
    assert raised.value.__cause__.http_exc.response.status_code == 404
    assert any(record.exc_info for record in caplog.records)


# ---------------------------------------------------------------------------
# DB routes
# ---------------------------------------------------------------------------


class TestDbRoutes:
    @pytest.mark.anyio
    async def test_db_health_not_configured(self) -> None:
        mock_settings = MagicMock()
        mock_settings.database_url = ""

        with patch("app.routes.db.get_settings", return_value=mock_settings):
            result = await db_health_check()
            assert result == {"db": "not_configured"}

    @pytest.mark.anyio
    async def test_stats_not_configured(self) -> None:
        mock_settings = MagicMock()
        mock_settings.database_url = ""

        with patch("app.routes.db.get_settings", return_value=mock_settings):
            result = await get_stats()
            assert result.error == "database_not_configured"
