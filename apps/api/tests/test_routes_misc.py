"""Tests for miscellaneous API routes: prefect, db, validation endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.routes.db import db_health_check, get_stats
from app.routes.prefect import (
    _get_flow_run_progress,
    get_flow_runs,
    prefect_health,
)
from prefect.client.schemas.objects import StateType

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
