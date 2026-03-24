"""Tests for miscellaneous API routes: prefect, db, validation endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from app.routes.db import db_health_check, get_stats
from app.routes.prefect import (
    _get_flow_names,
    _get_flow_run_progress,
    get_flow_runs,
    prefect_health,
)

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Prefect routes
# ---------------------------------------------------------------------------


class TestGetFlowNames:
    @pytest.mark.anyio
    async def test_returns_flow_id_to_name_map(self) -> None:
        client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "f1", "name": "sync_source"},
            {"id": "f2", "name": "sync_due_sources"},
        ]
        mock_response.raise_for_status = MagicMock()
        client.post.return_value = mock_response

        result = await _get_flow_names(client, "http://prefect:4200/api")
        assert result == {"f1": "sync_source", "f2": "sync_due_sources"}

    @pytest.mark.anyio
    async def test_returns_empty_on_error(self) -> None:
        client = AsyncMock()
        client.post.side_effect = Exception("connection refused")

        result = await _get_flow_names(client, "http://prefect:4200/api")
        assert result == {}


class TestGetFlowRunProgress:
    @pytest.mark.anyio
    async def test_returns_progress_dict(self) -> None:
        client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"state_type": "COMPLETED", "name": "task1"},
            {"state_type": "RUNNING", "name": "task2"},
            {"state_type": "PENDING", "name": "task3"},
        ]
        mock_response.raise_for_status = MagicMock()
        client.post.return_value = mock_response

        result = await _get_flow_run_progress(client, "http://prefect:4200/api", "run-123")
        assert result["total"] == 3
        assert result["completed"] == 1
        assert result["running"] == 1
        assert result["pending"] == 1
        assert result["current_task"] == "task2"
        assert result["percent"] == 33

    @pytest.mark.anyio
    async def test_returns_zeros_on_error(self) -> None:
        client = AsyncMock()
        client.post.side_effect = Exception("fail")

        result = await _get_flow_run_progress(client, "http://prefect:4200/api", "run-123")
        assert result["total"] == 0
        assert result["percent"] == 0


class TestGetFlowRuns:
    @pytest.mark.anyio
    async def test_returns_active_and_recent(self) -> None:
        mock_settings = MagicMock()
        mock_settings.prefect_api_url = "http://prefect:4200/api"

        with (
            patch("app.routes.prefect.get_settings", return_value=mock_settings),
            patch(
                "app.routes.prefect._get_flow_names",
                new_callable=AsyncMock,
                return_value={"f1": "sync_source"},
            ),
        ):
            # We need to mock httpx.AsyncClient
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "id": "r1",
                    "name": "run-1",
                    "flow_id": "f1",
                    "state_type": "RUNNING",
                    "state_name": "Running",
                    "start_time": "2024-01-01T00:00:00",
                    "end_time": None,
                    "parameters": {},
                    "total_run_time": 10,
                },
                {
                    "id": "r2",
                    "name": "run-2",
                    "flow_id": "f1",
                    "state_type": "COMPLETED",
                    "state_name": "Completed",
                    "start_time": "2024-01-01T00:00:00",
                    "end_time": "2024-01-01T00:01:00",
                    "parameters": {},
                    "total_run_time": 60,
                },
            ]
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            with patch("app.routes.prefect.httpx.AsyncClient") as mock_ac_cls:
                mock_ac_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_ac_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                with patch(
                    "app.routes.prefect._get_flow_run_progress",
                    new_callable=AsyncMock,
                    return_value={
                        "total": 5,
                        "completed": 2,
                        "failed": 0,
                        "running": 1,
                        "pending": 2,
                        "current_task": "chunking",
                        "percent": 40,
                    },
                ):
                    result = await get_flow_runs()
                    assert "active" in result
                    assert "recent" in result
                    assert len(result["active"]) == 1
                    assert len(result["recent"]) == 1

    @pytest.mark.anyio
    async def test_filters_sync_due_sources(self) -> None:
        """Scheduler runs (sync_due_sources) should be filtered out."""
        mock_settings = MagicMock()
        mock_settings.prefect_api_url = "http://prefect:4200/api"

        with patch("app.routes.prefect.get_settings", return_value=mock_settings):
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = [
                {
                    "id": "r1",
                    "name": "scheduler-run",
                    "flow_id": "f2",
                    "state_type": "COMPLETED",
                    "state_name": "Completed",
                    "start_time": None,
                    "end_time": None,
                    "parameters": {},
                    "total_run_time": 0,
                },
            ]
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            with patch("app.routes.prefect.httpx.AsyncClient") as mock_ac_cls:
                mock_ac_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_ac_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                with patch(
                    "app.routes.prefect._get_flow_names",
                    new_callable=AsyncMock,
                    return_value={"f2": "sync_due_sources"},
                ):
                    result = await get_flow_runs()
                    assert len(result["active"]) == 0
                    assert len(result["recent"]) == 0


class TestPrefectHealth:
    @pytest.mark.anyio
    async def test_healthy(self) -> None:
        mock_settings = MagicMock()
        mock_settings.prefect_api_url = "http://prefect:4200/api"

        with patch("app.routes.prefect.get_settings", return_value=mock_settings):
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response

            with patch("app.routes.prefect.httpx.AsyncClient") as mock_ac_cls:
                mock_ac_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_ac_cls.return_value.__aexit__ = AsyncMock(return_value=False)

                result = await prefect_health()
                assert result == {"prefect": "ok"}

    @pytest.mark.anyio
    async def test_unhealthy(self) -> None:
        mock_settings = MagicMock()
        mock_settings.prefect_api_url = "http://prefect:4200/api"

        with patch("app.routes.prefect.get_settings", return_value=mock_settings):
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")

            with patch("app.routes.prefect.httpx.AsyncClient") as mock_ac_cls:
                mock_ac_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_ac_cls.return_value.__aexit__ = AsyncMock(return_value=False)

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
            assert result == {"error": "database_not_configured"}
