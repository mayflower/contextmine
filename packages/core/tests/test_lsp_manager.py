"""Tests for LSP Manager pool management, caching, and lifecycle.

Tests the LspManager singleton, CachedServer, mock client injection,
get_client routing, cleanup of idle servers, and shutdown behavior
using mocked LSP clients.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.lsp.exceptions import LspNotAvailableError
from contextmine_core.lsp.languages import SupportedLanguage
from contextmine_core.lsp.manager import (
    CachedServer,
    LspManager,
    get_lsp_manager,
)

# ---------------------------------------------------------------------------
# CachedServer dataclass
# ---------------------------------------------------------------------------


class TestCachedServer:
    def test_creation(self) -> None:
        client = MagicMock()
        server = CachedServer(
            client=client,
            language=SupportedLanguage.PYTHON,
            project_root=Path("/tmp/project"),
        )
        assert server.client is client
        assert server.language == SupportedLanguage.PYTHON
        assert server.project_root == Path("/tmp/project")
        assert isinstance(server.last_used, datetime)

    def test_touch_updates_timestamp(self) -> None:
        client = MagicMock()
        server = CachedServer(
            client=client,
            language=SupportedLanguage.PYTHON,
            project_root=Path("/tmp/project"),
        )
        old_time = server.last_used
        # Force a tiny sleep to ensure timestamp differs
        import time

        time.sleep(0.01)
        server.touch()
        assert server.last_used >= old_time


# ---------------------------------------------------------------------------
# LspManager singleton
# ---------------------------------------------------------------------------


class TestLspManagerSingleton:
    def setup_method(self) -> None:
        LspManager.reset()

    def teardown_method(self) -> None:
        LspManager.reset()

    def test_get_instance_creates_singleton(self) -> None:
        instance = LspManager.get_instance()
        assert instance is not None
        assert LspManager.get_instance() is instance

    def test_reset_clears_singleton(self) -> None:
        instance = LspManager.get_instance()
        LspManager.reset()
        new_instance = LspManager.get_instance()
        assert new_instance is not instance

    def test_get_lsp_manager_helper(self) -> None:
        instance = get_lsp_manager()
        assert isinstance(instance, LspManager)


# ---------------------------------------------------------------------------
# Mock client injection
# ---------------------------------------------------------------------------


class TestMockClientInjection:
    def setup_method(self) -> None:
        LspManager.reset()

    def teardown_method(self) -> None:
        LspManager.reset()

    @pytest.mark.anyio
    async def test_set_mock_client_returns_mock(self) -> None:
        from contextmine_core.lsp.client import MockLspClient

        manager = LspManager()
        mock_client = MockLspClient()
        manager.set_mock_client(mock_client)

        result = await manager.get_client("/tmp/test.py")
        assert result is mock_client


# ---------------------------------------------------------------------------
# get_client routing
# ---------------------------------------------------------------------------


class TestGetClient:
    def setup_method(self) -> None:
        LspManager.reset()

    def teardown_method(self) -> None:
        LspManager.reset()

    @pytest.mark.anyio
    async def test_unsupported_file_type_raises(self) -> None:
        manager = LspManager()
        with pytest.raises(LspNotAvailableError, match="Unsupported file type"):
            await manager.get_client("/tmp/file.xyz_unknown_ext")

    @pytest.mark.anyio
    async def test_cached_server_returned(self) -> None:
        manager = LspManager()
        mock_client = MagicMock()

        key = (SupportedLanguage.PYTHON, Path("/tmp/project").resolve())
        manager._servers[key] = CachedServer(
            client=mock_client,
            language=SupportedLanguage.PYTHON,
            project_root=Path("/tmp/project").resolve(),
        )

        with (
            patch(
                "contextmine_core.lsp.manager.detect_language",
                return_value=SupportedLanguage.PYTHON,
            ),
            patch(
                "contextmine_core.lsp.manager.find_project_root",
                return_value=Path("/tmp/project").resolve(),
            ),
        ):
            result = await manager.get_client("/tmp/project/main.py")

        assert result is mock_client

    @pytest.mark.anyio
    async def test_new_server_started_when_not_cached(self) -> None:
        manager = LspManager()
        mock_client = MagicMock()

        with (
            patch(
                "contextmine_core.lsp.manager.detect_language",
                return_value=SupportedLanguage.PYTHON,
            ),
            patch(
                "contextmine_core.lsp.manager.find_project_root",
                return_value=Path("/tmp/project").resolve(),
            ),
            patch.object(
                manager, "_start_server", new_callable=AsyncMock, return_value=mock_client
            ),
            patch.object(manager, "_ensure_cleanup_task"),
        ):
            result = await manager.get_client("/tmp/project/main.py")

        assert result is mock_client
        assert len(manager._servers) == 1

    @pytest.mark.anyio
    async def test_explicit_project_root(self) -> None:
        manager = LspManager()
        mock_client = MagicMock()

        with (
            patch(
                "contextmine_core.lsp.manager.detect_language",
                return_value=SupportedLanguage.PYTHON,
            ),
            patch.object(
                manager, "_start_server", new_callable=AsyncMock, return_value=mock_client
            ),
            patch.object(manager, "_ensure_cleanup_task"),
        ):
            result = await manager.get_client("/tmp/project/main.py", project_root="/tmp/project")

        assert result is mock_client


# ---------------------------------------------------------------------------
# _start_server
# ---------------------------------------------------------------------------


class TestStartServer:
    @pytest.mark.anyio
    async def test_multilspy_not_installed(self) -> None:
        manager = LspManager()
        with (
            patch.dict("sys.modules", {"multilspy": None}),
            pytest.raises(LspNotAvailableError, match="multilspy not installed"),
        ):
            await manager._start_server(SupportedLanguage.PYTHON, Path("/tmp/project"))


# ---------------------------------------------------------------------------
# Cleanup idle servers
# ---------------------------------------------------------------------------


class TestCleanupIdleServers:
    @pytest.mark.anyio
    async def test_removes_idle_servers(self) -> None:
        manager = LspManager(idle_timeout_seconds=0)
        mock_client = AsyncMock()
        mock_client.stop = AsyncMock()

        key = (SupportedLanguage.PYTHON, Path("/tmp/project"))
        server = CachedServer(
            client=mock_client,
            language=SupportedLanguage.PYTHON,
            project_root=Path("/tmp/project"),
        )
        server.last_used = datetime.now() - timedelta(hours=1)
        manager._servers[key] = server

        await manager._cleanup_idle_servers()

        assert len(manager._servers) == 0
        mock_client.stop.assert_called_once()

    @pytest.mark.anyio
    async def test_keeps_active_servers(self) -> None:
        manager = LspManager(idle_timeout_seconds=3600)
        mock_client = AsyncMock()

        key = (SupportedLanguage.PYTHON, Path("/tmp/project"))
        manager._servers[key] = CachedServer(
            client=mock_client,
            language=SupportedLanguage.PYTHON,
            project_root=Path("/tmp/project"),
        )

        await manager._cleanup_idle_servers()

        assert len(manager._servers) == 1

    @pytest.mark.anyio
    async def test_stop_error_logged(self) -> None:
        manager = LspManager(idle_timeout_seconds=0)
        mock_client = AsyncMock()
        mock_client.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

        key = (SupportedLanguage.PYTHON, Path("/tmp/project"))
        server = CachedServer(
            client=mock_client,
            language=SupportedLanguage.PYTHON,
            project_root=Path("/tmp/project"),
        )
        server.last_used = datetime.now() - timedelta(hours=1)
        manager._servers[key] = server

        # Should not raise, just log
        await manager._cleanup_idle_servers()
        assert len(manager._servers) == 0


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    @pytest.mark.anyio
    async def test_shutdown_stops_all_servers(self) -> None:
        manager = LspManager()
        mock_client1 = AsyncMock()
        mock_client1.stop = AsyncMock()
        mock_client2 = AsyncMock()
        mock_client2.stop = AsyncMock()

        manager._servers = {
            (SupportedLanguage.PYTHON, Path("/p1")): CachedServer(
                client=mock_client1,
                language=SupportedLanguage.PYTHON,
                project_root=Path("/p1"),
            ),
            (SupportedLanguage.TYPESCRIPT, Path("/p2")): CachedServer(
                client=mock_client2,
                language=SupportedLanguage.TYPESCRIPT,
                project_root=Path("/p2"),
            ),
        }

        await manager.shutdown()

        assert len(manager._servers) == 0
        mock_client1.stop.assert_called_once()
        mock_client2.stop.assert_called_once()

    @pytest.mark.anyio
    async def test_shutdown_cancels_cleanup_task(self) -> None:
        manager = LspManager()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pytest.skip("No asyncio event loop (e.g. running under trio)")
            return

        # Create a real asyncio task that we can cancel
        async def _forever():
            await asyncio.sleep(3600)

        real_task = loop.create_task(_forever())
        manager._cleanup_task = real_task

        await manager.shutdown()
        assert real_task.cancelled()


# ---------------------------------------------------------------------------
# get_cached_languages
# ---------------------------------------------------------------------------


class TestGetCachedLanguages:
    def test_empty(self) -> None:
        manager = LspManager()
        assert manager.get_cached_languages() == []

    def test_with_servers(self) -> None:
        manager = LspManager()
        manager._servers = {
            (SupportedLanguage.PYTHON, Path("/p1")): CachedServer(
                client=MagicMock(),
                language=SupportedLanguage.PYTHON,
                project_root=Path("/p1"),
            ),
        }
        result = manager.get_cached_languages()
        assert len(result) == 1
        assert result[0] == ("python", "/p1")


# ---------------------------------------------------------------------------
# _ensure_cleanup_task
# ---------------------------------------------------------------------------


class TestEnsureCleanupTask:
    def test_creates_task_when_none(self) -> None:
        manager = LspManager()
        manager._cleanup_task = None

        # Without a running loop, it should silently skip
        manager._ensure_cleanup_task()
        # Should not raise

    def test_creates_task_when_done(self) -> None:
        manager = LspManager()
        mock_task = MagicMock()
        mock_task.done.return_value = True
        manager._cleanup_task = mock_task

        # Without a running loop, should not raise
        manager._ensure_cleanup_task()
