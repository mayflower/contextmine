"""LSP Manager: Singleton manager for language server lifecycle.

The LspManager handles:
- Lazy initialization of language servers
- Caching servers by (language, project_root) for reuse
- Automatic shutdown of idle servers
- Graceful degradation when servers are unavailable
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from contextmine_core.lsp.client import LspClient, MockLspClient
from contextmine_core.lsp.exceptions import LspNotAvailableError, LspTimeoutError
from contextmine_core.lsp.languages import (
    SupportedLanguage,
    detect_language,
    find_project_root,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CachedServer:
    """A cached language server instance."""

    client: LspClient
    language: SupportedLanguage
    project_root: Path
    last_used: datetime = field(default_factory=datetime.now)

    def touch(self) -> None:
        """Update last used time."""
        self.last_used = datetime.now()


class LspManager:
    """Manages language server lifecycle with caching and timeout.

    This is a singleton that:
    - Lazily starts language servers on first request
    - Caches servers by (language, project_root)
    - Stops idle servers after timeout
    - Handles graceful degradation when servers unavailable
    """

    _instance: LspManager | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        idle_timeout_seconds: float = 300.0,  # 5 minutes
        request_timeout_seconds: float = 30.0,
    ):
        """Initialize the manager.

        Args:
            idle_timeout_seconds: Time after which idle servers are stopped
            request_timeout_seconds: Timeout for individual LSP requests
        """
        self._idle_timeout = timedelta(seconds=idle_timeout_seconds)
        self._request_timeout = request_timeout_seconds
        self._servers: dict[tuple[SupportedLanguage, Path], CachedServer] = {}
        self._servers_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._mock_client: MockLspClient | None = None

    @classmethod
    def get_instance(cls) -> LspManager:
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        # At this point cls._instance is guaranteed to not be None
        assert cls._instance is not None
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. For testing only."""
        with cls._lock:
            if cls._instance:
                # Clear the instance
                cls._instance._mock_client = None
                cls._instance = None

    def set_mock_client(self, client: MockLspClient) -> None:
        """Set a mock client for testing.

        When set, all get_client calls will return this mock.

        Args:
            client: The mock client to use
        """
        self._mock_client = client

    async def get_client(
        self,
        file_path: str | Path,
        project_root: str | Path | None = None,
    ) -> LspClient | MockLspClient:
        """Get an LSP client for the given file.

        Args:
            file_path: Path to a source file
            project_root: Optional project root (auto-detected if None)

        Returns:
            LspClient ready for use

        Raises:
            LspNotAvailableError: If language is unsupported or server unavailable
        """
        # Return mock if set
        if self._mock_client is not None:
            return self._mock_client

        file_path = Path(file_path)
        language = detect_language(file_path)

        if language is None:
            raise LspNotAvailableError(f"Unsupported file type: {file_path.suffix}")

        if project_root is None:
            project_root = find_project_root(file_path)
        else:
            project_root = Path(project_root).resolve()

        cache_key = (language, project_root)

        async with self._servers_lock:
            if cache_key in self._servers:
                cached = self._servers[cache_key]
                cached.touch()
                return cached.client

            # Start new server
            client = await self._start_server(language, project_root)
            self._servers[cache_key] = CachedServer(
                client=client,
                language=language,
                project_root=project_root,
            )

            # Ensure cleanup task is running
            self._ensure_cleanup_task()

            return client

    async def _start_server(
        self,
        language: SupportedLanguage,
        project_root: Path,
    ) -> LspClient:
        """Start a language server for the given language and project.

        Args:
            language: The programming language
            project_root: Root directory of the project

        Returns:
            Initialized LspClient

        Raises:
            LspNotAvailableError: If server cannot be started
            LspTimeoutError: If server startup times out
        """
        try:
            from multilspy import LanguageServer
            from multilspy.multilspy_config import MultilspyConfig
            from multilspy.multilspy_logger import MultilspyLogger
        except ImportError as e:
            raise LspNotAvailableError("multilspy not installed. Run: pip install multilspy") from e

        try:
            config = MultilspyConfig.from_dict({"code_language": language.value})
            lsp_logger = MultilspyLogger()

            server = LanguageServer.create(config, lsp_logger, str(project_root))

            client = LspClient(server, project_root)
            await asyncio.wait_for(
                client.start(),
                timeout=self._request_timeout,
            )

            logger.info(
                "Started %s language server for %s",
                language.value,
                project_root,
            )
            return client

        except TimeoutError as e:
            raise LspTimeoutError(f"Timeout starting {language.value} server") from e
        except Exception as e:
            raise LspNotAvailableError(f"Failed to start {language.value} server: {e}") from e

    def _ensure_cleanup_task(self) -> None:
        """Ensure the cleanup background task is running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop())
            except RuntimeError:
                # No running loop - cleanup will happen on next request
                pass

    async def _cleanup_loop(self) -> None:
        """Periodically clean up idle servers."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self._cleanup_idle_servers()

    async def _cleanup_idle_servers(self) -> None:
        """Stop servers that have been idle too long."""
        now = datetime.now()
        to_remove: list[tuple[SupportedLanguage, Path]] = []

        async with self._servers_lock:
            for key, cached in self._servers.items():
                if now - cached.last_used > self._idle_timeout:
                    to_remove.append(key)

            for key in to_remove:
                cached = self._servers.pop(key)
                try:
                    await cached.client.stop()
                    logger.info(
                        "Stopped idle %s server for %s",
                        cached.language.value,
                        cached.project_root,
                    )
                except Exception as e:
                    logger.warning("Error stopping server: %s", e)

    async def shutdown(self) -> None:
        """Shutdown all language servers."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        async with self._servers_lock:
            for cached in self._servers.values():
                try:
                    await cached.client.stop()
                except Exception as e:
                    logger.warning("Error stopping server during shutdown: %s", e)
            self._servers.clear()

        logger.info("LSP manager shutdown complete")

    def get_cached_languages(self) -> list[tuple[str, str]]:
        """Get list of currently cached server languages and project roots.

        Returns:
            List of (language, project_root) tuples
        """
        return [(k[0].value, str(k[1])) for k in self._servers]


def get_lsp_manager() -> LspManager:
    """Get the global LSP manager instance.

    Returns:
        The singleton LspManager
    """
    return LspManager.get_instance()
