"""Native TypeScript 7 language-server integration for MultiLSPy."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from multilspy.language_server import LanguageServer
from multilspy.lsp_protocol_handler.lsp_types import InitializeParams
from multilspy.lsp_protocol_handler.server import ProcessLaunchInfo, Request
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

_LOG_PREFIX_CHARS = 8192
_LOG_SUFFIX_CHARS = 1024


def _bounded_message(message: str) -> str:
    original_chars = len(message)
    truncated = original_chars > _LOG_PREFIX_CHARS + _LOG_SUFFIX_CHARS
    if truncated:
        message = f"{message[:_LOG_PREFIX_CHARS]}...[truncated]...{message[-_LOG_SUFFIX_CHARS:]}"
    return f"original_chars={original_chars} truncated={str(truncated).lower()} message={message}"


def _format_log_message(params: Any) -> tuple[int, str]:
    if not isinstance(params, dict):
        return (
            logging.WARNING,
            f"LSP: invalid window/logMessage payload_type={type(params).__name__}",
        )

    message_type = params.get("type")
    message = params.get("message")
    if not isinstance(message, str):
        return (
            logging.WARNING,
            f"LSP: invalid window/logMessage message_type={type(message).__name__}",
        )
    if type(message_type) is not int or message_type not in {1, 2, 3, 4}:
        return (
            logging.WARNING,
            f"LSP: invalid window/logMessage {_bounded_message(message)}",
        )

    level = {1: logging.ERROR, 2: logging.WARNING}.get(message_type, logging.DEBUG)
    return (
        level,
        f"LSP: window/logMessage type={message_type} {_bounded_message(message)}",
    )


class NativeTypeScriptLanguageServer(LanguageServer):
    """Run the LSP built into TypeScript 7 instead of the removed tsserver."""

    def __init__(
        self,
        config: MultilspyConfig,
        logger: MultilspyLogger,
        repository_root_path: str,
        language_id: str,
    ) -> None:
        server_binary = config.server_binary
        if server_binary is None:
            raise ValueError("the native TypeScript server binary is required")

        protocol_log_level = os.getenv("LSP_PROTOCOL_LOG_LEVEL", "INFO").upper()
        if protocol_log_level not in {"DEBUG", "INFO"}:
            raise ValueError("LSP_PROTOCOL_LOG_LEVEL must be DEBUG or INFO")
        logger.logger = logging.getLogger("contextmine.lsp.typescript.protocol")
        logger.logger.setLevel(protocol_log_level)

        super().__init__(
            config,
            logger,
            repository_root_path,
            ProcessLaunchInfo(
                cmd=[server_binary, "--lsp", "--stdio"],
                cwd=repository_root_path,
            ),
            language_id,
        )

    def _get_initialize_params(self) -> InitializeParams:
        root = Path(self.repository_root_path)
        return {
            "processId": os.getpid(),
            "clientInfo": {"name": "contextmine", "version": "1"},
            "rootPath": str(root),
            "rootUri": root.as_uri(),
            "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
            "capabilities": {
                "workspace": {
                    "configuration": True,
                    "workspaceFolders": True,
                },
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                },
            },
            "trace": "off",
        }

    def _register_protocol_handlers(self) -> None:
        async def acknowledge(_params: dict[str, Any]) -> None:
            return None

        async def log_message(params: Any) -> None:
            level, message = _format_log_message(params)
            self.logger.log(message, level)

        self.server.on_request("client/registerCapability", acknowledge)
        self.server.on_request("window/workDoneProgress/create", acknowledge)
        self.server.on_request("workspace/diagnostic/refresh", acknowledge)
        self.server.on_notification("window/logMessage", log_message)
        self.server.on_notification("$/progress", acknowledge)
        self.server.on_notification("textDocument/publishDiagnostics", acknowledge)

    async def _request_shutdown(self) -> None:
        """Send TypeScript's parameterless shutdown request."""
        request = Request()
        request_id = self.server.request_id
        self.server.request_id += 1
        self.server._response_handlers[request_id] = request
        try:
            async with request.cv:
                await self.server._send_payload(
                    {"jsonrpc": "2.0", "id": request_id, "method": "shutdown"}
                )
                await request.cv.wait()
        finally:
            self.server._response_handlers.pop(request_id, None)
        if request.error is not None:
            raise request.error

    async def _shutdown_server(self) -> None:
        """Stop TypeScript without MultiLSPy's incompatible null parameters."""
        error: Exception | None = None
        try:
            # ponytail: remove this shim when pinned MultiLSPy omits null params.
            await asyncio.wait_for(self._request_shutdown(), timeout=30)
        except Exception as exc:  # noqa: BLE001
            error = exc

        self.server._received_shutdown = True
        try:
            await self.server._send_payload({"jsonrpc": "2.0", "method": "exit"})
        except Exception as exc:  # noqa: BLE001
            error = error or exc

        if error is not None:
            self.logger.log(
                f"Native TypeScript LSP shutdown failed: {_bounded_message(str(error))}",
                logging.WARNING,
            )

    @asynccontextmanager
    async def start_server(self) -> AsyncIterator[NativeTypeScriptLanguageServer]:
        """Start, initialize, and reliably stop the native server."""
        self._register_protocol_handlers()

        async with super().start_server():
            await self.server.start()
            try:
                response = await self.server.send.initialize(self._get_initialize_params())
                capabilities = response.get("capabilities") if isinstance(response, dict) else None
                if not isinstance(capabilities, dict):
                    raise RuntimeError(
                        "native TypeScript LSP initialize response has no capabilities"
                    )
                if not capabilities.get("hoverProvider"):
                    raise RuntimeError("native TypeScript LSP does not provide hover")
                if not capabilities.get("definitionProvider"):
                    raise RuntimeError("native TypeScript LSP does not provide definitions")

                self.server.notify.initialized({})
                if capabilities.get("completionProvider"):
                    self.completions_available.set()
                yield self
            finally:
                try:
                    await self._shutdown_server()
                finally:
                    await self.server.stop()
