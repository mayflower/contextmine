"""Native TypeScript 7 language-server integration for MultiLSPy."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from multilspy.language_server import LanguageServer
from multilspy.lsp_protocol_handler.lsp_types import InitializeParams
from multilspy.lsp_protocol_handler.server import ProcessLaunchInfo
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger


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

        async def log_message(params: dict[str, Any]) -> None:
            self.logger.log(f"LSP: window/logMessage: {params}", logging.INFO)

        self.server.on_request("client/registerCapability", acknowledge)
        self.server.on_request("window/workDoneProgress/create", acknowledge)
        self.server.on_request("workspace/diagnostic/refresh", acknowledge)
        self.server.on_notification("window/logMessage", log_message)
        self.server.on_notification("$/progress", acknowledge)
        self.server.on_notification("textDocument/publishDiagnostics", acknowledge)

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
                await self.server.shutdown()
                await self.server.stop()
