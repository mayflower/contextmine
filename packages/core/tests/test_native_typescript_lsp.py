"""Tests for ContextMine's native TypeScript 7 LSP adapter."""

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from contextmine_core.lsp.native_typescript import NativeTypeScriptLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger


def make_config(server_binary: str | None) -> MultilspyConfig:
    """Create the small MultiLSPy configuration used by the native adapter."""
    return MultilspyConfig.from_dict(
        {
            "code_language": "typescript",
            "server_binary": server_binary,
        }
    )


def test_native_server_uses_typescript_7_lsp_mode(tmp_path: Path) -> None:
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"),
        MultilspyLogger(),
        str(tmp_path),
        "typescript",
    )

    assert server.server.process_launch_info.cmd == [
        "/usr/local/bin/tsc",
        "--lsp",
        "--stdio",
    ]
    assert server.server.process_launch_info.cwd == str(tmp_path)


def test_native_server_initialization_targets_repository(tmp_path: Path) -> None:
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"),
        MultilspyLogger(),
        str(tmp_path),
        "typescript",
    )

    params = server._get_initialize_params()

    assert params["rootPath"] == str(tmp_path)
    assert params["rootUri"] == tmp_path.as_uri()
    assert params["workspaceFolders"] == [{"uri": tmp_path.as_uri(), "name": tmp_path.name}]
    assert params["capabilities"]["workspace"]["configuration"] is True


def test_native_server_requires_preinstalled_binary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="server binary is required"):
        NativeTypeScriptLanguageServer(
            make_config(None),
            MultilspyLogger(),
            str(tmp_path),
            "typescript",
        )


@pytest.mark.parametrize(
    ("configured_level", "expected_level"),
    [(None, logging.INFO), ("debug", logging.DEBUG)],
)
def test_native_server_protocol_log_level_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_level: str | None,
    expected_level: int,
) -> None:
    if configured_level is None:
        monkeypatch.delenv("LSP_PROTOCOL_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LSP_PROTOCOL_LOG_LEVEL", configured_level)

    logger = MultilspyLogger()
    NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"), logger, str(tmp_path), "typescript"
    )

    assert logger.logger.name == "contextmine.lsp.typescript.protocol"
    assert logger.logger.level == expected_level
    assert logger.logger.isEnabledFor(logging.DEBUG) is (expected_level == logging.DEBUG)


def test_native_server_rejects_invalid_protocol_log_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LSP_PROTOCOL_LOG_LEVEL", "WARNING")

    with pytest.raises(ValueError, match="LSP_PROTOCOL_LOG_LEVEL must be DEBUG or INFO"):
        NativeTypeScriptLanguageServer(
            make_config("/usr/local/bin/tsc"),
            MultilspyLogger(),
            str(tmp_path),
            "typescript",
        )


@pytest.mark.parametrize(
    ("message_type", "expected_level"),
    [
        (1, logging.ERROR),
        (2, logging.WARNING),
        (3, logging.DEBUG),
        (4, logging.DEBUG),
    ],
)
def test_native_server_maps_log_severity(
    tmp_path: Path, message_type: int, expected_level: int
) -> None:
    logger = MultilspyLogger()
    logger.log = Mock()
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"), logger, str(tmp_path), "typescript"
    )
    server._register_protocol_handlers()

    asyncio.run(
        server.server.on_notification_handlers["window/logMessage"](
            {"type": message_type, "message": "ready"}
        )
    )

    assert logger.log.call_args.args == (
        f"LSP: window/logMessage type={message_type} "
        "original_chars=5 truncated=false message=ready",
        expected_level,
    )


def test_native_server_bounds_log_messages(tmp_path: Path) -> None:
    logger = MultilspyLogger()
    logger.log = Mock()
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"), logger, str(tmp_path), "typescript"
    )
    server._register_protocol_handlers()
    message = "a" * 8193 + "b" * 1024

    asyncio.run(
        server.server.on_notification_handlers["window/logMessage"]({"type": 3, "message": message})
    )

    logged = logger.log.call_args.args[0]
    assert "original_chars=9217 truncated=true" in logged
    assert f"message={'a' * 8192}...[truncated]...{'b' * 1024}" in logged


def test_native_server_warns_for_invalid_log_payload(tmp_path: Path) -> None:
    logger = MultilspyLogger()
    logger.log = Mock()
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"), logger, str(tmp_path), "typescript"
    )
    server._register_protocol_handlers()

    asyncio.run(
        server.server.on_notification_handlers["window/logMessage"](
            {"type": 99, "message": "still useful", "extra": "must-not-be-logged"}
        )
    )

    assert logger.log.call_args.args[1] == logging.WARNING
    assert "invalid window/logMessage" in logger.log.call_args.args[0]
    assert "message=still useful" in logger.log.call_args.args[0]
    assert "must-not-be-logged" not in logger.log.call_args.args[0]


def test_native_server_shutdown_omits_params(tmp_path: Path) -> None:
    logger = MultilspyLogger()
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"), logger, str(tmp_path), "typescript"
    )
    payloads: list[dict[str, Any]] = []

    async def exercise() -> None:
        async def send(payload: dict[str, Any]) -> None:
            payloads.append(payload)
            if payload["method"] == "shutdown":

                async def respond() -> None:
                    await asyncio.sleep(0)
                    await server.server._response_handlers[payload["id"]].on_result(None)

                asyncio.create_task(respond())

        server.server._send_payload = send
        await server._shutdown_server()

    asyncio.run(exercise())

    assert payloads == [
        {"jsonrpc": "2.0", "id": 1, "method": "shutdown"},
        {"jsonrpc": "2.0", "method": "exit"},
    ]


def test_native_server_shutdown_failure_is_bounded(tmp_path: Path) -> None:
    logger = MultilspyLogger()
    logger.log = Mock()
    server = NativeTypeScriptLanguageServer(
        make_config("/usr/local/bin/tsc"), logger, str(tmp_path), "typescript"
    )
    payloads: list[dict[str, Any]] = []

    async def exercise() -> None:
        async def send(payload: dict[str, Any]) -> None:
            payloads.append(payload)
            if payload["method"] == "shutdown":
                raise RuntimeError("a" * 8193 + "b" * 1024)

        server.server._send_payload = send
        await server._shutdown_server()

    asyncio.run(exercise())

    assert payloads[-1] == {"jsonrpc": "2.0", "method": "exit"}
    assert logger.log.call_args.args[1] == logging.WARNING
    assert "original_chars=9217 truncated=true" in logger.log.call_args.args[0]
