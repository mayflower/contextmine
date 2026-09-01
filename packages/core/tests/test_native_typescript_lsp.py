"""Tests for ContextMine's native TypeScript 7 LSP adapter."""

from pathlib import Path

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
