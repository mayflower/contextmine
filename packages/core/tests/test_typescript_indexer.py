"""Tests for the TypeScript/JavaScript SCIP backend command construction."""

from __future__ import annotations

import json
from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.typescript import TypescriptIndexerBackend
from contextmine_core.semantic_snapshot.models import Language, ProjectTarget


def test_build_command_for_javascript_uses_generated_project_config(tmp_path: Path) -> None:
    backend = TypescriptIndexerBackend()
    target = ProjectTarget(language=Language.JAVASCRIPT, root_path=tmp_path)

    cmd, generated_project_config = backend._build_command(target, tmp_path / "index.scip")  # noqa: SLF001

    assert cmd[:2] == ["scip-typescript", "index"]
    assert "--project" in cmd
    assert generated_project_config is not None
    assert generated_project_config.exists()

    payload = json.loads(generated_project_config.read_text(encoding="utf-8"))
    compiler_options = dict(payload.get("compilerOptions") or {})
    assert compiler_options.get("allowJs") is True
    assert compiler_options.get("noEmit") is True

    generated_project_config.unlink(missing_ok=True)


def test_build_command_for_typescript_uses_default_mode(tmp_path: Path) -> None:
    backend = TypescriptIndexerBackend()
    target = ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path)

    cmd, generated_project_config = backend._build_command(target, tmp_path / "index.scip")  # noqa: SLF001

    assert cmd == ["scip-typescript", "index"]
    assert generated_project_config is None
