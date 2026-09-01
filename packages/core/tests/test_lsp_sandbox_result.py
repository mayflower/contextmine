"""Behavioral proof that LSP tools consume persisted sandbox results."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.twin.ops import _collect_lsp_symbols


@pytest.mark.anyio
async def test_lsp_uses_source_version_result_without_worker_checkout() -> None:
    scenario_id = uuid.uuid4()
    collection_id = uuid.uuid4()
    source_id = uuid.uuid4()
    source_version_id = uuid.uuid4()
    scenario = SimpleNamespace(id=scenario_id, version=7)
    file_node = SimpleNamespace(
        source_id=source_id,
        source_version_id=source_version_id,
        meta={"file_path": "src/app.ts"},
        natural_key="file:src/app.ts",
        updated_at=None,
    )
    lsp_symbol = {
        "name": "main",
        "kind_id": 12,
        "kind": "function",
        "file_path": "src/app.ts",
        "line_number": 3,
        "column": 0,
    }
    source_version = SimpleNamespace(
        source_id=source_id,
        stats={
            "scenario_id": str(scenario_id),
            "lsp": {"files_scanned": 1, "errors": [], "symbols": [lsp_symbol]},
        },
    )
    nodes_result = MagicMock()
    nodes_result.scalars.return_value.all.return_value = [file_node]
    versions_result = MagicMock()
    versions_result.scalars.return_value.all.return_value = [source_version]
    session = AsyncMock()
    session.execute.side_effect = [nodes_result, versions_result]

    with (
        patch(
            "contextmine_core.twin.ops._resolve_analysis_scenario",
            AsyncMock(return_value=scenario),
        ),
        patch(
            "contextmine_core.twin.ops.get_settings",
            return_value=SimpleNamespace(repos_root="/definitely/not/a/checkout"),
        ),
    ):
        result = await _collect_lsp_symbols(
            session,
            collection_id=collection_id,
            scenario_id=scenario_id,
            max_files=20,
        )

    assert result == {
        "scenario_id": str(scenario_id),
        "scenario_version": 7,
        "files_scanned": 1,
        "errors": [],
        "symbols": [lsp_symbol],
    }
