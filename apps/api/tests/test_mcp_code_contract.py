"""Public FastMCP contracts for code navigation and analysis."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp_domains.code import (
    AnalysisResult,
    CodeTextResult,
    expand,
    list_methods,
    outline,
    store_findings,
)
from app.mcp_server import mcp
from fastmcp.exceptions import ToolError

CODE_TOOL_NAMES = {
    "outline",
    "find_symbol",
    "definition",
    "references",
    "expand",
    "get_codebase_summary",
    "list_methods",
    "list_calls",
    "get_cfg",
    "get_variable_flow",
    "find_taint_sources",
    "find_taint_sinks",
    "find_taint_flows",
    "store_findings",
    "export_sarif",
}


def _database_patch(database):  # noqa: ANN001, ANN202
    context = patch("app.mcp_domains.code.get_db_session")
    session_factory = context.start()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=database)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.mark.anyio
async def test_code_tools_publish_native_structured_contracts() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert CODE_TOOL_NAMES.issubset(tools)
    for name in CODE_TOOL_NAMES:
        assert tools[name].output_schema is not None
        assert tools[name].output_schema.get("x-fastmcp-wrap-result") is not True
    assert tools["references"].parameters["properties"]["limit"]["maximum"] == 200
    assert tools["expand"].parameters["properties"]["depth"]["maximum"] == 3
    assert tools["list_methods"].parameters["properties"]["limit"]["maximum"] == 200
    assert tools["find_taint_flows"].parameters["properties"]["max_results"]["maximum"] == 200


@pytest.mark.anyio
async def test_outline_is_structured_and_repeatable() -> None:
    document = SimpleNamespace(id=uuid.uuid4())
    symbol = SimpleNamespace(
        parent_name=None,
        qualified_name="module.function",
        kind=SimpleNamespace(value="function"),
        name="function",
        start_line=1,
        end_line=3,
        signature="def function(): ...",
    )
    document_result = MagicMock()
    document_result.scalar_one_or_none.return_value = document
    symbol_result = MagicMock()
    symbol_result.scalars.return_value = [symbol]
    database = AsyncMock()
    database.execute = AsyncMock(
        side_effect=[document_result, symbol_result, document_result, symbol_result]
    )
    context = _database_patch(database)
    try:
        first = await outline(file_path="src/module.py")
        second = await outline(file_path="src/module.py")
    finally:
        context.stop()

    assert isinstance(first, CodeTextResult)
    assert first == second
    assert first.kind == "outline"
    assert first.count == 1
    assert "def function" in first.content


@pytest.mark.anyio
async def test_missing_outline_uses_native_tool_error() -> None:
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    database = AsyncMock()
    database.execute = AsyncMock(return_value=result)
    context = _database_patch(database)
    try:
        with pytest.raises(ToolError, match="No indexed document"):
            await outline(file_path="missing.py")
    finally:
        context.stop()


@pytest.mark.anyio
async def test_list_methods_delegates_to_existing_twin_query() -> None:
    collection = SimpleNamespace(id=uuid.uuid4())
    collection_result = MagicMock()
    collection_result.scalar_one_or_none.return_value = collection
    database = AsyncMock()
    database.execute = AsyncMock(return_value=collection_result)
    context = _database_patch(database)
    payload = {"items": [{"name": "run"}], "total": 1}
    try:
        with (
            patch(
                "app.mcp_domains.code.user_can_access_collection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("app.mcp_domains.code.get_current_user_id", return_value=None),
            patch(
                "app.mcp_domains.code.get_settings",
                return_value=SimpleNamespace(twin_analysis_cache_ttl_seconds=30),
            ),
            patch(
                "contextmine_core.twin.list_methods_multi",
                new_callable=AsyncMock,
                return_value=payload,
            ) as query,
        ):
            result = await list_methods(
                collection_id=str(collection.id),
                page=0,
                limit=25,
                engines="graphrag, lsp",
            )
    finally:
        context.stop()

    assert result == AnalysisResult(data=payload)
    query.assert_awaited_once()
    assert query.await_args.kwargs["limit"] == 25
    assert query.await_args.kwargs["engines"] == ["graphrag", "lsp"]


@pytest.mark.anyio
async def test_invalid_findings_and_seed_use_native_tool_errors() -> None:
    with pytest.raises(ToolError, match="valid JSON array"):
        await store_findings(collection_id=str(uuid.uuid4()), findings_json="not-json")
    with pytest.raises(ToolError, match="Invalid seed format"):
        await expand(seeds=["missing-separator"])
