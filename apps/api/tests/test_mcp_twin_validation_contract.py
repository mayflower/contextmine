"""Public FastMCP contracts for twin, exports, and validation."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp_domains.twin import (
    TwinGraphResult,
    TwinRefreshResult,
    get_twin_graph,
    refresh_twin,
)
from app.mcp_domains.validation import (
    ValidationDashboardResult,
    get_validation_dashboard,
)
from app.mcp_server import mcp
from fastmcp.exceptions import ToolError

TWIN_TOOL_NAMES = {
    "get_twin_graph",
    "query_twin_cypher",
    "get_twin_status",
    "get_twin_timeline",
    "refresh_twin",
    "export_twin_view",
}


def _database_patch(module: str, database):  # noqa: ANN001, ANN202
    context = patch(f"{module}.get_db_session")
    session_factory = context.start()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=database)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.mark.anyio
async def test_twin_and_validation_tools_publish_native_contracts() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert TWIN_TOOL_NAMES.issubset(tools)
    assert "get_validation_dashboard" in tools
    for name in TWIN_TOOL_NAMES | {"get_validation_dashboard"}:
        assert tools[name].output_schema is not None
        assert tools[name].output_schema.get("x-fastmcp-wrap-result") is not True
    assert tools["get_twin_graph"].parameters["properties"]["limit"]["maximum"] == 5000
    assert tools["get_twin_timeline"].parameters["properties"]["limit"]["maximum"] == 200
    assert (
        tools["refresh_twin"].parameters["properties"]["source_ids"]["anyOf"][0]["maxItems"] == 100
    )


@pytest.mark.anyio
async def test_twin_graph_is_structured_and_repeatable() -> None:
    scenario_id = uuid.uuid4()
    graph = {
        "nodes": [
            {"id": "code", "kind": "file", "natural_key": "src/app.py", "meta": {}},
            {"id": "test", "kind": "test_case", "natural_key": "test_app", "meta": {}},
        ],
        "edges": [{"source_node_id": "test", "target_node_id": "code", "kind": "tests"}],
        "total_nodes": 2,
    }
    database = AsyncMock()
    context = _database_patch("app.mcp_domains.twin", database)
    try:
        with patch(
            "contextmine_core.twin.get_scenario_graph",
            new_callable=AsyncMock,
            side_effect=[graph, graph],
        ):
            first = await get_twin_graph(scenario_id=str(scenario_id), facet="code")
            second = await get_twin_graph(scenario_id=str(scenario_id), facet="code")
    finally:
        context.stop()

    assert isinstance(first, TwinGraphResult)
    assert first == second
    assert [node["id"] for node in first.nodes] == ["code"]
    assert first.edges == []


@pytest.mark.anyio
async def test_refresh_returns_existing_source_version_ids() -> None:
    collection = SimpleNamespace(id=uuid.uuid4())
    source_version_id = uuid.uuid4()
    collection_result = MagicMock()
    collection_result.scalar_one_or_none.return_value = collection
    database = AsyncMock()
    database.execute = AsyncMock(return_value=collection_result)
    context = _database_patch("app.mcp_domains.twin", database)
    payload = {
        "collection_id": str(collection.id),
        "created": 1,
        "skipped": 0,
        "items": [
            {
                "source_id": str(uuid.uuid4()),
                "source_version_id": str(source_version_id),
                "status": "queued",
                "queued": True,
            }
        ],
    }
    try:
        with (
            patch("app.mcp_domains.twin.get_current_user_id", return_value=None),
            patch(
                "app.mcp_domains.twin.user_can_access_collection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "contextmine_core.twin.trigger_collection_refresh",
                new_callable=AsyncMock,
                return_value=payload,
            ),
        ):
            result = await refresh_twin(collection_id=str(collection.id))
    finally:
        context.stop()

    assert isinstance(result, TwinRefreshResult)
    assert result.items[0]["source_version_id"] == str(source_version_id)
    database.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_twin_and_validation_invalid_ids_use_tool_errors() -> None:
    with pytest.raises(ToolError, match="Invalid scenario_id"):
        await get_twin_graph(scenario_id="invalid")
    with pytest.raises(ToolError, match="Invalid collection_id"):
        await get_validation_dashboard(collection_id="invalid")


@pytest.mark.anyio
async def test_validation_delegates_to_existing_core_and_commits() -> None:
    database = AsyncMock()
    context = _database_patch("app.mcp_domains.validation", database)
    payload = {"overall": "healthy", "systems": []}
    try:
        with (
            patch(
                "app.mcp_domains.validation.refresh_validation_snapshots",
                new_callable=AsyncMock,
            ) as refresh,
            patch(
                "app.mcp_domains.validation.get_latest_validation_status",
                new_callable=AsyncMock,
                return_value=payload,
            ) as status,
        ):
            result = await get_validation_dashboard()
    finally:
        context.stop()

    assert result == ValidationDashboardResult(collection_id=None, status=payload)
    refresh.assert_awaited_once_with(database, None)
    status.assert_awaited_once_with(database, None)
    database.commit.assert_awaited_once()
