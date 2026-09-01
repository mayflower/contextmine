"""Public FastMCP contracts for research and architecture discovery."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp_domains.research import (
    ResearchResult,
    deep_research,
    get_research_report,
    graph_neighborhood,
    research_validation,
)
from app.mcp_server import mcp
from fastmcp.exceptions import ResourceError, ToolError

RESEARCH_TOOL_NAMES = {
    "deep_research",
    "research_validation",
    "research_data_model",
    "research_architecture",
    "graph_neighborhood",
    "trace_path",
    "graph_rag",
}


def _database_patch(database):  # noqa: ANN001, ANN202
    context = patch("app.mcp_domains.research.get_db_session")
    session_factory = context.start()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=database)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.mark.anyio
async def test_research_tools_publish_native_bounded_contracts() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert RESEARCH_TOOL_NAMES.issubset(tools)
    for name in RESEARCH_TOOL_NAMES:
        output_schema = tools[name].output_schema
        assert output_schema is not None
        assert "markdown" in output_schema["properties"]
    assert tools["deep_research"].parameters["properties"]["budget"]["maximum"] == 20
    assert tools["graph_neighborhood"].parameters["properties"]["depth"]["maximum"] == 3
    assert tools["graph_neighborhood"].parameters["properties"]["limit"]["maximum"] == 200
    assert tools["graph_rag"].parameters["properties"]["max_entities"]["maximum"] == 50


@pytest.mark.anyio
async def test_deep_research_returns_existing_full_run_id() -> None:
    run_id = str(uuid.uuid4())
    run = SimpleNamespace(
        run_id=run_id,
        evidence=[],
        budget_used=2,
        budget_steps=5,
        error_message=None,
        total_duration_ms=25,
        answer="The answer",
        status=SimpleNamespace(value="done"),
    )
    agent = MagicMock()
    agent.research = AsyncMock(return_value=run)
    with (
        patch("contextmine_core.research.ResearchAgent", return_value=agent),
        patch("contextmine_core.research.AgentConfig"),
        patch("contextmine_core.research.llm.get_research_llm_provider", return_value=object()),
    ):
        result = await deep_research(question="How does auth work?", budget=5, debug=True)

    assert isinstance(result, ResearchResult)
    assert result.run_id == run_id
    assert result.status == "done"
    assert result.data["artifact_report_uri"].endswith(f"{run_id}/report.md")


@pytest.mark.anyio
async def test_research_collection_filter_enforces_shared_access() -> None:
    requested = uuid.uuid4()
    database = AsyncMock()
    context = _database_patch(database)
    try:
        with (
            patch("app.mcp_domains.research.get_current_user_id", return_value=None),
            patch(
                "contextmine_core.search.get_accessible_collection_ids",
                new_callable=AsyncMock,
                return_value=[uuid.uuid4()],
            ),
            pytest.raises(ToolError, match="access denied"),
        ):
            await research_validation(code_path="auth.py", collection_id=str(requested))
    finally:
        context.stop()


@pytest.mark.anyio
async def test_graph_neighborhood_checks_collection_and_is_repeatable() -> None:
    node_id = uuid.uuid4()
    collection_id = uuid.uuid4()
    node = SimpleNamespace(id=node_id, collection_id=collection_id)
    collection = SimpleNamespace(id=collection_id)
    node_result = MagicMock()
    node_result.scalar_one_or_none.return_value = node
    collection_result = MagicMock()
    collection_result.scalar_one_or_none.return_value = collection
    database = AsyncMock()
    database.execute = AsyncMock(
        side_effect=[node_result, collection_result, node_result, collection_result]
    )
    context = _database_patch(database)
    result_pack = MagicMock()
    result_pack.entities = []
    result_pack.to_markdown.return_value = "# Neighborhood"
    result_pack.to_dict.return_value = {"entities": [], "edges": []}
    try:
        with (
            patch("app.mcp_domains.research.get_current_user_id", return_value=None),
            patch(
                "app.mcp_domains.research.user_can_access_collection",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "contextmine_core.graphrag.graph_neighborhood",
                new_callable=AsyncMock,
                side_effect=[result_pack, result_pack],
            ) as query,
        ):
            first = await graph_neighborhood(node_id=str(node_id))
            second = await graph_neighborhood(node_id=str(node_id))
    finally:
        context.stop()

    assert first == second
    assert first.kind == "graph_neighborhood"
    assert query.await_args.kwargs["collection_id"] == collection_id


@pytest.mark.anyio
async def test_missing_research_resource_uses_resource_error() -> None:
    store = MagicMock()
    store.get_report.return_value = None
    with (
        patch("contextmine_core.research.get_artifact_store", return_value=store),
        pytest.raises(ResourceError, match="Research run not found"),
    ):
        await get_research_report(str(uuid.uuid4()))
