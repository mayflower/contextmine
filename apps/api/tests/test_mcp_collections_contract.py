"""Public FastMCP contracts for collections and retrieval."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp_domains.collections import (
    CollectionListResult,
    DocumentListResult,
    MarkdownResult,
    get_context_markdown,
    list_collections,
    list_documents,
)
from app.mcp_server import mcp
from contextmine_core import CollectionVisibility
from fastmcp.exceptions import ToolError


def _database_patch(database):  # noqa: ANN001, ANN202
    context = patch("app.mcp_domains.collections.get_db_session")
    session_factory = context.start()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=database)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.mark.anyio
async def test_collection_tools_publish_native_structured_contracts() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert {
        "list_collections",
        "list_documents",
        "get_markdown",
    }.issubset(tools)
    collection_schema = tools["list_collections"].output_schema
    assert collection_schema is not None
    collection_properties = collection_schema["properties"]
    assert set(collection_properties) == {"collections", "total"}
    assert collection_properties["collections"]["type"] == "array"
    assert collection_properties["total"] == {"type": "integer"}
    assert tools["list_documents"].parameters["properties"]["limit"]["maximum"] == 100
    assert tools["get_markdown"].parameters["properties"]["max_chunks"]["maximum"] == 50
    markdown_schema = tools["get_markdown"].output_schema
    assert markdown_schema is not None
    assert "markdown" in markdown_schema["properties"]


@pytest.mark.anyio
async def test_list_collections_is_structured_and_stateless() -> None:
    collection_id = uuid.uuid4()
    result = MagicMock()
    result.all.return_value = [(collection_id, "Docs", "docs", CollectionVisibility.GLOBAL, 2)]
    database = AsyncMock()
    database.execute = AsyncMock(return_value=result)
    context = _database_patch(database)
    try:
        with patch("app.mcp_domains.collections.get_current_user_id", return_value=None):
            first = await list_collections()
            second = await list_collections()
    finally:
        context.stop()

    assert isinstance(first, CollectionListResult)
    assert first == second
    assert first.total == 1
    assert first.collections[0].id == str(collection_id)
    assert database.execute.await_count == 2


@pytest.mark.anyio
async def test_list_documents_is_bounded_and_reports_truncation() -> None:
    collection = SimpleNamespace(id=uuid.uuid4(), name="Docs")
    collection_result = MagicMock()
    collection_result.scalar_one_or_none.return_value = collection
    document_result = MagicMock()
    document_result.all.return_value = [
        (uuid.uuid4(), "doc://one", "One", "https://example.test/repo"),
        (uuid.uuid4(), "doc://two", "Two", "https://example.test/repo"),
    ]
    database = AsyncMock()
    database.execute = AsyncMock(side_effect=[collection_result, document_result])
    context = _database_patch(database)
    try:
        with (
            patch("app.mcp_domains.collections.get_current_user_id", return_value=None),
            patch(
                "app.mcp_domains.collections.user_can_access_collection",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await list_documents(collection_id=str(collection.id), limit=1)
    finally:
        context.stop()

    assert isinstance(result, DocumentListResult)
    assert [document.title for document in result.documents] == ["One"]
    assert result.truncated is True


@pytest.mark.anyio
async def test_list_documents_uses_native_tool_error() -> None:
    database = AsyncMock()
    context = _database_patch(database)
    try:
        with (
            patch("app.mcp_domains.collections.get_current_user_id", return_value=None),
            pytest.raises(ToolError, match="Invalid collection_id"),
        ):
            await list_documents(collection_id="not-a-uuid")
    finally:
        context.stop()


@pytest.mark.anyio
async def test_raw_markdown_result_is_structured() -> None:
    search_result = SimpleNamespace(
        uri="doc://auth",
        title="Authentication",
        content="Authentication uses signed sessions.",
    )
    with (
        patch("app.mcp_domains.collections.get_current_user_id", return_value=None),
        patch(
            "app.mcp_domains.collections.get_settings",
            return_value=SimpleNamespace(model_calls_enabled=False),
        ),
        patch(
            "app.mcp_domains.collections.hybrid_search",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(results=[search_result]),
        ),
    ):
        result = await get_context_markdown(query="auth", raw=True)

    assert isinstance(result, MarkdownResult)
    assert result.mode == "retrieval"
    assert result.chunks_used == 1
    assert result.sources[0].uri == "doc://auth"
    assert "signed sessions" in result.markdown


@pytest.mark.anyio
async def test_synthesis_failure_uses_native_tool_error() -> None:
    with (
        patch("app.mcp_domains.collections.get_current_user_id", return_value=None),
        patch(
            "app.mcp_domains.collections.get_settings",
            return_value=SimpleNamespace(model_calls_enabled=True),
        ),
        patch(
            "app.mcp_domains.collections.assemble_context",
            new_callable=AsyncMock,
            side_effect=RuntimeError("provider unavailable"),
        ),
        pytest.raises(ToolError, match="provider unavailable"),
    ):
        await get_context_markdown(query="auth")
