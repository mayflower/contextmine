"""Tests for the public FastMCP tool registry."""

import pytest
from app.mcp_server import mcp


@pytest.mark.anyio
async def test_mcp_tools_exist_in_native_registry() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert {
        "list_collections",
        "list_documents",
        "get_markdown",
        "get_twin_graph",
        "query_twin_cypher",
        "create_architecture_intent",
        "get_arc42",
        "arc42_drift_report",
        "list_ports_adapters",
    }.issubset(tools)

    properties = tools["get_markdown"].parameters["properties"]
    assert {
        "query",
        "collection_id",
        "topic",
        "max_chunks",
        "max_tokens",
        "offset",
        "raw",
    } == set(properties)
    output_schema = tools["get_markdown"].output_schema
    assert output_schema is not None
    assert "markdown" in output_schema["properties"]


@pytest.mark.anyio
async def test_get_twin_graph_schema_includes_behavioral_filters() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    properties = tools["get_twin_graph"].parameters["properties"]

    assert {
        "facet",
        "include_provenance_mode",
        "include_test_links",
        "include_ui_links",
    }.issubset(properties)
