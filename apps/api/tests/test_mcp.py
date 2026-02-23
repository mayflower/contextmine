"""Tests for MCP server functionality."""

from app.mcp_server import get_context_markdown_sync, get_tools


def test_mcp_tools_exist() -> None:
    """Test that all MCP tools are defined."""
    tools = get_tools()
    assert len(tools) >= 3

    tool_names = {t["name"] for t in tools}
    assert "context.list_collections" in tool_names
    assert "context.list_documents" in tool_names
    assert "context.get_markdown" in tool_names
    assert "context.get_twin_graph" in tool_names
    assert "context.query_twin_cypher" in tool_names
    assert "context.create_architecture_intent" in tool_names
    assert "context.get_arc42" in tool_names
    assert "context.arc42_drift_report" in tool_names
    assert "context.list_ports_adapters" in tool_names

    # Find get_markdown tool and verify its schema
    get_markdown = next(t for t in tools if t["name"] == "context.get_markdown")
    properties = get_markdown["inputSchema"].get("properties", {})
    assert "query" in properties
    assert "collection_id" in properties
    assert "topic" in properties
    assert "max_chunks" in properties
    assert "max_tokens" in properties
    assert "offset" in properties
    assert "raw" in properties


def test_mcp_tool_returns_markdown() -> None:
    """Test that the MCP tool returns a markdown string."""
    result = get_context_markdown_sync("test query")
    assert isinstance(result, str)
    assert "# Context for: test query" in result


def test_get_twin_graph_schema_includes_behavioral_filters() -> None:
    tools = get_tools()
    get_twin_graph = next(t for t in tools if t["name"] == "context.get_twin_graph")
    properties = get_twin_graph["inputSchema"].get("properties", {})
    assert "facet" in properties
    assert "include_provenance_mode" in properties
    assert "include_test_links" in properties
    assert "include_ui_links" in properties


def test_get_context_markdown_includes_sections() -> None:
    """Test that context markdown includes expected sections."""
    result = get_context_markdown_sync("example query")
    assert "# Context for: example query" in result
    assert "## Summary" in result
    assert "## Sources" in result
