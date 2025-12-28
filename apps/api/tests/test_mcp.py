"""Tests for MCP server functionality."""

from app.mcp_server import get_context_markdown_sync, get_tools


def test_mcp_tools_exist() -> None:
    """Test that all MCP tools are defined."""
    tools = get_tools()
    assert len(tools) == 3

    tool_names = {t["name"] for t in tools}
    assert "context.list_collections" in tool_names
    assert "context.list_documents" in tool_names
    assert "context.get_markdown" in tool_names

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


def test_get_context_markdown_includes_sections() -> None:
    """Test that context markdown includes expected sections."""
    result = get_context_markdown_sync("example query")
    assert "# Context for: example query" in result
    assert "## Summary" in result
    assert "## Sources" in result
