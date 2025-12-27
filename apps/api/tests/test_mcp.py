"""Tests for MCP server functionality."""

from app.mcp_server import get_context_markdown_sync, get_tools


def test_mcp_tool_exists() -> None:
    """Test that the context.get_markdown tool is defined."""
    tools = get_tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "context.get_markdown"
    assert "query" in tool["inputSchema"].get("properties", {})
    # Verify new parameters are in schema
    properties = tool["inputSchema"].get("properties", {})
    assert "collection_id" in properties
    assert "max_chunks" in properties
    assert "max_tokens" in properties


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
