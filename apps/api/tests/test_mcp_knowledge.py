"""Tests for Knowledge Graph MCP tools."""


class TestMCPToolsRegistered:
    """Tests that new MCP tools are registered with the FastMCP server."""

    def test_mcp_server_has_research_tools(self) -> None:
        """Test that the MCP server has research-focused tools registered."""
        from app.mcp_server import mcp

        # Get tool names from the FastMCP server
        tool_names = {tool.name for tool in mcp._tool_manager._tools.values()}

        # Verify all research-focused knowledge graph tools are registered
        expected_tools = {
            "research_validation",
            "research_data_model",
            "research_architecture",
            "graph_neighborhood",
            "trace_path",
            "graph_rag",
            "get_arc42",
            "arc42_drift_report",
        }

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Tool '{tool_name}' not registered"

    def test_existing_tools_still_available(self) -> None:
        """Test that existing MCP tools are still available."""
        from app.mcp_server import mcp

        tool_names = {tool.name for tool in mcp._tool_manager._tools.values()}

        # Verify existing tools are still registered
        expected_existing = {
            "list_collections",
            "list_documents",
            "get_markdown",
            "deep_research",
            "outline",
            "find_symbol",
            "definition",
            "references",
            "expand",
        }

        for tool_name in expected_existing:
            assert tool_name in tool_names, f"Existing tool '{tool_name}' not found"


class TestMCPInstructionsUpdated:
    """Tests that MCP instructions include research tools."""

    def test_instructions_include_graph_rag(self) -> None:
        """Test that instructions mention graph_rag."""
        from app.mcp_server import mcp

        assert "graph_rag" in mcp.instructions

    def test_instructions_include_research_validation(self) -> None:
        """Test that instructions mention research_validation tool."""
        from app.mcp_server import mcp

        assert "research_validation" in mcp.instructions

    def test_instructions_include_research_data_model(self) -> None:
        """Test that instructions mention research_data_model tool."""
        from app.mcp_server import mcp

        assert "research_data_model" in mcp.instructions

    def test_instructions_include_research_architecture(self) -> None:
        """Test that instructions mention research_architecture tool."""
        from app.mcp_server import mcp

        assert "research_architecture" in mcp.instructions

    def test_instructions_include_graph_tools(self) -> None:
        """Test that instructions mention graph neighborhood and trace."""
        from app.mcp_server import mcp

        assert "graph_neighborhood" in mcp.instructions
        assert "trace_path" in mcp.instructions

    def test_instructions_include_arc42(self) -> None:
        """Test that instructions mention arc42 tools."""
        from app.mcp_server import mcp

        assert "get_arc42" in mcp.instructions
        assert "arc42_drift_report" in mcp.instructions


class TestMCPToolSchemas:
    """Tests for MCP tool input schemas."""

    def test_research_validation_schema(self) -> None:
        """Test that research_validation has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "research_validation")
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Must have code_path, optional collection_id
        assert "code_path" in props
        assert "code_path" in required
        assert "collection_id" in props

    def test_research_data_model_schema(self) -> None:
        """Test that research_data_model has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "research_data_model")
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Must have entity, optional collection_id
        assert "entity" in props
        assert "entity" in required
        assert "collection_id" in props

    def test_research_architecture_schema(self) -> None:
        """Test that research_architecture has expected parameters."""
        from app.mcp_server import mcp

        tool = next(
            t for t in mcp._tool_manager._tools.values() if t.name == "research_architecture"
        )
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Must have topic, optional collection_id
        assert "topic" in props
        assert "topic" in required
        assert "collection_id" in props

    def test_graph_neighborhood_schema(self) -> None:
        """Test that graph_neighborhood has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "graph_neighborhood")
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Must have node_id
        assert "node_id" in props
        assert "node_id" in required
        assert "depth" in props
        assert "edge_kinds" in props
        assert "limit" in props

    def test_trace_path_schema(self) -> None:
        """Test that trace_path has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "trace_path")
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Must have from_node_id and to_node_id
        assert "from_node_id" in props
        assert "to_node_id" in props
        assert "from_node_id" in required
        assert "to_node_id" in required
        assert "max_hops" in props

    def test_graph_rag_schema(self) -> None:
        """Test that graph_rag has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "graph_rag")
        schema = tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Must have query
        assert "query" in props
        assert "query" in required
        assert "collection_id" in props
        assert "max_depth" in props
        # GraphRAG parameters (Leiden community-based)
        assert "max_communities" in props
        assert "max_entities" in props
        assert "format" in props
        # Map-reduce answering parameter
        assert "answer" in props

    def test_get_arc42_schema(self) -> None:
        """Test that get_arc42 has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "get_arc42")
        schema = tool.parameters
        props = schema.get("properties", {})

        # Should have optional collection_id, section, regenerate
        assert "collection_id" in props
        assert "section" in props
        assert "regenerate" in props

    def test_arc42_drift_report_schema(self) -> None:
        """Test that arc42_drift_report has expected parameters."""
        from app.mcp_server import mcp

        tool = next(t for t in mcp._tool_manager._tools.values() if t.name == "arc42_drift_report")
        schema = tool.parameters
        props = schema.get("properties", {})

        # Should have optional collection_id
        assert "collection_id" in props
