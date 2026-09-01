"""ContextMine FastMCP server wiring."""

from contextmine_core import get_settings
from fastmcp import FastMCP

from app.mcp_auth import ContextMineGitHubProvider
from app.mcp_domains.architecture import register_architecture_tools
from app.mcp_domains.code import register_code_tools
from app.mcp_domains.collections import register_collection_tools
from app.mcp_domains.research import register_research_tools
from app.mcp_domains.twin import register_twin_tools
from app.mcp_domains.validation import register_validation_tools

settings = get_settings()
try:
    auth = ContextMineGitHubProvider()
except ValueError as exc:
    if not settings.debug:
        raise RuntimeError(
            "MCP authentication required in production. "
            "Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET, or enable DEBUG mode for testing."
        ) from exc
    auth = None

mcp = FastMCP(
    auth=auth,
    name="contextmine",
    instructions="""ContextMine researches documentation, code, architecture, and digital twins.

Start with get_markdown for retrieval, graph_rag for graph-augmented context, or
 deep_research for a persisted multi-step investigation. Use collection, code,
 architecture, twin, export, and validation tools for their bounded structured results.
""",
)

register_collection_tools(mcp)
register_code_tools(mcp)
register_research_tools(mcp)
register_architecture_tools(mcp)
register_twin_tools(mcp)
register_validation_tools(mcp)

mcp_app = mcp.http_app(path="/", stateless_http=True)
mcp_lifespan = mcp_app.lifespan
