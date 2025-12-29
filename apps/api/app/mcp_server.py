"""MCP server implementation with FastMCP 2 and Streamable HTTP transport."""

import json
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Annotated

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Document,
    MCPApiToken,
    Source,
    Symbol,
    get_settings,
    verify_api_token,
)
from contextmine_core import get_session as get_db_session
from contextmine_core.context import assemble_context
from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec
from contextmine_core.search import hybrid_search
from fastmcp import FastMCP
from sqlalchemy import func, or_, select, update
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount

# Create FastMCP server
mcp = FastMCP(
    name="contextmine",
    instructions="""ContextMine - documentation and code retrieval.

## Quick Start
For simple questions: get_markdown(query="how do I X?")
For complex investigations: deep_research(question="how does X work?")

## Tool Selection Guide

**I need documentation/context:**
→ get_markdown - semantic search across indexed docs

**I need to understand code structure:**
→ outline - list functions/classes in a file
→ find_symbol - get source code of a specific function/class

**I need to trace code relationships:**
→ definition - jump to where something is defined
→ references - find all usages of a symbol
→ expand - explore call graphs and imports

**I have a complex question requiring investigation:**
→ deep_research - multi-step agent that searches, reads code, collects evidence

## Discovery (usually not needed)
- list_collections - see available doc collections
- list_documents - browse docs in a collection
""",
)

# Store user_id for the current request using contextvars (async-safe)
_current_user_id: ContextVar[uuid.UUID | None] = ContextVar("current_user_id", default=None)


def set_current_user_id(user_id: uuid.UUID | None) -> None:
    """Set the current user ID for authorization."""
    _current_user_id.set(user_id)


def get_current_user_id() -> uuid.UUID | None:
    """Get the current user ID."""
    return _current_user_id.get()


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate MCP requests with Bearer tokens and validate Origin."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Validate Bearer token and Origin header."""
        settings = get_settings()

        # Check Origin allowlist
        origin = request.headers.get("origin")
        allowed_origins = [o.strip() for o in settings.mcp_allowed_origins.split(",") if o.strip()]

        if allowed_origins and origin and origin not in allowed_origins:
            return JSONResponse(
                {"error": "Origin not allowed"},
                status_code=403,
            )

        # Check Bearer token
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Verify token against database
        user_id = await self._verify_token(token)
        if not user_id:
            return JSONResponse(
                {"error": "Invalid or revoked token"},
                status_code=401,
            )

        # Store user_id for tool handlers
        set_current_user_id(user_id)

        try:
            return await call_next(request)
        finally:
            # Clear user_id after request
            set_current_user_id(None)

    async def _verify_token(self, token: str) -> uuid.UUID | None:
        """Verify token against database and return user_id if valid."""
        async with get_db_session() as db:
            # Get all non-revoked tokens
            result = await db.execute(select(MCPApiToken).where(MCPApiToken.revoked_at.is_(None)))
            tokens = result.scalars().all()

            for db_token in tokens:
                if verify_api_token(token, db_token.token_hash):
                    # Update last_used_at
                    await db.execute(
                        update(MCPApiToken)
                        .where(MCPApiToken.id == db_token.id)
                        .values(last_used_at=datetime.now(UTC))
                    )
                    return db_token.user_id

        return None


@mcp.tool(name="list_collections")
async def list_collections(
    search: Annotated[str | None, "Optional search term to filter collections by name"] = None,
) -> str:
    """List available documentation collections. Usually not needed - just call get_markdown directly."""
    user_id = get_current_user_id()

    async with get_db_session() as db:
        # Build query for accessible collections
        query = select(
            Collection.id,
            Collection.name,
            Collection.slug,
            Collection.visibility,
            func.count(Source.id).label("source_count"),
        ).outerjoin(Source, Source.collection_id == Collection.id)

        # Filter by visibility/ownership
        if user_id:
            query = query.where(
                or_(
                    Collection.visibility == CollectionVisibility.GLOBAL,
                    Collection.owner_user_id == user_id,
                    Collection.id.in_(
                        select(CollectionMember.collection_id).where(
                            CollectionMember.user_id == user_id
                        )
                    ),
                )
            )
        else:
            query = query.where(Collection.visibility == CollectionVisibility.GLOBAL)

        # Optional search filter
        if search:
            query = query.where(
                or_(
                    Collection.name.ilike(f"%{search}%"),
                    Collection.slug.ilike(f"%{search}%"),
                )
            )

        query = query.group_by(Collection.id).order_by(Collection.name)
        result = await db.execute(query)
        rows = result.all()

    if not rows:
        return "# No Collections Found\n\nNo collections are available."

    lines = ["# Available Collections\n"]
    for row in rows:
        coll_id, name, slug, visibility, source_count = row
        lines.append(f"## {name}")
        lines.append(f"- **ID**: `{coll_id}`")
        lines.append(f"- **Slug**: {slug}")
        lines.append(f"- **Sources**: {source_count}")
        lines.append(f"- **Visibility**: {visibility.value}")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(name="list_documents")
async def list_documents(
    collection_id: Annotated[str, "The collection ID to list documents from"],
    topic: Annotated[str | None, "Optional topic/path filter (e.g., 'routing', 'api')"] = None,
    limit: Annotated[int, "Maximum number of documents to return"] = 50,
) -> str:
    """Browse documents in a collection. Usually not needed - just call get_markdown directly."""
    user_id = get_current_user_id()

    try:
        coll_uuid = uuid.UUID(collection_id)
    except ValueError:
        return f"# Error\n\nInvalid collection_id: {collection_id}"

    async with get_db_session() as db:
        # Verify access to collection
        coll_result = await db.execute(select(Collection).where(Collection.id == coll_uuid))
        collection = coll_result.scalar_one_or_none()

        if not collection:
            return "# Error\n\nCollection not found."

        # Check access
        has_access = collection.visibility == CollectionVisibility.GLOBAL
        if user_id and not has_access:
            if collection.owner_user_id == user_id:
                has_access = True
            else:
                member_result = await db.execute(
                    select(CollectionMember).where(
                        CollectionMember.collection_id == coll_uuid,
                        CollectionMember.user_id == user_id,
                    )
                )
                if member_result.scalar_one_or_none():
                    has_access = True

        if not has_access:
            return "# Error\n\nAccess denied to this collection."

        # Query documents
        query = (
            select(Document.id, Document.uri, Document.title, Source.url)
            .join(Source, Document.source_id == Source.id)
            .where(Source.collection_id == coll_uuid)
        )

        if topic:
            query = query.where(
                or_(
                    Document.title.ilike(f"%{topic}%"),
                    Document.uri.ilike(f"%{topic}%"),
                )
            )

        query = query.order_by(Document.title).limit(limit)
        result = await db.execute(query)
        rows = result.all()

    if not rows:
        msg = "# No Documents Found\n\nNo documents in collection"
        if topic:
            msg += f" matching topic '{topic}'"
        return msg + "."

    lines = [f"# Documents in {collection.name}\n"]
    if topic:
        lines.append(f"*Filtered by topic: {topic}*\n")

    for _doc_id, uri, title, _source_url in rows:
        lines.append(f"- **{title or 'Untitled'}**")
        lines.append(f"  - URI: `{uri}`")
        lines.append("")

    if len(rows) == limit:
        lines.append(f"*Showing first {limit} documents. Use topic filter to narrow results.*")

    return "\n".join(lines)


@mcp.tool(name="get_markdown")
async def get_context_markdown(
    query: Annotated[str, "Natural language question or search terms"],
    collection_id: Annotated[str | None, "Limit to specific collection (optional)"] = None,
    topic: Annotated[str | None, "Filter by topic path (e.g., 'api', 'hooks')"] = None,
    max_chunks: Annotated[int, "Number of results (1-50)"] = 10,
    max_tokens: Annotated[int, "Response length limit"] = 4000,
    offset: Annotated[int, "Pagination offset"] = 0,
    raw: Annotated[bool, "Return raw chunks without LLM synthesis (faster)"] = False,
) -> str:
    """Semantic search across all indexed documentation and code. Returns synthesized context."""
    user_id = get_current_user_id()

    # Parse collection_id if provided
    collection_uuid: uuid.UUID | None = None
    if collection_id:
        try:
            collection_uuid = uuid.UUID(collection_id)
        except ValueError:
            return f"# Error\n\nInvalid collection_id: {collection_id}"

    try:
        # If raw mode or topic filter, we need to do the search ourselves
        if raw or topic:
            return await _get_raw_chunks(
                query=query,
                user_id=user_id,
                collection_id=collection_uuid,
                topic=topic,
                max_chunks=max_chunks,
                offset=offset,
            )

        # Standard LLM-assembled response
        response = await assemble_context(
            query=query,
            user_id=user_id,
            collection_id=collection_uuid,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )
        return response.markdown
    except Exception as e:
        return f"# Error\n\nFailed to retrieve context: {e!s}"


async def _get_raw_chunks(
    query: str,
    user_id: uuid.UUID | None,
    collection_id: uuid.UUID | None,
    topic: str | None,
    max_chunks: int,
    offset: int,
) -> str:
    """Get raw search results without LLM assembly."""
    settings = get_settings()

    # Get query embedding
    try:
        emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(emb_provider, emb_model)
    except (ValueError, Exception):
        embedder = FakeEmbedder()

    embed_result = await embedder.embed_batch([query])
    query_embedding = embed_result.embeddings[0]

    # Search with extra results to handle topic filtering and offset
    search_limit = max_chunks + offset + 50 if topic else max_chunks + offset
    search_response = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_id,
        top_k=search_limit,
    )

    results = search_response.results

    # Apply topic filter if specified
    if topic:
        topic_lower = topic.lower()
        results = [
            r for r in results if topic_lower in r.title.lower() or topic_lower in r.uri.lower()
        ]

    # Apply offset and limit
    results = results[offset : offset + max_chunks]

    if not results:
        msg = f"# No Results\n\nNo content found for query: {query}"
        if topic:
            msg += f" (topic: {topic})"
        return msg

    # Build markdown output
    lines = [f"# Search Results for: {query}\n"]
    if topic:
        lines.append(f"*Filtered by topic: {topic}*\n")

    seen_uris: set[str] = set()
    for i, result in enumerate(results, 1):
        lines.append(f"## Result {i + offset}: {result.title}")
        lines.append(f"*Source: {result.uri}*\n")
        lines.append(result.content)
        lines.append("")
        seen_uris.add(result.uri)

    # Add sources section
    lines.append("## Sources\n")
    for uri in seen_uris:
        lines.append(f"- {uri}")

    return "\n".join(lines)


# =============================================================================
# Deep Research Tool
# =============================================================================


@mcp.tool(name="deep_research")
async def code_deep_research(
    question: Annotated[str, "Complex question requiring multi-step investigation"],
    scope: Annotated[str | None, "Limit to path pattern (e.g., 'src/api/**')"] = None,
    budget: Annotated[int, "Max investigation steps (1-20)"] = 10,
    debug: Annotated[bool, "Return run_id for trace inspection"] = False,
) -> str:
    """Multi-step research agent for complex codebase questions.

    Use this for questions like "how does X work?" or "where is Y implemented?"
    The agent searches, reads code, and collects evidence autonomously.

    Returns concise answer with citations. For full trace: @research://runs/{run_id}/report.md
    """
    # Validate budget
    budget = max(1, min(20, budget))

    try:
        from contextmine_core.research import AgentConfig, ResearchAgent
        from contextmine_core.research.actions.finalize import format_answer_with_citations
        from contextmine_core.research.llm import get_research_llm_provider

        # Get LLM provider
        llm_provider = get_research_llm_provider()

        # Create and run agent
        config = AgentConfig(
            max_steps=budget,
            store_artifacts=True,
        )
        agent = ResearchAgent(
            llm_provider=llm_provider,
            config=config,
        )

        run = await agent.research(question=question, scope=scope)

        # Build output
        if run.answer:
            citations = [
                {
                    "id": e.id,
                    "file": e.file_path,
                    "lines": f"{e.start_line}-{e.end_line}",
                    "provenance": e.provenance,
                }
                for e in run.evidence
            ]
            output = format_answer_with_citations(run.answer, citations, max_length=800)
        else:
            output = f"Research failed: {run.error_message or 'Unknown error'}"

        # Add run_id if debug mode or include for artifact access
        if debug:
            output += f"\n\n**Run ID:** {run.run_id}"
            output += f"\n**Status:** {run.status.value}"
            output += f"\n**Steps:** {run.budget_used}/{run.budget_steps}"
            output += f"\n\nUse @research://runs/{run.run_id}/report.md for the full report."
        else:
            # Always include minimal reference
            output += f"\n\n*Run ID: {run.run_id[:8]}... (use debug=true for full trace)*"

        return output

    except Exception as e:
        return f"Research error: {e}"


# =============================================================================
# Code Navigation Tools
# =============================================================================


@mcp.tool(name="outline")
async def code_outline(
    file_path: Annotated[str, "Document URI from search results"],
    include_children: Annotated[bool, "Include methods inside classes"] = True,
) -> str:
    """List all functions, classes, and methods in a file with line numbers."""
    # First, try to get symbols from the database (for indexed documents)
    try:
        async with get_db_session() as session:
            # Look up document by URI (for git:// URIs) or by file path in meta
            result = await session.execute(
                select(Document)
                .where(
                    or_(
                        Document.uri == file_path,
                        Document.uri.contains(file_path),
                    )
                )
                .limit(1)
            )
            doc = result.scalar_one_or_none()

            if doc:
                # Get symbols from database
                result = await session.execute(
                    select(Symbol).where(Symbol.document_id == doc.id).order_by(Symbol.start_line)
                )
                db_symbols = result.scalars().all()

                if db_symbols:
                    # Format database symbols
                    lines = [f"# Outline: {file_path}\n"]
                    lines.append("*From index (fast lookup)*\n")

                    # Group by parent for hierarchy
                    top_level = [s for s in db_symbols if s.parent_name is None]
                    children_map: dict[str, list[Symbol]] = {}
                    for s in db_symbols:
                        if s.parent_name:
                            children_map.setdefault(s.parent_name, []).append(s)

                    for sym in top_level:
                        lines.append(
                            f"## {sym.kind.value} `{sym.name}` (L{sym.start_line}-{sym.end_line})"
                        )
                        if sym.signature:
                            lines.append(f"```\n{sym.signature}\n```")

                        if include_children and sym.qualified_name in children_map:
                            for child in children_map[sym.qualified_name]:
                                lines.append(
                                    f"  - {child.kind.value} `{child.name}` "
                                    f"(L{child.start_line}-{child.end_line})"
                                )
                                if child.signature:
                                    lines.append(f"    `{child.signature}`")

                        lines.append("")

                    lines.append(f"\n*Found {len(top_level)} top-level symbols*")
                    return "\n".join(lines)
    except Exception as e:
        return f"# Document Not Found\n\nNo indexed document matches: `{file_path}`\n\nError: {e}"

    # Document exists but has no symbols (not a code file, or unsupported language)
    return (
        f"# No symbols found in {file_path}\n\nDocument exists but contains no extractable symbols."
    )


@mcp.tool(name="find_symbol")
async def code_find_symbol(
    file_path: Annotated[str, "Document URI from search results"],
    name: Annotated[str, "Function, class, or method name"],
) -> str:
    """Get the source code of a specific function or class by name."""
    # First, try to find in database (for indexed documents)
    try:
        async with get_db_session() as session:
            # Look up document by URI or file path
            result = await session.execute(
                select(Document)
                .where(
                    or_(
                        Document.uri == file_path,
                        Document.uri.contains(file_path),
                    )
                )
                .limit(1)
            )
            doc = result.scalar_one_or_none()

            if doc:
                # Find symbol by name in this document
                result = await session.execute(
                    select(Symbol).where(
                        Symbol.document_id == doc.id,
                        Symbol.name == name,
                    )
                )
                db_symbol = result.scalar_one_or_none()

                if db_symbol:
                    # Extract content from document markdown
                    content_lines = doc.content_markdown.split("\n")
                    start_idx = max(0, db_symbol.start_line - 1)
                    end_idx = min(len(content_lines), db_symbol.end_line)
                    content = "\n".join(content_lines[start_idx:end_idx])

                    return f"""# {db_symbol.kind.value} `{db_symbol.name}`

**File:** {file_path}
**Lines:** {db_symbol.start_line}-{db_symbol.end_line}
*From index (fast lookup)*

```
{content}
```
"""
    except Exception as e:
        return f"# Document Not Found\n\nNo indexed document matches: `{file_path}`\n\nError: {e}"

    # Document exists but symbol not found
    return f"# Symbol Not Found\n\nNo symbol named `{name}` in indexed document: {file_path}"


@mcp.tool(name="definition")
async def code_definition(
    file_path: Annotated[str, "Document URI"],
    line: Annotated[int, "Line number (1-indexed)"],
    column: Annotated[int, "Column position (0-indexed)"],
) -> str:
    """Jump to where a symbol is defined. Provide the location of a reference to find its definition."""
    from pathlib import Path

    # Try LSP first
    try:
        from contextmine_core.lsp import LspNotAvailableError, get_lsp_manager

        manager = get_lsp_manager()
        client = await manager.get_client(file_path)

        # Use LSP for accurate definition lookup
        locations = await client.get_definition(file_path, line, column)

        if locations:
            location = locations[0]  # Take first definition
            # Read the definition content
            def_path = Path(location.file_path)
            if def_path.exists():
                file_lines = def_path.read_text().split("\n")
                start_idx = max(0, location.start_line - 1)
                end_idx = min(len(file_lines), location.end_line + 10)
                content = "\n".join(file_lines[start_idx:end_idx])

                return f"""# Definition Found

**File:** {location.file_path}
**Lines:** {location.start_line}-{location.end_line}

```
{content}
```
"""
            return f"""# Definition Found

**File:** {location.file_path}
**Lines:** {location.start_line}-{location.end_line}
"""

        # LSP returned nothing, fall through to tree-sitter
    except (ImportError, LspNotAvailableError):
        # LSP not available, fall through to tree-sitter
        pass
    except Exception:
        # Log but continue to fallback
        pass

    # Fallback: Try tree-sitter to find enclosing symbol
    try:
        from contextmine_core.treesitter import find_enclosing_symbol, get_symbol_content

        symbol = find_enclosing_symbol(file_path, line)
        if symbol:
            content = get_symbol_content(symbol)
            return f"""# Enclosing Symbol (LSP not available)

**File:** {file_path}
**Symbol:** {symbol.kind.value} `{symbol.name}`
**Lines:** {symbol.start_line}-{symbol.end_line}

```
{content}
```

*Note: For accurate go-to-definition, an LSP server is needed.*
"""
    except Exception:
        pass

    return f"# Definition Not Found\n\nCould not find definition at {file_path}:{line}:{column}"


@mcp.tool(name="references")
async def code_references(
    file_path: Annotated[str, "Document URI"],
    line: Annotated[int, "Line number (1-indexed)"],
    column: Annotated[int, "Column position (0-indexed)"],
    limit: Annotated[int, "Max results"] = 20,
) -> str:
    """Find all usages of a symbol. Use for impact analysis before refactoring."""
    try:
        from contextmine_core.lsp import LspNotAvailableError, get_lsp_manager

        manager = get_lsp_manager()
        client = await manager.get_client(file_path)

        locations = await client.get_references(file_path, line, column)

        if not locations:
            return (
                f"# No References Found\n\nNo references to symbol at {file_path}:{line}:{column}"
            )

        output_lines = [f"# References ({len(locations)} found)\n"]

        for i, loc in enumerate(locations[:limit]):
            output_lines.append(f"## {i + 1}. {loc.file_path}:{loc.start_line}")
            output_lines.append(f"Lines {loc.start_line}-{loc.end_line}\n")

        if len(locations) > limit:
            output_lines.append(f"\n*Showing {limit} of {len(locations)} references*")

        return "\n".join(output_lines)

    except (ImportError, LspNotAvailableError):
        # LSP not available
        return """# LSP Not Available

To find references, an LSP server needs to be running for this language.

**Alternative:** Use `context.get_markdown` with the symbol name to search
for usages across the indexed codebase.

Example:
```
context.get_markdown(query="<symbol_name>", raw=True)
```
"""

    except Exception as e:
        return f"# Error\n\nFailed to find references: {e}"


@mcp.tool(name="expand")
async def code_expand(
    seeds: Annotated[
        list[str],
        "Starting points as 'file_uri::function_name' (e.g., 'git://repo/main.py::process_data')",
    ],
    depth: Annotated[int, "Hops to follow (1-3)"] = 2,
    edge_types: Annotated[
        list[str] | None, "Filter: 'calls', 'called_by', 'imports', 'inherits'"
    ] = None,
    limit: Annotated[int, "Max nodes"] = 30,
) -> str:
    """Explore code relationships from a starting symbol. Shows what it calls, what calls it, imports, etc.

    Seeds should be in format: file_path::symbol_name
    Example: src/auth.py::authenticate
    """
    if not seeds:
        return "# Error\n\nNo seed symbols provided. Use format: file_path::symbol_name"

    try:
        from contextmine_core.graph import EdgeType, get_graph_builder

        builder = get_graph_builder()

        if not builder.has_treesitter:
            return "# Error\n\nTree-sitter required for graph expansion but not available."

        # Extract file paths from seeds
        file_paths = set()
        for seed in seeds:
            if "::" in seed:
                file_path = seed.split("::")[0]
                file_paths.add(file_path)
            else:
                return f"# Error\n\nInvalid seed format: {seed}\nExpected: file_path::symbol_name"

        # Build graph
        graph = builder.build_multi_file_graph(list(file_paths))

        if graph.node_count() == 0:
            return f"# No Symbols Found\n\nNo symbols extracted from: {', '.join(file_paths)}"

        # Parse edge types
        parsed_edge_types = None
        if edge_types:
            parsed_edge_types = []
            for et in edge_types:
                try:
                    parsed_edge_types.append(EdgeType(et.lower()))
                except ValueError:
                    return f"# Error\n\nUnknown edge type: {et}\nValid: calls, called_by, imports, imported_by, inherits, inherited_by"

        # Find valid seeds
        all_node_ids = {n.id for n in graph.get_all_nodes()}
        valid_seeds = [s for s in seeds if s in all_node_ids]

        if not valid_seeds:
            # Try partial match
            for seed in seeds:
                symbol_name = seed.split("::")[-1] if "::" in seed else seed
                for node_id in all_node_ids:
                    if symbol_name in node_id:
                        valid_seeds.append(node_id)
                        break

        if not valid_seeds:
            available = list(all_node_ids)[:10]
            return f"""# Seed Symbols Not Found

The provided seeds were not found in the graph.

**Available symbols (first 10):**
{chr(10).join(f"- {s}" for s in available)}

**Tip:** Use `code.outline` to find exact symbol IDs.
"""

        # Expand graph
        from contextmine_core.graph import expand_graph

        subgraph = expand_graph(
            graph=graph,
            seeds=valid_seeds,
            depth=min(depth, 3),  # Cap depth at 3
            edge_types=parsed_edge_types,
        )

        nodes = list(subgraph.get_all_nodes())[:limit]
        edges = list(subgraph.get_all_edges())

        # Format output
        lines = ["# Graph Expansion\n"]
        lines.append(f"**Seeds:** {', '.join(valid_seeds)}")
        lines.append(f"**Depth:** {depth}")
        lines.append(f"**Nodes:** {len(nodes)} | **Edges:** {len(edges)}\n")

        lines.append("## Symbols\n")
        for node in nodes:
            lines.append(f"- **{node.kind}** `{node.name}` ({node.file_path}:{node.start_line})")

        if edges:
            lines.append("\n## Relationships\n")
            for edge in edges[:30]:  # Limit edges shown
                lines.append(
                    f"- {edge.source_id.split('::')[-1]} → {edge.edge_type.value} → {edge.target_id.split('::')[-1]}"
                )

        return "\n".join(lines)

    except ImportError as e:
        return f"# Error\n\nRequired module not available: {e}"
    except Exception as e:
        return f"# Error\n\nGraph expansion failed: {e}"


def get_context_markdown_sync(query: str) -> str:
    """Synchronous wrapper for testing. Returns a simple placeholder."""
    return f"""# Context for: {query}

## Summary

This is a test placeholder. Use the async MCP endpoint for real results.

## Sources

- No sources (test mode)
"""


# =============================================================================
# MCP Resources for Research Agent Artifacts
# =============================================================================
#
# These resources expose research run artifacts (traces, evidence, reports)
# so that Claude Code users can inspect them via @ mentions without polluting
# the main context with large outputs.
#
# URI Scheme:
#   research://runs                         - List recent runs
#   research://runs/{run_id}/trace.json     - Agent execution trace
#   research://runs/{run_id}/evidence.json  - Collected evidence
#   research://runs/{run_id}/report.md      - Human-readable report


@mcp.resource("research://runs")
async def list_research_runs() -> str:
    """List recent research runs.

    Returns a JSON list of research run metadata including run_id,
    question, status, and timestamps. Use the run_id to fetch
    individual artifacts.
    """
    from contextmine_core.research import get_artifact_store

    store = get_artifact_store()
    runs = store.list_runs(limit=20)

    result = []
    for meta in runs:
        result.append(
            {
                "run_id": meta.run_id,
                "question": meta.question,
                "status": meta.status,
                "created_at": meta.created_at.isoformat(),
                "completed_at": meta.completed_at.isoformat() if meta.completed_at else None,
            }
        )

    return json.dumps(result, indent=2)


@mcp.resource("research://runs/{run_id}/trace.json")
async def get_research_trace(run_id: str) -> str:
    """Get the execution trace for a research run.

    The trace shows each step the agent took, including:
    - Action name and input parameters
    - Output summary and timing
    - Any errors encountered
    - Evidence IDs collected in each step
    """
    from contextmine_core.research import get_artifact_store

    store = get_artifact_store()
    trace = store.get_trace(run_id)

    if trace is None:
        return json.dumps({"error": f"Run not found: {run_id}"})

    return json.dumps(trace, indent=2)


@mcp.resource("research://runs/{run_id}/evidence.json")
async def get_research_evidence(run_id: str) -> str:
    """Get the evidence collected during a research run.

    Evidence includes code snippets and documentation spans that
    support the answer, with:
    - File path and line range
    - Content excerpt
    - Reason for selection
    - How it was found (bm25, vector, lsp, graph, manual)
    - Symbol information if available
    """
    from contextmine_core.research import get_artifact_store

    store = get_artifact_store()
    evidence = store.get_evidence(run_id)

    if evidence is None:
        return json.dumps({"error": f"Run not found: {run_id}"})

    return json.dumps(evidence, indent=2)


@mcp.resource("research://runs/{run_id}/report.md")
async def get_research_report(run_id: str) -> str:
    """Get the markdown report for a research run.

    The report provides a human-readable summary including:
    - The original question and answer
    - All evidence with file locations and excerpts
    - Step-by-step trace of the investigation
    """
    from contextmine_core.research import get_artifact_store

    store = get_artifact_store()
    report = store.get_report(run_id)

    if report is None:
        return f"# Error\n\nRun not found: {run_id}"

    return report


# Get the HTTP app from FastMCP with root path (mounted at /mcp in main.py)
# Use stateless_http=True for simpler API testing without session management
_mcp_http_app = mcp.http_app(path="/", stateless_http=True)

# Export the MCP lifespan for integration with FastAPI
mcp_lifespan = _mcp_http_app.lifespan


# Create Starlette app with auth middleware wrapping the MCP HTTP app
mcp_app = Starlette(
    routes=[
        Mount("/", app=_mcp_http_app),
    ],
    middleware=[
        Middleware(MCPAuthMiddleware),
    ],
)


# Export tool list for tests
def get_tools() -> list[dict]:
    """Get list of available tools for testing."""
    return [
        {
            "name": "context.list_collections",
            "description": "List all documentation collections available to search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "Optional search term to filter collections by name",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "context.list_documents",
            "description": "List documents in a collection to explore available content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_id": {
                        "type": "string",
                        "description": "The collection ID to list documents from",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional topic/path filter",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return",
                        "default": 50,
                    },
                },
                "required": ["collection_id"],
            },
        },
        {
            "name": "context.get_markdown",
            "description": "Search and retrieve documentation context as Markdown.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documentation",
                    },
                    "collection_id": {
                        "type": "string",
                        "description": "Collection ID to search within",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Focus on specific topic like 'routing' or 'hooks'",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "description": "Maximum number of chunks to retrieve",
                        "default": 10,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for LLM response",
                        "default": 4000,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip first N results for pagination",
                        "default": 0,
                    },
                    "raw": {
                        "type": "boolean",
                        "description": "Return raw chunks instead of LLM-assembled response",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    ]
