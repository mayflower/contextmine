"""MCP server implementation with FastMCP 2 and Streamable HTTP transport."""

import json
import uuid
from typing import Annotated

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Document,
    Source,
    Symbol,
    SymbolEdge,
    get_settings,
)
from contextmine_core import get_session as get_db_session
from contextmine_core.context import assemble_context
from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec
from contextmine_core.search import hybrid_search
from fastmcp import FastMCP
from sqlalchemy import func, or_, select

from app.mcp_auth import ContextMineGitHubProvider, get_current_user_id


def escape_like_pattern(value: str) -> str:
    """Escape special characters in LIKE patterns to prevent SQL injection."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _resolve_collection_access(
    db,
    *,
    collection_id: str,
    user_id: uuid.UUID | None,
) -> tuple[Collection | None, str | None]:
    try:
        collection_uuid = uuid.UUID(collection_id)
    except ValueError:
        return None, f"Invalid collection_id: {collection_id}"

    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_uuid))
    ).scalar_one_or_none()
    if not collection:
        return None, "Collection not found."

    has_access = collection.visibility == CollectionVisibility.GLOBAL
    if user_id and not has_access:
        if collection.owner_user_id == user_id:
            has_access = True
        else:
            member_result = await db.execute(
                select(CollectionMember).where(
                    CollectionMember.collection_id == collection_uuid,
                    CollectionMember.user_id == user_id,
                )
            )
            has_access = member_result.scalar_one_or_none() is not None

    if not has_access:
        return None, "Access denied to this collection."
    return collection, None


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _node_kind_in_scope(kind: str, scope: str) -> bool:
    normalized = kind.strip().lower()
    if scope == "all":
        return True
    test_kinds = {"test_suite", "test_case", "test_fixture"}
    ui_kinds = {"ui_route", "ui_view", "ui_component", "interface_contract"}
    flow_kinds = {"user_flow", "flow_step"}
    if scope == "tests":
        return normalized in test_kinds
    if scope == "ui":
        return normalized in ui_kinds
    if scope == "flows":
        return normalized in flow_kinds
    if scope == "code":
        return normalized not in (test_kinds | ui_kinds | flow_kinds)
    return True


def _filter_graph_payload(
    graph: dict,
    *,
    scope: str = "all",
    provenance_mode: str | None = None,
    include_test_links: bool | None = None,
    include_ui_links: bool | None = None,
) -> dict:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])

    if scope != "all":
        allowed_ids = {
            str(node.get("id"))
            for node in nodes
            if _node_kind_in_scope(str(node.get("kind") or ""), scope)
        }
        nodes = [node for node in nodes if str(node.get("id")) in allowed_ids]
        edges = [
            edge
            for edge in edges
            if str(edge.get("source_node_id")) in allowed_ids
            and str(edge.get("target_node_id")) in allowed_ids
        ]

    if provenance_mode:
        mode = provenance_mode.strip().lower()
        if mode in {"deterministic", "inferred"}:
            allowed_ids = {
                str(node.get("id"))
                for node in nodes
                if str(((node.get("meta") or {}).get("provenance") or {}).get("mode") or "").lower()
                == mode
            }
            nodes = [node for node in nodes if str(node.get("id")) in allowed_ids]
            edges = [
                edge
                for edge in edges
                if str(edge.get("source_node_id")) in allowed_ids
                and str(edge.get("target_node_id")) in allowed_ids
            ]

    if include_test_links is False:
        edges = [
            edge for edge in edges if not str(edge.get("kind") or "").lower().startswith("test_")
        ]
    if include_ui_links is False:
        edges = [
            edge for edge in edges if not str(edge.get("kind") or "").lower().startswith("ui_")
        ]

    return {
        **graph,
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
    }


async def _resolve_collection_for_tool(
    db,
    *,
    collection_id: str | None,
    user_id: uuid.UUID | None,
) -> tuple[Collection | None, str | None]:
    if collection_id:
        return await _resolve_collection_access(db, collection_id=collection_id, user_id=user_id)

    query = select(Collection)
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

    collection = (
        await db.execute(query.order_by(Collection.created_at.desc()).limit(1))
    ).scalar_one_or_none()
    if not collection:
        return None, "No accessible collection found."
    return collection, None


# Create auth provider (uses GitHub OAuth)
settings = get_settings()
try:
    auth = ContextMineGitHubProvider()
except ValueError as e:
    # GitHub OAuth not configured
    if not settings.debug:
        raise RuntimeError(
            "MCP authentication required in production. "
            "Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET, or enable DEBUG mode for testing."
        ) from e
    auth = None

# Create FastMCP server with auth (if configured)
mcp = FastMCP(
    auth=auth,
    name="contextmine",
    instructions="""ContextMine - research assistant for documentation, code, and architecture.

## Quick Start
For simple questions: get_markdown(query="how do I X?")
For complex investigations: deep_research(question="how does X work?")
For knowledge graph queries: graph_rag(query="how do X and Y relate?")

## Primary Research Tool

**graph_rag(query)** - Graph-augmented retrieval with global + local context:
- Global: Community summaries (high-level architecture understanding)
- Local: Entity nodes (symbols, files, rules, tables) + graph expansion
- Citations: Evidence from source files

Examples:
→ graph_rag(query="how does auth work?") - combines community summaries + entity context
→ graph_rag(query="explain payment flow", format="json") - get structured output

## Research Tools

**I have a question about code/architecture:**
→ graph_rag(query="how does auth work?") - PRIMARY TOOL
→ deep_research(question="explain the payment flow") - multi-step investigation agent

**I need to understand validation/business rules:**
→ research_validation(code_path="auth.py") - find validation rules for specific code
→ research_validation(code_path="payment") - find payment-related business rules

**I need to understand data model:**
→ research_data_model(entity="users") - research tables, columns, relationships

**I need to understand architecture:**
→ research_architecture(topic="deployment") - infrastructure, CI/CD, jobs
→ research_architecture(topic="security") - auth patterns, access control
→ research_architecture(topic="api") - REST, GraphQL, RPC endpoints

## Code Navigation (for specific files/symbols)

**I need code structure of a specific file:**
→ outline(file_path="src/auth.py") - list functions/classes
→ find_symbol(file_path="src/auth.py", name="authenticate") - get source code

**I need to trace relationships:**
→ definition(file_path, line, column) - jump to definition
→ references(file_path, line, column) - find all usages
→ expand(seeds=["auth.py::login"], depth=2) - explore call graph

## Graph Exploration (for specific node IDs)

→ graph_neighborhood(node_id="...") - explore around a known node
→ trace_path(from_node_id="...", to_node_id="...") - find connection between nodes

## Discovery (usually not needed)
- list_collections - see available doc collections
- list_documents - browse docs in a collection
""",
)


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
            escaped_search = escape_like_pattern(search)
            query = query.where(
                or_(
                    Collection.name.ilike(f"%{escaped_search}%", escape="\\"),
                    Collection.slug.ilike(f"%{escaped_search}%", escape="\\"),
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
            escaped_topic = escape_like_pattern(topic)
            query = query.where(
                or_(
                    Document.title.ilike(f"%{escaped_topic}%", escape="\\"),
                    Document.uri.ilike(f"%{escaped_topic}%", escape="\\"),
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
        from contextmine_core.research import (
            AgentConfig,
            ResearchAgent,
            format_answer_with_citations,
        )
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
    """Jump to where a symbol is defined. Uses pre-indexed symbol data."""
    async with get_db_session() as db:
        # Find the document matching the file path
        escaped_path = escape_like_pattern(file_path)
        doc_query = select(Document).where(Document.uri.ilike(f"%{escaped_path}%", escape="\\"))
        doc_result = await db.execute(doc_query)
        doc = doc_result.scalar_one_or_none()

        if not doc:
            return f"# Document Not Found\n\nNo indexed document matches: `{file_path}`"

        # Find the symbol that contains this line (smallest enclosing symbol)
        symbol_query = (
            select(Symbol)
            .where(Symbol.document_id == doc.id)
            .where(Symbol.start_line <= line)
            .where(Symbol.end_line >= line)
            .order_by(Symbol.end_line - Symbol.start_line)  # Smallest first
        )
        symbol_result = await db.execute(symbol_query)
        symbol = symbol_result.scalars().first()

        if not symbol:
            return f"# No Symbol Found\n\nNo indexed symbol at {file_path}:{line}"

        # Get the content from the document's markdown
        content_lines = (doc.content_markdown or "").split("\n")
        start_idx = max(0, symbol.start_line - 1)
        end_idx = min(len(content_lines), symbol.end_line + 5)
        content = "\n".join(content_lines[start_idx:end_idx])

        # Build response
        docstring = symbol.meta.get("docstring", "") if symbol.meta else ""
        signature_info = f"**Signature:** `{symbol.signature}`\n" if symbol.signature else ""
        docstring_info = f"**Docstring:** {docstring}\n" if docstring else ""

        return f"""# Definition Found

**File:** {doc.uri}
**Symbol:** {symbol.kind.value} `{symbol.qualified_name}`
**Lines:** {symbol.start_line}-{symbol.end_line}
{signature_info}{docstring_info}
```
{content}
```
"""


@mcp.tool(name="references")
async def code_references(
    file_path: Annotated[str, "Document URI"],
    line: Annotated[int, "Line number (1-indexed)"],
    column: Annotated[int, "Column position (0-indexed)"],
    limit: Annotated[int, "Max results"] = 20,
) -> str:
    """Find all usages of a symbol. Uses pre-indexed symbol edge data."""
    from sqlalchemy.orm import selectinload

    async with get_db_session() as db:
        # Find the document matching the file path
        escaped_path = escape_like_pattern(file_path)
        doc_query = select(Document).where(Document.uri.ilike(f"%{escaped_path}%", escape="\\"))
        doc_result = await db.execute(doc_query)
        doc = doc_result.scalar_one_or_none()

        if not doc:
            return f"# Document Not Found\n\nNo indexed document matches: `{file_path}`"

        # Find the symbol at this location (smallest enclosing)
        symbol_query = (
            select(Symbol)
            .where(Symbol.document_id == doc.id)
            .where(Symbol.start_line <= line)
            .where(Symbol.end_line >= line)
            .order_by(Symbol.end_line - Symbol.start_line)  # Smallest first
        )
        symbol_result = await db.execute(symbol_query)
        target_symbol = symbol_result.scalars().first()

        if not target_symbol:
            return f"# No Symbol Found\n\nNo indexed symbol at {file_path}:{line}"

        # Find all edges where this symbol is the target (incoming references)
        edge_query = (
            select(SymbolEdge)
            .options(selectinload(SymbolEdge.source_symbol).selectinload(Symbol.document))
            .where(SymbolEdge.target_symbol_id == target_symbol.id)
            .limit(limit + 10)  # Get a few extra in case of filtering
        )
        edge_result = await db.execute(edge_query)
        edges = edge_result.scalars().all()

        if not edges:
            return f"# No References Found\n\nNo indexed references to `{target_symbol.qualified_name}`"

        output_lines = [f"# References to `{target_symbol.qualified_name}` ({len(edges)} found)\n"]

        for i, edge in enumerate(edges[:limit]):
            source = edge.source_symbol
            if source and source.document:
                ref_file = source.document.uri
                ref_line = edge.source_line or source.start_line
                output_lines.append(f"## {i + 1}. {source.qualified_name} ({edge.edge_type.value})")
                output_lines.append(f"**File:** {ref_file}:{ref_line}\n")

        if len(edges) > limit:
            output_lines.append(f"\n*Showing {limit} of {len(edges)} references*")

        return "\n".join(output_lines)


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


# =============================================================================
# Knowledge Graph Tools (Business Rules, ERD, Surfaces, GraphRAG)
# =============================================================================


@mcp.tool(name="research_validation")
async def research_validation(
    code_path: Annotated[str, "File path or function name to find validation rules for"],
    collection_id: Annotated[str | None, "Filter to specific collection"] = None,
) -> str:
    """Find validation rules, business logic, and constraints for specific code.

    Use this to understand what validation/authorization rules apply to a
    file, function, or feature. Returns business rules, their source code,
    and the conditions that trigger validation failures.

    Example queries:
    - "auth.py" - find rules in the auth module
    - "create_user" - find rules for user creation
    - "payment" - find payment validation rules
    """
    user_id = get_current_user_id()

    try:
        from contextmine_core.models import (
            KnowledgeEvidence,
            KnowledgeNode,
            KnowledgeNodeEvidence,
            KnowledgeNodeKind,
        )
        from contextmine_core.search import get_accessible_collection_ids

        async with get_db_session() as db:
            if collection_id:
                collection_ids = [uuid.UUID(collection_id)]
            else:
                collection_ids = await get_accessible_collection_ids(db, user_id)

            if not collection_ids:
                return "# No Collections Available\n\nNo accessible collections found."

            code_path_lower = code_path.lower()
            query_words = set(code_path_lower.replace("/", " ").replace("_", " ").split())

            # Find business rules matching the code path
            stmt = select(KnowledgeNode).where(
                KnowledgeNode.collection_id.in_(collection_ids),
                KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
            )
            result = await db.execute(stmt)
            all_rules = result.scalars().all()

            # Also find rule candidates (unlabeled) for more context
            stmt = select(KnowledgeNode).where(
                KnowledgeNode.collection_id.in_(collection_ids),
                KnowledgeNode.kind == KnowledgeNodeKind.RULE_CANDIDATE,
            )
            result = await db.execute(stmt)
            all_candidates = result.scalars().all()

            # Match rules by evidence file path or rule content
            matched_rules = []
            for rule in all_rules:
                meta = rule.meta or {}
                searchable = " ".join(
                    [
                        rule.name.lower(),
                        meta.get("natural_language", "").lower(),
                        meta.get("container_name", "").lower(),
                    ]
                ).replace("_", " ")

                # Check if matches by name or content
                if any(word in searchable for word in query_words if len(word) > 2):
                    matched_rules.append(rule)
                    continue

                # Check evidence file paths
                ev_stmt = (
                    select(KnowledgeEvidence)
                    .join(
                        KnowledgeNodeEvidence,
                        KnowledgeNodeEvidence.evidence_id == KnowledgeEvidence.id,
                    )
                    .where(KnowledgeNodeEvidence.node_id == rule.id)
                )
                ev_result = await db.execute(ev_stmt)
                for ev in ev_result.scalars().all():
                    if code_path_lower in ev.file_path.lower():
                        matched_rules.append(rule)
                        break

            # Match candidates similarly
            matched_candidates = []
            for candidate in all_candidates:
                meta = candidate.meta or {}
                searchable = " ".join(
                    [
                        candidate.name.lower(),
                        meta.get("container_name", "").lower(),
                        meta.get("file_path", "").lower(),
                    ]
                ).replace("_", " ")

                if any(word in searchable for word in query_words if len(word) > 2):
                    matched_candidates.append(candidate)

            if not matched_rules and not matched_candidates:
                return f"# No Validation Rules Found\n\nNo business rules or validation logic found for: `{code_path}`"

            lines = [f"# Validation Rules for: {code_path}\n"]

            if matched_rules:
                lines.append(f"## Business Rules ({len(matched_rules)} found)\n")
                for rule in matched_rules[:10]:
                    meta = rule.meta or {}
                    lines.append(f"### {rule.name}")
                    lines.append(
                        f"- **Category**: {meta.get('category', 'unknown')} | **Severity**: {meta.get('severity', 'unknown')}"
                    )
                    if meta.get("natural_language"):
                        lines.append(f"- **Rule**: {meta['natural_language']}")
                    if meta.get("predicate"):
                        lines.append(f"- **Condition**: `{meta['predicate'][:100]}`")
                    if meta.get("failure"):
                        lines.append(f"- **On Failure**: `{meta['failure'][:100]}`")
                    lines.append("")

            if matched_candidates:
                lines.append(f"## Validation Candidates ({len(matched_candidates)} unlabeled)\n")
                lines.append(
                    "*These are detected validation patterns not yet labeled as business rules.*\n"
                )
                for candidate in matched_candidates[:5]:
                    meta = candidate.meta or {}
                    lines.append(f"### {meta.get('container_name', candidate.name)}")
                    lines.append(f"- **File**: `{meta.get('file_path', 'unknown')}`")
                    if meta.get("predicate"):
                        lines.append(f"- **Condition**: `{meta['predicate'][:150]}`")
                    if meta.get("failure"):
                        lines.append(f"- **Failure**: `{meta['failure'][:100]}`")
                    lines.append("")

            return "\n".join(lines)

    except Exception as e:
        return f"# Error\n\nFailed to research validation: {e}"


@mcp.tool(name="research_data_model")
async def research_data_model(
    entity: Annotated[str, "Table name, entity, or data concept to research"],
    collection_id: Annotated[str | None, "Filter to specific collection"] = None,
) -> str:
    """Research the data model for a specific entity or concept.

    Returns database tables, columns, relationships, and related API endpoints
    for the given entity. Use this to understand how data is structured.

    Example queries:
    - "users" - find the users table and related entities
    - "order" - find order-related tables and APIs
    - "authentication" - find auth-related data structures
    """
    user_id = get_current_user_id()

    try:
        from contextmine_core.models import (
            KnowledgeArtifact,
            KnowledgeArtifactKind,
            KnowledgeNode,
            KnowledgeNodeKind,
        )
        from contextmine_core.search import get_accessible_collection_ids

        async with get_db_session() as db:
            if collection_id:
                collection_ids = [uuid.UUID(collection_id)]
            else:
                collection_ids = await get_accessible_collection_ids(db, user_id)

            if not collection_ids:
                return "# No Collections Available\n\nNo accessible collections found."

            entity_lower = entity.lower()
            query_words = set(entity_lower.replace("_", " ").split())

            # Find matching tables
            stmt = select(KnowledgeNode).where(
                KnowledgeNode.collection_id.in_(collection_ids),
                KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
            )
            result = await db.execute(stmt)
            tables = [
                t
                for t in result.scalars().all()
                if any(word in t.name.lower() for word in query_words if len(word) > 2)
            ]

            # Find matching columns
            stmt = select(KnowledgeNode).where(
                KnowledgeNode.collection_id.in_(collection_ids),
                KnowledgeNode.kind == KnowledgeNodeKind.DB_COLUMN,
            )
            result = await db.execute(stmt)
            columns = [
                c
                for c in result.scalars().all()
                if any(word in c.name.lower() for word in query_words if len(word) > 2)
            ]

            # Find related API endpoints
            stmt = select(KnowledgeNode).where(
                KnowledgeNode.collection_id.in_(collection_ids),
                KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
            )
            result = await db.execute(stmt)
            endpoints = []
            for ep in result.scalars().all():
                meta = ep.meta or {}
                searchable = " ".join([ep.name.lower(), meta.get("path", "").lower()])
                if any(word in searchable for word in query_words if len(word) > 2):
                    endpoints.append(ep)

            # Get ERD if available
            stmt = (
                select(KnowledgeArtifact)
                .where(
                    KnowledgeArtifact.collection_id.in_(collection_ids),
                    KnowledgeArtifact.kind == KnowledgeArtifactKind.MERMAID_ERD,
                )
                .order_by(KnowledgeArtifact.created_at.desc())
                .limit(1)
            )
            result = await db.execute(stmt)
            erd_artifact = result.scalar_one_or_none()

            if not tables and not columns and not endpoints:
                return f"# No Data Model Found\n\nNo tables, columns, or APIs found for: `{entity}`"

            lines = [f"# Data Model: {entity}\n"]

            if tables:
                lines.append(f"## Database Tables ({len(tables)} found)\n")
                for table in tables[:10]:
                    meta = table.meta or {}
                    pk = f" (PK: `{meta.get('primary_key')}`)" if meta.get("primary_key") else ""
                    lines.append(f"### {table.name}{pk}")
                    lines.append(f"- **Columns**: {meta.get('column_count', 'unknown')}")
                    if meta.get("columns"):
                        for col in meta["columns"][:8]:
                            nullable = "" if col.get("nullable", True) else " NOT NULL"
                            lines.append(f"  - `{col['name']}`: {col.get('type', '?')}{nullable}")
                    lines.append("")

            if columns:
                # Group columns by table
                cols_by_table: dict[str, list] = {}
                for col in columns:
                    table_name = (
                        col.natural_key.split(":")[1] if ":" in col.natural_key else "unknown"
                    )
                    cols_by_table.setdefault(table_name, []).append(col)

                lines.append(f"## Related Columns ({len(columns)} found)\n")
                for table_name, cols in cols_by_table.items():
                    lines.append(f"**{table_name}**:")
                    for col in cols[:5]:
                        meta = col.meta or {}
                        lines.append(f"- `{col.name}`: {meta.get('type', '?')}")
                    lines.append("")

            if endpoints:
                lines.append(f"## Related API Endpoints ({len(endpoints)} found)\n")
                for ep in endpoints[:8]:
                    meta = ep.meta or {}
                    lines.append(f"- `{meta.get('method', 'GET')} {meta.get('path', ep.name)}`")
                lines.append("")

            if erd_artifact and entity_lower in erd_artifact.content.lower():
                lines.append("## Entity Relationship Diagram\n")
                lines.append("```mermaid")
                # Extract relevant portion of ERD
                lines.append(erd_artifact.content)
                lines.append("```\n")

            return "\n".join(lines)

    except Exception as e:
        return f"# Error\n\nFailed to research data model: {e}"


@mcp.tool(name="research_architecture")
async def research_architecture(
    topic: Annotated[
        str,
        "Architecture topic to research: deployment|security|api|database|ui|tests|flows|rebuild",
    ],
    collection_id: Annotated[str | None, "Filter to specific collection"] = None,
) -> str:
    """Research system architecture for a specific topic.

    Returns relevant architecture documentation, components, patterns,
    and deployment information. Use this to understand how the system
    is designed and structured.

    Example topics:
    - "deployment" - infrastructure, jobs, CI/CD
    - "security" - authentication, authorization patterns
    - "api" - REST endpoints, GraphQL, RPC services
    - "database" - data model, tables, relationships
    - "ui" - screens, routes, and component composition
    - "tests" - suites, cases, fixtures, and obligations
    - "flows" - user flows and step evidence
    - "rebuild" - replacement readiness and known gaps
    """
    user_id = get_current_user_id()

    try:
        from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind
        from contextmine_core.search import get_accessible_collection_ids

        async with get_db_session() as db:
            if collection_id:
                collection_ids = [uuid.UUID(collection_id)]
            else:
                collection_ids = await get_accessible_collection_ids(db, user_id)

            if not collection_ids:
                return "# No Collections Available\n\nNo accessible collections found."

            topic_lower = topic.lower()

            lines = [f"# Architecture: {topic}\n"]

            # Topic-specific knowledge
            if any(word in topic_lower for word in ["api", "endpoint", "rest", "http"]):
                stmt = (
                    select(KnowledgeNode)
                    .where(
                        KnowledgeNode.collection_id.in_(collection_ids),
                        KnowledgeNode.kind == KnowledgeNodeKind.API_ENDPOINT,
                    )
                    .limit(20)
                )
                result = await db.execute(stmt)
                endpoints = result.scalars().all()
                if endpoints:
                    lines.append(f"## API Endpoints ({len(endpoints)} total)\n")
                    for ep in endpoints[:10]:
                        meta = ep.meta or {}
                        lines.append(f"- `{meta.get('method', 'GET')} {meta.get('path', ep.name)}`")
                    lines.append("")

            if any(word in topic_lower for word in ["deploy", "job", "ci", "cd", "workflow"]):
                stmt = (
                    select(KnowledgeNode)
                    .where(
                        KnowledgeNode.collection_id.in_(collection_ids),
                        KnowledgeNode.kind == KnowledgeNodeKind.JOB,
                    )
                    .limit(20)
                )
                result = await db.execute(stmt)
                jobs = result.scalars().all()
                if jobs:
                    lines.append(f"## Jobs & Workflows ({len(jobs)} total)\n")
                    for job in jobs[:10]:
                        meta = job.meta or {}
                        schedule = (
                            f" (schedule: `{meta['schedule']}`)" if meta.get("schedule") else ""
                        )
                        lines.append(
                            f"- **{job.name}** ({meta.get('job_type', 'unknown')}){schedule}"
                        )
                    lines.append("")

            if any(word in topic_lower for word in ["database", "data", "table", "schema"]):
                stmt = (
                    select(KnowledgeNode)
                    .where(
                        KnowledgeNode.collection_id.in_(collection_ids),
                        KnowledgeNode.kind == KnowledgeNodeKind.DB_TABLE,
                    )
                    .limit(20)
                )
                result = await db.execute(stmt)
                tables = result.scalars().all()
                if tables:
                    lines.append(f"## Database Tables ({len(tables)} total)\n")
                    for table in tables[:10]:
                        meta = table.meta or {}
                        lines.append(f"- **{table.name}**: {meta.get('column_count', '?')} columns")
                    lines.append("")

            if any(word in topic_lower for word in ["security", "auth", "permission", "access"]):
                stmt = select(KnowledgeNode).where(
                    KnowledgeNode.collection_id.in_(collection_ids),
                    KnowledgeNode.kind == KnowledgeNodeKind.BUSINESS_RULE,
                )
                result = await db.execute(stmt)
                auth_rules = [
                    r
                    for r in result.scalars().all()
                    if (r.meta or {}).get("category") in ("authorization", "authentication")
                ]
                if auth_rules:
                    lines.append(f"## Security Rules ({len(auth_rules)} found)\n")
                    for rule in auth_rules[:5]:
                        meta = rule.meta or {}
                        lines.append(f"### {rule.name}")
                        if meta.get("natural_language"):
                            lines.append(f"- {meta['natural_language']}")
                        lines.append("")

            if any(word in topic_lower for word in ["ui", "screen", "view", "route"]):
                ui_stmt = select(KnowledgeNode).where(
                    KnowledgeNode.collection_id.in_(collection_ids),
                    KnowledgeNode.kind.in_(
                        [
                            KnowledgeNodeKind.UI_ROUTE,
                            KnowledgeNodeKind.UI_VIEW,
                            KnowledgeNodeKind.UI_COMPONENT,
                        ]
                    ),
                )
                ui_rows = (await db.execute(ui_stmt)).scalars().all()
                if ui_rows:
                    by_kind: dict[str, int] = {}
                    for row in ui_rows:
                        key = row.kind.value
                        by_kind[key] = by_kind.get(key, 0) + 1
                    lines.append("## UI Topology\n")
                    for key, count in sorted(by_kind.items()):
                        lines.append(f"- **{key}**: {count}")
                    lines.append("")

            if any(word in topic_lower for word in ["test", "qa", "verification"]):
                test_stmt = select(KnowledgeNode).where(
                    KnowledgeNode.collection_id.in_(collection_ids),
                    KnowledgeNode.kind.in_(
                        [
                            KnowledgeNodeKind.TEST_SUITE,
                            KnowledgeNodeKind.TEST_CASE,
                            KnowledgeNodeKind.TEST_FIXTURE,
                        ]
                    ),
                )
                test_rows = (await db.execute(test_stmt)).scalars().all()
                if test_rows:
                    lines.append("## Test Semantics\n")
                    lines.append(
                        f"- Suites: {sum(1 for row in test_rows if row.kind == KnowledgeNodeKind.TEST_SUITE)}"
                    )
                    lines.append(
                        f"- Cases: {sum(1 for row in test_rows if row.kind == KnowledgeNodeKind.TEST_CASE)}"
                    )
                    lines.append(
                        f"- Fixtures: {sum(1 for row in test_rows if row.kind == KnowledgeNodeKind.TEST_FIXTURE)}"
                    )
                    lines.append("")

            if any(word in topic_lower for word in ["flow", "journey", "rebuild"]):
                flow_stmt = select(KnowledgeNode).where(
                    KnowledgeNode.collection_id.in_(collection_ids),
                    KnowledgeNode.kind.in_(
                        [KnowledgeNodeKind.USER_FLOW, KnowledgeNodeKind.FLOW_STEP]
                    ),
                )
                flow_rows = (await db.execute(flow_stmt)).scalars().all()
                if flow_rows:
                    lines.append("## User Flows\n")
                    lines.append(
                        f"- User flows: {sum(1 for row in flow_rows if row.kind == KnowledgeNodeKind.USER_FLOW)}"
                    )
                    lines.append(
                        f"- Flow steps: {sum(1 for row in flow_rows if row.kind == KnowledgeNodeKind.FLOW_STEP)}"
                    )
                    lines.append("")

            if "rebuild" in topic_lower:
                inferred_critical = 0
                critical_kinds = {
                    KnowledgeNodeKind.INTERFACE_CONTRACT,
                    KnowledgeNodeKind.USER_FLOW,
                    KnowledgeNodeKind.FLOW_STEP,
                    KnowledgeNodeKind.TEST_CASE,
                    KnowledgeNodeKind.UI_ROUTE,
                }
                readiness_stmt = select(KnowledgeNode).where(
                    KnowledgeNode.collection_id.in_(collection_ids),
                    KnowledgeNode.kind.in_(list(critical_kinds)),
                )
                for row in (await db.execute(readiness_stmt)).scalars().all():
                    prov = (row.meta or {}).get("provenance") or {}
                    if (
                        str(prov.get("mode") or "") == "inferred"
                        and float(prov.get("confidence") or 0.0) < 0.65
                    ):
                        inferred_critical += 1
                lines.append("## Rebuild Readiness Signals\n")
                lines.append(f"- Critical inferred-only nodes: {inferred_critical}")
                lines.append("- Use `export_twin_view(format='twin_manifest')` for full handoff.")
                lines.append("")

            if len(lines) == 1:
                lines.append(f"No specific architecture information found for topic: `{topic}`\n")
                lines.append(
                    "Try topics like: api, deployment, database, security, ui, tests, flows"
                )

            return "\n".join(lines)

    except Exception as e:
        return f"# Error\n\nFailed to research architecture: {e}"


@mcp.tool(name="graph_neighborhood")
async def mcp_graph_neighborhood(
    node_id: Annotated[str, "The starting node ID"],
    depth: Annotated[int, "Expansion depth (1-3)"] = 1,
    edge_kinds: Annotated[list[str] | None, "Filter by edge kinds"] = None,
    limit: Annotated[int, "Maximum nodes to return"] = 30,
) -> str:
    """Explore the knowledge graph neighborhood around a node.

    Returns connected nodes and edges, useful for understanding
    relationships between files, symbols, tables, rules, etc.
    """
    try:
        node_uuid = uuid.UUID(node_id)
    except ValueError:
        return f"# Error\n\nInvalid node_id: {node_id}"

    try:
        from contextmine_core.graphrag import graph_neighborhood

        async with get_db_session() as db:
            result = await graph_neighborhood(
                session=db,
                node_id=node_uuid,
                collection_id=None,
                depth=min(depth, 3),
                edge_kinds=edge_kinds,
                max_nodes=limit,
            )

            if not result.entities:
                return f"# No Neighborhood Found\n\nNode {node_id} has no connections."

            return result.to_markdown()

    except Exception as e:
        return f"# Error\n\nFailed to get graph neighborhood: {e}"


@mcp.tool(name="trace_path")
async def mcp_trace_path(
    from_node_id: Annotated[str, "Starting node ID"],
    to_node_id: Annotated[str, "Target node ID"],
    max_hops: Annotated[int, "Maximum path length (1-10)"] = 6,
) -> str:
    """Find the shortest path between two nodes in the knowledge graph.

    Useful for understanding how different code elements are connected,
    e.g., how a business rule relates to a database table.
    """
    try:
        from_uuid = uuid.UUID(from_node_id)
        to_uuid = uuid.UUID(to_node_id)
    except ValueError as e:
        return f"# Error\n\nInvalid node ID: {e}"

    try:
        from contextmine_core.graphrag import trace_path

        async with get_db_session() as db:
            result = await trace_path(
                session=db,
                from_node_id=from_uuid,
                to_node_id=to_uuid,
                collection_id=None,
                max_hops=min(max_hops, 10),
            )

            if not result.entities:
                return f"# No Path Found\n\nNo connection between {from_node_id[:8]}... and {to_node_id[:8]}..."

            return result.to_markdown()

    except Exception as e:
        return f"# Error\n\nFailed to trace path: {e}"


@mcp.tool(name="graph_rag")
async def mcp_graph_rag(
    query: Annotated[str, "Natural language query"],
    collection_id: Annotated[str | None, "Filter to specific collection"] = None,
    max_communities: Annotated[int, "Maximum community summaries to include (global context)"] = 5,
    max_entities: Annotated[int, "Maximum entities to include (local context)"] = 20,
    max_depth: Annotated[int, "Graph expansion depth (1-3)"] = 2,
    format: Annotated[str, "Output format: 'markdown' or 'json'"] = "markdown",
    answer: Annotated[bool, "Use map-reduce to generate synthesized answer (requires LLM)"] = False,
    twin_scope: Annotated[str, "Context scope: code|tests|ui|flows|all"] = "all",
    rebuild_mode: Annotated[bool, "Append rebuild-oriented architecture sections"] = False,
) -> str:
    """Graph-augmented retrieval with global (community) and local (entity) context.

    This is the PRIMARY research tool implementing Microsoft GraphRAG:
    1. Leiden-based hierarchical community detection
    2. Global context: Community summaries ranked by vector similarity
    3. Local context: Entity nodes from semantic search + community members
    4. Map-reduce answering: Parallel LLM calls over communities, then synthesis

    Set answer=true to get a synthesized answer using map-reduce over community summaries.
    Otherwise returns context (markdown/json) for you to analyze.

    Use this for questions that benefit from understanding:
    - High-level architecture (from community summaries)
    - Code structure and relationships (from entity context)
    - Evidence trail (from citations)
    """
    user_id = get_current_user_id()

    try:
        scope = (twin_scope or "all").strip().lower()
        if scope not in {"code", "tests", "ui", "flows", "all"}:
            return "# Error\n\nInvalid twin_scope. Use code|tests|ui|flows|all."

        def apply_scope_filter(result_pack):  # noqa: ANN001
            if scope == "all":
                return result_pack
            allowed_entities = [
                entity for entity in result_pack.entities if _node_kind_in_scope(entity.kind, scope)
            ]
            allowed_ids = {str(entity.node_id) for entity in allowed_entities}
            result_pack.entities = allowed_entities
            result_pack.edges = [
                edge
                for edge in result_pack.edges
                if edge.source_id in allowed_ids and edge.target_id in allowed_ids
            ]
            return result_pack

        collection_uuid = uuid.UUID(collection_id) if collection_id else None

        if answer:
            # Full GraphRAG map-reduce answering
            from contextmine_core import get_settings
            from contextmine_core.graphrag import graph_rag_query
            from contextmine_core.research.llm import get_llm_provider

            settings = get_settings()

            async with get_db_session() as db:
                # Get LLM provider for map-reduce
                try:
                    llm_provider = get_llm_provider(settings.default_llm_provider)
                except Exception:
                    return (
                        "# Error\n\nNo LLM provider configured. Set DEFAULT_LLM_PROVIDER env var."
                    )

                result = await graph_rag_query(
                    session=db,
                    query=query,
                    llm_provider=llm_provider,
                    collection_id=collection_uuid,
                    user_id=user_id,
                    max_communities=min(max_communities, 10),
                    max_entities=min(max_entities, 50),
                )
                if result.context:
                    result.context = apply_scope_filter(result.context)

                # Format response
                lines = [
                    f"# GraphRAG Answer: {query}\n",
                    result.final_answer,
                    "",
                    f"*Based on {result.communities_used} communities, {len(result.partial_answers)} relevant responses*",
                ]

                if result.context and result.context.citations:
                    lines.append("\n## Key Citations")
                    for cit in result.context.citations[:5]:
                        lines.append(f"- `{cit.format()}`")

                if rebuild_mode and result.context:
                    entities = result.context.entities
                    by_kind: dict[str, int] = {}
                    for entity in entities:
                        by_kind[entity.kind] = by_kind.get(entity.kind, 0) + 1
                    lines.append("\n## System Boundaries")
                    lines.append(f"- Entities in scope: {len(entities)} (scope: {scope})")
                    lines.append("\n## Interfaces")
                    lines.append(
                        f"- API/contract entities: {sum(by_kind.get(k, 0) for k in ['api_endpoint', 'service_rpc', 'interface_contract'])}"
                    )
                    lines.append("\n## UI Screens")
                    lines.append(
                        f"- UI entities: {sum(by_kind.get(k, 0) for k in ['ui_route', 'ui_view', 'ui_component'])}"
                    )
                    lines.append("\n## User Flows")
                    lines.append(
                        f"- Flow entities: {sum(by_kind.get(k, 0) for k in ['user_flow', 'flow_step'])}"
                    )
                    lines.append("\n## Test Obligations")
                    lines.append(
                        f"- Test entities: {sum(by_kind.get(k, 0) for k in ['test_suite', 'test_case', 'test_fixture'])}"
                    )
                    lines.append("\n## Known Gaps")
                    lines.append(
                        "- Prefer `export_twin_view(format='twin_manifest')` for full rebuild handoff."
                    )

                return "\n".join(lines)

        else:
            # Context retrieval only (no LLM)
            from contextmine_core.graphrag import graph_rag_context

            async with get_db_session() as db:
                result = await graph_rag_context(
                    session=db,
                    query=query,
                    collection_id=collection_uuid,
                    user_id=user_id,
                    max_communities=min(max_communities, 10),
                    max_entities=min(max_entities, 50),
                    max_depth=min(max_depth, 3),
                )
                result = apply_scope_filter(result)

                if format.lower() == "json":
                    payload = result.to_dict()
                    if rebuild_mode:
                        kinds = [entity.get("kind") for entity in payload.get("entities", [])]
                        payload["rebuild_sections"] = {
                            "System Boundaries": {"entity_count": len(payload.get("entities", []))},
                            "Interfaces": {
                                "entity_count": sum(
                                    1
                                    for kind in kinds
                                    if kind in {"api_endpoint", "service_rpc", "interface_contract"}
                                )
                            },
                            "UI Screens": {
                                "entity_count": sum(
                                    1
                                    for kind in kinds
                                    if kind in {"ui_route", "ui_view", "ui_component"}
                                )
                            },
                            "User Flows": {
                                "entity_count": sum(
                                    1 for kind in kinds if kind in {"user_flow", "flow_step"}
                                )
                            },
                            "Test Obligations": {
                                "entity_count": sum(
                                    1
                                    for kind in kinds
                                    if kind in {"test_suite", "test_case", "test_fixture"}
                                )
                            },
                            "Known Gaps": [
                                "Use twin_manifest export for end-to-end rebuild planning."
                            ],
                        }
                    return json.dumps(payload, indent=2)

                md = result.to_markdown()
                if (
                    not md.strip()
                    or md
                    == f"# GraphRAG Context: {query}\n\nFound 0 communities, 0 entities, 0 citations.\n"
                ):
                    return f"# No Results\n\nNo relevant content found for: {query}"

                if rebuild_mode:
                    md += (
                        "\n\n## Rebuild Mode\n"
                        f"- Scope: `{scope}`\n"
                        "- Add `format='json'` for machine-consumable rebuild sections.\n"
                    )

                return md

    except Exception as e:
        return f"# Error\n\nGraphRAG query failed: {e}"


@mcp.tool(name="create_architecture_intent")
async def mcp_create_architecture_intent(
    scenario_id: Annotated[str, "Scenario UUID"],
    action: Annotated[
        str,
        "Intent action: EXTRACT_DOMAIN|SPLIT_CONTAINER|MOVE_COMPONENT|DEFINE_INTERFACE|SET_VALIDATOR|APPLY_DATA_BOUNDARY",
    ],
    target_type: Annotated[str, "Target type: node|edge|context|service"],
    target_id: Annotated[str, "Target identifier"],
    expected_scenario_version: Annotated[int, "Optimistic locking scenario version"],
    params_json: Annotated[str | None, "Optional JSON object with action params"] = None,
) -> str:
    """Create and auto-execute an architecture intent."""
    try:
        from contextmine_core.architecture_intents import ArchitectureIntentV1
        from contextmine_core.models import Collection, TwinScenario
        from contextmine_core.twin import submit_intent

        user_id = get_current_user_id()
        if not user_id:
            return "# Error\n\nAuthentication required."

        scenario_uuid = uuid.UUID(scenario_id)
        params = json.loads(params_json) if params_json else {}

        async with get_db_session() as db:
            scenario = (
                await db.execute(select(TwinScenario).where(TwinScenario.id == scenario_uuid))
            ).scalar_one_or_none()
            if not scenario:
                return "# Error\n\nScenario not found."

            collection = (
                await db.execute(select(Collection).where(Collection.id == scenario.collection_id))
            ).scalar_one_or_none()
            if not collection or collection.owner_user_id != user_id:
                return "# Error\n\nOnly the collection owner can execute intents."

            payload = ArchitectureIntentV1(
                intent_version="1.0",
                scenario_id=scenario_uuid,
                action=action,
                target={"type": target_type, "id": target_id},
                params=params,
                expected_scenario_version=expected_scenario_version,
                requested_by=user_id,
            )

            intent = await submit_intent(
                session=db,
                scenario=scenario,
                intent=payload,
                requested_by=user_id,
                auto_execute=True,
            )
            await db.commit()

            return (
                f"# Intent Created\n\n"
                f"- ID: `{intent.id}`\n"
                f"- Status: `{intent.status.value}`\n"
                f"- Risk: `{intent.risk_level.value}`\n"
                f"- Requires Approval: `{intent.requires_approval}`\n"
                f"- Scenario Version: `{scenario.version}`"
            )

    except Exception as e:
        return f"# Error\n\nFailed to create intent: {e}"


@mcp.tool(name="approve_architecture_intent")
async def mcp_approve_architecture_intent(
    scenario_id: Annotated[str, "Scenario UUID"],
    intent_id: Annotated[str, "Blocked intent UUID to approve and execute"],
) -> str:
    """Approve a blocked high-risk intent and execute it."""
    try:
        from contextmine_core.models import Collection, TwinScenario
        from contextmine_core.twin import approve_and_execute_intent

        user_id = get_current_user_id()
        if not user_id:
            return "# Error\n\nAuthentication required."

        scenario_uuid = uuid.UUID(scenario_id)
        intent_uuid = uuid.UUID(intent_id)

        async with get_db_session() as db:
            scenario = (
                await db.execute(select(TwinScenario).where(TwinScenario.id == scenario_uuid))
            ).scalar_one_or_none()
            if not scenario:
                return "# Error\n\nScenario not found."

            collection = (
                await db.execute(select(Collection).where(Collection.id == scenario.collection_id))
            ).scalar_one_or_none()
            if not collection or collection.owner_user_id != user_id:
                return "# Error\n\nOnly the collection owner can approve intents."

            intent = await approve_and_execute_intent(db, scenario, intent_uuid)
            await db.commit()
            return (
                f"# Intent Approved\n\n"
                f"- ID: `{intent.id}`\n"
                f"- Status: `{intent.status.value}`\n"
                f"- Scenario Version: `{scenario.version}`"
            )
    except Exception as e:
        return f"# Error\n\nFailed to approve intent: {e}"


@mcp.tool(name="get_twin_graph")
async def mcp_get_twin_graph(
    scenario_id: Annotated[str, "Scenario UUID"],
    layer: Annotated[
        str | None,
        "Optional layer: portfolio_system|domain_container|component_interface|code_controlflow",
    ] = None,
    page: Annotated[int, "Page index"] = 0,
    limit: Annotated[int, "Nodes per page"] = 200,
    format: Annotated[str, "Output format: markdown or json"] = "markdown",
    facet: Annotated[str | None, "Facet filter: code|tests|ui|flows|all"] = None,
    include_provenance_mode: Annotated[
        str | None,
        "Optional provenance mode filter: deterministic|inferred",
    ] = None,
    include_test_links: Annotated[
        bool | None,
        "Include test semantic links (omit to keep existing default behavior)",
    ] = None,
    include_ui_links: Annotated[
        bool | None,
        "Include UI semantic links (omit to keep existing default behavior)",
    ] = None,
) -> str:
    """Read a paginated twin graph view."""
    try:
        from contextmine_core.models import TwinLayer
        from contextmine_core.twin import get_scenario_graph

        scenario_uuid = uuid.UUID(scenario_id)
        layer_value = TwinLayer(layer) if layer else None

        async with get_db_session() as db:
            graph = await get_scenario_graph(db, scenario_uuid, layer_value, page, min(limit, 5000))
        scope = (facet or "all").strip().lower()
        if scope not in {"code", "tests", "ui", "flows", "all"}:
            return "# Error\n\nInvalid facet. Use code|tests|ui|flows|all."
        graph = _filter_graph_payload(
            graph,
            scope=scope,
            provenance_mode=include_provenance_mode,
            include_test_links=include_test_links,
            include_ui_links=include_ui_links,
        )

        if format.lower() == "json":
            return json.dumps(graph, indent=2)

        lines = [
            f"# Twin Graph `{scenario_id}`",
            "",
            f"- Nodes: {len(graph['nodes'])}/{graph['total_nodes']}",
            f"- Edges: {len(graph['edges'])}",
            f"- Facet: {scope}",
            "",
            "## Nodes",
        ]
        for node in graph["nodes"][:50]:
            lines.append(f"- `{node['natural_key']}` ({node['kind']})")
        if len(graph["nodes"]) > 50:
            lines.append(f"- ... and {len(graph['nodes']) - 50} more")

        lines.append("")
        lines.append("## Edges")
        for edge in graph["edges"][:50]:
            lines.append(
                f"- `{edge['source_node_id']}` -[{edge['kind']}]-> `{edge['target_node_id']}`"
            )
        if len(graph["edges"]) > 50:
            lines.append(f"- ... and {len(graph['edges']) - 50} more")
        return "\n".join(lines)
    except Exception as e:
        return f"# Error\n\nFailed to read twin graph: {e}"


@mcp.tool(name="query_twin_cypher")
async def mcp_query_twin_cypher(
    scenario_id: Annotated[str, "Scenario UUID"],
    query: Annotated[str, "Read-only Cypher query"],
) -> str:
    """Run a read-only Cypher query against Apache AGE twin graph."""
    try:
        from contextmine_core.graph.age import run_read_only_cypher, sync_scenario_to_age

        scenario_uuid = uuid.UUID(scenario_id)
        async with get_db_session() as db:
            await sync_scenario_to_age(db, scenario_uuid)
            rows = await run_read_only_cypher(db, scenario_uuid, query)
            return json.dumps({"rows": rows, "count": len(rows)}, indent=2)
    except Exception as e:
        return f"# Error\n\nCypher query failed: {e}"


@mcp.tool(name="get_twin_status")
async def mcp_get_twin_status(
    collection_id: Annotated[str, "Collection UUID"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
) -> str:
    """Get digital twin freshness/status for a collection."""
    try:
        from contextmine_core.twin import get_collection_twin_status

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            scenario_uuid = uuid.UUID(scenario_id) if scenario_id else None
            payload = await get_collection_twin_status(
                db,
                collection_id=collection.id,
                scenario_id=scenario_uuid,
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to fetch twin status: {e}"


@mcp.tool(name="get_twin_timeline")
async def mcp_get_twin_timeline(
    collection_id: Annotated[str, "Collection UUID"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    source_id: Annotated[str | None, "Optional source UUID"] = None,
    event_type: Annotated[str | None, "Optional event type filter"] = None,
    status: Annotated[str | None, "Optional status filter"] = None,
    page: Annotated[int, "Page index"] = 0,
    limit: Annotated[int, "Items per page"] = 50,
) -> str:
    """Read paginated twin timeline events."""
    del scenario_id
    try:
        from contextmine_core.twin import list_collection_twin_events

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            source_uuid = uuid.UUID(source_id) if source_id else None
            payload = await list_collection_twin_events(
                db,
                collection_id=collection.id,
                page=max(page, 0),
                limit=max(min(limit, 200), 1),
                source_id=source_uuid,
                event_type=event_type,
                status=status,
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to fetch twin timeline: {e}"


@mcp.tool(name="refresh_twin")
async def mcp_refresh_twin(
    collection_id: Annotated[str, "Collection UUID"],
    source_ids: Annotated[str | None, "Comma-separated source UUIDs (optional)"] = None,
    force: Annotated[bool, "Force refresh even if revision unchanged"] = False,
) -> str:
    """Queue twin materialization refresh for one collection."""
    try:
        from contextmine_core.twin import coerce_source_ids, trigger_collection_refresh

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            parsed_source_ids = coerce_source_ids(_parse_csv_list(source_ids))
            payload = await trigger_collection_refresh(
                db,
                collection_id=collection.id,
                source_ids=parsed_source_ids or None,
                force=force,
            )
            await db.commit()
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to refresh twin: {e}"


@mcp.tool(name="get_arc42")
async def mcp_get_arc42(
    collection_id: Annotated[str | None, "Optional collection UUID"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    section: Annotated[str | None, "Optional section key or number (e.g. 5, quality)"] = None,
    regenerate: Annotated[bool, "Regenerate arc42 artifact instead of using cache"] = False,
) -> str:
    """Generate/read arc42 document for a collection scenario."""
    try:
        from contextmine_core.architecture import (
            SECTION_TITLES,
            build_architecture_facts,
            generate_arc42_from_facts,
            normalize_arc42_section_key,
        )
        from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind, TwinScenario
        from contextmine_core.twin import get_or_create_as_is_scenario

        settings = get_settings()
        if not settings.arch_docs_enabled:
            return "# Error\n\nArchitecture docs are disabled."

        section_key = normalize_arc42_section_key(section)
        if section and not section_key:
            return "# Error\n\nInvalid section."

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_for_tool(
                db,
                collection_id=collection_id,
                user_id=user_id,
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."

            if scenario_id:
                scenario = (
                    await db.execute(
                        select(TwinScenario).where(
                            TwinScenario.id == uuid.UUID(scenario_id),
                            TwinScenario.collection_id == collection.id,
                        )
                    )
                ).scalar_one_or_none()
                if not scenario:
                    return "# Error\n\nScenario not found in collection."
            else:
                scenario = (
                    await db.execute(
                        select(TwinScenario)
                        .where(
                            TwinScenario.collection_id == collection.id,
                            TwinScenario.is_as_is.is_(True),
                        )
                        .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if scenario is None:
                    scenario = await get_or_create_as_is_scenario(
                        db, collection.id, user_id=user_id
                    )

            artifact_name = f"{scenario.id}.arc42.md"
            existing = (
                await db.execute(
                    select(KnowledgeArtifact).where(
                        KnowledgeArtifact.collection_id == collection.id,
                        KnowledgeArtifact.kind == KnowledgeArtifactKind.ARC42,
                        KnowledgeArtifact.name == artifact_name,
                    )
                )
            ).scalar_one_or_none()

            if existing and not regenerate:
                meta = existing.meta or {}
                sections = meta.get("sections", {})
                if section_key and isinstance(sections, dict):
                    filtered = {section_key: sections.get(section_key, "")}
                    markdown = (
                        f"# arc42 - {scenario.name}\n\n"
                        f"## {SECTION_TITLES.get(section_key, section_key)}\n"
                        f"{filtered[section_key]}\n"
                    )
                    payload = {
                        "collection_id": str(collection.id),
                        "scenario_id": str(scenario.id),
                        "section": section_key,
                        "cached": True,
                        "facts_hash": meta.get("facts_hash"),
                        "warnings": meta.get("warnings", []),
                        "sections": filtered,
                        "markdown": markdown,
                    }
                    return json.dumps(payload, indent=2)

                payload = {
                    "collection_id": str(collection.id),
                    "scenario_id": str(scenario.id),
                    "section": section_key,
                    "cached": True,
                    "facts_hash": meta.get("facts_hash"),
                    "warnings": meta.get("warnings", []),
                    "sections": meta.get("sections", {}),
                    "markdown": existing.content,
                }
                return json.dumps(payload, indent=2)

            llm_provider = None
            if settings.arch_docs_llm_enrich and settings.default_llm_provider:
                try:
                    from contextmine_core.research.llm import get_llm_provider

                    llm_provider = get_llm_provider(settings.default_llm_provider)
                except Exception:
                    llm_provider = None

            facts = await build_architecture_facts(
                db,
                collection_id=collection.id,
                scenario_id=scenario.id,
                enable_llm_enrich=settings.arch_docs_llm_enrich,
                llm_provider=llm_provider,
            )
            full_doc = generate_arc42_from_facts(facts, scenario, options={})
            selected_doc = (
                generate_arc42_from_facts(facts, scenario, options={"section": section_key})
                if section_key
                else full_doc
            )

            if existing:
                existing.content = full_doc.markdown
                existing.meta = {
                    "scenario_id": str(scenario.id),
                    "generated_at": full_doc.generated_at.isoformat(),
                    "facts_hash": facts.facts_hash(),
                    "confidence_summary": full_doc.confidence_summary,
                    "section_coverage": full_doc.section_coverage,
                    "warnings": full_doc.warnings,
                    "sections": full_doc.sections,
                }
                artifact = existing
            else:
                artifact = KnowledgeArtifact(
                    collection_id=collection.id,
                    kind=KnowledgeArtifactKind.ARC42,
                    name=artifact_name,
                    content=full_doc.markdown,
                    meta={
                        "scenario_id": str(scenario.id),
                        "generated_at": full_doc.generated_at.isoformat(),
                        "facts_hash": facts.facts_hash(),
                        "confidence_summary": full_doc.confidence_summary,
                        "section_coverage": full_doc.section_coverage,
                        "warnings": full_doc.warnings,
                        "sections": full_doc.sections,
                    },
                )
                db.add(artifact)

            await db.commit()
            payload = {
                "collection_id": str(collection.id),
                "scenario_id": str(scenario.id),
                "artifact_id": str(artifact.id),
                "section": section_key,
                "cached": False,
                "facts_hash": facts.facts_hash(),
                "warnings": sorted(set([*facts.warnings, *full_doc.warnings])),
                "sections": selected_doc.sections,
                "markdown": selected_doc.markdown,
            }
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to get arc42: {e}"


@mcp.tool(name="arc42_drift_report")
async def mcp_arc42_drift_report(
    collection_id: Annotated[str | None, "Optional collection UUID"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    baseline_scenario_id: Annotated[str | None, "Optional baseline scenario UUID"] = None,
) -> str:
    """Compute advisory arc42 drift report between two scenarios."""
    try:
        from dataclasses import asdict

        from contextmine_core.architecture import build_architecture_facts, compute_arc42_drift
        from contextmine_core.models import TwinScenario
        from contextmine_core.twin import get_or_create_as_is_scenario

        settings = get_settings()
        if not settings.arch_docs_enabled:
            return "# Error\n\nArchitecture docs are disabled."

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_for_tool(
                db,
                collection_id=collection_id,
                user_id=user_id,
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."

            if scenario_id:
                scenario = (
                    await db.execute(
                        select(TwinScenario).where(
                            TwinScenario.id == uuid.UUID(scenario_id),
                            TwinScenario.collection_id == collection.id,
                        )
                    )
                ).scalar_one_or_none()
                if scenario is None:
                    return "# Error\n\nScenario not found in collection."
            else:
                scenario = (
                    await db.execute(
                        select(TwinScenario)
                        .where(
                            TwinScenario.collection_id == collection.id,
                            TwinScenario.is_as_is.is_(True),
                        )
                        .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if scenario is None:
                    scenario = await get_or_create_as_is_scenario(
                        db, collection.id, user_id=user_id
                    )

            baseline = None
            if baseline_scenario_id:
                baseline = (
                    await db.execute(
                        select(TwinScenario).where(
                            TwinScenario.id == uuid.UUID(baseline_scenario_id),
                            TwinScenario.collection_id == collection.id,
                        )
                    )
                ).scalar_one_or_none()
                if baseline is None:
                    return "# Error\n\nBaseline scenario not found in collection."
            elif scenario.base_scenario_id:
                baseline = (
                    await db.execute(
                        select(TwinScenario).where(
                            TwinScenario.id == scenario.base_scenario_id,
                            TwinScenario.collection_id == collection.id,
                        )
                    )
                ).scalar_one_or_none()
            if baseline is None:
                baseline = (
                    await db.execute(
                        select(TwinScenario)
                        .where(
                            TwinScenario.collection_id == collection.id,
                            TwinScenario.id != scenario.id,
                        )
                        .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()

            llm_provider = None
            if settings.arch_docs_llm_enrich and settings.default_llm_provider:
                try:
                    from contextmine_core.research.llm import get_llm_provider

                    llm_provider = get_llm_provider(settings.default_llm_provider)
                except Exception:
                    llm_provider = None

            current = await build_architecture_facts(
                db,
                collection_id=collection.id,
                scenario_id=scenario.id,
                enable_llm_enrich=settings.arch_docs_llm_enrich,
                llm_provider=llm_provider,
            )
            baseline_bundle = None
            if baseline:
                baseline_bundle = await build_architecture_facts(
                    db,
                    collection_id=collection.id,
                    scenario_id=baseline.id,
                    enable_llm_enrich=settings.arch_docs_llm_enrich,
                    llm_provider=llm_provider,
                )

            report = compute_arc42_drift(
                current,
                baseline_bundle,
                baseline_scenario_id=baseline.id if baseline else None,
            )
            deltas = [asdict(delta) for delta in report.deltas]

            by_type: dict[str, int] = {}
            for delta in deltas:
                key = str(delta.get("delta_type"))
                by_type[key] = by_type.get(key, 0) + 1

            payload = {
                "collection_id": str(collection.id),
                "scenario_id": str(scenario.id),
                "baseline_scenario_id": str(baseline.id) if baseline else None,
                "generated_at": report.generated_at.isoformat(),
                "current_hash": report.current_hash,
                "baseline_hash": report.baseline_hash,
                "summary": {
                    "total": len(deltas),
                    "by_type": by_type,
                    "severity": "low" if len(deltas) < 10 else "medium",
                },
                "deltas": deltas,
                "warnings": report.warnings,
            }
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to compute arc42 drift report: {e}"


@mcp.tool(name="list_ports_adapters")
async def mcp_list_ports_adapters(
    collection_id: Annotated[str | None, "Optional collection UUID"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    direction: Annotated[str | None, "Optional direction filter: inbound|outbound"] = None,
    container: Annotated[str | None, "Optional container name filter"] = None,
) -> str:
    """List inferred Ports-and-Adapters mapping with confidence and evidence references."""
    try:
        from dataclasses import asdict

        from contextmine_core.architecture import build_architecture_facts
        from contextmine_core.models import TwinScenario
        from contextmine_core.twin import get_or_create_as_is_scenario

        settings = get_settings()
        if not settings.arch_docs_enabled:
            return "# Error\n\nArchitecture docs are disabled."

        direction_value = direction.strip().lower() if direction else None
        if direction_value and direction_value not in {"inbound", "outbound"}:
            return "# Error\n\nInvalid direction. Use inbound or outbound."

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_for_tool(
                db,
                collection_id=collection_id,
                user_id=user_id,
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."

            if scenario_id:
                scenario = (
                    await db.execute(
                        select(TwinScenario).where(
                            TwinScenario.id == uuid.UUID(scenario_id),
                            TwinScenario.collection_id == collection.id,
                        )
                    )
                ).scalar_one_or_none()
                if scenario is None:
                    return "# Error\n\nScenario not found in collection."
            else:
                scenario = (
                    await db.execute(
                        select(TwinScenario)
                        .where(
                            TwinScenario.collection_id == collection.id,
                            TwinScenario.is_as_is.is_(True),
                        )
                        .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if scenario is None:
                    scenario = await get_or_create_as_is_scenario(
                        db, collection.id, user_id=user_id
                    )

            llm_provider = None
            if settings.arch_docs_llm_enrich and settings.default_llm_provider:
                try:
                    from contextmine_core.research.llm import get_llm_provider

                    llm_provider = get_llm_provider(settings.default_llm_provider)
                except Exception:
                    llm_provider = None

            bundle = await build_architecture_facts(
                db,
                collection_id=collection.id,
                scenario_id=scenario.id,
                enable_llm_enrich=settings.arch_docs_llm_enrich,
                llm_provider=llm_provider,
            )

            rows = bundle.ports_adapters
            if direction_value:
                rows = [row for row in rows if row.direction == direction_value]
            if container:
                target = container.strip().lower()
                rows = [row for row in rows if (row.container or "").strip().lower() == target]

            inbound = sum(1 for row in rows if row.direction == "inbound")
            outbound = sum(1 for row in rows if row.direction == "outbound")
            payload = {
                "collection_id": str(collection.id),
                "scenario_id": str(scenario.id),
                "summary": {"total": len(rows), "inbound": inbound, "outbound": outbound},
                "filters": {"direction": direction_value, "container": container},
                "warnings": bundle.warnings,
                "items": [asdict(row) for row in rows],
            }
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to list ports/adapters: {e}"


@mcp.tool(name="get_codebase_summary")
async def mcp_get_codebase_summary(
    collection_id: Annotated[str, "Collection UUID"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """Get multi-engine codebase summary."""
    try:
        from contextmine_core.twin import get_codebase_summary_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await get_codebase_summary_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to fetch codebase summary: {e}"


@mcp.tool(name="list_methods")
async def mcp_list_methods(
    collection_id: Annotated[str, "Collection UUID"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    query: Annotated[str | None, "Optional method name filter"] = None,
    page: Annotated[int, "Page index"] = 0,
    limit: Annotated[int, "Items per page"] = 50,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """List methods from selected analysis engines."""
    try:
        from contextmine_core.twin import list_methods_multi, sanitize_regex_query

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await list_methods_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                query=sanitize_regex_query(query),
                page=max(page, 0),
                limit=max(min(limit, 200), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to list methods: {e}"


@mcp.tool(name="list_calls")
async def mcp_list_calls(
    collection_id: Annotated[str, "Collection UUID"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    page: Annotated[int, "Page index"] = 0,
    limit: Annotated[int, "Items per page"] = 50,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """List call edges from selected analysis engines."""
    try:
        from contextmine_core.twin import list_calls_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await list_calls_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                page=max(page, 0),
                limit=max(min(limit, 200), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to list calls: {e}"


@mcp.tool(name="get_cfg")
async def mcp_get_cfg(
    collection_id: Annotated[str, "Collection UUID"],
    node_ref: Annotated[str, "Method/node reference"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    depth: Annotated[int, "Traversal depth"] = 2,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """Get CFG approximation from selected analysis engines."""
    try:
        from contextmine_core.twin import get_cfg_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await get_cfg_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                node_ref=node_ref,
                depth=max(min(depth, 8), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to fetch CFG: {e}"


@mcp.tool(name="get_variable_flow")
async def mcp_get_variable_flow(
    collection_id: Annotated[str, "Collection UUID"],
    node_ref: Annotated[str, "Method/node reference"],
    variable: Annotated[str | None, "Optional variable name"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    max_hops: Annotated[int, "Maximum traversal hops"] = 6,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """Get variable-flow approximation from selected analysis engines."""
    try:
        from contextmine_core.twin import get_variable_flow_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await get_variable_flow_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                node_ref=node_ref,
                variable=variable,
                max_hops=max(min(max_hops, 20), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to fetch variable flow: {e}"


@mcp.tool(name="find_taint_sources")
async def mcp_find_taint_sources(
    collection_id: Annotated[str, "Collection UUID"],
    language: Annotated[str | None, "Language pattern profile"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    limit: Annotated[int, "Maximum source candidates"] = 50,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """Find taint source candidates from selected engines."""
    try:
        from contextmine_core.twin import find_taint_sources_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await find_taint_sources_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                language=language,
                limit=max(min(limit, 300), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to find taint sources: {e}"


@mcp.tool(name="find_taint_sinks")
async def mcp_find_taint_sinks(
    collection_id: Annotated[str, "Collection UUID"],
    language: Annotated[str | None, "Language pattern profile"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    limit: Annotated[int, "Maximum sink candidates"] = 50,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """Find taint sink candidates from selected engines."""
    try:
        from contextmine_core.twin import find_taint_sinks_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await find_taint_sinks_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                language=language,
                limit=max(min(limit, 300), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to find taint sinks: {e}"


@mcp.tool(name="find_taint_flows")
async def mcp_find_taint_flows(
    collection_id: Annotated[str, "Collection UUID"],
    language: Annotated[str | None, "Language pattern profile"] = None,
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    max_hops: Annotated[int, "Maximum traversal hops"] = 6,
    max_results: Annotated[int, "Maximum flows to return"] = 50,
    engines: Annotated[str | None, "Comma-separated engines: graphrag,lsp,joern"] = None,
) -> str:
    """Find taint source->sink flows from selected engines."""
    try:
        from contextmine_core.twin import find_taint_flows_multi

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await find_taint_flows_multi(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                language=language,
                max_hops=max(min(max_hops, 20), 1),
                max_results=max(min(max_results, 200), 1),
                cache_ttl_seconds=settings.twin_analysis_cache_ttl_seconds,
                engines=_parse_csv_list(engines),
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to find taint flows: {e}"


@mcp.tool(name="store_findings")
async def mcp_store_findings(
    collection_id: Annotated[str, "Collection UUID"],
    findings_json: Annotated[str, "JSON array of normalized findings"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
) -> str:
    """Persist normalized findings in twin_findings."""
    try:
        from contextmine_core.twin import store_findings

        try:
            findings = json.loads(findings_json)
        except json.JSONDecodeError:
            return "# Error\n\nfindings_json must be a valid JSON array."
        if not isinstance(findings, list):
            return "# Error\n\nfindings_json must be a JSON array."

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await store_findings(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                findings=findings,
            )
            await db.commit()
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to store findings: {e}"


@mcp.tool(name="export_sarif")
async def mcp_export_sarif(
    collection_id: Annotated[str, "Collection UUID"],
    scenario_id: Annotated[str | None, "Optional scenario UUID"] = None,
    status: Annotated[str | None, "Optional finding status filter"] = None,
    min_severity: Annotated[str | None, "Optional severity threshold"] = None,
) -> str:
    """Export findings as SARIF."""
    try:
        from contextmine_core.twin import export_findings_sarif

        user_id = get_current_user_id()
        async with get_db_session() as db:
            collection, error = await _resolve_collection_access(
                db, collection_id=collection_id, user_id=user_id
            )
            if error:
                return f"# Error\n\n{error}"
            if collection is None:
                return "# Error\n\nCollection not found."
            payload = await export_findings_sarif(
                db,
                collection_id=collection.id,
                scenario_id=uuid.UUID(scenario_id) if scenario_id else None,
                status=status,
                min_severity=min_severity,
            )
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to export SARIF: {e}"


@mcp.tool(name="export_twin_view")
async def mcp_export_twin_view(
    scenario_id: Annotated[str, "Scenario UUID"],
    format: Annotated[str, "lpg_jsonl|cc_json|cx2|jgf|mermaid_c4|twin_manifest"],
) -> str:
    """Generate and persist an export artifact for a scenario."""
    try:
        from contextmine_core.exports import (
            export_codecharta_json,
            export_cx2,
            export_jgf,
            export_lpg_jsonl,
            export_mermaid_c4,
            export_twin_manifest,
        )
        from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind, TwinScenario
        from contextmine_core.twin import GraphProjection

        scenario_uuid = uuid.UUID(scenario_id)

        async with get_db_session() as db:
            scenario = (
                await db.execute(select(TwinScenario).where(TwinScenario.id == scenario_uuid))
            ).scalar_one_or_none()
            if not scenario:
                return "# Error\n\nScenario not found."

            if format == "lpg_jsonl":
                content = await export_lpg_jsonl(db, scenario.id)
                kind = KnowledgeArtifactKind.LPG_JSONL
                name = f"{scenario.name}.lpg.jsonl"
            elif format == "cc_json":
                content = await export_codecharta_json(
                    db,
                    scenario.id,
                    projection=GraphProjection.ARCHITECTURE,
                    entity_level="container",
                )
                kind = KnowledgeArtifactKind.CC_JSON
                name = f"{scenario.name}.cc.json"
            elif format == "cx2":
                content = await export_cx2(db, scenario.id)
                kind = KnowledgeArtifactKind.CX2
                name = f"{scenario.name}.cx2.json"
            elif format == "jgf":
                content = await export_jgf(db, scenario.id)
                kind = KnowledgeArtifactKind.JGF
                name = f"{scenario.name}.jgf.json"
            elif format == "mermaid_c4":
                content = await export_mermaid_c4(db, scenario.id)
                kind = (
                    KnowledgeArtifactKind.MERMAID_C4_ASIS
                    if scenario.is_as_is
                    else KnowledgeArtifactKind.MERMAID_C4_TOBE
                )
                name = f"{scenario.name}.mmd"
            elif format == "twin_manifest":
                content = await export_twin_manifest(db, scenario.id)
                kind = KnowledgeArtifactKind.TWIN_MANIFEST
                name = f"{scenario.name}.twin_manifest.json"
            else:
                return "# Error\n\nUnsupported export format."

            artifact = KnowledgeArtifact(
                id=uuid.uuid4(),
                collection_id=scenario.collection_id,
                kind=kind,
                name=name,
                content=content,
                meta={"scenario_id": str(scenario.id), "format": format},
            )
            db.add(artifact)
            await db.commit()

            return (
                f"# Export Created\n\n"
                f"- ID: `{artifact.id}`\n"
                f"- Name: `{artifact.name}`\n"
                f"- Kind: `{artifact.kind.value}`"
            )
    except Exception as e:
        return f"# Error\n\nExport failed: {e}"


@mcp.tool(name="get_validation_dashboard")
async def mcp_get_validation_dashboard(
    collection_id: Annotated[str | None, "Optional collection UUID"] = None,
) -> str:
    """Get normalized Tekton/Argo/Temporal status summary."""
    try:
        from contextmine_core.validation import (
            get_latest_validation_status,
            refresh_validation_snapshots,
        )

        collection_uuid = uuid.UUID(collection_id) if collection_id else None
        async with get_db_session() as db:
            await refresh_validation_snapshots(db, collection_uuid)
            payload = await get_latest_validation_status(db, collection_uuid)
            await db.commit()
            return json.dumps(payload, indent=2)
    except Exception as e:
        return f"# Error\n\nFailed to fetch validation dashboard: {e}"


# Get the HTTP app from FastMCP with root path (mounted at /mcp in main.py)
# FastMCP handles OAuth authentication via GitHubProvider
mcp_app = mcp.http_app(path="/", stateless_http=True)

# Export the MCP lifespan for integration with FastAPI
mcp_lifespan = mcp_app.lifespan


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
        {
            "name": "context.create_architecture_intent",
            "description": "Create an architecture intent and auto-execute low-risk actions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "action": {"type": "string"},
                    "target_type": {"type": "string"},
                    "target_id": {"type": "string"},
                    "expected_scenario_version": {"type": "integer"},
                    "params_json": {"type": "string"},
                },
                "required": [
                    "scenario_id",
                    "action",
                    "target_type",
                    "target_id",
                    "expected_scenario_version",
                ],
            },
        },
        {
            "name": "context.approve_architecture_intent",
            "description": "Approve and execute a blocked architecture intent.",
            "inputSchema": {
                "type": "object",
                "properties": {"scenario_id": {"type": "string"}, "intent_id": {"type": "string"}},
                "required": ["scenario_id", "intent_id"],
            },
        },
        {
            "name": "context.get_twin_graph",
            "description": "Read a paginated twin graph view with optional layer filter.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scenario_id": {"type": "string"},
                    "layer": {"type": "string"},
                    "page": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "format": {"type": "string"},
                    "facet": {"type": "string"},
                    "include_provenance_mode": {"type": "string"},
                    "include_test_links": {"type": "boolean"},
                    "include_ui_links": {"type": "boolean"},
                },
                "required": ["scenario_id"],
            },
        },
        {
            "name": "context.query_twin_cypher",
            "description": "Run read-only Cypher query against Apache AGE twin graph.",
            "inputSchema": {
                "type": "object",
                "properties": {"scenario_id": {"type": "string"}, "query": {"type": "string"}},
                "required": ["scenario_id", "query"],
            },
        },
        {
            "name": "context.get_arc42",
            "description": "Generate or read arc42 architecture document from twin + KG facts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_id": {"type": "string"},
                    "scenario_id": {"type": "string"},
                    "section": {"type": "string"},
                    "regenerate": {"type": "boolean", "default": False},
                },
                "required": [],
            },
        },
        {
            "name": "context.arc42_drift_report",
            "description": "Compute advisory arc42 drift report between scenario states.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_id": {"type": "string"},
                    "scenario_id": {"type": "string"},
                    "baseline_scenario_id": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "name": "context.list_ports_adapters",
            "description": "List inferred inbound/outbound ports and adapters with confidence.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_id": {"type": "string"},
                    "scenario_id": {"type": "string"},
                    "direction": {"type": "string"},
                    "container": {"type": "string"},
                },
                "required": [],
            },
        },
        {
            "name": "context.export_twin_view",
            "description": "Generate and persist lpg_jsonl, cc_json, cx2, jgf, mermaid_c4, or twin_manifest exports.",
            "inputSchema": {
                "type": "object",
                "properties": {"scenario_id": {"type": "string"}, "format": {"type": "string"}},
                "required": ["scenario_id", "format"],
            },
        },
        {
            "name": "context.get_validation_dashboard",
            "description": "Fetch normalized Tekton/Argo/Temporal metrics.",
            "inputSchema": {
                "type": "object",
                "properties": {"collection_id": {"type": "string"}},
                "required": [],
            },
        },
    ]
