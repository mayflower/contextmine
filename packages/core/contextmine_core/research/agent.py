"""Research Agent: LangGraph-based code research with tools.

Implements a proper LangGraph agent using ToolNode and bind_tools
for agentic code investigation with verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Any

from contextmine_core.research.artifacts import get_artifact_store
from contextmine_core.research.run import Evidence, ResearchRun
from contextmine_core.research.verification.models import VerificationStatus
from contextmine_core.research.verification.verifier import AnswerVerifier
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


def _escape_like_pattern(value: str) -> str:
    """Escape special characters in LIKE patterns to prevent SQL injection."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# =============================================================================
# STATE
# =============================================================================


class AgentState(TypedDict):
    """State for the research agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation messages including tool calls."""

    run: ResearchRun
    """The research run being executed."""

    pending_answer: str | None
    """Answer pending verification."""

    verification_attempts: int
    """Number of verification attempts."""


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""

    query: str = Field(description="Search query")
    k: int = Field(default=10, description="Number of results")


class OpenSpanInput(BaseModel):
    """Input for open span tool."""

    file_path: str = Field(description="Path to file")
    start_line: int = Field(description="Start line (1-indexed)")
    end_line: int = Field(description="End line (inclusive)")


class FinalizeInput(BaseModel):
    """Input for finalize tool."""

    answer: str = Field(description="The final answer with citations")
    confidence: float = Field(default=0.8, description="Confidence 0.0-1.0")


def create_tools(run_holder: dict[str, ResearchRun]) -> list:
    """Create tools that operate on a shared ResearchRun.

    Args:
        run_holder: Dict holding the current run (mutable reference)

    Returns:
        List of tool functions
    """

    @tool
    async def hybrid_search(query: str, k: int = 10) -> str:
        """Search the codebase using hybrid BM25 + vector retrieval.

        Use this to find relevant code snippets for your investigation.
        Returns matching code with file paths and line numbers.
        """
        run = run_holder["run"]

        try:
            from contextmine_core.embeddings import get_embedder, parse_embedding_model_spec
            from contextmine_core.search import hybrid_search as do_search
            from contextmine_core.settings import get_settings

            settings = get_settings()
            emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
            embedder = get_embedder(emb_provider, emb_model)

            embed_result = await embedder.embed_batch([query])
            query_embedding = embed_result.embeddings[0]

            results = await do_search(
                query=query,
                query_embedding=query_embedding,
                user_id=None,
                collection_id=None,
                top_k=k,
            )

            # Convert to evidence
            output_parts = []
            for i, r in enumerate(results.results[:k]):
                evidence = Evidence(
                    id=f"ev-{run.run_id[:8]}-{len(run.evidence) + i + 1:03d}",
                    file_path=r.uri or "unknown",
                    start_line=1,
                    end_line=len(r.content.split("\n")),
                    content=r.content,
                    reason=f"Matched query: {query}",
                    provenance="hybrid",
                    score=r.score,
                )
                run.add_evidence(evidence)
                output_parts.append(
                    f"[{evidence.id}] {evidence.file_path}\n```\n{r.content[:500]}\n```"
                )

            if not output_parts:
                return "No results found."

            return f"Found {len(results.results)} results:\n\n" + "\n\n".join(output_parts)

        except Exception as e:
            logger.exception("hybrid_search failed: %s", e)
            return f"Search failed: {e}"

    @tool
    async def open_span(file_path: str, start_line: int, end_line: int) -> str:
        """Read a specific range of lines from a file.

        Use this to examine code in detail after finding it via search.
        Registers the content as evidence for your answer.
        """
        run = run_holder["run"]

        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document
            from sqlalchemy import select

            async with get_async_session() as session:
                stmt = select(Document).where(Document.uri == file_path)
                result = await session.execute(stmt)
                doc = result.scalar_one_or_none()

                if not doc:
                    return f"File not found: {file_path}"

                lines = (doc.content or "").split("\n")
                start_idx = max(0, start_line - 1)
                end_idx = min(len(lines), end_line)
                content = "\n".join(lines[start_idx:end_idx])

                evidence = Evidence(
                    id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    reason="Opened for inspection",
                    provenance="manual",
                )
                run.add_evidence(evidence)

                return f"[{evidence.id}] {file_path}:{start_line}-{end_line}\n```\n{content}\n```"

        except Exception as e:
            logger.exception("open_span failed: %s", e)
            return f"Failed to read file: {e}"

    @tool
    async def finalize(answer: str, confidence: float = 0.8) -> str:
        """Submit your final answer with citations to evidence.

        Use this when you have gathered sufficient evidence to answer the question.
        Include citation IDs like [ev-abc-001] in your answer.
        The answer will be verified against the evidence before acceptance.
        """
        # This tool just returns the answer - verification happens in the graph
        run_holder["pending_answer"] = answer
        run_holder["confidence"] = confidence
        return f"Answer submitted for verification (confidence: {confidence})"

    # =========================================================================
    # DEFINITION/REFERENCE TOOLS (use pre-indexed Symbol and SymbolEdge data)
    # =========================================================================

    @tool
    async def goto_definition(symbol_name: str, file_path: str | None = None) -> str:
        """Jump to the definition of a symbol.

        Uses pre-indexed Symbol table to find where a symbol is defined.
        Optionally filter by file path if you know where it's used.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                # Find symbol by name
                stmt = (
                    select(Symbol)
                    .join(Document)
                    .where(Symbol.name == symbol_name)
                    .options(selectinload(Symbol.document))
                )
                if file_path:
                    # If file_path given, prioritize symbols in that file
                    stmt = stmt.order_by(
                        (Document.uri == file_path).desc(),
                        Symbol.start_line,
                    )
                stmt = stmt.limit(5)

                result = await session.execute(stmt)
                symbols = result.scalars().all()

                if not symbols:
                    return f"No definition found for '{symbol_name}'"

                output_parts = []
                for sym in symbols:
                    doc = sym.document
                    lines = (doc.content or "").split("\n")
                    start_idx = max(0, sym.start_line - 1)
                    end_idx = min(len(lines), sym.end_line)
                    content = "\n".join(lines[start_idx:end_idx])

                    evidence = Evidence(
                        id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                        file_path=doc.uri or "unknown",
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        content=content[:2000],
                        reason=f"Definition of '{symbol_name}'",
                        provenance="symbol_index",
                        symbol_id=sym.qualified_name,
                        symbol_kind=sym.kind.value,
                    )
                    run.add_evidence(evidence)
                    output_parts.append(
                        f"[{evidence.id}] {sym.kind.value} '{sym.name}' at "
                        f"{doc.uri}:{sym.start_line}-{sym.end_line}\n```\n{content[:500]}\n```"
                    )

                return f"Found {len(symbols)} definition(s):\n\n" + "\n\n".join(output_parts)

        except Exception as e:
            logger.warning("goto_definition failed: %s", e)
            return f"Goto definition failed: {e}"

    @tool
    async def find_references(symbol_name: str, file_path: str | None = None) -> str:
        """Find all usages/references of a symbol in the codebase.

        Uses pre-indexed SymbolEdge table to find where a symbol is referenced.
        Returns both callers (CALLS edges) and references (REFERENCES edges).
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol, SymbolEdge, SymbolEdgeType
            from sqlalchemy import or_, select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                # Find target symbol
                target_stmt = select(Symbol).join(Document).where(Symbol.name == symbol_name)
                if file_path:
                    target_stmt = target_stmt.where(Document.uri == file_path)
                target_stmt = target_stmt.limit(3)

                target_result = await session.execute(target_stmt)
                targets = target_result.scalars().all()

                if not targets:
                    return f"Symbol '{symbol_name}' not found"

                output_parts = []
                total_refs = 0

                for target in targets:
                    # Get incoming CALLS and REFERENCES edges
                    edges_stmt = (
                        select(SymbolEdge)
                        .where(SymbolEdge.target_symbol_id == target.id)
                        .where(
                            or_(
                                SymbolEdge.edge_type == SymbolEdgeType.CALLS,
                                SymbolEdge.edge_type == SymbolEdgeType.REFERENCES,
                            )
                        )
                        .options(
                            selectinload(SymbolEdge.source_symbol).selectinload(Symbol.document)
                        )
                        .limit(15)
                    )

                    edges_result = await session.execute(edges_stmt)
                    edges = edges_result.scalars().all()

                    for edge in edges[:10]:
                        ref_sym = edge.source_symbol
                        doc = ref_sym.document
                        lines = (doc.content or "").split("\n")

                        # Show context around the reference line
                        ref_line = edge.source_line or ref_sym.start_line
                        start_idx = max(0, ref_line - 3)
                        end_idx = min(len(lines), ref_line + 3)
                        snippet = "\n".join(lines[start_idx:end_idx])

                        evidence = Evidence(
                            id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                            file_path=doc.uri or "unknown",
                            start_line=max(1, ref_line - 2),
                            end_line=ref_line + 2,
                            content=snippet[:1500],
                            reason=f"{edge.edge_type.value} '{symbol_name}' from '{ref_sym.name}'",
                            provenance="symbol_index",
                            symbol_id=ref_sym.qualified_name,
                            symbol_kind=ref_sym.kind.value,
                        )
                        run.add_evidence(evidence)
                        output_parts.append(
                            f"[{evidence.id}] {edge.edge_type.value} in {ref_sym.kind.value} "
                            f"'{ref_sym.name}' at {doc.uri}:{ref_line}\n```\n{snippet[:300]}\n```"
                        )
                        total_refs += 1

                if not output_parts:
                    return f"No references found for '{symbol_name}'"

                summary = f"Found {total_refs} reference(s) to '{symbol_name}'"
                if total_refs > 10:
                    summary += " (showing first 10)"
                return summary + ":\n\n" + "\n\n".join(output_parts[:10])

        except Exception as e:
            logger.warning("find_references failed: %s", e)
            return f"Find references failed: {e}"

    @tool
    async def get_signature(symbol_name: str, file_path: str | None = None) -> str:
        """Get type signature and documentation for a symbol.

        Uses pre-indexed Symbol table to retrieve signature and docstring.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                stmt = (
                    select(Symbol)
                    .join(Document)
                    .where(Symbol.name == symbol_name)
                    .options(selectinload(Symbol.document))
                )
                if file_path:
                    stmt = stmt.where(Document.uri == file_path)
                stmt = stmt.limit(5)

                result = await session.execute(stmt)
                symbols = result.scalars().all()

                if not symbols:
                    return f"Symbol '{symbol_name}' not found"

                output_parts = []
                for sym in symbols:
                    doc = sym.document
                    content_parts = []

                    if sym.signature:
                        content_parts.append(f"Signature: {sym.signature}")
                    if sym.docstring:
                        content_parts.append(f"Documentation:\n{sym.docstring}")

                    if not content_parts:
                        # Fall back to showing the first few lines of the symbol
                        lines = (doc.content or "").split("\n")
                        start_idx = max(0, sym.start_line - 1)
                        end_idx = min(len(lines), sym.start_line + 5)
                        snippet = "\n".join(lines[start_idx:end_idx])
                        content_parts.append(f"Source:\n{snippet}")

                    content = "\n\n".join(content_parts)

                    evidence = Evidence(
                        id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                        file_path=doc.uri or "unknown",
                        start_line=sym.start_line,
                        end_line=sym.start_line,
                        content=content[:2000],
                        reason=f"Signature/docs for '{symbol_name}'",
                        provenance="symbol_index",
                        symbol_id=sym.qualified_name,
                        symbol_kind=sym.kind.value,
                    )
                    run.add_evidence(evidence)
                    output_parts.append(
                        f"[{evidence.id}] {sym.kind.value} '{sym.qualified_name}' "
                        f"at {doc.uri}:{sym.start_line}\n{content}"
                    )

                return "\n\n".join(output_parts)

        except Exception as e:
            logger.warning("get_signature failed: %s", e)
            return f"Get signature failed: {e}"

    # =========================================================================
    # SYMBOL INDEX TOOLS (use pre-indexed data from database)
    # =========================================================================

    @tool
    async def symbol_outline(file_path: str) -> str:
        """Get the outline of all indexed functions, classes, and symbols in a file.

        Uses the pre-indexed symbol database for fast lookup.
        Returns a list of symbols with their line numbers and signatures.
        """
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol
            from sqlalchemy import select

            async with get_async_session() as session:
                # Find document by URI
                doc_stmt = select(Document).where(Document.uri == file_path)
                doc_result = await session.execute(doc_stmt)
                doc = doc_result.scalar_one_or_none()

                if not doc:
                    return f"File not found in index: {file_path}"

                # Get all symbols for this document, ordered by line
                sym_stmt = (
                    select(Symbol).where(Symbol.document_id == doc.id).order_by(Symbol.start_line)
                )
                sym_result = await session.execute(sym_stmt)
                symbols = sym_result.scalars().all()

                if not symbols:
                    return f"No symbols indexed for {file_path}"

                outline_lines = []
                for sym in symbols:
                    indent = "  " if sym.parent_name else ""
                    sig = f" - {sym.signature}" if sym.signature else ""
                    outline_lines.append(
                        f"{indent}{sym.kind.value} {sym.name} (L{sym.start_line}-{sym.end_line}){sig}"
                    )

                summary = f"Found {len(symbols)} indexed symbols:\n" + "\n".join(outline_lines[:40])
                if len(outline_lines) > 40:
                    summary += f"\n... and {len(outline_lines) - 40} more"

                return summary

        except Exception as e:
            logger.warning("symbol_outline failed: %s", e)
            return f"Symbol outline failed: {e}"

    @tool
    async def symbol_find(name: str, file_path: str | None = None) -> str:
        """Find a symbol by name in the indexed codebase.

        Uses the pre-indexed symbol database. Optionally filter by file path.
        Returns the symbol's source code as evidence.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol
            from sqlalchemy import select

            async with get_async_session() as session:
                stmt = select(Symbol).join(Document)

                if file_path:
                    stmt = stmt.where(Document.uri == file_path)

                # Search by name (exact match first, then contains)
                stmt = stmt.where(Symbol.name == name)
                result = await session.execute(stmt)
                symbols = result.scalars().all()

                if not symbols:
                    # Try partial match
                    escaped_name = _escape_like_pattern(name)
                    stmt = (
                        select(Symbol)
                        .join(Document)
                        .where(Symbol.name.ilike(f"%{escaped_name}%", escape="\\"))
                    )
                    if file_path:
                        stmt = stmt.where(Document.uri == file_path)
                    stmt = stmt.limit(10)
                    result = await session.execute(stmt)
                    symbols = result.scalars().all()

                if not symbols:
                    return f"Symbol '{name}' not found in index"

                output_parts = []
                for sym in symbols[:5]:
                    # Get document content for the symbol
                    doc = sym.document
                    lines = (doc.content or "").split("\n")
                    start_idx = max(0, sym.start_line - 1)
                    end_idx = min(len(lines), sym.end_line)
                    content = "\n".join(lines[start_idx:end_idx])

                    evidence = Evidence(
                        id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                        file_path=doc.uri or "unknown",
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        content=content[:2000],
                        reason=f"Found indexed {sym.kind.value} '{sym.name}'",
                        provenance="symbol_index",
                        symbol_id=sym.qualified_name,
                        symbol_kind=sym.kind.value,
                    )
                    run.add_evidence(evidence)
                    output_parts.append(
                        f"[{evidence.id}] {sym.kind.value} '{sym.qualified_name}' at {doc.uri}:{sym.start_line}-{sym.end_line}\n```\n{content[:800]}\n```"
                    )

                return f"Found {len(symbols)} symbol(s):\n\n" + "\n\n".join(output_parts)

        except Exception as e:
            logger.warning("symbol_find failed: %s", e)
            return f"Symbol find failed: {e}"

    @tool
    async def symbol_callers(name: str, file_path: str | None = None) -> str:
        """Find all functions/methods that call a given symbol.

        Uses the pre-indexed symbol graph (SymbolEdge table).
        Returns callers as evidence.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol, SymbolEdge, SymbolEdgeType
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                # Find the target symbol
                stmt = select(Symbol).join(Document).where(Symbol.name == name)
                if file_path:
                    stmt = stmt.where(Document.uri == file_path)
                stmt = stmt.options(selectinload(Symbol.incoming_edges))
                result = await session.execute(stmt)
                target_symbols = result.scalars().all()

                if not target_symbols:
                    return f"Symbol '{name}' not found in index"

                output_parts = []
                for target in target_symbols[:3]:
                    # Get incoming CALLS edges
                    edges_stmt = (
                        select(SymbolEdge)
                        .where(SymbolEdge.target_symbol_id == target.id)
                        .where(SymbolEdge.edge_type == SymbolEdgeType.CALLS)
                        .options(
                            selectinload(SymbolEdge.source_symbol).selectinload(Symbol.document)
                        )
                    )
                    edges_result = await session.execute(edges_stmt)
                    edges = edges_result.scalars().all()

                    for edge in edges[:10]:
                        caller = edge.source_symbol
                        doc = caller.document
                        lines = (doc.content or "").split("\n")
                        start_idx = max(0, caller.start_line - 1)
                        end_idx = min(len(lines), caller.end_line)
                        content = "\n".join(lines[start_idx:end_idx])

                        evidence = Evidence(
                            id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                            file_path=doc.uri or "unknown",
                            start_line=caller.start_line,
                            end_line=caller.end_line,
                            content=content[:2000],
                            reason=f"Caller of '{name}' (line {edge.source_line})",
                            provenance="symbol_graph",
                            symbol_id=caller.qualified_name,
                            symbol_kind=caller.kind.value,
                        )
                        run.add_evidence(evidence)
                        output_parts.append(
                            f"[{evidence.id}] {caller.kind.value} '{caller.qualified_name}' calls '{name}' at line {edge.source_line}\n  {doc.uri}:{caller.start_line}"
                        )

                if not output_parts:
                    return f"No callers found for '{name}'"

                return f"Found {len(output_parts)} caller(s):\n" + "\n".join(output_parts)

        except Exception as e:
            logger.warning("symbol_callers failed: %s", e)
            return f"Symbol callers failed: {e}"

    @tool
    async def symbol_callees(name: str, file_path: str | None = None) -> str:
        """Find all functions/methods that a given symbol calls.

        Uses the pre-indexed symbol graph (SymbolEdge table).
        Returns callees as a list.
        """
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol, SymbolEdge, SymbolEdgeType
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload

            async with get_async_session() as session:
                # Find the source symbol
                stmt = select(Symbol).join(Document).where(Symbol.name == name)
                if file_path:
                    stmt = stmt.where(Document.uri == file_path)
                stmt = stmt.options(selectinload(Symbol.outgoing_edges))
                result = await session.execute(stmt)
                source_symbols = result.scalars().all()

                if not source_symbols:
                    return f"Symbol '{name}' not found in index"

                output_parts = []
                for source in source_symbols[:3]:
                    # Get outgoing CALLS edges
                    edges_stmt = (
                        select(SymbolEdge)
                        .where(SymbolEdge.source_symbol_id == source.id)
                        .where(SymbolEdge.edge_type == SymbolEdgeType.CALLS)
                        .options(
                            selectinload(SymbolEdge.target_symbol).selectinload(Symbol.document)
                        )
                    )
                    edges_result = await session.execute(edges_stmt)
                    edges = edges_result.scalars().all()

                    for edge in edges[:10]:
                        callee = edge.target_symbol
                        doc = callee.document
                        sig = f" - {callee.signature}" if callee.signature else ""

                        output_parts.append(
                            f"{callee.kind.value} '{callee.qualified_name}'{sig}\n  {doc.uri}:{callee.start_line}"
                        )

                if not output_parts:
                    return f"No callees found for '{name}'"

                return f"'{name}' calls {len(output_parts)} function(s):\n" + "\n".join(
                    output_parts
                )

        except Exception as e:
            logger.warning("symbol_callees failed: %s", e)
            return f"Symbol callees failed: {e}"

    @tool
    async def symbol_enclosing(file_path: str, line: int) -> str:
        """Find what function, class, or method contains a specific line.

        Uses the pre-indexed symbol database.
        Returns the enclosing symbol's information.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Document, Symbol
            from sqlalchemy import and_, select

            async with get_async_session() as session:
                # Find document by URI
                doc_stmt = select(Document).where(Document.uri == file_path)
                doc_result = await session.execute(doc_stmt)
                doc = doc_result.scalar_one_or_none()

                if not doc:
                    return f"File not found in index: {file_path}"

                # Find symbols that contain this line
                sym_stmt = (
                    select(Symbol)
                    .where(Symbol.document_id == doc.id)
                    .where(and_(Symbol.start_line <= line, Symbol.end_line >= line))
                    .order_by(Symbol.end_line - Symbol.start_line)  # Smallest first
                )
                sym_result = await session.execute(sym_stmt)
                symbols = sym_result.scalars().all()

                if not symbols:
                    return f"Line {line} is not inside any indexed symbol in {file_path}"

                # Get the innermost (smallest) symbol
                sym = symbols[0]
                lines = (doc.content or "").split("\n")
                start_idx = max(0, sym.start_line - 1)
                end_idx = min(len(lines), sym.end_line)
                content = "\n".join(lines[start_idx:end_idx])

                evidence = Evidence(
                    id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                    file_path=file_path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    content=content[:2000],
                    reason=f"Enclosing {sym.kind.value} for line {line}",
                    provenance="symbol_index",
                    symbol_id=sym.qualified_name,
                    symbol_kind=sym.kind.value,
                )
                run.add_evidence(evidence)

                return f"[{evidence.id}] Line {line} is inside {sym.kind.value} '{sym.name}' (L{sym.start_line}-{sym.end_line})\n```\n{content[:1000]}\n```"

        except Exception as e:
            logger.warning("symbol_enclosing failed: %s", e)
            return f"Symbol enclosing failed: {e}"

    @tool
    async def summarize_evidence(goal: str) -> str:
        """Use LLM to summarize collected evidence into a focused memo.

        Use this when you have gathered many evidence items and need to
        organize them before answering. Specify a goal to focus the summary.
        """
        run = run_holder["run"]

        if not run.evidence:
            return "No evidence collected yet to summarize."

        try:
            # Build evidence summary
            evidence_text = []
            for ev in run.evidence[:20]:  # Limit to 20 items
                evidence_text.append(
                    f"[{ev.id}] {ev.file_path}:{ev.start_line}-{ev.end_line}\n"
                    f"Reason: {ev.reason}\n"
                    f"```\n{ev.content[:500]}\n```"
                )

            system = "You are a code research assistant. Summarize evidence concisely."
            user_prompt = f"""Summarize the following code evidence to address this goal: {goal}

Evidence:
{chr(10).join(evidence_text)}

Provide a concise summary (2-3 paragraphs) focusing on the goal.
Reference evidence IDs like [ev-xxx-001] when making claims."""

            # Use the LLM provider
            from contextmine_core.research.llm import get_research_llm_provider

            provider = get_research_llm_provider()
            response = await provider.generate_text(
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=1000,
            )

            return f"Evidence Summary ({len(run.evidence)} items):\n\n{response}"

        except Exception as e:
            logger.warning("summarize_evidence failed: %s", e)
            return f"Summarize failed: {e}"

    # =========================================================================
    # GRAPH TRAVERSAL TOOLS (use pre-indexed SymbolEdge data)
    # =========================================================================

    @tool
    async def graph_expand(
        seed_names: list[str],
        edge_types: list[str] | None = None,
        depth: int = 2,
        limit: int = 50,
    ) -> str:
        """Expand from seed symbols following relationship types.

        Uses the pre-indexed symbol graph (SymbolEdge table).
        BFS traversal from seeds up to specified depth.

        Args:
            seed_names: List of symbol names to start from
            edge_types: Filter by edge types (calls, references, imports, inherits)
            depth: Max traversal depth (1-5)
            limit: Max nodes to return
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.models import Symbol, SymbolEdge, SymbolEdgeType
            from sqlalchemy import or_, select
            from sqlalchemy.orm import selectinload

            depth = max(1, min(5, depth))
            limit = max(1, min(100, limit))

            # Map edge type strings to enum
            edge_type_map = {
                "calls": SymbolEdgeType.CALLS,
                "references": SymbolEdgeType.REFERENCES,
                "imports": SymbolEdgeType.IMPORTS,
                "inherits": SymbolEdgeType.INHERITS,
                "contains": SymbolEdgeType.CONTAINS,
            }

            filter_types = None
            if edge_types:
                filter_types = [edge_type_map[t] for t in edge_types if t in edge_type_map]

            async with get_async_session() as session:
                # Find seed symbols
                seed_stmt = select(Symbol).where(or_(*[Symbol.name == name for name in seed_names]))
                seed_result = await session.execute(seed_stmt)
                seeds = {s.id: s for s in seed_result.scalars().all()}

                if not seeds:
                    return f"No symbols found matching: {seed_names}"

                visited = set(seeds.keys())
                frontier = set(seeds.keys())
                collected_symbols = dict(seeds)

                # BFS traversal
                for _d in range(depth):
                    if not frontier or len(collected_symbols) >= limit:
                        break

                    # Get edges from frontier
                    edge_stmt = select(SymbolEdge).where(
                        or_(
                            SymbolEdge.source_symbol_id.in_(frontier),
                            SymbolEdge.target_symbol_id.in_(frontier),
                        )
                    )
                    if filter_types:
                        edge_stmt = edge_stmt.where(SymbolEdge.edge_type.in_(filter_types))

                    edge_stmt = edge_stmt.options(
                        selectinload(SymbolEdge.source_symbol).selectinload(Symbol.document),
                        selectinload(SymbolEdge.target_symbol).selectinload(Symbol.document),
                    )

                    edge_result = await session.execute(edge_stmt)
                    edges = edge_result.scalars().all()

                    new_frontier = set()
                    for edge in edges:
                        for sym in [edge.source_symbol, edge.target_symbol]:
                            if sym.id not in visited and len(collected_symbols) < limit:
                                visited.add(sym.id)
                                new_frontier.add(sym.id)
                                collected_symbols[sym.id] = sym

                    frontier = new_frontier

                # Create evidence for collected symbols
                output_parts = []
                for sym in list(collected_symbols.values())[:limit]:
                    doc = sym.document
                    is_seed = sym.name in seed_names
                    marker = "[SEED] " if is_seed else ""

                    lines = (doc.content or "").split("\n")
                    start_idx = max(0, sym.start_line - 1)
                    end_idx = min(len(lines), sym.end_line)
                    content = "\n".join(lines[start_idx:end_idx])

                    evidence = Evidence(
                        id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                        file_path=doc.uri or "unknown",
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        content=content[:1500],
                        reason=f"Graph expansion from {seed_names}",
                        provenance="symbol_graph",
                        symbol_id=sym.qualified_name,
                        symbol_kind=sym.kind.value,
                    )
                    run.add_evidence(evidence)
                    output_parts.append(
                        f"{marker}[{evidence.id}] {sym.kind.value} '{sym.qualified_name}' "
                        f"at {doc.uri}:{sym.start_line}"
                    )

                return (
                    f"Expanded to {len(collected_symbols)} symbols "
                    f"(depth={depth}, seeds={len(seeds)}):\n" + "\n".join(output_parts[:30])
                )

        except Exception as e:
            logger.warning("graph_expand failed: %s", e)
            return f"Graph expand failed: {e}"

    @tool
    async def graph_pack(target_count: int = 10) -> str:
        """Select the most relevant evidence items from collected evidence.

        Scores evidence by:
        - Symbol importance (classes > functions > methods)
        - Content size (larger often more important)
        - Provenance diversity (multiple sources = bonus)

        Returns a ranked list of the most important evidence.
        """
        run = run_holder["run"]

        if not run.evidence:
            return "No evidence collected yet to pack."

        # Score each evidence item
        scored = []
        for ev in run.evidence:
            score = 0.0

            # Symbol kind scoring
            kind_scores = {
                "class": 5.0,
                "struct": 4.5,
                "interface": 4.0,
                "function": 3.0,
                "method": 2.5,
                "variable": 1.0,
            }
            if ev.symbol_kind:
                score += kind_scores.get(ev.symbol_kind.lower(), 1.0)

            # Size scoring (normalized)
            content_lines = len(ev.content.split("\n")) if ev.content else 0
            score += min(content_lines / 20.0, 3.0)  # Max 3 points for size

            # Provenance scoring
            provenance_scores = {
                "lsp": 2.0,
                "symbol_graph": 1.8,
                "symbol_index": 1.5,
                "hybrid": 1.2,
                "manual": 1.0,
            }
            score += provenance_scores.get(ev.provenance, 1.0)

            # Score bonus for having a score
            if ev.score:
                score += ev.score * 2.0

            scored.append((score, ev))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build output
        output_parts = [f"Top {min(target_count, len(scored))} evidence items:\n"]
        for i, (score, ev) in enumerate(scored[:target_count]):
            kind = f" ({ev.symbol_kind})" if ev.symbol_kind else ""
            output_parts.append(
                f"{i + 1}. [{ev.id}] score={score:.1f}{kind}\n"
                f"   {ev.file_path}:{ev.start_line}-{ev.end_line}\n"
                f"   Reason: {ev.reason}"
            )

        return "\n".join(output_parts)

    @tool
    async def graph_trace(
        from_symbol: str,
        to_symbol: str,
        edge_types: list[str] | None = None,
        max_depth: int = 5,
    ) -> str:
        """Find paths between two symbols in the code graph.

        Uses bidirectional BFS to find shortest paths.
        Useful for impact analysis and understanding code flow.

        Args:
            from_symbol: Source symbol name
            to_symbol: Target symbol name
            edge_types: Filter by edge types (calls, references, imports, inherits)
            max_depth: Maximum path length
        """
        run = run_holder["run"]
        try:
            from collections import deque

            from contextmine_core.database import get_async_session
            from contextmine_core.models import Symbol, SymbolEdge, SymbolEdgeType
            from sqlalchemy import or_, select
            from sqlalchemy.orm import selectinload

            max_depth = max(1, min(10, max_depth))

            edge_type_map = {
                "calls": SymbolEdgeType.CALLS,
                "references": SymbolEdgeType.REFERENCES,
                "imports": SymbolEdgeType.IMPORTS,
                "inherits": SymbolEdgeType.INHERITS,
            }

            filter_types = None
            if edge_types:
                filter_types = [edge_type_map[t] for t in edge_types if t in edge_type_map]

            async with get_async_session() as session:
                # Find source and target symbols
                sym_stmt = (
                    select(Symbol)
                    .where(or_(Symbol.name == from_symbol, Symbol.name == to_symbol))
                    .options(selectinload(Symbol.document))
                )
                sym_result = await session.execute(sym_stmt)
                symbols = {s.name: s for s in sym_result.scalars().all()}

                if from_symbol not in symbols:
                    return f"Source symbol '{from_symbol}' not found"
                if to_symbol not in symbols:
                    return f"Target symbol '{to_symbol}' not found"

                source = symbols[from_symbol]
                target = symbols[to_symbol]

                if source.id == target.id:
                    return "Source and target are the same symbol"

                # BFS to find path
                queue = deque([(source.id, [source.id])])
                visited = {source.id}
                found_paths = []

                while queue and len(found_paths) < 3:
                    current_id, path = queue.popleft()

                    if len(path) > max_depth:
                        continue

                    # Get outgoing edges
                    edge_stmt = select(SymbolEdge).where(SymbolEdge.source_symbol_id == current_id)
                    if filter_types:
                        edge_stmt = edge_stmt.where(SymbolEdge.edge_type.in_(filter_types))

                    edge_stmt = edge_stmt.options(
                        selectinload(SymbolEdge.target_symbol).selectinload(Symbol.document)
                    )

                    edge_result = await session.execute(edge_stmt)
                    edges = edge_result.scalars().all()

                    for edge in edges:
                        next_sym = edge.target_symbol
                        if next_sym.id == target.id:
                            found_paths.append(path + [next_sym.id])
                        elif next_sym.id not in visited:
                            visited.add(next_sym.id)
                            queue.append((next_sym.id, path + [next_sym.id]))

                if not found_paths:
                    return f"No path found from '{from_symbol}' to '{to_symbol}' within depth {max_depth}"

                # Get all symbols in paths
                all_ids = set()
                for p in found_paths:
                    all_ids.update(p)

                sym_stmt = (
                    select(Symbol)
                    .where(Symbol.id.in_(all_ids))
                    .options(selectinload(Symbol.document))
                )
                sym_result = await session.execute(sym_stmt)
                sym_map = {s.id: s for s in sym_result.scalars().all()}

                # Create evidence and output
                output_parts = [f"Found {len(found_paths)} path(s):\n"]
                for i, path in enumerate(found_paths[:3]):
                    path_names = []
                    for sym_id in path:
                        sym = sym_map.get(sym_id)
                        if sym:
                            path_names.append(sym.name)

                            # Add evidence for each symbol in path
                            doc = sym.document
                            lines = (doc.content or "").split("\n")
                            start_idx = max(0, sym.start_line - 1)
                            end_idx = min(len(lines), sym.end_line)
                            content = "\n".join(lines[start_idx:end_idx])

                            evidence = Evidence(
                                id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                                file_path=doc.uri or "unknown",
                                start_line=sym.start_line,
                                end_line=sym.end_line,
                                content=content[:1500],
                                reason=f"Path step {path.index(sym_id) + 1} from '{from_symbol}' to '{to_symbol}'",
                                provenance="symbol_graph",
                                symbol_id=sym.qualified_name,
                                symbol_kind=sym.kind.value,
                            )
                            run.add_evidence(evidence)

                    output_parts.append(f"Path {i + 1}: {' -> '.join(path_names)}")

                return "\n".join(output_parts)

        except Exception as e:
            logger.warning("graph_trace failed: %s", e)
            return f"Graph trace failed: {e}"

    # =========================================================================
    # GRAPHRAG TOOLS (use Knowledge Graph with community-aware retrieval)
    # =========================================================================

    @tool
    async def graphrag_search(query: str, max_entities: int = 15) -> str:
        """Search using GraphRAG with community-aware retrieval.

        This is the most powerful search tool - it combines:
        1. Community summaries (global context from code clusters)
        2. Hybrid search (BM25 + vector)
        3. Knowledge Graph expansion (files, symbols, DB tables, APIs, rules)

        Use this for broad architectural questions or to understand
        how different parts of the codebase relate.

        Args:
            query: Natural language query
            max_entities: Maximum entities to return (default 15)
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.graphrag import graph_rag_context

            async with get_async_session() as session:
                context = await graph_rag_context(
                    session=session,
                    query=query,
                    collection_id=None,
                    user_id=None,
                    max_communities=5,
                    max_entities=max_entities,
                    max_depth=2,
                )

            # Convert to evidence
            output_parts = []

            # Add community context
            if context.communities:
                output_parts.append(f"## Global Context ({len(context.communities)} communities)\n")
                for comm in context.communities[:3]:
                    output_parts.append(f"**{comm.title}** (relevance: {comm.relevance_score:.0%})")
                    if comm.summary:
                        summary = (
                            comm.summary[:300] + "..." if len(comm.summary) > 300 else comm.summary
                        )
                        output_parts.append(summary)
                    output_parts.append("")

            # Add entity context as evidence
            if context.entities:
                output_parts.append(f"## Local Context ({len(context.entities)} entities)\n")
                for entity in context.entities[:max_entities]:
                    # Create evidence for each entity
                    citations = entity.evidence[:1]
                    if citations:
                        cit = citations[0]
                        evidence = Evidence(
                            id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                            file_path=cit.file_path,
                            start_line=cit.start_line,
                            end_line=cit.end_line,
                            content=cit.snippet or "",
                            reason=f"GraphRAG: {entity.kind} '{entity.name}'",
                            provenance="graphrag",
                        )
                        run.add_evidence(evidence)
                        output_parts.append(
                            f"[{evidence.id}] {entity.kind}: {entity.name} "
                            f"({cit.file_path}:{cit.start_line})"
                        )
                    else:
                        output_parts.append(f"- {entity.kind}: {entity.name}")

            # Add relationship summary
            if context.edges:
                edge_counts: dict[str, int] = {}
                for edge in context.edges:
                    edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1
                output_parts.append(f"\n## Relationships ({len(context.edges)} edges)")
                for kind, count in sorted(edge_counts.items(), key=lambda x: -x[1])[:5]:
                    output_parts.append(f"- {kind}: {count}")

            if not output_parts:
                return "No relevant results found. Try a different query or use hybrid_search."

            return "\n".join(output_parts)

        except Exception as e:
            logger.warning("graphrag_search failed: %s", e)
            return f"GraphRAG search failed: {e}. Try hybrid_search as fallback."

    @tool
    async def kg_neighborhood(node_name: str, depth: int = 1, node_kind: str | None = None) -> str:
        """Explore the Knowledge Graph neighborhood of a node.

        Returns connected nodes including:
        - Files and symbols
        - Database tables and columns
        - API endpoints and schemas
        - Business rules

        Args:
            node_name: Name of the node to explore (file path, symbol name, table name)
            depth: Expansion depth (1-3, default 1)
            node_kind: Optional kind filter (FILE, SYMBOL, DB_TABLE, API_ENDPOINT, etc.)
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.graphrag import graph_neighborhood
            from contextmine_core.models import KnowledgeNode
            from sqlalchemy import select

            depth = max(1, min(3, depth))

            async with get_async_session() as session:
                # Find node by name
                stmt = select(KnowledgeNode).where(KnowledgeNode.name == node_name)
                if node_kind:
                    from contextmine_core.models import KnowledgeNodeKind

                    try:
                        kind_enum = KnowledgeNodeKind(node_kind.upper())
                        stmt = stmt.where(KnowledgeNode.kind == kind_enum)
                    except ValueError:
                        pass
                stmt = stmt.limit(1)

                result = await session.execute(stmt)
                node = result.scalar_one_or_none()

                if not node:
                    # Try by natural_key pattern
                    stmt = (
                        select(KnowledgeNode)
                        .where(KnowledgeNode.natural_key.ilike(f"%{node_name}%"))
                        .limit(1)
                    )
                    result = await session.execute(stmt)
                    node = result.scalar_one_or_none()

                if not node:
                    return f"Node '{node_name}' not found in Knowledge Graph"

                # Get neighborhood
                context = await graph_neighborhood(
                    session=session,
                    node_id=node.id,
                    collection_id=node.collection_id,
                    depth=depth,
                    max_nodes=30,
                )

            # Format output
            output_parts = [
                f"## Neighborhood of {node.kind.value}: {node.name}",
                f"Found {len(context.entities)} nodes, {len(context.edges)} edges\n",
            ]

            # Group by kind
            by_kind: dict[str, list] = {}
            for entity in context.entities:
                by_kind.setdefault(entity.kind, []).append(entity)

            for kind, entities in sorted(by_kind.items()):
                output_parts.append(f"**{kind}** ({len(entities)}):")
                for entity in entities[:10]:
                    # Create evidence
                    if entity.evidence:
                        cit = entity.evidence[0]
                        evidence = Evidence(
                            id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                            file_path=cit.file_path,
                            start_line=cit.start_line,
                            end_line=cit.end_line,
                            content=cit.snippet or "",
                            reason=f"KG neighbor: {entity.kind} '{entity.name}'",
                            provenance="knowledge_graph",
                        )
                        run.add_evidence(evidence)
                        output_parts.append(f"  [{evidence.id}] {entity.name}")
                    else:
                        output_parts.append(f"  - {entity.name}")
                if len(entities) > 10:
                    output_parts.append(f"  ... and {len(entities) - 10} more")
                output_parts.append("")

            # Show edge types
            if context.edges:
                edge_counts: dict[str, int] = {}
                for edge in context.edges:
                    edge_counts[edge.kind] = edge_counts.get(edge.kind, 0) + 1
                output_parts.append("**Relationships:**")
                for kind, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
                    output_parts.append(f"  - {kind}: {count}")

            return "\n".join(output_parts)

        except Exception as e:
            logger.warning("kg_neighborhood failed: %s", e)
            return f"Knowledge Graph neighborhood failed: {e}"

    @tool
    async def kg_path(from_name: str, to_name: str, max_hops: int = 6) -> str:
        """Find path between two nodes in the Knowledge Graph.

        Works across different node types (files, symbols, tables, APIs).
        Useful for understanding dependencies and impact.

        Args:
            from_name: Source node name
            to_name: Target node name
            max_hops: Maximum path length (default 6)
        """
        run = run_holder["run"]
        try:
            from contextmine_core.database import get_async_session
            from contextmine_core.graphrag import trace_path
            from contextmine_core.models import KnowledgeNode
            from sqlalchemy import select

            max_hops = max(1, min(10, max_hops))

            async with get_async_session() as session:
                # Find source node
                stmt = select(KnowledgeNode).where(KnowledgeNode.name == from_name).limit(1)
                result = await session.execute(stmt)
                from_node = result.scalar_one_or_none()

                if not from_node:
                    stmt = (
                        select(KnowledgeNode)
                        .where(KnowledgeNode.natural_key.ilike(f"%{from_name}%"))
                        .limit(1)
                    )
                    result = await session.execute(stmt)
                    from_node = result.scalar_one_or_none()

                if not from_node:
                    return f"Source node '{from_name}' not found"

                # Find target node
                stmt = select(KnowledgeNode).where(KnowledgeNode.name == to_name).limit(1)
                result = await session.execute(stmt)
                to_node = result.scalar_one_or_none()

                if not to_node:
                    stmt = (
                        select(KnowledgeNode)
                        .where(KnowledgeNode.natural_key.ilike(f"%{to_name}%"))
                        .limit(1)
                    )
                    result = await session.execute(stmt)
                    to_node = result.scalar_one_or_none()

                if not to_node:
                    return f"Target node '{to_name}' not found"

                # Find path
                context = await trace_path(
                    session=session,
                    from_node_id=from_node.id,
                    to_node_id=to_node.id,
                    collection_id=from_node.collection_id,
                    max_hops=max_hops,
                )

            if not context.entities:
                return f"No path found from '{from_name}' to '{to_name}' within {max_hops} hops"

            # Format output
            output_parts = [f"## Path: {from_name} → {to_name}\n"]

            if context.paths:
                output_parts.append(f"**Route:** {context.paths[0].description}\n")

            output_parts.append("**Steps:**")
            for i, entity in enumerate(context.entities):
                arrow = "→" if i < len(context.entities) - 1 else ""
                edge_kind = context.edges[i].kind if i < len(context.edges) else ""

                # Create evidence
                if entity.evidence:
                    cit = entity.evidence[0]
                    evidence = Evidence(
                        id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                        file_path=cit.file_path,
                        start_line=cit.start_line,
                        end_line=cit.end_line,
                        content=cit.snippet or "",
                        reason=f"Path step {i + 1}: {entity.kind} '{entity.name}'",
                        provenance="knowledge_graph",
                    )
                    run.add_evidence(evidence)
                    output_parts.append(f"  {i + 1}. [{evidence.id}] {entity.kind}: {entity.name}")
                else:
                    output_parts.append(f"  {i + 1}. {entity.kind}: {entity.name}")

                if edge_kind:
                    output_parts.append(f"     {arrow} ({edge_kind})")

            return "\n".join(output_parts)

        except Exception as e:
            logger.warning("kg_path failed: %s", e)
            return f"Knowledge Graph path failed: {e}"

    # Build tools list - all tools use pre-indexed data
    tools = [
        # Core tools
        hybrid_search,
        open_span,
        finalize,
        summarize_evidence,
        # Definition/reference tools (pre-indexed)
        goto_definition,
        find_references,
        get_signature,
        # Symbol index tools (pre-indexed)
        symbol_outline,
        symbol_find,
        symbol_enclosing,
        symbol_callers,
        symbol_callees,
        # Graph traversal tools (pre-indexed Symbol graph)
        graph_expand,
        graph_pack,
        graph_trace,
        # GraphRAG tools (Knowledge Graph with communities)
        graphrag_search,
        kg_neighborhood,
        kg_path,
    ]

    return tools


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for the research agent."""

    max_steps: int = 10
    """Maximum number of agent steps."""

    max_verification_retries: int = 2
    """Maximum verification retry attempts."""

    store_artifacts: bool = True
    """Whether to store artifacts."""


# =============================================================================
# RESEARCH AGENT
# =============================================================================


@dataclass
class ResearchAgent:
    """LangGraph-based research agent using proper tool architecture.

    Uses:
    - bind_tools() for LLM tool calling
    - ToolNode for tool execution
    - Verification node for answer grounding check
    """

    llm_provider: Any  # LLMProvider
    """LLM provider for the agent."""

    config: AgentConfig = field(default_factory=AgentConfig)
    """Agent configuration."""

    async def research(
        self,
        question: str,
        scope: str | None = None,
    ) -> ResearchRun:
        """Execute a research investigation.

        Args:
            question: The research question
            scope: Optional path pattern to limit search

        Returns:
            Completed ResearchRun with answer, evidence, and trace
        """
        run = ResearchRun.create(
            question=question,
            scope=scope,
            budget_steps=self.config.max_steps,
        )

        logger.info("Starting research run %s: %s", run.run_id[:8], question[:50])

        try:
            # Shared state for tools
            run_holder: dict[str, Any] = {
                "run": run,
                "pending_answer": None,
                "confidence": 0.8,
            }

            # Build and run the graph
            graph = self._build_graph(run_holder)

            # Initial state
            system_msg = self._build_system_prompt(question, scope)
            initial_state: AgentState = {
                "messages": [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=question),
                ],
                "run": run,
                "pending_answer": None,
                "verification_attempts": 0,
            }

            # Run the graph
            final_state = await graph.ainvoke(initial_state)
            run = final_state["run"]

        except Exception as e:
            logger.exception("Research failed: %s", e)
            run.fail(str(e))

        # Store artifacts
        if self.config.store_artifacts:
            try:
                store = get_artifact_store()
                store.save_run(run)
                logger.info("Saved artifacts for run %s", run.run_id[:8])
            except Exception as e:
                logger.warning("Failed to save artifacts: %s", e)

        return run

    def _build_graph(self, run_holder: dict[str, Any]) -> Any:
        """Build the LangGraph workflow."""
        # Create tools
        tools = create_tools(run_holder)

        # Get LangChain model and bind tools
        model = self.llm_provider._model
        model_with_tools = model.bind_tools(tools)

        # Define nodes
        async def agent_node(state: AgentState) -> dict:
            """Agent node - calls LLM with tools."""
            run = state["run"]

            # Check budget
            if run.budget_used >= run.budget_steps:
                if not run.answer:
                    run.complete(
                        "Research budget exhausted. Based on evidence collected, "
                        "I could not find sufficient information to answer the question."
                    )
                return {"run": run}

            response = await model_with_tools.ainvoke(state["messages"])
            run.budget_used = len([m for m in state["messages"] if isinstance(m, AIMessage)])

            return {"messages": [response], "run": run}

        async def verify_node(state: AgentState) -> dict[str, object]:
            """Verify the answer is grounded in evidence."""
            run = state["run"]
            pending = run_holder.get("pending_answer")
            attempts = state.get("verification_attempts", 0)

            if not pending:
                return dict(state)

            # Set answer for verification
            run.answer = pending

            # Verify
            verifier = AnswerVerifier(llm_provider=self.llm_provider)
            verification = await verifier.verify_async(run)
            run.verification = verification

            if verification.status == VerificationStatus.FAILED:
                run.answer = None
                run_holder["pending_answer"] = None

                logger.warning(
                    "Answer rejected for run %s (attempt %d): %s",
                    run.run_id[:8],
                    attempts + 1,
                    verification.issues[:2],
                )

                if attempts < self.config.max_verification_retries:
                    # Add feedback message for retry
                    feedback = (
                        f"Your answer was rejected because it's not grounded in evidence. "
                        f"Issues: {'; '.join(verification.issues[:3])}. "
                        f"Please gather more evidence and try again."
                    )
                    return {
                        "messages": [HumanMessage(content=feedback)],
                        "run": run,
                        "pending_answer": None,
                        "verification_attempts": attempts + 1,
                    }
                else:
                    run.fail(
                        f"Verification failed after {attempts + 1} attempts: "
                        f"{'; '.join(verification.issues[:3])}"
                    )
                    return {"run": run, "pending_answer": None}

            # Verification passed
            run.complete(pending)
            run_holder["pending_answer"] = None

            logger.info(
                "Research run %s verified and completed",
                run.run_id[:8],
            )

            return {"run": run, "pending_answer": None}

        def should_verify(state: AgentState) -> str:
            """Check if we need to verify an answer."""
            if run_holder.get("pending_answer"):
                return "verify"
            return "continue"

        def after_tools(state: AgentState) -> str:
            """Route after tool execution."""
            if run_holder.get("pending_answer"):
                return "verify"
            return "agent"

        def after_verify(state: AgentState) -> str:
            """Route after verification."""
            run = state["run"]
            if run.answer or run.error_message:
                return END
            # Retry - go back to agent
            return "agent"

        # Build graph
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("verify", verify_node)

        workflow.set_entry_point("agent")

        # Agent -> tools or end
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
        )

        # Tools -> verify or agent
        workflow.add_conditional_edges(
            "tools",
            after_tools,
            {"verify": "verify", "agent": "agent"},
        )

        # Verify -> end or retry
        workflow.add_conditional_edges(
            "verify",
            after_verify,
        )

        return workflow.compile()

    def _build_system_prompt(self, question: str, scope: str | None) -> str:
        """Build the system prompt for the agent."""
        scope_instruction = ""
        if scope:
            scope_instruction = f"\n\nLimit your search to files matching: {scope}"

        return f"""You are a code research agent investigating a codebase to answer questions.

Your task: {question}{scope_instruction}

## Available Tools (all use pre-indexed data)

### GraphRAG (Recommended for architectural questions)
- **graphrag_search** - MOST POWERFUL: Community-aware search combining global context (code clusters) + local entities (files, symbols, tables, APIs, rules)
- **kg_neighborhood** - Explore Knowledge Graph around a node (files, symbols, DB tables, APIs, business rules)
- **kg_path** - Find paths between any two nodes in the Knowledge Graph (cross-type: file→table, symbol→API, etc.)

### RAG Search
- **hybrid_search** - Search the codebase using BM25 + vector retrieval (fallback if GraphRAG unavailable)

### Definition & References
- **goto_definition** - Jump to where a symbol is defined
- **find_references** - Find all usages of a symbol across the codebase
- **get_signature** - Get type signature and documentation for a symbol

### Symbol Index
- **symbol_outline** - Get all symbols in a file (functions, classes, methods)
- **symbol_find** - Find a symbol by name across the codebase
- **symbol_enclosing** - Find what symbol (function/class) contains a specific line
- **symbol_callers** - Find all functions that call a given symbol
- **symbol_callees** - Find all functions that a symbol calls

### Symbol Graph Traversal
- **graph_expand** - Expand from seed symbols following relationships (calls, references, imports)
- **graph_pack** - Select the most relevant evidence items from collected evidence
- **graph_trace** - Find paths between two symbols (for impact analysis)

### Read & Summarize
- **open_span** - Read specific lines from a file
- **summarize_evidence** - Use LLM to compress collected evidence into a memo

### Finalize
- **finalize** - Submit your final answer with citations

## Instructions

1. For architectural questions, start with **graphrag_search** (includes community summaries + entities)
2. For specific symbol lookups, use hybrid_search or symbol_find
3. Use goto_definition/find_references for precise symbol navigation
4. Use kg_neighborhood to explore Knowledge Graph (includes DB tables, APIs, business rules)
5. Use kg_path to trace dependencies across different node types
6. Use symbol_* tools for detailed code structure exploration
7. Use graph_* tools for symbol-level multi-hop traversal
8. Use open_span to examine specific code sections in detail
9. Use summarize_evidence when you have many items and need to organize
10. Call finalize with your answer including citation IDs like [ev-abc-001]

## Important

- Your answer will be VERIFIED against the evidence
- Only make claims that are directly supported by the evidence
- If you cannot find relevant information, say so honestly
- Include citation IDs for all claims

Begin your investigation."""


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


async def run_research(
    question: str,
    scope: str | None = None,
    llm_provider: Any = None,
    max_steps: int = 10,
    store_artifacts: bool = True,
) -> ResearchRun:
    """Convenience function to run a research investigation.

    Args:
        question: The research question
        scope: Optional path pattern to limit search
        llm_provider: LLM provider (uses default if None)
        max_steps: Maximum steps
        store_artifacts: Whether to store artifacts

    Returns:
        Completed ResearchRun
    """
    if llm_provider is None:
        from contextmine_core.research.llm import get_research_llm_provider

        llm_provider = get_research_llm_provider()

    config = AgentConfig(
        max_steps=max_steps,
        store_artifacts=store_artifacts,
    )

    agent = ResearchAgent(
        llm_provider=llm_provider,
        config=config,
    )

    return await agent.research(question, scope)
