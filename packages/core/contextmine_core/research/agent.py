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
    # LSP TOOLS
    # =========================================================================

    @tool
    async def lsp_definition(file_path: str, line: int, column: int = 0) -> str:
        """Jump to the definition of a symbol using LSP.

        Use this to find where a function, class, or variable is defined.
        Provide the file path, line number (1-indexed), and column (0-indexed).
        """
        run = run_holder["run"]
        try:
            from contextmine_core.lsp import get_lsp_manager

            manager = get_lsp_manager()
            client = await manager.get_client(file_path)
            locations = await client.get_definition(file_path, line, column)

            if not locations:
                return f"No definition found at {file_path}:{line}:{column}"

            output_parts = []
            for loc in locations:
                # Read content at the location
                from pathlib import Path

                path = Path(loc.file_path)
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="replace")
                    lines = content.split("\n")
                    start_idx = max(0, loc.start_line - 1)
                    end_idx = min(len(lines), loc.end_line + 10)
                    snippet = "\n".join(lines[start_idx:end_idx])
                else:
                    snippet = "[File not accessible]"

                evidence = Evidence(
                    id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                    file_path=loc.file_path,
                    start_line=loc.start_line,
                    end_line=loc.end_line + 10,
                    content=snippet[:2000],
                    reason="Definition found via LSP",
                    provenance="lsp",
                    symbol_kind="definition",
                )
                run.add_evidence(evidence)
                output_parts.append(
                    f"[{evidence.id}] {loc.file_path}:{loc.start_line}\n```\n{snippet[:500]}\n```"
                )

            return f"Found {len(locations)} definition(s):\n\n" + "\n\n".join(output_parts)

        except Exception as e:
            logger.warning("lsp_definition failed: %s", e)
            return f"LSP definition failed: {e}"

    @tool
    async def lsp_references(file_path: str, line: int, column: int = 0) -> str:
        """Find all usages of a symbol in the codebase using LSP.

        Use this to find where a function, class, or variable is used.
        Provide the file path, line number (1-indexed), and column (0-indexed).
        """
        run = run_holder["run"]
        try:
            from contextmine_core.lsp import get_lsp_manager

            manager = get_lsp_manager()
            client = await manager.get_client(file_path)
            locations = await client.get_references(file_path, line, column)

            if not locations:
                return f"No references found at {file_path}:{line}:{column}"

            output_parts = []
            for i, loc in enumerate(locations[:10]):  # Limit to 10 references
                from pathlib import Path

                path = Path(loc.file_path)
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="replace")
                    lines = content.split("\n")
                    start_idx = max(0, loc.start_line - 3)
                    end_idx = min(len(lines), loc.end_line + 3)
                    snippet = "\n".join(lines[start_idx:end_idx])
                else:
                    snippet = "[File not accessible]"

                evidence = Evidence(
                    id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                    file_path=loc.file_path,
                    start_line=max(1, loc.start_line - 2),
                    end_line=loc.end_line + 2,
                    content=snippet[:1500],
                    reason=f"Reference {i + 1} of {len(locations)} via LSP",
                    provenance="lsp",
                    symbol_kind="reference",
                )
                run.add_evidence(evidence)
                output_parts.append(
                    f"[{evidence.id}] {loc.file_path}:{loc.start_line}\n```\n{snippet[:300]}\n```"
                )

            summary = f"Found {len(locations)} reference(s)"
            if len(locations) > 10:
                summary += " (showing first 10)"
            return summary + ":\n\n" + "\n\n".join(output_parts)

        except Exception as e:
            logger.warning("lsp_references failed: %s", e)
            return f"LSP references failed: {e}"

    @tool
    async def lsp_hover(file_path: str, line: int, column: int = 0) -> str:
        """Get type signature and documentation for a symbol using LSP.

        Use this to understand what a function or class does without reading all its code.
        Provide the file path, line number (1-indexed), and column (0-indexed).
        """
        run = run_holder["run"]
        try:
            from contextmine_core.lsp import get_lsp_manager

            manager = get_lsp_manager()
            client = await manager.get_client(file_path)
            info = await client.get_hover(file_path, line, column)

            if not info:
                return f"No hover info at {file_path}:{line}:{column}"

            content_parts = []
            if info.signature:
                content_parts.append(f"Signature: {info.signature}")
            if info.documentation:
                content_parts.append(f"Documentation:\n{info.documentation}")

            content = "\n\n".join(content_parts) or f"{info.kind}: {info.name}"

            evidence = Evidence(
                id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                file_path=file_path,
                start_line=line,
                end_line=line,
                content=content[:2000],
                reason=f"Hover info for {info.name} ({info.kind})",
                provenance="lsp",
                symbol_id=info.name,
                symbol_kind=info.kind,
            )
            run.add_evidence(evidence)

            return f"[{evidence.id}] {info.kind} '{info.name}':\n{content}"

        except Exception as e:
            logger.warning("lsp_hover failed: %s", e)
            return f"LSP hover failed: {e}"

    # =========================================================================
    # TREE-SITTER TOOLS
    # =========================================================================

    @tool
    async def ts_outline(file_path: str) -> str:
        """Get the outline of all functions, classes, and symbols in a file.

        Use this to quickly understand the structure of a file without reading all code.
        Returns a list of symbols with their line numbers.
        """
        try:
            from contextmine_core.treesitter import extract_outline

            symbols = extract_outline(file_path, include_children=True)

            if not symbols:
                return f"No symbols found in {file_path}"

            outline_lines = []
            for sym in symbols:
                outline_lines.append(
                    f"{sym.kind.value} {sym.name} (L{sym.start_line}-{sym.end_line})"
                )
                for child in sym.children:
                    outline_lines.append(
                        f"  {child.kind.value} {child.name} (L{child.start_line}-{child.end_line})"
                    )

            summary = f"Found {len(symbols)} top-level symbols:\n" + "\n".join(outline_lines[:30])
            if len(outline_lines) > 30:
                summary += f"\n... and {len(outline_lines) - 30} more"

            return summary

        except Exception as e:
            logger.warning("ts_outline failed: %s", e)
            return f"Tree-sitter outline failed: {e}"

    @tool
    async def ts_find_symbol(file_path: str, name: str) -> str:
        """Find a specific function, class, or method by name in a file.

        Use this when you know the symbol name but want to see its full implementation.
        Returns the symbol's source code as evidence.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.treesitter import find_symbol_by_name, get_symbol_content

            symbol = find_symbol_by_name(file_path, name)

            if not symbol:
                return f"Symbol '{name}' not found in {file_path}"

            content = get_symbol_content(symbol)

            evidence = Evidence(
                id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                file_path=file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                content=content[:2000],
                reason=f"Found {symbol.kind.value} '{name}' via Tree-sitter",
                provenance="treesitter",
                symbol_id=symbol.name,
                symbol_kind=symbol.kind.value,
            )
            run.add_evidence(evidence)

            return f"[{evidence.id}] {symbol.kind.value} '{name}' at {file_path}:{symbol.start_line}-{symbol.end_line}\n```\n{content[:1000]}\n```"

        except Exception as e:
            logger.warning("ts_find_symbol failed: %s", e)
            return f"Tree-sitter find_symbol failed: {e}"

    @tool
    async def ts_enclosing_symbol(file_path: str, line: int) -> str:
        """Find what function, class, or method contains a specific line.

        Use this to understand the context of a code location.
        Returns the enclosing symbol's source code as evidence.
        """
        run = run_holder["run"]
        try:
            from contextmine_core.treesitter import find_enclosing_symbol, get_symbol_content

            symbol = find_enclosing_symbol(file_path, line)

            if not symbol:
                return f"Line {line} is not inside any symbol in {file_path}"

            content = get_symbol_content(symbol)

            evidence = Evidence(
                id=f"ev-{run.run_id[:8]}-{len(run.evidence) + 1:03d}",
                file_path=file_path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                content=content[:2000],
                reason=f"Enclosing {symbol.kind.value} for line {line}",
                provenance="treesitter",
                symbol_id=symbol.name,
                symbol_kind=symbol.kind.value,
            )
            run.add_evidence(evidence)

            return f"[{evidence.id}] Line {line} is inside {symbol.kind.value} '{symbol.name}' (L{symbol.start_line}-{symbol.end_line})\n```\n{content[:1000]}\n```"

        except Exception as e:
            logger.warning("ts_enclosing_symbol failed: %s", e)
            return f"Tree-sitter enclosing_symbol failed: {e}"

    # Build tools list
    tools = [hybrid_search, open_span, finalize]

    # Add LSP tools (may fail if LSP not available)
    try:
        from contextmine_core.lsp import get_lsp_manager  # noqa: F401

        tools.extend([lsp_definition, lsp_references, lsp_hover])
    except ImportError:
        logger.info("LSP tools not available (multilspy not installed)")

    # Add Tree-sitter tools (may fail if tree-sitter not available)
    try:
        from contextmine_core.treesitter import extract_outline  # noqa: F401

        tools.extend([ts_outline, ts_find_symbol, ts_enclosing_symbol])
    except ImportError:
        logger.info("Tree-sitter tools not available")

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

## Available Tools

### Search & Read
- **hybrid_search** - Search the codebase using BM25 + vector retrieval
- **open_span** - Read specific lines from a file

### LSP (Language Server Protocol)
- **lsp_definition** - Jump to where a symbol is defined
- **lsp_references** - Find all usages of a symbol across the codebase
- **lsp_hover** - Get type signature and documentation for a symbol

### Tree-sitter (Code Structure)
- **ts_outline** - Get outline of functions, classes, and symbols in a file
- **ts_find_symbol** - Find a specific symbol by name and get its source code
- **ts_enclosing_symbol** - Find what function/class contains a specific line

### Finalize
- **finalize** - Submit your final answer with citations

## Instructions

1. Start by searching for relevant code using hybrid_search
2. Use LSP tools to navigate definitions and find usages
3. Use Tree-sitter tools to understand file structure
4. Use open_span to examine specific code sections in detail
5. Collect evidence until you can confidently answer the question
6. Call finalize with your answer including citation IDs like [ev-abc-001]

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
