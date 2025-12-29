"""Action schemas for the research agent.

These Pydantic models define the structured inputs and outputs for each
action the research agent can take during investigation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# ACTION INPUT SCHEMAS
# =============================================================================


class HybridSearchInput(BaseModel):
    """Input for hybrid_search action."""

    query: str = Field(description="Search query for finding relevant code")
    k: int = Field(default=10, description="Number of results to return", ge=1, le=50)


class OpenSpanInput(BaseModel):
    """Input for open_span action."""

    file_path: str = Field(description="Path to the file to read")
    start_line: int = Field(description="Starting line number (1-indexed)", ge=1)
    end_line: int = Field(description="Ending line number (1-indexed, inclusive)", ge=1)


class SummarizeEvidenceInput(BaseModel):
    """Input for summarize_evidence action."""

    goal: str = Field(description="What aspect of the evidence to summarize")


class FinalizeInput(BaseModel):
    """Input for finalize action."""

    answer: str = Field(description="The final answer to the research question")
    confidence: float = Field(
        default=0.8,
        description="Confidence in the answer (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


# =============================================================================
# LSP ACTION INPUT SCHEMAS
# =============================================================================


class LspPositionInput(BaseModel):
    """Input for LSP position-based actions (definition, references, hover)."""

    file_path: str = Field(description="Path to the source file")
    line: int = Field(description="Line number (1-indexed)", ge=1)
    column: int = Field(description="Column number (0-indexed)", ge=0)


class LspDiagnosticsInput(BaseModel):
    """Input for lsp_diagnostics action."""

    file_paths: list[str] = Field(description="List of file paths to check for diagnostics")


# =============================================================================
# TREE-SITTER ACTION INPUT SCHEMAS
# =============================================================================


class TsOutlineInput(BaseModel):
    """Input for ts_outline action."""

    file_path: str = Field(description="Path to the source file to outline")


class TsFindSymbolInput(BaseModel):
    """Input for ts_find_symbol action."""

    file_path: str = Field(description="Path to the source file")
    name: str = Field(description="Name of the symbol to find")


class TsEnclosingSymbolInput(BaseModel):
    """Input for ts_enclosing_symbol action."""

    file_path: str = Field(description="Path to the source file")
    line: int = Field(description="Line number (1-indexed)", ge=1)


# =============================================================================
# GRAPH ACTION INPUT SCHEMAS
# =============================================================================


class GraphExpandInput(BaseModel):
    """Input for graph_expand action."""

    seeds: list[str] = Field(
        description="Seed symbol IDs to expand from (format: 'file_path::symbol_name')"
    )
    edge_types: list[str] | None = Field(
        default=None,
        description="Edge types to follow: contains, defines, references, calls, imports, inherits",
    )
    depth: int = Field(default=2, description="Maximum traversal depth", ge=1, le=5)
    limit: int = Field(default=50, description="Maximum nodes to collect", ge=1, le=200)


class GraphPackInput(BaseModel):
    """Input for graph_pack action."""

    node_ids: list[str] | None = Field(
        default=None,
        description="Specific node IDs to consider (None for all in context)",
    )
    target_count: int = Field(default=10, description="Maximum nodes to select", ge=1, le=50)


class GraphTraceInput(BaseModel):
    """Input for graph_trace action."""

    from_symbol: str = Field(description="Starting symbol ID (format: 'file_path::symbol_name')")
    to_symbol: str = Field(description="Target symbol ID (format: 'file_path::symbol_name')")
    edge_types: list[str] | None = Field(
        default=None,
        description="Edge types to follow (None for all)",
    )


# =============================================================================
# ACTION SELECTION SCHEMA
# =============================================================================


class ActionSelection(BaseModel):
    """The agent's choice of next action.

    This is the structured output from the LLM when deciding what to do next.
    """

    action: Literal[
        "hybrid_search",
        "open_span",
        "summarize_evidence",
        "finalize",
        "lsp_definition",
        "lsp_references",
        "lsp_hover",
        "lsp_diagnostics",
        "ts_outline",
        "ts_find_symbol",
        "ts_enclosing_symbol",
        "graph_expand",
        "graph_pack",
        "graph_trace",
    ] = Field(description="The action to take")
    reasoning: str = Field(description="Brief explanation of why this action was chosen")

    # Action-specific parameters (only one should be set based on action)
    hybrid_search: HybridSearchInput | None = Field(
        default=None,
        description="Parameters for hybrid_search action",
    )
    open_span: OpenSpanInput | None = Field(
        default=None,
        description="Parameters for open_span action",
    )
    summarize_evidence: SummarizeEvidenceInput | None = Field(
        default=None,
        description="Parameters for summarize_evidence action",
    )
    finalize: FinalizeInput | None = Field(
        default=None,
        description="Parameters for finalize action",
    )

    # LSP action parameters
    lsp_definition: LspPositionInput | None = Field(
        default=None,
        description="Parameters for lsp_definition action",
    )
    lsp_references: LspPositionInput | None = Field(
        default=None,
        description="Parameters for lsp_references action",
    )
    lsp_hover: LspPositionInput | None = Field(
        default=None,
        description="Parameters for lsp_hover action",
    )
    lsp_diagnostics: LspDiagnosticsInput | None = Field(
        default=None,
        description="Parameters for lsp_diagnostics action",
    )

    # Tree-sitter action parameters
    ts_outline: TsOutlineInput | None = Field(
        default=None,
        description="Parameters for ts_outline action",
    )
    ts_find_symbol: TsFindSymbolInput | None = Field(
        default=None,
        description="Parameters for ts_find_symbol action",
    )
    ts_enclosing_symbol: TsEnclosingSymbolInput | None = Field(
        default=None,
        description="Parameters for ts_enclosing_symbol action",
    )

    # Graph action parameters
    graph_expand: GraphExpandInput | None = Field(
        default=None,
        description="Parameters for graph_expand action",
    )
    graph_pack: GraphPackInput | None = Field(
        default=None,
        description="Parameters for graph_pack action",
    )
    graph_trace: GraphTraceInput | None = Field(
        default=None,
        description="Parameters for graph_trace action",
    )

    def get_action_input(self) -> BaseModel | None:
        """Get the input parameters for the selected action."""
        return getattr(self, self.action)


# =============================================================================
# ACTION OUTPUT SCHEMAS
# =============================================================================


class SearchResult(BaseModel):
    """A single search result from hybrid_search."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    provenance: Literal["bm25", "vector", "hybrid"]


class HybridSearchOutput(BaseModel):
    """Output from hybrid_search action."""

    results: list[SearchResult]
    total_found: int


class OpenSpanOutput(BaseModel):
    """Output from open_span action."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    evidence_id: str


class SummarizeEvidenceOutput(BaseModel):
    """Output from summarize_evidence action."""

    summary: str
    evidence_count: int
    key_files: list[str]


class FinalizeOutput(BaseModel):
    """Output from finalize action."""

    answer: str
    citations: list[str]  # Evidence IDs
    confidence: float


# =============================================================================
# LSP ACTION OUTPUT SCHEMAS
# =============================================================================


class LspLocation(BaseModel):
    """A location in source code from LSP."""

    file_path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int


class LspDefinitionOutput(BaseModel):
    """Output from lsp_definition action."""

    locations: list[LspLocation]
    symbol_name: str | None = None


class LspReferencesOutput(BaseModel):
    """Output from lsp_references action."""

    locations: list[LspLocation]
    total_references: int


class LspHoverOutput(BaseModel):
    """Output from lsp_hover action."""

    name: str
    kind: str  # function, class, method, variable, etc.
    signature: str | None
    documentation: str | None


class LspDiagnostic(BaseModel):
    """A single diagnostic from the language server."""

    file_path: str
    line: int
    column: int
    message: str
    severity: Literal["error", "warning", "info", "hint"]
    code: str | None = None


class LspDiagnosticsOutput(BaseModel):
    """Output from lsp_diagnostics action."""

    diagnostics: list[LspDiagnostic]
    files_checked: int
    error_count: int
    warning_count: int


# =============================================================================
# TREE-SITTER ACTION OUTPUT SCHEMAS
# =============================================================================


class TsSymbol(BaseModel):
    """A code symbol from Tree-sitter."""

    name: str
    kind: str  # function, class, method, etc.
    file_path: str
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    signature: str | None = None
    parent: str | None = None


class TsOutlineOutput(BaseModel):
    """Output from ts_outline action."""

    symbols: list[TsSymbol]
    file_path: str


class TsFindSymbolOutput(BaseModel):
    """Output from ts_find_symbol action."""

    symbol: TsSymbol | None
    found: bool


class TsEnclosingSymbolOutput(BaseModel):
    """Output from ts_enclosing_symbol action."""

    symbol: TsSymbol | None
    found: bool
    line: int


# =============================================================================
# GRAPH ACTION OUTPUT SCHEMAS
# =============================================================================


class GraphNode(BaseModel):
    """A node in the code graph."""

    id: str  # Qualified ID: file_path::symbol_name
    name: str
    kind: str  # function, class, method, etc.
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    parent_id: str | None = None


class GraphEdge(BaseModel):
    """An edge in the code graph."""

    source_id: str
    target_id: str
    edge_type: str  # contains, defines, references, calls, imports, inherits
    weight: float = 1.0


class GraphExpandOutput(BaseModel):
    """Output from graph_expand action."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    seeds_found: int
    total_expanded: int


class GraphPackedNode(BaseModel):
    """A node selected by graph_pack with reason."""

    node: GraphNode
    reason: str
    score: float


class GraphPackOutput(BaseModel):
    """Output from graph_pack action."""

    selected: list[GraphPackedNode]
    total_considered: int


class GraphPathStep(BaseModel):
    """A step in a path between symbols."""

    node: GraphNode
    edge_type: str | None = None  # None for starting node
    direction: str  # forward or backward


class GraphTraceOutput(BaseModel):
    """Output from graph_trace action."""

    paths: list[list[GraphPathStep]]
    found: bool
    shortest_length: int | None = None


# =============================================================================
# AGENT STATE
# =============================================================================


class AgentState(BaseModel):
    """Current state of the research agent.

    This is passed to the LLM to help it decide the next action.
    """

    question: str = Field(description="The research question being investigated")
    scope: str | None = Field(default=None, description="Path pattern limiting search")
    steps_taken: int = Field(default=0, description="Number of steps completed")
    steps_remaining: int = Field(description="Number of steps left in budget")
    evidence_count: int = Field(default=0, description="Number of evidence items collected")
    evidence_summary: str = Field(
        default="",
        description="Brief summary of evidence collected so far",
    )
    last_action: str | None = Field(default=None, description="The last action taken")
    last_result_summary: str | None = Field(
        default=None,
        description="Summary of the last action's result",
    )
