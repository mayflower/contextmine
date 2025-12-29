"""Action registry for the research agent.

The registry maintains available actions and their metadata for the agent loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextmine_core.research.run import Evidence, ResearchRun


@dataclass
class ActionResult:
    """Result from executing an action."""

    success: bool
    """Whether the action completed successfully."""

    output_summary: str
    """Brief summary of what happened (for trace)."""

    evidence: list[Evidence] = field(default_factory=list)
    """New evidence collected by this action."""

    data: dict[str, Any] = field(default_factory=dict)
    """Full action output data."""

    error: str | None = None
    """Error message if action failed."""

    should_stop: bool = False
    """Whether the agent should stop after this action (e.g., finalize)."""


class Action(ABC):
    """Base class for research agent actions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this action."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this action does."""
        ...

    @abstractmethod
    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute the action.

        Args:
            run: The current research run (for context and evidence access)
            params: Action parameters from ActionSelection

        Returns:
            ActionResult with output and any collected evidence
        """
        ...


class ActionRegistry:
    """Registry of available actions for the research agent."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        """Register an action."""
        self._actions[action.name] = action

    def get(self, name: str) -> Action | None:
        """Get an action by name."""
        return self._actions.get(name)

    def list_actions(self) -> list[str]:
        """List all registered action names."""
        return list(self._actions.keys())

    def get_action_descriptions(self) -> list[str]:
        """Get formatted descriptions of all actions for prompts."""
        descriptions = []
        for action in self._actions.values():
            descriptions.append(f"- **{action.name}**: {action.description}")
        return descriptions


# Global registry instance
_registry: ActionRegistry | None = None


def get_action_registry() -> ActionRegistry:
    """Get the global action registry, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = ActionRegistry()
        _register_default_actions(_registry)
    return _registry


def _register_default_actions(registry: ActionRegistry) -> None:
    """Register the default set of actions."""
    # Import here to avoid circular imports
    from contextmine_core.research.actions.finalize import FinalizeAction
    from contextmine_core.research.actions.open_span import OpenSpanAction
    from contextmine_core.research.actions.search import HybridSearchAction
    from contextmine_core.research.actions.summarize import SummarizeEvidenceAction

    registry.register(HybridSearchAction())
    registry.register(OpenSpanAction())
    registry.register(SummarizeEvidenceAction())
    registry.register(FinalizeAction())

    # Register LSP actions
    _register_lsp_actions(registry)

    # Register Tree-sitter actions
    _register_treesitter_actions(registry)

    # Register Graph actions
    _register_graph_actions(registry)


def _register_lsp_actions(registry: ActionRegistry) -> None:
    """Register LSP-based actions."""
    try:
        from contextmine_core.research.actions.lsp import (
            LspDefinitionAction,
            LspDiagnosticsAction,
            LspHoverAction,
            LspReferencesAction,
        )

        registry.register(LspDefinitionAction())
        registry.register(LspReferencesAction())
        registry.register(LspHoverAction())
        registry.register(LspDiagnosticsAction())

    except ImportError:
        # LSP module not available (optional dependency)
        pass


def _register_treesitter_actions(registry: ActionRegistry) -> None:
    """Register Tree-sitter based actions."""
    try:
        from contextmine_core.research.actions.treesitter import (
            TsEnclosingSymbolAction,
            TsFindSymbolAction,
            TsOutlineAction,
        )

        registry.register(TsOutlineAction())
        registry.register(TsFindSymbolAction())
        registry.register(TsEnclosingSymbolAction())

    except ImportError:
        # Tree-sitter module not available (optional dependency)
        pass


def _register_graph_actions(registry: ActionRegistry) -> None:
    """Register Graph-based actions."""
    try:
        from contextmine_core.research.actions.graph import (
            GraphExpandAction,
            GraphPackAction,
            GraphTraceAction,
        )

        registry.register(GraphExpandAction())
        registry.register(GraphPackAction())
        registry.register(GraphTraceAction())

    except ImportError:
        # Graph module not available (optional dependency)
        pass


def reset_action_registry() -> None:
    """Reset the global action registry. Used for testing."""
    global _registry
    _registry = None
