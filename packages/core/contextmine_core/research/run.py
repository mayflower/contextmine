"""ResearchRun: Core data structures for research agent runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextmine_core.research.verification import VerificationResult


class RunStatus(str, Enum):
    """Status of a research run."""

    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Evidence:
    """A piece of evidence collected during research.

    Evidence items are code snippets or documentation spans that
    support the answer to the research question.
    """

    id: str
    """Unique identifier for this evidence item."""

    file_path: str
    """Path to the file containing this evidence."""

    start_line: int
    """Starting line number (1-indexed)."""

    end_line: int
    """Ending line number (1-indexed, inclusive)."""

    content: str
    """The actual text content of the evidence."""

    reason: str
    """Why this evidence was selected."""

    provenance: str
    """How this evidence was found: 'bm25', 'vector', 'lsp', 'graph', 'manual'."""

    score: float | None = None
    """Relevance score if available."""

    symbol_id: str | None = None
    """Symbol identifier if LSP-derived (e.g., 'MyClass.my_method')."""

    symbol_kind: str | None = None
    """Symbol kind if available (e.g., 'function', 'class', 'method')."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "reason": self.reason,
            "provenance": self.provenance,
            "score": self.score,
            "symbol_id": self.symbol_id,
            "symbol_kind": self.symbol_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        """Create Evidence from dictionary."""
        return cls(
            id=data["id"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            content=data["content"],
            reason=data["reason"],
            provenance=data["provenance"],
            score=data.get("score"),
            symbol_id=data.get("symbol_id"),
            symbol_kind=data.get("symbol_kind"),
        )


@dataclass
class ActionStep:
    """A single step in the research agent's execution trace."""

    step_number: int
    """Sequential step number (1-indexed)."""

    action: str
    """Name of the action invoked (e.g., 'hybrid_search', 'lsp_definition')."""

    input: dict[str, Any]
    """Input parameters for the action."""

    output_summary: str
    """Brief summary of the action output."""

    duration_ms: int
    """Time taken to execute this step in milliseconds."""

    error: str | None = None
    """Error message if the step failed."""

    evidence_ids: list[str] = field(default_factory=list)
    """IDs of evidence items collected in this step."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "input": self.input,
            "output_summary": self.output_summary,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "evidence_ids": self.evidence_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionStep:
        """Create ActionStep from dictionary."""
        return cls(
            step_number=data["step_number"],
            action=data["action"],
            input=data["input"],
            output_summary=data["output_summary"],
            duration_ms=data["duration_ms"],
            error=data.get("error"),
            evidence_ids=data.get("evidence_ids", []),
        )


@dataclass
class ResearchRun:
    """A complete research run with trace and evidence.

    This is the core data structure for tracking a research agent's
    investigation of a codebase question.
    """

    run_id: str
    """Unique identifier for this run (ULID or UUID)."""

    question: str
    """The research question being investigated."""

    status: RunStatus
    """Current status of the run."""

    created_at: datetime
    """When the run was started."""

    steps: list[ActionStep] = field(default_factory=list)
    """Trace of actions taken during the run."""

    evidence: list[Evidence] = field(default_factory=list)
    """Evidence collected during the run."""

    answer: str | None = None
    """Final answer if the run completed successfully."""

    error_message: str | None = None
    """Error message if the run failed."""

    completed_at: datetime | None = None
    """When the run finished (success or error)."""

    scope: str | None = None
    """Path pattern limiting the search scope."""

    budget_steps: int = 10
    """Maximum number of steps allowed."""

    budget_used: int = 0
    """Number of steps actually used."""

    total_duration_ms: int = 0
    """Total time taken for the run in milliseconds."""

    verification: VerificationResult | None = None
    """Verification result if the run has been verified."""

    @classmethod
    def create(
        cls,
        question: str,
        scope: str | None = None,
        budget_steps: int = 10,
    ) -> ResearchRun:
        """Create a new research run."""
        return cls(
            run_id=str(uuid.uuid4()),
            question=question,
            status=RunStatus.RUNNING,
            created_at=datetime.now(UTC),
            scope=scope,
            budget_steps=budget_steps,
        )

    def add_step(self, step: ActionStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)
        self.budget_used = len(self.steps)
        self.total_duration_ms += step.duration_ms

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the collection."""
        self.evidence.append(evidence)

    def complete(self, answer: str) -> None:
        """Mark the run as successfully completed."""
        self.status = RunStatus.DONE
        self.answer = answer
        self.completed_at = datetime.now(UTC)

    def fail(self, error_message: str) -> None:
        """Mark the run as failed."""
        self.status = RunStatus.ERROR
        self.error_message = error_message
        self.completed_at = datetime.now(UTC)

    def get_evidence_by_id(self, evidence_id: str) -> Evidence | None:
        """Get evidence item by ID."""
        for e in self.evidence:
            if e.id == evidence_id:
                return e
        return None

    def to_trace_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for JSON serialization."""
        result = {
            "run_id": self.run_id,
            "question": self.question,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "scope": self.scope,
            "budget_steps": self.budget_steps,
            "budget_used": self.budget_used,
            "total_duration_ms": self.total_duration_ms,
            "error_message": self.error_message,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.verification:
            result["verification"] = self.verification.to_dict()
        return result

    def to_evidence_dict(self) -> dict[str, Any]:
        """Convert evidence to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "question": self.question,
            "evidence_count": len(self.evidence),
            "evidence": [e.to_dict() for e in self.evidence],
        }

    def to_report_markdown(self) -> str:
        """Generate a markdown report for the run."""
        lines = [
            f"# Research Report: {self.run_id[:8]}",
            "",
            f"**Question:** {self.question}",
            f"**Status:** {self.status.value}",
            f"**Created:** {self.created_at.isoformat()}",
        ]

        if self.completed_at:
            lines.append(f"**Completed:** {self.completed_at.isoformat()}")

        if self.scope:
            lines.append(f"**Scope:** `{self.scope}`")

        lines.extend(
            [
                f"**Steps used:** {self.budget_used}/{self.budget_steps}",
                f"**Duration:** {self.total_duration_ms}ms",
                "",
            ]
        )

        if self.answer:
            lines.extend(
                [
                    "## Answer",
                    "",
                    self.answer,
                    "",
                ]
            )

        if self.error_message:
            lines.extend(
                [
                    "## Error",
                    "",
                    f"```\n{self.error_message}\n```",
                    "",
                ]
            )

        if self.evidence:
            lines.extend(
                [
                    "## Evidence",
                    "",
                ]
            )
            for e in self.evidence:
                lines.extend(
                    [
                        f"### [{e.id}] {e.file_path}:{e.start_line}-{e.end_line}",
                        "",
                        f"**Reason:** {e.reason}",
                        f"**Provenance:** {e.provenance}",
                        "",
                        "```",
                        e.content,
                        "```",
                        "",
                    ]
                )

        if self.steps:
            lines.extend(
                [
                    "## Trace",
                    "",
                ]
            )
            for step in self.steps:
                status = "ERROR" if step.error else "OK"
                lines.append(
                    f"{step.step_number}. **{step.action}** ({step.duration_ms}ms) [{status}]"
                )
                if step.error:
                    lines.append(f"   - Error: {step.error}")
                else:
                    lines.append(f"   - {step.output_summary}")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchRun:
        """Create ResearchRun from combined trace + evidence dictionaries."""
        run = cls(
            run_id=data["run_id"],
            question=data["question"],
            status=RunStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            scope=data.get("scope"),
            budget_steps=data.get("budget_steps", 10),
            budget_used=data.get("budget_used", 0),
            total_duration_ms=data.get("total_duration_ms", 0),
            error_message=data.get("error_message"),
            answer=data.get("answer"),
        )
        if data.get("completed_at"):
            run.completed_at = datetime.fromisoformat(data["completed_at"])
        run.steps = [ActionStep.from_dict(s) for s in data.get("steps", [])]
        run.evidence = [Evidence.from_dict(e) for e in data.get("evidence", [])]

        # Handle verification if present
        if data.get("verification"):
            from contextmine_core.research.verification import VerificationResult

            run.verification = VerificationResult.from_dict(data["verification"])

        return run
