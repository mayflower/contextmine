"""Research agent infrastructure for deep code investigation."""

from typing import Any

from contextmine_core.research.agent import (
    AgentConfig,
    ResearchAgent,
    run_research,
)
from contextmine_core.research.artifacts import (
    ArtifactStore,
    FileArtifactStore,
    MemoryArtifactStore,
    get_artifact_store,
    reset_artifact_store,
)
from contextmine_core.research.eval import (
    EvalDataset,
    EvalMetrics,
    EvalQuestion,
    EvalRun,
    EvalRunner,
    QuestionResult,
    calculate_metrics,
)
from contextmine_core.research.run import (
    ActionStep,
    Evidence,
    ResearchRun,
    RunStatus,
)
from contextmine_core.research.verification import (
    AnswerVerifier,
    CitationVerification,
    ConfidenceCalibration,
    EvidenceSupportScore,
    VerificationResult,
    VerificationStatus,
)


def format_answer_with_citations(
    answer: str,
    citations: list[dict[str, Any]],
    max_length: int = 800,
) -> str:
    """Format answer with citations for MCP tool output.

    Args:
        answer: The answer text
        citations: List of citation dicts with 'id', 'file', 'lines', 'provenance'
        max_length: Maximum length for the answer portion

    Returns:
        Formatted string with answer and citations
    """
    # Truncate answer if too long
    if len(answer) > max_length:
        answer = answer[:max_length] + "..."

    parts = [answer, "", "**Citations:**"]

    for c in citations[:10]:  # Limit to 10 citations in output
        parts.append(f"- [{c['id']}] {c['file']}:{c['lines']} ({c['provenance']})")

    if len(citations) > 10:
        parts.append(f"  ... and {len(citations) - 10} more")

    return "\n".join(parts)


__all__ = [
    # Agent
    "AgentConfig",
    "ResearchAgent",
    "run_research",
    "format_answer_with_citations",
    # Artifacts
    "ArtifactStore",
    "FileArtifactStore",
    "MemoryArtifactStore",
    "get_artifact_store",
    "reset_artifact_store",
    # Run
    "ActionStep",
    "Evidence",
    "ResearchRun",
    "RunStatus",
    # Verification
    "AnswerVerifier",
    "CitationVerification",
    "ConfidenceCalibration",
    "EvidenceSupportScore",
    "VerificationResult",
    "VerificationStatus",
    # Evaluation
    "EvalDataset",
    "EvalMetrics",
    "EvalQuestion",
    "EvalRun",
    "EvalRunner",
    "QuestionResult",
    "calculate_metrics",
]
