"""Research agent infrastructure for deep code investigation."""

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

__all__ = [
    # Agent
    "AgentConfig",
    "ResearchAgent",
    "run_research",
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
