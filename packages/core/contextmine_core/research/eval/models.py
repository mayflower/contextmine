"""Evaluation models for testing research agent quality."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from contextmine_core.research.run import ResearchRun
    from contextmine_core.research.verification import VerificationResult


@dataclass
class EvalQuestion:
    """A single evaluation question.

    Represents a test case for the research agent with optional
    expected answers and evidence files for scoring.
    """

    id: str
    """Unique identifier for the question."""

    question: str
    """The research question to ask."""

    expected_answer: str | None = None
    """Optional expected answer for accuracy scoring."""

    expected_evidence_files: list[str] | None = None
    """Optional list of files that should be in evidence."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorizing questions."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_evidence_files": self.expected_evidence_files,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalQuestion:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            expected_answer=data.get("expected_answer"),
            expected_evidence_files=data.get("expected_evidence_files"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvalDataset:
    """A collection of evaluation questions.

    Datasets can be loaded from YAML files or created programmatically.
    """

    id: str
    """Unique identifier for the dataset."""

    name: str
    """Human-readable name."""

    questions: list[EvalQuestion]
    """List of questions in the dataset."""

    description: str = ""
    """Optional description of the dataset."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalDataset:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            questions=[EvalQuestion.from_dict(q) for q in data.get("questions", [])],
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalDataset:
        """Load dataset from YAML file.

        Expected YAML format:
        ```yaml
        id: my-dataset
        name: My Test Dataset
        description: Test questions for validation
        questions:
          - id: q001
            question: "What is the main entry point?"
            expected_evidence_files:
              - "src/main.py"
            tags: ["entry-point"]
        ```

        Args:
            path: Path to YAML file

        Returns:
            Loaded dataset
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def filter_by_tags(self, tags: list[str]) -> EvalDataset:
        """Create filtered dataset with questions matching any tag.

        Args:
            tags: Tags to filter by

        Returns:
            New dataset with filtered questions
        """
        tag_set = set(tags)
        filtered = [q for q in self.questions if tag_set & set(q.tags)]
        return EvalDataset(
            id=f"{self.id}-filtered",
            name=f"{self.name} (filtered)",
            description=f"Filtered by tags: {', '.join(tags)}",
            questions=filtered,
        )


@dataclass
class QuestionResult:
    """Result of running a single evaluation question.

    Contains the research run, verification result, and metadata
    about the execution.
    """

    question: EvalQuestion
    """The question that was asked."""

    run: ResearchRun
    """The completed research run."""

    verification: VerificationResult
    """Verification result for the run."""

    duration_seconds: float
    """Time taken to run the question."""

    success: bool
    """Whether the run completed without error."""

    error: str | None = None
    """Error message if run failed."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""

        return {
            "question": self.question.to_dict(),
            "run_id": self.run.run_id,
            "run_status": self.run.status.value,
            "verification": self.verification.to_dict(),
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
        }

    @property
    def evidence_file_recall(self) -> float | None:
        """Calculate recall of expected evidence files.

        Returns:
            Recall (0.0-1.0) or None if no expected files
        """
        if not self.question.expected_evidence_files:
            return None

        expected = set(self.question.expected_evidence_files)
        found = {e.file_path for e in self.run.evidence}

        # Check partial matches (expected might be suffix)
        matched = 0
        for exp in expected:
            for f in found:
                if f.endswith(exp) or exp.endswith(f):
                    matched += 1
                    break

        return matched / len(expected) if expected else 1.0


@dataclass
class EvalRun:
    """A complete evaluation run across a dataset.

    Tracks all question results and metadata about the evaluation.
    """

    id: str
    """Unique identifier for the eval run."""

    dataset_id: str
    """ID of the dataset being evaluated."""

    results: list[QuestionResult]
    """Results for each question."""

    started_at: str
    """ISO timestamp of when evaluation started."""

    completed_at: str | None = None
    """ISO timestamp of when evaluation completed."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (model, settings, etc.)."""

    @classmethod
    def create(cls, dataset_id: str, metadata: dict[str, Any] | None = None) -> EvalRun:
        """Create a new evaluation run.

        Args:
            dataset_id: ID of the dataset
            metadata: Optional metadata

        Returns:
            New EvalRun instance
        """
        return cls(
            id=str(uuid.uuid4()),
            dataset_id=dataset_id,
            results=[],
            started_at=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )

    def add_result(self, result: QuestionResult) -> None:
        """Add a question result.

        Args:
            result: The result to add
        """
        self.results.append(result)

    def complete(self) -> None:
        """Mark the evaluation as complete."""
        self.completed_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }
