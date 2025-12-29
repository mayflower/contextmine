"""Evaluation metrics for research agent quality assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextmine_core.research.eval.models import QuestionResult
from contextmine_core.research.verification import VerificationStatus


@dataclass
class EvalMetrics:
    """Aggregate metrics from an evaluation run.

    Computed from a list of QuestionResults to provide
    summary statistics about research agent performance.
    """

    # Completion metrics
    success_rate: float
    """Fraction of questions completed without error (0.0-1.0)."""

    total_questions: int
    """Total number of questions evaluated."""

    successful_questions: int
    """Number of questions completed without error."""

    # Quality metrics
    avg_confidence: float
    """Average stated confidence across runs."""

    avg_evidence_count: float
    """Average number of evidence items collected."""

    avg_action_count: float
    """Average number of actions taken per question."""

    avg_duration_seconds: float
    """Average time to answer a question."""

    # Verification metrics
    citation_validity_rate: float
    """Fraction of citations that reference valid evidence."""

    calibration_score: float
    """Average calibration score (1 - calibration_delta)."""

    verification_pass_rate: float
    """Fraction of runs passing verification."""

    # Evidence recall (if expected files provided)
    avg_evidence_recall: float | None
    """Average recall of expected evidence files."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success_rate": self.success_rate,
            "total_questions": self.total_questions,
            "successful_questions": self.successful_questions,
            "avg_confidence": self.avg_confidence,
            "avg_evidence_count": self.avg_evidence_count,
            "avg_action_count": self.avg_action_count,
            "avg_duration_seconds": self.avg_duration_seconds,
            "citation_validity_rate": self.citation_validity_rate,
            "calibration_score": self.calibration_score,
            "verification_pass_rate": self.verification_pass_rate,
            "avg_evidence_recall": self.avg_evidence_recall,
        }

    def to_report_markdown(self) -> str:
        """Generate markdown report of metrics."""
        lines = [
            "# Evaluation Metrics",
            "",
            "## Completion",
            f"- **Success Rate:** {self.success_rate:.1%}",
            f"- **Total Questions:** {self.total_questions}",
            f"- **Successful:** {self.successful_questions}",
            "",
            "## Quality",
            f"- **Avg Confidence:** {self.avg_confidence:.2f}",
            f"- **Avg Evidence Count:** {self.avg_evidence_count:.1f}",
            f"- **Avg Actions:** {self.avg_action_count:.1f}",
            f"- **Avg Duration:** {self.avg_duration_seconds:.1f}s",
            "",
            "## Verification",
            f"- **Citation Validity:** {self.citation_validity_rate:.1%}",
            f"- **Calibration Score:** {self.calibration_score:.2f}",
            f"- **Verification Pass Rate:** {self.verification_pass_rate:.1%}",
        ]

        if self.avg_evidence_recall is not None:
            lines.extend(
                [
                    "",
                    "## Evidence Recall",
                    f"- **Avg Recall:** {self.avg_evidence_recall:.1%}",
                ]
            )

        return "\n".join(lines)


def calculate_metrics(results: list[QuestionResult]) -> EvalMetrics:
    """Calculate aggregate metrics from question results.

    Args:
        results: List of question results from an eval run

    Returns:
        Computed metrics
    """
    if not results:
        return EvalMetrics(
            success_rate=0.0,
            total_questions=0,
            successful_questions=0,
            avg_confidence=0.0,
            avg_evidence_count=0.0,
            avg_action_count=0.0,
            avg_duration_seconds=0.0,
            citation_validity_rate=0.0,
            calibration_score=0.0,
            verification_pass_rate=0.0,
            avg_evidence_recall=None,
        )

    total = len(results)
    successful = [r for r in results if r.success]
    num_successful = len(successful)

    # Success rate
    success_rate = num_successful / total if total > 0 else 0.0

    # Calculate averages from successful runs only
    if successful:
        # Confidence
        confidences = [r.verification.confidence_calibration.stated_confidence for r in successful]
        avg_confidence = sum(confidences) / len(confidences)

        # Evidence count
        evidence_counts = [len(r.run.evidence) for r in successful]
        avg_evidence_count = sum(evidence_counts) / len(evidence_counts)

        # Action count
        action_counts = [len(r.run.steps) for r in successful]
        avg_action_count = sum(action_counts) / len(action_counts)

        # Duration
        durations = [r.duration_seconds for r in successful]
        avg_duration = sum(durations) / len(durations)

        # Citation validity
        citation_validities = [r.verification.citation_validity_rate for r in successful]
        citation_validity_rate = sum(citation_validities) / len(citation_validities)

        # Calibration score (1 - delta, clamped to [0, 1])
        calibration_scores = [
            max(0, 1 - r.verification.confidence_calibration.calibration_delta) for r in successful
        ]
        calibration_score = sum(calibration_scores) / len(calibration_scores)

        # Verification pass rate
        passed = sum(1 for r in successful if r.verification.status == VerificationStatus.PASSED)
        verification_pass_rate = passed / len(successful)

        # Evidence recall (only for questions with expected files)
        recalls = [r.evidence_file_recall for r in successful if r.evidence_file_recall is not None]
        avg_evidence_recall = sum(recalls, 0.0) / len(recalls) if recalls else None
    else:
        # No successful runs
        avg_confidence = 0.0
        avg_evidence_count = 0.0
        avg_action_count = 0.0
        avg_duration = 0.0
        citation_validity_rate = 0.0
        calibration_score = 0.0
        verification_pass_rate = 0.0
        avg_evidence_recall = None

    return EvalMetrics(
        success_rate=success_rate,
        total_questions=total,
        successful_questions=num_successful,
        avg_confidence=avg_confidence,
        avg_evidence_count=avg_evidence_count,
        avg_action_count=avg_action_count,
        avg_duration_seconds=avg_duration,
        citation_validity_rate=citation_validity_rate,
        calibration_score=calibration_score,
        verification_pass_rate=verification_pass_rate,
        avg_evidence_recall=avg_evidence_recall,
    )
