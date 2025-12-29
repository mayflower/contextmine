"""Evaluation module for testing research agent quality."""

from contextmine_core.research.eval.metrics import EvalMetrics, calculate_metrics
from contextmine_core.research.eval.models import (
    EvalDataset,
    EvalQuestion,
    EvalRun,
    QuestionResult,
)
from contextmine_core.research.eval.runner import EvalRunner

__all__ = [
    # Models
    "EvalDataset",
    "EvalQuestion",
    "EvalRun",
    "QuestionResult",
    # Metrics
    "EvalMetrics",
    "calculate_metrics",
    # Runner
    "EvalRunner",
]
