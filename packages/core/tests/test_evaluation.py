"""Tests for the evaluation module."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from contextmine_core.research.eval import (
    EvalDataset,
    EvalMetrics,
    EvalQuestion,
    EvalRun,
    EvalRunner,
    QuestionResult,
    calculate_metrics,
)
from contextmine_core.research.run import Evidence, ResearchRun, RunStatus
from contextmine_core.research.verification import (
    ConfidenceCalibration,
    EvidenceSupportScore,
    VerificationResult,
    VerificationStatus,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_questions() -> list[EvalQuestion]:
    """Create sample evaluation questions."""
    return [
        EvalQuestion(
            id="q001",
            question="What is the main entry point?",
            expected_evidence_files=["src/main.py"],
            tags=["entry-point", "basic"],
        ),
        EvalQuestion(
            id="q002",
            question="How does authentication work?",
            expected_evidence_files=["src/auth.py", "src/utils.py"],
            tags=["auth", "security"],
        ),
    ]


@pytest.fixture
def sample_dataset(sample_questions: list[EvalQuestion]) -> EvalDataset:
    """Create a sample dataset."""
    return EvalDataset(
        id="test-dataset",
        name="Test Dataset",
        description="A test dataset for evaluation",
        questions=sample_questions,
    )


@pytest.fixture
def sample_verification() -> VerificationResult:
    """Create a sample verification result."""
    return VerificationResult(
        status=VerificationStatus.PASSED,
        citations=[],
        evidence_support=EvidenceSupportScore(
            score=0.8,
            reasoning="Good evidence",
            supporting_evidence_ids=["ev-001"],
        ),
        semantic_grounding=None,
        confidence_calibration=ConfidenceCalibration(
            stated_confidence=0.8,
            evidence_confidence=0.75,
            calibration_delta=0.05,
            is_calibrated=True,
        ),
        issues=[],
        verified_at=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_run() -> ResearchRun:
    """Create a sample research run."""
    run = ResearchRun(
        run_id="test-run-001",
        question="Test question",
        status=RunStatus.DONE,
        created_at=datetime.now(UTC),
        answer="Test answer",
        evidence=[
            Evidence(
                id="ev-001",
                file_path="src/main.py",
                start_line=1,
                end_line=10,
                content="def main():",
                reason="Entry point",
                provenance="search",
                score=0.9,
            )
        ],
    )
    run.complete("Test answer")
    return run


# =============================================================================
# EVAL QUESTION TESTS
# =============================================================================


class TestEvalQuestion:
    """Tests for EvalQuestion dataclass."""

    def test_to_dict(self, sample_questions: list[EvalQuestion]):
        """Test serialization to dict."""
        q = sample_questions[0]
        d = q.to_dict()

        assert d["id"] == "q001"
        assert d["question"] == "What is the main entry point?"
        assert d["expected_evidence_files"] == ["src/main.py"]
        assert "entry-point" in d["tags"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "id": "q999",
            "question": "Test question?",
            "expected_answer": "Test answer",
            "expected_evidence_files": ["file.py"],
            "tags": ["test"],
            "metadata": {"custom": "data"},
        }
        q = EvalQuestion.from_dict(d)

        assert q.id == "q999"
        assert q.question == "Test question?"
        assert q.expected_answer == "Test answer"
        assert q.expected_evidence_files == ["file.py"]
        assert q.metadata == {"custom": "data"}


# =============================================================================
# EVAL DATASET TESTS
# =============================================================================


class TestEvalDataset:
    """Tests for EvalDataset class."""

    def test_to_dict(self, sample_dataset: EvalDataset):
        """Test serialization to dict."""
        d = sample_dataset.to_dict()

        assert d["id"] == "test-dataset"
        assert d["name"] == "Test Dataset"
        assert len(d["questions"]) == 2

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "id": "new-dataset",
            "name": "New Dataset",
            "description": "A new dataset",
            "questions": [
                {"id": "q1", "question": "Question 1?"},
                {"id": "q2", "question": "Question 2?"},
            ],
        }
        dataset = EvalDataset.from_dict(d)

        assert dataset.id == "new-dataset"
        assert len(dataset.questions) == 2
        assert dataset.questions[0].id == "q1"

    def test_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
id: yaml-dataset
name: YAML Dataset
description: Loaded from YAML
questions:
  - id: y1
    question: "YAML question 1?"
    tags:
      - yaml
      - test
  - id: y2
    question: "YAML question 2?"
    expected_evidence_files:
      - src/file.py
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            dataset = EvalDataset.from_yaml(f.name)

            assert dataset.id == "yaml-dataset"
            assert dataset.name == "YAML Dataset"
            assert len(dataset.questions) == 2
            assert "yaml" in dataset.questions[0].tags

            # Cleanup
            Path(f.name).unlink()

    def test_filter_by_tags(self, sample_dataset: EvalDataset):
        """Test filtering by tags."""
        filtered = sample_dataset.filter_by_tags(["auth"])

        assert len(filtered.questions) == 1
        assert filtered.questions[0].id == "q002"

    def test_filter_by_multiple_tags(self, sample_dataset: EvalDataset):
        """Test filtering by multiple tags (OR)."""
        filtered = sample_dataset.filter_by_tags(["basic", "security"])

        assert len(filtered.questions) == 2


# =============================================================================
# QUESTION RESULT TESTS
# =============================================================================


class TestQuestionResult:
    """Tests for QuestionResult dataclass."""

    def test_to_dict(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test serialization to dict."""
        result = QuestionResult(
            question=sample_questions[0],
            run=sample_run,
            verification=sample_verification,
            duration_seconds=5.5,
            success=True,
            error=None,
        )

        d = result.to_dict()

        assert d["question"]["id"] == "q001"
        assert d["run_id"] == "test-run-001"
        assert d["verification"]["status"] == "passed"
        assert d["duration_seconds"] == 5.5
        assert d["success"] is True

    def test_evidence_file_recall_full_match(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test evidence file recall with full match."""
        result = QuestionResult(
            question=sample_questions[0],  # expects ["src/main.py"]
            run=sample_run,  # has evidence from "src/main.py"
            verification=sample_verification,
            duration_seconds=1.0,
            success=True,
        )

        assert result.evidence_file_recall == 1.0

    def test_evidence_file_recall_partial_match(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test evidence file recall with partial match."""
        result = QuestionResult(
            question=sample_questions[1],  # expects ["src/auth.py", "src/utils.py"]
            run=sample_run,  # only has "src/main.py"
            verification=sample_verification,
            duration_seconds=1.0,
            success=True,
        )

        assert result.evidence_file_recall == 0.0

    def test_evidence_file_recall_no_expected(
        self,
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test evidence file recall when no expected files."""
        question = EvalQuestion(
            id="q-no-expected",
            question="Generic question",
        )
        result = QuestionResult(
            question=question,
            run=sample_run,
            verification=sample_verification,
            duration_seconds=1.0,
            success=True,
        )

        assert result.evidence_file_recall is None


# =============================================================================
# EVAL RUN TESTS
# =============================================================================


class TestEvalRun:
    """Tests for EvalRun dataclass."""

    def test_create(self):
        """Test creating a new eval run."""
        run = EvalRun.create(
            dataset_id="test-dataset",
            metadata={"model": "test-model"},
        )

        assert run.id  # Should have a UUID
        assert run.dataset_id == "test-dataset"
        assert run.started_at
        assert run.completed_at is None
        assert run.metadata["model"] == "test-model"

    def test_add_result(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test adding results to eval run."""
        eval_run = EvalRun.create(dataset_id="test")

        result = QuestionResult(
            question=sample_questions[0],
            run=sample_run,
            verification=sample_verification,
            duration_seconds=1.0,
            success=True,
        )

        eval_run.add_result(result)

        assert len(eval_run.results) == 1
        assert eval_run.results[0].question.id == "q001"

    def test_complete(self):
        """Test completing an eval run."""
        eval_run = EvalRun.create(dataset_id="test")
        assert eval_run.completed_at is None

        eval_run.complete()

        assert eval_run.completed_at is not None

    def test_to_dict(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test serialization to dict."""
        eval_run = EvalRun.create(dataset_id="test")
        result = QuestionResult(
            question=sample_questions[0],
            run=sample_run,
            verification=sample_verification,
            duration_seconds=1.0,
            success=True,
        )
        eval_run.add_result(result)
        eval_run.complete()

        d = eval_run.to_dict()

        assert d["dataset_id"] == "test"
        assert d["started_at"]
        assert d["completed_at"]
        assert len(d["results"]) == 1


# =============================================================================
# EVAL METRICS TESTS
# =============================================================================


class TestEvalMetrics:
    """Tests for EvalMetrics and calculate_metrics."""

    def test_calculate_metrics_empty(self):
        """Test metrics calculation with no results."""
        metrics = calculate_metrics([])

        assert metrics.success_rate == 0.0
        assert metrics.total_questions == 0
        assert metrics.successful_questions == 0

    def test_calculate_metrics_all_success(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test metrics with all successful results."""
        results = [
            QuestionResult(
                question=q,
                run=sample_run,
                verification=sample_verification,
                duration_seconds=2.0,
                success=True,
            )
            for q in sample_questions
        ]

        metrics = calculate_metrics(results)

        assert metrics.success_rate == 1.0
        assert metrics.total_questions == 2
        assert metrics.successful_questions == 2
        assert metrics.avg_duration_seconds == 2.0

    def test_calculate_metrics_partial_success(
        self,
        sample_questions: list[EvalQuestion],
        sample_run: ResearchRun,
        sample_verification: VerificationResult,
    ):
        """Test metrics with partial success."""
        results = [
            QuestionResult(
                question=sample_questions[0],
                run=sample_run,
                verification=sample_verification,
                duration_seconds=1.0,
                success=True,
            ),
            QuestionResult(
                question=sample_questions[1],
                run=sample_run,
                verification=sample_verification,
                duration_seconds=1.0,
                success=False,
                error="Test error",
            ),
        ]

        metrics = calculate_metrics(results)

        assert metrics.success_rate == 0.5
        assert metrics.total_questions == 2
        assert metrics.successful_questions == 1

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = EvalMetrics(
            success_rate=0.8,
            total_questions=10,
            successful_questions=8,
            avg_confidence=0.75,
            avg_evidence_count=3.5,
            avg_action_count=5.0,
            avg_duration_seconds=2.5,
            citation_validity_rate=0.9,
            calibration_score=0.85,
            verification_pass_rate=0.7,
            avg_evidence_recall=0.6,
        )

        d = metrics.to_dict()

        assert d["success_rate"] == 0.8
        assert d["total_questions"] == 10
        assert d["avg_evidence_recall"] == 0.6

    def test_metrics_to_report_markdown(self):
        """Test markdown report generation."""
        metrics = EvalMetrics(
            success_rate=0.8,
            total_questions=10,
            successful_questions=8,
            avg_confidence=0.75,
            avg_evidence_count=3.5,
            avg_action_count=5.0,
            avg_duration_seconds=2.5,
            citation_validity_rate=0.9,
            calibration_score=0.85,
            verification_pass_rate=0.7,
            avg_evidence_recall=0.6,
        )

        report = metrics.to_report_markdown()

        assert "# Evaluation Metrics" in report
        assert "80.0%" in report  # success rate
        assert "10" in report  # total questions


# =============================================================================
# EVAL RUNNER TESTS
# =============================================================================


class TestEvalRunner:
    """Tests for EvalRunner class."""

    @pytest.mark.anyio
    async def test_run_single(self, sample_run: ResearchRun):
        """Test running a single ad-hoc question."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(return_value=sample_run)

        runner = EvalRunner(agent=mock_agent)
        result = await runner.run_single("Test question?")

        assert result.question.id == "adhoc"
        assert result.question.question == "Test question?"
        assert result.success is True
        mock_agent.research.assert_called_once()

    @pytest.mark.anyio
    async def test_run_dataset_serial(self, sample_dataset: EvalDataset, sample_run: ResearchRun):
        """Test running a dataset serially."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(return_value=sample_run)

        runner = EvalRunner(agent=mock_agent)
        eval_run = await runner.run_dataset(sample_dataset, max_parallel=1)

        assert eval_run.dataset_id == "test-dataset"
        assert len(eval_run.results) == 2
        assert eval_run.completed_at is not None
        assert mock_agent.research.call_count == 2

    @pytest.mark.anyio
    async def test_run_dataset_parallel(self, sample_dataset: EvalDataset, sample_run: ResearchRun):
        """Test running a dataset in parallel."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(return_value=sample_run)

        runner = EvalRunner(agent=mock_agent)
        eval_run = await runner.run_dataset(sample_dataset, max_parallel=2)

        assert len(eval_run.results) == 2
        assert mock_agent.research.call_count == 2

    @pytest.mark.anyio
    async def test_run_handles_errors(self, sample_dataset: EvalDataset):
        """Test that runner handles errors gracefully."""
        # Mock agent that raises
        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(side_effect=ValueError("Test error"))

        runner = EvalRunner(agent=mock_agent)
        eval_run = await runner.run_dataset(sample_dataset, max_parallel=1)

        # Should still complete, but with failed results
        assert len(eval_run.results) == 2
        assert all(not r.success for r in eval_run.results)
        assert all("Test error" in r.error for r in eval_run.results)
