"""Evaluation runner for testing research agent quality."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import anyio
from contextmine_core.research.eval.models import (
    EvalDataset,
    EvalQuestion,
    EvalRun,
    QuestionResult,
)

if TYPE_CHECKING:
    from contextmine_core.research.agent import ResearchAgent
    from contextmine_core.research.verification import AnswerVerifier

logger = logging.getLogger(__name__)


class EvalRunner:
    """Runner for evaluating research agent on test datasets.

    Executes questions from a dataset, verifies results, and
    collects metrics about agent performance.
    """

    def __init__(
        self,
        agent: ResearchAgent,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        """Initialize the eval runner.

        Args:
            agent: Research agent to evaluate
            verifier: Optional verifier (creates default if None)
        """
        self._agent = agent
        if verifier is None:
            from contextmine_core.research.verification import AnswerVerifier

            verifier = AnswerVerifier()
        self._verifier = verifier

    async def run_dataset(
        self,
        dataset: EvalDataset,
        max_parallel: int = 1,
    ) -> EvalRun:
        """Run all questions in a dataset.

        Args:
            dataset: The dataset to evaluate
            max_parallel: Maximum concurrent questions (default: 1 = serial)

        Returns:
            Completed evaluation run with all results
        """
        eval_run = EvalRun.create(
            dataset_id=dataset.id,
            metadata={
                "dataset_name": dataset.name,
                "question_count": len(dataset.questions),
                "max_parallel": max_parallel,
            },
        )

        logger.info(
            "Starting eval run %s on dataset '%s' (%d questions)",
            eval_run.id[:8],
            dataset.name,
            len(dataset.questions),
        )

        if max_parallel <= 1:
            # Serial execution
            for i, question in enumerate(dataset.questions):
                logger.info(
                    "Running question %d/%d: %s",
                    i + 1,
                    len(dataset.questions),
                    question.id,
                )
                result = await self._run_question(question)
                eval_run.add_result(result)
        else:
            # Parallel execution with semaphore using anyio
            results: list[QuestionResult] = []
            semaphore = anyio.Semaphore(max_parallel)

            async def run_with_semaphore(q: EvalQuestion) -> QuestionResult:
                async with semaphore:
                    return await self._run_question(q)

            async with anyio.create_task_group() as tg:

                async def run_and_collect(q: EvalQuestion) -> None:
                    result = await run_with_semaphore(q)
                    results.append(result)

                for question in dataset.questions:
                    tg.start_soon(run_and_collect, question)

            for result in results:
                eval_run.add_result(result)

        eval_run.complete()

        logger.info(
            "Eval run %s completed: %d/%d successful",
            eval_run.id[:8],
            sum(1 for r in eval_run.results if r.success),
            len(eval_run.results),
        )

        return eval_run

    async def _run_question(self, question: EvalQuestion) -> QuestionResult:
        """Run a single evaluation question.

        Args:
            question: The question to run

        Returns:
            Result with run, verification, and timing
        """
        start_time = time.time()
        error: str | None = None
        success = True

        try:
            # Run the research agent
            run = await self._agent.research(
                question=question.question,
                scope=None,  # Could add scope to EvalQuestion if needed
            )

            # Verify the result
            verification = self._verifier.verify(run)

        except Exception as e:
            logger.exception("Failed to run question %s: %s", question.id, e)

            # Create a failed run
            from contextmine_core.research.run import ResearchRun
            from contextmine_core.research.verification import (
                ConfidenceCalibration,
                EvidenceSupportScore,
                VerificationResult,
                VerificationStatus,
            )

            run = ResearchRun.create(
                question=question.question,
                budget_steps=1,
            )
            run.fail(str(e))

            verification = VerificationResult(
                status=VerificationStatus.FAILED,
                citations=[],
                evidence_support=EvidenceSupportScore(
                    score=0.0,
                    reasoning="Run failed with exception",
                    supporting_evidence_ids=[],
                ),
                semantic_grounding=None,
                confidence_calibration=ConfidenceCalibration(
                    stated_confidence=0.0,
                    evidence_confidence=0.0,
                    calibration_delta=0.0,
                    is_calibrated=True,
                ),
                issues=[f"Exception: {e}"],
                verified_at="",
            )

            error = str(e)
            success = False

        duration = time.time() - start_time

        return QuestionResult(
            question=question,
            run=run,
            verification=verification,
            duration_seconds=duration,
            success=success,
            error=error,
        )

    async def run_single(self, question: str) -> QuestionResult:
        """Run a single ad-hoc question.

        Convenience method for testing individual questions.

        Args:
            question: The question text

        Returns:
            Question result
        """
        eval_question = EvalQuestion(
            id="adhoc",
            question=question,
        )
        return await self._run_question(eval_question)
