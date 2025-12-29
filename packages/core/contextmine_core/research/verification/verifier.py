"""Answer verifier for validating research run results."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from contextmine_core.research.verification.models import (
    CitationVerification,
    ConfidenceCalibration,
    EvidenceSupportScore,
    SemanticGrounding,
    VerificationResult,
    VerificationStatus,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider
    from contextmine_core.research.run import ResearchRun
    from contextmine_core.settings import Settings

logger = logging.getLogger(__name__)

# Regex pattern for citation IDs: [ev-xxx-nnn] or [ev_xxx_nnn]
CITATION_PATTERN = re.compile(r"\[ev[-_][a-zA-Z0-9]+[-_]\d+\]")


class GroundingCheckResult(BaseModel):
    """LLM output for semantic grounding check."""

    is_grounded: bool
    grounding_score: float
    ungrounded_claims: list[str]
    reasoning: str


class AnswerVerifier:
    """Verifies research run answers against evidence.

    Performs four types of verification:
    1. Citation validation: Check that citation IDs reference real evidence
    2. Evidence support: Score how well evidence supports the answer
    3. Semantic grounding: LLM-based check that answer claims are in evidence
    4. Confidence calibration: Compare stated confidence vs evidence quality
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            settings: Optional settings instance. Uses defaults if None.
            llm_provider: Optional LLM provider for semantic grounding check.
                          If None, semantic grounding is skipped.
        """
        if settings is None:
            from contextmine_core.settings import get_settings

            settings = get_settings()

        self._settings = settings
        self._llm_provider = llm_provider

    async def verify_async(self, run: ResearchRun) -> VerificationResult:
        """Verify a completed research run with semantic grounding.

        This is the preferred method when an LLM provider is available,
        as it performs full semantic grounding check.

        Args:
            run: The research run to verify

        Returns:
            VerificationResult with status, scores, and issues
        """
        # Check citations
        citations = self._verify_citations(run)

        # Assess evidence support
        evidence_support = self._assess_evidence_support(run)

        # Semantic grounding check (requires LLM)
        semantic_grounding = None
        if self._llm_provider and run.answer and run.evidence:
            semantic_grounding = await self._check_semantic_grounding(run)

        # Check confidence calibration
        confidence_calibration = self._check_confidence_calibration(run)

        # Collect issues
        issues = self._collect_issues(
            run, citations, evidence_support, confidence_calibration, semantic_grounding
        )

        # Determine overall status
        status = self._determine_status(
            citations, evidence_support, confidence_calibration, semantic_grounding, issues
        )

        return VerificationResult(
            status=status,
            citations=citations,
            evidence_support=evidence_support,
            semantic_grounding=semantic_grounding,
            confidence_calibration=confidence_calibration,
            issues=issues,
            verified_at=datetime.now(UTC).isoformat(),
        )

    def verify(self, run: ResearchRun) -> VerificationResult:
        """Verify a completed research run (sync, no semantic grounding).

        Note: This synchronous method does NOT perform semantic grounding.
        Use verify_async() for full verification including semantic grounding.

        Args:
            run: The research run to verify

        Returns:
            VerificationResult with status, scores, and issues
        """
        # Check citations
        citations = self._verify_citations(run)

        # Assess evidence support
        evidence_support = self._assess_evidence_support(run)

        # Check confidence calibration
        confidence_calibration = self._check_confidence_calibration(run)

        # Collect issues (no semantic grounding in sync mode)
        issues = self._collect_issues(
            run, citations, evidence_support, confidence_calibration, None
        )

        # Determine overall status
        status = self._determine_status(
            citations, evidence_support, confidence_calibration, None, issues
        )

        return VerificationResult(
            status=status,
            citations=citations,
            evidence_support=evidence_support,
            semantic_grounding=None,
            confidence_calibration=confidence_calibration,
            issues=issues,
            verified_at=datetime.now(UTC).isoformat(),
        )

    async def _check_semantic_grounding(self, run: ResearchRun) -> SemanticGrounding:
        """Check if answer claims are grounded in evidence using LLM.

        Args:
            run: The research run with answer and evidence

        Returns:
            SemanticGrounding result
        """
        if not self._llm_provider:
            return SemanticGrounding(
                is_grounded=False,
                grounding_score=0.0,
                ungrounded_claims=["Cannot verify - no LLM provider"],
                reasoning="Semantic grounding check requires LLM provider",
            )

        # Build evidence context
        evidence_text = self._format_evidence_for_grounding(run)

        system_prompt = """You are a verification assistant. Your job is to check if an answer is grounded in the provided evidence.

CRITICAL: An answer is ONLY grounded if ALL its claims can be found IN the evidence text.

If the answer makes claims about topics NOT present in the evidence, it is NOT grounded.
If the answer invents information not in the evidence, it is NOT grounded.
If the evidence is about a completely different topic than the answer, grounding_score should be 0.0.

Be STRICT. It's better to reject a good answer than to accept a hallucinated one."""

        user_message = f"""## Question
{run.question}

## Answer to verify
{run.answer}

## Evidence collected
{evidence_text}

Check if EVERY claim in the answer is supported by the evidence above.
List any claims that are NOT supported by the evidence."""

        try:
            result = await self._llm_provider.generate_structured(
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                output_schema=GroundingCheckResult,
                max_tokens=1024,
                temperature=0.0,
            )

            return SemanticGrounding(
                is_grounded=result.is_grounded,
                grounding_score=max(0.0, min(1.0, result.grounding_score)),
                ungrounded_claims=result.ungrounded_claims,
                reasoning=result.reasoning,
            )

        except Exception as e:
            logger.warning("Semantic grounding check failed: %s", e)
            return SemanticGrounding(
                is_grounded=False,
                grounding_score=0.0,
                ungrounded_claims=[f"Verification failed: {e}"],
                reasoning=f"Error during semantic grounding check: {e}",
            )

    def _format_evidence_for_grounding(self, run: ResearchRun) -> str:
        """Format evidence for grounding check."""
        parts = []
        for i, e in enumerate(run.evidence[:20], 1):  # Limit to 20 items
            parts.append(f"### Evidence {i}: {e.file_path}:{e.start_line}-{e.end_line}")
            parts.append(f"```\n{e.content[:1000]}\n```")  # Limit content length
            parts.append("")
        return "\n".join(parts)

    def _verify_citations(self, run: ResearchRun) -> list[CitationVerification]:
        """Verify all citations in the answer.

        Extracts citation IDs from the answer text and checks whether
        each one corresponds to an evidence item in the run.

        Args:
            run: The research run

        Returns:
            List of citation verification results
        """
        if not run.answer:
            return []

        # Extract citation IDs from the answer
        citation_ids = self._extract_citation_ids(run.answer)

        verifications = []
        for cid in citation_ids:
            # Clean the citation ID (remove brackets)
            clean_id = cid.strip("[]")

            # Try to find the evidence
            evidence = run.get_evidence_by_id(clean_id)

            if evidence:
                snippet = evidence.content[:100] if evidence.content else None
                verifications.append(
                    CitationVerification(
                        citation_id=clean_id,
                        found=True,
                        evidence_snippet=snippet,
                    )
                )
            else:
                verifications.append(
                    CitationVerification(
                        citation_id=clean_id,
                        found=False,
                        evidence_snippet=None,
                    )
                )

        return verifications

    def _extract_citation_ids(self, text: str) -> list[str]:
        """Extract citation IDs from text.

        Args:
            text: Text containing citations like [ev-abc-001]

        Returns:
            List of citation IDs found (including brackets)
        """
        return CITATION_PATTERN.findall(text)

    def _assess_evidence_support(self, run: ResearchRun) -> EvidenceSupportScore:
        """Assess how well evidence supports the answer.

        Uses heuristics based on:
        - Number of evidence items
        - Evidence relevance scores
        - Diversity of evidence sources

        Args:
            run: The research run

        Returns:
            Evidence support score with reasoning
        """
        if not run.evidence:
            return EvidenceSupportScore(
                score=0.0,
                reasoning="No evidence collected",
                supporting_evidence_ids=[],
            )

        evidence_count = len(run.evidence)
        evidence_ids = [e.id for e in run.evidence]

        # Calculate average relevance score
        scored_evidence = [e.score for e in run.evidence if e.score is not None]
        avg_score = sum(scored_evidence, 0.0) / len(scored_evidence) if scored_evidence else 0.5

        # Calculate provenance diversity
        provenances = {e.provenance for e in run.evidence}
        diversity_bonus = min(len(provenances) * 0.1, 0.3)  # Up to 0.3 bonus

        # Calculate final score
        # - Base: evidence count factor (logarithmic, caps at 10 items)
        count_factor = min(evidence_count / 5, 1.0)  # 5+ items = full score
        score = count_factor * 0.4 + avg_score * 0.4 + diversity_bonus + 0.1

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        reasoning_parts = [
            f"{evidence_count} evidence items collected",
            f"average relevance score: {avg_score:.2f}",
            f"{len(provenances)} provenance types",
        ]

        return EvidenceSupportScore(
            score=score,
            reasoning="; ".join(reasoning_parts),
            supporting_evidence_ids=evidence_ids,
        )

    def _check_confidence_calibration(self, run: ResearchRun) -> ConfidenceCalibration:
        """Check if stated confidence matches evidence quality.

        Args:
            run: The research run

        Returns:
            Confidence calibration result
        """
        # Extract stated confidence from answer
        stated_confidence = self._extract_confidence(run.answer)

        # Calculate evidence-based confidence
        evidence_confidence = self._calculate_evidence_confidence(run)

        # Calculate calibration delta
        calibration_delta = abs(stated_confidence - evidence_confidence)

        # Check if within tolerance
        tolerance = getattr(self._settings, "verification_confidence_tolerance", 0.2)
        is_calibrated = calibration_delta <= tolerance

        return ConfidenceCalibration(
            stated_confidence=stated_confidence,
            evidence_confidence=evidence_confidence,
            calibration_delta=calibration_delta,
            is_calibrated=is_calibrated,
        )

    def _extract_confidence(self, answer: str | None) -> float:
        """Extract confidence level from answer text.

        Looks for patterns like:
        - "Confidence: high/medium/low"
        - "Confidence: 0.8"
        - "confidence level: 80%"

        Args:
            answer: The answer text

        Returns:
            Confidence as float (0.0 to 1.0), defaults to 0.5
        """
        if not answer:
            return 0.5

        # Pattern for explicit percentage
        pct_match = re.search(r"[Cc]onfidence[:\s]+(\d+)%", answer)
        if pct_match:
            return int(pct_match.group(1)) / 100

        # Pattern for explicit decimal
        decimal_match = re.search(r"[Cc]onfidence[:\s]+(0?\.\d+|1\.0)", answer)
        if decimal_match:
            return float(decimal_match.group(1))

        # Pattern for word-based confidence
        lower_answer = answer.lower()
        if "confidence: high" in lower_answer or "high confidence" in lower_answer:
            return 0.85
        if "confidence: medium" in lower_answer or "medium confidence" in lower_answer:
            return 0.6
        if "confidence: low" in lower_answer or "low confidence" in lower_answer:
            return 0.35

        # Default confidence if not stated
        return 0.5

    def _calculate_evidence_confidence(self, run: ResearchRun) -> float:
        """Calculate confidence based on evidence quality.

        Args:
            run: The research run

        Returns:
            Calculated confidence (0.0 to 1.0)
        """
        if not run.evidence:
            return 0.2  # Low confidence with no evidence

        # Factors:
        # 1. Evidence count (more = higher confidence)
        count_factor = min(len(run.evidence) / 5, 1.0) * 0.3

        # 2. Average relevance score
        scored = [e.score for e in run.evidence if e.score is not None]
        avg_score = sum(scored, 0.0) / len(scored) if scored else 0.5
        score_factor = avg_score * 0.4

        # 3. Completion (did run finish successfully?)
        from contextmine_core.research.run import RunStatus

        completion_factor = 0.3 if run.status == RunStatus.DONE else 0.1

        confidence = count_factor + score_factor + completion_factor
        return max(0.0, min(1.0, confidence))

    def _collect_issues(
        self,
        run: ResearchRun,
        citations: list[CitationVerification],
        evidence_support: EvidenceSupportScore,
        confidence_calibration: ConfidenceCalibration,
        semantic_grounding: SemanticGrounding | None,
    ) -> list[str]:
        """Collect all verification issues.

        Args:
            run: The research run
            citations: Citation verification results
            evidence_support: Evidence support score
            confidence_calibration: Confidence calibration result
            semantic_grounding: Semantic grounding check result

        Returns:
            List of issue descriptions
        """
        issues = []

        # CRITICAL: Check semantic grounding first (most important check)
        if semantic_grounding is not None and not semantic_grounding.is_grounded:
            issues.append(
                f"HALLUCINATION DETECTED: Answer is not grounded in evidence "
                f"(score: {semantic_grounding.grounding_score:.2f})"
            )
            if semantic_grounding.ungrounded_claims:
                for claim in semantic_grounding.ungrounded_claims[:3]:
                    issues.append(f"  - Ungrounded: {claim}")

        # Check for missing citations
        require_citations = getattr(self._settings, "verification_require_citations", True)
        missing_citations = [c for c in citations if not c.found]
        if missing_citations:
            ids = ", ".join(c.citation_id for c in missing_citations)
            issues.append(f"Invalid citation IDs: {ids}")
        elif require_citations and run.answer and not citations:
            # Answer exists but no citations
            issues.append("Answer contains no citations to evidence")

        # Check evidence support
        min_support = getattr(self._settings, "verification_min_evidence_support", 0.5)
        if evidence_support.score < min_support:
            issues.append(
                f"Evidence support score ({evidence_support.score:.2f}) "
                f"below minimum ({min_support})"
            )

        # Check confidence calibration
        if not confidence_calibration.is_calibrated:
            issues.append(
                f"Confidence miscalibrated: stated {confidence_calibration.stated_confidence:.2f}, "
                f"evidence suggests {confidence_calibration.evidence_confidence:.2f}"
            )

        # Check for no answer
        if not run.answer:
            issues.append("No answer provided")

        # Check for no evidence
        if not run.evidence:
            issues.append("No evidence collected")

        return issues

    def _determine_status(
        self,
        citations: list[CitationVerification],
        evidence_support: EvidenceSupportScore,
        confidence_calibration: ConfidenceCalibration,
        semantic_grounding: SemanticGrounding | None,
        issues: list[str],
    ) -> VerificationStatus:
        """Determine overall verification status.

        Args:
            citations: Citation verification results
            evidence_support: Evidence support score
            confidence_calibration: Confidence calibration result
            semantic_grounding: Semantic grounding check result
            issues: List of issues found

        Returns:
            Overall verification status
        """
        # CRITICAL: Semantic grounding failure = immediate fail
        if semantic_grounding is not None and not semantic_grounding.is_grounded:
            return VerificationStatus.FAILED

        # Critical failures
        missing_citations = [c for c in citations if not c.found]
        if missing_citations:
            return VerificationStatus.FAILED

        if evidence_support.score < 0.2:
            return VerificationStatus.FAILED

        # Warnings (non-critical issues)
        if issues:
            return VerificationStatus.WARNING

        return VerificationStatus.PASSED
