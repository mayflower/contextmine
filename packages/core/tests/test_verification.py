"""Tests for the verification module.

Tests actual verification behavior, not dataclass serialization.
"""

from datetime import UTC, datetime

import pytest
from contextmine_core.research.run import Evidence, ResearchRun, RunStatus
from contextmine_core.research.verification import (
    AnswerVerifier,
    VerificationStatus,
)


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    """Create sample evidence items."""
    return [
        Evidence(
            id="ev-abc-001",
            file_path="src/auth.py",
            start_line=10,
            end_line=20,
            content="def authenticate(user, password):\n    return check_credentials(user, password)",
            reason="Authentication function",
            provenance="search",
            score=0.85,
        ),
        Evidence(
            id="ev-abc-002",
            file_path="src/utils.py",
            start_line=5,
            end_line=15,
            content="def check_credentials(user, password):\n    return db.verify(user, password)",
            reason="Credential verification",
            provenance="lsp",
            score=0.75,
        ),
    ]


@pytest.fixture
def verifier() -> AnswerVerifier:
    """Create an answer verifier."""
    return AnswerVerifier()


class TestAnswerVerifier:
    """Tests for the AnswerVerifier class - actual behavior tests."""

    def test_verify_passing_run(self, verifier: AnswerVerifier, sample_evidence: list[Evidence]):
        """Test verification of a good run passes."""
        run = ResearchRun(
            run_id="test-run-001",
            question="How does authentication work?",
            status=RunStatus.DONE,
            created_at=datetime.now(UTC),
            answer="Authentication uses the `authenticate` function [ev-abc-001] which calls `check_credentials` [ev-abc-002]. Confidence: high",
            evidence=sample_evidence,
        )
        run.complete(run.answer)

        result = verifier.verify(run)

        assert result.status in (VerificationStatus.PASSED, VerificationStatus.WARNING)
        assert len(result.citations) == 2
        assert all(c.found for c in result.citations)
        assert result.evidence_support.score > 0.5

    def test_verify_invalid_citations(
        self, verifier: AnswerVerifier, sample_evidence: list[Evidence]
    ):
        """Test verification catches invalid citations."""
        run = ResearchRun(
            run_id="test-run-002",
            question="Test question",
            status=RunStatus.DONE,
            created_at=datetime.now(UTC),
            answer="See [ev-invalid-001] and [ev-notfound-002]",
            evidence=sample_evidence,
        )
        run.complete(run.answer)

        result = verifier.verify(run)

        assert result.status == VerificationStatus.FAILED
        assert not any(c.found for c in result.citations)
        assert any("Invalid citation" in issue for issue in result.issues)

    def test_verify_no_evidence(self, verifier: AnswerVerifier):
        """Test verification of run with no evidence."""
        run = ResearchRun(
            run_id="test-run-003",
            question="Test question",
            status=RunStatus.DONE,
            created_at=datetime.now(UTC),
            answer="I don't know",
            evidence=[],
        )
        run.complete(run.answer)

        result = verifier.verify(run)

        assert result.evidence_support.score == 0.0
        assert "No evidence collected" in result.issues

    def test_verify_no_answer(self, verifier: AnswerVerifier):
        """Test verification of run with no answer."""
        run = ResearchRun(
            run_id="test-run-004",
            question="Test question",
            status=RunStatus.ERROR,
            created_at=datetime.now(UTC),
            error_message="Failed",
        )

        result = verifier.verify(run)

        assert "No answer provided" in result.issues

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Confidence: high", 0.85),
            ("I have high confidence in this", 0.85),
            ("Confidence: medium", 0.6),
            ("Confidence: low", 0.35),
            ("Confidence: 80%", 0.8),
            ("My confidence: 75%", 0.75),
            ("Confidence: 0.9", 0.9),
            ("confidence: 0.65", 0.65),
            ("Just an answer without confidence", 0.5),
            (None, 0.5),
        ],
    )
    def test_extract_confidence(self, verifier: AnswerVerifier, text: str | None, expected: float):
        """Test confidence extraction from various formats."""
        assert verifier._extract_confidence(text) == expected

    @pytest.mark.parametrize(
        "text,expected_count",
        [
            ("See [ev-abc-001] and also [ev-xyz-123] for details", 2),
            ("See [ev_abc_001]", 1),
            ("No citations here", 0),
            ("[ev-a-1][ev-b-2][ev-c-3]", 3),  # Pattern: ev-{alpha}-{digits}
        ],
    )
    def test_extract_citation_ids(self, verifier: AnswerVerifier, text: str, expected_count: int):
        """Test citation ID extraction."""
        citations = verifier._extract_citation_ids(text)
        assert len(citations) == expected_count
