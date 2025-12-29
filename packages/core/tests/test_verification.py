"""Tests for the verification module."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from contextmine_core.research.run import Evidence, ResearchRun, RunStatus
from contextmine_core.research.verification import (
    AnswerVerifier,
    CitationVerification,
    ConfidenceCalibration,
    EvidenceSupportScore,
    VerificationResult,
    VerificationStatus,
)

# =============================================================================
# FIXTURES
# =============================================================================


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
def completed_run(sample_evidence: list[Evidence]) -> ResearchRun:
    """Create a completed research run with evidence."""
    run = ResearchRun(
        run_id="test-run-001",
        question="How does authentication work?",
        status=RunStatus.DONE,
        created_at=datetime.now(UTC),
        answer="Authentication uses the `authenticate` function [ev-abc-001] which calls `check_credentials` [ev-abc-002]. Confidence: high",
        evidence=sample_evidence,
    )
    run.complete(
        "Authentication uses the `authenticate` function [ev-abc-001] which calls `check_credentials` [ev-abc-002]. Confidence: high"
    )
    return run


@pytest.fixture
def verifier() -> AnswerVerifier:
    """Create an answer verifier."""
    return AnswerVerifier()


# =============================================================================
# CITATION VERIFICATION TESTS
# =============================================================================


class TestCitationVerification:
    """Tests for CitationVerification dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        cv = CitationVerification(
            citation_id="ev-abc-001",
            found=True,
            evidence_snippet="def authenticate",
        )
        d = cv.to_dict()
        assert d["citation_id"] == "ev-abc-001"
        assert d["found"] is True
        assert d["evidence_snippet"] == "def authenticate"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "citation_id": "ev-xyz-002",
            "found": False,
            "evidence_snippet": None,
        }
        cv = CitationVerification.from_dict(d)
        assert cv.citation_id == "ev-xyz-002"
        assert cv.found is False
        assert cv.evidence_snippet is None


class TestEvidenceSupportScore:
    """Tests for EvidenceSupportScore dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        score = EvidenceSupportScore(
            score=0.85,
            reasoning="Good coverage",
            supporting_evidence_ids=["ev-001", "ev-002"],
        )
        d = score.to_dict()
        assert d["score"] == 0.85
        assert d["reasoning"] == "Good coverage"
        assert d["supporting_evidence_ids"] == ["ev-001", "ev-002"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "score": 0.6,
            "reasoning": "Some evidence",
            "supporting_evidence_ids": ["ev-003"],
        }
        score = EvidenceSupportScore.from_dict(d)
        assert score.score == 0.6
        assert score.reasoning == "Some evidence"
        assert score.supporting_evidence_ids == ["ev-003"]


class TestConfidenceCalibration:
    """Tests for ConfidenceCalibration dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        cal = ConfidenceCalibration(
            stated_confidence=0.8,
            evidence_confidence=0.75,
            calibration_delta=0.05,
            is_calibrated=True,
        )
        d = cal.to_dict()
        assert d["stated_confidence"] == 0.8
        assert d["evidence_confidence"] == 0.75
        assert d["calibration_delta"] == 0.05
        assert d["is_calibrated"] is True

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "stated_confidence": 0.9,
            "evidence_confidence": 0.5,
            "calibration_delta": 0.4,
            "is_calibrated": False,
        }
        cal = ConfidenceCalibration.from_dict(d)
        assert cal.stated_confidence == 0.9
        assert cal.calibration_delta == 0.4
        assert cal.is_calibrated is False


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        result = VerificationResult(
            status=VerificationStatus.PASSED,
            citations=[
                CitationVerification("ev-001", found=True),
            ],
            evidence_support=EvidenceSupportScore(
                score=0.8,
                reasoning="Good",
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
            verified_at="2024-01-01T00:00:00",
        )

        d = result.to_dict()
        restored = VerificationResult.from_dict(d)

        assert restored.status == VerificationStatus.PASSED
        assert len(restored.citations) == 1
        assert restored.citations[0].found is True
        assert restored.evidence_support.score == 0.8
        assert restored.confidence_calibration.is_calibrated is True
        assert restored.issues == []

    def test_citation_validity_rate(self):
        """Test citation validity rate calculation."""
        result = VerificationResult(
            status=VerificationStatus.WARNING,
            citations=[
                CitationVerification("ev-001", found=True),
                CitationVerification("ev-002", found=True),
                CitationVerification("ev-003", found=False),
            ],
            evidence_support=EvidenceSupportScore(0.5, "ok", []),
            semantic_grounding=None,
            confidence_calibration=ConfidenceCalibration(0.5, 0.5, 0.0, True),
            issues=[],
            verified_at="",
        )
        assert result.citation_validity_rate == pytest.approx(2 / 3)

    def test_citation_validity_rate_no_citations(self):
        """Test that no citations gives 100% validity."""
        result = VerificationResult(
            status=VerificationStatus.PASSED,
            citations=[],
            evidence_support=EvidenceSupportScore(0.5, "ok", []),
            semantic_grounding=None,
            confidence_calibration=ConfidenceCalibration(0.5, 0.5, 0.0, True),
            issues=[],
            verified_at="",
        )
        assert result.citation_validity_rate == 1.0

    def test_create_passed(self):
        """Test factory method for passed results."""
        result = VerificationResult.create_passed(
            citations=[],
            evidence_support=EvidenceSupportScore(0.8, "good", ["ev-001"]),
            confidence_calibration=ConfidenceCalibration(0.8, 0.8, 0.0, True),
        )
        assert result.status == VerificationStatus.PASSED
        assert result.issues == []
        assert result.verified_at  # Should have a timestamp

    def test_create_failed(self):
        """Test factory method for failed results."""
        result = VerificationResult.create_failed(
            citations=[],
            evidence_support=EvidenceSupportScore(0.2, "poor", []),
            confidence_calibration=ConfidenceCalibration(0.9, 0.2, 0.7, False),
            issues=["Low evidence support", "Miscalibrated confidence"],
        )
        assert result.status == VerificationStatus.FAILED
        assert len(result.issues) == 2


# =============================================================================
# ANSWER VERIFIER TESTS
# =============================================================================


class TestAnswerVerifier:
    """Tests for the AnswerVerifier class."""

    def test_verify_passing_run(self, verifier: AnswerVerifier, completed_run: ResearchRun):
        """Test verification of a good run passes."""
        result = verifier.verify(completed_run)

        assert result.status in (VerificationStatus.PASSED, VerificationStatus.WARNING)
        assert len(result.citations) == 2
        assert all(c.found for c in result.citations)
        assert result.evidence_support.score > 0.5
        assert result.confidence_calibration.stated_confidence > 0

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
        run.complete("See [ev-invalid-001] and [ev-notfound-002]")

        result = verifier.verify(run)

        assert result.status == VerificationStatus.FAILED
        assert len(result.citations) == 2
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
        run.complete("I don't know")

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
        assert len(result.citations) == 0

    def test_extract_confidence_high(self, verifier: AnswerVerifier):
        """Test extracting high confidence."""
        assert verifier._extract_confidence("Confidence: high") == 0.85
        assert verifier._extract_confidence("I have high confidence in this") == 0.85

    def test_extract_confidence_medium(self, verifier: AnswerVerifier):
        """Test extracting medium confidence."""
        assert verifier._extract_confidence("Confidence: medium") == 0.6

    def test_extract_confidence_low(self, verifier: AnswerVerifier):
        """Test extracting low confidence."""
        assert verifier._extract_confidence("Confidence: low") == 0.35
        assert verifier._extract_confidence("I have low confidence") == 0.35

    def test_extract_confidence_percentage(self, verifier: AnswerVerifier):
        """Test extracting percentage confidence."""
        assert verifier._extract_confidence("Confidence: 80%") == 0.8
        assert verifier._extract_confidence("My confidence: 75%") == 0.75

    def test_extract_confidence_decimal(self, verifier: AnswerVerifier):
        """Test extracting decimal confidence."""
        assert verifier._extract_confidence("Confidence: 0.9") == 0.9
        assert verifier._extract_confidence("confidence: 0.65") == 0.65

    def test_extract_confidence_default(self, verifier: AnswerVerifier):
        """Test default confidence when not specified."""
        assert verifier._extract_confidence("Just an answer without confidence") == 0.5
        assert verifier._extract_confidence(None) == 0.5

    def test_extract_citation_ids(self, verifier: AnswerVerifier):
        """Test citation ID extraction."""
        text = "See [ev-abc-001] and also [ev-xyz-123] for details"
        citations = verifier._extract_citation_ids(text)
        assert len(citations) == 2
        assert "[ev-abc-001]" in citations
        assert "[ev-xyz-123]" in citations

    def test_extract_citation_ids_with_underscore(self, verifier: AnswerVerifier):
        """Test citation ID extraction with underscores."""
        text = "See [ev_abc_001]"
        citations = verifier._extract_citation_ids(text)
        assert len(citations) == 1
        assert "[ev_abc_001]" in citations


# =============================================================================
# VERIFICATION RESULT WITH RUN INTEGRATION
# =============================================================================


class TestVerificationWithRun:
    """Tests for verification integration with ResearchRun."""

    def test_run_to_dict_includes_verification(self, sample_evidence: list[Evidence]):
        """Test that verification is included in run serialization."""
        run = ResearchRun(
            run_id="test-run-005",
            question="Test",
            status=RunStatus.DONE,
            created_at=datetime.now(UTC),
            answer="Test answer",
            evidence=sample_evidence,
        )

        # Add verification
        run.verification = VerificationResult.create_passed(
            citations=[],
            evidence_support=EvidenceSupportScore(0.8, "good", []),
            confidence_calibration=ConfidenceCalibration(0.8, 0.8, 0.0, True),
        )

        d = run.to_trace_dict()
        assert "verification" in d
        assert d["verification"]["status"] == "passed"

    def test_run_from_dict_restores_verification(self, sample_evidence: list[Evidence]):
        """Test that verification is restored from dict."""
        run = ResearchRun(
            run_id="test-run-006",
            question="Test",
            status=RunStatus.DONE,
            created_at=datetime.now(UTC),
            answer="Test answer",
            evidence=sample_evidence,
        )
        run.verification = VerificationResult.create_passed(
            citations=[],
            evidence_support=EvidenceSupportScore(0.8, "good", []),
            confidence_calibration=ConfidenceCalibration(0.8, 0.8, 0.0, True),
        )

        # Serialize and restore
        d = run.to_trace_dict()
        d["evidence"] = [e.to_dict() for e in sample_evidence]
        restored = ResearchRun.from_dict(d)

        assert restored.verification is not None
        assert restored.verification.status == VerificationStatus.PASSED
