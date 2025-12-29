"""Tests for ResearchRun data structures."""

from contextmine_core.research.run import (
    ActionStep,
    Evidence,
    ResearchRun,
    RunStatus,
)


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        evidence = Evidence(
            id="ev-001",
            file_path="src/main.py",
            start_line=10,
            end_line=20,
            content="def hello():\n    print('hello')",
            reason="Contains the hello function",
            provenance="bm25",
            score=0.85,
            symbol_id="main.hello",
            symbol_kind="function",
        )

        d = evidence.to_dict()
        assert d["id"] == "ev-001"
        assert d["file_path"] == "src/main.py"
        assert d["start_line"] == 10
        assert d["end_line"] == 20
        assert d["provenance"] == "bm25"
        assert d["score"] == 0.85
        assert d["symbol_id"] == "main.hello"

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "id": "ev-002",
            "file_path": "lib/utils.py",
            "start_line": 5,
            "end_line": 15,
            "content": "class Helper:\n    pass",
            "reason": "Utility class",
            "provenance": "vector",
        }

        evidence = Evidence.from_dict(data)
        assert evidence.id == "ev-002"
        assert evidence.file_path == "lib/utils.py"
        assert evidence.score is None
        assert evidence.symbol_id is None


class TestActionStep:
    """Tests for ActionStep dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        step = ActionStep(
            step_number=1,
            action="hybrid_search",
            input={"query": "hello world", "top_k": 10},
            output_summary="Found 5 results",
            duration_ms=150,
            evidence_ids=["ev-001", "ev-002"],
        )

        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["action"] == "hybrid_search"
        assert d["input"]["query"] == "hello world"
        assert d["duration_ms"] == 150
        assert "ev-001" in d["evidence_ids"]

    def test_error_step(self) -> None:
        """Test step with error."""
        step = ActionStep(
            step_number=2,
            action="lsp_definition",
            input={"file": "missing.py", "line": 1},
            output_summary="",
            duration_ms=50,
            error="File not found",
        )

        d = step.to_dict()
        assert d["error"] == "File not found"


class TestResearchRun:
    """Tests for ResearchRun dataclass."""

    def test_create(self) -> None:
        """Test creating a new run."""
        run = ResearchRun.create(
            question="What does the main function do?",
            scope="src/",
            budget_steps=5,
        )

        assert run.run_id  # UUID is generated
        assert run.question == "What does the main function do?"
        assert run.status == RunStatus.RUNNING
        assert run.scope == "src/"
        assert run.budget_steps == 5
        assert run.budget_used == 0
        assert run.created_at is not None

    def test_add_step(self) -> None:
        """Test adding steps to a run."""
        run = ResearchRun.create(question="Test question")

        step = ActionStep(
            step_number=1,
            action="hybrid_search",
            input={"query": "test"},
            output_summary="Found results",
            duration_ms=100,
        )
        run.add_step(step)

        assert run.budget_used == 1
        assert run.total_duration_ms == 100
        assert len(run.steps) == 1

    def test_add_evidence(self) -> None:
        """Test adding evidence to a run."""
        run = ResearchRun.create(question="Test question")

        evidence = Evidence(
            id="ev-001",
            file_path="test.py",
            start_line=1,
            end_line=5,
            content="test code",
            reason="Test reason",
            provenance="manual",
        )
        run.add_evidence(evidence)

        assert len(run.evidence) == 1
        assert run.get_evidence_by_id("ev-001") == evidence
        assert run.get_evidence_by_id("nonexistent") is None

    def test_complete(self) -> None:
        """Test completing a run."""
        run = ResearchRun.create(question="Test question")
        run.complete("The answer is 42")

        assert run.status == RunStatus.DONE
        assert run.answer == "The answer is 42"
        assert run.completed_at is not None

    def test_fail(self) -> None:
        """Test failing a run."""
        run = ResearchRun.create(question="Test question")
        run.fail("Something went wrong")

        assert run.status == RunStatus.ERROR
        assert run.error_message == "Something went wrong"
        assert run.completed_at is not None

    def test_to_trace_dict(self) -> None:
        """Test trace serialization."""
        run = ResearchRun.create(question="Test question")
        run.add_step(
            ActionStep(
                step_number=1,
                action="test",
                input={},
                output_summary="ok",
                duration_ms=10,
            )
        )
        run.complete("Done")

        trace = run.to_trace_dict()
        assert trace["run_id"] == run.run_id
        assert trace["question"] == "Test question"
        assert trace["status"] == "done"
        assert len(trace["steps"]) == 1

    def test_to_evidence_dict(self) -> None:
        """Test evidence serialization."""
        run = ResearchRun.create(question="Test question")
        run.add_evidence(
            Evidence(
                id="ev-001",
                file_path="test.py",
                start_line=1,
                end_line=5,
                content="code",
                reason="reason",
                provenance="bm25",
            )
        )

        evidence = run.to_evidence_dict()
        assert evidence["run_id"] == run.run_id
        assert evidence["evidence_count"] == 1
        assert len(evidence["evidence"]) == 1

    def test_to_report_markdown(self) -> None:
        """Test markdown report generation."""
        run = ResearchRun.create(question="What is X?")
        run.add_evidence(
            Evidence(
                id="ev-001",
                file_path="src/x.py",
                start_line=10,
                end_line=15,
                content="class X:\n    pass",
                reason="Definition of X",
                provenance="lsp",
            )
        )
        run.add_step(
            ActionStep(
                step_number=1,
                action="lsp_definition",
                input={"symbol": "X"},
                output_summary="Found X at src/x.py:10",
                duration_ms=50,
            )
        )
        run.complete("X is a class in src/x.py")

        report = run.to_report_markdown()
        assert "# Research Report:" in report
        assert "What is X?" in report
        assert "X is a class in src/x.py" in report
        assert "src/x.py:10-15" in report
        assert "lsp_definition" in report

    def test_from_dict_roundtrip(self) -> None:
        """Test serialization roundtrip."""
        run = ResearchRun.create(question="Test question", scope="src/")
        run.add_step(
            ActionStep(
                step_number=1,
                action="test",
                input={"key": "value"},
                output_summary="ok",
                duration_ms=100,
            )
        )
        run.add_evidence(
            Evidence(
                id="ev-001",
                file_path="test.py",
                start_line=1,
                end_line=5,
                content="code",
                reason="reason",
                provenance="manual",
            )
        )
        run.complete("Done")

        # Combine trace and evidence for full roundtrip
        combined = {
            **run.to_trace_dict(),
            "evidence": [e.to_dict() for e in run.evidence],
            "answer": run.answer,
        }

        restored = ResearchRun.from_dict(combined)
        assert restored.run_id == run.run_id
        assert restored.question == run.question
        assert restored.status == run.status
        assert len(restored.steps) == 1
        assert len(restored.evidence) == 1
        assert restored.answer == "Done"
