"""Tests for MCP research resources.

These tests verify the MCP resource handlers work correctly with the artifact store.
The artifact store behavior is comprehensively tested in packages/core/tests/test_artifacts.py.
"""

from contextmine_core.research import (
    MemoryArtifactStore,
    ResearchRun,
)
from contextmine_core.research.run import ActionStep, Evidence


def create_sample_run() -> ResearchRun:
    """Create a sample research run for testing."""
    run = ResearchRun.create(question="How does search work?")
    run.add_step(
        ActionStep(
            step_number=1,
            action="hybrid_search",
            input={"query": "search"},
            output_summary="Found 3 results",
            duration_ms=150,
            evidence_ids=["ev-001"],
        )
    )
    run.add_evidence(
        Evidence(
            id="ev-001",
            file_path="src/search.py",
            start_line=10,
            end_line=20,
            content="def search():\n    pass",
            reason="Search function",
            provenance="bm25",
        )
    )
    run.complete("Search uses hybrid retrieval.")
    return run


class TestMCPResourcesContract:
    """Contract tests verifying MCP resources can list and read artifacts.

    These tests verify that:
    1. Resources can list runs from the artifact store
    2. Resources can read trace, evidence, and report for a run
    3. The data format matches what MCP resources would return
    """

    def test_list_runs_returns_metadata(self) -> None:
        """Test that listing runs returns proper metadata for MCP resources."""
        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        # Verify list behavior that MCP resource depends on
        runs = store.list_runs(limit=20)
        assert len(runs) == 1
        meta = runs[0]
        assert meta.run_id == run.run_id
        assert meta.question == "How does search work?"
        assert meta.status == "done"
        assert meta.created_at is not None

    def test_get_trace_returns_json_serializable(self) -> None:
        """Test that trace is JSON-serializable for MCP resources."""
        import json

        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        trace = store.get_trace(run.run_id)
        assert trace is not None

        # Verify it's JSON-serializable (what MCP resource returns)
        json_str = json.dumps(trace, indent=2)
        assert "hybrid_search" in json_str
        assert run.run_id in json_str

    def test_get_evidence_returns_json_serializable(self) -> None:
        """Test that evidence is JSON-serializable for MCP resources."""
        import json

        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        evidence = store.get_evidence(run.run_id)
        assert evidence is not None

        # Verify it's JSON-serializable (what MCP resource returns)
        json_str = json.dumps(evidence, indent=2)
        assert "ev-001" in json_str
        assert "src/search.py" in json_str

    def test_get_report_returns_markdown(self) -> None:
        """Test that report is proper markdown for MCP resources."""
        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        report = store.get_report(run.run_id)
        assert report is not None

        # Verify markdown content (what MCP resource returns)
        assert "# Research Report:" in report
        assert "How does search work?" in report
        assert "src/search.py" in report

    def test_nonexistent_run_returns_none(self) -> None:
        """Test that missing runs return None (MCP resources convert to error)."""
        store = MemoryArtifactStore()

        assert store.get_trace("nonexistent") is None
        assert store.get_evidence("nonexistent") is None
        assert store.get_report("nonexistent") is None
