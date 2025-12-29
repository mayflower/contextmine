"""Tests for ArtifactStore implementations."""

import tempfile
import time
from pathlib import Path

from contextmine_core.research.artifacts import (
    FileArtifactStore,
    MemoryArtifactStore,
    reset_artifact_store,
)
from contextmine_core.research.run import (
    ActionStep,
    Evidence,
    ResearchRun,
)


def create_sample_run(question: str = "Test question") -> ResearchRun:
    """Create a sample research run for testing."""
    run = ResearchRun.create(question=question)
    run.add_step(
        ActionStep(
            step_number=1,
            action="hybrid_search",
            input={"query": "test"},
            output_summary="Found 3 results",
            duration_ms=150,
            evidence_ids=["ev-001"],
        )
    )
    run.add_evidence(
        Evidence(
            id="ev-001",
            file_path="src/main.py",
            start_line=10,
            end_line=20,
            content="def main():\n    pass",
            reason="Entry point",
            provenance="bm25",
            score=0.9,
        )
    )
    run.complete("The main function is the entry point.")
    return run


class TestMemoryArtifactStore:
    """Tests for MemoryArtifactStore."""

    def test_save_and_get_trace(self) -> None:
        """Test saving and retrieving trace."""
        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        trace = store.get_trace(run.run_id)
        assert trace is not None
        assert trace["run_id"] == run.run_id
        assert trace["question"] == "Test question"
        assert len(trace["steps"]) == 1

    def test_save_and_get_evidence(self) -> None:
        """Test saving and retrieving evidence."""
        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        evidence = store.get_evidence(run.run_id)
        assert evidence is not None
        assert evidence["evidence_count"] == 1
        assert evidence["evidence"][0]["id"] == "ev-001"

    def test_save_and_get_report(self) -> None:
        """Test saving and retrieving report."""
        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        report = store.get_report(run.run_id)
        assert report is not None
        assert "# Research Report:" in report
        assert "Test question" in report

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent run returns None."""
        store = MemoryArtifactStore()

        assert store.get_trace("nonexistent") is None
        assert store.get_evidence("nonexistent") is None
        assert store.get_report("nonexistent") is None

    def test_list_runs(self) -> None:
        """Test listing runs returns newest first."""
        store = MemoryArtifactStore()

        run1 = create_sample_run("First question")
        store.save_run(run1)
        time.sleep(0.01)  # Ensure different timestamps

        run2 = create_sample_run("Second question")
        store.save_run(run2)

        runs = store.list_runs()
        assert len(runs) == 2
        # Newest first
        assert runs[0].run_id == run2.run_id
        assert runs[1].run_id == run1.run_id

    def test_list_runs_limit(self) -> None:
        """Test listing runs with limit."""
        store = MemoryArtifactStore()

        for i in range(5):
            run = create_sample_run(f"Question {i}")
            store.save_run(run)
            time.sleep(0.01)

        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_delete_run(self) -> None:
        """Test deleting a run."""
        store = MemoryArtifactStore()
        run = create_sample_run()
        store.save_run(run)

        assert store.run_exists(run.run_id)
        assert store.delete_run(run.run_id)
        assert not store.run_exists(run.run_id)
        assert not store.delete_run(run.run_id)  # Second delete returns False

    def test_max_runs_eviction(self) -> None:
        """Test that old runs are evicted when max_runs is exceeded."""
        store = MemoryArtifactStore(max_runs=3)

        runs = []
        for i in range(5):
            run = create_sample_run(f"Question {i}")
            runs.append(run)
            store.save_run(run)
            time.sleep(0.01)

        # Should only have 3 runs (the newest ones)
        all_runs = store.list_runs(limit=10)
        assert len(all_runs) == 3

        # Oldest runs should be evicted
        assert not store.run_exists(runs[0].run_id)
        assert not store.run_exists(runs[1].run_id)
        # Newest runs should still exist
        assert store.run_exists(runs[2].run_id)
        assert store.run_exists(runs[3].run_id)
        assert store.run_exists(runs[4].run_id)

    def test_ttl_eviction(self) -> None:
        """Test TTL-based eviction."""
        # Use very short TTL for testing
        store = MemoryArtifactStore(ttl_minutes=0)  # 0 minutes = immediate expiry

        run = create_sample_run()
        store.save_run(run)

        # Wait a tiny bit so expiration happens
        time.sleep(0.1)

        # Evict expired
        evicted = store.evict_expired()
        assert evicted == 1
        assert not store.run_exists(run.run_id)


class TestFileArtifactStore:
    """Tests for FileArtifactStore."""

    def test_save_and_get_trace(self) -> None:
        """Test saving and retrieving trace from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir)
            run = create_sample_run()
            store.save_run(run)

            trace = store.get_trace(run.run_id)
            assert trace is not None
            assert trace["run_id"] == run.run_id

            # Verify file exists
            trace_path = Path(tmpdir) / run.run_id / "trace.json"
            assert trace_path.exists()

    def test_save_and_get_evidence(self) -> None:
        """Test saving and retrieving evidence from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir)
            run = create_sample_run()
            store.save_run(run)

            evidence = store.get_evidence(run.run_id)
            assert evidence is not None
            assert evidence["evidence_count"] == 1

    def test_save_and_get_report(self) -> None:
        """Test saving and retrieving report from files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir)
            run = create_sample_run()
            store.save_run(run)

            report = store.get_report(run.run_id)
            assert report is not None
            assert "# Research Report:" in report

    def test_list_runs(self) -> None:
        """Test listing runs from file store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir)

            run1 = create_sample_run("First")
            store.save_run(run1)
            time.sleep(0.01)

            run2 = create_sample_run("Second")
            store.save_run(run2)

            runs = store.list_runs()
            assert len(runs) == 2
            assert runs[0].run_id == run2.run_id  # Newest first

    def test_delete_run(self) -> None:
        """Test deleting a run removes files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir)
            run = create_sample_run()
            store.save_run(run)

            run_dir = Path(tmpdir) / run.run_id
            assert run_dir.exists()

            store.delete_run(run.run_id)
            assert not run_dir.exists()

    def test_max_runs_eviction(self) -> None:
        """Test max_runs eviction for file store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir, max_runs=3)

            runs = []
            for i in range(5):
                run = create_sample_run(f"Question {i}")
                runs.append(run)
                store.save_run(run)
                time.sleep(0.01)

            # Should only have 3 runs
            all_runs = store.list_runs(limit=10)
            assert len(all_runs) == 3

    def test_persistence_across_instances(self) -> None:
        """Test that data persists across store instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with first instance
            store1 = FileArtifactStore(artifact_dir=tmpdir)
            run = create_sample_run()
            store1.save_run(run)

            # Read with second instance
            store2 = FileArtifactStore(artifact_dir=tmpdir)
            trace = store2.get_trace(run.run_id)
            assert trace is not None
            assert trace["run_id"] == run.run_id

    def test_handles_corrupted_files(self) -> None:
        """Test graceful handling of corrupted JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(artifact_dir=tmpdir)
            run = create_sample_run()
            store.save_run(run)

            # Corrupt the trace file
            trace_path = Path(tmpdir) / run.run_id / "trace.json"
            trace_path.write_text("not valid json {{{")

            # Should return None, not crash
            assert store.get_trace(run.run_id) is None


class TestGetArtifactStore:
    """Tests for the global artifact store singleton."""

    def setup_method(self) -> None:
        """Reset the global store before each test."""
        reset_artifact_store()

    def test_default_memory_store(self) -> None:
        """Test that default store is memory-based."""
        from contextmine_core.research.artifacts import get_artifact_store

        store = get_artifact_store()
        assert isinstance(store, MemoryArtifactStore)

    def test_singleton_behavior(self) -> None:
        """Test that get_artifact_store returns the same instance."""
        from contextmine_core.research.artifacts import get_artifact_store

        store1 = get_artifact_store()
        store2 = get_artifact_store()
        assert store1 is store2
