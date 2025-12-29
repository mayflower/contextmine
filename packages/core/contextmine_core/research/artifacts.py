"""ArtifactStore: Storage for research run artifacts."""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextmine_core.research.run import ResearchRun


@dataclass
class ArtifactMetadata:
    """Metadata for a stored artifact."""

    run_id: str
    question: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    expires_at: datetime | None


class ArtifactStore(ABC):
    """Abstract base class for artifact storage."""

    @abstractmethod
    def save_run(self, run: ResearchRun) -> None:
        """Save a research run's artifacts (trace, evidence, report)."""
        ...

    @abstractmethod
    def get_trace(self, run_id: str) -> dict | None:
        """Get the trace JSON for a run."""
        ...

    @abstractmethod
    def get_evidence(self, run_id: str) -> dict | None:
        """Get the evidence JSON for a run."""
        ...

    @abstractmethod
    def get_report(self, run_id: str) -> str | None:
        """Get the markdown report for a run."""
        ...

    @abstractmethod
    def list_runs(self, limit: int = 20) -> list[ArtifactMetadata]:
        """List recent runs, newest first."""
        ...

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete a run's artifacts. Returns True if deleted."""
        ...

    @abstractmethod
    def evict_expired(self) -> int:
        """Remove expired artifacts. Returns count of evicted runs."""
        ...

    @abstractmethod
    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        ...


class MemoryArtifactStore(ArtifactStore):
    """In-memory artifact store with TTL eviction.

    Suitable for development and testing. Data is lost on restart.
    """

    def __init__(self, ttl_minutes: int = 60, max_runs: int = 100):
        self._ttl_minutes = ttl_minutes
        self._max_runs = max_runs
        self._runs: dict[str, dict] = {}  # run_id -> {trace, evidence, report, meta}
        self._lock = threading.Lock()

    def save_run(self, run: ResearchRun) -> None:
        """Save a research run's artifacts."""

        now = datetime.now(UTC)
        expires_at = datetime.fromtimestamp(now.timestamp() + self._ttl_minutes * 60, tz=UTC)

        with self._lock:
            self._runs[run.run_id] = {
                "trace": run.to_trace_dict(),
                "evidence": run.to_evidence_dict(),
                "report": run.to_report_markdown(),
                "meta": ArtifactMetadata(
                    run_id=run.run_id,
                    question=run.question,
                    status=run.status.value,
                    created_at=run.created_at,
                    completed_at=run.completed_at,
                    expires_at=expires_at,
                ),
            }
            self._enforce_max_runs()

    def _enforce_max_runs(self) -> None:
        """Remove oldest runs if over max limit. Must hold lock."""
        if len(self._runs) > self._max_runs:
            # Sort by created_at, remove oldest
            sorted_ids = sorted(
                self._runs.keys(),
                key=lambda rid: self._runs[rid]["meta"].created_at,
            )
            to_remove = len(self._runs) - self._max_runs
            for run_id in sorted_ids[:to_remove]:
                del self._runs[run_id]

    def get_trace(self, run_id: str) -> dict | None:
        """Get the trace JSON for a run."""
        with self._lock:
            run_data = self._runs.get(run_id)
            return run_data["trace"] if run_data else None

    def get_evidence(self, run_id: str) -> dict | None:
        """Get the evidence JSON for a run."""
        with self._lock:
            run_data = self._runs.get(run_id)
            return run_data["evidence"] if run_data else None

    def get_report(self, run_id: str) -> str | None:
        """Get the markdown report for a run."""
        with self._lock:
            run_data = self._runs.get(run_id)
            return run_data["report"] if run_data else None

    def list_runs(self, limit: int = 20) -> list[ArtifactMetadata]:
        """List recent runs, newest first."""
        with self._lock:
            metas = [r["meta"] for r in self._runs.values()]
            metas.sort(key=lambda m: m.created_at, reverse=True)
            return metas[:limit]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run's artifacts."""
        with self._lock:
            if run_id in self._runs:
                del self._runs[run_id]
                return True
            return False

    def evict_expired(self) -> int:
        """Remove expired artifacts."""
        now = datetime.now(UTC)
        evicted = 0
        with self._lock:
            to_delete = []
            for run_id, run_data in self._runs.items():
                meta = run_data["meta"]
                if meta.expires_at and meta.expires_at < now:
                    to_delete.append(run_id)
            for run_id in to_delete:
                del self._runs[run_id]
                evicted += 1
        return evicted

    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        with self._lock:
            return run_id in self._runs


class FileArtifactStore(ArtifactStore):
    """File-based artifact store with TTL eviction.

    Stores artifacts in a directory structure:
    {artifact_dir}/
      {run_id}/
        trace.json
        evidence.json
        report.md
        meta.json
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        ttl_minutes: int = 60,
        max_runs: int = 100,
    ):
        self._artifact_dir = Path(artifact_dir)
        self._ttl_minutes = ttl_minutes
        self._max_runs = max_runs
        self._lock = threading.Lock()

        # Ensure directory exists
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        # Add to gitignore if in a git repo
        self._ensure_gitignore()

    def _ensure_gitignore(self) -> None:
        """Add artifact directory to .gitignore if needed."""
        # Find repository root by looking for .git directory
        current = self._artifact_dir.resolve()
        while current != current.parent:
            git_dir = current / ".git"
            if git_dir.is_dir():
                gitignore = current / ".gitignore"
                artifact_rel = self._artifact_dir.resolve().relative_to(current)
                pattern = f"/{artifact_rel}/"

                if gitignore.exists():
                    content = gitignore.read_text()
                    if pattern not in content and str(artifact_rel) not in content:
                        with gitignore.open("a") as f:
                            f.write(f"\n# Research agent artifacts\n{pattern}\n")
                break
            current = current.parent

    def _run_dir(self, run_id: str) -> Path:
        """Get the directory for a run."""
        return self._artifact_dir / run_id

    def save_run(self, run: ResearchRun) -> None:
        """Save a research run's artifacts."""
        now = datetime.now(UTC)
        expires_at = datetime.fromtimestamp(now.timestamp() + self._ttl_minutes * 60, tz=UTC)

        with self._lock:
            run_dir = self._run_dir(run.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save trace
            trace_path = run_dir / "trace.json"
            trace_path.write_text(json.dumps(run.to_trace_dict(), indent=2))

            # Save evidence
            evidence_path = run_dir / "evidence.json"
            evidence_path.write_text(json.dumps(run.to_evidence_dict(), indent=2))

            # Save report
            report_path = run_dir / "report.md"
            report_path.write_text(run.to_report_markdown())

            # Save metadata
            meta_path = run_dir / "meta.json"
            meta = {
                "run_id": run.run_id,
                "question": run.question,
                "status": run.status.value,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "expires_at": expires_at.isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2))

            self._enforce_max_runs()

    def _enforce_max_runs(self) -> None:
        """Remove oldest runs if over max limit. Must hold lock."""
        runs = list(self._artifact_dir.iterdir())
        if len(runs) <= self._max_runs:
            return

        # Sort by created_at from meta.json
        run_times: list[tuple[Path, datetime]] = []
        for run_dir in runs:
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    created_at = datetime.fromisoformat(meta["created_at"])
                    run_times.append((run_dir, created_at))
                except (json.JSONDecodeError, KeyError):
                    # Invalid meta, use mtime
                    run_times.append(
                        (run_dir, datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC))
                    )

        run_times.sort(key=lambda x: x[1])
        to_remove = len(run_times) - self._max_runs
        for run_dir, _ in run_times[:to_remove]:
            self._delete_run_dir(run_dir)

    def _delete_run_dir(self, run_dir: Path) -> None:
        """Delete a run directory and all its contents."""
        for file in run_dir.iterdir():
            file.unlink()
        run_dir.rmdir()

    def get_trace(self, run_id: str) -> dict | None:
        """Get the trace JSON for a run."""
        trace_path = self._run_dir(run_id) / "trace.json"
        if not trace_path.exists():
            return None
        try:
            return json.loads(trace_path.read_text())
        except json.JSONDecodeError:
            return None

    def get_evidence(self, run_id: str) -> dict | None:
        """Get the evidence JSON for a run."""
        evidence_path = self._run_dir(run_id) / "evidence.json"
        if not evidence_path.exists():
            return None
        try:
            return json.loads(evidence_path.read_text())
        except json.JSONDecodeError:
            return None

    def get_report(self, run_id: str) -> str | None:
        """Get the markdown report for a run."""
        report_path = self._run_dir(run_id) / "report.md"
        if not report_path.exists():
            return None
        return report_path.read_text()

    def list_runs(self, limit: int = 20) -> list[ArtifactMetadata]:
        """List recent runs, newest first."""
        metas: list[ArtifactMetadata] = []

        with self._lock:
            for run_dir in self._artifact_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                meta_path = run_dir / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    data = json.loads(meta_path.read_text())
                    metas.append(
                        ArtifactMetadata(
                            run_id=data["run_id"],
                            question=data["question"],
                            status=data["status"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            completed_at=(
                                datetime.fromisoformat(data["completed_at"])
                                if data.get("completed_at")
                                else None
                            ),
                            expires_at=(
                                datetime.fromisoformat(data["expires_at"])
                                if data.get("expires_at")
                                else None
                            ),
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

        metas.sort(key=lambda m: m.created_at, reverse=True)
        return metas[:limit]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run's artifacts."""
        with self._lock:
            run_dir = self._run_dir(run_id)
            if not run_dir.exists():
                return False
            self._delete_run_dir(run_dir)
            return True

    def evict_expired(self) -> int:
        """Remove expired artifacts."""
        now = datetime.now(UTC)
        evicted = 0

        with self._lock:
            for run_dir in list(self._artifact_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                meta_path = run_dir / "meta.json"
                if not meta_path.exists():
                    continue
                try:
                    data = json.loads(meta_path.read_text())
                    if data.get("expires_at"):
                        expires_at = datetime.fromisoformat(data["expires_at"])
                        if expires_at < now:
                            self._delete_run_dir(run_dir)
                            evicted += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        return evicted

    def run_exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return self._run_dir(run_id).exists()


# Global store instance
_store: ArtifactStore | None = None
_store_lock = threading.Lock()


def get_artifact_store() -> ArtifactStore:
    """Get the configured artifact store singleton.

    Uses settings from contextmine_core.settings if available,
    otherwise defaults to in-memory store.
    """
    global _store

    with _store_lock:
        if _store is not None:
            return _store

        # Try to load settings
        try:
            from contextmine_core.settings import get_settings

            settings = get_settings()
            store_type = getattr(settings, "artifact_store", "memory")
            ttl_minutes = getattr(settings, "artifact_ttl_minutes", 60)
            max_runs = getattr(settings, "artifact_max_runs", 100)
            artifact_dir = getattr(settings, "artifact_dir", ".mcp_artifacts")

            if store_type == "file":
                _store = FileArtifactStore(
                    artifact_dir=artifact_dir,
                    ttl_minutes=ttl_minutes,
                    max_runs=max_runs,
                )
            else:
                _store = MemoryArtifactStore(
                    ttl_minutes=ttl_minutes,
                    max_runs=max_runs,
                )
        except Exception:
            # Fallback to in-memory store
            _store = MemoryArtifactStore()

        return _store


def reset_artifact_store() -> None:
    """Reset the global artifact store. Used for testing."""
    global _store
    with _store_lock:
        _store = None
