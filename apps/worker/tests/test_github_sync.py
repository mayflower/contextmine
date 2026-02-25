"""Tests for git-derived sync helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextmine_worker.github_sync import compute_git_evolution_snapshots
from git import Repo


def _create_repo_with_commit(tmp_path: Path) -> Repo:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    file_a = src_dir / "a.py"
    file_b = src_dir / "b.py"
    file_a.write_text("print('a')\n", encoding="utf-8")
    file_b.write_text("print('b')\n", encoding="utf-8")
    repo.index.add(["src/a.py", "src/b.py"])
    repo.index.commit("seed")
    return repo


def test_compute_git_evolution_snapshots_falls_back_when_since_filter_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _create_repo_with_commit(tmp_path)
    original_iter_commits = repo.iter_commits

    def _iter_commits(*args: object, **kwargs: object):
        if "since" in kwargs:
            return []
        return original_iter_commits(*args, **kwargs)

    monkeypatch.setattr(repo, "iter_commits", _iter_commits)

    payload = compute_git_evolution_snapshots(
        repo,
        {"src/a.py", "src/b.py"},
        window_days=365,
    )

    assert "git_since_filter_fallback_used" in payload["warnings"]
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert int(stats.get("commits_considered", 0)) >= 1
    assert int(stats.get("commits_scanned", 0)) >= 1
    assert int(stats.get("ownership_rows", 0)) >= 1


def test_compute_git_evolution_snapshots_marks_empty_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _create_repo_with_commit(tmp_path)

    def _iter_commits(*args: object, **kwargs: object):
        return []

    monkeypatch.setattr(repo, "iter_commits", _iter_commits)

    payload = compute_git_evolution_snapshots(
        repo,
        {"src/a.py"},
        window_days=365,
    )

    assert "no_commits_in_window" in payload["warnings"]
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert int(stats.get("commits_considered", 0)) == 0
    assert int(stats.get("commits_scanned", 0)) == 0
