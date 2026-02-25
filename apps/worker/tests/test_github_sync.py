"""Tests for git-derived sync helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextmine_worker.github_sync import clone_or_pull_repo, compute_git_evolution_snapshots
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


def _create_remote_origin(tmp_path: Path) -> tuple[Repo, Path]:
    remote_worktree = tmp_path / "remote_worktree"
    remote_worktree.mkdir(parents=True, exist_ok=True)
    remote_repo = Repo.init(remote_worktree)
    with remote_repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")

    readme = remote_worktree / "README.md"
    readme.write_text("initial\n", encoding="utf-8")
    remote_repo.index.add(["README.md"])
    remote_repo.index.commit("initial commit")
    remote_repo.git.branch("-M", "main")

    bare_origin = tmp_path / "origin.git"
    Repo.init(bare_origin, bare=True)
    remote_repo.create_remote("origin", str(bare_origin))
    remote_repo.git.push("--set-upstream", "origin", "main")
    return remote_repo, bare_origin


def test_clone_or_pull_repo_resets_divergence_and_cleans_worktree(tmp_path: Path) -> None:
    remote_repo, origin_path = _create_remote_origin(tmp_path)
    local_path = tmp_path / "local_clone"

    clone_or_pull_repo(local_path, str(origin_path), branch="main")
    local_repo = Repo(local_path)

    # Simulate indexer-generated artifacts and a local divergent commit.
    generated_dir = local_path / "node_modules"
    generated_dir.mkdir(parents=True, exist_ok=True)
    (generated_dir / "temp.js").write_text("console.log('temp')\n", encoding="utf-8")
    (local_path / "README.md").write_text("local divergence\n", encoding="utf-8")
    local_repo.index.add(["README.md"])
    local_repo.index.commit("local divergence")

    # Upstream advances independently.
    remote_readme = Path(remote_repo.working_tree_dir or "") / "README.md"
    remote_readme.write_text("upstream update\n", encoding="utf-8")
    remote_repo.index.add(["README.md"])
    upstream_commit = remote_repo.index.commit("upstream update")
    remote_repo.git.push("origin", "main")

    refreshed_repo = clone_or_pull_repo(local_path, str(origin_path), branch="main")

    assert refreshed_repo.head.commit.hexsha == upstream_commit.hexsha
    assert not generated_dir.exists()
    assert (local_path / "README.md").read_text(encoding="utf-8") == "upstream update\n"


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
