"""Tests for git-derived sync helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextmine_worker.github_sync import (
    _load_git_numstat_commits,
    clone_or_pull_repo,
    compute_git_evolution_snapshots,
    is_eligible_file,
)
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
    monkeypatch.setattr(
        "contextmine_worker.github_sync._load_git_numstat_commits",
        lambda *_args, **_kwargs: None,
    )
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
    monkeypatch.setattr(
        "contextmine_worker.github_sync._load_git_numstat_commits",
        lambda *_args, **_kwargs: None,
    )

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


def test_compute_git_evolution_snapshots_caps_pairing_for_large_commits(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")

    target_files: set[str] = set()
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(6):
        file_path = src_dir / f"f{idx}.py"
        file_path.write_text(f"print('{idx}')\n", encoding="utf-8")
        target_files.add(f"src/f{idx}.py")
    repo.index.add(sorted(target_files))
    repo.index.commit("touch many files")

    payload = compute_git_evolution_snapshots(
        repo,
        target_files,
        window_days=365,
        max_files_per_commit=3,
    )

    warnings = payload["warnings"]
    assert isinstance(warnings, list)
    assert any(
        str(item).startswith("temporal_coupling_pairing_capped:") for item in warnings
    ), warnings

    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert int(stats.get("max_files_per_commit", 0)) == 3
    assert int(stats.get("pairing_truncated_commits", 0)) == 1
    assert int(stats.get("ownership_rows", 0)) == 6

    file_rows = [
        row
        for row in payload["coupling_rows"]
        if isinstance(row, dict) and row.get("entity_level") == "file"
    ]
    assert len(file_rows) == 3


def test_load_git_numstat_commits_uses_pathspec_filters() -> None:
    class _DummyGit:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def log(self, *args: object) -> str:
            self.calls.append(args)
            return ""

    class _DummyRepo:
        def __init__(self) -> None:
            self.git = _DummyGit()

    repo = _DummyRepo()
    result = _load_git_numstat_commits(repo, {"src/a.py", "src/b.py"}, since_days=7)

    assert result == ([], False)
    assert repo.git.calls
    first_call = repo.git.calls[0]
    assert "--" in first_call
    assert "src/a.py" in first_call
    assert "src/b.py" in first_call


def test_is_eligible_file_skips_custom_composer_vendor_dir(tmp_path: Path) -> None:
    composer_json = tmp_path / "composer.json"
    composer_json.write_text(
        '{"config":{"vendor-dir":"src/libs"}}',
        encoding="utf-8",
    )
    lib_file = tmp_path / "src" / "libs" / "vendor_file.php"
    lib_file.parent.mkdir(parents=True, exist_ok=True)
    lib_file.write_text("<?php echo 'vendor';", encoding="utf-8")

    app_file = tmp_path / "src" / "app" / "main.php"
    app_file.parent.mkdir(parents=True, exist_ok=True)
    app_file.write_text("<?php echo 'app';", encoding="utf-8")

    assert not is_eligible_file(Path("src/libs/vendor_file.php"), tmp_path)
    assert is_eligible_file(Path("src/app/main.php"), tmp_path)
