from datetime import UTC, datetime, timedelta
from pathlib import Path

from contextmine_worker.github_sync import compute_git_change_metrics
from git import Actor, Repo


def _commit_file(repo: Repo, relative_path: str, content: str, when: datetime) -> None:
    file_path = Path(repo.working_tree_dir or ".") / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    repo.index.add([relative_path])
    actor = Actor("ContextMine Test", "test@example.com")
    timestamp = when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S%z")
    repo.index.commit(
        f"update {relative_path}",
        author=actor,
        committer=actor,
        author_date=timestamp,
        commit_date=timestamp,
    )


def test_compute_git_change_metrics_respects_since_window(tmp_path: Path) -> None:
    repo = Repo.init(tmp_path)
    now = datetime.now(UTC)

    _commit_file(repo, "src/example.py", "print('old')\n", now - timedelta(days=30))
    _commit_file(repo, "src/example.py", "print('new')\n", now - timedelta(days=1))

    all_history = compute_git_change_metrics(repo, {"src/example.py"})
    recent_window = compute_git_change_metrics(repo, {"src/example.py"}, since_days=7)

    assert all_history["src/example.py"]["change_frequency"] == 2
    assert recent_window["src/example.py"]["change_frequency"] == 1
    assert recent_window["src/example.py"]["churn"] <= all_history["src/example.py"]["churn"]
