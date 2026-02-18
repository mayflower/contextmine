"""GitHub repository sync service."""

import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

logger = logging.getLogger(__name__)

# File extensions to index (code + docs)
ALLOWED_EXTENSIONS = {
    # Documentation
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".adoc",
    # Code
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".graphql",
    ".proto",
    ".dockerfile",
    # Config
    ".env.example",
    ".gitignore",
    ".dockerignore",
}

# Special filenames to always include
ALLOWED_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "CMakeLists.txt",
    "README",
    "LICENSE",
    "CHANGELOG",
    "CONTRIBUTING",
}

# Maximum file size in bytes (1MB)
MAX_FILE_SIZE = 1024 * 1024

# Base path for cloned repos
REPOS_BASE_PATH = Path("/data/repos")


@dataclass
class SyncStats:
    """Statistics from a sync operation."""

    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    docs_created: int = 0
    docs_updated: int = 0
    docs_deleted: int = 0


def is_eligible_file(file_path: Path, repo_root: Path) -> bool:
    """Check if a file should be indexed."""
    # Check extension
    ext = file_path.suffix.lower()
    name = file_path.name

    if ext not in ALLOWED_EXTENSIONS and name not in ALLOWED_FILENAMES:
        return False

    # Check file size
    full_path = repo_root / file_path
    if full_path.exists():
        try:
            if full_path.stat().st_size > MAX_FILE_SIZE:
                return False
        except OSError:
            return False

    # Skip hidden files and directories
    for part in file_path.parts:
        if part.startswith(".") and part not in {".github", ".gitlab"}:
            return False

    # Skip common non-content directories
    skip_dirs = {"node_modules", "vendor", "dist", "build", "__pycache__", ".git", "venv", ".venv"}
    return not any(part in skip_dirs for part in file_path.parts)


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_file_title(file_path: Path) -> str:
    """Generate a title for a file."""
    return str(file_path)


def build_uri(owner: str, repo: str, file_path: str, branch: str) -> str:
    """Build a URI for a document."""
    return f"git://github.com/{owner}/{repo}/{file_path}?ref={branch}"


def https_url_to_ssh(https_url: str) -> str:
    """Convert HTTPS GitHub URL to SSH URL.

    Args:
        https_url: URL like https://github.com/owner/repo.git

    Returns:
        SSH URL like git@github.com:owner/repo.git
    """
    # Remove https:// prefix and any auth
    url = https_url.replace("https://", "")
    if "@" in url:
        url = url.split("@", 1)[1]

    # github.com/owner/repo.git -> git@github.com:owner/repo.git
    if url.startswith("github.com/"):
        path = url[len("github.com/") :]
        return f"git@github.com:{path}"

    return https_url  # Fallback


def clone_or_pull_repo(
    repo_path: Path,
    clone_url: str,
    branch: str,
    token: str | None = None,
    ssh_private_key: str | None = None,
) -> Repo:
    """Clone a repo or pull latest changes.

    Args:
        repo_path: Local path for the repo
        clone_url: HTTPS URL of the repo
        branch: Branch to checkout
        token: Optional GitHub token for private repos (used if no SSH key)
        ssh_private_key: Optional SSH private key for private repos (preferred)

    Returns:
        Git Repo object
    """
    env = os.environ.copy()
    key_file_path: str | None = None

    try:
        # If SSH key provided, set up SSH authentication
        if ssh_private_key:
            # Write key to a temporary file (GitPython needs a file path)
            fd, key_file_path = tempfile.mkstemp(prefix="deploy_key_", suffix=".pem")
            with os.fdopen(fd, "w") as f:
                f.write(ssh_private_key)
                if not ssh_private_key.endswith("\n"):
                    f.write("\n")
            os.chmod(key_file_path, 0o600)

            # Set GIT_SSH_COMMAND to use this key
            # StrictHostKeyChecking=no to avoid interactive prompts
            # UserKnownHostsFile=/dev/null to avoid host key issues
            env["GIT_SSH_COMMAND"] = (
                f"ssh -i {key_file_path} "
                f"-o StrictHostKeyChecking=accept-new "
                f"-o UserKnownHostsFile=/dev/null"
            )

            # Convert HTTPS URL to SSH URL
            clone_url = https_url_to_ssh(clone_url)
        elif token and clone_url.startswith("https://"):
            # Add token to URL if provided
            # Insert token into URL: https://token@github.com/...
            clone_url = clone_url.replace("https://", f"https://{token}@")

        if repo_path.exists():
            # Pull latest
            repo = Repo(repo_path)
            origin = repo.remotes.origin

            # Update remote URL
            origin.set_url(clone_url)

            # Fetch and checkout branch with custom environment
            with repo.git.custom_environment(**env):
                origin.fetch()
                repo.git.checkout(branch)
                repo.git.pull("origin", branch)
        else:
            # Clone
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            repo = Repo.clone_from(clone_url, repo_path, branch=branch, env=env)

        return repo

    finally:
        # Clean up the temporary key file
        if key_file_path and os.path.exists(key_file_path):
            os.unlink(key_file_path)


def get_changed_files(
    repo: Repo,
    old_sha: str | None,
    new_sha: str,
) -> tuple[list[str], list[str]]:
    """Get list of changed and deleted files between commits.

    Args:
        repo: Git repo object
        old_sha: Previous commit SHA (None for full index)
        new_sha: Current commit SHA

    Returns:
        Tuple of (changed_files, deleted_files)
    """
    if old_sha is None:
        # Full index - get all files in current tree
        changed = []
        for item in repo.commit(new_sha).tree.traverse():
            if item.type == "blob":  # type: ignore[union-attr]
                changed.append(item.path)  # type: ignore[union-attr]
        return changed, []

    # Incremental - diff between commits
    try:
        diff = repo.commit(old_sha).diff(repo.commit(new_sha))
    except GitCommandError:
        # Old commit might be gone (force push), do full index
        changed = []
        for item in repo.commit(new_sha).tree.traverse():
            if item.type == "blob":  # type: ignore[union-attr]
                changed.append(item.path)  # type: ignore[union-attr]
        return changed, []

    changed = []
    deleted = []

    for diff_item in diff:
        if diff_item.deleted_file:
            if diff_item.a_path:
                deleted.append(diff_item.a_path)
        elif diff_item.new_file or diff_item.renamed_file or diff_item.a_blob != diff_item.b_blob:
            if diff_item.b_path:
                changed.append(diff_item.b_path)
            # Handle renamed files - delete old path
            if diff_item.renamed_file and diff_item.a_path:
                deleted.append(diff_item.a_path)

    return changed, deleted


def compute_git_change_metrics(repo: Repo, target_files: set[str]) -> dict[str, dict[str, int]]:
    """Compute git-derived change frequency and churn for target files.

    Returns:
        Mapping of file path -> metrics:
            - change_frequency: non-merge commit touch count
            - insertions: total insertions across matching commits
            - deletions: total deletions across matching commits
            - churn: total insertions + deletions
    """
    if not target_files:
        return {}

    metrics: dict[str, dict[str, int]] = {
        path: {"change_frequency": 0, "insertions": 0, "deletions": 0, "churn": 0}
        for path in target_files
    }

    try:
        for commit in repo.iter_commits("HEAD", no_merges=True):
            files = commit.stats.files or {}
            for file_path, file_stats in files.items():
                if file_path not in metrics:
                    continue

                insertions = int(file_stats.get("insertions", 0) or 0)
                deletions = int(file_stats.get("deletions", 0) or 0)

                entry = metrics[file_path]
                entry["change_frequency"] += 1
                entry["insertions"] += insertions
                entry["deletions"] += deletions
                entry["churn"] += insertions + deletions
    except Exception:
        logger.warning("Failed to compute git change metrics; defaulting to zeros", exc_info=True)
        return {
            path: {"change_frequency": 0, "insertions": 0, "deletions": 0, "churn": 0}
            for path in target_files
        }

    return metrics


def read_file_content(repo_path: Path, file_path: str) -> str | None:
    """Read file content, returning None if not readable."""
    full_path = repo_path / file_path
    try:
        content = full_path.read_text(encoding="utf-8")
        return content
    except (OSError, UnicodeDecodeError):
        return None


def cleanup_repo(repo_path: Path) -> None:
    """Remove a cloned repository."""
    if repo_path.exists():
        shutil.rmtree(repo_path)


def ensure_repos_dir() -> None:
    """Ensure the repos directory exists."""
    REPOS_BASE_PATH.mkdir(parents=True, exist_ok=True)


def get_repo_path(source_id: str) -> Path:
    """Get the local path for a source's repo."""
    return REPOS_BASE_PATH / source_id
