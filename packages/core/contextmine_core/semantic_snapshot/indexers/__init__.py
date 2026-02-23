"""SCIP indexers module.

Provides polyglot SCIP indexing for Python, TypeScript, JavaScript, Java, and PHP.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.base import BaseIndexerBackend
from contextmine_core.semantic_snapshot.indexers.detection import (
    detect_project_at,
    detect_projects,
    detect_projects_with_diagnostics,
)
from contextmine_core.semantic_snapshot.indexers.java import JavaIndexerBackend
from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend
from contextmine_core.semantic_snapshot.indexers.python import PythonIndexerBackend
from contextmine_core.semantic_snapshot.indexers.runner import (
    CmdResult,
    CommandFailedError,
    CommandNotFoundError,
    CommandTimeoutError,
    check_tool_version,
    run_cmd,
)
from contextmine_core.semantic_snapshot.indexers.typescript import (
    TypescriptIndexerBackend,
)
from contextmine_core.semantic_snapshot.models import (
    IndexArtifact,
    IndexConfig,
    Language,
    ProjectTarget,
)

logger = logging.getLogger(__name__)

# Backend registry - order matters for language detection priority
BACKENDS: list[BaseIndexerBackend] = [
    TypescriptIndexerBackend(),
    PythonIndexerBackend(),
    JavaIndexerBackend(),
    PhpIndexerBackend(),
]

__all__ = [
    # Detection
    "detect_projects",
    "detect_project_at",
    "detect_projects_with_diagnostics",
    # Orchestration
    "index_repo",
    "index_project",
    # Backends
    "BaseIndexerBackend",
    "TypescriptIndexerBackend",
    "PythonIndexerBackend",
    "JavaIndexerBackend",
    "PhpIndexerBackend",
    "BACKENDS",
    # Runner utilities
    "run_cmd",
    "check_tool_version",
    "CmdResult",
    "CommandNotFoundError",
    "CommandTimeoutError",
    "CommandFailedError",
]


def index_repo(
    repo_root: Path | str,
    cfg: IndexConfig | None = None,
) -> list[IndexArtifact]:
    """Main entry point: detect and index all projects in a repository.

    This function:
    1. Detects all indexable projects in the repository
    2. Filters by enabled languages
    3. Runs the appropriate SCIP indexer for each project
    4. Collects and returns the results

    Args:
        repo_root: Path to the repository root
        cfg: Indexing configuration (defaults will be used if not provided)

    Returns:
        List of IndexArtifact results (both successful and failed)
    """
    repo_root = Path(repo_root)
    cfg = cfg or IndexConfig()

    # Ensure output directory exists
    if cfg.output_dir is None:
        cfg.output_dir = Path(tempfile.mkdtemp(prefix="scip_index_"))
    else:
        cfg.output_dir = Path(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Detect all projects
    projects = detect_projects(repo_root)
    logger.info("Detected %d projects in %s", len(projects), repo_root)

    # Index each project
    artifacts: list[IndexArtifact] = []
    for project in projects:
        # Skip disabled languages
        if project.language not in cfg.enabled_languages:
            logger.debug(
                "Skipping %s project (disabled): %s", project.language.value, project.root_path
            )
            continue

        artifact = index_project(project, cfg)
        artifacts.append(artifact)

        if artifact.success:
            logger.info(
                "Indexed %s project: %s (%.2fs)",
                project.language.value,
                project.root_path,
                artifact.duration_s,
            )
        else:
            logger.warning(
                "Failed to index %s project: %s - %s",
                project.language.value,
                project.root_path,
                artifact.error_message,
            )

            if not cfg.best_effort:
                # Stop on first failure
                break

    return artifacts


def index_project(target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
    """Index a single project.

    Args:
        target: Project to index
        cfg: Indexing configuration

    Returns:
        IndexArtifact with the result
    """
    # Find a backend that can handle this project
    for backend in BACKENDS:
        if backend.can_handle(target):
            try:
                return backend.index(target, cfg)
            except CommandNotFoundError as e:
                return IndexArtifact(
                    language=target.language,
                    project_root=target.root_path,
                    scip_path=Path("/dev/null"),
                    logs_path=None,
                    tool_name=backend.TOOL_NAME,
                    tool_version="unknown",
                    duration_s=0,
                    success=False,
                    error_message=str(e),
                )
            except Exception as e:
                logger.exception("Unexpected error indexing %s", target.root_path)
                return IndexArtifact(
                    language=target.language,
                    project_root=target.root_path,
                    scip_path=Path("/dev/null"),
                    logs_path=None,
                    tool_name=backend.TOOL_NAME,
                    tool_version="unknown",
                    duration_s=0,
                    success=False,
                    error_message=f"Unexpected error: {e}",
                )

    # No backend found
    return IndexArtifact(
        language=target.language,
        project_root=target.root_path,
        scip_path=Path("/dev/null"),
        logs_path=None,
        tool_name="unknown",
        tool_version="unknown",
        duration_s=0,
        success=False,
        error_message=f"No backend found for language: {target.language.value}",
    )


def get_available_tools() -> dict[Language, tuple[bool, str]]:
    """Check which SCIP tools are available.

    Returns:
        Dict mapping Language to (available, version) tuples
    """
    result: dict[Language, tuple[bool, str]] = {}

    for backend in BACKENDS:
        available, version = backend.check_tool_available()
        # Map backend to language(s) it handles
        if isinstance(backend, TypescriptIndexerBackend):
            result[Language.TYPESCRIPT] = (available, version)
            result[Language.JAVASCRIPT] = (available, version)
        elif isinstance(backend, PythonIndexerBackend):
            result[Language.PYTHON] = (available, version)
        elif isinstance(backend, JavaIndexerBackend):
            result[Language.JAVA] = (available, version)
        elif isinstance(backend, PhpIndexerBackend):
            result[Language.PHP] = (available, version)

    return result
