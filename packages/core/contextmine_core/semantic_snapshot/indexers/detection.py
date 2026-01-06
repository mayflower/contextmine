"""Project detection for SCIP indexing.

Detects projects by looking for language-specific marker files.
Supports monorepos with multiple project roots.
"""

from __future__ import annotations

import logging
from pathlib import Path

from contextmine_core.semantic_snapshot.models import Language, ProjectTarget

logger = logging.getLogger(__name__)

# Directories to skip during traversal
IGNORE_DIRS = {
    "node_modules",
    "vendor",
    ".git",
    "dist",
    "build",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    "egg-info",
    ".gradle",
    ".m2",
    ".idea",
    ".vscode",
}


def detect_projects(repo_root: Path | str) -> list[ProjectTarget]:
    """Detect all indexable projects in a repository.

    Scans the repository for language-specific marker files:
    - Python: pyproject.toml / setup.cfg / setup.py / requirements.txt
    - TypeScript: package.json + tsconfig.json
    - JavaScript: package.json (no tsconfig.json)
    - Java: pom.xml / build.gradle / build.gradle.kts / build.sbt
    - PHP: composer.json + composer.lock

    Monorepo support:
    - Multiple project roots are allowed
    - Nested projects: prefer deepest match (nearest to source files)
    - Deduplicates overlapping detections

    Args:
        repo_root: Path to the repository root

    Returns:
        List of ProjectTarget objects, sorted by path depth (deepest first)
    """
    repo_root = Path(repo_root)
    projects: list[ProjectTarget] = []

    # Track detected roots to avoid duplicates
    seen_roots: set[Path] = set()

    def should_skip_dir(dir_path: Path) -> bool:
        """Check if directory should be skipped."""
        return dir_path.name in IGNORE_DIRS or dir_path.name.startswith(".")

    def walk_directory(current: Path) -> None:
        """Recursively walk directory tree."""
        if should_skip_dir(current):
            return

        # Check for project markers in this directory
        project = detect_project_at(current)
        if project and current not in seen_roots:
            projects.append(project)
            seen_roots.add(current)

        # Continue walking subdirectories
        try:
            for child in current.iterdir():
                if child.is_dir():
                    walk_directory(child)
        except PermissionError:
            logger.debug("Permission denied: %s", current)

    walk_directory(repo_root)

    # Sort by path depth (deepest first) for monorepo handling
    # This ensures nested projects are processed before their parents
    projects.sort(key=lambda p: len(p.root_path.parts), reverse=True)

    return projects


def detect_project_at(directory: Path) -> ProjectTarget | None:
    """Detect a project at a specific directory.

    Args:
        directory: Directory to check

    Returns:
        ProjectTarget if a project is detected, None otherwise
    """
    # Check each language in order of specificity
    # Python
    project = _detect_python(directory)
    if project:
        return project

    # TypeScript/JavaScript (check TS first as it's more specific)
    project = _detect_typescript_or_javascript(directory)
    if project:
        return project

    # Java
    project = _detect_java(directory)
    if project:
        return project

    # PHP
    project = _detect_php(directory)
    if project:
        return project

    return None


def _detect_python(directory: Path) -> ProjectTarget | None:
    """Detect a Python project.

    Markers (any of):
    - pyproject.toml
    - setup.cfg
    - setup.py
    - requirements.txt
    """
    has_pyproject = (directory / "pyproject.toml").exists()
    has_setup_cfg = (directory / "setup.cfg").exists()
    has_setup_py = (directory / "setup.py").exists()
    has_requirements = (directory / "requirements.txt").exists()

    if has_pyproject or has_setup_cfg or has_setup_py or has_requirements:
        return ProjectTarget(
            language=Language.PYTHON,
            root_path=directory,
            metadata={
                "has_pyproject": has_pyproject,
                "has_setup_cfg": has_setup_cfg,
                "has_setup_py": has_setup_py,
                "has_requirements": has_requirements,
            },
        )
    return None


def _detect_typescript_or_javascript(directory: Path) -> ProjectTarget | None:
    """Detect a TypeScript or JavaScript project.

    TypeScript: package.json + tsconfig.json
    JavaScript: package.json (no tsconfig.json)
    """
    has_package_json = (directory / "package.json").exists()
    if not has_package_json:
        return None

    has_tsconfig = (directory / "tsconfig.json").exists()

    # Detect package manager
    package_manager = "npm"  # default
    if (directory / "pnpm-lock.yaml").exists():
        package_manager = "pnpm"
    elif (directory / "yarn.lock").exists():
        package_manager = "yarn"
    elif (directory / "bun.lockb").exists():
        package_manager = "bun"

    if has_tsconfig:
        return ProjectTarget(
            language=Language.TYPESCRIPT,
            root_path=directory,
            metadata={
                "has_tsconfig": True,
                "package_manager": package_manager,
            },
        )
    else:
        return ProjectTarget(
            language=Language.JAVASCRIPT,
            root_path=directory,
            metadata={
                "has_tsconfig": False,
                "package_manager": package_manager,
            },
        )


def _detect_java(directory: Path) -> ProjectTarget | None:
    """Detect a Java project.

    Markers (any of):
    - pom.xml (Maven)
    - build.gradle (Gradle)
    - build.gradle.kts (Gradle Kotlin DSL)
    - build.sbt (sbt)
    """
    has_pom = (directory / "pom.xml").exists()
    has_gradle = (directory / "build.gradle").exists()
    has_gradle_kts = (directory / "build.gradle.kts").exists()
    has_sbt = (directory / "build.sbt").exists()

    if has_pom or has_gradle or has_gradle_kts or has_sbt:
        # Determine build tool
        if has_pom:
            build_tool = "maven"
        elif has_gradle or has_gradle_kts:
            build_tool = "gradle"
        elif has_sbt:
            build_tool = "sbt"
        else:
            build_tool = "unknown"

        return ProjectTarget(
            language=Language.JAVA,
            root_path=directory,
            metadata={
                "build_tool": build_tool,
                "has_pom": has_pom,
                "has_gradle": has_gradle,
                "has_gradle_kts": has_gradle_kts,
                "has_sbt": has_sbt,
            },
        )
    return None


def _detect_php(directory: Path) -> ProjectTarget | None:
    """Detect a PHP project.

    Requires both:
    - composer.json
    - composer.lock
    """
    has_composer_json = (directory / "composer.json").exists()
    has_composer_lock = (directory / "composer.lock").exists()

    # Require both for a proper PHP project
    if has_composer_json and has_composer_lock:
        return ProjectTarget(
            language=Language.PHP,
            root_path=directory,
            metadata={
                "has_composer_lock": True,
            },
        )
    return None
