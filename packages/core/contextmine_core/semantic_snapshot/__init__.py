"""Semantic Snapshot: Language-agnostic code intelligence layer.

This module provides a unified interface for extracting semantic information
from source code using SCIP (Sourcegraph Code Intelligence Protocol).

SCIP provides fully resolved cross-file semantic analysis from language-specific
indexers:
- scip-python for Python
- scip-typescript for TypeScript/JavaScript
- scip-java for Java
- scip-php for PHP

The Snapshot model is stable and serializable, allowing semantic information
to be persisted and used by downstream consumers (knowledge graph, GraphRAG).
"""

from __future__ import annotations

import logging
from pathlib import Path

from contextmine_core.semantic_snapshot.models import (
    # Snapshot models
    FileInfo,
    IndexArtifact,
    IndexConfig,
    InstallDepsMode,
    Language,
    Occurrence,
    OccurrenceRole,
    ProjectTarget,
    Range,
    Relation,
    RelationKind,
    Snapshot,
    Symbol,
    SymbolKind,
)
from contextmine_core.semantic_snapshot.scip import (
    SCIPProvider,
    build_snapshot_scip,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Snapshot Models
    "FileInfo",
    "Symbol",
    "SymbolKind",
    "Range",
    "Occurrence",
    "OccurrenceRole",
    "Relation",
    "RelationKind",
    "Snapshot",
    # Indexer Models
    "Language",
    "ProjectTarget",
    "IndexArtifact",
    "IndexConfig",
    "InstallDepsMode",
    # SCIP Provider
    "SCIPProvider",
    # Entry functions
    "build_snapshot",
    "build_snapshot_scip",
    # Indexer functions (lazy import)
    "detect_projects",
    "index_repo",
]


def build_snapshot(scip_path: Path | str) -> Snapshot:
    """Build a semantic snapshot from a SCIP index file.

    This is the primary entry point for parsing SCIP indexes.
    SCIP files are produced by running language-specific indexers
    via index_repo() or external tools.

    Args:
        scip_path: Path to the .scip index file

    Returns:
        Snapshot containing semantic information

    Raises:
        FileNotFoundError: If SCIP file doesn't exist
    """
    return build_snapshot_scip(scip_path)


def detect_projects(repo_root: Path | str) -> list[ProjectTarget]:
    """Detect all indexable projects in a repository.

    Scans for language-specific marker files:
    - Python: pyproject.toml / setup.cfg / requirements.txt
    - TypeScript: package.json + tsconfig.json
    - JavaScript: package.json (no tsconfig)
    - Java: pom.xml / build.gradle
    - PHP: composer.json + composer.lock

    Args:
        repo_root: Path to the repository root

    Returns:
        List of detected ProjectTarget objects
    """
    from contextmine_core.semantic_snapshot.indexers import (
        detect_projects as _detect_projects,
    )

    return _detect_projects(repo_root)


def index_repo(
    repo_root: Path | str,
    cfg: IndexConfig | None = None,
) -> list[IndexArtifact]:
    """Index all projects in a repository using SCIP indexers.

    This function:
    1. Detects all indexable projects
    2. Runs the appropriate SCIP indexer for each
    3. Returns IndexArtifact results

    Args:
        repo_root: Path to the repository root
        cfg: Indexing configuration (uses defaults if not provided)

    Returns:
        List of IndexArtifact results
    """
    from contextmine_core.semantic_snapshot.indexers import (
        index_repo as _index_repo,
    )

    return _index_repo(repo_root, cfg)
