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

from contextmine_core.semantic_snapshot.lsif import (
    LSIFProvider,
    build_snapshot_lsif,
)
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
    "LSIFProvider",
    # Entry functions
    "build_snapshot",
    "build_snapshot_scip",
    "build_snapshot_lsif",
    # Indexer functions (lazy import)
    "detect_projects",
    "index_repo",
]


def build_snapshot(index_path: Path | str) -> Snapshot:
    """Build a semantic snapshot from SCIP or LSIF index files.

    Supported:
    - `.scip` protobuf files
    - LSIF JSON/JSONL (`.lsif`, `.json`, `.jsonl`)
    """
    path = Path(index_path)
    suffix = path.suffix.lower()
    if suffix == ".scip":
        return build_snapshot_scip(path)
    if suffix in {".lsif", ".json", ".jsonl"}:
        return build_snapshot_lsif(path)
    raise ValueError(f"Unsupported semantic index format: {path}")


def detect_projects(repo_root: Path | str) -> list[ProjectTarget]:
    """Detect all indexable projects in a repository.

    Uses a language census (cloc-first) to identify all involved languages,
    then uses repository markers only to refine project roots.

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
