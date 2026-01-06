"""Base class for SCIP indexer backends.

Each language has its own backend that implements the indexing logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contextmine_core.semantic_snapshot.indexers.runner import check_tool_version
from contextmine_core.semantic_snapshot.models import (
    IndexArtifact,
    IndexConfig,
    ProjectTarget,
)


class BaseIndexerBackend(ABC):
    """Base class for language-specific SCIP indexer backends.

    Subclasses implement indexing for specific languages by:
    1. Implementing can_handle() to check language compatibility
    2. Implementing index() to run the language-specific indexer
    """

    # Override in subclasses
    TOOL_NAME: str = ""

    @abstractmethod
    def can_handle(self, target: ProjectTarget) -> bool:
        """Check if this backend can handle the given project.

        Args:
            target: Project to check

        Returns:
            True if this backend can index the project
        """
        pass

    @abstractmethod
    def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
        """Run the indexer and produce a SCIP artifact.

        Args:
            target: Project to index
            cfg: Indexing configuration

        Returns:
            IndexArtifact with the result

        Raises:
            CommandNotFoundError: If the indexer tool is not installed
            CommandTimeoutError: If indexing times out
            CommandFailedError: If indexing fails
        """
        pass

    def check_tool_available(self) -> tuple[bool, str]:
        """Check if the required tool is installed and return version.

        Returns:
            Tuple of (available, version_string)
        """
        if not self.TOOL_NAME:
            return False, ""
        return check_tool_version(self.TOOL_NAME)
