"""Tree-sitter Manager: Singleton manager for parser lifecycle and caching.

The TreeSitterManager handles:
- Lazy initialization of parsers per language
- LRU caching of parsed trees with content-hash invalidation
- Graceful degradation when tree-sitter is unavailable
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from contextmine_core.treesitter.languages import TreeSitterLanguage, detect_language

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CachedTree:
    """A cached parsed tree."""

    tree: Any  # tree_sitter.Tree
    content_hash: str
    file_path: str


class TreeSitterManager:
    """Manages tree-sitter parsers and cached parse trees.

    This is a singleton that:
    - Lazily loads parsers per language
    - Caches parsed trees with LRU eviction
    - Uses content-hash for cache invalidation
    - Handles graceful degradation if tree-sitter unavailable
    """

    _instance: TreeSitterManager | None = None
    _lock = threading.Lock()

    def __init__(self, cache_size: int = 100):
        """Initialize the manager.

        Args:
            cache_size: Maximum number of parsed trees to cache
        """
        self._cache_size = cache_size
        self._parsers: dict[TreeSitterLanguage, Any] = {}  # language -> Parser
        self._trees: OrderedDict[str, CachedTree] = OrderedDict()  # file_path -> CachedTree
        self._parsers_lock = threading.Lock()
        self._available: bool | None = None

    @classmethod
    def get_instance(cls, cache_size: int = 100) -> TreeSitterManager:
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_size=cache_size)
        assert cls._instance is not None
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. For testing only."""
        with cls._lock:
            cls._instance = None

    def is_available(self) -> bool:
        """Check if tree-sitter is available."""
        if self._available is None:
            try:
                import tree_sitter_language_pack  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "tree-sitter-language-pack not installed. "
                    "Tree-sitter features will be disabled."
                )
        return self._available

    def get_parser(self, language: TreeSitterLanguage) -> Any:
        """Get a parser for the specified language.

        Args:
            language: The programming language

        Returns:
            tree_sitter.Parser configured for the language

        Raises:
            ImportError: If tree-sitter-language-pack is not installed
            ValueError: If language is not supported
        """
        if not self.is_available():
            raise ImportError(
                "tree-sitter-language-pack not installed. "
                "Run: pip install tree-sitter-language-pack"
            )

        with self._parsers_lock:
            if language not in self._parsers:
                self._parsers[language] = self._create_parser(language)
            return self._parsers[language]

    def _create_parser(self, language: TreeSitterLanguage) -> Any:
        """Create a parser for the specified language.

        Args:
            language: The programming language

        Returns:
            Configured Parser instance
        """
        from tree_sitter_language_pack import get_parser

        # tree-sitter-language-pack uses language names directly
        return get_parser(language.value)

    def parse(
        self,
        file_path: str | Path,
        content: str | None = None,
        force_reparse: bool = False,
    ) -> Any:
        """Parse a file and return its syntax tree.

        Uses cached tree if content hasn't changed.

        Args:
            file_path: Path to the source file
            content: Optional file content (reads from file if None)
            force_reparse: Force reparsing even if cached

        Returns:
            tree_sitter.Tree

        Raises:
            ImportError: If tree-sitter not available
            ValueError: If language not supported
            FileNotFoundError: If file doesn't exist and no content provided
        """
        file_path = str(Path(file_path).resolve())

        # Read content if not provided
        if content is None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            content = path.read_text(encoding="utf-8", errors="replace")

        # Compute content hash for cache invalidation
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Check cache
        if not force_reparse and file_path in self._trees:
            cached = self._trees[file_path]
            if cached.content_hash == content_hash:
                # Move to end (LRU)
                self._trees.move_to_end(file_path)
                return cached.tree

        # Detect language
        language = detect_language(file_path)
        if language is None:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        # Parse
        parser = self.get_parser(language)
        tree = parser.parse(content.encode("utf-8"))

        # Cache the result
        self._trees[file_path] = CachedTree(
            tree=tree,
            content_hash=content_hash,
            file_path=file_path,
        )

        # Evict oldest if over capacity
        while len(self._trees) > self._cache_size:
            self._trees.popitem(last=False)

        return tree

    def invalidate(self, file_path: str | Path) -> None:
        """Remove a file from the cache.

        Args:
            file_path: Path to the file to invalidate
        """
        file_path = str(Path(file_path).resolve())
        self._trees.pop(file_path, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._trees.clear()

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_trees": len(self._trees),
            "max_size": self._cache_size,
            "parsers_loaded": len(self._parsers),
        }


def get_treesitter_manager() -> TreeSitterManager:
    """Get the global TreeSitterManager instance.

    Returns:
        The singleton TreeSitterManager
    """
    return TreeSitterManager.get_instance()
