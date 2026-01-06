"""Python SCIP indexer backend.

Uses scip-python to index Python projects.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.base import BaseIndexerBackend
from contextmine_core.semantic_snapshot.indexers.runner import (
    CommandNotFoundError,
    run_cmd,
)
from contextmine_core.semantic_snapshot.models import (
    IndexArtifact,
    IndexConfig,
    Language,
    ProjectTarget,
)

logger = logging.getLogger(__name__)


class PythonIndexerBackend(BaseIndexerBackend):
    """Backend for scip-python indexer.

    Tool: scip-python (npm install -g @sourcegraph/scip-python)

    Note: scip-python is a Node.js tool that uses Pyright for type analysis.
    """

    TOOL_NAME = "scip-python"

    def can_handle(self, target: ProjectTarget) -> bool:
        """Check if this backend can handle the project."""
        return target.language == Language.PYTHON

    def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
        """Run scip-python to index the project.

        Args:
            target: Project to index
            cfg: Indexing configuration

        Returns:
            IndexArtifact with the result
        """
        import time

        start_time = time.monotonic()

        # Determine output directory
        output_dir = cfg.output_dir or Path(tempfile.mkdtemp(prefix="scip_"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        scip_path = output_dir / f"{target.root_path.name}.scip"
        logs_path = output_dir / f"{target.root_path.name}.log"

        # Check tool availability
        available, version = self.check_tool_available()
        if not available:
            raise CommandNotFoundError(
                f"{self.TOOL_NAME} not found. Install with: npm install -g @sourcegraph/scip-python"
            )

        # Build command
        cmd = self._build_command(target, cfg)

        # Build environment
        env = dict(cfg.env_overrides)
        if cfg.node_memory_mb:
            # Set Node.js memory limit (scip-python runs on Node.js)
            node_options = env.get("NODE_OPTIONS", "")
            node_options += f" --max-old-space-size={cfg.node_memory_mb}"
            env["NODE_OPTIONS"] = node_options.strip()

        # Run indexer
        timeout = cfg.timeout_s_by_language.get(Language.PYTHON, 300)

        try:
            result = run_cmd(
                cmd=cmd,
                cwd=target.root_path,
                env=env if env else None,
                timeout_s=timeout,
                logs_path=logs_path,
            )
        except CommandNotFoundError:
            raise

        duration = time.monotonic() - start_time

        if result.timed_out:
            return IndexArtifact(
                language=target.language,
                project_root=target.root_path,
                scip_path=scip_path,
                logs_path=logs_path,
                tool_name=self.TOOL_NAME,
                tool_version=version,
                duration_s=duration,
                success=False,
                error_message=f"Timeout after {timeout}s",
            )

        if result.exit_code != 0:
            return IndexArtifact(
                language=target.language,
                project_root=target.root_path,
                scip_path=scip_path,
                logs_path=logs_path,
                tool_name=self.TOOL_NAME,
                tool_version=version,
                duration_s=duration,
                success=False,
                error_message=f"Exit code {result.exit_code}: {result.stderr_tail}",
            )

        # Check if SCIP file was created
        # scip-python outputs to index.scip by default
        default_scip = target.root_path / "index.scip"
        if default_scip.exists():
            shutil.move(str(default_scip), str(scip_path))

        if not scip_path.exists():
            return IndexArtifact(
                language=target.language,
                project_root=target.root_path,
                scip_path=scip_path,
                logs_path=logs_path,
                tool_name=self.TOOL_NAME,
                tool_version=version,
                duration_s=duration,
                success=False,
                error_message="SCIP file was not created",
            )

        return IndexArtifact(
            language=target.language,
            project_root=target.root_path,
            scip_path=scip_path,
            logs_path=logs_path,
            tool_name=self.TOOL_NAME,
            tool_version=version,
            duration_s=duration,
            success=True,
        )

    def _build_command(self, target: ProjectTarget, cfg: IndexConfig) -> list[str]:
        """Build the scip-python command."""
        cmd = [self.TOOL_NAME, "index", "."]

        # Add project name
        cmd.extend(["--project-name", cfg.project_name])

        # Add project version if set
        if cfg.project_version:
            cmd.extend(["--project-version", cfg.project_version])

        return cmd
