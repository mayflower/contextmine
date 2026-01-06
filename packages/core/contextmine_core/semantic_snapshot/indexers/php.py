"""PHP SCIP indexer backend.

Uses scip-php to index PHP projects.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.base import BaseIndexerBackend
from contextmine_core.semantic_snapshot.indexers.runner import (
    CmdResult,
    CommandNotFoundError,
    run_cmd,
)
from contextmine_core.semantic_snapshot.models import (
    IndexArtifact,
    IndexConfig,
    InstallDepsMode,
    Language,
    ProjectTarget,
)

logger = logging.getLogger(__name__)


class PhpIndexerBackend(BaseIndexerBackend):
    """Backend for scip-php indexer.

    Requires composer.json + composer.lock to be present.

    Tool: scip-php (install via composer global require)
    """

    TOOL_NAME = "scip-php"

    def can_handle(self, target: ProjectTarget) -> bool:
        """Check if this backend can handle the project."""
        return target.language == Language.PHP

    def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
        """Run scip-php to index the project.

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
                f"{self.TOOL_NAME} not found. Install with: composer global require nicosantangelo/scip-php"
            )

        # Install dependencies if needed
        if self._should_install_deps(target, cfg):
            self._install_deps(target, cfg.timeout_s_by_language.get(Language.PHP, 300))

        # Build command
        cmd = self._build_command(target)

        # Run indexer
        timeout = cfg.timeout_s_by_language.get(Language.PHP, 300)

        try:
            result = run_cmd(
                cmd=cmd,
                cwd=target.root_path,
                env=dict(cfg.env_overrides) if cfg.env_overrides else None,
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

    def _build_command(self, target: ProjectTarget) -> list[str]:
        """Build the scip-php command."""
        return [self.TOOL_NAME]

    def _should_install_deps(self, target: ProjectTarget, cfg: IndexConfig) -> bool:
        """Check if dependencies should be installed."""
        if cfg.install_deps_mode == InstallDepsMode.NEVER:
            return False

        if cfg.install_deps_mode == InstallDepsMode.ALWAYS:
            return True

        # AUTO mode: check if vendor exists
        vendor = target.root_path / "vendor"
        return not vendor.exists()

    def _install_deps(self, target: ProjectTarget, timeout_s: int) -> CmdResult:
        """Install project dependencies using composer.

        IMPORTANT: Uses composer install, NOT composer require.
        This respects the lockfile and doesn't modify composer.json.
        """
        cmd = [
            "composer",
            "install",
            "--no-interaction",
            "--prefer-dist",
            "--no-progress",
        ]

        logger.info("Installing dependencies with composer in %s", target.root_path)
        return run_cmd(cmd=cmd, cwd=target.root_path, timeout_s=timeout_s)
