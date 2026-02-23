"""PHP SCIP indexer backend.

Uses scip-php to index PHP projects.
"""

from __future__ import annotations

import json
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
    _FALLBACK_TOOL_PATHS = (
        Path("/opt/composer/vendor/bin/scip-php"),
        Path.home() / ".composer" / "vendor" / "bin" / "scip-php",
    )

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

        # Resolve tool path up front so PATH issues do not break indexing.
        tool_path = self._resolve_tool_path()
        if tool_path is None:
            raise CommandNotFoundError(
                f"{self.TOOL_NAME} not found. Install with: composer global require davidrjenni/scip-php"
            )
        available, version = self._check_tool_available(tool_path)
        if not available:
            raise CommandNotFoundError(f"{self.TOOL_NAME} is not executable at {tool_path}")

        # Install dependencies if needed
        if self._should_install_deps(target, cfg):
            deps_result = self._install_deps(
                target, cfg.timeout_s_by_language.get(Language.PHP, 300)
            )
            if deps_result.timed_out:
                return IndexArtifact(
                    language=target.language,
                    project_root=target.root_path,
                    scip_path=scip_path,
                    logs_path=logs_path,
                    tool_name=self.TOOL_NAME,
                    tool_version=version,
                    duration_s=time.monotonic() - start_time,
                    success=False,
                    error_message="composer install timed out",
                )
            if deps_result.exit_code != 0:
                return IndexArtifact(
                    language=target.language,
                    project_root=target.root_path,
                    scip_path=scip_path,
                    logs_path=logs_path,
                    tool_name=self.TOOL_NAME,
                    tool_version=version,
                    duration_s=time.monotonic() - start_time,
                    success=False,
                    error_message=f"composer install failed: {deps_result.stderr_tail}",
                )

        # Build command
        cmd = self._build_command(target, tool_path)

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

    def _build_command(self, target: ProjectTarget, tool_path: str) -> list[str]:
        """Build the scip-php command."""
        return [tool_path]

    def _resolve_tool_path(self) -> str | None:
        """Resolve scip-php executable path.

        Falls back to common Composer global bin locations if PATH is incomplete.
        """
        resolved = shutil.which(self.TOOL_NAME)
        if resolved:
            return resolved
        for candidate in self._FALLBACK_TOOL_PATHS:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return None

    def _check_tool_available(self, tool_path: str) -> tuple[bool, str]:
        """Check whether resolved tool path is executable and retrieve version."""
        # Import locally to avoid expanding module-level surface for a small helper.
        from contextmine_core.semantic_snapshot.indexers.runner import check_tool_version

        return check_tool_version(tool_path)

    def _should_install_deps(self, target: ProjectTarget, cfg: IndexConfig) -> bool:
        """Check if dependencies should be installed."""
        if cfg.install_deps_mode == InstallDepsMode.NEVER:
            return False

        if cfg.install_deps_mode == InstallDepsMode.ALWAYS:
            return True

        # AUTO mode: check if configured Composer vendor dir exists.
        vendor = target.root_path / self._resolve_vendor_dir(target.root_path)
        return not vendor.exists()

    def _resolve_vendor_dir(self, project_root: Path) -> Path:
        """Resolve Composer vendor-dir from composer.json config.

        Defaults to ``vendor`` when composer.json is missing or unparsable.
        """
        composer_json = project_root / "composer.json"
        try:
            content = json.loads(composer_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return Path("vendor")

        config = content.get("config")
        if not isinstance(config, dict):
            return Path("vendor")
        vendor_dir = str(config.get("vendor-dir") or "vendor").strip()
        if not vendor_dir:
            return Path("vendor")
        return Path(vendor_dir)

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
            "--ignore-platform-reqs",
        ]

        logger.info("Installing dependencies with composer in %s", target.root_path)
        return run_cmd(cmd=cmd, cwd=target.root_path, timeout_s=timeout_s)
