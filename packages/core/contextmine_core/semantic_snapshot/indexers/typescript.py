"""TypeScript/JavaScript SCIP indexer backend.

Uses scip-typescript to index TypeScript and JavaScript projects.
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


class TypescriptIndexerBackend(BaseIndexerBackend):
    """Backend for scip-typescript indexer.

    Handles both TypeScript and JavaScript projects.

    Tool: scip-typescript (npm install -g @sourcegraph/scip-typescript)
    """

    TOOL_NAME = "scip-typescript"

    def can_handle(self, target: ProjectTarget) -> bool:
        """Check if this backend can handle the project."""
        return target.language in (Language.TYPESCRIPT, Language.JAVASCRIPT)

    def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
        """Run scip-typescript to index the project.

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
                f"{self.TOOL_NAME} not found. Install with: npm install -g @sourcegraph/scip-typescript"
            )

        # Install dependencies if needed
        if self._should_install_deps(target, cfg):
            self._install_deps(target, cfg.timeout_s_by_language.get(target.language, 600))

        # Build command (JS may require an explicit project config with allowJs=true)
        cmd, generated_project_config = self._build_command(target, scip_path)

        # Build environment
        env = dict(cfg.env_overrides)
        if cfg.node_memory_mb:
            # Set Node.js memory limit
            node_options = env.get("NODE_OPTIONS", "")
            node_options += f" --max-old-space-size={cfg.node_memory_mb}"
            env["NODE_OPTIONS"] = node_options.strip()

        # Run indexer
        timeout = cfg.timeout_s_by_language.get(target.language, 600)

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
        finally:
            if generated_project_config and generated_project_config.exists():
                generated_project_config.unlink(missing_ok=True)

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
        # scip-typescript outputs to index.scip by default
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

    def _build_command(
        self,
        target: ProjectTarget,
        output_path: Path,
    ) -> tuple[list[str], Path | None]:
        """Build the scip-typescript command."""
        cmd = [self.TOOL_NAME, "index"]

        # For JavaScript, force a project config that enables JS coverage.
        if target.language == Language.JAVASCRIPT:
            project_config = self._create_javascript_project_config(target.root_path)
            cmd.extend(["--project", str(project_config)])
            return cmd, project_config

        return cmd, None

    def _create_javascript_project_config(self, root_path: Path) -> Path:
        """Create a temporary tsconfig for JavaScript indexing with allowJs enabled."""
        generated_config_path = root_path / ".contextmine.scip.javascript.json"
        existing_tsconfig = root_path / "tsconfig.json"

        config: dict[str, object] = {
            "compilerOptions": {
                "allowJs": True,
                "checkJs": False,
                "noEmit": True,
                "skipLibCheck": True,
            },
            "include": ["**/*.js", "**/*.jsx", "**/*.mjs", "**/*.cjs"],
            "exclude": [
                "**/node_modules/**",
                "**/dist/**",
                "**/build/**",
                "**/vendor/**",
                "**/.git/**",
            ],
        }
        if existing_tsconfig.exists():
            config["extends"] = "./tsconfig.json"

        generated_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return generated_config_path

    def _should_install_deps(self, target: ProjectTarget, cfg: IndexConfig) -> bool:
        """Check if dependencies should be installed."""
        if cfg.install_deps_mode == InstallDepsMode.NEVER:
            return False

        if cfg.install_deps_mode == InstallDepsMode.ALWAYS:
            return True

        # AUTO mode: check if node_modules exists
        node_modules = target.root_path / "node_modules"
        return not node_modules.exists()

    def _install_deps(self, target: ProjectTarget, timeout_s: int) -> CmdResult:
        """Install project dependencies."""
        package_manager = target.metadata.get("package_manager", "npm")

        if package_manager == "pnpm":
            cmd = ["pnpm", "install", "--frozen-lockfile"]
        elif package_manager == "yarn":
            cmd = ["yarn", "install", "--frozen-lockfile"]
        elif package_manager == "bun":
            cmd = ["bun", "install", "--frozen-lockfile"]
        else:
            # npm - use ci for lockfile, install otherwise
            if (target.root_path / "package-lock.json").exists():
                cmd = ["npm", "ci"]
            else:
                cmd = ["npm", "install"]

        logger.info("Installing dependencies with %s in %s", package_manager, target.root_path)
        return run_cmd(cmd=cmd, cwd=target.root_path, timeout_s=timeout_s)
