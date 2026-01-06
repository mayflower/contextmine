"""Java SCIP indexer backend.

Uses scip-java to index Java, Kotlin, and Scala projects.
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

# JVM exports needed for Java 17+ (module system compatibility)
JAVA_17_EXPORTS = [
    "--add-exports=jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
]


class JavaIndexerBackend(BaseIndexerBackend):
    """Backend for scip-java indexer.

    Supports Maven, Gradle, and sbt projects.

    Tool: scip-java (install via Coursier: cs install scip-java)
    """

    TOOL_NAME = "scip-java"

    def can_handle(self, target: ProjectTarget) -> bool:
        """Check if this backend can handle the project."""
        return target.language == Language.JAVA

    def index(self, target: ProjectTarget, cfg: IndexConfig) -> IndexArtifact:
        """Run scip-java to index the project.

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
                f"{self.TOOL_NAME} not found. Install with: cs install scip-java (requires Coursier)"
            )

        # Build command
        cmd = self._build_command(target, scip_path, cfg)

        # Build environment with JVM exports for Java 17+
        env = dict(cfg.env_overrides)
        java_tool_options = env.get("JAVA_TOOL_OPTIONS", "")
        java_tool_options += " " + " ".join(JAVA_17_EXPORTS)
        env["JAVA_TOOL_OPTIONS"] = java_tool_options.strip()

        # Run indexer
        timeout = cfg.timeout_s_by_language.get(Language.JAVA, 900)

        try:
            result = run_cmd(
                cmd=cmd,
                cwd=target.root_path,
                env=env,
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
        # scip-java might output to index.scip if --output not used
        default_scip = target.root_path / "index.scip"
        if default_scip.exists() and default_scip != scip_path:
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
        self, target: ProjectTarget, output_path: Path, cfg: IndexConfig
    ) -> list[str]:
        """Build the scip-java command."""
        cmd = [self.TOOL_NAME, "index"]

        # Specify output path
        cmd.extend(["--output", str(output_path)])

        # Add any extra build args (e.g., for Gradle tasks)
        if cfg.java_build_args:
            cmd.append("--")
            cmd.extend(cfg.java_build_args)

        return cmd
