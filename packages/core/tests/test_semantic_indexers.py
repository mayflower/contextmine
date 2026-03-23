"""Tests for semantic snapshot indexers: runner, detection, index_project, index_repo."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from contextmine_core.semantic_snapshot.indexers import (
    _NULL_SCIP_PATH,
    BACKENDS,
    index_project,
    index_repo,
)
from contextmine_core.semantic_snapshot.indexers.runner import (
    CmdResult,
    CommandFailedError,
    CommandNotFoundError,
    check_tool_version,
    run_cmd,
)
from contextmine_core.semantic_snapshot.models import (
    IndexArtifact,
    IndexConfig,
    Language,
    ProjectTarget,
)

# ===========================================================================
# CmdResult dataclass
# ===========================================================================


class TestCmdResult:
    def test_default_timed_out_false(self):
        r = CmdResult(exit_code=0, stdout_tail="ok", stderr_tail="", elapsed_s=1.0)
        assert r.timed_out is False

    def test_fields(self):
        r = CmdResult(
            exit_code=1, stdout_tail="out", stderr_tail="err", elapsed_s=2.5, timed_out=True
        )
        assert r.exit_code == 1
        assert r.stdout_tail == "out"
        assert r.stderr_tail == "err"
        assert r.elapsed_s == 2.5
        assert r.timed_out is True


# ===========================================================================
# CommandFailedError
# ===========================================================================


class TestCommandFailedError:
    def test_attributes(self):
        err = CommandFailedError("failed", exit_code=2, stderr="some error")
        assert err.exit_code == 2
        assert err.stderr == "some error"
        assert "failed" in str(err)


# ===========================================================================
# run_cmd
# ===========================================================================


class TestRunCmd:
    def test_successful_command(self, tmp_path: Path):
        result = run_cmd(["echo", "hello"], cwd=tmp_path)
        assert result.exit_code == 0
        assert "hello" in result.stdout_tail
        assert result.timed_out is False
        assert result.elapsed_s >= 0

    def test_command_not_found_raises(self, tmp_path: Path):
        with pytest.raises(CommandNotFoundError, match="Command not found"):
            run_cmd(["this-command-does-not-exist-xyz"], cwd=tmp_path)

    def test_nonzero_exit_code(self, tmp_path: Path):
        result = run_cmd(["false"], cwd=tmp_path)
        assert result.exit_code != 0
        assert result.timed_out is False

    def test_timeout_returns_timed_out(self, tmp_path: Path):
        result = run_cmd(["sleep", "10"], cwd=tmp_path, timeout_s=1)
        assert result.timed_out is True
        assert result.exit_code == -1

    def test_env_override(self, tmp_path: Path):
        result = run_cmd(
            ["env"],
            cwd=tmp_path,
            env={"MY_TEST_VAR": "hello_from_test"},
        )
        assert "MY_TEST_VAR=hello_from_test" in result.stdout_tail

    def test_logs_path_writes_output(self, tmp_path: Path):
        logs_file = tmp_path / "logs" / "cmd.log"
        result = run_cmd(["echo", "logged"], cwd=tmp_path, logs_path=logs_file)
        assert result.exit_code == 0
        assert logs_file.exists()
        content = logs_file.read_text()
        assert "logged" in content
        assert "Command:" in content
        assert "Exit code:" in content

    def test_logs_path_on_timeout(self, tmp_path: Path):
        logs_file = tmp_path / "timeout.log"
        run_cmd(["sleep", "10"], cwd=tmp_path, timeout_s=1, logs_path=logs_file)
        assert logs_file.exists()
        content = logs_file.read_text()
        assert "TIMEOUT" in content

    def test_tail_chars_limits_output(self, tmp_path: Path):
        # Generate output longer than tail_chars
        script = tmp_path / "gen.sh"
        script.write_text("#!/bin/sh\nprintf '%0.sA' {1..200}")
        script.chmod(0o755)
        result = run_cmd(["sh", str(script)], cwd=tmp_path, tail_chars=50)
        assert len(result.stdout_tail) <= 50

    def test_stderr_captured(self, tmp_path: Path):
        result = run_cmd(["sh", "-c", "echo 'err msg' >&2"], cwd=tmp_path)
        assert "err msg" in result.stderr_tail


# ===========================================================================
# check_tool_version
# ===========================================================================


class TestCheckToolVersion:
    def test_existing_tool(self):
        # "echo" is always available
        available, version = check_tool_version("echo")
        assert available is True

    def test_nonexistent_tool(self):
        available, version = check_tool_version("definitely-not-a-tool-xyz123")
        assert available is False
        assert version == ""

    @patch("contextmine_core.semantic_snapshot.indexers.runner.subprocess.run")
    def test_version_flag_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="1.2.3\n", stderr="")
        available, version = check_tool_version("some-tool")
        assert available is True
        assert version == "1.2.3"

    @patch("contextmine_core.semantic_snapshot.indexers.runner.subprocess.run")
    def test_fallback_to_v_flag(self, mock_run):
        # First call (--version) fails, second call (-v) succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="2.0.0\n", stderr=""),
        ]
        available, version = check_tool_version("some-tool")
        assert available is True
        assert version == "2.0.0"

    @patch("contextmine_core.semantic_snapshot.indexers.runner.subprocess.run")
    def test_both_version_flags_fail(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        available, version = check_tool_version("some-tool")
        assert available is True
        assert version == "unknown"

    @patch("contextmine_core.semantic_snapshot.indexers.runner.subprocess.run")
    def test_timeout_returns_unknown_version(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="tool", timeout=10)
        available, version = check_tool_version("some-tool")
        assert available is True
        assert version == "unknown"


# ===========================================================================
# BACKENDS registry
# ===========================================================================


class TestBackendsRegistry:
    def test_backends_list_non_empty(self):
        assert len(BACKENDS) > 0

    def test_all_backends_have_tool_name(self):
        for backend in BACKENDS:
            assert hasattr(backend, "TOOL_NAME")
            assert isinstance(backend.TOOL_NAME, str)

    def test_all_backends_have_can_handle(self):
        for backend in BACKENDS:
            assert hasattr(backend, "can_handle")
            assert callable(backend.can_handle)


# ===========================================================================
# index_project
# ===========================================================================


class TestIndexProject:
    def test_no_backend_found(self):
        """When no backend can handle the target, return failure artifact."""
        # Create a target for a language that no backend can handle
        # We use a mock language or project that won't match
        target = ProjectTarget(
            language=Language.PYTHON,
            root_path=Path("/nonexistent"),
            metadata={},
        )
        cfg = IndexConfig(output_dir=Path("/tmp/test-scip"))

        # Patch all backends to not handle this target
        with (
            patch.object(BACKENDS[0], "can_handle", return_value=False),
            patch.object(BACKENDS[1], "can_handle", return_value=False),
            patch.object(BACKENDS[2], "can_handle", return_value=False),
            patch.object(BACKENDS[3], "can_handle", return_value=False),
        ):
            artifact = index_project(target, cfg)

        assert artifact.success is False
        assert "No backend found" in (artifact.error_message or "")
        assert artifact.scip_path == _NULL_SCIP_PATH

    def test_command_not_found_error(self):
        """When the indexer tool is not installed, return graceful failure."""
        target = ProjectTarget(
            language=Language.PYTHON,
            root_path=Path("/tmp"),
        )
        cfg = IndexConfig(output_dir=Path("/tmp/test-scip"))

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.TOOL_NAME = "scip-python"
        mock_backend.index.side_effect = CommandNotFoundError("scip-python not found")

        with patch(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        ):
            artifact = index_project(target, cfg)

        assert artifact.success is False
        assert "not found" in (artifact.error_message or "")

    def test_unexpected_error(self):
        """When an unexpected error occurs, return graceful failure."""
        target = ProjectTarget(
            language=Language.PYTHON,
            root_path=Path("/tmp"),
        )
        cfg = IndexConfig(output_dir=Path("/tmp/test-scip"))

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.TOOL_NAME = "scip-python"
        mock_backend.index.side_effect = RuntimeError("disk full")

        with patch(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        ):
            artifact = index_project(target, cfg)

        assert artifact.success is False
        assert "Unexpected error" in (artifact.error_message or "")

    def test_successful_indexing(self):
        """When backend succeeds, return the artifact it produced."""
        target = ProjectTarget(
            language=Language.PYTHON,
            root_path=Path("/tmp"),
        )
        cfg = IndexConfig(output_dir=Path("/tmp/test-scip"))

        expected_artifact = IndexArtifact(
            language=Language.PYTHON,
            project_root=Path("/tmp"),
            scip_path=Path("/tmp/test-scip/index.scip"),
            logs_path=None,
            tool_name="scip-python",
            tool_version="1.0.0",
            duration_s=2.5,
            success=True,
        )

        mock_backend = MagicMock()
        mock_backend.can_handle.return_value = True
        mock_backend.index.return_value = expected_artifact

        with patch(
            "contextmine_core.semantic_snapshot.indexers.BACKENDS",
            [mock_backend],
        ):
            artifact = index_project(target, cfg)

        assert artifact.success is True
        assert artifact.tool_name == "scip-python"
        assert artifact.duration_s == 2.5


# ===========================================================================
# index_repo
# ===========================================================================


class TestIndexRepo:
    def test_skips_disabled_languages(self, tmp_path: Path):
        """Languages not in enabled_languages are skipped."""
        # Create a project that detect_projects would find
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "dummy.py").write_text("x = 1\n")

        cfg = IndexConfig(
            enabled_languages={Language.TYPESCRIPT},  # Python disabled
            output_dir=tmp_path / "output",
        )

        with patch(
            "contextmine_core.semantic_snapshot.indexers.detect_projects",
            return_value=[
                ProjectTarget(language=Language.PYTHON, root_path=tmp_path),
            ],
        ):
            artifacts = index_repo(tmp_path, cfg)

        assert len(artifacts) == 0

    def test_best_effort_continues_after_failure(self, tmp_path: Path):
        """In best_effort mode, continues indexing after a failure."""
        cfg = IndexConfig(
            enabled_languages={Language.PYTHON, Language.TYPESCRIPT},
            output_dir=tmp_path / "output",
            best_effort=True,
        )

        projects = [
            ProjectTarget(language=Language.PYTHON, root_path=tmp_path / "py"),
            ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path / "ts"),
        ]

        fail_artifact = IndexArtifact(
            language=Language.PYTHON,
            project_root=tmp_path / "py",
            scip_path=_NULL_SCIP_PATH,
            logs_path=None,
            tool_name="scip-python",
            tool_version="unknown",
            duration_s=0,
            success=False,
            error_message="Tool not found",
        )
        ok_artifact = IndexArtifact(
            language=Language.TYPESCRIPT,
            project_root=tmp_path / "ts",
            scip_path=tmp_path / "output" / "ts.scip",
            logs_path=None,
            tool_name="scip-typescript",
            tool_version="1.0",
            duration_s=1.0,
            success=True,
        )

        with (
            patch(
                "contextmine_core.semantic_snapshot.indexers.detect_projects",
                return_value=projects,
            ),
            patch(
                "contextmine_core.semantic_snapshot.indexers.index_project",
                side_effect=[fail_artifact, ok_artifact],
            ),
        ):
            artifacts = index_repo(tmp_path, cfg)

        assert len(artifacts) == 2

    def test_stops_on_first_failure_when_not_best_effort(self, tmp_path: Path):
        cfg = IndexConfig(
            enabled_languages={Language.PYTHON, Language.TYPESCRIPT},
            output_dir=tmp_path / "output",
            best_effort=False,
        )

        projects = [
            ProjectTarget(language=Language.PYTHON, root_path=tmp_path / "py"),
            ProjectTarget(language=Language.TYPESCRIPT, root_path=tmp_path / "ts"),
        ]

        fail_artifact = IndexArtifact(
            language=Language.PYTHON,
            project_root=tmp_path / "py",
            scip_path=_NULL_SCIP_PATH,
            logs_path=None,
            tool_name="scip-python",
            tool_version="unknown",
            duration_s=0,
            success=False,
            error_message="Tool not found",
        )

        with (
            patch(
                "contextmine_core.semantic_snapshot.indexers.detect_projects",
                return_value=projects,
            ),
            patch(
                "contextmine_core.semantic_snapshot.indexers.index_project",
                return_value=fail_artifact,
            ),
        ):
            artifacts = index_repo(tmp_path, cfg)

        # Should stop after first failure
        assert len(artifacts) == 1
        assert artifacts[0].success is False

    def test_creates_temp_output_dir_when_none(self, tmp_path: Path):
        """When output_dir is None, a temp directory is created."""
        cfg = IndexConfig(output_dir=None)

        with patch(
            "contextmine_core.semantic_snapshot.indexers.detect_projects",
            return_value=[],
        ):
            artifacts = index_repo(tmp_path, cfg)

        assert len(artifacts) == 0
        # After the call, cfg.output_dir should be set to a temp dir
        assert cfg.output_dir is not None
        assert cfg.output_dir.exists()

    def test_creates_output_dir_if_not_exists(self, tmp_path: Path):
        output = tmp_path / "new" / "nested" / "dir"
        cfg = IndexConfig(output_dir=output)

        with patch(
            "contextmine_core.semantic_snapshot.indexers.detect_projects",
            return_value=[],
        ):
            index_repo(tmp_path, cfg)

        assert output.exists()


# ===========================================================================
# IndexConfig defaults
# ===========================================================================


class TestIndexConfig:
    def test_default_enabled_languages(self):
        cfg = IndexConfig()
        assert Language.PYTHON in cfg.enabled_languages
        assert Language.TYPESCRIPT in cfg.enabled_languages
        assert Language.JAVASCRIPT in cfg.enabled_languages
        assert Language.JAVA in cfg.enabled_languages
        assert Language.PHP in cfg.enabled_languages

    def test_default_best_effort(self):
        cfg = IndexConfig()
        assert cfg.best_effort is True

    def test_default_timeouts(self):
        cfg = IndexConfig()
        assert cfg.timeout_s_by_language[Language.PYTHON] == 300
        assert cfg.timeout_s_by_language[Language.JAVA] == 900


# ===========================================================================
# ProjectTarget serialization
# ===========================================================================


class TestProjectTarget:
    def test_to_dict_roundtrip(self):
        target = ProjectTarget(
            language=Language.PYTHON,
            root_path=Path("/my/project"),
            metadata={"has_pyproject": True},
        )
        d = target.to_dict()
        restored = ProjectTarget.from_dict(d)
        assert restored.language == Language.PYTHON
        assert str(restored.root_path) == "/my/project"
        assert restored.metadata["has_pyproject"] is True

    def test_from_dict_missing_metadata(self):
        d = {"language": "python", "root_path": "/tmp"}
        target = ProjectTarget.from_dict(d)
        assert target.metadata == {}


# ===========================================================================
# IndexArtifact serialization
# ===========================================================================


class TestIndexArtifact:
    def test_to_dict_roundtrip(self):
        artifact = IndexArtifact(
            language=Language.TYPESCRIPT,
            project_root=Path("/project"),
            scip_path=Path("/out/index.scip"),
            logs_path=Path("/out/logs.txt"),
            tool_name="scip-typescript",
            tool_version="0.5.0",
            duration_s=12.3,
            success=True,
            error_message=None,
        )
        d = artifact.to_dict()
        restored = IndexArtifact.from_dict(d)
        assert restored.language == Language.TYPESCRIPT
        assert restored.tool_version == "0.5.0"
        assert restored.success is True
        assert restored.logs_path == Path("/out/logs.txt")

    def test_to_dict_with_none_logs_path(self):
        artifact = IndexArtifact(
            language=Language.PYTHON,
            project_root=Path("/p"),
            scip_path=Path("/s"),
            logs_path=None,
            tool_name="t",
            tool_version="v",
            duration_s=0,
        )
        d = artifact.to_dict()
        assert d["logs_path"] is None
        restored = IndexArtifact.from_dict(d)
        assert restored.logs_path is None

    def test_from_dict_defaults(self):
        d = {
            "language": "java",
            "project_root": "/p",
            "scip_path": "/s",
            "logs_path": None,
            "tool_name": "t",
            "tool_version": "v",
            "duration_s": 1.0,
        }
        artifact = IndexArtifact.from_dict(d)
        assert artifact.success is True
        assert artifact.error_message is None
