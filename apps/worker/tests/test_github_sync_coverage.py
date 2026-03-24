"""Coverage tests for github_sync.py utility functions.

Targets:
- is_eligible_file (lines 124-156): extension checks, file size, hidden files, skip dirs
- _path_is_within (lines 159-164)
- _cached_composer_vendor_dirs (lines 167-189)
- _composer_vendor_dirs (lines 192-199)
- SyncStats dataclass
- https_url_to_ssh
"""

from __future__ import annotations

import json
from pathlib import Path

from contextmine_worker.github_sync import (
    SyncStats,
    _path_is_within,
    is_eligible_file,
)


class TestIsEligibleFile:
    def test_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("# code")
        assert is_eligible_file(Path("main.py"), tmp_path) is True

    def test_disallowed_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "image.png"
        f.write_bytes(b"\x00")
        assert is_eligible_file(Path("image.png"), tmp_path) is False

    def test_allowed_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "Dockerfile"
        f.write_text("FROM python:3.11")
        assert is_eligible_file(Path("Dockerfile"), tmp_path) is True

    def test_file_too_large(self, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        f.write_bytes(b"x" * (1024 * 1024 + 1))
        assert is_eligible_file(Path("big.py"), tmp_path) is False

    def test_hidden_file_excluded(self, tmp_path: Path) -> None:
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        f = hidden_dir / "secret.py"
        f.write_text("# secret")
        assert is_eligible_file(Path(".hidden/secret.py"), tmp_path) is False

    def test_github_dir_allowed(self, tmp_path: Path) -> None:
        github_dir = tmp_path / ".github" / "workflows"
        github_dir.mkdir(parents=True)
        f = github_dir / "ci.yml"
        f.write_text("name: CI")
        assert is_eligible_file(Path(".github/workflows/ci.yml"), tmp_path) is True

    def test_skip_dir_excluded(self, tmp_path: Path) -> None:
        node_dir = tmp_path / "node_modules" / "pkg"
        node_dir.mkdir(parents=True)
        f = node_dir / "index.js"
        f.write_text("module.exports = {}")
        assert is_eligible_file(Path("node_modules/pkg/index.js"), tmp_path) is False

    def test_nonexistent_file_passes_size_check(self, tmp_path: Path) -> None:
        # File doesn't exist - size check is skipped
        assert is_eligible_file(Path("doesnt_exist.py"), tmp_path) is True

    def test_venv_excluded(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / "venv" / "lib"
        venv_dir.mkdir(parents=True)
        f = venv_dir / "site.py"
        f.write_text("# venv")
        assert is_eligible_file(Path("venv/lib/site.py"), tmp_path) is False

    def test_pycache_excluded(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        f = cache_dir / "module.pyc"
        f.write_bytes(b"\x00")
        assert is_eligible_file(Path("__pycache__/module.pyc"), tmp_path) is False

    def test_dist_excluded(self, tmp_path: Path) -> None:
        assert is_eligible_file(Path("dist/main.js"), tmp_path) is False


class TestPathIsWithin:
    def test_within(self) -> None:
        assert _path_is_within(Path("vendor/lib/a.py"), Path("vendor")) is True

    def test_not_within(self) -> None:
        assert _path_is_within(Path("src/main.py"), Path("vendor")) is False

    def test_same_path(self) -> None:
        assert _path_is_within(Path("vendor"), Path("vendor")) is True


class TestComposerVendorDirs:
    def test_no_composer_json(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()

    def test_composer_with_vendor_dir(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps(
                {
                    "config": {"vendor-dir": "src/libs"},
                }
            )
        )
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ("src/libs",)

    def test_composer_no_vendor_dir(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"require": {}}))
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()

    def test_composer_invalid_json(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text("{invalid")
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()

    def test_composer_vendor_dirs_wrapper(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import (
            _cached_composer_vendor_dirs,
            _composer_vendor_dirs,
        )

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps(
                {
                    "config": {"vendor-dir": "custom/vendor"},
                }
            )
        )
        result = _composer_vendor_dirs(tmp_path)
        assert len(result) == 1
        assert result[0] == Path("custom/vendor")


class TestSyncStats:
    def test_defaults(self) -> None:
        stats = SyncStats()
        assert stats.files_scanned == 0
        assert stats.files_indexed == 0
        assert stats.docs_created == 0

    def test_custom(self) -> None:
        stats = SyncStats(files_scanned=10, docs_created=5)
        assert stats.files_scanned == 10
        assert stats.docs_created == 5


class TestHttpsUrlToSsh:
    def test_github_url(self) -> None:
        from contextmine_worker.github_sync import https_url_to_ssh

        result = https_url_to_ssh("https://github.com/owner/repo.git")
        assert result == "git@github.com:owner/repo.git"

    def test_non_github_url(self) -> None:
        from contextmine_worker.github_sync import https_url_to_ssh

        result = https_url_to_ssh("https://gitlab.com/owner/repo.git")
        assert "gitlab.com" in result

    def test_already_ssh(self) -> None:
        from contextmine_worker.github_sync import https_url_to_ssh

        result = https_url_to_ssh("git@github.com:owner/repo.git")
        assert result == "git@github.com:owner/repo.git"


class TestIsEligibleFileWithComposerVendor:
    def test_file_in_custom_vendor_dir(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()

        # Create composer.json with custom vendor dir
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps(
                {
                    "config": {"vendor-dir": "custom_libs"},
                }
            )
        )

        # File in custom vendor dir
        custom_libs = tmp_path / "custom_libs" / "pkg"
        custom_libs.mkdir(parents=True)
        f = custom_libs / "lib.php"
        f.write_text("<?php // lib")

        result = is_eligible_file(Path("custom_libs/pkg/lib.php"), tmp_path)
        assert result is False
