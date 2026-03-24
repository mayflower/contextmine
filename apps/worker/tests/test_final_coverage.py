"""Targeted tests to close coverage gaps in apps/worker modules.

Each section targets specific uncovered lines from --cov-report=term-missing.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from contextmine_worker.flows import (
    IGNORED_REPO_PATH_PARTS,
    _embedding_batch_timeout_seconds,
    _is_ignored_repo_path,
    _joern_parse_timeout_seconds,
    _knowledge_graph_build_timeout_seconds,
    _log_background_task_failure,
    _sync_blocking_step_timeout_seconds,
    _sync_document_step_timeout_seconds,
    _sync_documents_per_run_limit,
    _sync_source_timeout_seconds,
    _sync_temporal_coupling_max_files_per_commit,
    _twin_graph_build_timeout_seconds,
    _uri_to_file_path,
)
from contextmine_worker.github_sync import (
    MAX_FILE_SIZE,
    SyncStats,
    _cached_composer_vendor_dirs,
    _GitCommitTouch,
    _path_is_within,
    build_uri,
    compute_content_hash,
    get_file_title,
    https_url_to_ssh,
    is_eligible_file,
)
from contextmine_worker.web_sync import (
    WebPage,
    WebSyncStats,
    extract_markdown_with_trafilatura,
    get_page_title,
    is_url_in_scope,
)

# ============================================================================
# 1. github_sync.py — pure helper functions
#    Missing: various lines in is_eligible_file, composer vendor dirs,
#    https_url_to_ssh, clone_or_pull_repo edge cases
# ============================================================================


class TestIsEligibleFile:
    """Cover lines in is_eligible_file (124-156)."""

    def test_allowed_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("code")
        assert is_eligible_file(Path("main.py"), tmp_path) is True

    def test_disallowed_extension(self, tmp_path: Path) -> None:
        assert is_eligible_file(Path("image.bmp"), tmp_path) is False

    def test_allowed_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "Dockerfile"
        f.write_text("FROM python:3.12")
        assert is_eligible_file(Path("Dockerfile"), tmp_path) is True

    def test_too_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "big.py"
        f.write_bytes(b"x" * (MAX_FILE_SIZE + 1))
        assert is_eligible_file(Path("big.py"), tmp_path) is False

    def test_hidden_directory(self, tmp_path: Path) -> None:
        assert is_eligible_file(Path(".hidden/main.py"), tmp_path) is False

    def test_github_directory_allowed(self, tmp_path: Path) -> None:
        d = tmp_path / ".github" / "workflows"
        d.mkdir(parents=True)
        f = d / "ci.yml"
        f.write_text("on: push")
        assert is_eligible_file(Path(".github/workflows/ci.yml"), tmp_path) is True

    def test_skip_dirs(self, tmp_path: Path) -> None:
        assert is_eligible_file(Path("node_modules/pkg/index.js"), tmp_path) is False

    def test_nonexistent_file_no_crash(self, tmp_path: Path) -> None:
        # File does not exist, but has valid extension - should pass
        assert is_eligible_file(Path("nonexistent.py"), tmp_path) is True


class TestComposerVendorDirs:
    """Cover _cached_composer_vendor_dirs and _composer_vendor_dirs."""

    def test_no_composer_json(self, tmp_path: Path) -> None:
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()

    def test_with_custom_vendor_dir(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        # Need to clear lru_cache for this test
        _cached_composer_vendor_dirs.cache_clear()

        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"config": {"vendor-dir": "src/libs"}}))
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ("src/libs",)
        _cached_composer_vendor_dirs.cache_clear()

    def test_invalid_json(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text("not json")
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()
        _cached_composer_vendor_dirs.cache_clear()

    def test_no_config_key(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text('{"name": "test/pkg"}')
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()
        _cached_composer_vendor_dirs.cache_clear()

    def test_vendor_dir_blocked(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"config": {"vendor-dir": "src/libs"}}))

        # Now is_eligible_file should reject files under src/libs
        assert is_eligible_file(Path("src/libs/vendor/foo.php"), tmp_path) is False
        _cached_composer_vendor_dirs.cache_clear()

    def test_empty_vendor_dir(self, tmp_path: Path) -> None:
        from contextmine_worker.github_sync import _cached_composer_vendor_dirs

        _cached_composer_vendor_dirs.cache_clear()
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"config": {"vendor-dir": ""}}))
        result = _cached_composer_vendor_dirs(str(tmp_path))
        assert result == ()
        _cached_composer_vendor_dirs.cache_clear()


class TestPathIsWithin:
    def test_within(self) -> None:
        assert _path_is_within(Path("src/libs/foo.py"), Path("src/libs")) is True

    def test_not_within(self) -> None:
        assert _path_is_within(Path("src/main.py"), Path("src/libs")) is False


class TestBuildUri:
    def test_basic(self) -> None:
        result = build_uri("owner", "repo", "src/main.py", "main")
        assert result == "git://github.com/owner/repo/src/main.py?ref=main"


class TestComputeContentHash:
    def test_deterministic(self) -> None:
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("hello")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_content(self) -> None:
        assert compute_content_hash("a") != compute_content_hash("b")


class TestGetFileTitle:
    def test_basic(self) -> None:
        assert get_file_title(Path("src/main.py")) == "src/main.py"


class TestHttpsUrlToSsh:
    def test_github_https(self) -> None:
        result = https_url_to_ssh("https://github.com/owner/repo.git")
        assert result == "git@github.com:owner/repo.git"

    def test_with_auth_in_url(self) -> None:
        result = https_url_to_ssh("https://user@github.com/owner/repo.git")
        assert result == "git@github.com:owner/repo.git"

    def test_non_github_fallback(self) -> None:
        result = https_url_to_ssh("https://gitlab.com/owner/repo.git")
        # Non-github falls back to original URL
        assert result == "https://gitlab.com/owner/repo.git"


class TestSyncStats:
    def test_defaults(self) -> None:
        stats = SyncStats()
        assert stats.files_scanned == 0
        assert stats.files_indexed == 0
        assert stats.files_skipped == 0
        assert stats.files_deleted == 0
        assert stats.docs_created == 0
        assert stats.docs_updated == 0
        assert stats.docs_deleted == 0

    def test_modification(self) -> None:
        stats = SyncStats()
        stats.files_scanned = 10
        stats.docs_created = 5
        assert stats.files_scanned == 10


class TestGitCommitTouch:
    def test_creation(self) -> None:
        touch = _GitCommitTouch(
            authored_at=datetime.now(UTC),
            author_name="Test",
            author_email="test@example.com",
            files={"src/main.py": (10, 5)},
        )
        assert touch.author_name == "Test"
        assert "src/main.py" in touch.files


# ============================================================================
# 2. flows.py — pure helper functions (lines 98-182)
# ============================================================================


class TestUriToFilePath:
    def test_git_uri(self) -> None:
        result = _uri_to_file_path("git://github.com/owner/repo/src/main.py?ref=main")
        assert result == "src/main.py"

    def test_simple_path(self) -> None:
        # _uri_to_file_path splits on "/" and takes [-1] for 5+ parts
        result = _uri_to_file_path("src/main.py")
        assert result == "main.py"  # only 2 parts, takes last after split(5)

    def test_no_slash(self) -> None:
        result = _uri_to_file_path("main.py")
        assert result == "main.py"


class TestIsIgnoredRepoPath:
    def test_node_modules(self) -> None:
        assert _is_ignored_repo_path("node_modules/pkg/index.js") is True

    def test_vendor(self) -> None:
        assert _is_ignored_repo_path("vendor/lib/file.php") is True

    def test_pycache(self) -> None:
        assert _is_ignored_repo_path("src/__pycache__/module.pyc") is True

    def test_venv(self) -> None:
        assert _is_ignored_repo_path("venv/lib/python/site.py") is True
        assert _is_ignored_repo_path(".venv/lib/python/site.py") is True

    def test_normal_path(self) -> None:
        assert _is_ignored_repo_path("src/main.py") is False

    def test_dist(self) -> None:
        assert _is_ignored_repo_path("dist/bundle.js") is True

    def test_build(self) -> None:
        assert _is_ignored_repo_path("build/output.js") is True

    def test_src_libs(self) -> None:
        assert _is_ignored_repo_path("src/libs/vendor/file.php") is True

    def test_backslash_normalization(self) -> None:
        assert _is_ignored_repo_path("node_modules\\pkg\\index.js") is True


class TestIgnoredRepoParts:
    def test_expected_entries(self) -> None:
        assert "node_modules" in IGNORED_REPO_PATH_PARTS
        assert "vendor" in IGNORED_REPO_PATH_PARTS
        assert "__pycache__" in IGNORED_REPO_PATH_PARTS
        assert ".git" in IGNORED_REPO_PATH_PARTS


class TestLogBackgroundTaskFailure:
    def test_cancelled_task(self) -> None:
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True
        _log_background_task_failure(task)  # Should not raise

    def test_task_with_exception(self) -> None:
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("test")
        _log_background_task_failure(task)  # Should not raise

    def test_task_no_exception(self) -> None:
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = None
        _log_background_task_failure(task)  # Should not raise


# ============================================================================
# 3. web_sync.py — pure helpers (lines 29-94)
# ============================================================================


class TestIsUrlInScope:
    def test_same_host_root(self) -> None:
        assert is_url_in_scope("https://docs.example.com/page", "https://docs.example.com/") is True

    def test_different_host(self) -> None:
        assert is_url_in_scope("https://other.com/page", "https://docs.example.com/") is False

    def test_different_scheme(self) -> None:
        assert is_url_in_scope("http://example.com/page", "https://example.com/") is False

    def test_same_host_subpath(self) -> None:
        assert is_url_in_scope("https://example.com/docs/page", "https://example.com/docs/") is True

    def test_same_host_different_path(self) -> None:
        assert (
            is_url_in_scope("https://example.com/other/page", "https://example.com/docs/") is False
        )

    def test_root_base(self) -> None:
        assert is_url_in_scope("https://example.com/anything", "https://example.com/") is True

    def test_empty_base_path(self) -> None:
        assert is_url_in_scope("https://example.com/page", "https://example.com") is True


class TestGetPageTitle:
    def test_with_title(self) -> None:
        page = WebPage(
            url="https://example.com", title="My Page", markdown="# Title", content_hash="abc"
        )
        assert get_page_title(page) == "My Page"

    def test_without_title(self) -> None:
        page = WebPage(
            url="https://example.com/docs/guide", title="", markdown="# Guide", content_hash="def"
        )
        assert get_page_title(page) == "/docs/guide"


class TestExtractMarkdownWithTrafilatura:
    def test_empty(self) -> None:
        assert extract_markdown_with_trafilatura("") is None

    def test_basic_html(self) -> None:
        html = "<html><body><p>Hello world, this is a test paragraph with some content.</p></body></html>"
        result = extract_markdown_with_trafilatura(html)
        # May or may not extract depending on content length
        # Just ensure it doesn't crash
        assert result is None or isinstance(result, str)

    def test_invalid_html(self) -> None:
        result = extract_markdown_with_trafilatura("<not>valid<<html")
        assert result is None or isinstance(result, str)


class TestWebSyncStats:
    def test_defaults(self) -> None:
        stats = WebSyncStats()
        assert stats.pages_crawled == 0
        assert stats.errors == []


class TestWebPage:
    def test_creation(self) -> None:
        page = WebPage(url="u", title="t", markdown="m", content_hash="h")
        assert page.etag is None
        assert page.last_modified is None


class TestFlowsTimeoutHelpers:
    """Test timeout configuration helpers from flows.py."""

    def test_sync_source_timeout(self) -> None:
        result = _sync_source_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 0

    def test_embedding_batch_timeout(self) -> None:
        result = _embedding_batch_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 10

    def test_knowledge_graph_build_timeout(self) -> None:
        result = _knowledge_graph_build_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 120

    def test_twin_graph_build_timeout(self) -> None:
        result = _twin_graph_build_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 120

    def test_sync_blocking_step_timeout(self) -> None:
        result = _sync_blocking_step_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 30

    def test_sync_document_step_timeout(self) -> None:
        result = _sync_document_step_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 10

    def test_sync_documents_per_run_limit(self) -> None:
        result = _sync_documents_per_run_limit()
        assert isinstance(result, int)
        assert result >= 0

    def test_sync_temporal_coupling_max_files(self) -> None:
        result = _sync_temporal_coupling_max_files_per_commit()
        assert isinstance(result, int)
        assert result >= 0

    def test_joern_parse_timeout(self) -> None:
        result = _joern_parse_timeout_seconds()
        assert isinstance(result, int)
        assert result >= 30
