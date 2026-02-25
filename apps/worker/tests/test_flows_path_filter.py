"""Tests for repository path filtering in worker flows."""

from contextmine_worker.flows import _is_ignored_repo_path


def test_is_ignored_repo_path_skips_dependency_and_build_dirs() -> None:
    assert _is_ignored_repo_path("node_modules/react/index.js")
    assert _is_ignored_repo_path("vendor/symfony/http-foundation.php")
    assert _is_ignored_repo_path("frontend/dist/main.js")
    assert _is_ignored_repo_path("service/.venv/lib/python3.12/site-packages/pkg.py")
    assert _is_ignored_repo_path(r"backend\\venv\\lib\\python3.12\\site-packages\\pkg.py")
    assert _is_ignored_repo_path("phpmyfaq/src/libs/aws/aws-sdk-php/src/S3/S3Client.php")


def test_is_ignored_repo_path_keeps_first_party_paths() -> None:
    assert not _is_ignored_repo_path("src/app/main.py")
    assert not _is_ignored_repo_path("phpmyfaq/src/phpMyFAQ/Database.php")
    assert not _is_ignored_repo_path(".github/workflows/ci.yml")
