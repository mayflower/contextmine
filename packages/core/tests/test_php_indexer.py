"""Unit tests for PHP SCIP indexer backend behavior."""

from pathlib import Path

from contextmine_core.semantic_snapshot.indexers.php import PhpIndexerBackend
from contextmine_core.semantic_snapshot.models import (
    IndexConfig,
    InstallDepsMode,
    Language,
    ProjectTarget,
)


def test_resolve_vendor_dir_from_composer_config(tmp_path: Path) -> None:
    """Use composer.json config.vendor-dir when present."""
    project_root = tmp_path
    (project_root / "composer.json").write_text(
        '{"name":"acme/test","config":{"vendor-dir":"phpmyfaq/src/libs"}}',
        encoding="utf-8",
    )

    backend = PhpIndexerBackend()
    vendor_dir = backend._resolve_vendor_dir(project_root)

    assert vendor_dir.as_posix() == "phpmyfaq/src/libs"


def test_should_install_deps_checks_custom_vendor_dir(tmp_path: Path) -> None:
    """AUTO mode should skip install if configured vendor-dir already exists."""
    project_root = tmp_path
    (project_root / "composer.json").write_text(
        '{"name":"acme/test","config":{"vendor-dir":"phpmyfaq/src/libs"}}',
        encoding="utf-8",
    )
    (project_root / "phpmyfaq" / "src" / "libs").mkdir(parents=True)
    (project_root / "phpmyfaq" / "src" / "libs" / "autoload.php").write_text(
        "<?php\n",
        encoding="utf-8",
    )

    backend = PhpIndexerBackend()
    target = ProjectTarget(language=Language.PHP, root_path=project_root)
    cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)

    assert backend._should_install_deps(target, cfg) is False


def test_should_install_deps_when_autoload_missing(tmp_path: Path) -> None:
    """AUTO mode should install deps if vendor dir exists but autoload is missing."""
    project_root = tmp_path
    (project_root / "composer.json").write_text(
        '{"name":"acme/test","config":{"vendor-dir":"phpmyfaq/src/libs"}}',
        encoding="utf-8",
    )
    (project_root / "phpmyfaq" / "src" / "libs").mkdir(parents=True)

    backend = PhpIndexerBackend()
    target = ProjectTarget(language=Language.PHP, root_path=project_root)
    cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)

    assert backend._should_install_deps(target, cfg) is True


def test_should_install_deps_when_forced_via_metadata(tmp_path: Path) -> None:
    """Force flag should bypass AUTO heuristics and always install."""
    project_root = tmp_path
    (project_root / "composer.json").write_text(
        '{"name":"acme/test","config":{"vendor-dir":"vendor"}}',
        encoding="utf-8",
    )
    (project_root / "vendor").mkdir(parents=True)
    (project_root / "vendor" / "autoload.php").write_text(
        "<?php\n",
        encoding="utf-8",
    )

    backend = PhpIndexerBackend()
    target = ProjectTarget(
        language=Language.PHP,
        root_path=project_root,
        metadata={"force_install_deps": True},
    )
    cfg = IndexConfig(install_deps_mode=InstallDepsMode.AUTO)

    assert backend._should_install_deps(target, cfg) is True
