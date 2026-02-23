"""SCIP Polyglot Indexing smoke tests.

These tests verify that the SCIP integration works end-to-end:
- Project detection for all supported languages
- SCIP parsing from fixture files
- Model serialization round-trips

Note: Actual indexer execution tests require the SCIP tools to be installed.
"""

import tempfile
from pathlib import Path

import pytest
from contextmine_core.semantic_snapshot import (
    Snapshot,
    build_snapshot,
    detect_projects,
)
from contextmine_core.semantic_snapshot.indexers import (
    detect_projects_with_diagnostics,
)
from contextmine_core.semantic_snapshot.models import (
    IndexConfig,
    InstallDepsMode,
    Language,
    ProjectTarget,
)


class TestSCIPProjectDetection:
    """Test project detection for all supported languages."""

    def test_detect_python_project(self) -> None:
        """Test detection of Python projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[build-system]")
            (Path(tmpdir) / "main.py").write_text("print('hello')")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.PYTHON

    def test_detect_typescript_project(self) -> None:
        """Test detection of TypeScript projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "tsconfig.json").write_text("{}")
            (Path(tmpdir) / "index.ts").write_text("console.log('hello')")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.TYPESCRIPT

    def test_detect_javascript_project(self) -> None:
        """Test detection of JavaScript projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "index.js").write_text("console.log('hello')")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.JAVASCRIPT

    def test_detect_java_maven_project(self) -> None:
        """Test detection of Java Maven projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pom.xml").write_text("<project></project>")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.JAVA
            assert projects[0].metadata.get("build_tool") == "maven"

    def test_detect_java_gradle_project(self) -> None:
        """Test detection of Java Gradle projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "build.gradle").write_text("plugins {}")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.JAVA
            assert projects[0].metadata.get("build_tool") == "gradle"

    def test_detect_php_project(self) -> None:
        """Test detection of PHP projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "composer.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "composer.lock").write_text("{}")
            (Path(tmpdir) / "index.php").write_text("<?php echo 'ok';")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.PHP

    def test_detect_php_without_composer_lock(self) -> None:
        """Test PHP detection without lockfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "composer.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "index.php").write_text("<?php echo 'ok';")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.PHP

    def test_detect_php_ignores_custom_vendor_dir_subprojects(self) -> None:
        """Do not descend into Composer custom vendor-dir dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "composer.json").write_text(
                '{"name":"root/app","config":{"vendor-dir":"phpmyfaq/src/libs"}}'
            )
            (root / "composer.lock").write_text("{}")
            (root / "index.php").write_text("<?php echo 'ok';")

            dep = root / "phpmyfaq" / "src" / "libs" / "some-package"
            dep.mkdir(parents=True)
            (dep / "composer.json").write_text('{"name":"dep/pkg"}')
            (dep / "dep.php").write_text("<?php echo 'dep';")

            projects = detect_projects(tmpdir)

            php_roots = [p.root_path.resolve() for p in projects if p.language == Language.PHP]
            assert php_roots == [root.resolve()]

    def test_detect_monorepo(self) -> None:
        """Test detection of multiple projects in a monorepo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Python backend
            backend = Path(tmpdir) / "backend"
            backend.mkdir()
            (backend / "pyproject.toml").write_text("[build-system]")

            # TypeScript frontend
            frontend = Path(tmpdir) / "frontend"
            frontend.mkdir()
            (frontend / "package.json").write_text('{"name": "frontend"}')
            (frontend / "tsconfig.json").write_text("{}")

            projects = detect_projects(tmpdir)

            assert len(projects) == 2
            languages = {p.language for p in projects}
            assert Language.PYTHON in languages
            assert Language.TYPESCRIPT in languages

    def test_ignore_node_modules(self) -> None:
        """Test that node_modules is ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "index.ts").write_text("export const x = 1;")
            (Path(tmpdir) / "tsconfig.json").write_text("{}")

            node_modules = Path(tmpdir) / "node_modules" / "some-package"
            node_modules.mkdir(parents=True)
            (node_modules / "package.json").write_text('{"name": "dep"}')
            (node_modules / "dep.ts").write_text("export const y = 2;")

            projects = detect_projects(tmpdir)

            # Should only detect root project
            assert len(projects) == 1

    def test_detect_polyglot_root_languages(self) -> None:
        """Detect TS + PHP in one repository root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "composer.json").write_text('{"name": "test"}')
            (root / "package.json").write_text('{"name": "test"}')
            (root / "tsconfig.json").write_text("{}")
            (root / "app.php").write_text("<?php echo 'ok';")
            (root / "app.ts").write_text("export const app = true;")

            projects, diagnostics = detect_projects_with_diagnostics(root)

            languages = {project.language for project in projects}
            assert Language.PHP in languages
            assert Language.TYPESCRIPT in languages
            assert "php" in diagnostics.languages_detected
            assert "typescript" in diagnostics.languages_detected


class TestSCIPModels:
    """Test SCIP model serialization and configuration."""

    def test_project_target_roundtrip(self) -> None:
        """Test ProjectTarget serialization round-trip."""
        target = ProjectTarget(
            language=Language.PYTHON,
            root_path=Path("/src/project"),
            metadata={"has_pyproject": True},
        )
        d = target.to_dict()
        restored = ProjectTarget.from_dict(d)

        assert restored.language == target.language
        assert restored.root_path == target.root_path
        assert restored.metadata == target.metadata

    def test_index_config_defaults(self) -> None:
        """Test IndexConfig default values."""
        cfg = IndexConfig()

        assert Language.PYTHON in cfg.enabled_languages
        assert Language.TYPESCRIPT in cfg.enabled_languages
        assert Language.JAVA in cfg.enabled_languages
        assert Language.PHP in cfg.enabled_languages
        assert cfg.install_deps_mode == InstallDepsMode.AUTO
        assert cfg.best_effort is True

    def test_snapshot_merge(self) -> None:
        """Test merging two snapshots."""
        from contextmine_core.semantic_snapshot.models import (
            FileInfo,
            Range,
            Symbol,
            SymbolKind,
        )

        s1 = Snapshot(
            files=[FileInfo("/a.py", "python")],
            symbols=[Symbol("a:1:0:fn", SymbolKind.FUNCTION, "/a.py", Range(1, 0, 5, 0), "foo")],
        )
        s2 = Snapshot(
            files=[FileInfo("/b.ts", "typescript")],
            symbols=[Symbol("b:1:0:fn", SymbolKind.FUNCTION, "/b.ts", Range(1, 0, 5, 0), "bar")],
        )

        merged = s1.merge(s2)

        assert len(merged.files) == 2
        assert len(merged.symbols) == 2


class TestSCIPParsing:
    """Test SCIP file parsing."""

    def test_build_snapshot_missing_file(self) -> None:
        """Test that missing SCIP file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_snapshot("/nonexistent.scip")

    def test_snapshot_to_dict_roundtrip(self) -> None:
        """Test Snapshot serialization round-trip."""
        import json

        from contextmine_core.semantic_snapshot.models import (
            FileInfo,
            Occurrence,
            OccurrenceRole,
            Range,
            Relation,
            RelationKind,
            Symbol,
            SymbolKind,
        )

        snapshot = Snapshot(
            files=[FileInfo("/src/main.py", "python")],
            symbols=[
                Symbol(
                    def_id="main.py:1:0:function",
                    kind=SymbolKind.FUNCTION,
                    file_path="/src/main.py",
                    range=Range(1, 0, 10, 1),
                    name="main",
                )
            ],
            occurrences=[
                Occurrence(
                    file_path="/src/main.py",
                    range=Range(1, 0, 1, 4),
                    role=OccurrenceRole.DEFINITION,
                    def_id="main.py:1:0:function",
                )
            ],
            relations=[
                Relation(
                    src_def_id="main.py:1:0:function",
                    kind=RelationKind.CALLS,
                    dst_def_id="unresolved:print",
                    resolved=False,
                )
            ],
            meta={"provider": "test"},
        )

        d = snapshot.to_dict()
        json_str = json.dumps(d)  # Ensure JSON serializable
        restored_dict = json.loads(json_str)
        restored = Snapshot.from_dict(restored_dict)

        assert len(restored.files) == 1
        assert len(restored.symbols) == 1
        assert len(restored.occurrences) == 1
        assert len(restored.relations) == 1
        assert restored.meta["provider"] == "test"


class TestSCIPSettings:
    """Test SCIP settings integration."""

    def test_settings_have_scip_fields(self) -> None:
        """Test that Settings class has all SCIP fields."""
        from contextmine_core.settings import Settings

        settings = Settings()

        assert hasattr(settings, "scip_languages")
        assert hasattr(settings, "scip_install_deps_mode")
        assert hasattr(settings, "scip_timeout_python")
        assert hasattr(settings, "scip_timeout_typescript")
        assert hasattr(settings, "scip_timeout_java")
        assert hasattr(settings, "scip_timeout_php")
        assert hasattr(settings, "scip_node_memory_mb")
        assert hasattr(settings, "scip_best_effort")

    def test_settings_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test SCIP settings default values."""
        from contextmine_core.settings import Settings

        for key in (
            "SCIP_LANGUAGES",
            "SCIP_INSTALL_DEPS_MODE",
            "SCIP_TIMEOUT_PYTHON",
            "SCIP_TIMEOUT_TYPESCRIPT",
            "SCIP_TIMEOUT_JAVA",
            "SCIP_TIMEOUT_PHP",
            "SCIP_NODE_MEMORY_MB",
            "SCIP_BEST_EFFORT",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)

        settings = Settings()

        assert "python" in settings.scip_languages
        assert settings.scip_install_deps_mode == "auto"
        assert settings.scip_timeout_python == 300
        assert settings.scip_timeout_typescript == 600
        assert settings.scip_timeout_java == 900
        assert settings.scip_timeout_php == 900
        assert settings.scip_node_memory_mb == 4096
        assert settings.scip_best_effort is True
