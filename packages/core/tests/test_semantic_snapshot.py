"""Tests for the Semantic Snapshot layer.

This module tests the SCIP-based semantic snapshot functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest
from contextmine_core.semantic_snapshot import (
    FileInfo,
    IndexConfig,
    InstallDepsMode,
    Language,
    Occurrence,
    OccurrenceRole,
    ProjectTarget,
    Range,
    Relation,
    RelationKind,
    Snapshot,
    Symbol,
    SymbolKind,
    build_snapshot,
    detect_projects,
)


class TestModels:
    """Tests for Snapshot data models."""

    def test_range_to_dict_roundtrip(self) -> None:
        """Test Range serialization round-trip."""
        range_ = Range(start_line=10, start_col=4, end_line=20, end_col=1)
        d = range_.to_dict()
        restored = Range.from_dict(d)
        assert restored == range_

    def test_file_info_to_dict_roundtrip(self) -> None:
        """Test FileInfo serialization round-trip."""
        file_info = FileInfo(path="/src/main.py", language="python")
        d = file_info.to_dict()
        restored = FileInfo.from_dict(d)
        assert restored.path == file_info.path
        assert restored.language == file_info.language

    def test_symbol_to_dict_roundtrip(self) -> None:
        """Test Symbol serialization round-trip."""
        symbol = Symbol(
            def_id="main.py:10:0:function",
            kind=SymbolKind.FUNCTION,
            file_path="/src/main.py",
            range=Range(10, 0, 20, 1),
            name="my_function",
            container_def_id=None,
        )
        d = symbol.to_dict()
        restored = Symbol.from_dict(d)
        assert restored.def_id == symbol.def_id
        assert restored.kind == symbol.kind
        assert restored.name == symbol.name

    def test_occurrence_to_dict_roundtrip(self) -> None:
        """Test Occurrence serialization round-trip."""
        occ = Occurrence(
            file_path="/src/main.py",
            range=Range(10, 0, 10, 10),
            role=OccurrenceRole.DEFINITION,
            def_id="main.py:10:0:function",
        )
        d = occ.to_dict()
        restored = Occurrence.from_dict(d)
        assert restored.def_id == occ.def_id
        assert restored.role == occ.role

    def test_relation_to_dict_roundtrip(self) -> None:
        """Test Relation serialization round-trip."""
        rel = Relation(
            src_def_id="a:1:0:function",
            kind=RelationKind.CALLS,
            dst_def_id="b:10:0:function",
            resolved=False,
            weight=0.5,
            meta={"callee_name": "foo"},
        )
        d = rel.to_dict()
        restored = Relation.from_dict(d)
        assert restored.src_def_id == rel.src_def_id
        assert restored.kind == rel.kind
        assert restored.resolved == rel.resolved
        assert restored.meta == rel.meta

    def test_snapshot_to_dict_roundtrip(self) -> None:
        """Test Snapshot serialization round-trip."""
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
        # Ensure it's JSON serializable
        json_str = json.dumps(d)
        restored_dict = json.loads(json_str)
        restored = Snapshot.from_dict(restored_dict)

        assert len(restored.files) == 1
        assert len(restored.symbols) == 1
        assert len(restored.occurrences) == 1
        assert len(restored.relations) == 1
        assert restored.meta["provider"] == "test"

    def test_snapshot_merge(self) -> None:
        """Test merging two snapshots."""
        s1 = Snapshot(
            files=[FileInfo("/a.py", "python")],
            symbols=[Symbol("a:1:0:fn", SymbolKind.FUNCTION, "/a.py", Range(1, 0, 5, 0), "foo")],
        )
        s2 = Snapshot(
            files=[FileInfo("/b.py", "python")],
            symbols=[Symbol("b:1:0:fn", SymbolKind.FUNCTION, "/b.py", Range(1, 0, 5, 0), "bar")],
        )

        merged = s1.merge(s2)
        assert len(merged.files) == 2
        assert len(merged.symbols) == 2

    def test_snapshot_get_symbol_by_def_id(self) -> None:
        """Test finding symbol by def_id."""
        snapshot = Snapshot(
            symbols=[
                Symbol("a:1:0:fn", SymbolKind.FUNCTION, "/a.py", Range(1, 0, 5, 0), "foo"),
                Symbol("a:10:0:fn", SymbolKind.FUNCTION, "/a.py", Range(10, 0, 15, 0), "bar"),
            ]
        )

        found = snapshot.get_symbol_by_def_id("a:10:0:fn")
        assert found is not None
        assert found.name == "bar"

        not_found = snapshot.get_symbol_by_def_id("nonexistent")
        assert not_found is None


class TestIndexerModels:
    """Tests for SCIP indexer configuration models."""

    def test_language_enum(self) -> None:
        """Test Language enum values."""
        assert Language.PYTHON.value == "python"
        assert Language.TYPESCRIPT.value == "typescript"
        assert Language.JAVA.value == "java"
        assert Language.PHP.value == "php"

    def test_install_deps_mode_enum(self) -> None:
        """Test InstallDepsMode enum values."""
        assert InstallDepsMode.AUTO.value == "auto"
        assert InstallDepsMode.ALWAYS.value == "always"
        assert InstallDepsMode.NEVER.value == "never"

    def test_project_target_to_dict_roundtrip(self) -> None:
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
        assert cfg.install_deps_mode == InstallDepsMode.AUTO
        assert cfg.best_effort is True


class TestProjectDetection:
    """Tests for project detection."""

    def test_detect_python_project(self) -> None:
        """Test detection of Python projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python project
            (Path(tmpdir) / "pyproject.toml").write_text("[build-system]")
            (Path(tmpdir) / "main.py").write_text("print('hello')")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.PYTHON
            assert projects[0].metadata.get("has_pyproject") is True

    def test_detect_typescript_project(self) -> None:
        """Test detection of TypeScript projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a TypeScript project
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "tsconfig.json").write_text('{"compilerOptions": {}}')
            (Path(tmpdir) / "main.ts").write_text("console.log('hello')")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.TYPESCRIPT
            assert projects[0].metadata.get("has_tsconfig") is True

    def test_detect_javascript_project(self) -> None:
        """Test detection of JavaScript projects (no tsconfig)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a JavaScript project (package.json but no tsconfig)
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "main.js").write_text("console.log('hello')")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.JAVASCRIPT
            assert projects[0].metadata.get("has_tsconfig") is False

    def test_detect_java_maven_project(self) -> None:
        """Test detection of Java Maven projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Maven project
            (Path(tmpdir) / "pom.xml").write_text("<project></project>")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.JAVA
            assert projects[0].metadata.get("build_tool") == "maven"

    def test_detect_java_gradle_project(self) -> None:
        """Test detection of Java Gradle projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Gradle project
            (Path(tmpdir) / "build.gradle").write_text("plugins {}")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.JAVA
            assert projects[0].metadata.get("build_tool") == "gradle"

    def test_detect_php_project(self) -> None:
        """Test detection of PHP projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a PHP project (requires both composer.json and composer.lock)
            (Path(tmpdir) / "composer.json").write_text('{"name": "test"}')
            (Path(tmpdir) / "composer.lock").write_text("{}")

            projects = detect_projects(tmpdir)

            assert len(projects) == 1
            assert projects[0].language == Language.PHP
            assert projects[0].metadata.get("has_composer_lock") is True

    def test_detect_monorepo(self) -> None:
        """Test detection of multiple projects in a monorepo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a monorepo with multiple projects
            backend = Path(tmpdir) / "backend"
            backend.mkdir()
            (backend / "pyproject.toml").write_text("[build-system]")

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
            # Create a project with node_modules
            (Path(tmpdir) / "package.json").write_text('{"name": "test"}')

            node_modules = Path(tmpdir) / "node_modules" / "some-package"
            node_modules.mkdir(parents=True)
            (node_modules / "package.json").write_text('{"name": "dep"}')

            projects = detect_projects(tmpdir)

            # Should only detect the root project, not the one in node_modules
            assert len(projects) == 1


class TestBuildSnapshot:
    """Tests for the build_snapshot entry function."""

    def test_build_snapshot_missing_scip_file(self) -> None:
        """Test that missing SCIP file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_snapshot("/nonexistent.scip")
