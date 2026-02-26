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
from contextmine_core.semantic_snapshot.scip import SCIPProvider


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


class TestSCIPSymbolInference:
    """Tests for symbol-kind/name fallback inference from SCIP descriptors."""

    def test_infer_function_symbol(self) -> None:
        provider = SCIPProvider("/nonexistent.scip")
        kind, name = provider._infer_kind_and_name_from_symbol(  # noqa: SLF001
            "scip-python python project 0.0.0 `pkg.mod`/run_sync()."
        )
        assert kind == SymbolKind.FUNCTION
        assert name == "run_sync"

    def test_infer_method_and_parameter_symbols(self) -> None:
        provider = SCIPProvider("/nonexistent.scip")
        method_kind, method_name = provider._infer_kind_and_name_from_symbol(  # noqa: SLF001
            "scip-python python project 0.0.0 `pkg.mod`/Worker#execute()."
        )
        parameter_kind, parameter_name = provider._infer_kind_and_name_from_symbol(  # noqa: SLF001
            "scip-python python project 0.0.0 `pkg.mod`/Worker#execute().(context)"
        )
        assert method_kind == SymbolKind.METHOD
        assert method_name == "execute"
        assert parameter_kind == SymbolKind.PARAMETER
        assert parameter_name == "context"

    def test_infer_class_and_property_symbols(self) -> None:
        provider = SCIPProvider("/nonexistent.scip")
        class_kind, class_name = provider._infer_kind_and_name_from_symbol(  # noqa: SLF001
            "scip-python python project 0.0.0 `pkg.mod`/SyncConfig#"
        )
        field_kind, field_name = provider._infer_kind_and_name_from_symbol(  # noqa: SLF001
            "scip-python python project 0.0.0 `pkg.mod`/SyncConfig#batch_size."
        )
        assert class_kind == SymbolKind.CLASS
        assert class_name == "SyncConfig"
        assert field_kind == SymbolKind.PROPERTY
        assert field_name == "batch_size"

    def test_ignore_local_symbol(self) -> None:
        provider = SCIPProvider("/nonexistent.scip")
        kind, name = provider._infer_kind_and_name_from_symbol("local 42")  # noqa: SLF001
        assert kind == SymbolKind.UNKNOWN
        assert name is None


class TestSCIPOccurrenceRelations:
    """Tests for occurrence-derived relation extraction from SCIP indexes."""

    def test_occurrence_relations_include_calls(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-php"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "PHP"
        doc.relative_path = "src/app.php"

        caller_symbol = "scip-php php phpmyfaq 0.0.0 src/app.php/foo()."
        callee_symbol = "scip-php php phpmyfaq 0.0.0 src/app.php/bar()."

        caller = doc.symbols.add()
        caller.symbol = caller_symbol
        caller.kind = scip_pb2.SymbolInformation.Kind.Function
        caller.display_name = "foo"

        callee = doc.symbols.add()
        callee.symbol = callee_symbol
        callee.kind = scip_pb2.SymbolInformation.Kind.Function
        callee.display_name = "bar"

        caller_def = doc.occurrences.add()
        caller_def.symbol = caller_symbol
        caller_def.symbol_roles = scip_pb2.SymbolRole.Definition
        caller_def.range.extend([0, 0, 2, 1])

        callee_def = doc.occurrences.add()
        callee_def.symbol = callee_symbol
        callee_def.symbol_roles = scip_pb2.SymbolRole.Definition
        callee_def.range.extend([4, 0, 5, 1])

        call_ref = doc.occurrences.add()
        call_ref.symbol = callee_symbol
        call_ref.symbol_roles = scip_pb2.SymbolRole.ReadAccess
        call_ref.syntax_kind = scip_pb2.SyntaxKind.IdentifierFunction
        call_ref.range.extend([1, 4, 1, 7])
        call_ref.enclosing_range.extend([0, 0, 2, 1])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        relations = {(rel.src_def_id, rel.kind, rel.dst_def_id) for rel in snapshot.relations}

        assert (caller_symbol, RelationKind.CALLS, callee_symbol) in relations

    def test_occurrence_relations_use_enclosing_range_when_definition_is_narrow(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-python"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "Python"
        doc.relative_path = "pkg/service.py"

        caller_symbol = "scip-python python demo 0.0.0 `pkg.service`/run()."
        callee_symbol = "scip-python python demo 0.0.0 `pkg.service`/helper()."

        caller = doc.symbols.add()
        caller.symbol = caller_symbol
        caller.kind = scip_pb2.SymbolInformation.Kind.Function
        caller.display_name = "run"

        callee = doc.symbols.add()
        callee.symbol = callee_symbol
        callee.kind = scip_pb2.SymbolInformation.Kind.Function
        callee.display_name = "helper"

        caller_def = doc.occurrences.add()
        caller_def.symbol = caller_symbol
        caller_def.symbol_roles = scip_pb2.SymbolRole.Definition
        caller_def.range.extend([0, 4, 0, 7])  # identifier-only
        caller_def.enclosing_range.extend([0, 0, 3, 1])  # full function body

        callee_def = doc.occurrences.add()
        callee_def.symbol = callee_symbol
        callee_def.symbol_roles = scip_pb2.SymbolRole.Definition
        callee_def.range.extend([5, 0, 7, 1])

        call_ref = doc.occurrences.add()
        call_ref.symbol = callee_symbol
        call_ref.symbol_roles = scip_pb2.SymbolRole.ReadAccess
        call_ref.syntax_kind = scip_pb2.SyntaxKind.UnspecifiedSyntaxKind
        call_ref.range.extend([2, 8, 2, 14])
        call_ref.enclosing_range.extend([0, 0, 3, 1])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        relations = {(rel.src_def_id, rel.kind, rel.dst_def_id) for rel in snapshot.relations}

        assert (caller_symbol, RelationKind.CALLS, callee_symbol) in relations

    def test_occurrence_relations_fallback_to_module_for_global_imports(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-typescript"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "TypeScript"
        doc.relative_path = "src/index.ts"

        module_symbol = "scip-typescript typescript demo 0.0.0 src/index.ts/"
        imported_symbol = "scip-typescript typescript demo 0.0.0 src/lib.ts/dep()."

        module = doc.symbols.add()
        module.symbol = module_symbol
        module.kind = scip_pb2.SymbolInformation.Kind.Module
        module.display_name = "index"

        imported = doc.symbols.add()
        imported.symbol = imported_symbol
        imported.kind = scip_pb2.SymbolInformation.Kind.Function
        imported.display_name = "dep"

        module_def = doc.occurrences.add()
        module_def.symbol = module_symbol
        module_def.symbol_roles = scip_pb2.SymbolRole.Definition
        module_def.range.extend([0, 0, 0, 1])  # narrow/no enclosing range on purpose

        imported_def = doc.occurrences.add()
        imported_def.symbol = imported_symbol
        imported_def.symbol_roles = scip_pb2.SymbolRole.Definition
        imported_def.range.extend([1, 0, 1, 3])

        import_ref = doc.occurrences.add()
        import_ref.symbol = imported_symbol
        import_ref.symbol_roles = scip_pb2.SymbolRole.Import
        import_ref.range.extend([6, 0, 6, 6])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        relations = {(rel.src_def_id, rel.kind, rel.dst_def_id) for rel in snapshot.relations}

        assert (module_symbol, RelationKind.IMPORTS, imported_symbol) in relations

    def test_range_nesting_recovers_contains_when_enclosing_symbol_missing(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-php"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "PHP"
        doc.relative_path = "src/worker.php"

        class_symbol = "scip-php php demo 0.0.0 src/worker.php/Worker#"
        method_symbol = "scip-php php demo 0.0.0 src/worker.php/Worker#execute()."

        class_info = doc.symbols.add()
        class_info.symbol = class_symbol
        class_info.kind = scip_pb2.SymbolInformation.Kind.Class
        class_info.display_name = "Worker"

        method_info = doc.symbols.add()
        method_info.symbol = method_symbol
        method_info.kind = scip_pb2.SymbolInformation.Kind.Method
        method_info.display_name = "execute"

        class_def = doc.occurrences.add()
        class_def.symbol = class_symbol
        class_def.symbol_roles = scip_pb2.SymbolRole.Definition
        class_def.range.extend([0, 0, 0, 6])
        class_def.enclosing_range.extend([0, 0, 9, 1])

        method_def = doc.occurrences.add()
        method_def.symbol = method_symbol
        method_def.symbol_roles = scip_pb2.SymbolRole.Definition
        method_def.range.extend([2, 4, 2, 11])
        method_def.enclosing_range.extend([2, 4, 8, 1])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        relations = {(rel.src_def_id, rel.kind, rel.dst_def_id) for rel in snapshot.relations}

        assert (class_symbol, RelationKind.CONTAINS, method_symbol) in relations

    def test_occurrence_call_detection_allows_unknown_target_kind(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-php"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "PHP"
        doc.relative_path = "src/app.php"

        caller_symbol = "scip-php php demo 0.0.0 src/app.php/run()."
        callee_symbol = "scip-php php demo 0.0.0 src/app.php/UnknownTarget"

        caller = doc.symbols.add()
        caller.symbol = caller_symbol
        caller.kind = scip_pb2.SymbolInformation.Kind.Function
        caller.display_name = "run"

        callee = doc.symbols.add()
        callee.symbol = callee_symbol
        callee.kind = scip_pb2.SymbolInformation.Kind.UnspecifiedKind
        callee.display_name = "UnknownTarget"

        caller_def = doc.occurrences.add()
        caller_def.symbol = caller_symbol
        caller_def.symbol_roles = scip_pb2.SymbolRole.Definition
        caller_def.range.extend([0, 0, 3, 1])

        callee_def = doc.occurrences.add()
        callee_def.symbol = callee_symbol
        callee_def.symbol_roles = scip_pb2.SymbolRole.Definition
        callee_def.range.extend([5, 0, 5, 12])

        call_ref = doc.occurrences.add()
        call_ref.symbol = callee_symbol
        call_ref.symbol_roles = scip_pb2.SymbolRole.ReadAccess
        call_ref.syntax_kind = scip_pb2.SyntaxKind.IdentifierFunction
        call_ref.range.extend([1, 4, 1, 16])
        call_ref.enclosing_range.extend([0, 0, 3, 1])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        relations = {(rel.src_def_id, rel.kind, rel.dst_def_id) for rel in snapshot.relations}

        assert (caller_symbol, RelationKind.CALLS, callee_symbol) in relations

    def test_scip_php_contextual_caller_fallback_without_enclosing_ranges(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-php"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "PHP"
        doc.relative_path = "src/app.php"

        caller_symbol = "scip-php php demo 0.0.0 src/app.php/run()."
        callee_symbol = "scip-php php demo 0.0.0 src/app.php/helper()."

        caller = doc.symbols.add()
        caller.symbol = caller_symbol
        caller.kind = scip_pb2.SymbolInformation.Kind.UnspecifiedKind

        callee = doc.symbols.add()
        callee.symbol = callee_symbol
        callee.kind = scip_pb2.SymbolInformation.Kind.UnspecifiedKind

        caller_def = doc.occurrences.add()
        caller_def.symbol = caller_symbol
        caller_def.symbol_roles = scip_pb2.SymbolRole.Definition
        caller_def.syntax_kind = scip_pb2.SyntaxKind.IdentifierFunctionDefinition
        caller_def.range.extend([10, 12, 10, 15])

        callee_def = doc.occurrences.add()
        callee_def.symbol = callee_symbol
        callee_def.symbol_roles = scip_pb2.SymbolRole.Definition
        callee_def.syntax_kind = scip_pb2.SyntaxKind.IdentifierFunctionDefinition
        callee_def.range.extend([30, 12, 30, 18])

        call_ref = doc.occurrences.add()
        call_ref.symbol = callee_symbol
        call_ref.symbol_roles = 0
        call_ref.syntax_kind = scip_pb2.SyntaxKind.Identifier
        call_ref.range.extend([14, 8, 14, 14])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        relations = {(rel.src_def_id, rel.kind, rel.dst_def_id) for rel in snapshot.relations}

        assert (caller_symbol, RelationKind.CALLS, callee_symbol) in relations

    def test_unknown_kind_symbol_is_kept_with_fallback_kind(self) -> None:
        from contextmine_core.semantic_snapshot.proto import scip_pb2

        provider = SCIPProvider("/nonexistent.scip")
        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-php"
        index.metadata.tool_info.version = "0.0.0"

        doc = index.documents.add()
        doc.language = "PHP"
        doc.relative_path = "src/app.php"

        unknown_symbol = "scip-php php demo 0.0.0 src/app.php/UnknownTarget"
        symbol = doc.symbols.add()
        symbol.symbol = unknown_symbol
        symbol.kind = scip_pb2.SymbolInformation.Kind.UnspecifiedKind

        symbol_def = doc.occurrences.add()
        symbol_def.symbol = unknown_symbol
        symbol_def.symbol_roles = scip_pb2.SymbolRole.Definition
        symbol_def.range.extend([1, 0, 1, 12])

        snapshot = provider._convert_index(index)  # noqa: SLF001
        symbol_by_id = {item.def_id: item for item in snapshot.symbols}

        assert unknown_symbol in symbol_by_id
        assert symbol_by_id[unknown_symbol].kind == SymbolKind.MODULE
