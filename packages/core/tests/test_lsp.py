"""Tests for the LSP module."""

from pathlib import Path

import pytest
from contextmine_core.lsp import (
    EXTENSION_TO_LANGUAGE,
    Location,
    LspManager,
    LspNotAvailableError,
    MockLspClient,
    SupportedLanguage,
    SymbolInfo,
    detect_language,
    find_project_root,
)
from contextmine_core.research.actions.lsp import (
    LspDefinitionAction,
    LspHoverAction,
    LspReferencesAction,
    MockLspDefinitionAction,
)
from contextmine_core.research.actions.schemas import (
    ActionSelection,
    LspDiagnosticsInput,
    LspPositionInput,
)
from contextmine_core.research.run import ResearchRun


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_python(self) -> None:
        """Test detecting Python files."""
        assert detect_language("test.py") == SupportedLanguage.PYTHON
        assert detect_language("test.pyi") == SupportedLanguage.PYTHON
        assert detect_language("/path/to/file.py") == SupportedLanguage.PYTHON

    def test_detect_typescript(self) -> None:
        """Test detecting TypeScript files."""
        assert detect_language("test.ts") == SupportedLanguage.TYPESCRIPT
        assert detect_language("test.tsx") == SupportedLanguage.TYPESCRIPT

    def test_detect_javascript(self) -> None:
        """Test detecting JavaScript files."""
        assert detect_language("test.js") == SupportedLanguage.JAVASCRIPT
        assert detect_language("test.jsx") == SupportedLanguage.JAVASCRIPT

    def test_detect_rust(self) -> None:
        """Test detecting Rust files."""
        assert detect_language("test.rs") == SupportedLanguage.RUST

    def test_detect_go(self) -> None:
        """Test detecting Go files."""
        assert detect_language("test.go") == SupportedLanguage.GO

    def test_detect_unsupported(self) -> None:
        """Test that unsupported file types return None."""
        assert detect_language("test.txt") is None
        assert detect_language("test.md") is None
        assert detect_language("test.json") is None

    def test_extension_to_language_completeness(self) -> None:
        """Test that extension mapping covers expected languages."""
        # Verify all supported languages have at least one extension
        languages_with_extensions = set(EXTENSION_TO_LANGUAGE.values())
        assert SupportedLanguage.PYTHON in languages_with_extensions
        assert SupportedLanguage.TYPESCRIPT in languages_with_extensions
        assert SupportedLanguage.JAVASCRIPT in languages_with_extensions
        assert SupportedLanguage.RUST in languages_with_extensions
        assert SupportedLanguage.GO in languages_with_extensions


class TestProjectRootDetection:
    """Tests for project root detection."""

    def test_find_project_root_with_git(self, tmp_path: Path) -> None:
        """Test finding project root with .git directory."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        sub_dir = project_dir / "src" / "module"
        sub_dir.mkdir(parents=True)

        test_file = sub_dir / "test.py"
        test_file.touch()

        root = find_project_root(test_file)
        assert root == project_dir

    def test_find_project_root_with_pyproject(self, tmp_path: Path) -> None:
        """Test finding project root with pyproject.toml."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").touch()

        sub_dir = project_dir / "src"
        sub_dir.mkdir()

        test_file = sub_dir / "main.py"
        test_file.touch()

        root = find_project_root(test_file)
        assert root == project_dir

    def test_find_project_root_fallback(self, tmp_path: Path) -> None:
        """Test fallback to file's directory when no markers found."""
        # Create a file without any project markers
        test_file = tmp_path / "orphan.py"
        test_file.touch()

        root = find_project_root(test_file)
        assert root == tmp_path


class TestMockLspClient:
    """Tests for the mock LSP client."""

    @pytest.mark.anyio
    async def test_mock_client_definition(self) -> None:
        """Test mock client returns configured definitions."""
        client = MockLspClient()
        client.set_definition(
            "test.py",
            10,
            5,
            [
                Location(
                    file_path="other.py",
                    start_line=20,
                    start_column=0,
                    end_line=25,
                    end_column=10,
                )
            ],
        )

        locations = await client.get_definition("test.py", 10, 5)
        assert len(locations) == 1
        assert locations[0].file_path == "other.py"
        assert locations[0].start_line == 20

    @pytest.mark.anyio
    async def test_mock_client_references(self) -> None:
        """Test mock client returns configured references."""
        client = MockLspClient()
        client.set_references(
            "test.py",
            10,
            5,
            [
                Location("a.py", 1, 0, 1, 10),
                Location("b.py", 5, 0, 5, 10),
            ],
        )

        locations = await client.get_references("test.py", 10, 5)
        assert len(locations) == 2

    @pytest.mark.anyio
    async def test_mock_client_hover(self) -> None:
        """Test mock client returns configured hover info."""
        client = MockLspClient()
        client.set_hover(
            "test.py",
            10,
            5,
            SymbolInfo(
                name="my_function",
                kind="function",
                signature="def my_function(x: int) -> str",
                documentation="Does something useful.",
            ),
        )

        info = await client.get_hover("test.py", 10, 5)
        assert info is not None
        assert info.name == "my_function"
        assert info.kind == "function"

    @pytest.mark.anyio
    async def test_mock_client_empty_results(self) -> None:
        """Test mock client returns empty results when not configured."""
        client = MockLspClient()

        assert await client.get_definition("unknown.py", 1, 0) == []
        assert await client.get_references("unknown.py", 1, 0) == []
        assert await client.get_hover("unknown.py", 1, 0) is None


class TestLspManager:
    """Tests for the LSP manager."""

    def test_singleton_pattern(self) -> None:
        """Test that LspManager is a singleton."""
        LspManager.reset()
        manager1 = LspManager.get_instance()
        manager2 = LspManager.get_instance()
        assert manager1 is manager2
        LspManager.reset()

    @pytest.mark.anyio
    async def test_mock_client_injection(self) -> None:
        """Test that mock client can be injected for testing."""
        LspManager.reset()
        manager = LspManager.get_instance()

        mock_client = MockLspClient()
        manager.set_mock_client(mock_client)

        client = await manager.get_client("test.py")
        assert client is mock_client

        LspManager.reset()

    @pytest.mark.anyio
    async def test_unsupported_file_type(self) -> None:
        """Test that unsupported file types raise error."""
        LspManager.reset()
        manager = LspManager.get_instance()

        with pytest.raises(LspNotAvailableError):
            await manager.get_client("test.txt")

        LspManager.reset()


class TestLspActionSchemas:
    """Tests for LSP action schemas."""

    def test_lsp_position_input(self) -> None:
        """Test LspPositionInput validation."""
        input_data = LspPositionInput(file_path="test.py", line=10, column=5)
        assert input_data.file_path == "test.py"
        assert input_data.line == 10
        assert input_data.column == 5

    def test_lsp_diagnostics_input(self) -> None:
        """Test LspDiagnosticsInput validation."""
        input_data = LspDiagnosticsInput(file_paths=["a.py", "b.py"])
        assert len(input_data.file_paths) == 2

    def test_action_selection_with_lsp_definition(self) -> None:
        """Test ActionSelection with lsp_definition action."""
        selection = ActionSelection(
            action="lsp_definition",
            reasoning="Need to find where function is defined",
            lsp_definition={"file_path": "test.py", "line": 10, "column": 5},
        )
        assert selection.action == "lsp_definition"
        input_params = selection.get_action_input()
        assert input_params is not None
        assert input_params.file_path == "test.py"  # type: ignore[union-attr]

    def test_action_selection_with_lsp_references(self) -> None:
        """Test ActionSelection with lsp_references action."""
        selection = ActionSelection(
            action="lsp_references",
            reasoning="Find all usages",
            lsp_references={"file_path": "test.py", "line": 20, "column": 0},
        )
        assert selection.action == "lsp_references"


class TestLspActions:
    """Tests for LSP actions with mock client."""

    @pytest.mark.anyio
    async def test_lsp_definition_action(self) -> None:
        """Test LspDefinitionAction with mock client."""
        mock_client = MockLspClient()
        mock_client.set_definition(
            "test.py",
            10,
            5,
            [
                Location(
                    file_path="other.py",
                    start_line=20,
                    start_column=0,
                    end_line=25,
                    end_column=10,
                )
            ],
        )

        action = LspDefinitionAction(lsp_client=mock_client)
        run = ResearchRun.create(question="Where is function X defined?")

        result = await action.execute(
            run,
            {"file_path": "test.py", "line": 10, "column": 5},
        )

        assert result.success
        assert "Found 1 definition" in result.output_summary

    @pytest.mark.anyio
    async def test_lsp_definition_action_no_results(self) -> None:
        """Test LspDefinitionAction when no definition found."""
        mock_client = MockLspClient()
        action = LspDefinitionAction(lsp_client=mock_client)
        run = ResearchRun.create(question="Where is X defined?")

        result = await action.execute(
            run,
            {"file_path": "test.py", "line": 10, "column": 5},
        )

        assert result.success
        assert "No definition found" in result.output_summary

    @pytest.mark.anyio
    async def test_lsp_references_action(self) -> None:
        """Test LspReferencesAction with mock client."""
        mock_client = MockLspClient()
        mock_client.set_references(
            "test.py",
            10,
            5,
            [
                Location("a.py", 1, 0, 1, 10),
                Location("b.py", 5, 0, 5, 10),
                Location("c.py", 10, 0, 10, 10),
            ],
        )

        action = LspReferencesAction(lsp_client=mock_client)
        run = ResearchRun.create(question="Where is X used?")

        result = await action.execute(
            run,
            {"file_path": "test.py", "line": 10, "column": 5},
        )

        assert result.success
        assert "Found 3 reference" in result.output_summary

    @pytest.mark.anyio
    async def test_lsp_hover_action(self) -> None:
        """Test LspHoverAction with mock client."""
        mock_client = MockLspClient()
        mock_client.set_hover(
            "test.py",
            10,
            5,
            SymbolInfo(
                name="process",
                kind="function",
                signature="def process(data: list) -> dict",
                documentation="Process the data.",
            ),
        )

        action = LspHoverAction(lsp_client=mock_client)
        run = ResearchRun.create(question="What is process()?")

        result = await action.execute(
            run,
            {"file_path": "test.py", "line": 10, "column": 5},
        )

        assert result.success
        assert "process" in result.output_summary
        assert len(result.evidence) == 1
        assert result.evidence[0].symbol_id == "process"
        assert result.evidence[0].symbol_kind == "function"

    @pytest.mark.anyio
    async def test_lsp_action_missing_file_path(self) -> None:
        """Test that LSP actions require file_path."""
        mock_client = MockLspClient()
        action = LspDefinitionAction(lsp_client=mock_client)
        run = ResearchRun.create(question="Test")

        result = await action.execute(run, {"line": 10, "column": 5})

        assert not result.success
        assert "file_path" in result.error.lower()


class TestMockLspDefinitionAction:
    """Tests for the mock LSP definition action."""

    @pytest.mark.anyio
    async def test_mock_action_returns_configured_results(self) -> None:
        """Test MockLspDefinitionAction returns configured locations."""
        action = MockLspDefinitionAction(
            mock_locations=[
                {
                    "file_path": "definitions.py",
                    "start_line": 100,
                    "end_line": 110,
                    "content": "def my_function(): pass",
                }
            ]
        )

        run = ResearchRun.create(question="Find definition")
        result = await action.execute(
            run,
            {"file_path": "test.py", "line": 10, "column": 5},
        )

        assert result.success
        assert len(result.evidence) == 1
        assert result.evidence[0].file_path == "definitions.py"
        assert result.evidence[0].provenance == "lsp"

    @pytest.mark.anyio
    async def test_mock_action_empty_results(self) -> None:
        """Test MockLspDefinitionAction with no configured results."""
        action = MockLspDefinitionAction()
        run = ResearchRun.create(question="Find definition")

        result = await action.execute(
            run,
            {"file_path": "test.py", "line": 10, "column": 5},
        )

        assert result.success
        assert "No definition found" in result.output_summary
        assert len(result.evidence) == 0
