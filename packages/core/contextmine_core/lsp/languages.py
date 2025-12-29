"""Language detection and server configuration for LSP."""

from enum import Enum
from pathlib import Path


class SupportedLanguage(str, Enum):
    """Languages supported by the LSP manager."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CSHARP = "csharp"


# File extension to language mapping
EXTENSION_TO_LANGUAGE: dict[str, SupportedLanguage] = {
    # Python
    ".py": SupportedLanguage.PYTHON,
    ".pyi": SupportedLanguage.PYTHON,
    ".pyw": SupportedLanguage.PYTHON,
    # TypeScript
    ".ts": SupportedLanguage.TYPESCRIPT,
    ".tsx": SupportedLanguage.TYPESCRIPT,
    ".mts": SupportedLanguage.TYPESCRIPT,
    ".cts": SupportedLanguage.TYPESCRIPT,
    # JavaScript
    ".js": SupportedLanguage.JAVASCRIPT,
    ".jsx": SupportedLanguage.JAVASCRIPT,
    ".mjs": SupportedLanguage.JAVASCRIPT,
    ".cjs": SupportedLanguage.JAVASCRIPT,
    # Rust
    ".rs": SupportedLanguage.RUST,
    # Go
    ".go": SupportedLanguage.GO,
    # Java
    ".java": SupportedLanguage.JAVA,
    # C#
    ".cs": SupportedLanguage.CSHARP,
}


def detect_language(file_path: str | Path) -> SupportedLanguage | None:
    """Detect language from file extension.

    Args:
        file_path: Path to the source file

    Returns:
        SupportedLanguage if detected, None otherwise
    """
    suffix = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix)


# Common project root markers for auto-detection
PROJECT_ROOT_MARKERS: list[str] = [
    # Version control
    ".git",
    ".hg",
    ".svn",
    # Python
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    # JavaScript/TypeScript
    "package.json",
    "tsconfig.json",
    # Rust
    "Cargo.toml",
    # Go
    "go.mod",
    "go.sum",
    # Java
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    # C#
    "*.csproj",
    "*.sln",
]


def find_project_root(file_path: str | Path) -> Path:
    """Find project root by looking for common markers.

    Args:
        file_path: Path to a file in the project

    Returns:
        Project root directory, or file's parent if no markers found
    """
    file_path = Path(file_path).resolve()

    # Start from file's directory and walk up
    current = file_path.parent if file_path.is_file() else file_path

    while current != current.parent:
        for marker in PROJECT_ROOT_MARKERS:
            if "*" in marker:
                # Glob pattern
                if list(current.glob(marker)):
                    return current
            elif (current / marker).exists():
                return current
        current = current.parent

    # Fallback to file's directory
    return file_path.parent if file_path.is_file() else file_path
