"""LSP-specific exceptions."""


class LspError(Exception):
    """Base class for LSP errors."""

    pass


class LspNotAvailableError(LspError):
    """Raised when LSP is not available for a language/file.

    This can happen when:
    - The file type is not supported
    - The language server is not installed
    - The multilspy library is not available
    """

    pass


class LspTimeoutError(LspError):
    """Raised when an LSP operation times out."""

    pass


class LspServerError(LspError):
    """Raised when the language server returns an error."""

    pass
