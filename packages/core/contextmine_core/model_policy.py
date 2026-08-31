"""Central policy gate for external embedding and generative-model calls."""

from __future__ import annotations


class ModelCallsDisabledError(RuntimeError):
    """Raised when an external model call is blocked by configuration."""


def ensure_model_calls_enabled() -> None:
    """Fail closed before initializing or invoking an external model client."""
    from contextmine_core.settings import get_settings

    if not get_settings().model_calls_enabled:
        raise ModelCallsDisabledError(
            "External model calls are disabled by MODEL_CALLS_ENABLED=false"
        )
