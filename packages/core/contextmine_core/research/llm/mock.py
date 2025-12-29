"""Mock LLM Provider for testing.

Provides deterministic responses for testing the research agent
without making actual API calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from contextmine_core.research.llm.provider import LLMProvider
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing.

    Returns canned responses based on configured patterns or defaults.
    Useful for unit tests and development without API calls.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        default_text_response: str = "This is a mock response.",
        structured_responses: dict[str, Any] | None = None,
    ):
        """Initialize mock provider.

        Args:
            model_name: Name to report as model
            default_text_response: Default response for generate_text
            structured_responses: Dict mapping schema names to response dicts
        """
        self._model_name = model_name
        self._default_text_response = default_text_response
        self._structured_responses = structured_responses or {}
        self._call_history: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        """Get the mock model name."""
        return self._model_name

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Get history of all calls made to this provider."""
        return self._call_history

    def reset_history(self) -> None:
        """Clear call history."""
        self._call_history = []

    def set_text_response(self, response: str) -> None:
        """Set the default text response."""
        self._default_text_response = response

    def set_structured_response(self, schema_name: str, response: dict[str, Any]) -> None:
        """Set a canned response for a specific schema.

        Args:
            schema_name: Name of the Pydantic model class
            response: Dict to return (will be validated against schema)
        """
        self._structured_responses[schema_name] = response

    async def generate_text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate mock text response."""
        self._call_history.append(
            {
                "method": "generate_text",
                "system": system,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return self._default_text_response

    async def generate_structured(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> T:
        """Generate mock structured response."""
        self._call_history.append(
            {
                "method": "generate_structured",
                "system": system,
                "messages": list(messages),
                "output_schema": output_schema.__name__,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        schema_name = output_schema.__name__

        # Check for configured response
        if schema_name in self._structured_responses:
            return output_schema.model_validate(self._structured_responses[schema_name])

        # Generate a default response based on schema fields
        default_data = self._generate_default_for_schema(output_schema)
        return output_schema.model_validate(default_data)

    def _generate_default_for_schema(self, schema: type[BaseModel]) -> dict[str, Any]:
        """Generate default values for a Pydantic schema.

        Uses schema field info to generate sensible defaults.
        """
        result: dict[str, Any] = {}

        for field_name, field_info in schema.model_fields.items():
            annotation = field_info.annotation

            # Handle Optional types
            if hasattr(annotation, "__origin__"):
                origin = getattr(annotation, "__origin__", None)
                if origin is type(None):
                    result[field_name] = None
                    continue

            # Generate defaults based on type
            if annotation is str:
                result[field_name] = f"mock_{field_name}"
            elif annotation is int:
                result[field_name] = 0
            elif annotation is float:
                result[field_name] = 0.0
            elif annotation is bool:
                result[field_name] = False
            elif annotation is list or (
                hasattr(annotation, "__origin__") and annotation.__origin__ is list
            ):
                result[field_name] = []
            elif annotation is dict or (
                hasattr(annotation, "__origin__") and annotation.__origin__ is dict
            ):
                result[field_name] = {}
            elif field_info.default is not None:
                result[field_name] = field_info.default
            else:
                result[field_name] = None

        return result


class FailingMockProvider(LLMProvider):
    """Mock provider that fails for testing error handling."""

    def __init__(
        self,
        model_name: str = "failing-mock",
        fail_count: int = 2,
        error_type: type[Exception] = ConnectionError,
    ):
        """Initialize failing mock.

        Args:
            model_name: Name to report
            fail_count: Number of times to fail before succeeding
            error_type: Exception type to raise
        """
        self._model_name = model_name
        self._fail_count = fail_count
        self._error_type = error_type
        self._call_count = 0
        self._success_response = "Success after retries"

    @property
    def model_name(self) -> str:
        return self._model_name

    def reset(self) -> None:
        """Reset call count."""
        self._call_count = 0

    async def generate_text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._error_type(f"Mock failure {self._call_count}/{self._fail_count}")
        return self._success_response

    async def generate_structured(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> T:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise self._error_type(f"Mock failure {self._call_count}/{self._fail_count}")

        # Return minimal valid response
        default_data: dict[str, Any] = {}
        for field_name, field_info in output_schema.model_fields.items():
            if field_info.annotation is str:
                default_data[field_name] = "success"
            elif field_info.annotation is int:
                default_data[field_name] = 1
            elif field_info.annotation is bool:
                default_data[field_name] = True
            else:
                default_data[field_name] = None

        return output_schema.model_validate(default_data)
