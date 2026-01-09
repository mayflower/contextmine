"""Custom span helpers for LLM and embedding instrumentation.

These helpers follow OpenTelemetry semantic conventions for GenAI operations.
See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

F = TypeVar("F", bound=Callable[..., Any])


def trace_llm_call(
    provider: str,
    model: str,
    operation: str = "chat",
) -> Callable[[F], F]:
    """Decorator to trace LLM API calls with semantic attributes.

    Adds OpenTelemetry GenAI semantic attributes to the span:
    - gen_ai.system: The LLM provider (e.g., "anthropic", "openai")
    - gen_ai.request.model: The model name
    - gen_ai.operation.name: The operation type (e.g., "chat", "completion")

    Usage:
        @trace_llm_call(provider="anthropic", model="claude-3-sonnet")
        async def generate(self, prompt: str) -> str:
            ...

    Args:
        provider: The LLM provider name (e.g., "anthropic", "openai", "gemini")
        model: The model name (e.g., "claude-3-sonnet", "gpt-4")
        operation: The operation type (default: "chat")

    Returns:
        Decorated async function with tracing
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"llm.{operation}",
                attributes={
                    "gen_ai.system": provider,
                    "gen_ai.request.model": model,
                    "gen_ai.operation.name": operation,
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    # Add response attributes if available
                    if hasattr(result, "usage"):
                        usage = result.usage
                        if hasattr(usage, "prompt_tokens"):
                            span.set_attribute(
                                "gen_ai.usage.input_tokens", usage.prompt_tokens
                            )
                        if hasattr(usage, "completion_tokens"):
                            span.set_attribute(
                                "gen_ai.usage.output_tokens", usage.completion_tokens
                            )
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


@asynccontextmanager
async def trace_embedding_call(
    provider: str,
    model: str,
    batch_size: int = 1,
) -> AsyncIterator[trace.Span]:
    """Context manager for tracing embedding API calls.

    Creates a span with GenAI semantic attributes for embedding operations.
    The span is yielded so callers can add additional attributes.

    Usage:
        async with trace_embedding_call("openai", "text-embedding-3-small", len(texts)) as span:
            embeddings = await client.embed(texts)
            span.set_attribute("embedding.tokens_used", total_tokens)

    Args:
        provider: The embedding provider (e.g., "openai", "gemini")
        model: The embedding model name
        batch_size: Number of texts in the batch

    Yields:
        The active span for adding additional attributes
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(
        "embedding.embed",
        attributes={
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "embedding.batch_size": batch_size,
        },
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def trace_db_operation(
    operation: str,
    table: str | None = None,
) -> AsyncIterator[trace.Span]:
    """Context manager for tracing database operations.

    Creates a span with database semantic attributes. Use this for
    custom database operations that aren't automatically instrumented
    by the SQLAlchemy instrumentation.

    Usage:
        async with trace_db_operation("SELECT", "chunks") as span:
            result = await session.execute(query)
            span.set_attribute("db.rows_affected", len(result))

    Args:
        operation: The database operation (e.g., "SELECT", "INSERT", "UPDATE")
        table: The table name (optional)

    Yields:
        The active span for adding additional attributes
    """
    tracer = trace.get_tracer(__name__)
    attrs: dict[str, Any] = {"db.operation": operation}
    if table:
        attrs["db.sql.table"] = table

    with tracer.start_as_current_span(
        f"db.{operation.lower()}",
        attributes=attrs,
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def trace_sync_llm_call(
    provider: str,
    model: str,
    operation: str = "chat",
) -> Callable[[F], F]:
    """Decorator to trace synchronous LLM API calls.

    Same as trace_llm_call but for synchronous functions.

    Args:
        provider: The LLM provider name
        model: The model name
        operation: The operation type (default: "chat")

    Returns:
        Decorated sync function with tracing
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"llm.{operation}",
                attributes={
                    "gen_ai.system": provider,
                    "gen_ai.request.model": model,
                    "gen_ai.operation.name": operation,
                },
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    if hasattr(result, "usage"):
                        usage = result.usage
                        if hasattr(usage, "prompt_tokens"):
                            span.set_attribute(
                                "gen_ai.usage.input_tokens", usage.prompt_tokens
                            )
                        if hasattr(usage, "completion_tokens"):
                            span.set_attribute(
                                "gen_ai.usage.output_tokens", usage.completion_tokens
                            )
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator
