"""Prefect-compatible telemetry decorators for flows and tasks.

Prefect 3.x does not have official OTEL instrumentation, so we provide
manual decorators that wrap flows/tasks with OpenTelemetry spans.

Usage:
    from contextmine_worker.telemetry import traced_flow, traced_task

    @flow(...)
    @traced_flow()
    async def my_flow():
        ...

    @task(...)
    @traced_task()
    async def my_task():
        ...
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

F = TypeVar("F", bound=Callable[..., Any])

# W3C Trace Context propagator for cross-service trace propagation
propagator = TraceContextTextMapPropagator()


def traced_flow(name: str | None = None) -> Callable[[F], F]:
    """Decorator to add tracing to Prefect flows.

    Creates a parent span for the entire flow execution. Should be applied
    AFTER the @flow decorator (decorators are applied bottom-up).

    Usage:
        @flow(name="sync-source")
        @traced_flow()
        async def sync_source(source_id: str):
            ...

    Args:
        name: Custom span name. Defaults to "flow.{function_name}"

    Returns:
        Decorated async function with tracing
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            span_name = name or f"flow.{func.__name__}"

            with tracer.start_as_current_span(
                span_name,
                attributes={
                    "prefect.flow.name": func.__name__,
                    "prefect.type": "flow",
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def traced_task(name: str | None = None) -> Callable[[F], F]:
    """Decorator to add tracing to Prefect tasks.

    Creates a child span for the task execution. Should be applied
    AFTER the @task decorator.

    Usage:
        @task(name="process-document")
        @traced_task()
        async def process_document(doc_id: str):
            ...

    Args:
        name: Custom span name. Defaults to "task.{function_name}"

    Returns:
        Decorated async function with tracing
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer(__name__)
            span_name = name or f"task.{func.__name__}"

            with tracer.start_as_current_span(
                span_name,
                attributes={
                    "prefect.task.name": func.__name__,
                    "prefect.type": "task",
                },
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def inject_trace_context() -> dict[str, str]:
    """Get current trace context for propagation to other services.

    Use this when triggering a Prefect flow run from the API to pass
    the trace context as a flow parameter.

    Returns:
        Dictionary with W3C Trace Context headers (traceparent, tracestate)

    Usage (in API):
        from contextmine_worker.telemetry import inject_trace_context

        trace_context = inject_trace_context()
        await run_deployment(
            name="sync-single-source-deployment",
            parameters={"source_id": source_id, "trace_context": trace_context},
        )
    """
    carrier: dict[str, str] = {}
    propagator.inject(carrier)
    return carrier


def extract_trace_context(carrier: dict[str, str]) -> otel_context.Context:
    """Extract trace context from headers.

    Use this at the start of a Prefect flow to continue a trace
    started by the API.

    Args:
        carrier: Dictionary with W3C Trace Context headers

    Returns:
        OpenTelemetry Context that can be attached

    Usage (in flow):
        from contextmine_worker.telemetry import extract_trace_context

        @flow
        async def my_flow(source_id: str, trace_context: dict | None = None):
            if trace_context:
                ctx = extract_trace_context(trace_context)
                token = otel_context.attach(ctx)
            try:
                # ... flow logic ...
            finally:
                if trace_context:
                    otel_context.detach(token)
    """
    return propagator.extract(carrier)


def attach_trace_context(carrier: dict[str, str] | None) -> otel_context.Token | None:
    """Attach trace context from a carrier dictionary.

    Convenience wrapper that extracts and attaches in one call.

    Args:
        carrier: Dictionary with W3C Trace Context headers, or None

    Returns:
        Context token for detaching later, or None if no carrier provided

    Usage:
        token = attach_trace_context(trace_context)
        try:
            # ... flow logic with proper trace parent ...
        finally:
            if token:
                otel_context.detach(token)
    """
    if not carrier:
        return None
    ctx = extract_trace_context(carrier)
    return otel_context.attach(ctx)


def detach_trace_context(token: otel_context.Token | None) -> None:
    """Detach a previously attached trace context.

    Args:
        token: The token returned by attach_trace_context, or None
    """
    if token is not None:
        otel_context.detach(token)
