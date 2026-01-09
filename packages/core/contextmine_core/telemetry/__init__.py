"""OpenTelemetry initialization and utilities.

This module provides optional telemetry support that is completely disabled by default.
When OTEL_ENABLED=false (the default), there is zero overhead - no OTEL imports occur.

Usage:
    from contextmine_core.telemetry import init_telemetry, shutdown_telemetry

    # In your application startup:
    telemetry_enabled = init_telemetry(service_suffix="-api")

    # In your application shutdown:
    await shutdown_telemetry()
"""

from contextmine_core.telemetry.setup import (
    get_meter,
    get_tracer,
    init_telemetry,
    shutdown_telemetry,
)
from contextmine_core.telemetry.spans import (
    trace_db_operation,
    trace_embedding_call,
    trace_llm_call,
)

__all__ = [
    "init_telemetry",
    "shutdown_telemetry",
    "get_tracer",
    "get_meter",
    "trace_llm_call",
    "trace_embedding_call",
    "trace_db_operation",
]
