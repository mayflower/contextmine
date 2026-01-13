"""OpenTelemetry setup with conditional initialization.

This module handles OTEL SDK initialization. All OTEL imports are lazy
to avoid any overhead when telemetry is disabled.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

# Module-level state
_tracer: Tracer | None = None
_meter: Meter | None = None
_initialized = False

logger = logging.getLogger(__name__)


def init_telemetry(
    service_suffix: str = "",
    extra_resource_attributes: dict[str, str] | None = None,
) -> bool:
    """Initialize OpenTelemetry if enabled.

    This function is safe to call multiple times - subsequent calls are no-ops.
    All OTEL imports happen lazily inside this function to avoid overhead
    when telemetry is disabled.

    Args:
        service_suffix: Suffix to append to service name (e.g., "-api", "-worker")
        extra_resource_attributes: Additional resource attributes to include

    Returns:
        True if telemetry was initialized, False if disabled or already initialized
    """
    global _tracer, _meter, _initialized

    if _initialized:
        return True

    from contextmine_core.settings import get_settings

    settings = get_settings()

    if not settings.otel_enabled:
        logger.debug("OpenTelemetry disabled (OTEL_ENABLED=false)")
        return False

    # Import OTEL SDK components (only when enabled)
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_OFF,
        ALWAYS_ON,
        ParentBasedTraceIdRatio,
        TraceIdRatioBased,
    )

    # Build service name with suffix
    service_name = settings.otel_service_name + service_suffix

    # Create resource with attributes
    resource_attrs: dict[str, str] = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: "0.1.0",
        "deployment.environment": "development" if settings.debug else "production",
    }
    if extra_resource_attributes:
        resource_attrs.update(extra_resource_attributes)
    resource = Resource.create(resource_attrs)

    # Configure sampler based on settings
    sampler_map = {
        "always_on": ALWAYS_ON,
        "always_off": ALWAYS_OFF,
        "traceidratio": TraceIdRatioBased(settings.otel_traces_sampler_arg),
        "parentbased_traceidratio": ParentBasedTraceIdRatio(settings.otel_traces_sampler_arg),
    }
    sampler = sampler_map.get(settings.otel_traces_sampler, ALWAYS_ON)

    # Setup Traces
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)
    span_exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(__name__)

    # Setup Metrics
    metric_exporter = OTLPMetricExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=60000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(__name__)

    # Note: Log export is disabled because Tempo only supports traces.
    # Logs are collected by Promtail and sent to Loki instead.
    # To enable OTEL logs, configure a separate logs endpoint (e.g., Alloy/Loki OTLP).

    _initialized = True
    logger.info(
        f"OpenTelemetry initialized: service={service_name}, "
        f"endpoint={settings.otel_exporter_otlp_endpoint}"
    )
    return True


async def shutdown_telemetry() -> None:
    """Gracefully shutdown telemetry exporters.

    This flushes any pending telemetry data before the application exits.
    Safe to call even if telemetry was never initialized.
    """
    global _initialized

    if not _initialized:
        return

    from opentelemetry import metrics, trace

    # Shutdown tracer provider
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, "shutdown"):
        tracer_provider.shutdown()

    # Shutdown meter provider
    meter_provider = metrics.get_meter_provider()
    if hasattr(meter_provider, "shutdown"):
        meter_provider.shutdown()

    logger.info("OpenTelemetry shutdown complete")


def get_tracer(name: str = __name__) -> Tracer:
    """Get a tracer instance.

    Returns a no-op tracer if telemetry is disabled.

    Args:
        name: The tracer name (typically __name__ of the calling module)

    Returns:
        A Tracer instance (no-op if OTEL disabled)
    """
    from opentelemetry import trace

    if _tracer is not None:
        return _tracer
    return trace.get_tracer(name)


def get_meter(name: str = __name__) -> Meter:
    """Get a meter instance.

    Returns a no-op meter if telemetry is disabled.

    Args:
        name: The meter name (typically __name__ of the calling module)

    Returns:
        A Meter instance (no-op if OTEL disabled)
    """
    from opentelemetry import metrics

    if _meter is not None:
        return _meter
    return metrics.get_meter(name)
