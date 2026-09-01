"""Emit real ContextMine traces and a metric through the OTLP exporter."""

from __future__ import annotations

import asyncio
import json

from contextmine_core import close_engine
from contextmine_core.telemetry import (
    get_meter,
    init_telemetry,
    shutdown_telemetry,
)


async def main() -> None:
    """Run an empty real sync flow and flush its telemetry."""
    initialized = init_telemetry(
        service_suffix="-smoke-worker",
        extra_resource_attributes={"contextmine.smoke.gate": "otel-enabled"},
    )
    if not initialized:
        raise AssertionError("OpenTelemetry must be enabled for this smoke gate")

    try:
        # Import after SDK initialization so all application spans use the
        # configured provider and exporter.
        from contextmine_worker.flows import sync_due_sources

        result = await sync_due_sources()
        expected = {"scheduled": 0, "sources": []}
        if result != expected:
            raise AssertionError(f"Unexpected empty sync result: {result!r}")

        counter = get_meter("contextmine.smoke").create_counter(
            "contextmine.smoke.sync_due_sources.runs",
            description="Successful sync_due_sources smoke executions",
            unit="{run}",
        )
        counter.add(
            1,
            {
                "contextmine.smoke.result": "empty",
                "contextmine.smoke.scheduled": 0,
            },
        )
    finally:
        await close_engine()
        await shutdown_telemetry()

    print(json.dumps({"otel_enabled": True, "sync_due_sources": result}, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
