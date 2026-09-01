"""Verify that the collector received useful ContextMine telemetry."""

from __future__ import annotations

import sys


def main() -> None:
    output = sys.stdin.read()
    required_signals = {
        "API service resource": ("service.name: Str(contextmine-api)",),
        "API health span": (
            "http.route: Str(/api/health)",
            "http.method: Str(GET)",
            "http.status_code: Int(200)",
        ),
        "worker service resource": ("service.name: Str(contextmine-smoke-worker)",),
        "real sync flow span": (
            "prefect.flow.name: Str(sync_due_sources)",
            "prefect.type: Str(flow)",
        ),
        "SQLAlchemy telemetry": ("db.client.connections.usage",),
        "smoke gate resource attribute": (
            "contextmine.smoke.gate",
            "otel-enabled",
        ),
        "sync result metric": (
            "contextmine.smoke.sync_due_sources.runs",
            "contextmine.smoke.result",
            "empty",
        ),
    }
    missing = [
        label
        for label, markers in required_signals.items()
        if not all(marker in output for marker in markers)
    ]
    if missing:
        raise AssertionError("Collector output is missing required signals: " + ", ".join(missing))

    print("OpenTelemetry collector received API, worker flow, and metric signals")


if __name__ == "__main__":
    main()
