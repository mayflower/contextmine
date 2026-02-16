"""Polyglot real metrics extraction package."""

from contextmine_core.metrics.coverage_reports import (
    detect_coverage_protocol,
    parse_coverage_reports,
)
from contextmine_core.metrics.models import FileMetricRecord, MetricsGateError, ProjectMetricBundle
from contextmine_core.metrics.pipeline import (
    flatten_metric_bundles,
    is_relevant_production_file,
    run_polyglot_metrics_pipeline,
)

__all__ = [
    "FileMetricRecord",
    "MetricsGateError",
    "ProjectMetricBundle",
    "detect_coverage_protocol",
    "parse_coverage_reports",
    "flatten_metric_bundles",
    "is_relevant_production_file",
    "run_polyglot_metrics_pipeline",
]
