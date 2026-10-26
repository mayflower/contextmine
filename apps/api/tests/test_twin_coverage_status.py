"""Missing coverage is reported as missing, not as 0 % coverage."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from app.routes.twin import _compute_metric_averages, _resolve_metrics_reason
from contextmine_core.models import SourceType
from contextmine_core.twin.service import _build_metric_snapshot


def _metric(coverage: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        coverage=coverage,
        complexity=2.0,
        coupling=1.0,
        cohesion=1.0,
        instability=0.0,
        duplication_ratio=0.0,
        crap_score=None,
        fan_in=0,
        fan_out=0,
        cycle_participation=False,
        cycle_size=0,
        change_frequency=0.0,
        meta={},
    )


def _db(*results: list) -> MagicMock:
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[
            MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=rows))))
            for rows in results
        ]
    )
    return db


def test_snapshot_keeps_missing_coverage_null() -> None:
    meta = {"loc": 10, "coupling": 1.0, "complexity": 2.0}
    assert _build_metric_snapshot(uuid.uuid4(), "file:a.py", meta).coverage is None
    meta["coverage"] = 0.0
    assert _build_metric_snapshot(uuid.uuid4(), "file:a.py", meta).coverage == 0.0


def test_coverage_average_without_any_coverage_is_none() -> None:
    averages = _compute_metric_averages([_metric(None), _metric(None)], metrics_ready=True)
    assert averages["coverage_avg"] is None
    assert averages["complexity_avg"] == 2.0


def test_coverage_average_ignores_files_without_coverage() -> None:
    averages = _compute_metric_averages([_metric(60.0), _metric(None)], metrics_ready=True)
    assert averages["coverage_avg"] == 60.0


@pytest.mark.anyio
async def test_reason_is_ok_with_coverage() -> None:
    db = _db()
    reason = await _resolve_metrics_reason(
        db, uuid.uuid4(), uuid.uuid4(), metrics_ready=True, has_coverage=True
    )
    assert reason == "ok"
    db.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_ready_metrics_without_coverage_await_ci_coverage() -> None:
    source = SimpleNamespace(id=uuid.uuid4(), type=SourceType.GITHUB)
    reason = await _resolve_metrics_reason(
        _db([source], []), uuid.uuid4(), uuid.uuid4(), metrics_ready=True, has_coverage=False
    )
    assert reason == "awaiting_ci_coverage"


@pytest.mark.anyio
async def test_ready_metrics_report_a_failed_coverage_ingest() -> None:
    source = SimpleNamespace(id=uuid.uuid4(), type=SourceType.GITHUB)
    job = SimpleNamespace(status="failed")
    reason = await _resolve_metrics_reason(
        _db([source], [job]), uuid.uuid4(), uuid.uuid4(), metrics_ready=True, has_coverage=False
    )
    assert reason == "coverage_ingest_failed"


@pytest.mark.anyio
async def test_ready_metrics_without_github_source_are_ok() -> None:
    source = SimpleNamespace(id=uuid.uuid4(), type=SourceType.WEB)
    reason = await _resolve_metrics_reason(
        _db([source]), uuid.uuid4(), uuid.uuid4(), metrics_ready=True, has_coverage=False
    )
    assert reason == "ok"
