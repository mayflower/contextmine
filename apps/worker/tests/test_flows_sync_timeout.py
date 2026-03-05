"""Tests for sync timeout behavior in worker flows."""

from types import SimpleNamespace

import contextmine_worker.flows as flows


def test_sync_source_timeout_allows_zero(monkeypatch) -> None:
    monkeypatch.setattr(
        flows,
        "get_settings",
        lambda: SimpleNamespace(sync_source_timeout_seconds=0),
    )
    assert flows._sync_source_timeout_seconds() == 0


def test_sync_source_timeout_still_clamps_negative(monkeypatch) -> None:
    monkeypatch.setattr(
        flows,
        "get_settings",
        lambda: SimpleNamespace(sync_source_timeout_seconds=-42),
    )
    assert flows._sync_source_timeout_seconds() == 0
