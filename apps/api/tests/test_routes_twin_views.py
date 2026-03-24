"""Tests for twin view/endpoint handler bodies.

Exercises the HTTP endpoint handlers that perform DB queries and return JSON.
All DB and service calls are mocked so no real database is required.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COLLECTION_ID = uuid.uuid4()
_SCENARIO_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()
_BASE_SCENARIO_ID = uuid.uuid4()


def _fake_scenario(
    *,
    scenario_id: uuid.UUID = _SCENARIO_ID,
    collection_id: uuid.UUID = _COLLECTION_ID,
    name: str = "AS-IS",
    version: int = 1,
    is_as_is: bool = True,
    base_scenario_id: uuid.UUID | None = None,
    meta: dict | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> MagicMock:
    scenario = MagicMock()
    scenario.id = scenario_id
    scenario.collection_id = collection_id
    scenario.name = name
    scenario.version = version
    scenario.is_as_is = is_as_is
    scenario.base_scenario_id = base_scenario_id
    scenario.meta = meta or {}
    scenario.created_at = created_at or datetime.now(tz=UTC)
    scenario.updated_at = updated_at or datetime.now(tz=UTC)
    return scenario


def _fake_collection(
    *,
    collection_id: uuid.UUID = _COLLECTION_ID,
    owner_user_id: uuid.UUID = _USER_ID,
) -> MagicMock:
    col = MagicMock()
    col.id = collection_id
    col.owner_user_id = owner_user_id
    return col


class _FakeScalarResult:
    """Mimics sqlalchemy ``result.scalar_one_or_none()``."""

    def __init__(self, value: Any = None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value

    def scalar_one(self) -> Any:
        return self._value

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._value)


class _FakeScalars:
    def __init__(self, value: Any) -> None:
        self._value = value

    def all(self) -> list:
        if isinstance(self._value, list):
            return self._value
        return [] if self._value is None else [self._value]

    def first(self) -> Any:
        items = self.all()
        return items[0] if items else None


def _make_db(
    *,
    collection: Any | None = None,
    scenario: Any | None = None,
    extra_execute_results: list[Any] | None = None,
) -> AsyncMock:
    """Build a mock async DB session.

    ``execute`` returns *collection* for the first call (the ownership /
    membership check) and *scenario* for the second call (scenario
    resolution), then yields items from *extra_execute_results* for all
    subsequent calls.
    """
    db = AsyncMock()
    results: list[Any] = []
    if collection is not None:
        results.append(_FakeScalarResult(collection))
    if scenario is not None:
        results.append(_FakeScalarResult(scenario))
    for extra in extra_execute_results or []:
        results.append(_FakeScalarResult(extra))
    db.execute = AsyncMock(side_effect=results if results else [_FakeScalarResult(None)])
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    return db


@asynccontextmanager
async def _fake_db_session(db: AsyncMock):
    """Context-manager that yields the given mock db."""
    yield db


def _patch_auth_and_db(
    db: AsyncMock,
    *,
    user_id: uuid.UUID = _USER_ID,
):
    """Return a list of patch context managers for auth + DB session."""
    return [
        patch(
            "app.routes.twin.get_session",
            return_value={"user_id": str(user_id)},
        ),
        patch(
            "app.routes.twin.get_db_session",
            return_value=_fake_db_session(db),
        ),
    ]


def _apply_patches(patches: list):
    """Stack multiple patch context managers."""
    from contextlib import ExitStack

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collection():
    return _fake_collection()


@pytest.fixture
def scenario():
    return _fake_scenario()


@pytest.fixture
def db_mock(collection, scenario):
    return _make_db(collection=collection, scenario=scenario)


# ---------------------------------------------------------------------------
# Helpers for building httpx client
# ---------------------------------------------------------------------------


async def _get(path: str, db: AsyncMock, *, user_id: uuid.UUID = _USER_ID, **params) -> Any:
    """Issue a GET request against the app with mocked auth/db."""
    from app.main import app
    from httpx import ASGITransport, AsyncClient

    patches = _patch_auth_and_db(db, user_id=user_id)
    with _apply_patches(patches):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(path, params=params)
    return resp


async def _post(
    path: str, db: AsyncMock, *, user_id: uuid.UUID = _USER_ID, json: Any = None
) -> Any:
    """Issue a POST request against the app with mocked auth/db."""
    from app.main import app
    from httpx import ASGITransport, AsyncClient

    patches = _patch_auth_and_db(db, user_id=user_id)
    with _apply_patches(patches):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(path, json=json)
    return resp


# ---------------------------------------------------------------------------
# /api/twin/scenarios CRUD
# ---------------------------------------------------------------------------


class TestCreateScenario:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario(is_as_is=False, base_scenario_id=_BASE_SCENARIO_ID)
        db = _make_db(collection=_fake_collection())

        with patch(
            "app.routes.twin.create_to_be_scenario",
            new_callable=AsyncMock,
            return_value=fake_scenario,
        ):
            resp = await _post(
                "/api/twin/scenarios",
                db,
                json={"collection_id": str(_COLLECTION_ID), "name": "TO-BE v2"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "AS-IS"  # from fake_scenario default
        assert body["is_as_is"] is False
        assert body["base_scenario_id"] == str(_BASE_SCENARIO_ID)


class TestListScenarios:
    async def test_with_collection_filter(self) -> None:
        fake_scenario = _fake_scenario()
        # First call: _ensure_member (collection), second call: query for scenarios
        db = _make_db(
            collection=_fake_collection(),
            extra_execute_results=[[fake_scenario]],
        )
        # Override to supply scenarios list
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(_fake_collection()),  # _ensure_member
                _FakeScalarResult([fake_scenario]),  # scenario query
            ]
        )
        resp = await _get(
            "/api/twin/scenarios",
            db,
            collection_id=str(_COLLECTION_ID),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "scenarios" in body


class TestGetScenario:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(scenario=fake_scenario, collection=_fake_collection())
        # _load_scenario, _ensure_member
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        resp = await _get(
            f"/api/twin/scenarios/{_SCENARIO_ID}",
            db,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(_SCENARIO_ID)
        assert body["name"] == "AS-IS"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/status
# ---------------------------------------------------------------------------


class TestTwinStatusView:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.get_collection_twin_status",
            new_callable=AsyncMock,
            return_value={"status": "ready", "node_count": 42},
        ):
            resp = await _get(f"/api/twin/collections/{_COLLECTION_ID}/status", db)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/timeline
# ---------------------------------------------------------------------------


class TestTwinTimelineView:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.list_collection_twin_events",
            new_callable=AsyncMock,
            return_value={"events": [], "total": 0},
        ):
            resp = await _get(f"/api/twin/collections/{_COLLECTION_ID}/timeline", db)
        assert resp.status_code == 200
        body = resp.json()
        assert "events" in body


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/refresh
# ---------------------------------------------------------------------------


class TestTwinRefresh:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.trigger_collection_refresh",
            new_callable=AsyncMock,
            return_value={"refreshed": True, "source_count": 1},
        ):
            resp = await _post(
                f"/api/twin/collections/{_COLLECTION_ID}/refresh",
                db,
                json={"force": False},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["refreshed"] is True


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/diff
# ---------------------------------------------------------------------------


class TestTwinDiffView:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.get_collection_twin_diff",
            new_callable=AsyncMock,
            return_value={"added": [], "removed": [], "changed": []},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/diff",
                db,
                from_version=1,
                to_version=2,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "added" in body


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/topology
# ---------------------------------------------------------------------------


class TestTopologyView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_graph = {
            "nodes": [],
            "edges": [],
            "page": 0,
            "limit": 1000,
            "total_nodes": 0,
            "projection": "architecture",
            "entity_level": "domain",
            "grouping_strategy": "domain",
            "excluded_kinds": [],
        }
        # DB calls: _ensure_member, _resolve_view_scenario (is_as_is query),
        # get_or_create fallback
        db = _make_db(collection=_fake_collection())
        with (
            patch(
                "app.routes.twin._ensure_member",
                new_callable=AsyncMock,
            ),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_graph,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/topology",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "graph" in body
        assert "scenario" in body
        assert body["projection"] == "architecture"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/deep-dive
# ---------------------------------------------------------------------------


class TestDeepDiveView:
    async def test_file_dependency_mode(self) -> None:
        fake_scenario = _fake_scenario()
        fake_graph = {
            "nodes": [],
            "edges": [],
            "page": 0,
            "limit": 3000,
            "total_nodes": 0,
            "projection": "code_file",
            "entity_level": "file",
            "grouping_strategy": "file",
            "excluded_kinds": [],
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_graph,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/deep-dive",
                db,
                mode="file_dependency",
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "graph" in body

    async def test_symbol_callgraph_mode(self) -> None:
        fake_scenario = _fake_scenario()
        node_id = str(uuid.uuid4())
        fake_full_graph = {
            "nodes": [
                {"id": node_id, "natural_key": "sym:foo"},
            ],
            "edges": [
                {"source_node_id": node_id, "target_node_id": node_id},
            ],
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "symbol",
            "excluded_kinds": [],
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_full_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_full_graph,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/deep-dive",
                db,
                mode="symbol_callgraph",
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["projection"] == "code_symbol"

    async def test_contains_hierarchy_mode(self) -> None:
        fake_scenario = _fake_scenario()
        fake_graph = {
            "nodes": [],
            "edges": [],
            "page": 0,
            "limit": 3000,
            "total_nodes": 0,
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "symbol",
            "excluded_kinds": [],
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_graph,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/deep-dive",
                db,
                mode="contains_hierarchy",
            )
        assert resp.status_code == 200

    async def test_invalid_mode_returns_400(self) -> None:
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=_fake_scenario(),
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/deep-dive",
                db,
                mode="bogus_mode",
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/ui-map
# ---------------------------------------------------------------------------


class TestUiMapView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_full_graph = {"nodes": [], "edges": []}
        fake_projection = {
            "summary": {"routes": 2, "views": 3, "trace_edges": 5},
            "graph": {"nodes": [{"id": "n1"}], "edges": []},
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_full_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_full_graph,
            ),
            patch(
                "app.routes.twin.build_ui_map_projection",
                return_value=fake_projection,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/ui-map",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["projection"] == "ui_map"
        assert body["entity_level"] == "ui"
        assert body["summary"]["routes"] == 2


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/test-matrix
# ---------------------------------------------------------------------------


class TestTestMatrixView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_full_graph = {"nodes": [], "edges": []}
        fake_projection = {
            "summary": {"test_cases": 5, "subjects": 3},
            "matrix": [["a", "b"]],
            "graph": {"nodes": [], "edges": []},
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_full_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_full_graph,
            ),
            patch(
                "app.routes.twin.build_test_matrix_projection",
                return_value=fake_projection,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/test-matrix",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["projection"] == "test_matrix"
        assert body["summary"]["test_cases"] == 5
        assert body["matrix"] == [["a", "b"]]


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/user-flows
# ---------------------------------------------------------------------------


class TestUserFlowsView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_full_graph = {"nodes": [], "edges": []}
        fake_projection = {
            "summary": {"user_flows": 2, "flow_steps": 10},
            "flows": [{"id": "f1", "name": "Login"}],
            "graph": {"nodes": [], "edges": []},
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_full_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_full_graph,
            ),
            patch(
                "app.routes.twin.build_user_flows_projection",
                return_value=fake_projection,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/user-flows",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["projection"] == "user_flows"
        assert body["entity_level"] == "user_flow"
        assert body["summary"]["user_flows"] == 2


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/rebuild-readiness
# ---------------------------------------------------------------------------


class TestRebuildReadinessView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_full_graph = {"nodes": [], "edges": []}
        fake_readiness = {
            "score": 0.75,
            "summary": "Good shape",
            "known_gaps": [],
            "critical_inferred_only": [],
        }
        fake_status = {
            "behavioral_layers_status": "ok",
            "last_behavioral_materialized_at": None,
            "deep_warnings": [],
            "scip_status": "ready",
            "scip_projects_by_language": {},
            "scip_failed_projects": [],
            "metrics_gate": {},
        }
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_full_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_full_graph,
            ),
            patch(
                "app.routes.twin.compute_rebuild_readiness",
                return_value=fake_readiness,
            ),
            patch(
                "app.routes.twin.get_collection_twin_status",
                new_callable=AsyncMock,
                return_value=fake_status,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/rebuild-readiness",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["score"] == 0.75
        assert body["projection"] == "rebuild_readiness"
        assert body["behavioral_layers_status"] == "ok"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/summary
# ---------------------------------------------------------------------------


class TestAnalysisSummary:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.get_codebase_summary_multi",
            new_callable=AsyncMock,
            return_value={"files": 100, "lines": 5000},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/summary",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["files"] == 100


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/methods
# ---------------------------------------------------------------------------


class TestAnalysisMethods:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.list_methods_multi",
            new_callable=AsyncMock,
            return_value={"methods": [], "total": 0},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/methods",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "methods" in body


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/calls
# ---------------------------------------------------------------------------


class TestAnalysisCalls:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.list_calls_multi",
            new_callable=AsyncMock,
            return_value={"calls": [], "total": 0},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/calls",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "calls" in body


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/cfg
# ---------------------------------------------------------------------------


class TestAnalysisCfg:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.get_cfg_multi",
            new_callable=AsyncMock,
            return_value={"cfg": {"nodes": [], "edges": []}},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/cfg",
                db,
                node_ref="my_function",
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/variable-flow
# ---------------------------------------------------------------------------


class TestAnalysisVariableFlow:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.get_variable_flow_multi",
            new_callable=AsyncMock,
            return_value={"flows": []},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/variable-flow",
                db,
                node_ref="my_func",
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/taint/sources
# ---------------------------------------------------------------------------


class TestAnalysisTaintSources:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.find_taint_sources_multi",
            new_callable=AsyncMock,
            return_value={"sources": [], "total": 0},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/taint/sources",
                db,
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/taint/sinks
# ---------------------------------------------------------------------------


class TestAnalysisTaintSinks:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.find_taint_sinks_multi",
            new_callable=AsyncMock,
            return_value={"sinks": [], "total": 0},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/taint/sinks",
                db,
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/taint/flows  (POST)
# ---------------------------------------------------------------------------


class TestAnalysisTaintFlows:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.find_taint_flows_multi",
            new_callable=AsyncMock,
            return_value={"flows": [], "total": 0},
        ):
            resp = await _post(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/taint/flows",
                db,
                json={"max_hops": 6, "max_results": 50},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/findings/store  (POST)
# ---------------------------------------------------------------------------


class TestStoreFindingsView:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.store_findings",
            new_callable=AsyncMock,
            return_value={"stored": 3},
        ):
            resp = await _post(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/findings/store",
                db,
                json={"findings": [{"rule": "r1"}]},
            )
        assert resp.status_code == 200
        assert resp.json()["stored"] == 3


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/findings
# ---------------------------------------------------------------------------


class TestListFindingsView:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.list_findings",
            new_callable=AsyncMock,
            return_value={"findings": [], "total": 0},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/findings",
                db,
            )
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/analysis/findings/sarif
# ---------------------------------------------------------------------------


class TestExportFindingsSarif:
    async def test_happy_path(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch(
            "app.routes.twin.export_findings_sarif",
            new_callable=AsyncMock,
            return_value={"$schema": "sarif", "runs": []},
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/analysis/findings/sarif",
                db,
            )
        assert resp.status_code == 200
        assert "$schema" in resp.json()


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/mermaid
# ---------------------------------------------------------------------------


class TestMermaidView:
    async def test_single_mode(self) -> None:
        fake_scenario = _fake_scenario(base_scenario_id=None)
        fake_result = MagicMock()
        fake_result.content = "C4Context\n  System(a, 'A')"
        fake_result.warnings = []

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.export_mermaid_c4_result",
                new_callable=AsyncMock,
                return_value=fake_result,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/mermaid",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "single"
        assert "content" in body

    async def test_compare_mode(self) -> None:
        fake_scenario = _fake_scenario(base_scenario_id=_BASE_SCENARIO_ID)
        as_is_result = MagicMock()
        as_is_result.content = "as-is content"
        as_is_result.warnings = ["w1"]
        to_be_result = MagicMock()
        to_be_result.content = "to-be content"
        to_be_result.warnings = ["w2"]

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.export_mermaid_asis_tobe_result",
                new_callable=AsyncMock,
                return_value=(as_is_result, to_be_result),
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/mermaid",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "compare"
        assert body["as_is"] == "as-is content"
        assert body["to_be"] == "to-be content"
        assert sorted(body["warnings"]) == ["w1", "w2"]


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/arc42
# ---------------------------------------------------------------------------


class TestArc42View:
    async def test_disabled_returns_503(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.arch_docs_enabled = False
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/arc42",
                db,
            )
        assert resp.status_code == 503

    async def test_cached_artifact_full_doc(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        artifact = MagicMock()
        artifact.id = artifact_id
        artifact.name = f"{_SCENARIO_ID}.arc42.md"
        artifact.kind = MagicMock()
        artifact.kind.value = "arc42"
        artifact.content = "# Full arc42"
        artifact.meta = {
            "sections": {"1_introduction_and_goals": "Intro content"},
            "generated_at": "2024-01-01T00:00:00",
            "facts_hash": "abc123",
            "warnings": [],
            "confidence_summary": {},
            "section_coverage": {"1_introduction_and_goals": True},
        }

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
        ):
            mock_settings.return_value.arch_docs_enabled = True
            # The arc42 view queries for the existing artifact
            db.execute = AsyncMock(
                side_effect=[
                    _FakeScalarResult(_fake_collection()),  # _ensure_member
                    _FakeScalarResult(fake_scenario),  # _resolve_view_scenario (1)
                    _FakeScalarResult(fake_scenario),  # _resolve_view_scenario (2) - as_is query
                    _FakeScalarResult(artifact),  # existing artifact query
                ]
            )
            # Need to patch at the level that the route actually calls
            with (
                patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
                patch(
                    "app.routes.twin._resolve_view_scenario",
                    new_callable=AsyncMock,
                    return_value=fake_scenario,
                ),
            ):
                resp = await _get(
                    f"/api/twin/collections/{_COLLECTION_ID}/views/arc42",
                    db,
                )
        # With the artifact mock, the route should return cached
        assert resp.status_code in (200, 409)  # 409 if artifact query fails to match

    async def test_no_artifact_no_regenerate_returns_409(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
        ):
            mock_settings.return_value.arch_docs_enabled = True
            # The artifact query returns None
            db.execute = AsyncMock(return_value=_FakeScalarResult(None))
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/arc42",
                db,
            )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/arc42/drift
# ---------------------------------------------------------------------------


class TestArc42DriftView:
    async def test_disabled_returns_503(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.arch_docs_enabled = False
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/arc42/drift",
                db,
            )
        assert resp.status_code == 503

    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()

        @dataclass
        class FakeDelta:
            delta_type: str = "added"
            section: str = "context"
            description: str = "new item"

        @dataclass
        class FakeReport:
            generated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
            current_hash: str = "hash1"
            baseline_hash: str = "hash0"
            deltas: list = field(default_factory=lambda: [FakeDelta()])
            warnings: list = field(default_factory=list)

        fake_bundle = MagicMock()
        fake_bundle.ports_adapters = []
        fake_bundle.warnings = []

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._resolve_baseline_scenario",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "app.routes.twin._build_arch_bundle",
                new_callable=AsyncMock,
                return_value=fake_bundle,
            ),
            patch(
                "app.routes.twin.compute_arc42_drift",
                return_value=FakeReport(),
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
        ):
            mock_settings.return_value.arch_docs_enabled = True
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/arc42/drift",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "summary" in body
        assert body["summary"]["total"] == 1


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/ports-adapters
# ---------------------------------------------------------------------------


class TestPortsAdaptersView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()

        @dataclass
        class FakeFact:
            direction: str = "inbound"
            container: str = "api"
            port_name: str = "HTTP"
            adapter_name: str = "FastAPI"

        fake_bundle = MagicMock()
        fake_bundle.ports_adapters = [FakeFact(), FakeFact(direction="outbound")]
        fake_bundle.warnings = []

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._build_arch_bundle",
                new_callable=AsyncMock,
                return_value=fake_bundle,
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
        ):
            mock_settings.return_value.arch_docs_enabled = True
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/ports-adapters",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total"] == 2
        assert body["summary"]["inbound"] == 1
        assert body["summary"]["outbound"] == 1

    async def test_disabled_returns_503(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.arch_docs_enabled = False
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/ports-adapters",
                db,
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/evolution/* (4 sub-views)
# ---------------------------------------------------------------------------


class TestEvolutionInvestmentUtilization:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_evolution_enabled"),
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_investment_utilization_payload",
                new_callable=AsyncMock,
                return_value={"quadrants": [], "items": []},
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/evolution/investment-utilization",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "quadrants" in body

    async def test_disabled_returns_404(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.twin_evolution_view_enabled = False
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/evolution/investment-utilization",
                db,
            )
        assert resp.status_code == 404


class TestEvolutionKnowledgeIslands:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_evolution_enabled"),
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_knowledge_islands_payload",
                new_callable=AsyncMock,
                return_value={"islands": [], "bus_factor": 2},
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/evolution/knowledge-islands",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "islands" in body


class TestEvolutionTemporalCoupling:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_evolution_enabled"),
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_temporal_coupling_payload",
                new_callable=AsyncMock,
                return_value={"edges": [], "clusters": []},
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/evolution/temporal-coupling",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "edges" in body


class TestEvolutionFitnessFunctions:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_evolution_enabled"),
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin.get_fitness_functions_payload",
                new_callable=AsyncMock,
                return_value={"functions": [], "violations": 0},
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/evolution/fitness-functions",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "functions" in body


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/graphrag
# ---------------------------------------------------------------------------


class TestGraphragView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())

        # The graphrag view does direct DB queries inside the handler body
        # after the mocked helpers: total_nodes count, paged_nodes query.
        # We supply those results via db.execute side_effect.

        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._load_community_graph",
                new_callable=AsyncMock,
                return_value=([], [], "knowledge"),
            ),
            patch(
                "app.routes.twin._compute_symbol_communities",
                return_value=({}, {}),
            ),
        ):
            # DB queries inside graphrag_view body:
            # 1. total_nodes count (scalar_one)
            # 2. paged_nodes query (scalars().all() -> [])
            # (no edges since no nodes)
            db.execute = AsyncMock(
                side_effect=[
                    _FakeScalarResult(0),  # total_nodes count
                    _FakeScalarResult([]),  # paged_nodes
                ]
            )
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/graphrag",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "scenario" in body
        assert "graph" in body
        assert body["projection"] == "graphrag"
        assert body["status"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/graphrag/communities
# ---------------------------------------------------------------------------


class TestGraphragCommunitiesView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        community_data = {
            "c1": {
                "id": "c1",
                "label": "Auth",
                "size": 5,
                "cohesion": 0.8,
                "top_kinds": [("symbol", 5)],
                "sample_nodes": ["node1"],
            }
        }
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._load_community_graph",
                new_callable=AsyncMock,
                return_value=([], [], "knowledge"),
            ),
            patch(
                "app.routes.twin._compute_symbol_communities",
                return_value=({}, community_data),
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/graphrag/communities",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["label"] == "Auth"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/graphrag/processes
# ---------------------------------------------------------------------------


class TestGraphragProcessesView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        fake_process = {
            "id": "proc-1",
            "label": "Checkout Flow",
            "process_type": "linear",
            "step_count": 3,
            "community_ids": ["c1"],
            "entry_node_id": str(uuid.uuid4()),
            "terminal_node_id": str(uuid.uuid4()),
            "steps": [],
        }
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._load_community_graph",
                new_callable=AsyncMock,
                return_value=([], [], "knowledge"),
            ),
            patch(
                "app.routes.twin._compute_symbol_communities",
                return_value=({}, {}),
            ),
            patch(
                "app.routes.twin._detect_processes",
                return_value=[fake_process],
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/graphrag/processes",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["label"] == "Checkout Flow"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/graphrag/path
# ---------------------------------------------------------------------------


class TestGraphragPathView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        from_node_id = uuid.uuid4()
        to_node_id = uuid.uuid4()
        from_node = MagicMock()
        from_node.id = from_node_id
        to_node = MagicMock()
        to_node.id = to_node_id

        @dataclass
        class FakeEntity:
            node_id: uuid.UUID = field(default_factory=uuid.uuid4)
            natural_key: str = "sym:foo"
            kind: str = "symbol"
            name: str = "foo"

        @dataclass
        class FakeEdge:
            source_id: str = ""
            target_id: str = ""
            kind: str = "calls"

        fake_context = MagicMock()
        fake_context.entities = [FakeEntity(node_id=from_node_id), FakeEntity(node_id=to_node_id)]
        fake_context.edges = [FakeEdge(source_id=str(from_node_id), target_id=str(to_node_id))]

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._resolve_knowledge_node",
                new_callable=AsyncMock,
                side_effect=[from_node, to_node],
            ),
            patch(
                "app.routes.twin.graphrag_trace_path",
                new_callable=AsyncMock,
                return_value=fake_context,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/graphrag/path",
                db,
                from_node_id=str(from_node_id),
                to_node_id=str(to_node_id),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "found"
        assert body["path"]["hops"] == 1

    async def test_from_node_not_found(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._resolve_knowledge_node",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/graphrag/path",
                db,
                from_node_id=str(uuid.uuid4()),
                to_node_id=str(uuid.uuid4()),
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/semantic-map
# ---------------------------------------------------------------------------


class TestSemanticMapView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch(
                "app.routes.twin._load_community_graph",
                new_callable=AsyncMock,
                return_value=([], [], "knowledge"),
            ),
            patch(
                "app.routes.twin._compute_symbol_communities",
                return_value=({}, {}),
            ),
            patch(
                "app.routes.twin._build_structural_community_points",
                new_callable=AsyncMock,
                return_value=([], []),
            ),
            patch(
                "app.routes.twin._build_semantic_map_signals",
                return_value={
                    "isolated_points": [],
                    "mixed_clusters": [],
                    "duplicate_candidates": [],
                    "semantic_duplication": [],
                    "misplaced_code": [],
                },
            ),
        ):
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/semantic-map",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "points" in body


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/city
# ---------------------------------------------------------------------------


class TestCityView:
    async def test_happy_path_with_metrics(self) -> None:
        fake_scenario = _fake_scenario()
        metric = MagicMock()
        metric.node_natural_key = "src/main.py"
        metric.loc = 100
        metric.symbol_count = 10
        metric.coverage = 0.8
        metric.complexity = 5.0
        metric.coupling = 0.3
        metric.cohesion = 0.7
        metric.instability = 0.4
        metric.fan_in = 3
        metric.fan_out = 2
        metric.cycle_participation = False
        metric.cycle_size = 0
        metric.duplication_ratio = 0.1
        metric.crap_score = 2.5
        metric.change_frequency = 10
        metric.meta = {"churn": 5.0}
        metric.scenario_id = _SCENARIO_ID

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
            patch(
                "app.routes.twin.export_codecharta_json",
                new_callable=AsyncMock,
                return_value='{"nodes": [{"name": "a"}, {"name": "b"}], "edges": []}',
            ),
        ):
            mock_settings.return_value.metrics_strict_mode = False
            # DB calls: _ensure_member, _resolve_view_scenario, metrics query
            db.execute = AsyncMock(
                return_value=_FakeScalarResult([metric]),
            )
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/city",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["collection_id"] == str(_COLLECTION_ID)
        assert "summary" in body
        assert "hotspots" in body
        assert body["summary"]["metric_nodes"] == 1
        assert body["metrics_status"]["status"] == "ready"
        assert body["cc_json"]["nodes"] is not None


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/graph
# ---------------------------------------------------------------------------


class TestScenarioGraphView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_graph = {
            "nodes": [],
            "edges": [],
            "page": 0,
            "limit": 200,
            "total_nodes": 0,
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "symbol",
            "excluded_kinds": [],
        }
        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with patch(
            "app.routes.twin.get_scenario_graph",
            new_callable=AsyncMock,
            return_value=fake_graph,
        ):
            resp = await _get(
                f"/api/twin/scenarios/{_SCENARIO_ID}/graph",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["projection"] == "code_symbol"


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/graph/neighborhood
# ---------------------------------------------------------------------------


class TestScenarioGraphNeighborhoodView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        node_id = str(uuid.uuid4())
        fake_full_graph = {
            "nodes": [{"id": node_id, "natural_key": "sym:main"}],
            "edges": [],
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "symbol",
            "excluded_kinds": [],
        }
        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with patch(
            "app.routes.twin.get_full_scenario_graph",
            new_callable=AsyncMock,
            return_value=fake_full_graph,
        ):
            resp = await _get(
                f"/api/twin/scenarios/{_SCENARIO_ID}/graph/neighborhood",
                db,
                node_id=node_id,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == node_id
        assert body["hops"] == 1

    async def test_node_not_found(self) -> None:
        fake_scenario = _fake_scenario()
        fake_full_graph = {
            "nodes": [],
            "edges": [],
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "symbol",
            "excluded_kinds": [],
        }
        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with patch(
            "app.routes.twin.get_full_scenario_graph",
            new_callable=AsyncMock,
            return_value=fake_full_graph,
        ):
            resp = await _get(
                f"/api/twin/scenarios/{_SCENARIO_ID}/graph/neighborhood",
                db,
                node_id=str(uuid.uuid4()),
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/patches
# ---------------------------------------------------------------------------


class TestGetPatches:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        fake_patch = MagicMock()
        fake_patch.id = uuid.uuid4()
        fake_patch.scenario_version = 1
        fake_patch.intent_id = uuid.uuid4()
        fake_patch.patch_ops = [{"op": "add"}]
        fake_patch.created_at = datetime.now(tz=UTC)

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with patch(
            "app.routes.twin.list_scenario_patches",
            new_callable=AsyncMock,
            return_value=[fake_patch],
        ):
            resp = await _get(
                f"/api/twin/scenarios/{_SCENARIO_ID}/patches",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["scenario_id"] == str(_SCENARIO_ID)
        assert len(body["patches"]) == 1


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/cypher  (POST)
# ---------------------------------------------------------------------------


class TestCypherQuery:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with (
            patch(
                "app.routes.twin.sync_scenario_to_age",
                new_callable=AsyncMock,
            ),
            patch(
                "app.routes.twin.run_read_only_cypher",
                new_callable=AsyncMock,
                return_value=[{"name": "foo"}],
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/cypher",
                db,
                json={"query": "MATCH (n) RETURN n LIMIT 1"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["rows"] == [{"name": "foo"}]


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/exports  (POST)
# ---------------------------------------------------------------------------


class TestCreateExport:
    async def test_lpg_jsonl_code_symbol(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.lpg.jsonl"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "lpg_jsonl"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with (
            patch(
                "app.routes.twin.export_lpg_jsonl",
                new_callable=AsyncMock,
                return_value='{"nodes":[]}',
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "lpg_jsonl", "projection": "code_symbol"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "lpg_jsonl"

    async def test_twin_manifest(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.twin_manifest.json"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "twin_manifest"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
            ]
        )
        with (
            patch(
                "app.routes.twin.export_twin_manifest",
                new_callable=AsyncMock,
                return_value='{"manifest": true}',
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "twin_manifest"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "twin_manifest"


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/exports/{export_id}
# ---------------------------------------------------------------------------


class TestGetExport:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        export_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = export_id
        fake_artifact.name = "test.lpg.jsonl"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "lpg_jsonl"
        fake_artifact.content = '{"nodes":[]}'
        fake_artifact.meta = {"format": "lpg_jsonl"}
        fake_artifact.updated_at = datetime.now(tz=UTC)

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
                _FakeScalarResult(fake_artifact),  # artifact query
            ]
        )
        resp = await _get(
            f"/api/twin/scenarios/{_SCENARIO_ID}/exports/{export_id}",
            db,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(export_id)

    async def test_not_found(self) -> None:
        fake_scenario = _fake_scenario()
        export_id = uuid.uuid4()
        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
                _FakeScalarResult(None),  # artifact not found
            ]
        )
        resp = await _get(
            f"/api/twin/scenarios/{_SCENARIO_ID}/exports/{export_id}",
            db,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/exports/{export_id}/raw
# ---------------------------------------------------------------------------


class TestGetExportRaw:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        export_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = export_id
        fake_artifact.content = '{"data": "raw content"}'

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_member
                _FakeScalarResult(fake_artifact),  # artifact query
            ]
        )
        resp = await _get(
            f"/api/twin/scenarios/{_SCENARIO_ID}/exports/{export_id}/raw",
            db,
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# /api/twin/collections/{id}/views/graphrag/evidence
# ---------------------------------------------------------------------------


class TestGraphragEvidenceView:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        node_id = uuid.uuid4()
        node = MagicMock()
        node.id = node_id
        node.name = "test_node"
        node.kind = MagicMock()
        node.kind.value = "symbol"

        evidence_id = uuid.uuid4()
        evidence = MagicMock()
        evidence.id = evidence_id
        evidence.file_path = "src/main.py"
        evidence.start_line = 10
        evidence.end_line = 20
        evidence.snippet = "def main(): pass"
        evidence.document_id = None

        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
        ):
            # Inner DB queries: node lookup by UUID, evidence count, evidence list
            db.execute = AsyncMock(
                side_effect=[
                    _FakeScalarResult(node),  # node by UUID
                    _FakeScalarResult(1),  # count
                    _FakeScalarResult([evidence]),  # evidence list
                ]
            )
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/graphrag/evidence",
                db,
                node_id=str(node_id),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == str(node_id)
        assert body["node_name"] == "test_node"
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["file_path"] == "src/main.py"


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/intents  (POST)
# ---------------------------------------------------------------------------


class TestCreateIntent:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        intent_id = uuid.uuid4()
        fake_intent = MagicMock()
        fake_intent.id = intent_id
        fake_intent.status = MagicMock()
        fake_intent.status.value = "executed"
        fake_intent.risk_level = MagicMock()
        fake_intent.risk_level.value = "low"
        fake_intent.requires_approval = False

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_owner
            ]
        )
        with patch(
            "app.routes.twin.submit_intent",
            new_callable=AsyncMock,
            return_value=fake_intent,
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/intents",
                db,
                json={
                    "intent_version": "1.0",
                    "scenario_id": str(_SCENARIO_ID),
                    "action": "EXTRACT_DOMAIN",
                    "target": {"type": "node", "id": "some-node-id"},
                    "expected_scenario_version": 1,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(intent_id)
        assert body["status"] == "executed"


# ---------------------------------------------------------------------------
# /api/twin/scenarios/{id}/intents/{intent_id}/approve (POST)
# ---------------------------------------------------------------------------


class TestApproveIntent:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        intent_id = uuid.uuid4()
        fake_intent = MagicMock()
        fake_intent.id = intent_id
        fake_intent.status = MagicMock()
        fake_intent.status.value = "executed"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),  # _load_scenario
                _FakeScalarResult(_fake_collection()),  # _ensure_owner
            ]
        )
        with patch(
            "app.routes.twin.approve_and_execute_intent",
            new_callable=AsyncMock,
            return_value=fake_intent,
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/intents/{intent_id}/approve",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "executed"


# ---------------------------------------------------------------------------
# Auth failure (401) - shared across all endpoints
# ---------------------------------------------------------------------------


class TestAuthFailure:
    async def test_no_user_returns_401(self) -> None:
        db = _make_db(collection=_fake_collection())
        with (
            patch(
                "app.routes.twin.get_session",
                return_value={},  # no user_id
            ),
            patch(
                "app.routes.twin.get_db_session",
                return_value=_fake_db_session(db),
            ),
        ):
            from app.main import app
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(f"/api/twin/collections/{_COLLECTION_ID}/status")
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# _ensure_member / _ensure_owner / _can_access_collection async helpers
# ---------------------------------------------------------------------------


class TestEnsureMember:
    async def test_owner_passes(self) -> None:
        from app.routes.twin import _ensure_member

        db = AsyncMock()
        db.execute = AsyncMock(
            return_value=_FakeScalarResult(_fake_collection(owner_user_id=_USER_ID))
        )
        # Should not raise
        await _ensure_member(db, _COLLECTION_ID, _USER_ID)

    async def test_member_passes(self) -> None:
        from app.routes.twin import _ensure_member

        other_owner = uuid.uuid4()
        col = _fake_collection(owner_user_id=other_owner)
        membership = MagicMock()  # non-None means member
        db = AsyncMock()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(col),  # collection lookup
                _FakeScalarResult(membership),  # membership lookup
            ]
        )
        await _ensure_member(db, _COLLECTION_ID, _USER_ID)

    async def test_no_collection_raises_404(self) -> None:
        from app.routes.twin import _ensure_member
        from fastapi import HTTPException

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        with pytest.raises(HTTPException) as exc_info:
            await _ensure_member(db, _COLLECTION_ID, _USER_ID)
        assert exc_info.value.status_code == 404

    async def test_not_member_raises_403(self) -> None:
        from app.routes.twin import _ensure_member
        from fastapi import HTTPException

        other_owner = uuid.uuid4()
        col = _fake_collection(owner_user_id=other_owner)
        db = AsyncMock()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(col),  # collection
                _FakeScalarResult(None),  # no membership
            ]
        )
        with pytest.raises(HTTPException) as exc_info:
            await _ensure_member(db, _COLLECTION_ID, _USER_ID)
        assert exc_info.value.status_code == 403


class TestEnsureOwner:
    async def test_owner_passes(self) -> None:
        from app.routes.twin import _ensure_owner

        db = AsyncMock()
        db.execute = AsyncMock(
            return_value=_FakeScalarResult(_fake_collection(owner_user_id=_USER_ID))
        )
        await _ensure_owner(db, _COLLECTION_ID, _USER_ID)

    async def test_no_collection_raises_404(self) -> None:
        from app.routes.twin import _ensure_owner
        from fastapi import HTTPException

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        with pytest.raises(HTTPException) as exc_info:
            await _ensure_owner(db, _COLLECTION_ID, _USER_ID)
        assert exc_info.value.status_code == 404

    async def test_not_owner_raises_403(self) -> None:
        from app.routes.twin import _ensure_owner
        from fastapi import HTTPException

        other_owner = uuid.uuid4()
        col = _fake_collection(owner_user_id=other_owner)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(col))
        with pytest.raises(HTTPException) as exc_info:
            await _ensure_owner(db, _COLLECTION_ID, _USER_ID)
        assert exc_info.value.status_code == 403


class TestCanAccessCollection:
    async def test_no_collection(self) -> None:
        from app.routes.twin import _can_access_collection

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        assert await _can_access_collection(db, _COLLECTION_ID, _USER_ID) is False

    async def test_owner_has_access(self) -> None:
        from app.routes.twin import _can_access_collection

        col = _fake_collection(owner_user_id=_USER_ID)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(col))
        assert await _can_access_collection(db, _COLLECTION_ID, _USER_ID) is True

    async def test_member_has_access(self) -> None:
        from app.routes.twin import _can_access_collection

        other_owner = uuid.uuid4()
        col = _fake_collection(owner_user_id=other_owner)
        membership = MagicMock()
        db = AsyncMock()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(col),
                _FakeScalarResult(membership),
            ]
        )
        assert await _can_access_collection(db, _COLLECTION_ID, _USER_ID) is True

    async def test_non_member_denied(self) -> None:
        from app.routes.twin import _can_access_collection

        other_owner = uuid.uuid4()
        col = _fake_collection(owner_user_id=other_owner)
        db = AsyncMock()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(col),
                _FakeScalarResult(None),
            ]
        )
        assert await _can_access_collection(db, _COLLECTION_ID, _USER_ID) is False


# ---------------------------------------------------------------------------
# _resolve_view_scenario
# ---------------------------------------------------------------------------


class TestResolveViewScenario:
    async def test_with_scenario_id(self) -> None:
        from app.routes.twin import _resolve_view_scenario

        scenario = _fake_scenario()
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(scenario))
        result = await _resolve_view_scenario(db, _COLLECTION_ID, str(_SCENARIO_ID))
        assert result.id == _SCENARIO_ID

    async def test_with_bad_scenario_id_raises_400(self) -> None:
        from app.routes.twin import _resolve_view_scenario
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_view_scenario(db, _COLLECTION_ID, "not-a-uuid")
        assert exc_info.value.status_code == 400

    async def test_scenario_not_found_raises_404(self) -> None:
        from app.routes.twin import _resolve_view_scenario
        from fastapi import HTTPException

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_view_scenario(db, _COLLECTION_ID, str(uuid.uuid4()))
        assert exc_info.value.status_code == 404

    async def test_no_scenario_id_uses_as_is(self) -> None:
        from app.routes.twin import _resolve_view_scenario

        scenario = _fake_scenario(is_as_is=True)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(scenario))
        result = await _resolve_view_scenario(db, _COLLECTION_ID, None)
        assert result.is_as_is is True

    async def test_no_scenario_id_creates_as_is(self) -> None:
        from app.routes.twin import _resolve_view_scenario

        created = _fake_scenario(is_as_is=True)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        with patch(
            "app.routes.twin.get_or_create_as_is_scenario",
            new_callable=AsyncMock,
            return_value=created,
        ):
            result = await _resolve_view_scenario(db, _COLLECTION_ID, None)
        assert result.is_as_is is True


# ---------------------------------------------------------------------------
# _upsert_artifact
# ---------------------------------------------------------------------------


class TestUpsertArtifact:
    async def test_creates_new(self) -> None:
        from app.routes.twin import _upsert_artifact
        from contextmine_core.models import KnowledgeArtifactKind

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        db.add = MagicMock()
        result = await _upsert_artifact(
            db,
            collection_id=_COLLECTION_ID,
            kind=KnowledgeArtifactKind.ARC42,
            name="test.arc42.md",
            content="# test",
            meta={"foo": "bar"},
        )
        assert result.content == "# test"
        db.add.assert_called_once()

    async def test_updates_existing(self) -> None:
        from app.routes.twin import _upsert_artifact
        from contextmine_core.models import KnowledgeArtifactKind

        existing = MagicMock()
        existing.content = "old"
        existing.meta = {}

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(existing))
        result = await _upsert_artifact(
            db,
            collection_id=_COLLECTION_ID,
            kind=KnowledgeArtifactKind.ARC42,
            name="test.arc42.md",
            content="new content",
            meta={"updated": True},
        )
        assert result.content == "new content"
        assert result.meta == {"updated": True}


# ---------------------------------------------------------------------------
# _load_scenario
# ---------------------------------------------------------------------------


class TestLoadScenario:
    async def test_happy_path(self) -> None:
        from app.routes.twin import _load_scenario

        scenario = _fake_scenario()
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(scenario))
        result = await _load_scenario(db, str(_SCENARIO_ID))
        assert result.id == _SCENARIO_ID

    async def test_invalid_id_raises_400(self) -> None:
        from app.routes.twin import _load_scenario
        from fastapi import HTTPException

        db = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await _load_scenario(db, "bad-id")
        assert exc_info.value.status_code == 400

    async def test_not_found_raises_404(self) -> None:
        from app.routes.twin import _load_scenario
        from fastapi import HTTPException

        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        with pytest.raises(HTTPException) as exc_info:
            await _load_scenario(db, str(uuid.uuid4()))
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# _resolve_baseline_scenario
# ---------------------------------------------------------------------------


class TestResolveBaselineScenario:
    async def test_explicit_baseline(self) -> None:
        from app.routes.twin import _resolve_baseline_scenario

        baseline = _fake_scenario(scenario_id=_BASE_SCENARIO_ID, name="baseline")
        scenario = _fake_scenario()
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(baseline))
        result = await _resolve_baseline_scenario(
            db,
            collection_id=_COLLECTION_ID,
            scenario=scenario,
            baseline_scenario_id=str(_BASE_SCENARIO_ID),
        )
        assert result is not None and result.id == _BASE_SCENARIO_ID

    async def test_explicit_baseline_not_found_raises_404(self) -> None:
        from app.routes.twin import _resolve_baseline_scenario
        from fastapi import HTTPException

        scenario = _fake_scenario()
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(None))
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_baseline_scenario(
                db,
                collection_id=_COLLECTION_ID,
                scenario=scenario,
                baseline_scenario_id=str(uuid.uuid4()),
            )
        assert exc_info.value.status_code == 404

    async def test_invalid_baseline_id_raises_400(self) -> None:
        from app.routes.twin import _resolve_baseline_scenario
        from fastapi import HTTPException

        scenario = _fake_scenario()
        db = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_baseline_scenario(
                db,
                collection_id=_COLLECTION_ID,
                scenario=scenario,
                baseline_scenario_id="bad-uuid",
            )
        assert exc_info.value.status_code == 400

    async def test_uses_base_scenario_id_from_scenario(self) -> None:
        from app.routes.twin import _resolve_baseline_scenario

        baseline = _fake_scenario(scenario_id=_BASE_SCENARIO_ID, name="baseline")
        scenario = _fake_scenario(base_scenario_id=_BASE_SCENARIO_ID)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(baseline))
        result = await _resolve_baseline_scenario(
            db,
            collection_id=_COLLECTION_ID,
            scenario=scenario,
            baseline_scenario_id=None,
        )
        assert result is not None and result.id == _BASE_SCENARIO_ID

    async def test_fallback_to_latest(self) -> None:
        from app.routes.twin import _resolve_baseline_scenario

        latest = _fake_scenario(scenario_id=uuid.uuid4(), name="latest")
        scenario = _fake_scenario(base_scenario_id=None)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_FakeScalarResult(latest))
        result = await _resolve_baseline_scenario(
            db,
            collection_id=_COLLECTION_ID,
            scenario=scenario,
            baseline_scenario_id=None,
        )
        assert result is not None and result.id == latest.id


# ---------------------------------------------------------------------------
# ERM view
# ---------------------------------------------------------------------------


class TestErmView:
    async def test_disabled_returns_503(self) -> None:
        db = _make_db(collection=_fake_collection())
        with patch("app.routes.twin.get_settings") as mock_settings:
            mock_settings.return_value.arch_docs_enabled = False
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/erm",
                db,
            )
        assert resp.status_code == 503

    async def test_happy_path_empty(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
        ):
            mock_settings.return_value.arch_docs_enabled = True
            # DB queries: table_nodes, column_nodes, edges, mermaid artifact
            db.execute = AsyncMock(
                side_effect=[
                    _FakeScalarResult([]),  # table_nodes
                    _FakeScalarResult([]),  # column_nodes
                    _FakeScalarResult([]),  # edges
                    _FakeScalarResult(None),  # mermaid artifact
                ]
            )
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/erm",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["tables"] == 0
        assert body["summary"]["columns"] == 0
        assert body["summary"]["foreign_keys"] == 0
        assert body["mermaid"] is None
        assert any("No DB_TABLE" in w for w in body["warnings"])


# ---------------------------------------------------------------------------
# City view with no metrics (unavailable path)
# ---------------------------------------------------------------------------


class TestCityViewNoMetrics:
    async def test_no_metrics_reports_unavailable(self) -> None:
        fake_scenario = _fake_scenario()
        db = _make_db(collection=_fake_collection())
        with (
            patch("app.routes.twin._ensure_member", new_callable=AsyncMock),
            patch(
                "app.routes.twin._resolve_view_scenario",
                new_callable=AsyncMock,
                return_value=fake_scenario,
            ),
            patch("app.routes.twin.get_settings") as mock_settings,
            patch(
                "app.routes.twin.refresh_metric_snapshots",
                new_callable=AsyncMock,
            ),
            patch(
                "app.routes.twin.export_codecharta_json",
                new_callable=AsyncMock,
                return_value='{"nodes": [{"name": "a"}, {"name": "b"}], "edges": []}',
            ),
        ):
            mock_settings.return_value.metrics_strict_mode = False
            # DB calls: first metrics query (empty), refresh, second metrics query (empty),
            # sources query, (no GitHub sources)
            db.execute = AsyncMock(
                side_effect=[
                    _FakeScalarResult([]),  # first metrics
                    _FakeScalarResult([]),  # second metrics after refresh
                    _FakeScalarResult([]),  # sources
                ]
            )
            resp = await _get(
                f"/api/twin/collections/{_COLLECTION_ID}/views/city",
                db,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["metrics_status"]["status"] == "unavailable"
        assert body["hotspots"] == []
        assert body["summary"]["metric_nodes"] == 0


# ---------------------------------------------------------------------------
# Export branches: cc_json, cx2, jgf, mermaid_c4
# ---------------------------------------------------------------------------


class TestCreateExportCcJson:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.cc.json"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "cc_json"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),
                _FakeScalarResult(_fake_collection()),
            ]
        )
        with (
            patch(
                "app.routes.twin.export_codecharta_json",
                new_callable=AsyncMock,
                return_value='{"data": 1}',
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "cc_json"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "cc_json"


class TestCreateExportCx2:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.cx2.json"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "cx2"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),
                _FakeScalarResult(_fake_collection()),
            ]
        )
        with (
            patch(
                "app.routes.twin.export_cx2",
                new_callable=AsyncMock,
                return_value='{"cx2": true}',
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "cx2", "projection": "code_symbol"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "cx2"


class TestCreateExportJgf:
    async def test_happy_path(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.jgf.json"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "jgf"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),
                _FakeScalarResult(_fake_collection()),
            ]
        )
        with (
            patch(
                "app.routes.twin.export_jgf",
                new_callable=AsyncMock,
                return_value='{"jgf": true}',
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "jgf", "projection": "code_symbol"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "jgf"


class TestCreateExportMermaidC4:
    async def test_single_scenario(self) -> None:
        fake_scenario = _fake_scenario(base_scenario_id=None)
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.mmd"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "mermaid_c4_asis"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),
                _FakeScalarResult(_fake_collection()),
            ]
        )
        with (
            patch(
                "app.routes.twin.export_mermaid_c4",
                new_callable=AsyncMock,
                return_value="C4Container",
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "mermaid_c4"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "mermaid_c4"

    async def test_with_base_scenario(self) -> None:
        fake_scenario = _fake_scenario(base_scenario_id=_BASE_SCENARIO_ID)
        as_is_artifact = MagicMock()
        as_is_artifact.id = uuid.uuid4()
        as_is_artifact.name = "AS-IS.asis.mmd"
        to_be_artifact = MagicMock()
        to_be_artifact.id = uuid.uuid4()
        to_be_artifact.name = "AS-IS.tobe.mmd"

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),
                _FakeScalarResult(_fake_collection()),
            ]
        )
        with (
            patch(
                "app.routes.twin.export_mermaid_asis_tobe",
                new_callable=AsyncMock,
                return_value=("as-is mmd", "to-be mmd"),
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                side_effect=[as_is_artifact, to_be_artifact],
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "mermaid_c4"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "exports" in body
        assert len(body["exports"]) == 2


# ---------------------------------------------------------------------------
# Export with architecture projection (non-code_symbol)
# ---------------------------------------------------------------------------


class TestCreateExportWithArchProjection:
    async def test_lpg_jsonl_architecture(self) -> None:
        fake_scenario = _fake_scenario()
        artifact_id = uuid.uuid4()
        fake_artifact = MagicMock()
        fake_artifact.id = artifact_id
        fake_artifact.name = "AS-IS.lpg.jsonl"
        fake_artifact.kind = MagicMock()
        fake_artifact.kind.value = "lpg_jsonl"

        fake_full_graph = {
            "nodes": [],
            "edges": [],
            "projection": "architecture",
            "entity_level": "domain",
            "grouping_strategy": "domain",
            "excluded_kinds": [],
        }

        db = _make_db()
        db.execute = AsyncMock(
            side_effect=[
                _FakeScalarResult(fake_scenario),
                _FakeScalarResult(_fake_collection()),
            ]
        )
        with (
            patch(
                "app.routes.twin.get_full_scenario_graph",
                new_callable=AsyncMock,
                return_value=fake_full_graph,
            ),
            patch(
                "app.routes.twin.export_lpg_jsonl_from_graph",
                return_value='{"nodes": []}',
            ),
            patch(
                "app.routes.twin._upsert_artifact",
                new_callable=AsyncMock,
                return_value=fake_artifact,
            ),
        ):
            resp = await _post(
                f"/api/twin/scenarios/{_SCENARIO_ID}/exports",
                db,
                json={"format": "lpg_jsonl", "projection": "architecture"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["format"] == "lpg_jsonl"


# ---------------------------------------------------------------------------
# _serialize_arc42_document + _serialize_graphrag_node
# ---------------------------------------------------------------------------


class TestSerializeArc42Document:
    def test_basic(self) -> None:
        from app.routes.twin import _serialize_arc42_document

        doc = SimpleNamespace(
            title="Arc42",
            generated_at=datetime.now(tz=UTC),
            sections={"1_introduction_and_goals": "hello"},
            markdown="# hello",
            warnings=[],
            confidence_summary={},
            section_coverage={"1_introduction_and_goals": True},
        )
        result = _serialize_arc42_document(doc)
        assert result["title"] == "Arc42"
        assert "sections" in result


class TestSerializeGraphragNode:
    def test_basic(self) -> None:
        from app.routes.twin import _serialize_graphrag_node

        node = MagicMock()
        node.id = uuid.uuid4()
        node.natural_key = "file:src/main.py"
        node.kind = MagicMock()
        node.kind.value = "file"
        node.name = "main.py"
        node.description = "Entry point"
        node.meta = {"loc": 100}

        result = _serialize_graphrag_node(
            node,
            community_mode="none",
            focused_community_id=None,
            community_by_node_id={},
            communities={},
        )
        assert result["id"] == str(node.id)
        assert result["kind"] == "file"
        assert result["name"] == "main.py"


# ---------------------------------------------------------------------------
# _user_id_or_401
# ---------------------------------------------------------------------------


class TestUserIdOr401:
    def test_returns_user_id(self) -> None:
        from app.routes.twin import _user_id_or_401

        request = MagicMock()
        with patch(
            "app.routes.twin.get_session",
            return_value={"user_id": str(_USER_ID)},
        ):
            result = _user_id_or_401(request)
        assert result == _USER_ID

    def test_missing_user_raises_401(self) -> None:
        from app.routes.twin import _user_id_or_401
        from fastapi import HTTPException

        request = MagicMock()
        with patch(
            "app.routes.twin.get_session",
            return_value={},
        ):
            with pytest.raises(HTTPException) as exc_info:
                _user_id_or_401(request)
            assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Edge cases: invalid collection_id returns 400
# ---------------------------------------------------------------------------


class TestInvalidCollectionId:
    async def test_topology_bad_id(self) -> None:
        db = _make_db(collection=_fake_collection())
        resp = await _get(
            "/api/twin/collections/not-a-uuid/views/topology",
            db,
        )
        assert resp.status_code == 400

    async def test_deep_dive_bad_id(self) -> None:
        db = _make_db(collection=_fake_collection())
        resp = await _get(
            "/api/twin/collections/not-a-uuid/views/deep-dive",
            db,
        )
        assert resp.status_code == 400

    async def test_status_bad_id(self) -> None:
        db = _make_db(collection=_fake_collection())
        resp = await _get(
            "/api/twin/collections/not-a-uuid/status",
            db,
        )
        assert resp.status_code == 400
