from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.exports import mermaid_c4 as mermaid_export
from contextmine_core.twin import GraphProjection


class _Result:
    def __init__(self, scenario_name: str, collection_id=None) -> None:
        self._scenario = SimpleNamespace(name=scenario_name, collection_id=collection_id or uuid4())

    def scalar_one(self) -> SimpleNamespace:
        return self._scenario


class _Scalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _ScalarResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return _Scalars(self._values)


class _FakeSession:
    def __init__(self, jobs=None):
        self.jobs = jobs or []

    async def execute(self, stmt):  # noqa: ANN001
        statement = str(stmt)
        if "knowledge_nodes" in statement:
            return _ScalarResult(self.jobs)
        return _Result("AS-IS", collection_id=uuid4())


@pytest.mark.anyio
async def test_mermaid_c4_container_uses_architecture_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_get_full_scenario_graph(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "nodes": [
                {
                    "id": "arch:container:1",
                    "natural_key": "container|billing|api|",
                    "kind": "container",
                    "name": "api",
                    "meta": {"member_count": 3},
                }
            ],
            "edges": [],
            "total_nodes": 1,
            "projection": "architecture",
            "entity_level": "container",
            "grouping_strategy": "heuristic",
            "excluded_kinds": [],
        }

    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_get_full_scenario_graph)

    content = await mermaid_export.export_mermaid_c4(_FakeSession(), uuid4(), c4_view="container")

    assert captured["projection"] == GraphProjection.ARCHITECTURE
    assert captured["include_kinds"] == {"file"}
    assert "C4Container" in content
    assert "Container(" in content


@pytest.mark.anyio
async def test_component_view_renders_components(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_get_full_scenario_graph(**kwargs):  # noqa: ANN003
        return {
            "nodes": [
                {
                    "id": "arch:component:1",
                    "natural_key": "component|billing|api|invoice",
                    "kind": "component",
                    "name": "invoice",
                    "meta": {"container": "api", "component": "invoice", "member_count": 5},
                }
            ],
            "edges": [],
            "total_nodes": 1,
            "projection": "architecture",
            "entity_level": "component",
            "grouping_strategy": "heuristic",
            "excluded_kinds": [],
        }

    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_get_full_scenario_graph)
    result = await mermaid_export.export_mermaid_c4_result(
        _FakeSession(),
        uuid4(),
        c4_view="component",
        c4_scope="api",
    )

    assert result.c4_view == "component"
    assert "C4Component" in result.content
    assert "Component(" in result.content
    assert result.warnings == []


@pytest.mark.anyio
async def test_code_view_warns_on_calls_fallback_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    symbol_nodes = [
        {
            "id": f"s{index}",
            "natural_key": f"symbol:Node{index}",
            "kind": "method",
            "name": f"node_{index}",
            "meta": {"file_path": "services/billing/api/invoice.py"},
        }
        for index in range(1, 13)
    ]
    symbol_edges = [
        {
            "id": f"e{index}",
            "source_node_id": f"s{index}",
            "target_node_id": f"s{index + 1}",
            "kind": "symbol_references_symbol",
            "meta": {},
        }
        for index in range(1, 12)
    ]

    async def _fake_get_full_scenario_graph(**kwargs):  # noqa: ANN003
        projection = kwargs.get("projection")
        if projection == GraphProjection.CODE_FILE:
            return {
                "nodes": [
                    {
                        "id": "f1",
                        "natural_key": "file:services/billing/api/invoice.py",
                        "kind": "file",
                        "name": "services/billing/api/invoice.py",
                        "meta": {},
                    }
                ],
                "edges": [],
                "total_nodes": 1,
                "projection": "code_file",
                "entity_level": "file",
                "grouping_strategy": "heuristic",
                "excluded_kinds": [],
            }

        return {
            "nodes": symbol_nodes,
            "edges": symbol_edges,
            "total_nodes": len(symbol_nodes),
            "projection": "code_symbol",
            "entity_level": "symbol",
            "grouping_strategy": "heuristic",
            "excluded_kinds": [],
        }

    monkeypatch.setattr(mermaid_export, "get_full_scenario_graph", _fake_get_full_scenario_graph)

    result = await mermaid_export.export_mermaid_c4_result(
        _FakeSession(),
        uuid4(),
        c4_view="code",
        max_nodes=10,
    )

    assert "C4Component" in result.content
    assert any(
        "symbol_calls_symbol" in warning or "fallback" in warning for warning in result.warnings
    )
    assert any("limited" in warning.lower() for warning in result.warnings)


@pytest.mark.anyio
async def test_context_view_sparse_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_surface_counts(_session, _collection_id):
        return {
            "api_endpoint": 0,
            "graphql_operation": 0,
            "service_rpc": 0,
            "job": 0,
            "message_schema": 0,
            "db_table": 0,
        }

    monkeypatch.setattr(mermaid_export, "_surface_counts", _fake_surface_counts)

    result = await mermaid_export._render_context_view(
        session=_FakeSession(),
        collection_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
    )

    assert "C4Context" in result.content
    assert result.warnings


@pytest.mark.anyio
async def test_deployment_view_sparse_emits_warning() -> None:
    result = await mermaid_export._render_deployment_view(
        session=_FakeSession(jobs=[]),
        collection_id=uuid4(),
        scenario_id=uuid4(),
        scenario_name="AS-IS",
        c4_scope=None,
        max_nodes=50,
    )

    assert "C4Deployment" in result.content
    assert result.warnings
