from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from contextmine_core.exports import mermaid_c4 as mermaid_export
from contextmine_core.twin import GraphProjection


class _Result:
    def __init__(self, scenario_name: str) -> None:
        self._scenario = SimpleNamespace(name=scenario_name)

    def scalar_one(self) -> SimpleNamespace:
        return self._scenario


class _FakeSession:
    async def execute(self, _stmt):  # noqa: ANN001
        return _Result("AS-IS")


@pytest.mark.anyio
async def test_mermaid_c4_uses_architecture_projection(monkeypatch: pytest.MonkeyPatch) -> None:
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

    content = await mermaid_export.export_mermaid_c4(_FakeSession(), uuid4())

    assert captured["projection"] == GraphProjection.ARCHITECTURE
    assert "Container(" in content
    assert " class " not in content
    assert " method " not in content
