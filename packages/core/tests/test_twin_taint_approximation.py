"""The graphrag (keyword-substring) taint engine must flag its output as approximate.

It matches keyword lists against symbol *names*; it is not data-flow analysis. The
output is tagged so consumers don't mistake it for sound taint findings.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from contextmine_core.twin import ops

pytestmark = pytest.mark.anyio


@pytest.fixture
def _stub_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the analysis cache + DB lookups so the _compute closures run directly."""
    scenario = SimpleNamespace(id=uuid.uuid4())

    async def fake_cache(session, *, compute, **_kwargs):  # noqa: ANN001
        return await compute(session, scenario)

    async def fake_pattern_nodes(session, *, scenario_id, patterns, limit):  # noqa: ANN001
        return []

    monkeypatch.setattr(ops, "_with_analysis_cache", fake_cache)
    monkeypatch.setattr(ops, "_find_pattern_nodes", fake_pattern_nodes)


async def test_taint_sources_marked_approximation(_stub_db: None) -> None:
    result = await ops.find_taint_sources(
        object(),
        collection_id=uuid.uuid4(),
        scenario_id=None,
        language="python",
        limit=10,
        cache_ttl_seconds=0,
    )
    assert result["approximation"] is True
    assert "data-flow" in result["approximation_reason"]


async def test_taint_sinks_marked_approximation(_stub_db: None) -> None:
    result = await ops.find_taint_sinks(
        object(),
        collection_id=uuid.uuid4(),
        scenario_id=None,
        language="python",
        limit=10,
        cache_ttl_seconds=0,
    )
    assert result["approximation"] is True
    assert "data-flow" in result["approximation_reason"]


async def test_taint_flows_marked_approximation(_stub_db: None) -> None:
    result = await ops.find_taint_flows(
        object(),
        collection_id=uuid.uuid4(),
        scenario_id=None,
        language="python",
        max_hops=4,
        max_results=10,
        cache_ttl_seconds=0,
    )
    assert result["approximation"] is True
    assert "data-flow" in result["approximation_reason"]
