"""Tests for AGE read-only cypher guard."""

from __future__ import annotations

import pytest
from contextmine_core.graph.age import ensure_read_only_cypher


def test_read_only_query_allowed() -> None:
    ensure_read_only_cypher("MATCH (n:Node) RETURN n LIMIT 10")


@pytest.mark.parametrize(
    "query",
    [
        "CREATE (n:Node {id:'1'})",
        "MATCH (n) SET n.x = 1 RETURN n",
        "MATCH (n) DELETE n",
        "MERGE (n:Node {id:'1'}) RETURN n",
    ],
)
def test_mutating_query_blocked(query: str) -> None:
    with pytest.raises(ValueError):
        ensure_read_only_cypher(query)
