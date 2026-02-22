"""Unit tests for ports/adapters extraction helpers."""

from uuid import uuid4

from contextmine_core.architecture.facts import (
    _extract_inbound_ports,
    _extract_outbound_ports,
)
from contextmine_core.architecture.schemas import EvidenceRef
from contextmine_core.models import (
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)


def test_extract_inbound_ports_marks_deterministic_with_evidence() -> None:
    node_id = uuid4()
    node = KnowledgeNode(
        id=node_id,
        collection_id=uuid4(),
        kind=KnowledgeNodeKind.API_ENDPOINT,
        natural_key="endpoint:post:/orders",
        name="POST /orders",
        meta={"method": "POST", "path": "/orders"},
    )

    facts = _extract_inbound_ports(
        [node],
        {node_id: (EvidenceRef(kind="file", ref="services/orders/api/routes.py", start_line=10),)},
    )

    assert len(facts) == 1
    fact = facts[0]
    assert fact.direction == "inbound"
    assert fact.protocol == "http"
    assert fact.source == "deterministic"
    assert fact.container == "api"


def test_extract_outbound_ports_builds_adapter_mapping() -> None:
    symbol = KnowledgeNode(
        id=uuid4(),
        collection_id=uuid4(),
        kind=KnowledgeNodeKind.SYMBOL,
        natural_key="symbol:orders_repo",
        name="OrdersRepository",
        meta={"file_path": "services/orders/infra/repository.py"},
    )
    target = KnowledgeNode(
        id=uuid4(),
        collection_id=uuid4(),
        kind=KnowledgeNodeKind.DB_TABLE,
        natural_key="db:orders",
        name="orders",
        meta={},
    )
    edge = KnowledgeEdge(
        id=uuid4(),
        collection_id=uuid4(),
        source_node_id=symbol.id,
        target_node_id=target.id,
        kind=KnowledgeEdgeKind.SYMBOL_REFERENCES_SYMBOL,
    )

    facts = _extract_outbound_ports({symbol.id: symbol}, {target.id: target}, [edge])

    assert len(facts) == 1
    fact = facts[0]
    assert fact.direction == "outbound"
    assert fact.port_name == "orders"
    assert fact.adapter_name == "OrdersRepository"
    assert fact.protocol == "sql"
