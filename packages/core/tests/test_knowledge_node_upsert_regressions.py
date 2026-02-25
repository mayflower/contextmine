"""Regression tests for extractor graph builders upserting KnowledgeNode rows."""

from __future__ import annotations

import uuid

import pytest
from contextmine_core.analyzer.extractors.jobs import JobDef, JobsExtraction, build_jobs_graph
from contextmine_core.analyzer.extractors.schema import (
    AggregatedSchema,
    ColumnDef,
    TableDef,
    build_schema_graph,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeResult:
    def __init__(
        self, *, one: uuid.UUID | None = None, one_or_none: uuid.UUID | None = None
    ) -> None:
        self._one = one
        self._one_or_none = one_or_none

    def scalar_one(self) -> uuid.UUID:
        if self._one is None:
            raise AssertionError("scalar_one requested but no value set")
        return self._one

    def scalar_one_or_none(self) -> uuid.UUID | None:
        return self._one_or_none


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    async def execute(self, statement: object) -> _FakeResult:
        if getattr(statement, "is_select", False):
            return _FakeResult(one_or_none=None)
        return _FakeResult(one=uuid.uuid4())

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        for obj in self.added:
            if hasattr(obj, "id") and getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_build_schema_graph_uses_supported_knowledge_node_columns() -> None:
    session = _FakeSession()
    schema = AggregatedSchema(
        tables={
            "users": TableDef(
                name="users",
                columns=[ColumnDef(name="id", type_name="UUID", nullable=False, primary_key=True)],
                primary_keys=["id"],
                description="User accounts",
            )
        },
        foreign_keys=[],
        sources=["db/schema.sql"],
    )

    stats = await build_schema_graph(session=session, collection_id=uuid.uuid4(), schema=schema)

    assert stats["table_nodes_created"] == 1
    assert stats["column_nodes_created"] == 1
    assert stats["edges_created"] == 1
    assert stats["evidence_created"] == 1


@pytest.mark.anyio
async def test_build_jobs_graph_uses_supported_knowledge_node_columns() -> None:
    session = _FakeSession()
    extraction = JobsExtraction(
        file_path=".github/workflows/ci.yml",
        jobs=[
            JobDef(
                name="nightly",
                framework="github_actions",
                file_path=".github/workflows/ci.yml",
                description="Nightly build",
            )
        ],
    )

    stats = await build_jobs_graph(
        session=session,
        collection_id=uuid.uuid4(),
        extractions=[extraction],
    )

    assert stats["job_nodes_created"] == 1
    assert stats["edges_created"] == 0
    assert stats["evidence_created"] == 1
