"""persist_semantic_entities against a real PostgreSQL schema."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import pytest
from contextmine_core.knowledge.extraction import (
    ExtractionBatch,
    SemanticEntity,
    SemanticRelationship,
    persist_semantic_entities,
)
from contextmine_core.models import (
    Base,
    Chunk,
    Collection,
    Document,
    EmbeddingModel,
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeEvidence,
    KnowledgeNodeKind,
    Source,
    SourceType,
    User,
)
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def _entity(name: str, *, files: list[str], description: str = "d") -> SemanticEntity:
    return SemanticEntity(
        id=name, name=name, entity_type="CONCEPT", description=description, source_files=files
    )


@asynccontextmanager
async def _schema():
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    schema = f"semantic_entities_test_{uuid.uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    enums = {
        "collection_visibility": ["global", "private"],
        "source_type": [kind.value for kind in SourceType],
        "knowledge_node_kind": [kind.value for kind in KnowledgeNodeKind],
        "knowledge_edge_kind": [kind.value for kind in KnowledgeEdgeKind],
    }
    async with admin_engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        for name, values in enums.items():
            labels = ", ".join(f"'{value}'" for value in values)
            await connection.execute(text(f'CREATE TYPE "{schema}".{name} AS ENUM ({labels})'))
    engine = create_async_engine(
        database_url, connect_args={"server_settings": {"search_path": f"{schema},public"}}
    )
    # The model declares a GIN index on the JSON meta column that only works on
    # the migrated (JSONB) schema; leave it out of this ad-hoc schema.
    node_indexes = KnowledgeNode.__table__.indexes
    meta_index = next(index for index in node_indexes if index.name == "ix_knowledge_node_meta")
    node_indexes.discard(meta_index)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection,
                    tables=[
                        User.__table__,
                        Collection.__table__,
                        Source.__table__,
                        Document.__table__,
                        EmbeddingModel.__table__,
                        Chunk.__table__,
                        KnowledgeNode.__table__,
                        KnowledgeEdge.__table__,
                        KnowledgeEvidence.__table__,
                        KnowledgeNodeEvidence.__table__,
                    ],
                )
            )
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        node_indexes.add(meta_index)
        await engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()


async def _collection(session) -> uuid.UUID:  # noqa: ANN001
    user = User(github_user_id=303, github_login="semantic-test")
    session.add(user)
    await session.flush()
    collection = Collection(
        slug=f"c-{uuid.uuid4().hex[:8]}", name="c", visibility="private", owner_user_id=user.id
    )
    session.add(collection)
    await session.flush()
    for path in ("ISSUES.md", "docs/a.md"):
        session.add(
            KnowledgeNode(
                collection_id=collection.id,
                kind=KnowledgeNodeKind.FILE,
                natural_key=f"file:{path}",
                name=path,
                meta={},
            )
        )
    await session.commit()
    return collection.id


async def _count(session, collection_id: uuid.UUID, model) -> int:  # noqa: ANN001
    return (
        await session.execute(
            select(func.count()).select_from(model).where(model.collection_id == collection_id)
        )
    ).scalar_one()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL semantic entity test",
)
async def test_existing_and_duplicate_entities_are_reused() -> None:
    async with _schema() as factory:
        async with factory() as session:
            collection_id = await _collection(session)

        # First run creates the entity from an unchanged document.
        async with factory() as session:
            await persist_semantic_entities(
                session,
                collection_id,
                ExtractionBatch(entities=[_entity("Architecture recovery", files=["docs/a.md"])]),
                changed_doc_ids=None,
            )
            await session.commit()

        # A later incremental run finds the same entity again (also twice in one
        # batch, in different capitalization) and relates it to another entity.
        async with factory() as session:
            stats = await persist_semantic_entities(
                session,
                collection_id,
                ExtractionBatch(
                    entities=[
                        _entity("Architecture Recovery", files=["ISSUES.md", "docs/a.md"]),
                        _entity("architecture recovery", files=["ISSUES.md"]),
                        _entity("Evidence", files=["ISSUES.md"]),
                    ],
                    relationships=[
                        SemanticRelationship(
                            source_entity="Architecture Recovery",
                            target_entity="Evidence",
                            relationship_type="USES",
                            description="cites",
                            strength=0.9,
                        ),
                        SemanticRelationship(
                            source_entity="architecture recovery",
                            target_entity="Evidence",
                            relationship_type="USES",
                            description="cites",
                            strength=0.9,
                        ),
                    ],
                ),
                changed_doc_ids=[str(uuid.uuid4())],
            )
            await session.commit()

            entities = (
                (
                    await session.execute(
                        select(KnowledgeNode).where(
                            KnowledgeNode.collection_id == collection_id,
                            KnowledgeNode.kind == KnowledgeNodeKind.SEMANTIC_ENTITY,
                        )
                    )
                )
                .scalars()
                .all()
            )

        by_key = {node.natural_key: node for node in entities}
        assert sorted(by_key) == ["entity:architecture_recovery", "entity:evidence"]
        recovery = by_key["entity:architecture_recovery"]
        assert sorted(recovery.meta["source_files"]) == ["ISSUES.md", "docs/a.md"]
        assert stats["entities_created"] == 1
        assert stats["entities_merged"] == 2
        assert stats["relationships_created"] == 1
        async with factory() as session:
            assert await _count(session, collection_id, KnowledgeEdge) == 4  # 1 rel + 3 mentions
