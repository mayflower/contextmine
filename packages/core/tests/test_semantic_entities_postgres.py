"""persist_semantic_entities against a real PostgreSQL schema."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import contextmine_core
import pytest
from contextmine_core.knowledge.extraction import (
    ENTITY_EXTRACTION_SYSTEM,
    ExtractedEntity,
    ExtractionBatch,
    ExtractionResult,
    SemanticEntity,
    SemanticRelationship,
    extract_and_persist_semantic_entities,
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
    SemanticExtractionChunk,
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
                        SemanticExtractionChunk.__table__,
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


class _FakeLlm:
    """Names one entity after the chunk's file; fails for files in ``broken``."""

    provider_name = "fake"
    model_name = "m"

    def __init__(self, broken: set[str] | None = None) -> None:
        self.broken = broken or set()
        self.entity_calls: list[str] = []

    async def generate_structured(self, *, system, messages, output_schema, **_kwargs):  # noqa: ANN001, ANN003
        if system != ENTITY_EXTRACTION_SYSTEM:
            return output_schema()
        path = messages[0]["content"].split("FILE PATH: ", 1)[1].split("\n", 1)[0]
        self.entity_calls.append(path)
        if path in self.broken:
            raise ValueError("unusable answer")
        name = path.rsplit("/", 1)[-1].split(".")[0].title() + " Module"
        return ExtractionResult(
            entities=[ExtractedEntity(name=name, type="COMPONENT", description=f"Code in {path}")]
        )


class _FakeEmbedder:
    """Distinct vectors per text, so entities never merge; can fail after N calls."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    async def embed_batch(self, texts):  # noqa: ANN001
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("embedding service down")
        import random

        vectors = []
        for text_ in texts:
            rnd = random.Random(text_)
            vectors.append([rnd.uniform(-1, 1) for _ in range(64)])
        return type("Batch", (), {"embeddings": vectors})()


async def _documents(session, collection_id: uuid.UUID, paths: list[str]) -> dict[str, uuid.UUID]:  # noqa: ANN001
    from datetime import UTC, datetime

    source = Source(collection_id=collection_id, type=SourceType.GITHUB, url="https://x/y")
    session.add(source)
    await session.flush()
    doc_ids = {}
    for path in paths:
        document = Document(
            source_id=source.id,
            uri=path,
            title=path,
            content_markdown="x",
            content_hash=uuid.uuid4().hex,
            last_seen_at=datetime.now(UTC),
        )
        session.add(document)
        await session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                chunk_hash=uuid.uuid4().hex,
                content=f'class Thing:\n    """Lives in {path}."""\n',
                meta={"start_line": 1, "end_line": 2},
            )
        )
        doc_ids[path] = document.id
    await session.commit()
    return doc_ids


async def _recorded_paths(session, collection_id: uuid.UUID) -> list[str]:  # noqa: ANN001
    rows = await session.execute(
        select(Document.uri)
        .join(Chunk, Chunk.document_id == Document.id)
        .join(SemanticExtractionChunk, SemanticExtractionChunk.chunk_id == Chunk.id)
        .where(SemanticExtractionChunk.collection_id == collection_id)
    )
    return sorted(rows.scalars().all())


async def _entity_names(session, collection_id: uuid.UUID) -> list[str]:  # noqa: ANN001
    rows = await session.execute(
        select(KnowledgeNode.name).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SEMANTIC_ENTITY,
        )
    )
    return sorted(rows.scalars().all())


PATHS = ["src/alpha.py", "src/beta.py", "src/gamma.py"]
_needs_db = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL semantic extraction test",
)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@_needs_db
async def test_batches_commit_and_a_rerun_skips_finished_chunks(monkeypatch) -> None:  # noqa: ANN001
    async with _schema() as factory:
        monkeypatch.setattr(contextmine_core, "get_session", factory)
        async with factory() as session:
            collection_id = await _collection(session)
            await _documents(session, collection_id, PATHS)

        llm = _FakeLlm()
        stats = await extract_and_persist_semantic_entities(
            collection_id, llm, _FakeEmbedder(), changed_doc_ids=None, batch_size=1
        )
        assert stats["batches"] == 3
        assert stats["chunks_processed"] == 3
        assert stats["chunks_remaining"] == 0

        rerun = await extract_and_persist_semantic_entities(
            collection_id, llm, _FakeEmbedder(), changed_doc_ids=[], batch_size=1
        )
        assert rerun["chunks_processed"] == 0
        assert len(llm.entity_calls) == 3

        async with factory() as session:
            assert await _recorded_paths(session, collection_id) == PATHS
            assert await _entity_names(session, collection_id) == [
                "Alpha Module",
                "Beta Module",
                "Gamma Module",
            ]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@_needs_db
async def test_failed_chunk_is_not_recorded_and_retried(monkeypatch) -> None:  # noqa: ANN001
    async with _schema() as factory:
        monkeypatch.setattr(contextmine_core, "get_session", factory)
        async with factory() as session:
            collection_id = await _collection(session)
            await _documents(session, collection_id, PATHS)

        stats = await extract_and_persist_semantic_entities(
            collection_id, _FakeLlm(broken={"src/beta.py"}), _FakeEmbedder(), changed_doc_ids=None
        )
        assert stats["chunks_failed"] == 1
        assert stats["chunks_remaining"] == 1
        async with factory() as session:
            assert await _recorded_paths(session, collection_id) == ["src/alpha.py", "src/gamma.py"]

        retry = _FakeLlm()
        await extract_and_persist_semantic_entities(
            collection_id, retry, _FakeEmbedder(), changed_doc_ids=[]
        )
        assert retry.entity_calls == ["src/beta.py"]
        async with factory() as session:
            assert await _recorded_paths(session, collection_id) == PATHS


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@_needs_db
async def test_aborted_run_keeps_committed_batches(monkeypatch) -> None:  # noqa: ANN001
    async with _schema() as factory:
        monkeypatch.setattr(contextmine_core, "get_session", factory)
        async with factory() as session:
            collection_id = await _collection(session)
            await _documents(session, collection_id, PATHS)

        with pytest.raises(RuntimeError, match="embedding service down"):
            await extract_and_persist_semantic_entities(
                collection_id,
                _FakeLlm(),
                _FakeEmbedder(fail_after=1),
                changed_doc_ids=None,
                batch_size=1,
            )

        async with factory() as session:
            assert await _recorded_paths(session, collection_id) == ["src/alpha.py"]
            assert await _entity_names(session, collection_id) == ["Alpha Module"]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@_needs_db
async def test_changed_document_is_reset_and_extracted_again(monkeypatch) -> None:  # noqa: ANN001
    async with _schema() as factory:
        monkeypatch.setattr(contextmine_core, "get_session", factory)
        async with factory() as session:
            collection_id = await _collection(session)
            doc_ids = await _documents(session, collection_id, PATHS)
        await extract_and_persist_semantic_entities(
            collection_id, _FakeLlm(), _FakeEmbedder(), changed_doc_ids=None
        )

        llm = _FakeLlm()
        await extract_and_persist_semantic_entities(
            collection_id,
            llm,
            _FakeEmbedder(),
            changed_doc_ids=[str(doc_ids["src/beta.py"])],
        )

        assert llm.entity_calls == ["src/beta.py"]
        async with factory() as session:
            assert await _recorded_paths(session, collection_id) == PATHS
            assert await _entity_names(session, collection_id) == [
                "Alpha Module",
                "Beta Module",
                "Gamma Module",
            ]
