"""Tests for knowledge graph builder.

Note: These tests require PostgreSQL and are skipped when using SQLite.
Run with a real test database for full coverage.
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
from contextmine_core.database import Base
from contextmine_core.knowledge.builder import (
    build_knowledge_graph_for_source,
    cleanup_orphan_nodes,
)
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    Document,
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
    Source,
    SourceType,
    Symbol,
    SymbolKind,
    User,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Skip all tests in this module - they require PostgreSQL
pytestmark = pytest.mark.skip(reason="Requires PostgreSQL (uses ON CONFLICT)")


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend."""
    return "asyncio"


@pytest.fixture
async def test_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session.

    Uses SQLite in-memory for unit tests.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def user_id() -> uuid.UUID:
    """Generate a test user ID."""
    return uuid.uuid4()


@pytest.fixture
def collection_id() -> uuid.UUID:
    """Generate a test collection ID."""
    return uuid.uuid4()


@pytest.fixture
def source_id() -> uuid.UUID:
    """Generate a test source ID."""
    return uuid.uuid4()


class TestKnowledgeGraphBuilder:
    """Tests for knowledge graph builder."""

    async def test_build_creates_file_nodes(
        self,
        test_session: AsyncSession,
        user_id: uuid.UUID,
        collection_id: uuid.UUID,
        source_id: uuid.UUID,
    ) -> None:
        """Test that building graph creates FILE nodes for documents."""
        # Setup: Create user, collection, source, and document
        user = User(
            id=user_id,
            github_user_id=12345,
            github_login="testuser",
        )
        test_session.add(user)

        collection = Collection(
            id=collection_id,
            slug="test-collection",
            name="Test Collection",
            visibility=CollectionVisibility.PRIVATE,
            owner_user_id=user_id,
        )
        test_session.add(collection)

        source = Source(
            id=source_id,
            collection_id=collection_id,
            type=SourceType.GITHUB,
            url="https://github.com/test/repo",
            config={"owner": "test", "repo": "repo"},
        )
        test_session.add(source)

        doc = Document(
            source_id=source_id,
            uri="git://github.com/test/repo/src/main.py?ref=main",
            title="main.py",
            content_markdown="# Main module",
            content_hash="abc123",
            meta={"file_path": "src/main.py"},
            last_seen_at=datetime.now(UTC),
        )
        test_session.add(doc)
        await test_session.commit()

        # Build knowledge graph
        stats = await build_knowledge_graph_for_source(test_session, source_id)
        await test_session.commit()

        # Verify FILE node was created
        assert stats["file_nodes_created"] >= 1

        # Query for the created node
        result = await test_session.execute(
            select(KnowledgeNode).where(
                KnowledgeNode.collection_id == collection_id,
                KnowledgeNode.kind == KnowledgeNodeKind.FILE,
            )
        )
        nodes = result.scalars().all()
        assert len(nodes) == 1
        assert nodes[0].name == "main.py"
        assert nodes[0].meta["file_path"] == "src/main.py"

    async def test_build_creates_symbol_nodes(
        self,
        test_session: AsyncSession,
        user_id: uuid.UUID,
        collection_id: uuid.UUID,
        source_id: uuid.UUID,
    ) -> None:
        """Test that building graph creates SYMBOL nodes."""
        # Setup
        user = User(id=user_id, github_user_id=12345, github_login="testuser")
        test_session.add(user)

        collection = Collection(
            id=collection_id,
            slug="test-collection",
            name="Test Collection",
            visibility=CollectionVisibility.PRIVATE,
            owner_user_id=user_id,
        )
        test_session.add(collection)

        source = Source(
            id=source_id,
            collection_id=collection_id,
            type=SourceType.GITHUB,
            url="https://github.com/test/repo",
        )
        test_session.add(source)

        doc = Document(
            source_id=source_id,
            uri="git://github.com/test/repo/src/main.py?ref=main",
            title="main.py",
            content_markdown="def hello(): pass",
            content_hash="abc123",
            meta={"file_path": "src/main.py"},
            last_seen_at=datetime.now(UTC),
        )
        test_session.add(doc)
        await test_session.flush()

        # Add a symbol
        symbol = Symbol(
            document_id=doc.id,
            qualified_name="hello",
            name="hello",
            kind=SymbolKind.FUNCTION,
            start_line=1,
            end_line=1,
            signature="def hello()",
        )
        test_session.add(symbol)
        await test_session.commit()

        # Build knowledge graph
        stats = await build_knowledge_graph_for_source(test_session, source_id)
        await test_session.commit()

        # Verify
        assert stats["symbol_nodes_created"] >= 1
        assert stats["edges_created"] >= 1  # FILE_DEFINES_SYMBOL edge

        result = await test_session.execute(
            select(KnowledgeNode).where(
                KnowledgeNode.collection_id == collection_id,
                KnowledgeNode.kind == KnowledgeNodeKind.SYMBOL,
            )
        )
        symbol_nodes = result.scalars().all()
        assert len(symbol_nodes) == 1
        assert symbol_nodes[0].name == "hello"

    async def test_build_creates_containment_edges(
        self,
        test_session: AsyncSession,
        user_id: uuid.UUID,
        collection_id: uuid.UUID,
        source_id: uuid.UUID,
    ) -> None:
        """Test that nested symbols create SYMBOL_CONTAINS_SYMBOL edges."""
        # Setup
        user = User(id=user_id, github_user_id=12345, github_login="testuser")
        test_session.add(user)

        collection = Collection(
            id=collection_id,
            slug="test-collection",
            name="Test Collection",
            visibility=CollectionVisibility.PRIVATE,
            owner_user_id=user_id,
        )
        test_session.add(collection)

        source = Source(
            id=source_id,
            collection_id=collection_id,
            type=SourceType.GITHUB,
            url="https://github.com/test/repo",
        )
        test_session.add(source)

        doc = Document(
            source_id=source_id,
            uri="git://github.com/test/repo/src/main.py?ref=main",
            title="main.py",
            content_markdown="class Foo:\n    def bar(self): pass",
            content_hash="abc123",
            meta={"file_path": "src/main.py"},
            last_seen_at=datetime.now(UTC),
        )
        test_session.add(doc)
        await test_session.flush()

        # Add class and method symbols
        class_symbol = Symbol(
            document_id=doc.id,
            qualified_name="Foo",
            name="Foo",
            kind=SymbolKind.CLASS,
            start_line=1,
            end_line=2,
        )
        test_session.add(class_symbol)

        method_symbol = Symbol(
            document_id=doc.id,
            qualified_name="Foo.bar",
            name="bar",
            kind=SymbolKind.METHOD,
            start_line=2,
            end_line=2,
            parent_name="Foo",
        )
        test_session.add(method_symbol)
        await test_session.commit()

        # Build knowledge graph
        await build_knowledge_graph_for_source(test_session, source_id)
        await test_session.commit()

        # Verify containment edge exists
        result = await test_session.execute(
            select(KnowledgeEdge).where(
                KnowledgeEdge.collection_id == collection_id,
                KnowledgeEdge.kind == KnowledgeEdgeKind.SYMBOL_CONTAINS_SYMBOL,
            )
        )
        edges = result.scalars().all()
        assert len(edges) == 1

    async def test_build_is_idempotent(
        self,
        test_session: AsyncSession,
        user_id: uuid.UUID,
        collection_id: uuid.UUID,
        source_id: uuid.UUID,
    ) -> None:
        """Test that running builder twice doesn't duplicate nodes."""
        # Setup
        user = User(id=user_id, github_user_id=12345, github_login="testuser")
        test_session.add(user)

        collection = Collection(
            id=collection_id,
            slug="test-collection",
            name="Test Collection",
            visibility=CollectionVisibility.PRIVATE,
            owner_user_id=user_id,
        )
        test_session.add(collection)

        source = Source(
            id=source_id,
            collection_id=collection_id,
            type=SourceType.GITHUB,
            url="https://github.com/test/repo",
        )
        test_session.add(source)

        doc = Document(
            source_id=source_id,
            uri="git://github.com/test/repo/src/main.py?ref=main",
            title="main.py",
            content_markdown="# Main",
            content_hash="abc123",
            meta={"file_path": "src/main.py"},
            last_seen_at=datetime.now(UTC),
        )
        test_session.add(doc)
        await test_session.commit()

        # Build twice
        await build_knowledge_graph_for_source(test_session, source_id)
        await test_session.commit()

        await build_knowledge_graph_for_source(test_session, source_id)
        await test_session.commit()

        # Verify only one FILE node exists
        result = await test_session.execute(
            select(KnowledgeNode).where(
                KnowledgeNode.collection_id == collection_id,
                KnowledgeNode.kind == KnowledgeNodeKind.FILE,
            )
        )
        nodes = result.scalars().all()
        assert len(nodes) == 1

    async def test_cleanup_removes_orphan_nodes(
        self,
        test_session: AsyncSession,
        user_id: uuid.UUID,
        collection_id: uuid.UUID,
        source_id: uuid.UUID,
    ) -> None:
        """Test that cleanup removes nodes for deleted documents."""
        from sqlalchemy import delete

        # Setup
        user = User(id=user_id, github_user_id=12345, github_login="testuser")
        test_session.add(user)

        collection = Collection(
            id=collection_id,
            slug="test-collection",
            name="Test Collection",
            visibility=CollectionVisibility.PRIVATE,
            owner_user_id=user_id,
        )
        test_session.add(collection)

        source = Source(
            id=source_id,
            collection_id=collection_id,
            type=SourceType.GITHUB,
            url="https://github.com/test/repo",
        )
        test_session.add(source)

        doc = Document(
            source_id=source_id,
            uri="git://github.com/test/repo/src/main.py?ref=main",
            title="main.py",
            content_markdown="# Main",
            content_hash="abc123",
            meta={"file_path": "src/main.py"},
            last_seen_at=datetime.now(UTC),
        )
        test_session.add(doc)
        await test_session.commit()

        doc_id = doc.id

        # Build graph
        await build_knowledge_graph_for_source(test_session, source_id)
        await test_session.commit()

        # Delete the document
        await test_session.execute(delete(Document).where(Document.id == doc_id))
        await test_session.commit()

        # Run cleanup
        stats = await cleanup_orphan_nodes(test_session, collection_id, source_id)
        await test_session.commit()

        # Verify node was removed
        assert stats["nodes_deleted"] == 1

        result = await test_session.execute(
            select(KnowledgeNode).where(
                KnowledgeNode.collection_id == collection_id,
            )
        )
        nodes = result.scalars().all()
        assert len(nodes) == 0
