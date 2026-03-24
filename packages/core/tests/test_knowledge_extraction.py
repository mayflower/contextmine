"""Unit tests for knowledge/extraction.py.

Tests LLM-based entity and relationship extraction with mocked providers:
- Pydantic model validation (ExtractedEntity, ExtractedRelationship, ExtractionResult)
- extract_entities_from_chunk() with mocked LLM provider
- extract_relationships_from_chunk() with mocked LLM provider
- SemanticEntity / SemanticRelationship / ExtractionBatch dataclasses
- _merge_entity_groups() pure logic
- _extract_cross_document_relationships() pure logic
- _resolve_entities_with_embeddings() with mocked embedder
- persist_semantic_entities() with mocked session
- extract_from_documents() with mocked session/LLM/embedder
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from contextmine_core.knowledge.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionBatch,
    ExtractionResult,
    SemanticEntity,
    SemanticRelationship,
    _extract_cross_document_relationships,
    _merge_entity_groups,
    _resolve_entities_with_embeddings,
    extract_entities_from_chunk,
    extract_from_documents,
    extract_relationships_from_chunk,
    persist_semantic_entities,
)
from pydantic import ValidationError

# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestExtractedEntity:
    """Tests for ExtractedEntity Pydantic model."""

    def test_valid_creation(self) -> None:
        entity = ExtractedEntity(
            name="User Authentication",
            type="CONCEPT",
            description="Handles login and session management",
        )
        assert entity.name == "User Authentication"
        assert entity.type == "CONCEPT"
        assert entity.aliases == []
        assert entity.source_symbols == []

    def test_with_aliases_and_symbols(self) -> None:
        entity = ExtractedEntity(
            name="Auth",
            type="COMPONENT",
            description="Auth component",
            aliases=["authentication", "login"],
            source_symbols=["AuthService", "LoginController"],
        )
        assert len(entity.aliases) == 2
        assert len(entity.source_symbols) == 2


class TestExtractedRelationship:
    """Tests for ExtractedRelationship Pydantic model."""

    def test_valid_creation(self) -> None:
        rel = ExtractedRelationship(
            source="Auth",
            target="User",
            relationship="USES",
            description="Auth uses User model",
            strength=0.8,
        )
        assert rel.source == "Auth"
        assert rel.target == "User"
        assert rel.strength == 0.8

    def test_strength_bounds(self) -> None:
        ExtractedRelationship(
            source="A", target="B", relationship="R", description="D", strength=0.0
        )
        ExtractedRelationship(
            source="A", target="B", relationship="R", description="D", strength=1.0
        )
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                source="A", target="B", relationship="R", description="D", strength=1.5
            )
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                source="A", target="B", relationship="R", description="D", strength=-0.1
            )


class TestExtractionResult:
    """Tests for ExtractionResult Pydantic model."""

    def test_empty_defaults(self) -> None:
        result = ExtractionResult()
        assert result.entities == []
        assert result.relationships == []

    def test_with_data(self) -> None:
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="A", type="CONCEPT", description="desc"),
            ],
            relationships=[
                ExtractedRelationship(
                    source="A", target="B", relationship="USES", description="d", strength=0.5
                ),
            ],
        )
        assert len(result.entities) == 1
        assert len(result.relationships) == 1


# ===========================================================================
# Dataclass tests
# ===========================================================================


class TestSemanticEntity:
    """Tests for SemanticEntity dataclass."""

    def test_natural_key(self) -> None:
        entity = SemanticEntity(
            id="abc123",
            name="User Authentication",
            entity_type="CONCEPT",
            description="Handles login",
        )
        assert entity.natural_key == "entity:user_authentication"

    def test_natural_key_strips_spaces(self) -> None:
        entity = SemanticEntity(
            id="abc",
            name="Data Access Layer",
            entity_type="COMPONENT",
            description="DAL",
        )
        assert entity.natural_key == "entity:data_access_layer"

    def test_defaults(self) -> None:
        entity = SemanticEntity(id="x", name="N", entity_type="T", description="D")
        assert entity.aliases == []
        assert entity.source_symbols == []
        assert entity.source_files == []


class TestSemanticRelationship:
    """Tests for SemanticRelationship dataclass."""

    def test_creation(self) -> None:
        rel = SemanticRelationship(
            source_entity="Auth",
            target_entity="User",
            relationship_type="USES",
            description="Auth uses User",
        )
        assert rel.strength == 1.0  # Default

    def test_custom_strength(self) -> None:
        rel = SemanticRelationship(
            source_entity="A",
            target_entity="B",
            relationship_type="R",
            description="D",
            strength=0.5,
        )
        assert rel.strength == 0.5


class TestExtractionBatch:
    """Tests for ExtractionBatch dataclass."""

    def test_defaults(self) -> None:
        batch = ExtractionBatch()
        assert batch.entities == []
        assert batch.relationships == []
        assert batch.chunks_processed == 0
        assert batch.errors == []
        assert batch.entity_occurrences == {}

    def test_mutable_defaults_not_shared(self) -> None:
        b1 = ExtractionBatch()
        b2 = ExtractionBatch()
        b1.entities.append(SemanticEntity(id="x", name="N", entity_type="T", description="D"))
        assert b2.entities == []


# ===========================================================================
# extract_entities_from_chunk tests
# ===========================================================================


class TestExtractEntitiesFromChunk:
    """Tests for extract_entities_from_chunk with mocked LLM."""

    @pytest.mark.anyio
    async def test_empty_code_returns_empty(self) -> None:
        mock_llm = AsyncMock()
        result = await extract_entities_from_chunk(mock_llm, "", "test.py")
        assert result == []
        mock_llm.generate_structured.assert_not_called()

    @pytest.mark.anyio
    async def test_whitespace_only_returns_empty(self) -> None:
        mock_llm = AsyncMock()
        result = await extract_entities_from_chunk(mock_llm, "   \n  ", "test.py")
        assert result == []

    @pytest.mark.anyio
    async def test_returns_entities_from_llm(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = ExtractionResult(
            entities=[
                ExtractedEntity(name="Auth", type="CONCEPT", description="Handles auth"),
                ExtractedEntity(name="DB", type="COMPONENT", description="Database layer"),
            ]
        )

        result = await extract_entities_from_chunk(mock_llm, "def login(): pass", "auth.py")
        assert len(result) == 2
        assert result[0].name == "Auth"

    @pytest.mark.anyio
    async def test_truncates_long_code(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = ExtractionResult(entities=[])

        long_code = "x" * 10000
        await extract_entities_from_chunk(mock_llm, long_code, "big.py")

        call_args = mock_llm.generate_structured.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "truncated" in prompt

    @pytest.mark.anyio
    async def test_llm_exception_returns_empty(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.side_effect = RuntimeError("LLM error")

        result = await extract_entities_from_chunk(mock_llm, "code here", "test.py")
        assert result == []


# ===========================================================================
# extract_relationships_from_chunk tests
# ===========================================================================


class TestExtractRelationshipsFromChunk:
    """Tests for extract_relationships_from_chunk with mocked LLM."""

    @pytest.mark.anyio
    async def test_empty_entities_returns_empty(self) -> None:
        mock_llm = AsyncMock()
        result = await extract_relationships_from_chunk(mock_llm, [], "code")
        assert result == []

    @pytest.mark.anyio
    async def test_single_entity_returns_empty(self) -> None:
        mock_llm = AsyncMock()
        entities = [ExtractedEntity(name="A", type="CONCEPT", description="D")]
        result = await extract_relationships_from_chunk(mock_llm, entities, "code")
        assert result == []

    @pytest.mark.anyio
    async def test_returns_relationships_from_llm(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = ExtractionResult(
            relationships=[
                ExtractedRelationship(
                    source="Auth",
                    target="User",
                    relationship="USES",
                    description="Auth uses User",
                    strength=0.9,
                ),
            ]
        )

        entities = [
            ExtractedEntity(name="Auth", type="CONCEPT", description="Auth module"),
            ExtractedEntity(name="User", type="DATA_MODEL", description="User model"),
        ]

        result = await extract_relationships_from_chunk(
            mock_llm, entities, "class Auth: def get_user(): ..."
        )
        assert len(result) == 1
        assert result[0].source == "Auth"
        assert result[0].target == "User"

    @pytest.mark.anyio
    async def test_truncates_long_code(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = ExtractionResult(relationships=[])

        entities = [
            ExtractedEntity(name="A", type="CONCEPT", description="D1"),
            ExtractedEntity(name="B", type="CONCEPT", description="D2"),
        ]
        long_code = "x" * 5000
        await extract_relationships_from_chunk(mock_llm, entities, long_code)

        prompt = mock_llm.generate_structured.call_args.kwargs["messages"][0]["content"]
        assert "truncated" in prompt

    @pytest.mark.anyio
    async def test_llm_exception_returns_empty(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.side_effect = RuntimeError("LLM error")

        entities = [
            ExtractedEntity(name="A", type="C", description="D1"),
            ExtractedEntity(name="B", type="C", description="D2"),
        ]
        result = await extract_relationships_from_chunk(mock_llm, entities, "code")
        assert result == []


# ===========================================================================
# _merge_entity_groups tests
# ===========================================================================


class TestMergeEntityGroups:
    """Tests for _merge_entity_groups pure logic."""

    def test_single_entity_group(self) -> None:
        entity = ExtractedEntity(
            name="Auth",
            type="CONCEPT",
            description="Handles authentication",
            aliases=["login"],
            source_symbols=["AuthService"],
        )
        groups = {0: [entity]}
        merged = _merge_entity_groups(groups)

        assert len(merged) == 1
        assert merged[0].name == "Auth"
        assert merged[0].entity_type == "CONCEPT"
        assert "login" in merged[0].aliases
        # "Auth" is the best_name so it's NOT in aliases (aliases = all - {best_name})
        assert "Auth" not in merged[0].aliases

    def test_multiple_entities_merged(self) -> None:
        e1 = ExtractedEntity(
            name="Auth",
            type="CONCEPT",
            description="Short",
            aliases=["login"],
            source_symbols=["AuthService"],
        )
        e2 = ExtractedEntity(
            name="Authentication",
            type="CONCEPT",
            description="Much longer description of auth",
            aliases=["auth"],
            source_symbols=["LoginController"],
        )
        groups = {0: [e1, e2]}
        merged = _merge_entity_groups(groups)

        assert len(merged) == 1
        # Best name is from entity with longest description
        assert merged[0].name == "Authentication"
        assert "Auth" in merged[0].aliases
        assert "login" in merged[0].aliases
        assert "auth" in merged[0].aliases

    def test_type_picked_by_majority(self) -> None:
        e1 = ExtractedEntity(name="A", type="CONCEPT", description="D1")
        e2 = ExtractedEntity(name="B", type="COMPONENT", description="D2")
        e3 = ExtractedEntity(name="C", type="CONCEPT", description="D3")
        groups = {0: [e1, e2, e3]}
        merged = _merge_entity_groups(groups)

        assert merged[0].entity_type == "CONCEPT"

    def test_empty_group_skipped(self) -> None:
        groups = {0: [], 1: [ExtractedEntity(name="A", type="C", description="D")]}
        merged = _merge_entity_groups(groups)
        assert len(merged) == 1

    def test_multiple_groups(self) -> None:
        e1 = ExtractedEntity(name="Auth", type="CONCEPT", description="Auth")
        e2 = ExtractedEntity(name="DB", type="COMPONENT", description="Database")
        groups = {0: [e1], 1: [e2]}
        merged = _merge_entity_groups(groups)
        assert len(merged) == 2
        names = {m.name for m in merged}
        assert "Auth" in names
        assert "DB" in names


# ===========================================================================
# _extract_cross_document_relationships tests
# ===========================================================================


class TestExtractCrossDocumentRelationships:
    """Tests for _extract_cross_document_relationships pure logic."""

    def test_no_co_occurrences_returns_empty(self) -> None:
        entities = [
            SemanticEntity(id="a", name="Auth", entity_type="C", description="D"),
        ]
        occurrences = {"doc1.py": ["Auth"]}
        result = _extract_cross_document_relationships(entities, occurrences, [])
        assert result == []

    def test_co_occurring_entities_create_relationship(self) -> None:
        entities = [
            SemanticEntity(
                id="a",
                name="Auth",
                entity_type="C",
                description="D",
                aliases=["login"],
            ),
            SemanticEntity(
                id="b",
                name="User",
                entity_type="D",
                description="U",
                aliases=["account"],
            ),
        ]
        occurrences = {"doc1.py": ["Auth", "User"]}
        result = _extract_cross_document_relationships(entities, occurrences, [])

        assert len(result) == 1
        assert result[0].relationship_type == "CO_OCCURS"
        assert result[0].strength == 0.3  # 1 doc -> 0.3

    def test_strength_scales_with_count(self) -> None:
        entities = [
            SemanticEntity(id="a", name="Alpha", entity_type="C", description="D"),
            SemanticEntity(id="b", name="Beta", entity_type="C", description="D"),
        ]
        # 5+ co-occurrences -> 0.8
        occurrences = {f"doc{i}.py": ["Alpha", "Beta"] for i in range(6)}
        result = _extract_cross_document_relationships(entities, occurrences, [])

        assert len(result) == 1
        assert result[0].strength == 0.8

    def test_skips_existing_relationships(self) -> None:
        entities = [
            SemanticEntity(id="a", name="Auth", entity_type="C", description="D"),
            SemanticEntity(id="b", name="User", entity_type="C", description="D"),
        ]
        occurrences = {"doc1.py": ["Auth", "User"]}
        existing = [
            SemanticRelationship(
                source_entity="Auth",
                target_entity="User",
                relationship_type="USES",
                description="Existing",
            )
        ]
        result = _extract_cross_document_relationships(entities, occurrences, existing)
        assert result == []

    def test_resolves_aliases_to_canonical_names(self) -> None:
        entities = [
            SemanticEntity(
                id="a",
                name="Auth",
                entity_type="C",
                description="D",
                aliases=["login", "authentication"],
            ),
            SemanticEntity(
                id="b",
                name="User",
                entity_type="D",
                description="U",
                aliases=["account"],
            ),
        ]
        # Uses alias names in occurrences
        occurrences = {"doc1.py": ["login", "account"]}
        result = _extract_cross_document_relationships(entities, occurrences, [])

        assert len(result) == 1
        # Should use canonical names
        entity_names = {result[0].source_entity, result[0].target_entity}
        assert "Auth" in entity_names
        assert "User" in entity_names


# ===========================================================================
# _resolve_entities_with_embeddings tests
# ===========================================================================


class TestResolveEntitiesWithEmbeddings:
    """Tests for _resolve_entities_with_embeddings with mocked embedder."""

    @pytest.mark.anyio
    async def test_empty_entities_returns_empty(self) -> None:
        mock_embedder = AsyncMock()
        result = await _resolve_entities_with_embeddings([], mock_embedder)
        assert result == []

    @pytest.mark.anyio
    async def test_single_entity(self) -> None:
        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = MagicMock(embeddings=[[1.0, 0.0, 0.0]])

        entities = [
            ExtractedEntity(name="Auth", type="CONCEPT", description="Handles auth"),
        ]
        result = await _resolve_entities_with_embeddings(entities, mock_embedder)
        assert len(result) == 1
        assert result[0].name == "Auth"

    @pytest.mark.anyio
    async def test_similar_entities_merged(self) -> None:
        mock_embedder = AsyncMock()
        # Nearly identical embeddings -> should merge
        mock_embedder.embed_batch.return_value = MagicMock(
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
            ]
        )

        entities = [
            ExtractedEntity(name="Auth", type="CONCEPT", description="Short"),
            ExtractedEntity(name="Authentication", type="CONCEPT", description="Longer auth desc"),
        ]
        result = await _resolve_entities_with_embeddings(
            entities, mock_embedder, similarity_threshold=0.85
        )
        assert len(result) == 1

    @pytest.mark.anyio
    async def test_dissimilar_entities_kept_separate(self) -> None:
        mock_embedder = AsyncMock()
        # Orthogonal embeddings -> should not merge
        mock_embedder.embed_batch.return_value = MagicMock(
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        entities = [
            ExtractedEntity(name="Auth", type="CONCEPT", description="Auth module"),
            ExtractedEntity(name="Database", type="COMPONENT", description="DB layer"),
        ]
        result = await _resolve_entities_with_embeddings(entities, mock_embedder)
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_legacy_embed_texts_interface(self) -> None:
        """Test fallback to embed_texts when embed_batch is not available."""
        mock_embedder = MagicMock(spec=[])
        # Remove embed_batch, add embed_texts
        mock_embedder.embed_texts = AsyncMock(return_value=[[1.0, 0.0, 0.0]])

        entities = [
            ExtractedEntity(name="Auth", type="CONCEPT", description="Auth"),
        ]
        result = await _resolve_entities_with_embeddings(entities, mock_embedder)
        assert len(result) == 1

    @pytest.mark.anyio
    async def test_no_embed_interface_raises(self) -> None:
        mock_embedder = MagicMock(spec=[])
        entities = [
            ExtractedEntity(name="A", type="C", description="D"),
        ]
        with pytest.raises(ValueError, match="embed_batch"):
            await _resolve_entities_with_embeddings(entities, mock_embedder)

    @pytest.mark.anyio
    async def test_empty_embeddings_raises(self) -> None:
        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = MagicMock(embeddings=[])

        entities = [
            ExtractedEntity(name="A", type="C", description="D"),
        ]
        with pytest.raises(ValueError, match="no embeddings"):
            await _resolve_entities_with_embeddings(entities, mock_embedder)


# ===========================================================================
# persist_semantic_entities tests
# ===========================================================================


class TestPersistSemanticEntities:
    """Tests for persist_semantic_entities with mocked session."""

    @pytest.mark.anyio
    async def test_empty_batch_returns_zero_stats(self) -> None:
        session = AsyncMock()
        coll_id = uuid4()

        # Mock: no existing entities
        mock_existing = MagicMock()
        mock_existing.fetchall.return_value = []

        # Mock: no file nodes
        mock_files = MagicMock()
        mock_files.all.return_value = []

        session.execute.side_effect = [mock_existing, mock_files]

        batch = ExtractionBatch()
        stats = await persist_semantic_entities(session, coll_id, batch)

        assert stats["entities_created"] == 0
        assert stats["relationships_created"] == 0
        assert stats["file_edges_created"] == 0

    @pytest.mark.anyio
    async def test_creates_entities_and_relationships(self) -> None:
        session = AsyncMock()
        coll_id = uuid4()

        # Mock: no existing entities
        mock_existing = MagicMock()
        mock_existing.fetchall.return_value = []

        # Mock: no file nodes
        mock_files = MagicMock()
        mock_files.all.return_value = []

        session.execute.side_effect = [mock_existing, mock_files]

        # Make session.add set a mock .id on the node so flush/entity creation works.
        # persist_semantic_entities reads node.id after flush.
        node_ids = iter([uuid4(), uuid4()])

        original_add = session.add

        def mock_add(obj):
            # Set id on KnowledgeNode-like objects that will be flushed
            if hasattr(obj, "kind") and not hasattr(obj, "source_node_id"):
                obj.id = next(node_ids)
            return original_add(obj)

        session.add = mock_add

        batch = ExtractionBatch(
            entities=[
                SemanticEntity(
                    id="a1",
                    name="Auth",
                    entity_type="CONCEPT",
                    description="Auth module",
                ),
                SemanticEntity(
                    id="b1",
                    name="User",
                    entity_type="DATA_MODEL",
                    description="User model",
                ),
            ],
            relationships=[
                SemanticRelationship(
                    source_entity="Auth",
                    target_entity="User",
                    relationship_type="USES",
                    description="Auth uses User",
                    strength=0.9,
                ),
            ],
        )
        stats = await persist_semantic_entities(session, coll_id, batch)

        assert stats["entities_created"] == 2
        assert stats["relationships_created"] == 1

    @pytest.mark.anyio
    async def test_deletes_existing_entities_first(self) -> None:
        session = AsyncMock()
        coll_id = uuid4()
        old_id = uuid4()

        mock_existing = MagicMock()
        mock_existing.fetchall.return_value = [(old_id,)]

        mock_files = MagicMock()
        mock_files.all.return_value = []

        session.execute.side_effect = [mock_existing, MagicMock(), MagicMock(), mock_files]

        batch = ExtractionBatch()
        await persist_semantic_entities(session, coll_id, batch)

        # Should have called execute for the delete statements
        assert (
            session.execute.call_count >= 3
        )  # select existing + delete edges + delete nodes + select files

    @pytest.mark.anyio
    async def test_creates_file_mention_edges(self) -> None:
        session = AsyncMock()
        coll_id = uuid4()
        file_node_id = uuid4()

        mock_existing = MagicMock()
        mock_existing.fetchall.return_value = []

        mock_files = MagicMock()
        mock_files.all.return_value = [(file_node_id, "file:src/auth.py")]

        session.execute.side_effect = [mock_existing, mock_files]

        batch = ExtractionBatch(
            entities=[
                SemanticEntity(
                    id="a1",
                    name="Auth",
                    entity_type="CONCEPT",
                    description="Auth",
                    source_symbols=["src/auth.py"],
                ),
            ],
        )
        stats = await persist_semantic_entities(session, coll_id, batch)
        assert stats["file_edges_created"] == 1


# ===========================================================================
# extract_from_documents tests
# ===========================================================================


class TestExtractFromDocuments:
    """Tests for extract_from_documents with mocked session/LLM/embedder."""

    @pytest.mark.anyio
    async def test_raises_without_embedder(self) -> None:
        with pytest.raises(ValueError, match="embedder is required"):
            await extract_from_documents(
                collection_id=uuid4(),
                llm_provider=AsyncMock(),
                embedder=None,
            )

    @pytest.mark.anyio
    async def test_empty_documents(self) -> None:
        mock_llm = AsyncMock()
        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = MagicMock(embeddings=[])

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session.execute.return_value = mock_result

        # get_session is imported locally inside extract_from_documents
        with patch(
            "contextmine_core.get_session",
        ) as mock_get_session:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_session
            ctx.__aexit__.return_value = False
            mock_get_session.return_value = ctx

            batch = await extract_from_documents(
                collection_id=uuid4(),
                llm_provider=mock_llm,
                embedder=mock_embedder,
            )
            assert batch.chunks_processed == 0
            assert batch.entities == []

    @pytest.mark.anyio
    async def test_processes_documents_and_extracts(self) -> None:
        mock_llm = AsyncMock()

        entity = ExtractedEntity(name="Auth", type="CONCEPT", description="Auth module")
        mock_llm.generate_structured.return_value = ExtractionResult(
            entities=[entity],
            relationships=[],
        )

        mock_embedder = AsyncMock()
        mock_embedder.embed_batch.return_value = MagicMock(embeddings=[[1.0, 0.0, 0.0]])

        doc_id = uuid4()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = [
            (doc_id, "src/auth.py", "def login(): pass"),
        ]
        mock_session.execute.return_value = mock_result

        # get_session is imported locally inside extract_from_documents
        with patch(
            "contextmine_core.get_session",
        ) as mock_get_session:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = mock_session
            ctx.__aexit__.return_value = False
            mock_get_session.return_value = ctx

            batch = await extract_from_documents(
                collection_id=uuid4(),
                llm_provider=mock_llm,
                embedder=mock_embedder,
            )
            assert batch.chunks_processed == 1
            assert len(batch.entities) >= 1
