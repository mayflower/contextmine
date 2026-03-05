"""Compatibility tests for embedding interfaces in GraphRAG extraction."""

from __future__ import annotations

import pytest
from contextmine_core.embeddings import FakeEmbedder
from contextmine_core.knowledge.extraction import ExtractedEntity, _resolve_entities_with_embeddings


@pytest.mark.anyio
async def test_fake_embedder_supports_embed_texts_compat() -> None:
    embedder = FakeEmbedder(dimension=16)
    vectors = await embedder.embed_texts(["alpha", "beta"])
    assert len(vectors) == 2
    assert all(len(vector) == 16 for vector in vectors)


@pytest.mark.anyio
async def test_entity_resolution_works_with_embed_batch_embedder() -> None:
    entities = [
        ExtractedEntity(
            name="User Management",
            type="CONCEPT",
            description="Handles users and roles.",
            aliases=["users"],
            source_symbols=["UserService"],
        )
    ]
    resolved = await _resolve_entities_with_embeddings(
        entities=entities,
        embedder=FakeEmbedder(dimension=32),
        similarity_threshold=0.85,
    )
    assert len(resolved) == 1
    assert resolved[0].name == "User Management"
