"""Embedding service - re-exports from contextmine_core for backwards compatibility."""

from contextmine_core.embeddings import (
    Embedder,
    EmbeddingResult,
    FakeEmbedder,
    GeminiEmbedder,
    OpenAIEmbedder,
    get_embedder,
    parse_embedding_model_spec,
)

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "FakeEmbedder",
    "GeminiEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
    "parse_embedding_model_spec",
]
