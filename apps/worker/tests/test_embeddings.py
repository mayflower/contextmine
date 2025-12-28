"""Tests for embedding service."""

import pytest
from contextmine_core import EmbeddingProvider
from contextmine_worker.embeddings import (
    EmbeddingResult,
    FakeEmbedder,
    GeminiEmbedder,
    OpenAIEmbedder,
    get_embedder,
    parse_embedding_model_spec,
)

pytestmark = pytest.mark.anyio


class TestFakeEmbedder:
    """Tests for the FakeEmbedder class."""

    async def test_embed_batch_returns_embeddings(self) -> None:
        """Test that embed_batch returns embeddings for all texts."""
        embedder = FakeEmbedder(dimension=128)
        texts = ["Hello world", "Test text", "Another sample"]

        result = await embedder.embed_batch(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert all(len(emb) == 128 for emb in result.embeddings)

    async def test_embed_batch_deterministic(self) -> None:
        """Test that the same text produces the same embedding."""
        embedder = FakeEmbedder(dimension=128)
        text = "Consistent text"

        result1 = await embedder.embed_batch([text])
        result2 = await embedder.embed_batch([text])

        assert result1.embeddings[0] == result2.embeddings[0]

    async def test_embed_batch_different_texts_produce_different_embeddings(
        self,
    ) -> None:
        """Test that different texts produce different embeddings."""
        embedder = FakeEmbedder(dimension=128)

        result = await embedder.embed_batch(["Text A", "Text B"])

        assert result.embeddings[0] != result.embeddings[1]

    async def test_embed_batch_normalized_vectors(self) -> None:
        """Test that embeddings are normalized (magnitude ~= 1)."""
        embedder = FakeEmbedder(dimension=128)

        result = await embedder.embed_batch(["Test text"])
        embedding = result.embeddings[0]

        magnitude = sum(x * x for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 0.0001

    async def test_embed_batch_tokens_used(self) -> None:
        """Test that tokens_used is estimated from word count."""
        embedder = FakeEmbedder(dimension=128)
        texts = ["one two three", "four five"]

        result = await embedder.embed_batch(texts)

        # 3 words + 2 words = 5 tokens
        assert result.tokens_used == 5

    def test_provider_property(self) -> None:
        """Test the provider property."""
        embedder = FakeEmbedder()
        assert embedder.provider == EmbeddingProvider.OPENAI

    def test_model_name_property(self) -> None:
        """Test the model_name property."""
        embedder = FakeEmbedder()
        assert embedder.model_name == "fake-embedding-model"

    def test_dimension_property(self) -> None:
        """Test the dimension property."""
        embedder = FakeEmbedder(dimension=256)
        assert embedder.dimension == 256

    def test_default_dimension(self) -> None:
        """Test the default dimension is 1536."""
        embedder = FakeEmbedder()
        assert embedder.dimension == 1536


class TestOpenAIEmbedder:
    """Tests for the OpenAIEmbedder class."""

    def test_init_with_valid_model(self) -> None:
        """Test initialization with a valid model."""
        embedder = OpenAIEmbedder(
            model_name="text-embedding-3-small",
            api_key="test-key",
        )
        assert embedder.model_name == "text-embedding-3-small"
        assert embedder.dimension == 1536

    def test_init_with_unknown_model_raises_error(self) -> None:
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown OpenAI model"):
            OpenAIEmbedder(model_name="unknown-model", api_key="test-key")

    def test_init_without_api_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises ValueError."""
        import contextmine_core.settings as settings_module

        # Clear any cached settings and ensure no API key is set
        monkeypatch.setattr(settings_module, "_settings", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key required"):
            OpenAIEmbedder(model_name="text-embedding-3-small", api_key=None)

    def test_provider_property(self) -> None:
        """Test the provider property."""
        embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder.provider == EmbeddingProvider.OPENAI

    def test_dimensions_for_models(self) -> None:
        """Test dimensions for different models."""
        embedder_small = OpenAIEmbedder(model_name="text-embedding-3-small", api_key="test")
        embedder_large = OpenAIEmbedder(model_name="text-embedding-3-large", api_key="test")
        embedder_ada = OpenAIEmbedder(model_name="text-embedding-ada-002", api_key="test")

        assert embedder_small.dimension == 1536
        assert embedder_large.dimension == 3072
        assert embedder_ada.dimension == 1536


class TestGeminiEmbedder:
    """Tests for the GeminiEmbedder class."""

    def test_init_with_valid_model(self) -> None:
        """Test initialization with a valid model."""
        embedder = GeminiEmbedder(
            model_name="text-embedding-004",
            api_key="test-key",
        )
        assert embedder.model_name == "text-embedding-004"
        assert embedder.dimension == 768

    def test_init_with_unknown_model_raises_error(self) -> None:
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown Gemini model"):
            GeminiEmbedder(model_name="unknown-model", api_key="test-key")

    def test_init_without_api_key_raises_error(self) -> None:
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="API key required"):
            GeminiEmbedder(model_name="text-embedding-004", api_key=None)

    def test_provider_property(self) -> None:
        """Test the provider property."""
        embedder = GeminiEmbedder(api_key="test-key")
        assert embedder.provider == EmbeddingProvider.GEMINI


class TestGetEmbedder:
    """Tests for the get_embedder factory function."""

    def test_get_openai_embedder(self) -> None:
        """Test getting an OpenAI embedder."""
        embedder = get_embedder(EmbeddingProvider.OPENAI, api_key="test-key")
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.model_name == "text-embedding-3-small"

    def test_get_openai_embedder_with_custom_model(self) -> None:
        """Test getting an OpenAI embedder with custom model."""
        embedder = get_embedder(
            EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-large",
            api_key="test-key",
        )
        assert embedder.model_name == "text-embedding-3-large"
        assert embedder.dimension == 3072

    def test_get_gemini_embedder(self) -> None:
        """Test getting a Gemini embedder."""
        embedder = get_embedder(EmbeddingProvider.GEMINI, api_key="test-key")
        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.model_name == "text-embedding-004"

    def test_get_embedder_with_string_provider(self) -> None:
        """Test getting an embedder with string provider."""
        embedder = get_embedder("openai", api_key="test-key")
        assert isinstance(embedder, OpenAIEmbedder)

    def test_get_embedder_unknown_provider(self) -> None:
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError):
            get_embedder("unknown", api_key="test-key")


class TestParseEmbeddingModelSpec:
    """Tests for the parse_embedding_model_spec function."""

    def test_parse_openai_spec(self) -> None:
        """Test parsing OpenAI model spec."""
        provider, model_name = parse_embedding_model_spec("openai:text-embedding-3-small")
        assert provider == EmbeddingProvider.OPENAI
        assert model_name == "text-embedding-3-small"

    def test_parse_gemini_spec(self) -> None:
        """Test parsing Gemini model spec."""
        provider, model_name = parse_embedding_model_spec("gemini:text-embedding-004")
        assert provider == EmbeddingProvider.GEMINI
        assert model_name == "text-embedding-004"

    def test_parse_invalid_spec_no_colon(self) -> None:
        """Test that spec without colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model spec"):
            parse_embedding_model_spec("openai-text-embedding-3-small")

    def test_parse_invalid_provider(self) -> None:
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError):
            parse_embedding_model_spec("unknown:some-model")
