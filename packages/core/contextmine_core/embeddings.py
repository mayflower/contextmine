"""Embedding service for generating vector embeddings."""

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass

from contextmine_core.models import EmbeddingProvider
from contextmine_core.settings import get_settings
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class EmbeddingResult:
    """Result of embedding a batch of texts."""

    embeddings: list[list[float]]
    model_name: str
    dimension: int
    tokens_used: int = 0


class Embedder(ABC):
    """Abstract base class for embedders."""

    @property
    @abstractmethod
    def provider(self) -> EmbeddingProvider:
        """Get the provider enum value."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass


class FakeEmbedder(Embedder):
    """Fake embedder for testing that produces deterministic vectors.

    Generates consistent embeddings based on text hash, allowing
    tests to verify embedding behavior without external API calls.
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.OPENAI

    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            embedding = self._generate_deterministic_embedding(text)
            embeddings.append(embedding)

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimension=self._dimension,
            tokens_used=sum(len(t.split()) for t in texts),
        )

    def _generate_deterministic_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text hash.

        Uses SHA-256 hash to seed a simple PRNG for consistent results.
        """
        # Create hash of the text
        text_hash = hashlib.sha256(text.encode("utf-8")).digest()

        # Generate deterministic floats from hash bytes
        embedding = []
        hash_bytes = text_hash

        while len(embedding) < self._dimension:
            # Use 4 bytes at a time to create floats
            for i in range(0, len(hash_bytes), 4):
                if len(embedding) >= self._dimension:
                    break
                # Convert 4 bytes to float in range [-1, 1]
                value = struct.unpack("f", hash_bytes[i : i + 4])[0]
                # Normalize to [-1, 1]
                normalized = (value % 2) - 1
                embedding.append(normalized)

            # Re-hash to get more bytes if needed
            if len(embedding) < self._dimension:
                hash_bytes = hashlib.sha256(hash_bytes).digest()

        # Normalize the vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding[: self._dimension]


class OpenAIEmbedder(Embedder):
    """OpenAI embedder using the embeddings API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key or get_settings().openai_api_key

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if self._model_name not in self._dimensions:
            raise ValueError(f"Unknown OpenAI model: {self._model_name}")

        if not self._api_key:
            raise ValueError("OpenAI API key required")

    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.OPENAI

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimensions[self._model_name]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts using OpenAI API."""
        from contextmine_core.telemetry.spans import trace_embedding_call
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)

        async with trace_embedding_call("openai", self._model_name, len(texts)) as span:
            response = await client.embeddings.create(
                model=self._model_name,
                input=texts,
            )

            embeddings = [data.embedding for data in response.data]
            tokens_used = response.usage.total_tokens

            # Record token usage on the span
            span.set_attribute("embedding.tokens_used", tokens_used)

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self._model_name,
            dimension=self.dimension,
            tokens_used=tokens_used,
        )


class GeminiEmbedder(Embedder):
    """Google Gemini embedder using the embeddings API."""

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: str | None = None,
    ):
        self._model_name = model_name
        self._api_key = api_key or get_settings().gemini_api_key

        # Model dimensions
        self._dimensions = {
            "text-embedding-004": 768,
            "embedding-001": 768,
        }

        if self._model_name not in self._dimensions:
            raise ValueError(f"Unknown Gemini model: {self._model_name}")

        if not self._api_key:
            raise ValueError("Gemini API key required")

    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.GEMINI

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimensions[self._model_name]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts using Gemini API."""
        from contextmine_core.telemetry.spans import trace_embedding_call
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

        async with trace_embedding_call("gemini", self._model_name, len(texts)):
            result = await client.aio.models.embed_content(
                model=self._model_name,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )

            if result.embeddings is None:
                raise ValueError("Gemini API returned no embeddings")
            embeddings = [e.values for e in result.embeddings if e.values is not None]

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self._model_name,
            dimension=self.dimension,
            tokens_used=0,  # Gemini doesn't report token usage
        )


def get_embedder(
    provider: EmbeddingProvider | str,
    model_name: str | None = None,
    api_key: str | None = None,
) -> Embedder:
    """Get an embedder instance for the given provider.

    Args:
        provider: Provider enum or string like "openai" or "gemini"
        model_name: Model name (uses provider default if not specified)
        api_key: API key (uses settings if not specified)

    Returns:
        Embedder instance
    """
    if isinstance(provider, str):
        provider = EmbeddingProvider(provider)

    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder(
            model_name=model_name or "text-embedding-3-small",
            api_key=api_key,
        )
    elif provider == EmbeddingProvider.GEMINI:
        return GeminiEmbedder(
            model_name=model_name or "text-embedding-004",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def parse_embedding_model_spec(spec: str) -> tuple[EmbeddingProvider, str]:
    """Parse an embedding model specification like 'openai:text-embedding-3-small'.

    Returns:
        Tuple of (provider, model_name)
    """
    if ":" not in spec:
        raise ValueError(f"Invalid model spec: {spec}. Expected 'provider:model_name'")

    provider_str, model_name = spec.split(":", 1)
    provider = EmbeddingProvider(provider_str)
    return provider, model_name
