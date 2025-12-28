"""Context assembly service for generating Markdown documents from retrieved chunks."""

import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from contextmine_core.search import SearchResult, hybrid_search
from contextmine_core.settings import get_settings


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


@dataclass
class ContextRequest:
    """Request for context assembly."""

    query: str
    collection_id: str | None = None
    max_chunks: int = 10
    max_tokens: int = 4000
    provider: LLMProvider = LLMProvider.OPENAI
    model: str | None = None


@dataclass
class ContextResponse:
    """Response from context assembly."""

    markdown: str
    query: str
    chunks_used: int
    sources: list[dict]


SYSTEM_PROMPT = """You are a helpful documentation assistant. Your task is to create a clear,
well-organized Markdown document that answers the user's query based ONLY on the provided context chunks.

IMPORTANT RULES:
1. Use ONLY the information from the provided chunks. Do not make up or hallucinate any information.
2. Preserve all code fences exactly as they appear in the chunks. Do not modify code examples.
3. If the chunks don't contain enough information to answer the query, say so clearly.
4. Structure your response with clear headings and sections.
5. At the end, include a "## Sources" section listing the document URIs used.

Format code blocks with the appropriate language identifier when known."""


def build_context_prompt(query: str, chunks: list[SearchResult]) -> str:
    """Build the prompt for context assembly.

    Args:
        query: The user's query
        chunks: Retrieved chunks to use as context

    Returns:
        The formatted prompt string
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"### Chunk {i} (from: {chunk.uri})\n\n{chunk.content}")

    context_text = "\n\n---\n\n".join(context_parts)

    return f"""Based on the following context chunks, answer the query.

## Query
{query}

## Context Chunks
{context_text}

Please provide a comprehensive answer using ONLY the information from the chunks above.
Include a "## Sources" section at the end listing the URIs of the chunks you used."""


def extract_sources(chunks: list[SearchResult]) -> list[dict]:
    """Extract source information from chunks.

    Args:
        chunks: List of search results

    Returns:
        List of source dictionaries with uri, title, and optional file_path
    """
    seen_uris: set[str] = set()
    sources = []

    for chunk in chunks:
        if chunk.uri not in seen_uris:
            seen_uris.add(chunk.uri)
            source = {
                "uri": chunk.uri,
                "title": chunk.title,
            }
            # Extract file_path from git:// URIs
            if chunk.uri.startswith("git://"):
                # Format: git://github.com/{owner}/{repo}/{path}?ref={branch}
                parts = chunk.uri.split("?")[0].split("/")
                if len(parts) > 5:
                    source["file_path"] = "/".join(parts[5:])
            sources.append(source)

    return sources


class LLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt with context
            max_tokens: Maximum tokens for the response

        Returns:
            The generated text
        """
        pass

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            system_prompt: The system prompt
            user_prompt: The user prompt with context
            max_tokens: Maximum tokens for the response

        Yields:
            Text chunks as they are generated
        """
        # Default implementation: just yield the full response
        result = await self.generate(system_prompt, user_prompt, max_tokens)
        yield result


class FakeLLM(LLM):
    """Fake LLM for testing that returns deterministic Markdown."""

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate a deterministic response for testing."""
        # Extract query from user prompt
        lines = user_prompt.split("\n")
        query = ""
        for i, line in enumerate(lines):
            if line.strip() == "## Query":
                if i + 1 < len(lines):
                    query = lines[i + 1].strip()
                break

        # Extract chunks and their content
        chunks_content = []
        in_chunk = False
        current_chunk = []

        for line in lines:
            if line.startswith("### Chunk"):
                if current_chunk:
                    chunks_content.append("\n".join(current_chunk))
                current_chunk = []
                in_chunk = True
            elif line.strip() == "---":
                if current_chunk:
                    chunks_content.append("\n".join(current_chunk))
                current_chunk = []
                in_chunk = False
            elif in_chunk:
                current_chunk.append(line)

        if current_chunk:
            chunks_content.append("\n".join(current_chunk))

        # Extract URIs from chunks
        uris = []
        for line in lines:
            if "(from:" in line:
                start = line.find("(from:") + 7
                end = line.find(")", start)
                if end > start:
                    uris.append(line[start:end].strip())

        # Build deterministic response
        response_parts = [
            f"# Response to: {query}",
            "",
            "## Summary",
            "",
            f"Based on {len(chunks_content)} retrieved chunks, here is the relevant information:",
            "",
        ]

        # Include first chunk content (preserving code fences)
        if chunks_content:
            response_parts.append("## Relevant Content")
            response_parts.append("")
            response_parts.append(chunks_content[0])
            response_parts.append("")

        # Add sources section
        response_parts.append("## Sources")
        response_parts.append("")
        for uri in uris:
            response_parts.append(f"- {uri}")

        return "\n".join(response_parts)


class OpenAILLM(LLM):
    """OpenAI LLM implementation."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or get_settings().openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key required")

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate using OpenAI API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Generate streaming response using OpenAI API."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLM(LLM):
    """Anthropic LLM implementation."""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or get_settings().anthropic_api_key
        if not self.api_key:
            raise ValueError("Anthropic API key required")

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate using Anthropic API."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)  # type: ignore[union-attr]
        return "".join(text_parts)

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Generate streaming response using Anthropic API."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        async with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GeminiLLM(LLM):
    """Google Gemini LLM implementation."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or get_settings().gemini_api_key
        if not self.api_key:
            raise ValueError("Gemini API key required")

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Generate using Gemini API."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Generate streaming response using Gemini API."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        stream = await client.aio.models.generate_content_stream(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text


def get_llm(
    provider: LLMProvider | str,
    model: str | None = None,
    api_key: str | None = None,
) -> LLM:
    """Get an LLM instance for the given provider.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Optional model name
        api_key: Optional API key

    Returns:
        LLM instance
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider)

    settings = get_settings()

    if provider == LLMProvider.OPENAI:
        return OpenAILLM(
            model=model or settings.default_llm_model,
            api_key=api_key,
        )
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicLLM(
            model=model or "claude-3-haiku-20240307",
            api_key=api_key,
        )
    elif provider == LLMProvider.GEMINI:
        return GeminiLLM(
            model=model or "gemini-2.0-flash",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def assemble_context(
    query: str,
    user_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    max_chunks: int = 10,
    max_tokens: int = 4000,
    provider: LLMProvider | str = LLMProvider.OPENAI,
    model: str | None = None,
    llm: LLM | None = None,
) -> ContextResponse:
    """Assemble a context document from retrieved chunks.

    Args:
        query: The user's query
        user_id: Optional user ID for access control
        collection_id: Optional collection to search within
        max_chunks: Maximum number of chunks to retrieve
        max_tokens: Maximum tokens for LLM response
        provider: LLM provider to use
        model: Optional model override
        llm: Optional LLM instance (for testing with FakeLLM)

    Returns:
        ContextResponse with assembled Markdown and metadata
    """
    # Import here to avoid circular imports
    from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec

    settings = get_settings()

    # Get query embedding
    try:
        emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(emb_provider, emb_model)
    except (ValueError, Exception):
        embedder = FakeEmbedder()

    embed_result = await embedder.embed_batch([query])
    query_embedding = embed_result.embeddings[0]

    # Retrieve chunks
    search_response = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_id,
        top_k=max_chunks,
    )

    chunks = search_response.results

    if not chunks:
        return ContextResponse(
            markdown=f"# No Results\n\nNo relevant content found for query: {query}",
            query=query,
            chunks_used=0,
            sources=[],
        )

    # Build prompt
    user_prompt = build_context_prompt(query, chunks)

    # Get or create LLM
    if llm is None:
        try:
            llm = get_llm(provider, model)
        except ValueError:
            # Fall back to FakeLLM if no API key
            llm = FakeLLM()

    # Generate response
    markdown = await llm.generate(SYSTEM_PROMPT, user_prompt, max_tokens)

    # Extract sources
    sources = extract_sources(chunks)

    return ContextResponse(
        markdown=markdown,
        query=query,
        chunks_used=len(chunks),
        sources=sources,
    )


@dataclass
class StreamingContextMetadata:
    """Metadata sent before streaming content."""

    query: str
    chunks_used: int
    sources: list[dict]


async def assemble_context_stream(
    query: str,
    user_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    max_chunks: int = 10,
    max_tokens: int = 4000,
    provider: LLMProvider | str = LLMProvider.OPENAI,
    model: str | None = None,
) -> AsyncIterator[str | StreamingContextMetadata]:
    """Assemble a context document with streaming response.

    Yields:
        First yield: StreamingContextMetadata with query info and sources
        Subsequent yields: Text chunks from the LLM

    Args:
        query: The user's query
        user_id: Optional user ID for access control
        collection_id: Optional collection to search within
        max_chunks: Maximum number of chunks to retrieve
        max_tokens: Maximum tokens for LLM response
        provider: LLM provider to use
        model: Optional model override
    """
    from contextmine_core.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec

    settings = get_settings()

    # Get query embedding
    try:
        emb_provider, emb_model = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(emb_provider, emb_model)
    except (ValueError, Exception):
        embedder = FakeEmbedder()

    embed_result = await embedder.embed_batch([query])
    query_embedding = embed_result.embeddings[0]

    # Retrieve chunks
    search_response = await hybrid_search(
        query=query,
        query_embedding=query_embedding,
        user_id=user_id,
        collection_id=collection_id,
        top_k=max_chunks,
    )

    chunks = search_response.results

    if not chunks:
        yield StreamingContextMetadata(query=query, chunks_used=0, sources=[])
        yield f"# No Results\n\nNo relevant content found for query: {query}"
        return

    # Extract sources and yield metadata first
    sources = extract_sources(chunks)
    yield StreamingContextMetadata(query=query, chunks_used=len(chunks), sources=sources)

    # Build prompt and stream response
    user_prompt = build_context_prompt(query, chunks)

    try:
        llm = get_llm(provider, model)
    except ValueError:
        llm = FakeLLM()

    async for text_chunk in llm.generate_stream(SYSTEM_PROMPT, user_prompt, max_tokens):
        yield text_chunk
