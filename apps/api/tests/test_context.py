"""Tests for context assembly functionality."""

import pytest
from contextmine_core.context import (
    FakeLLM,
    build_context_prompt,
    extract_sources,
)
from contextmine_core.search import SearchResult


def make_search_result(
    chunk_id: str,
    uri: str,
    title: str,
    content: str,
    score: float = 0.9,
    fts_rank: int | None = 1,
    vector_rank: int | None = None,
) -> SearchResult:
    """Helper to create SearchResult with default values."""
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="source-1",
        collection_id="collection-1",
        uri=uri,
        title=title,
        content=content,
        score=score,
        fts_rank=fts_rank,
        vector_rank=vector_rank,
        fts_score=score if fts_rank else None,
        vector_score=score if vector_rank else None,
    )


class TestBuildContextPrompt:
    """Tests for building context prompts."""

    def test_build_prompt_basic(self) -> None:
        """Test building a prompt with basic chunks."""
        chunks = [
            make_search_result(
                chunk_id="chunk-1",
                uri="https://example.com/doc1",
                title="Document 1",
                content="This is the first chunk.",
                score=0.9,
                fts_rank=1,
            ),
            make_search_result(
                chunk_id="chunk-2",
                uri="https://example.com/doc2",
                title="Document 2",
                content="This is the second chunk.",
                score=0.8,
                fts_rank=2,
            ),
        ]

        prompt = build_context_prompt("How do I test?", chunks)

        assert "## Query" in prompt
        assert "How do I test?" in prompt
        assert "## Context Chunks" in prompt
        assert "### Chunk 1 (from: https://example.com/doc1)" in prompt
        assert "### Chunk 2 (from: https://example.com/doc2)" in prompt
        assert "This is the first chunk." in prompt
        assert "This is the second chunk." in prompt

    def test_build_prompt_preserves_code_fences(self) -> None:
        """Test that code fences are preserved in the prompt."""
        code_content = """Here's how to do it:

```python
def hello():
    print("Hello, world!")
```

And that's it."""

        chunks = [
            make_search_result(
                chunk_id="chunk-1",
                uri="https://example.com/code",
                title="Code Example",
                content=code_content,
            ),
        ]

        prompt = build_context_prompt("Show me code", chunks)

        # Code fence should be preserved exactly
        assert "```python" in prompt
        assert 'def hello():' in prompt
        assert '    print("Hello, world!")' in prompt
        assert "```" in prompt


class TestExtractSources:
    """Tests for extracting sources from chunks."""

    def test_extract_sources_basic(self) -> None:
        """Test extracting sources from chunks."""
        chunks = [
            make_search_result(
                chunk_id="chunk-1",
                uri="https://example.com/doc1",
                title="Document 1",
                content="Content 1",
            ),
            make_search_result(
                chunk_id="chunk-2",
                uri="https://example.com/doc2",
                title="Document 2",
                content="Content 2",
                score=0.8,
                fts_rank=2,
            ),
        ]

        sources = extract_sources(chunks)

        assert len(sources) == 2
        assert sources[0]["uri"] == "https://example.com/doc1"
        assert sources[0]["title"] == "Document 1"
        assert sources[1]["uri"] == "https://example.com/doc2"
        assert sources[1]["title"] == "Document 2"

    def test_extract_sources_deduplicates(self) -> None:
        """Test that duplicate URIs are deduplicated."""
        chunks = [
            make_search_result(
                chunk_id="chunk-1",
                uri="https://example.com/doc1",
                title="Document 1",
                content="Content 1",
            ),
            make_search_result(
                chunk_id="chunk-2",
                uri="https://example.com/doc1",  # Same URI
                title="Document 1",
                content="Content 2 from same doc",
                score=0.8,
                fts_rank=2,
            ),
        ]

        sources = extract_sources(chunks)

        assert len(sources) == 1
        assert sources[0]["uri"] == "https://example.com/doc1"

    def test_extract_sources_git_uri(self) -> None:
        """Test extracting file_path from git:// URIs."""
        chunks = [
            make_search_result(
                chunk_id="chunk-1",
                uri="git://github.com/owner/repo/src/main.py?ref=main",
                title="Main Module",
                content="Content",
            ),
        ]

        sources = extract_sources(chunks)

        assert len(sources) == 1
        assert sources[0]["uri"] == "git://github.com/owner/repo/src/main.py?ref=main"
        assert sources[0]["file_path"] == "src/main.py"


class TestFakeLLM:
    """Tests for the FakeLLM deterministic test helper."""

    @pytest.mark.anyio
    async def test_fake_llm_returns_deterministic_output(self) -> None:
        """Test that FakeLLM returns consistent output."""
        llm = FakeLLM()

        prompt = """Based on the following context chunks, answer the query.

## Query
How do I use the API?

## Context Chunks
### Chunk 1 (from: https://docs.example.com/api)

Use the API like this:

```python
import api
api.call()
```

---

### Chunk 2 (from: https://docs.example.com/auth)

First authenticate with your key.

Please provide a comprehensive answer using ONLY the information from the chunks above.
Include a "## Sources" section at the end listing the URIs of the chunks you used."""

        result1 = await llm.generate("system prompt", prompt, 1000)
        result2 = await llm.generate("system prompt", prompt, 1000)

        # Should be deterministic
        assert result1 == result2

    @pytest.mark.anyio
    async def test_fake_llm_includes_query(self) -> None:
        """Test that FakeLLM includes the query in response."""
        llm = FakeLLM()

        prompt = """Based on the following context chunks, answer the query.

## Query
How do I configure settings?

## Context Chunks
### Chunk 1 (from: https://docs.example.com/settings)

Configure settings in config.yaml.

Please provide a comprehensive answer."""

        result = await llm.generate("system prompt", prompt, 1000)

        assert "How do I configure settings?" in result

    @pytest.mark.anyio
    async def test_fake_llm_includes_sources(self) -> None:
        """Test that FakeLLM includes sources section."""
        llm = FakeLLM()

        prompt = """Based on the following context chunks, answer the query.

## Query
Test query

## Context Chunks
### Chunk 1 (from: https://docs.example.com/source1)

Content 1

---

### Chunk 2 (from: https://docs.example.com/source2)

Content 2

Please provide a comprehensive answer."""

        result = await llm.generate("system prompt", prompt, 1000)

        assert "## Sources" in result
        assert "https://docs.example.com/source1" in result
        assert "https://docs.example.com/source2" in result

    @pytest.mark.anyio
    async def test_fake_llm_preserves_code_fences(self) -> None:
        """Test that FakeLLM preserves code fences from chunks."""
        llm = FakeLLM()

        prompt = """Based on the following context chunks, answer the query.

## Query
Show me code

## Context Chunks
### Chunk 1 (from: https://docs.example.com/code)

Here's the code:

```python
def example():
    return 42
```

Please provide a comprehensive answer."""

        result = await llm.generate("system prompt", prompt, 1000)

        # Code fence should be preserved
        assert "```python" in result
        assert "def example():" in result
        assert "return 42" in result


class TestContextEndpoint:
    """Tests for the context assembly API endpoint."""

    @pytest.mark.anyio
    async def test_context_endpoint_invalid_provider(self) -> None:
        """Test that invalid provider returns 400."""
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post(
            "/api/context",
            json={
                "query": "test query",
                "provider": "invalid_provider",
            },
        )

        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]

    @pytest.mark.anyio
    async def test_context_endpoint_invalid_collection_id(self) -> None:
        """Test that invalid collection_id returns 400."""
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.post(
            "/api/context",
            json={
                "query": "test query",
                "collection_id": "not-a-uuid",
            },
        )

        assert response.status_code == 400
        assert "Invalid collection_id" in response.json()["detail"]
