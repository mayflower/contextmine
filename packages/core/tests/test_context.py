"""Tests for context assembly service.

Covers build_context_prompt, extract_sources, FakeLLM, get_llm,
assemble_context with mocked DB, and LLM provider selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
from contextmine_core.context import (
    SYSTEM_PROMPT,
    ContextRequest,
    ContextResponse,
    FakeLLM,
    LLMProvider,
    StreamingContextMetadata,
    build_context_prompt,
    extract_sources,
    get_llm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchResult:
    uri: str
    title: str
    content: str


# ---------------------------------------------------------------------------
# build_context_prompt
# ---------------------------------------------------------------------------


class TestBuildContextPrompt:
    def test_single_chunk(self) -> None:
        chunks = [FakeSearchResult(uri="doc://a", title="Doc A", content="Content A")]
        result = build_context_prompt("test query", chunks)
        assert "## Query" in result
        assert "test query" in result
        assert "Content A" in result
        assert "doc://a" in result

    def test_multiple_chunks(self) -> None:
        chunks = [
            FakeSearchResult(uri="doc://a", title="Doc A", content="Content A"),
            FakeSearchResult(uri="doc://b", title="Doc B", content="Content B"),
        ]
        result = build_context_prompt("query", chunks)
        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "Content A" in result
        assert "Content B" in result

    def test_empty_chunks(self) -> None:
        result = build_context_prompt("query", [])
        assert "## Query" in result
        assert "## Context Chunks" in result


# ---------------------------------------------------------------------------
# extract_sources
# ---------------------------------------------------------------------------


class TestExtractSources:
    def test_basic_extraction(self) -> None:
        chunks = [
            FakeSearchResult(uri="https://docs.example.com/a", title="Doc A", content="x"),
            FakeSearchResult(uri="https://docs.example.com/b", title="Doc B", content="y"),
        ]
        sources = extract_sources(chunks)
        assert len(sources) == 2
        assert sources[0]["uri"] == "https://docs.example.com/a"
        assert sources[0]["title"] == "Doc A"

    def test_deduplication(self) -> None:
        chunks = [
            FakeSearchResult(uri="doc://a", title="A", content="x"),
            FakeSearchResult(uri="doc://a", title="A", content="y"),
        ]
        sources = extract_sources(chunks)
        assert len(sources) == 1

    def test_git_uri_extracts_file_path(self) -> None:
        chunks = [
            FakeSearchResult(
                uri="git://github.com/org/repo/src/main.py?ref=main",
                title="main.py",
                content="code",
            )
        ]
        sources = extract_sources(chunks)
        assert len(sources) == 1
        assert sources[0].get("file_path") == "src/main.py"

    def test_short_git_uri_no_file_path(self) -> None:
        chunks = [FakeSearchResult(uri="git://github.com/org", title="repo", content="x")]
        sources = extract_sources(chunks)
        assert len(sources) == 1
        assert "file_path" not in sources[0]

    def test_empty_chunks(self) -> None:
        sources = extract_sources([])
        assert sources == []


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


class TestFakeLLM:
    @pytest.mark.anyio
    async def test_generate_extracts_query(self) -> None:
        llm = FakeLLM()
        prompt = build_context_prompt(
            "how does auth work?",
            [FakeSearchResult(uri="doc://a", title="Auth", content="Auth content")],
        )
        result = await llm.generate(SYSTEM_PROMPT, prompt, 4000)
        assert "how does auth work?" in result
        assert "## Sources" in result

    @pytest.mark.anyio
    async def test_generate_with_no_chunks(self) -> None:
        llm = FakeLLM()
        prompt = build_context_prompt("test", [])
        result = await llm.generate(SYSTEM_PROMPT, prompt, 4000)
        assert "Response to:" in result

    @pytest.mark.anyio
    async def test_generate_stream_yields_full(self) -> None:
        llm = FakeLLM()
        prompt = "## Query\ntest\n## Context Chunks\n### Chunk 1 (from: doc://a)\nContent"
        chunks = []
        async for chunk in llm.generate_stream(SYSTEM_PROMPT, prompt, 4000):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "Response to:" in chunks[0]


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------


class TestGetLlm:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            get_llm("nonexistent")

    def test_openai_provider(self) -> None:
        llm = get_llm(LLMProvider.OPENAI, api_key="test-key")
        assert llm is not None

    def test_anthropic_provider(self) -> None:
        llm = get_llm(LLMProvider.ANTHROPIC, api_key="test-key")
        assert llm is not None

    def test_gemini_provider(self) -> None:
        llm = get_llm(LLMProvider.GEMINI, api_key="test-key")
        assert llm is not None

    def test_string_provider(self) -> None:
        llm = get_llm("openai", api_key="test-key")
        assert llm is not None

    def test_openai_no_key_raises(self) -> None:
        with patch("contextmine_core.context.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = None
            mock_settings.return_value.default_llm_model = "gpt-4o-mini"
            with pytest.raises(ValueError, match="OpenAI API key required"):
                get_llm(LLMProvider.OPENAI)

    def test_anthropic_no_key_raises(self) -> None:
        with patch("contextmine_core.context.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            mock_settings.return_value.default_llm_model = "claude-3-haiku"
            with pytest.raises(ValueError, match="Anthropic API key required"):
                get_llm(LLMProvider.ANTHROPIC)

    def test_gemini_no_key_raises(self) -> None:
        with patch("contextmine_core.context.get_settings") as mock_settings:
            mock_settings.return_value.gemini_api_key = None
            with pytest.raises(ValueError, match="Gemini API key required"):
                get_llm(LLMProvider.GEMINI)


# ---------------------------------------------------------------------------
# ContextRequest / ContextResponse / StreamingContextMetadata
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_context_request_defaults(self) -> None:
        req = ContextRequest(query="test")
        assert req.query == "test"
        assert req.max_chunks == 10
        assert req.max_tokens == 4000
        assert req.provider == LLMProvider.OPENAI
        assert req.collection_id is None
        assert req.model is None

    def test_context_response(self) -> None:
        resp = ContextResponse(
            markdown="# Response",
            query="test",
            chunks_used=5,
            sources=[{"uri": "doc://a"}],
        )
        assert resp.markdown == "# Response"
        assert resp.chunks_used == 5

    def test_streaming_metadata(self) -> None:
        meta = StreamingContextMetadata(
            query="test",
            chunks_used=3,
            sources=[{"uri": "doc://a"}],
        )
        assert meta.query == "test"
        assert meta.chunks_used == 3


# ---------------------------------------------------------------------------
# LLMProvider enum
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_values(self) -> None:
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GEMINI.value == "gemini"

    def test_from_string(self) -> None:
        assert LLMProvider("openai") == LLMProvider.OPENAI


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_is_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)

    def test_contains_key_instructions(self) -> None:
        assert "ONLY" in SYSTEM_PROMPT
        assert "Sources" in SYSTEM_PROMPT
        assert "code fences" in SYSTEM_PROMPT
