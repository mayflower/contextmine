"""Tests for the fail-closed external model-call policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import contextmine_core.settings as settings_module
import pytest
from contextmine_core.architecture.agent_sdk import generate_arc42_with_claude_sdk
from contextmine_core.context import LLMProvider, OpenAILLM, get_llm
from contextmine_core.embeddings import OpenAIEmbedder, get_embedder
from contextmine_core.model_policy import ModelCallsDisabledError
from contextmine_core.research.llm.provider import LangChainProvider, get_llm_provider
from contextmine_core.settings import Settings


@pytest.fixture
def model_calls_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings(model_calls_enabled=False),
    )


def test_model_calls_are_enabled_by_default() -> None:
    assert Settings.model_fields["model_calls_enabled"].default is True


def test_model_factories_fail_closed(model_calls_disabled: None) -> None:
    with pytest.raises(ModelCallsDisabledError):
        get_embedder("openai", api_key="test-key")
    with pytest.raises(ModelCallsDisabledError):
        get_llm(LLMProvider.OPENAI, api_key="test-key")
    with pytest.raises(ModelCallsDisabledError):
        get_llm_provider("openai", api_key="test-key")


@pytest.mark.anyio
async def test_direct_clients_cannot_bypass_policy(model_calls_disabled: None) -> None:
    embedder = OpenAIEmbedder(api_key="test-key")
    llm = OpenAILLM(api_key="test-key")
    research_provider = LangChainProvider(
        model=MagicMock(),
        model_name="test-model",
    )

    with pytest.raises(ModelCallsDisabledError):
        await embedder.embed_batch(["content"])
    with pytest.raises(ModelCallsDisabledError):
        await llm.generate("system", "user", 10)
    with pytest.raises(ModelCallsDisabledError):
        await research_provider.generate_text(system="system", messages=[])


@pytest.mark.anyio
async def test_agent_sdk_generation_cannot_bypass_policy(
    model_calls_disabled: None,
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelCallsDisabledError):
        await generate_arc42_with_claude_sdk(
            collection_id=uuid4(),
            scenario_id=uuid4(),
            scenario_name="AS-IS",
            repo_path=tmp_path,
        )
