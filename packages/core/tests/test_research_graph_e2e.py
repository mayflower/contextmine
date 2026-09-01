"""End-to-end coverage for the compiled research LangGraph."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from contextmine_core.research.agent import AgentConfig, ResearchAgent
from contextmine_core.research.llm.mock import MockLLMProvider
from contextmine_core.research.run import RunStatus
from contextmine_core.research.verification import VerificationStatus
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedToolCallingModel:
    """Drive one search/finalize cycle while recording the real graph messages."""

    def __init__(self) -> None:
        self.bound_tool_names: list[str] = []
        self.invocations: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[Any]) -> ScriptedToolCallingModel:
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.invocations.append(list(messages))

        if len(self.invocations) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "hybrid_search",
                        "args": {"query": "authentication password verification", "k": 1},
                        "id": "search-call",
                        "type": "tool_call",
                    }
                ],
            )

        if len(self.invocations) == 2:
            tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
            assert len(tool_messages) == 1
            assert tool_messages[0].name == "hybrid_search"

            citation = re.search(r"\[(ev-[a-zA-Z0-9]+-\d+)\]", str(tool_messages[0].content))
            assert citation is not None
            evidence_id = citation.group(1)

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "finalize",
                        "args": {
                            "answer": (
                                "Authentication delegates password checks to verify_password "
                                f"[{evidence_id}].\n\nConfidence: 0.5"
                            ),
                            "confidence": 0.5,
                        },
                        "id": "finalize-call",
                        "type": "tool_call",
                    }
                ],
            )

        raise AssertionError("The compiled graph requested an unexpected third model turn")


class ScriptedResearchProvider(MockLLMProvider):
    """Combine deterministic tool calls with deterministic semantic grounding."""

    def __init__(self, model: ScriptedToolCallingModel) -> None:
        super().__init__(
            structured_responses={
                "GroundingCheckResult": {
                    "is_grounded": True,
                    "grounding_score": 1.0,
                    "ungrounded_claims": [],
                    "reasoning": "Every answer claim is present in the fixed evidence.",
                }
            }
        )
        self._model = model


@pytest.mark.anyio
async def test_research_executes_the_compiled_graph_end_to_end() -> None:
    """Exercise agent -> tools -> agent -> tools -> verify -> END without graph mocks."""
    model = ScriptedToolCallingModel()
    provider = ScriptedResearchProvider(model)
    agent = ResearchAgent(
        llm_provider=provider,
        config=AgentConfig(max_steps=4, store_artifacts=False),
    )

    settings = SimpleNamespace(
        default_embedding_model="mock:fixture",
        verification_confidence_tolerance=0.2,
        verification_min_evidence_support=0.5,
        verification_require_citations=True,
    )
    embedder = AsyncMock()
    embedder.embed_batch.return_value = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
    search_result = SimpleNamespace(
        results=[
            SimpleNamespace(
                uri="src/auth.py",
                content=(
                    "def authenticate(credentials):\n"
                    "    return verify_password(credentials.password)"
                ),
                score=1.0,
            )
        ]
    )

    with (
        patch("contextmine_core.settings.get_settings", return_value=settings),
        patch(
            "contextmine_core.embeddings.parse_embedding_model_spec",
            return_value=("mock", "fixture"),
        ),
        patch("contextmine_core.embeddings.get_embedder", return_value=embedder),
        patch(
            "contextmine_core.search.hybrid_search",
            new_callable=AsyncMock,
            return_value=search_result,
        ) as search,
    ):
        result = await agent.research("How are passwords verified?")

    assert result.status is RunStatus.DONE
    assert result.error_message is None
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.file_path == "src/auth.py"
    assert evidence.provenance == "hybrid"
    assert evidence.score == 1.0
    assert result.answer == (
        "Authentication delegates password checks to verify_password "
        f"[{evidence.id}].\n\nConfidence: 0.5"
    )

    assert result.verification is not None
    assert result.verification.status is VerificationStatus.PASSED
    assert [citation.citation_id for citation in result.verification.citations] == [evidence.id]

    assert {"hybrid_search", "finalize"}.issubset(model.bound_tool_names)
    assert len(model.invocations) == 2
    assert isinstance(model.invocations[0][0], SystemMessage)
    assert isinstance(model.invocations[0][1], HumanMessage)
    assert any(isinstance(message, ToolMessage) for message in model.invocations[1])

    embedder.embed_batch.assert_awaited_once_with(["authentication password verification"])
    search.assert_awaited_once()
    assert [call["method"] for call in provider.call_history] == ["generate_structured"]
