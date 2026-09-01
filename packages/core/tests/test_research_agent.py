"""Behavioral tests for the persistent LangGraph research agent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.research.agent import (
    AgentConfig,
    AgentState,
    FinalizeInput,
    HybridSearchInput,
    OpenSpanInput,
    ResearchAgent,
    _escape_like_pattern,
    create_tools,
)
from contextmine_core.research.run import Evidence, ResearchRun, RunStatus
from contextmine_core.research.verification.models import (
    ConfidenceCalibration,
    EvidenceSupportScore,
    VerificationResult,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _run(question: str = "How does auth work?", budget: int = 10) -> ResearchRun:
    return ResearchRun.create(question=question, budget_steps=budget)


def _passed_verification() -> VerificationResult:
    return VerificationResult.create_passed(
        citations=[],
        evidence_support=EvidenceSupportScore(
            score=1.0,
            reasoning="The answer is supported.",
        ),
        confidence_calibration=ConfidenceCalibration(
            stated_confidence=0.9,
            evidence_confidence=0.9,
            calibration_delta=0.0,
            is_calibrated=True,
        ),
    )


def _state(run: ResearchRun, tool_name: str, args: dict[str, Any]) -> AgentState:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": "call-1", "type": "tool_call"}],
            )
        ],
        "run": run.to_dict(),
        "pending_answer": None,
        "verification_attempts": 0,
        "confidence": 0.8,
    }


async def _invoke_tool(
    tool_name: str,
    args: dict[str, Any],
    run: ResearchRun | None = None,
) -> AgentState:
    workflow = StateGraph(AgentState)
    workflow.add_node("tools", ToolNode(create_tools()))
    workflow.set_entry_point("tools")
    workflow.set_finish_point("tools")
    return await workflow.compile().ainvoke(_state(run or _run(), tool_name, args))


class _FinalizeModel:
    def __init__(self, answer: str = "Authentication uses signed sessions.") -> None:
        self.answer = answer
        self.calls = 0

    async def ainvoke(self, _messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "finalize",
                    "args": {"answer": self.answer, "confidence": 0.9},
                    "id": f"finalize-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


class _Provider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self, model: _FinalizeModel | None = None) -> None:
        self.model = model or _FinalizeModel()

    def bind_tools(self, _tools: list[Any]) -> _FinalizeModel:
        return self.model


def test_input_contracts_and_escape_behavior() -> None:
    assert HybridSearchInput(query="auth").k == 10
    assert OpenSpanInput(file_path="src/main.py", start_line=1, end_line=5).end_line == 5
    assert FinalizeInput(answer="answer").confidence == 0.8
    assert _escape_like_pattern("a\\b%c_d") == "a\\\\b\\%c\\_d"


def test_tool_contract_exposes_only_model_controlled_arguments() -> None:
    tools = {tool.name: tool for tool in create_tools()}

    assert len(tools) == 18
    assert "runtime" not in tools["hybrid_search"].args
    assert set(tools["finalize"].args) == {"answer", "confidence"}
    assert set(tools["graph_trace"].args) == {
        "from_symbol",
        "to_symbol",
        "edge_types",
        "max_depth",
    }


@pytest.mark.anyio
async def test_finalize_updates_native_graph_state_and_emits_tool_message() -> None:
    result = await _invoke_tool("finalize", {"answer": "answer", "confidence": 1.4})

    assert result["pending_answer"] == "answer"
    assert result["confidence"] == 1.0
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].tool_call_id == "call-1"


@pytest.mark.anyio
async def test_evidence_tool_updates_serializable_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def query(_name: str, _path: str | None, run: ResearchRun) -> str:
        evidence = Evidence(
            id="temporary",
            file_path="src/auth.py",
            start_line=10,
            end_line=20,
            content="def authenticate(): ...",
            reason="definition",
            provenance="symbol_index",
        )
        run.add_evidence(evidence)
        return f"[{evidence.id}] definition"

    monkeypatch.setattr("contextmine_core.research.agent._query_goto_definition", query)
    result = await _invoke_tool("goto_definition", {"symbol_name": "authenticate"})
    restored = ResearchRun.from_dict(result["run"])

    assert len(restored.evidence) == 1
    assert restored.evidence[0].file_path == "src/auth.py"
    assert restored.evidence[0].id in result["messages"][-1].content


def test_evidence_is_deduplicated_across_replayed_tool_results() -> None:
    run = _run()
    first = Evidence("first", "src/a.py", 1, 2, "same", "reason", "manual")
    replay = Evidence("second", "src/a.py", 1, 2, "same", "reason", "manual")

    run.add_evidence(first)
    run.add_evidence(replay)

    assert len(run.evidence) == 1
    assert replay.id == first.id
    assert ResearchRun.from_dict(run.to_dict()).evidence[0].id == first.id


def test_graph_compiles_once_with_public_provider_tool_binding() -> None:
    provider = MagicMock()
    provider.bind_tools.return_value = MagicMock()
    agent = ResearchAgent(provider, checkpointer=InMemorySaver())

    first = agent._graph_for_execution()
    second = agent._graph_for_execution()

    assert first is second
    provider.bind_tools.assert_called_once()
    assert {node for node in first.get_graph().nodes} >= {"agent", "tools", "verify"}


@pytest.mark.anyio
async def test_run_persists_by_run_id_and_completed_resume_is_a_noop() -> None:
    saver = InMemorySaver()
    model = _FinalizeModel()
    agent = ResearchAgent(
        _Provider(model),
        AgentConfig(store_artifacts=False),
        checkpointer=saver,
    )
    verification = _passed_verification()

    with patch(
        "contextmine_core.research.agent.AnswerVerifier.verify_async",
        new=AsyncMock(return_value=verification),
    ):
        completed = await agent.research("How does auth work?")

    snapshot = await agent._graph_for_execution().aget_state(
        {"configurable": {"thread_id": completed.run_id}}
    )
    restarted = ResearchAgent(
        _Provider(_FinalizeModel("must not run")),
        AgentConfig(store_artifacts=False),
        checkpointer=saver,
    )
    resumed = await restarted.resume(completed.run_id)

    assert completed.status == RunStatus.DONE
    assert snapshot.values["run"]["run_id"] == completed.run_id
    assert resumed.to_dict() == completed.to_dict()
    assert restarted.llm_provider.model.calls == 0


@pytest.mark.anyio
async def test_interrupted_checkpoint_resumes_after_agent_recreation() -> None:
    saver = InMemorySaver()
    first = ResearchAgent(_Provider(), AgentConfig(store_artifacts=False), checkpointer=saver)
    run = _run("Where is auth?")
    graph_config = {"configurable": {"thread_id": run.run_id}}
    await first._graph_for_execution().aupdate_state(
        graph_config,
        first._initial_state(run),
        as_node="__start__",
    )

    restarted = ResearchAgent(_Provider(), AgentConfig(store_artifacts=False), checkpointer=saver)
    verification = _passed_verification()
    with patch(
        "contextmine_core.research.agent.AnswerVerifier.verify_async",
        new=AsyncMock(return_value=verification),
    ):
        resumed = await restarted.resume(run.run_id)

    assert resumed.status == RunStatus.DONE
    assert resumed.run_id == run.run_id
    assert restarted.llm_provider.model.calls == 1


@pytest.mark.anyio
async def test_cancelled_checkpoint_does_not_restart() -> None:
    saver = InMemorySaver()
    agent = ResearchAgent(_Provider(), AgentConfig(store_artifacts=False), checkpointer=saver)
    run = _run("Where is auth?")
    graph_config = {"configurable": {"thread_id": run.run_id}}
    await agent._graph_for_execution().aupdate_state(
        graph_config,
        agent._initial_state(run),
        as_node="__start__",
    )

    cancelled = await agent.cancel(run.run_id)
    restarted = ResearchAgent(_Provider(), AgentConfig(store_artifacts=False), checkpointer=saver)
    resumed = await restarted.resume(run.run_id)

    assert cancelled.status == RunStatus.CANCELLED
    assert resumed.status == RunStatus.CANCELLED
    assert restarted.llm_provider.model.calls == 0
