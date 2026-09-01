"""PostgreSQL integration test for resumable research runs."""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from contextmine_core.research.agent import AgentConfig, ResearchAgent
from contextmine_core.research.checkpoints import _checkpoint_connection_string
from contextmine_core.research.run import Evidence, ResearchRun, RunStatus
from contextmine_core.research.verification.models import (
    ConfidenceCalibration,
    EvidenceSupportScore,
    VerificationResult,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class _FinalizeModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, _messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "finalize",
                    "args": {"answer": "Authentication uses signed sessions.", "confidence": 0.9},
                    "id": f"finalize-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


class _Provider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self) -> None:
        self.model = _FinalizeModel()

    def bind_tools(self, _tools: list[Any]) -> _FinalizeModel:
        return self.model


def _passed_verification() -> VerificationResult:
    return VerificationResult.create_passed(
        citations=[],
        evidence_support=EvidenceSupportScore(1.0, "The answer is supported."),
        confidence_calibration=ConfidenceCalibration(0.9, 0.9, 0.0, True),
    )


async def _delete_thread(saver: AsyncPostgresSaver, thread_id: str) -> None:
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        await saver.conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set - skipping PostgreSQL checkpoint test",
)
async def test_research_resumes_across_postgres_connections_without_duplicate_evidence() -> None:
    database_url = os.environ["TEST_DATABASE_URL"]
    if not database_url.startswith(("postgresql+asyncpg://", "postgresql://")):
        pytest.skip("TEST_DATABASE_URL must point to PostgreSQL")
    connection_string = _checkpoint_connection_string(database_url)
    run = ResearchRun.create("How does auth work?")
    run.run_id = f"checkpoint-test-{uuid.uuid4()}"
    run.add_evidence(
        Evidence(
            id="auth-evidence",
            file_path="src/auth.py",
            start_line=1,
            end_line=5,
            content="def authenticate(): ...",
            reason="authentication implementation",
            provenance="test",
        )
    )

    try:
        async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
            await saver.setup()
            first = ResearchAgent(
                _Provider(),
                AgentConfig(store_artifacts=False),
                checkpointer=saver,
            )
            await first._graph_for_execution().aupdate_state(
                {"configurable": {"thread_id": run.run_id}},
                first._initial_state(run),
                as_node="__start__",
            )

        async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
            restarted = ResearchAgent(
                _Provider(),
                AgentConfig(store_artifacts=False),
                checkpointer=saver,
            )
            with patch(
                "contextmine_core.research.agent.AnswerVerifier.verify_async",
                new=AsyncMock(return_value=_passed_verification()),
            ):
                completed = await restarted.resume(run.run_id)

        async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
            terminal_reader = ResearchAgent(
                _Provider(),
                AgentConfig(store_artifacts=False),
                checkpointer=saver,
            )
            restored = await terminal_reader.resume(run.run_id)
            assert completed.status == RunStatus.DONE
            assert restored.to_dict() == completed.to_dict()
            assert len(restored.evidence) == 1
            assert terminal_reader.llm_provider.model.calls == 0
    finally:
        async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
            await _delete_thread(saver, run.run_id)
