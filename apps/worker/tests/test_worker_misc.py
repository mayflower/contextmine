"""Tests for worker misc modules: main.py, init_prefect.py, telemetry.py, chunking extras."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_worker.chunking import (
    ChunkResult,
    chunk_document,
    chunk_with_headers,
    compute_chunk_hash,
    is_markdown_file,
    split_code_file,
    split_markdown_preserving_code_fences,
)
from contextmine_worker.telemetry import (
    attach_trace_context,
    detach_trace_context,
    extract_trace_context,
    inject_trace_context,
    traced_flow,
    traced_task,
)

pytestmark = pytest.mark.anyio

# ---------------------------------------------------------------------------
# Worker telemetry decorators
# ---------------------------------------------------------------------------


class TestTracedFlow:
    async def test_traced_flow_returns_result(self) -> None:
        @traced_flow()
        async def my_flow(x: int) -> int:
            return x * 2

        result = await my_flow(5)
        assert result == 10

    async def test_traced_flow_custom_name(self) -> None:
        @traced_flow(name="custom.flow")
        async def another_flow() -> str:
            return "ok"

        result = await another_flow()
        assert result == "ok"

    async def test_traced_flow_propagates_exception(self) -> None:
        @traced_flow()
        async def failing_flow() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await failing_flow()


class TestTracedTask:
    async def test_traced_task_returns_result(self) -> None:
        @traced_task()
        async def my_task(x: int) -> int:
            return x + 1

        result = await my_task(10)
        assert result == 11

    async def test_traced_task_custom_name(self) -> None:
        @traced_task(name="custom.task")
        async def named_task() -> str:
            return "done"

        assert await named_task() == "done"

    async def test_traced_task_propagates_exception(self) -> None:
        @traced_task()
        async def failing_task() -> None:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            await failing_task()


class TestTraceContextPropagation:
    def test_inject_returns_dict(self) -> None:
        carrier = inject_trace_context()
        assert isinstance(carrier, dict)

    def test_extract_returns_context(self) -> None:
        ctx = extract_trace_context({"traceparent": "00-abc-def-01"})
        # Should not raise
        assert ctx is not None

    def test_attach_none_returns_none(self) -> None:
        assert attach_trace_context(None) is None

    def test_detach_none_is_safe(self) -> None:
        detach_trace_context(None)  # Should not raise


# ---------------------------------------------------------------------------
# init_prefect
# ---------------------------------------------------------------------------


class TestInitPrefect:
    async def test_init_concurrency_limits(self) -> None:
        mock_client = AsyncMock()
        # Simulate existing limit
        mock_existing = MagicMock()
        mock_existing.concurrency_limit = 5
        mock_client.read_concurrency_limit_by_tag.return_value = mock_existing

        with patch(
            "contextmine_worker.init_prefect.get_client",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            from contextmine_worker.init_prefect import init_concurrency_limits

            await init_concurrency_limits()
            # Should have been called for each tag
            assert mock_client.read_concurrency_limit_by_tag.call_count > 0

    async def test_init_concurrency_creates_new_when_not_found(self) -> None:
        mock_client = AsyncMock()
        mock_client.read_concurrency_limit_by_tag.side_effect = Exception("not found")
        mock_client.create_concurrency_limit.return_value = None

        with patch(
            "contextmine_worker.init_prefect.get_client",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            from contextmine_worker.init_prefect import init_concurrency_limits

            await init_concurrency_limits()
            assert mock_client.create_concurrency_limit.call_count > 0


# ---------------------------------------------------------------------------
# main.py scheduler loop -- uses asyncio.create_task so run under asyncio only
# ---------------------------------------------------------------------------


class TestSchedulerLoop:
    def test_scheduler_loop_runs_and_can_be_cancelled(self) -> None:
        """Use asyncio.run directly to avoid trio parametrization
        since the scheduler uses asyncio.create_task."""

        async def _inner() -> None:
            with patch("contextmine_worker.main.sync_due_sources") as mock_flow:
                mock_flow.fn = AsyncMock(return_value={"synced": 0})

                from contextmine_worker.main import scheduler_loop

                task = asyncio.create_task(scheduler_loop())
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(_inner())


# ---------------------------------------------------------------------------
# chunking extras
# ---------------------------------------------------------------------------


class TestChunkingExtras:
    def test_compute_chunk_hash_deterministic(self) -> None:
        h1 = compute_chunk_hash("hello world")
        h2 = compute_chunk_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_is_markdown_file(self) -> None:
        assert is_markdown_file("README.md") is True
        assert is_markdown_file("notes.mdx") is True
        assert is_markdown_file("docs.rst") is True
        assert is_markdown_file("file.txt") is True
        assert is_markdown_file("file.adoc") is True
        assert is_markdown_file("main.py") is False
        assert is_markdown_file("app.js") is False

    def test_chunk_document_empty_content(self) -> None:
        assert chunk_document("") == []
        assert chunk_document("   ") == []

    def test_chunk_document_returns_chunk_results(self) -> None:
        content = "Hello world " * 200
        results = chunk_document(content, file_path=None, chunk_size=100, chunk_overlap=10)
        assert len(results) > 0
        assert all(isinstance(r, ChunkResult) for r in results)

    def test_split_code_file_python(self) -> None:
        code = "def foo():\n    pass\n\ndef bar():\n    return 1\n" * 50
        chunks = split_code_file(code, "main.py", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0

    def test_split_code_file_unknown_extension(self) -> None:
        code = "some content " * 50
        chunks = split_code_file(code, "file.xyz", chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 0

    def test_chunk_document_code_file(self) -> None:
        code = "def foo():\n    pass\n" * 100
        results = chunk_document(code, file_path="main.py", chunk_size=100, chunk_overlap=10)
        assert len(results) > 0

    def test_chunk_with_headers(self) -> None:
        content = "# Section 1\nContent A\n## Section 1.1\nContent B\n# Section 2\nContent C\n"
        results = chunk_with_headers(content, chunk_size=5000)
        assert len(results) > 0

    def test_split_markdown_with_code_blocks(self) -> None:
        content = (
            "Some intro text.\n\n```python\ndef hello():\n    print('hi')\n```\n\nMore text here.\n"
        )
        chunks = split_markdown_preserving_code_fences(content, chunk_size=5000)
        assert len(chunks) >= 1
        # Code block should remain intact
        full = "\n".join(chunks)
        assert "def hello()" in full
