"""Tests for CI coverage ingest routes."""

from __future__ import annotations

import hashlib
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core import SourceType
from httpx import AsyncClient


@pytest.mark.anyio
class TestCoverageIngestRoutes:
    """Route-level tests for coverage ingest endpoints."""

    async def test_coverage_ingest_requires_token(self, client: AsyncClient) -> None:
        source_id = str(uuid.uuid4())
        response = await client.post(
            f"/api/sources/{source_id}/metrics/coverage-ingest",
            data={"commit_sha": "a" * 40},
            files=[("reports", ("coverage.xml", b"<coverage/>", "application/xml"))],
        )
        assert response.status_code == 401
        assert "INGEST_AUTH_INVALID" in response.json()["detail"]

    async def test_coverage_ingest_invalid_sha_rejected(self, client: AsyncClient) -> None:
        source_id = str(uuid.uuid4())
        response = await client.post(
            f"/api/sources/{source_id}/metrics/coverage-ingest",
            headers={"X-ContextMine-Ingest-Token": "token"},
            data={"commit_sha": "not-a-sha"},
            files=[("reports", ("coverage.xml", b"<coverage/>", "application/xml"))],
        )
        assert response.status_code == 400
        assert "commit_sha" in response.json()["detail"]

    @patch("app.routes.metrics_ingest._trigger_prefect_ingest_flow")
    @patch("app.routes.metrics_ingest.get_db_session")
    async def test_coverage_ingest_queues_job_and_triggers_prefect(
        self,
        mock_get_db_session: Any,
        mock_trigger_prefect: Any,
        client: AsyncClient,
    ) -> None:
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        token = "cmi_token_123456"
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        commit_sha = "a" * 40

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id
        mock_source.type = SourceType.GITHUB
        mock_source.cursor = commit_sha

        mock_token_row = MagicMock()
        mock_token_row.token_hash = token_hash
        mock_token_row.last_used_at = None

        async def execute_side_effect(query: Any) -> MagicMock:
            result = MagicMock()
            text = str(query)
            if "FROM sources" in text:
                result.scalar_one_or_none.return_value = mock_source
            elif "FROM source_ingest_tokens" in text:
                result.scalar_one_or_none.return_value = mock_token_row
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=execute_side_effect)
        mock_db.flush = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()

        @asynccontextmanager
        async def db_session():
            yield mock_db

        mock_get_db_session.return_value = db_session()
        mock_trigger_prefect.return_value = {"id": "flow-run-1"}

        response = await client.post(
            f"/api/sources/{source_id}/metrics/coverage-ingest",
            headers={"X-ContextMine-Ingest-Token": token},
            data={"commit_sha": commit_sha, "provider": "github_actions"},
            files=[("reports", ("coverage.xml", b"<coverage/>", "application/xml"))],
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "queued"
        assert payload["flow_run_id"] == "flow-run-1"

    async def test_job_status_requires_auth(self, client: AsyncClient) -> None:
        source_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        response = await client.get(f"/api/sources/{source_id}/metrics/coverage-ingest/{job_id}")
        assert response.status_code == 401
