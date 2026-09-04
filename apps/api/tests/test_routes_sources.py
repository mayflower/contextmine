"""Tests for source management routes.

Extends the existing test_sources.py with deeper coverage of
success paths, update/sync flows, deploy key operations, and
ingest token management.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


def _mock_db_session(mock_db: MagicMock):
    """Create an async context manager that yields the mock DB."""

    mock_db.commit = AsyncMock()

    @asynccontextmanager
    async def session():
        yield mock_db

    return session()


# ---------------------------------------------------------------------------
# Pure helper function tests
# ---------------------------------------------------------------------------


class TestValidateGithubUrl:
    """Tests for validate_github_url helper."""

    def test_valid_url(self) -> None:
        from app.routes.sources import validate_github_url

        result = validate_github_url("https://github.com/owner/repo")
        assert result["owner"] == "owner"
        assert result["repo"] == "repo"
        # branch is left unset so the worker resolves the repo's actual default branch
        assert result["branch"] is None

    def test_valid_url_with_git_suffix(self) -> None:
        from app.routes.sources import validate_github_url

        result = validate_github_url("https://github.com/owner/repo.git")
        assert result["owner"] == "owner"
        assert result["repo"] == "repo"

    def test_invalid_url_raises_400(self) -> None:
        from app.routes.sources import validate_github_url
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_github_url("not-a-url")
        assert exc_info.value.status_code == 400
        assert "Invalid GitHub URL" in exc_info.value.detail

    def test_non_github_url_raises_400(self) -> None:
        from app.routes.sources import validate_github_url
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_github_url("https://gitlab.com/owner/repo")
        assert exc_info.value.status_code == 400

    def test_github_url_missing_repo(self) -> None:
        from app.routes.sources import validate_github_url
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_github_url("https://github.com/owner")
        assert exc_info.value.status_code == 400


class TestValidateWebUrl:
    """Tests for validate_web_url helper."""

    def test_valid_https_url(self) -> None:
        from app.routes.sources import validate_web_url

        result = validate_web_url("https://docs.example.com/guide/intro")
        assert "start_url" in result
        assert "base_url" in result
        assert result["start_url"] == "https://docs.example.com/guide/intro"

    def test_valid_http_url(self) -> None:
        from app.routes.sources import validate_web_url

        result = validate_web_url("http://example.com/docs/")
        assert result["start_url"] == "http://example.com/docs/"

    def test_non_http_url_raises_400(self) -> None:
        from app.routes.sources import validate_web_url
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_web_url("ftp://example.com")
        assert exc_info.value.status_code == 400
        assert "Invalid URL" in exc_info.value.detail


class TestMarkCoveragePatternsDeprecated:
    """Tests for mark_coverage_patterns_deprecated helper."""

    def test_adds_deprecated_marker(self) -> None:
        from app.routes.sources import mark_coverage_patterns_deprecated

        result = mark_coverage_patterns_deprecated({}, ["*.xml"])
        assert result["metrics"]["deprecated"] is True
        assert result["metrics"]["deprecated_field"] == "coverage_report_patterns"
        assert result["metrics"]["coverage_report_patterns_ignored"] == ["*.xml"]

    def test_preserves_existing_config(self) -> None:
        from app.routes.sources import mark_coverage_patterns_deprecated

        result = mark_coverage_patterns_deprecated({"owner": "test"}, None)
        assert result["owner"] == "test"
        assert result["metrics"]["deprecated"] is True

    def test_none_patterns(self) -> None:
        from app.routes.sources import mark_coverage_patterns_deprecated

        result = mark_coverage_patterns_deprecated({}, None)
        assert "coverage_report_patterns_ignored" not in result["metrics"]


class TestHashIngestToken:
    """Tests for hash_ingest_token helper."""

    def test_deterministic(self) -> None:
        from app.routes.sources import hash_ingest_token

        assert hash_ingest_token("test") == hash_ingest_token("test")

    def test_different_inputs(self) -> None:
        from app.routes.sources import hash_ingest_token

        assert hash_ingest_token("abc") != hash_ingest_token("def")

    def test_returns_hex(self) -> None:
        from app.routes.sources import hash_ingest_token

        result = hash_ingest_token("test_token")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestMakeTokenPreview:
    """Tests for make_token_preview helper."""

    def test_short_token(self) -> None:
        from app.routes.sources import make_token_preview

        assert make_token_preview("short") == "********"

    def test_long_token(self) -> None:
        from app.routes.sources import make_token_preview

        result = make_token_preview("cmi_abcdefghijklmnop")
        assert result.startswith("cmi_ab")
        assert result.endswith("mnop")
        assert "..." in result


# ---------------------------------------------------------------------------
# Source creation (success paths)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestCreateSource:
    """Tests for POST /api/collections/{collection_id}/sources."""

    async def test_create_source_requires_body_fields(self, client: AsyncClient) -> None:
        """Test that missing required body fields are rejected."""
        # Missing 'type' and 'url'
        response = await client.post(
            f"/api/collections/{uuid.uuid4()}/sources",
            json={},
        )
        assert response.status_code == 422

        # Missing 'url'
        response = await client.post(
            f"/api/collections/{uuid.uuid4()}/sources",
            json={"type": "github"},
        )
        assert response.status_code == 422

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_create_source_invalid_collection_id(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        response = await client.post(
            "/api/collections/not-a-uuid/sources",
            json={"type": "github", "url": "https://github.com/owner/repo"},
        )
        assert response.status_code == 400
        assert "Invalid collection ID" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Source update
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestUpdateSource:
    """Tests for PATCH /api/sources/{source_id}."""

    async def test_update_requires_auth(self, client: AsyncClient) -> None:
        response = await client.patch(
            f"/api/sources/{uuid.uuid4()}",
            json={"enabled": False},
        )
        assert response.status_code == 401

    @patch("app.routes.sources.get_session")
    async def test_update_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.patch(
            "/api/sources/not-a-uuid",
            json={"enabled": False},
        )
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_update_schedule_too_low_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/sources/{source_id}",
            json={"schedule_interval_minutes": 0},
        )
        assert response.status_code == 400
        assert "at least 1 minute" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_update_max_pages_non_web_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        from contextmine_core import SourceType

        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id
        mock_source.type = SourceType.GITHUB

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/sources/{source_id}",
            json={"max_pages": 50},
        )
        assert response.status_code == 400
        assert "only supported for web sources" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_update_max_pages_out_of_range_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        from contextmine_core import SourceType

        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id
        mock_source.type = SourceType.WEB

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/sources/{source_id}",
            json={"max_pages": 5000},
        )
        assert response.status_code == 400
        assert "max_pages must be between 1 and 1000" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Sync now
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestSyncNow:
    """Tests for POST /api/sources/{source_id}/sync-now."""

    @patch("app.routes.sources.get_session")
    async def test_sync_now_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post("/api/sources/not-a-uuid/sync-now")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_sync_now_source_not_found(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.post(f"/api/sources/{uuid.uuid4()}/sync-now")
        assert response.status_code == 404
        assert "Source not found" in response.json()["detail"]

    async def test_sync_now_starts_visible_prefect_run(self, client: AsyncClient) -> None:
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        sync_run_id = uuid.uuid4()
        flow_run_id = uuid.uuid4()
        source = MagicMock(
            id=source_id,
            collection_id=collection_id,
            url="https://github.com/example/repository",
            schedule_interval_minutes=60,
        )
        scheduled_run = None
        mock_db = MagicMock()

        def add(row: Any) -> None:
            nonlocal scheduled_run
            row.id = sync_run_id
            scheduled_run = row

        mock_db.add = add
        mock_db.flush = AsyncMock()

        async def execute(_statement: Any) -> MagicMock:
            result = MagicMock()
            if mock_db.execute.await_count == 1:
                result.scalar_one_or_none.return_value = source
            elif mock_db.execute.await_count == 2:
                result.scalar_one_or_none.return_value = None
            else:
                result.scalar_one.return_value = scheduled_run
            return result

        mock_db.execute = AsyncMock(side_effect=execute)

        with (
            patch("app.routes.sources.get_session", return_value={"user_id": str(uuid.uuid4())}),
            patch(
                "app.routes.sources.get_db_session",
                side_effect=lambda: _mock_db_session(mock_db),
            ),
            patch("app.routes.sources._get_collection_with_access", new=AsyncMock()),
            patch(
                "app.routes.prefect.start_source_sync",
                new=AsyncMock(return_value=str(flow_run_id)),
            ) as start_sync,
        ):
            response = await client.post(f"/api/sources/{source_id}/sync-now")

        assert response.status_code == 200
        assert response.json() == {
            "status": "sync_scheduled",
            "sync_run_id": str(sync_run_id),
            "flow_run_id": str(flow_run_id),
        }
        start_sync.assert_awaited_once_with(str(source_id), source.url, str(sync_run_id))
        assert scheduled_run.flow_run_id == str(flow_run_id)

    async def test_sync_now_reuses_active_prefect_run(self, client: AsyncClient) -> None:
        source_id = uuid.uuid4()
        flow_run_id = str(uuid.uuid4())
        source = MagicMock(
            id=source_id,
            collection_id=uuid.uuid4(),
            url="https://github.com/example/repository",
        )
        active_run = MagicMock(id=uuid.uuid4(), flow_run_id=flow_run_id)
        source_result = MagicMock()
        source_result.scalar_one_or_none.return_value = source
        active_result = MagicMock()
        active_result.scalar_one_or_none.return_value = active_run
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=[source_result, active_result])

        with (
            patch("app.routes.sources.get_session", return_value={"user_id": str(uuid.uuid4())}),
            patch(
                "app.routes.sources.get_db_session",
                side_effect=lambda: _mock_db_session(mock_db),
            ),
            patch("app.routes.sources._get_collection_with_access", new=AsyncMock()),
            patch("app.routes.prefect.start_source_sync", new=AsyncMock()) as start_sync,
            # Prefect reports nothing terminal, i.e. the flow is still in flight.
            patch("app.routes.prefect.terminal_flow_run_state", AsyncMock(return_value=None)),
        ):
            response = await client.post(f"/api/sources/{source_id}/sync-now")

        assert response.status_code == 200
        assert response.json()["status"] == "already_scheduled"
        assert response.json()["flow_run_id"] == flow_run_id
        start_sync.assert_not_awaited()

    async def test_sync_now_replaces_a_run_whose_flow_already_crashed(
        self, client: AsyncClient
    ) -> None:
        """A crashed flow must not block the source from syncing again."""
        from contextmine_core import SyncRunStatus

        source_id = uuid.uuid4()
        dead_flow_run_id = str(uuid.uuid4())
        new_flow_run_id = str(uuid.uuid4())
        source = MagicMock(
            id=source_id,
            collection_id=uuid.uuid4(),
            url="https://github.com/example/repository",
            schedule_interval_minutes=60,
        )
        stale_run = MagicMock(
            id=uuid.uuid4(),
            flow_run_id=dead_flow_run_id,
            status=SyncRunStatus.SCHEDULED,
            finished_at=None,
            error=None,
        )
        replacement_id = uuid.uuid4()

        def add(row: Any) -> None:
            row.id = replacement_id

        mock_db = MagicMock()
        mock_db.add = add
        mock_db.flush = AsyncMock()
        source_result = MagicMock()
        source_result.scalar_one_or_none.return_value = source
        active_result = MagicMock()
        active_result.scalar_one_or_none.return_value = stale_run
        run_result = MagicMock()
        run_result.scalar_one_or_none.return_value = stale_run
        mock_db.execute = AsyncMock(
            side_effect=[source_result, active_result, run_result, run_result, run_result]
        )

        with (
            patch("app.routes.sources.get_session", return_value={"user_id": str(uuid.uuid4())}),
            patch(
                "app.routes.sources.get_db_session",
                side_effect=lambda: _mock_db_session(mock_db),
            ),
            patch("app.routes.sources._get_collection_with_access", new=AsyncMock()),
            patch(
                "app.routes.prefect.terminal_flow_run_state",
                AsyncMock(return_value="CRASHED"),
            ),
            patch(
                "app.routes.prefect.start_source_sync",
                AsyncMock(return_value=new_flow_run_id),
            ) as start_sync,
        ):
            response = await client.post(f"/api/sources/{source_id}/sync-now")

        assert response.status_code == 200
        assert response.json()["status"] == "sync_scheduled"
        assert response.json()["flow_run_id"] == new_flow_run_id
        start_sync.assert_awaited_once()
        assert stale_run.status is SyncRunStatus.FAILED

    async def test_sync_now_prefect_outage_finishes_business_run(self, client: AsyncClient) -> None:
        source_id = uuid.uuid4()
        sync_run_id = uuid.uuid4()
        source = MagicMock(
            id=source_id,
            collection_id=uuid.uuid4(),
            url="https://github.com/example/repository",
            schedule_interval_minutes=60,
        )
        scheduled_run = None
        mock_db = MagicMock()

        def add(row: Any) -> None:
            nonlocal scheduled_run
            row.id = sync_run_id
            scheduled_run = row

        mock_db.add = add
        mock_db.flush = AsyncMock()

        async def execute(_statement: Any) -> MagicMock:
            result = MagicMock()
            if mock_db.execute.await_count == 1:
                result.scalar_one_or_none.return_value = source
            elif mock_db.execute.await_count == 2:
                result.scalar_one_or_none.return_value = None
            else:
                result.scalar_one.return_value = scheduled_run
            return result

        mock_db.execute = AsyncMock(side_effect=execute)

        with (
            patch("app.routes.sources.get_session", return_value={"user_id": str(uuid.uuid4())}),
            patch(
                "app.routes.sources.get_db_session",
                side_effect=lambda: _mock_db_session(mock_db),
            ),
            patch("app.routes.sources._get_collection_with_access", new=AsyncMock()),
            patch(
                "app.routes.prefect.start_source_sync",
                new=AsyncMock(side_effect=RuntimeError("prefect unavailable")),
            ),
        ):
            response = await client.post(f"/api/sources/{source_id}/sync-now")

        assert response.status_code == 502
        assert scheduled_run.status.value == "failed"
        assert scheduled_run.finished_at is not None
        assert scheduled_run.error == "Prefect scheduling failed: prefect unavailable"


# ---------------------------------------------------------------------------
# Deploy key routes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDeployKeyRoutes:
    """Tests for deploy key management routes."""

    async def test_get_deploy_key_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/sources/{uuid.uuid4()}/deploy-key")
        assert response.status_code == 401

    async def test_set_deploy_key_requires_auth(self, client: AsyncClient) -> None:
        response = await client.put(
            f"/api/sources/{uuid.uuid4()}/deploy-key",
            json={"private_key": "fake-key"},
        )
        assert response.status_code == 401

    async def test_delete_deploy_key_requires_auth(self, client: AsyncClient) -> None:
        response = await client.delete(f"/api/sources/{uuid.uuid4()}/deploy-key")
        assert response.status_code == 401

    @patch("app.routes.sources.get_session")
    async def test_get_deploy_key_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/sources/bad-id/deploy-key")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    async def test_set_deploy_key_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.put(
            "/api/sources/bad-id/deploy-key",
            json={"private_key": "fake"},
        )
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.validate_ssh_private_key")
    @patch("app.routes.sources.get_session")
    async def test_set_deploy_key_invalid_ssh_key(
        self,
        mock_get_session: Any,
        mock_validate: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        mock_validate.return_value = False

        response = await client.put(
            f"/api/sources/{uuid.uuid4()}/deploy-key",
            json={"private_key": "not-a-valid-key"},
        )
        assert response.status_code == 400
        assert "Invalid SSH private key" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_get_deploy_key_source_not_found(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{uuid.uuid4()}/deploy-key")
        assert response.status_code == 404
        assert "Source not found" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_get_deploy_key_not_owner_returns_403(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = owner_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/deploy-key")
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_get_deploy_key_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id
        mock_source.deploy_key_fingerprint = "SHA256:abc123"
        mock_source.deploy_key_encrypted = b"encrypted"

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/deploy-key")
        assert response.status_code == 200
        data = response.json()
        assert data["has_key"] is True
        assert data["fingerprint"] == "SHA256:abc123"

    @patch("app.routes.sources.get_session")
    async def test_delete_deploy_key_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.delete("/api/sources/bad-id/deploy-key")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_delete_deploy_key_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.delete(f"/api/sources/{source_id}/deploy-key")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# Ingest token routes
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestIngestTokenRoutes:
    """Tests for coverage ingest token routes."""

    async def test_rotate_token_requires_auth(self, client: AsyncClient) -> None:
        response = await client.post(
            f"/api/sources/{uuid.uuid4()}/metrics/coverage-ingest-token/rotate"
        )
        assert response.status_code == 401

    async def test_get_token_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/sources/{uuid.uuid4()}/metrics/coverage-ingest-token")
        assert response.status_code == 401

    @patch("app.routes.sources.get_session")
    async def test_rotate_token_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.post("/api/sources/bad-id/metrics/coverage-ingest-token/rotate")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    async def test_get_token_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/sources/bad-id/metrics/coverage-ingest-token")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_get_token_no_token_exists(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            elif call_count == 2:
                result.scalar_one_or_none.return_value = None  # No token
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/metrics/coverage-ingest-token")
        assert response.status_code == 200
        data = response.json()
        assert data["has_token"] is False
        assert data["token_preview"] is None

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_get_token_with_existing_token(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_token = MagicMock()
        mock_token.token_preview = "cmi_ab...wxyz"
        mock_token.created_at = now
        mock_token.rotated_at = now
        mock_token.last_used_at = None

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            elif call_count == 2:
                result.scalar_one_or_none.return_value = mock_token
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/metrics/coverage-ingest-token")
        assert response.status_code == 200
        data = response.json()
        assert data["has_token"] is True
        assert data["token_preview"] == "cmi_ab...wxyz"


# ---------------------------------------------------------------------------
# List sources
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestListSources:
    """Tests for GET /api/collections/{collection_id}/sources."""

    @patch("app.routes.sources.get_session")
    async def test_list_sources_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/collections/not-a-uuid/sources")
        assert response.status_code == 400
        assert "Invalid collection ID" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_list_sources_collection_not_found(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/collections/{uuid.uuid4()}/sources")
        assert response.status_code == 404
        assert "Collection not found" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_delete_source_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.delete = AsyncMock()
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.delete(f"/api/sources/{source_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestTerminalFlowRunState:
    """Tests for reading a flow run's terminal state from Prefect."""

    @staticmethod
    def _client_returning(flow_run: Any) -> Any:
        @asynccontextmanager
        async def factory():
            client = MagicMock()
            client.read_flow_run = AsyncMock(return_value=flow_run)
            yield client

        return factory

    @pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "CRASHED", "CANCELLED"])
    async def test_reports_states_a_run_never_leaves(self, state: str) -> None:
        from app.routes import prefect as prefect_routes

        flow_run = MagicMock(state_type=MagicMock(value=state))
        with patch.object(prefect_routes, "get_client", self._client_returning(flow_run)):
            assert await prefect_routes.terminal_flow_run_state(str(uuid.uuid4())) == state

    @pytest.mark.parametrize("state", ["RUNNING", "SCHEDULED", "PENDING", "PAUSED"])
    async def test_reports_nothing_while_the_run_may_still_finish(self, state: str) -> None:
        from app.routes import prefect as prefect_routes

        flow_run = MagicMock(state_type=MagicMock(value=state))
        with patch.object(prefect_routes, "get_client", self._client_returning(flow_run)):
            assert await prefect_routes.terminal_flow_run_state(str(uuid.uuid4())) is None

    async def test_treats_an_unknown_run_as_gone(self) -> None:
        from app.routes import prefect as prefect_routes
        from prefect.exceptions import ObjectNotFound

        @asynccontextmanager
        async def factory():
            client = MagicMock()
            client.read_flow_run = AsyncMock(
                side_effect=ObjectNotFound(http_exc=Exception("404 flow run not found"))
            )
            yield client

        with patch.object(prefect_routes, "get_client", factory):
            result = await prefect_routes.terminal_flow_run_state(str(uuid.uuid4()))
        assert result == prefect_routes.FLOW_RUN_MISSING

    async def test_treats_a_malformed_id_as_gone(self) -> None:
        from app.routes import prefect as prefect_routes

        assert await prefect_routes.terminal_flow_run_state("not-a-uuid") == (
            prefect_routes.FLOW_RUN_MISSING
        )

    async def test_stays_silent_when_prefect_is_unreachable(self) -> None:
        """An unreachable server must not look like a finished run."""
        from app.routes import prefect as prefect_routes

        @asynccontextmanager
        async def factory():
            client = MagicMock()
            client.read_flow_run = AsyncMock(side_effect=ConnectionError("boom"))
            yield client

        with patch.object(prefect_routes, "get_client", factory):
            assert await prefect_routes.terminal_flow_run_state(str(uuid.uuid4())) is None


@pytest.mark.anyio
class TestReleaseFinishedSyncRun:
    """A finished flow must not keep its source blocked from syncing again."""

    @staticmethod
    def _run() -> MagicMock:
        from contextmine_core import SyncRunStatus

        return MagicMock(
            id=uuid.uuid4(),
            status=SyncRunStatus.SCHEDULED,
            finished_at=None,
            error=None,
        )

    async def _release(self, run: MagicMock | None, flow_state: str | None) -> bool:
        from app.routes import sources as sources_routes

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = run
        mock_db.execute = AsyncMock(return_value=mock_result)

        with (
            patch.object(sources_routes, "get_db_session", lambda: _mock_db_session(mock_db)),
            patch(
                "app.routes.prefect.terminal_flow_run_state",
                AsyncMock(return_value=flow_state),
            ),
        ):
            return await sources_routes._release_finished_sync_run(
                run.id if run else uuid.uuid4(), str(uuid.uuid4())
            )

    async def test_marks_a_crashed_run_failed(self) -> None:
        from contextmine_core import SyncRunStatus

        run = self._run()
        assert await self._release(run, "CRASHED") is True
        assert run.status is SyncRunStatus.FAILED
        assert run.finished_at is not None
        assert "RECONCILED_WITH_PREFECT" in run.error

    async def test_marks_a_completed_run_successful_without_an_error(self) -> None:
        from contextmine_core import SyncRunStatus

        run = self._run()
        assert await self._release(run, "COMPLETED") is True
        assert run.status is SyncRunStatus.SUCCESS
        assert run.error is None

    async def test_keeps_waiting_while_the_flow_may_still_finish(self) -> None:
        from contextmine_core import SyncRunStatus

        run = self._run()
        assert await self._release(run, None) is False
        assert run.status is SyncRunStatus.SCHEDULED
        assert run.finished_at is None

    async def test_treats_a_vanished_flow_run_as_failed(self) -> None:
        from app.routes import prefect as prefect_routes
        from contextmine_core import SyncRunStatus

        run = self._run()
        assert await self._release(run, prefect_routes.FLOW_RUN_MISSING) is True
        assert run.status is SyncRunStatus.FAILED
        assert "an unknown run" in run.error

    async def test_does_not_reconcile_a_run_someone_else_already_closed(self) -> None:
        from contextmine_core import SyncRunStatus

        run = self._run()
        run.status = SyncRunStatus.FAILED
        assert await self._release(run, "CRASHED") is False

    async def test_does_nothing_when_the_run_is_gone(self) -> None:
        assert await self._release(None, "CRASHED") is False
