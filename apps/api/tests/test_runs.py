"""Tests for sync runs management."""

from typing import Any
from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
class TestRunsRoutes:
    """Tests for runs route handlers."""

    async def test_list_runs_requires_auth(self, client: AsyncClient) -> None:
        """Test that listing runs requires authentication."""
        response = await client.get("/api/runs?source_id=some-id")
        assert response.status_code == 401

    @patch("app.routes.runs.get_session")
    @patch("app.routes.runs.get_db_session")
    async def test_list_runs_invalid_source_id(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that invalid source ID is rejected."""
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        response = await client.get("/api/runs?source_id=invalid-uuid")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.runs.get_session")
    @patch("app.routes.runs.get_db_session")
    async def test_list_runs_source_not_found(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that nonexistent source returns 404."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(f"/api/runs?source_id={uuid.uuid4()}")
        assert response.status_code == 404
        assert "Source not found" in response.json()["detail"]


@pytest.mark.anyio
class TestRunsAccess:
    """Tests for runs access control."""

    @patch("app.routes.runs.get_session")
    @patch("app.routes.runs.get_db_session")
    async def test_list_runs_requires_collection_access(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that only users with collection access can list runs."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from contextmine_core import CollectionVisibility

        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()  # Different user
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()

        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock source
        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        # Mock private collection NOT owned by user
        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = owner_id
        mock_collection.visibility = CollectionVisibility.PRIVATE

        mock_db = MagicMock()

        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            else:
                # Membership check - no membership
                result.scalar_one_or_none.return_value = None
            call_count += 1
            return result

        mock_db.execute = mock_execute

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(f"/api/runs?source_id={source_id}")
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    @patch("app.routes.runs.get_session")
    @patch("app.routes.runs.get_db_session")
    async def test_list_runs_allows_owner(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that collection owner can list runs."""
        import uuid
        from contextlib import asynccontextmanager
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from contextmine_core import CollectionVisibility, SyncRunStatus

        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        run_id = uuid.uuid4()

        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock source
        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        # Mock collection owned by user
        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id
        mock_collection.visibility = CollectionVisibility.PRIVATE

        # Mock run
        mock_run = MagicMock()
        mock_run.id = run_id
        mock_run.source_id = source_id
        mock_run.started_at = datetime.now(UTC)
        mock_run.finished_at = datetime.now(UTC)
        mock_run.status = SyncRunStatus.SUCCESS
        mock_run.stats = {"documents_synced": 0}
        mock_run.error = None

        mock_db = MagicMock()

        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            else:
                result.scalars.return_value.all.return_value = [mock_run]
            call_count += 1
            return result

        mock_db.execute = mock_execute

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(f"/api/runs?source_id={source_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == str(run_id)
        assert data[0]["status"] == "success"

    @patch("app.routes.runs.get_session")
    @patch("app.routes.runs.get_db_session")
    async def test_list_runs_allows_global_collection(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that any user can list runs for global collections."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        from contextmine_core import CollectionVisibility

        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()  # Different user
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()

        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock source
        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        # Mock global collection (not owned by user but visible to all)
        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = owner_id
        mock_collection.visibility = CollectionVisibility.GLOBAL

        mock_db = MagicMock()

        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_source
            elif call_count == 1:
                result.scalar_one.return_value = mock_collection
            else:
                result.scalars.return_value.all.return_value = []
            call_count += 1
            return result

        mock_db.execute = mock_execute

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(f"/api/runs?source_id={source_id}")
        assert response.status_code == 200
        assert response.json() == []
