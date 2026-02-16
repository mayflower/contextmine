"""Tests for sources management."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
class TestSourceRoutes:
    """Tests for source route handlers."""

    async def test_create_source_requires_auth(self, client: AsyncClient) -> None:
        """Test that creating a source requires authentication."""
        response = await client.post(
            "/api/collections/some-id/sources",
            json={"type": "github", "url": "https://github.com/owner/repo"},
        )
        assert response.status_code == 401

    async def test_list_sources_requires_auth(self, client: AsyncClient) -> None:
        """Test that listing sources requires authentication."""
        response = await client.get("/api/collections/some-id/sources")
        assert response.status_code == 401

    async def test_delete_source_requires_auth(self, client: AsyncClient) -> None:
        """Test that deleting a source requires authentication."""
        response = await client.delete("/api/sources/some-id")
        assert response.status_code == 401

    async def test_sync_now_requires_auth(self, client: AsyncClient) -> None:
        """Test that sync now requires authentication."""
        response = await client.post("/api/sources/some-id/sync-now")
        assert response.status_code == 401


@pytest.mark.anyio
class TestSourceValidation:
    """Tests for source URL validation."""

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_invalid_source_type_rejected(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that invalid source type is rejected."""
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        response = await client.post(
            "/api/collections/some-id/sources",
            json={"type": "invalid", "url": "https://example.com"},
        )
        assert response.status_code == 400
        assert "Invalid type" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_invalid_github_url_rejected(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that invalid GitHub URL is rejected."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        from contextmine_core import CollectionVisibility

        user_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock collection owned by user
        mock_collection = MagicMock()
        mock_collection.id = uuid.uuid4()
        mock_collection.owner_user_id = user_id
        mock_collection.visibility = CollectionVisibility.PRIVATE

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_collection
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.post(
            f"/api/collections/{mock_collection.id}/sources",
            json={"type": "github", "url": "not-a-valid-url"},
        )
        assert response.status_code == 400
        assert "Invalid GitHub URL" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_invalid_web_url_rejected(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that invalid web URL is rejected."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        from contextmine_core import CollectionVisibility

        user_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock collection owned by user
        mock_collection = MagicMock()
        mock_collection.id = uuid.uuid4()
        mock_collection.owner_user_id = user_id
        mock_collection.visibility = CollectionVisibility.PRIVATE

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_collection
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.post(
            f"/api/collections/{mock_collection.id}/sources",
            json={"type": "web", "url": "not-a-valid-url"},
        )
        assert response.status_code == 400
        assert "Invalid URL" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_coverage_patterns_deprecated_marker_for_web_source(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that coverage_report_patterns is accepted as deprecated no-op for web sources."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        from contextmine_core import SourceType

        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id
        mock_source.type = SourceType.WEB
        mock_source.url = "https://example.com/docs"
        mock_source.config = {"start_url": "https://example.com/docs"}
        mock_source.enabled = True
        mock_source.schedule_interval_minutes = 60
        mock_source.next_run_at = None
        mock_source.last_run_at = None
        mock_source.created_at = datetime.now(UTC)
        mock_source.deploy_key_fingerprint = None

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()

        async def mock_execute(query: Any) -> MagicMock:
            result = MagicMock()
            query_text = str(query)
            if "FROM sources" in query_text:
                result.scalar_one_or_none.return_value = mock_source
            elif "FROM collections" in query_text:
                result.scalar_one.return_value = mock_collection
            else:
                result.scalar.return_value = 0
            return result

        mock_db.execute = AsyncMock(side_effect=mock_execute)
        mock_db.flush = AsyncMock()

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.patch(
            f"/api/sources/{source_id}",
            json={"coverage_report_patterns": ["**/coverage.xml"]},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["config"]["metrics"]["deprecated"] is True
        assert payload["config"]["metrics"]["deprecated_field"] == "coverage_report_patterns"

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_empty_coverage_patterns_deprecated_marker_for_github_source(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that empty coverage_report_patterns is accepted as deprecated no-op."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        from contextmine_core import SourceType

        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id
        mock_source.type = SourceType.GITHUB
        mock_source.url = "https://github.com/org/repo"
        mock_source.config = {"owner": "org", "repo": "repo", "branch": "main"}
        mock_source.enabled = True
        mock_source.schedule_interval_minutes = 60
        mock_source.next_run_at = None
        mock_source.last_run_at = None
        mock_source.created_at = datetime.now(UTC)
        mock_source.deploy_key_fingerprint = None

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()

        async def mock_execute(query: Any) -> MagicMock:
            result = MagicMock()
            query_text = str(query)
            if "FROM sources" in query_text:
                result.scalar_one_or_none.return_value = mock_source
            elif "FROM collections" in query_text:
                result.scalar_one.return_value = mock_collection
            else:
                result.scalar.return_value = 0
            return result

        mock_db.execute = AsyncMock(side_effect=mock_execute)
        mock_db.flush = AsyncMock()

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.patch(
            f"/api/sources/{source_id}",
            json={"coverage_report_patterns": []},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["config"]["metrics"]["deprecated"] is True
        assert payload["config"]["metrics"]["coverage_report_patterns_ignored"] == []


@pytest.mark.anyio
class TestSourceAccess:
    """Tests for source access control."""

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_create_source_requires_owner(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that only collection owner can create sources."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        from contextmine_core import CollectionVisibility

        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()  # Different user
        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock collection NOT owned by user
        mock_collection = MagicMock()
        mock_collection.id = uuid.uuid4()
        mock_collection.owner_user_id = owner_id
        mock_collection.visibility = CollectionVisibility.PRIVATE

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_collection
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.post(
            f"/api/collections/{mock_collection.id}/sources",
            json={"type": "github", "url": "https://github.com/owner/repo"},
        )
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_delete_source_requires_owner(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that only collection owner can delete sources."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock

        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()  # Different user
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()

        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock source
        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        # Mock collection NOT owned by user
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
            else:
                result.scalar_one.return_value = mock_collection
            call_count += 1
            return result

        mock_db.execute = mock_execute

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.delete(f"/api/sources/{source_id}")
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]

    @patch("app.routes.sources.get_session")
    @patch("app.routes.sources.get_db_session")
    async def test_source_not_found_returns_404(
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

        response = await client.delete(f"/api/sources/{uuid.uuid4()}")
        assert response.status_code == 404
        assert "Source not found" in response.json()["detail"]
