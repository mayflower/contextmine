"""Tests for collections management and access control."""

from typing import Any
from unittest.mock import patch

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
class TestCollectionRoutes:
    """Tests for collection route handlers."""

    async def test_create_collection_requires_auth(self, client: AsyncClient) -> None:
        """Test that creating a collection requires authentication."""
        response = await client.post(
            "/api/collections",
            json={"name": "Test Collection", "slug": "test", "visibility": "private"},
        )
        assert response.status_code == 401

    async def test_list_collections_requires_auth(self, client: AsyncClient) -> None:
        """Test that listing collections requires authentication."""
        response = await client.get("/api/collections")
        assert response.status_code == 401

    async def test_list_members_requires_auth(self, client: AsyncClient) -> None:
        """Test that listing members requires authentication."""
        response = await client.get("/api/collections/some-id/members")
        assert response.status_code == 401

    async def test_list_invites_requires_auth(self, client: AsyncClient) -> None:
        """Test that listing invites requires authentication."""
        response = await client.get("/api/collections/some-id/invites")
        assert response.status_code == 401

    async def test_share_collection_requires_auth(self, client: AsyncClient) -> None:
        """Test that sharing a collection requires authentication."""
        response = await client.post(
            "/api/collections/some-id/share",
            json={"github_login": "testuser"},
        )
        assert response.status_code == 401

    async def test_unshare_collection_requires_auth(self, client: AsyncClient) -> None:
        """Test that unsharing a collection requires authentication."""
        response = await client.delete("/api/collections/some-id/share/testuser")
        assert response.status_code == 401


@pytest.mark.anyio
class TestCollectionVisibility:
    """Tests for collection visibility and access control."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_invalid_visibility_rejected(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that invalid visibility values are rejected."""
        import uuid

        # Mock authenticated session
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        response = await client.post(
            "/api/collections",
            json={"name": "Test", "slug": "test", "visibility": "invalid"},
        )
        assert response.status_code == 400
        assert "Invalid visibility" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_slug_uniqueness_enforced(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that duplicate slugs are rejected."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        user_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock existing collection with same slug
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()  # Slug exists
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.post(
            "/api/collections",
            json={"name": "Test", "slug": "existing-slug", "visibility": "private"},
        )
        assert response.status_code == 400
        assert "Slug already exists" in response.json()["detail"]


@pytest.mark.anyio
class TestCollectionAccess:
    """Tests for collection access control."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_invalid_collection_id_rejected(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that invalid collection IDs are rejected."""
        import uuid

        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        response = await client.get("/api/collections/not-a-uuid/members")
        assert response.status_code == 400
        assert "Invalid collection ID" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_nonexistent_collection_returns_404(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that nonexistent collection returns 404."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock empty result (collection not found)
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(f"/api/collections/{collection_id}/members")
        assert response.status_code == 404
        assert "Collection not found" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_share_requires_owner(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that only owner can share a collection."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        from contextmine_core import CollectionVisibility

        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()  # Different user
        collection_id = uuid.uuid4()

        mock_get_session.return_value = {"user_id": str(user_id)}

        # Mock collection with different owner
        mock_collection = MagicMock()
        mock_collection.id = collection_id
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
            f"/api/collections/{collection_id}/share",
            json={"github_login": "otheruser"},
        )
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]


@pytest.mark.anyio
class TestInviteAutoAccept:
    """Tests for invite auto-acceptance on login."""

    @patch("app.routes.auth.get_db_session")
    @patch("app.routes.auth.get_session")
    @patch("app.routes.auth.get_github_user")
    @patch("app.routes.auth.exchange_code_for_token")
    async def test_invites_accepted_on_login(
        self,
        mock_exchange: Any,
        mock_get_github_user: Any,
        mock_get_session: Any,
        mock_get_db_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that pending invites are auto-accepted on login."""
        import uuid
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        # Setup mocks
        mock_exchange.return_value = "fake-token"
        mock_get_github_user.return_value = {
            "id": 12345,
            "login": "testuser",
            "name": "Test User",
            "avatar_url": "https://github.com/test.png",
        }

        session_data: dict[str, str] = {"oauth_state": "valid-state"}
        mock_get_session.return_value = session_data

        # Mock user
        mock_user = MagicMock()
        mock_user.id = uuid.uuid4()
        mock_user.github_login = "testuser"

        # Mock pending invite
        mock_invite = MagicMock()
        mock_invite.collection_id = uuid.uuid4()
        mock_invite.github_login = "testuser"

        mock_db = MagicMock()

        # First call returns None (new user), then user, then invites
        async def mock_execute(query: Any) -> MagicMock:
            result = MagicMock()
            if "users" in str(query) and "github_user_id" in str(query):
                result.scalar_one_or_none.return_value = None  # New user
            elif "collection_invites" in str(query):
                result.scalars.return_value.all.return_value = [mock_invite]
            elif "collection_members" in str(query):
                result.scalar_one_or_none.return_value = None  # Not a member yet
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.delete = AsyncMock()

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(
            "/api/auth/callback",
            params={"code": "test-code", "state": "valid-state"},
            follow_redirects=False,
        )

        # Should redirect to frontend after successful login
        assert response.status_code == 302

        # Verify invite was processed (delete called for invite)
        mock_db.delete.assert_called()
