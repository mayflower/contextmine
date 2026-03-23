"""Tests for collection CRUD routes with mocked DB sessions.

Extends the existing test_collections.py with deeper coverage of
success paths, update/delete flows, and edge cases.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core import CollectionVisibility
from httpx import AsyncClient


def _mock_db_session(mock_db: MagicMock):
    """Create an async context manager that yields the mock DB."""

    @asynccontextmanager
    async def session():
        yield mock_db

    return session()


# ---------------------------------------------------------------------------
# Collection creation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestCreateCollection:
    """Tests for POST /api/collections."""

    async def test_create_requires_name_slug_visibility(self, client: AsyncClient) -> None:
        """Test that missing fields are rejected by Pydantic validation."""
        # Missing 'slug'
        response = await client.post(
            "/api/collections",
            json={"name": "Test"},
        )
        assert response.status_code == 422

        # Missing 'name'
        response = await client.post(
            "/api/collections",
            json={"slug": "test", "visibility": "private"},
        )
        assert response.status_code == 422

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_user_not_found_returns_401(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = None  # Slug not taken
            elif call_count == 1:
                result.scalar_one_or_none.return_value = None  # User not found
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.post(
            "/api/collections",
            json={"name": "Test", "slug": "test", "visibility": "private"},
        )
        assert response.status_code == 401
        assert "User not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Collection listing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestListCollections:
    """Tests for GET /api/collections."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_list_empty(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get("/api/collections")
        assert response.status_code == 200
        assert response.json() == []

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_list_returns_collections(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.slug = "test"
        mock_collection.name = "Test"
        mock_collection.visibility = CollectionVisibility.GLOBAL
        mock_collection.owner_user_id = user_id
        mock_collection.created_at = now

        mock_owner = MagicMock()
        mock_owner.id = user_id
        mock_owner.github_login = "testuser"

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                # Collection+User join query
                result.all.return_value = [(mock_collection, mock_owner)]
            elif call_count == 1:
                # Member count
                result.scalar.return_value = 2
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get("/api/collections")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["slug"] == "test"
        assert data[0]["is_owner"] is True
        assert data[0]["member_count"] == 2


# ---------------------------------------------------------------------------
# Collection update
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestUpdateCollection:
    """Tests for PATCH /api/collections/{collection_id}."""

    async def test_update_requires_auth(self, client: AsyncClient) -> None:
        response = await client.patch(
            f"/api/collections/{uuid.uuid4()}",
            json={"name": "Updated"},
        )
        assert response.status_code == 401

    @patch("app.routes.collections.get_session")
    async def test_update_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.patch(
            "/api/collections/not-a-uuid",
            json={"name": "Updated"},
        )
        assert response.status_code == 400
        assert "Invalid collection ID" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_update_collection_not_found(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/collections/{collection_id}",
            json={"name": "Updated"},
        )
        assert response.status_code == 404
        assert "Collection not found" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_update_not_owner_returns_403(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = owner_id

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_collection
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/collections/{collection_id}",
            json={"name": "Updated"},
        )
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_update_invalid_visibility_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.slug = "test"
        mock_collection.name = "Test"
        mock_collection.owner_user_id = user_id
        mock_collection.visibility = CollectionVisibility.PRIVATE
        mock_collection.created_at = now

        mock_owner = MagicMock()
        mock_owner.id = user_id
        mock_owner.github_login = "testuser"

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                # Collection lookup
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                # Owner lookup
                result.scalar_one.return_value = mock_owner
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/collections/{collection_id}",
            json={"visibility": "invalid_value"},
        )
        assert response.status_code == 400
        assert "Invalid visibility" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_update_name_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.slug = "test"
        mock_collection.name = "Old Name"
        mock_collection.owner_user_id = user_id
        mock_collection.visibility = CollectionVisibility.PRIVATE
        mock_collection.created_at = now

        mock_owner = MagicMock()
        mock_owner.id = user_id
        mock_owner.github_login = "testuser"

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalar_one.return_value = mock_owner
            elif call_count == 2:
                result.scalar.return_value = 0  # member count
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.patch(
            f"/api/collections/{collection_id}",
            json={"name": "New Name"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_owner"] is True


# ---------------------------------------------------------------------------
# Collection deletion
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDeleteCollection:
    """Tests for DELETE /api/collections/{collection_id}."""

    async def test_delete_requires_auth(self, client: AsyncClient) -> None:
        response = await client.delete(f"/api/collections/{uuid.uuid4()}")
        assert response.status_code == 401

    @patch("app.routes.collections.get_session")
    async def test_delete_invalid_collection_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.delete("/api/collections/not-a-uuid")
        assert response.status_code == 400
        assert "Invalid collection ID" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_delete_not_owner_returns_403(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = owner_id

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_collection
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.delete(f"/api/collections/{collection_id}")
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_delete_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                # Collection lookup
                result.scalar_one_or_none.return_value = mock_collection
            else:
                # Various delete queries + source/doc/chunk lookups
                result.scalars.return_value.all.return_value = []
                result.scalar_one_or_none.return_value = None
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.delete = AsyncMock()
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.delete(f"/api/collections/{collection_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["collection_id"] == str(collection_id)


# ---------------------------------------------------------------------------
# Share / unshare
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestShareCollection:
    """Tests for POST /api/collections/{collection_id}/share."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_share_with_owner_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_target_user = MagicMock()
        mock_target_user.id = user_id  # Same as owner
        mock_target_user.github_login = "owneruser"

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalar_one_or_none.return_value = mock_target_user
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.post(
            f"/api/collections/{collection_id}/share",
            json={"github_login": "owneruser"},
        )
        assert response.status_code == 400
        assert "Cannot share with collection owner" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_share_duplicate_member_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        target_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_target_user = MagicMock()
        mock_target_user.id = target_id
        mock_target_user.github_login = "targetuser"

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalar_one_or_none.return_value = mock_target_user
            elif call_count == 2:
                # Already a member
                result.scalar_one_or_none.return_value = MagicMock()
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.post(
            f"/api/collections/{collection_id}/share",
            json={"github_login": "targetuser"},
        )
        assert response.status_code == 400
        assert "already a member" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_share_creates_invite_for_unknown_user(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                # User not found
                result.scalar_one_or_none.return_value = None
            elif call_count == 2:
                # No existing invite
                result.scalar_one_or_none.return_value = None
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.post(
            f"/api/collections/{collection_id}/share",
            json={"github_login": "unknownuser"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "invite_created"
        assert data["github_login"] == "unknownuser"

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_share_duplicate_invite_returns_400(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalar_one_or_none.return_value = None  # User not found
            elif call_count == 2:
                result.scalar_one_or_none.return_value = MagicMock()  # Invite exists
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.post(
            f"/api/collections/{collection_id}/share",
            json={"github_login": "someone"},
        )
        assert response.status_code == 400
        assert "Invite already exists" in response.json()["detail"]


@pytest.mark.anyio
class TestUnshareCollection:
    """Tests for DELETE /api/collections/{collection_id}/share/{identifier}."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_unshare_member_not_found_returns_404(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            else:
                # Neither member nor invite found
                result.scalar_one_or_none.return_value = None
                result.first.return_value = None
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.delete(f"/api/collections/{collection_id}/share/unknownuser")
        assert response.status_code == 404
        assert "Member or invite not found" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_unshare_by_user_id_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        target_user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_member = MagicMock()

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalar_one_or_none.return_value = mock_member
            call_count += 1
            return result

        mock_db.execute = mock_execute
        mock_db.delete = AsyncMock()
        mock_db.flush = AsyncMock()

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.delete(f"/api/collections/{collection_id}/share/{target_user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "member_removed"


# ---------------------------------------------------------------------------
# List members and invites
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestListMembers:
    """Tests for GET /api/collections/{collection_id}/members."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_list_members_private_access_denied(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

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
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalar_one_or_none.return_value = None  # Not a member
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/collections/{collection_id}/members")
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


@pytest.mark.anyio
class TestListInvites:
    """Tests for GET /api/collections/{collection_id}/invites."""

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_list_invites_not_owner_returns_403(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        owner_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = owner_id

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_collection
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/collections/{collection_id}/invites")
        assert response.status_code == 403
        assert "Only the owner" in response.json()["detail"]

    @patch("app.routes.collections.get_session")
    @patch("app.routes.collections.get_db_session")
    async def test_list_invites_success(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id

        mock_invite = MagicMock()
        mock_invite.github_login = "inviteduser"
        mock_invite.created_at = now

        mock_db = MagicMock()
        call_count = 0

        async def mock_execute(query: Any) -> MagicMock:
            nonlocal call_count
            result = MagicMock()
            if call_count == 0:
                result.scalar_one_or_none.return_value = mock_collection
            elif call_count == 1:
                result.scalars.return_value.all.return_value = [mock_invite]
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/collections/{collection_id}/invites")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["github_login"] == "inviteduser"
