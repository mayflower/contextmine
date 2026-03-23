"""Tests for document listing and count routes.

Covers authentication, input validation, access control, pagination,
and success paths for the documents API.
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
# Authentication tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDocumentAuth:
    """Tests for document route authentication."""

    async def test_list_documents_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/sources/{uuid.uuid4()}/documents")
        assert response.status_code == 401

    async def test_document_count_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get(f"/api/sources/{uuid.uuid4()}/documents/count")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDocumentValidation:
    """Tests for document route input validation."""

    @patch("app.routes.documents.get_session")
    async def test_list_documents_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/sources/not-a-uuid/documents")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.documents.get_session")
    async def test_document_count_invalid_source_id(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        response = await client.get("/api/sources/not-a-uuid/documents/count")
        assert response.status_code == 400
        assert "Invalid source ID" in response.json()["detail"]

    @patch("app.routes.documents.get_session")
    async def test_list_documents_page_validation(
        self, mock_get_session: Any, client: AsyncClient
    ) -> None:
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}
        source_id = str(uuid.uuid4())

        # page < 1 should fail validation
        response = await client.get(f"/api/sources/{source_id}/documents?page=0")
        assert response.status_code == 422

        # page_size > 100 should fail validation
        response = await client.get(f"/api/sources/{source_id}/documents?page_size=101")
        assert response.status_code == 422

        # page_size < 1 should fail validation
        response = await client.get(f"/api/sources/{source_id}/documents?page_size=0")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Source not found
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDocumentSourceNotFound:
    """Tests for document routes when source is not found."""

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_list_documents_source_not_found(
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

        response = await client.get(f"/api/sources/{uuid.uuid4()}/documents")
        assert response.status_code == 404
        assert "Source not found" in response.json()["detail"]

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_document_count_source_not_found(
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

        response = await client.get(f"/api/sources/{uuid.uuid4()}/documents/count")
        assert response.status_code == 404
        assert "Source not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDocumentAccessControl:
    """Tests for document route access control."""

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_list_documents_private_collection_access_denied(
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
            elif call_count == 2:
                # Not a member
                result.scalar_one_or_none.return_value = None
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents")
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_document_count_private_collection_access_denied(
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
            elif call_count == 2:
                result.scalar_one_or_none.return_value = None  # Not a member
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents/count")
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


@pytest.mark.anyio
class TestDocumentListSuccess:
    """Tests for successful document listing."""

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_list_documents_empty(
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
            elif call_count == 2:
                result.scalar.return_value = 0  # total count
            elif call_count == 3:
                result.scalars.return_value.all.return_value = []  # documents
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_list_documents_with_results(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        user_id = uuid.uuid4()
        source_id = uuid.uuid4()
        collection_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        now = datetime.now(UTC)
        mock_get_session.return_value = {"user_id": str(user_id)}

        mock_source = MagicMock()
        mock_source.id = source_id
        mock_source.collection_id = collection_id

        mock_collection = MagicMock()
        mock_collection.id = collection_id
        mock_collection.owner_user_id = user_id
        mock_collection.visibility = CollectionVisibility.GLOBAL

        mock_doc = MagicMock()
        mock_doc.id = doc_id
        mock_doc.source_id = source_id
        mock_doc.uri = "https://github.com/owner/repo/blob/main/README.md"
        mock_doc.title = "README.md"
        mock_doc.content_hash = "abc123"
        mock_doc.meta = {"language": "markdown"}
        mock_doc.last_seen_at = now
        mock_doc.created_at = now
        mock_doc.updated_at = now

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
                result.scalar.return_value = 1  # total count
            elif call_count == 3:
                result.scalars.return_value.all.return_value = [mock_doc]
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert len(data["documents"]) == 1
        assert data["documents"][0]["id"] == str(doc_id)
        assert data["documents"][0]["uri"] == "https://github.com/owner/repo/blob/main/README.md"
        assert data["documents"][0]["title"] == "README.md"
        assert data["documents"][0]["content_hash"] == "abc123"
        assert data["documents"][0]["meta"] == {"language": "markdown"}

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_list_documents_custom_pagination(
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
            elif call_count == 2:
                result.scalar.return_value = 100
            elif call_count == 3:
                result.scalars.return_value.all.return_value = []
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents?page=3&page_size=25")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 3
        assert data["page_size"] == 25
        assert data["total"] == 100


@pytest.mark.anyio
class TestDocumentCountSuccess:
    """Tests for successful document count retrieval."""

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_document_count_success(
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
            elif call_count == 2:
                result.scalar.return_value = 42  # document count
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents/count")
        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == str(source_id)
        assert data["document_count"] == 42

    @patch("app.routes.documents.get_session")
    @patch("app.routes.documents.get_db_session")
    async def test_document_count_global_collection_access(
        self,
        mock_get_db_session: Any,
        mock_get_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that a non-owner can access document count for global collections."""
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
            elif call_count == 2:
                result.scalar.return_value = 10
            call_count += 1
            return result

        mock_db.execute = mock_execute

        mock_get_db_session.return_value = _mock_db_session(mock_db)

        response = await client.get(f"/api/sources/{source_id}/documents/count")
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 10
