"""Tests for MCP token management and authentication."""

from typing import Any
from unittest.mock import patch

import pytest
from contextmine_core import generate_api_token, hash_api_token, verify_api_token
from httpx import AsyncClient


class TestTokenHashing:
    """Tests for API token hashing functions."""

    def test_generate_token_is_random(self) -> None:
        """Test that generate_api_token produces unique values."""
        token1 = generate_api_token()
        token2 = generate_api_token()
        assert token1 != token2
        assert len(token1) >= 32

    def test_hash_and_verify_token(self) -> None:
        """Test that tokens can be hashed and verified."""
        token = generate_api_token()
        token_hash = hash_api_token(token)

        # Hash should be different from token
        assert token_hash != token

        # Verification should work
        assert verify_api_token(token, token_hash) is True

    def test_verify_wrong_token(self) -> None:
        """Test that wrong tokens are rejected."""
        token = generate_api_token()
        token_hash = hash_api_token(token)
        wrong_token = generate_api_token()

        assert verify_api_token(wrong_token, token_hash) is False


@pytest.mark.anyio
class TestMCPTokenRoutes:
    """Tests for MCP token route handlers."""

    async def test_create_token_requires_auth(self, client: AsyncClient) -> None:
        """Test that creating a token requires authentication."""
        response = await client.post(
            "/api/mcp-tokens",
            json={"name": "Test Token"},
        )
        assert response.status_code == 401

    async def test_list_tokens_requires_auth(self, client: AsyncClient) -> None:
        """Test that listing tokens requires authentication."""
        response = await client.get("/api/mcp-tokens")
        assert response.status_code == 401

    async def test_revoke_token_requires_auth(self, client: AsyncClient) -> None:
        """Test that revoking a token requires authentication."""
        response = await client.delete("/api/mcp-tokens/some-id")
        assert response.status_code == 401


@pytest.mark.anyio
class TestMCPEndpointAuth:
    """Tests for MCP endpoint authentication."""

    async def test_mcp_requires_bearer_token(self, client: AsyncClient) -> None:
        """Test that /mcp returns 401 without a token."""
        response = await client.get("/mcp/sse")
        assert response.status_code == 401
        data = response.json()
        assert "Missing or invalid Authorization header" in data["error"]

    @patch("app.mcp_server.get_db_session")
    async def test_mcp_rejects_invalid_token(
        self, mock_get_db_session: Any, client: AsyncClient
    ) -> None:
        """Test that /mcp returns 401 with invalid token."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        # Mock empty token list
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        response = await client.get(
            "/mcp/sse",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "Invalid or revoked token" in data["error"]

    @patch("app.mcp_server.get_settings")
    async def test_mcp_origin_check(self, mock_get_settings: Any, client: AsyncClient) -> None:
        """Test that /mcp validates origin when allowlist is configured."""
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.mcp_allowed_origins = "https://allowed.com,https://other.com"
        mock_get_settings.return_value = mock_settings

        # Request with disallowed origin should fail
        response = await client.get(
            "/mcp/sse",
            headers={
                "Authorization": "Bearer some-token",
                "Origin": "https://evil.com",
            },
        )
        assert response.status_code == 403
        data = response.json()
        assert "Origin not allowed" in data["error"]

    @patch("app.mcp_server.get_db_session")
    @patch("app.mcp_server.get_settings")
    async def test_mcp_origin_allowed_when_empty(
        self,
        mock_get_settings: Any,
        mock_get_db_session: Any,
        client: AsyncClient,
    ) -> None:
        """Test that /mcp allows all origins when allowlist is empty (dev mode)."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        mock_settings = MagicMock()
        mock_settings.mcp_allowed_origins = ""  # Empty = allow all
        mock_get_settings.return_value = mock_settings

        # Mock empty token list
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def mock_session():
            yield mock_db

        mock_get_db_session.return_value = mock_session()

        # Should proceed to token check (not origin error)
        response = await client.get(
            "/mcp/sse",
            headers={
                "Authorization": "Bearer some-token",
                "Origin": "https://any-origin.com",
            },
        )
        # Should fail on token check, not origin
        assert response.status_code == 401
        data = response.json()
        assert "Invalid or revoked token" in data["error"]
