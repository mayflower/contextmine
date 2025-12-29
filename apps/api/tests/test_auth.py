"""Tests for authentication routes and services."""

from unittest.mock import AsyncMock, patch

import pytest
from contextmine_core import decrypt_token, encrypt_token, generate_state
from httpx import AsyncClient


class TestAuthService:
    """Tests for auth service functions."""

    def test_generate_state_is_random(self) -> None:
        """Test that generate_state produces unique values."""
        state1 = generate_state()
        state2 = generate_state()
        assert state1 != state2
        assert len(state1) >= 32  # URL-safe base64 of 32 bytes

    def test_encrypt_decrypt_token(self) -> None:
        """Test that tokens can be encrypted and decrypted."""
        original_token = "ghp_test_access_token_12345"
        encrypted = encrypt_token(original_token)
        assert encrypted != original_token
        decrypted = decrypt_token(encrypted)
        assert decrypted == original_token

    def test_encrypted_tokens_are_different(self) -> None:
        """Test that encrypting the same token produces different ciphertext."""
        token = "ghp_test_token"
        encrypted1 = encrypt_token(token)
        encrypted2 = encrypt_token(token)
        # Fernet includes timestamp and random IV, so ciphertexts differ
        assert encrypted1 != encrypted2
        # But both decrypt to the same value
        assert decrypt_token(encrypted1) == decrypt_token(encrypted2) == token


@pytest.mark.anyio
class TestAuthRoutes:
    """Tests for auth route handlers."""

    async def test_login_requires_oauth_config(self, client: AsyncClient) -> None:
        """Test that /api/auth/login returns 500 when OAuth not configured."""
        from unittest.mock import MagicMock

        # Mock settings to return None for github_client_id
        mock_settings = MagicMock()
        mock_settings.github_client_id = None

        with patch("app.routes.auth.get_settings", return_value=mock_settings):
            response = await client.get("/api/auth/login", follow_redirects=False)
            # When OAuth is not configured, it should return 500
            assert response.status_code == 500
            data = response.json()
            assert "not configured" in data["detail"].lower()

    @patch("app.routes.auth.get_settings")
    @patch("app.routes.auth.get_github_authorize_url")
    async def test_login_redirects_to_github(
        self,
        mock_get_authorize_url: AsyncMock,
        mock_get_settings: AsyncMock,
        client: AsyncClient,
    ) -> None:
        """Test that /api/auth/login redirects to GitHub OAuth when configured."""
        from unittest.mock import MagicMock

        # Mock settings with GitHub OAuth configured
        mock_settings = MagicMock()
        mock_settings.github_client_id = "test_client_id"
        mock_get_settings.return_value = mock_settings
        mock_get_authorize_url.return_value = (
            "https://github.com/login/oauth/authorize?client_id=test&state=abc"
        )

        response = await client.get("/api/auth/login", follow_redirects=False)
        assert response.status_code == 302
        location = response.headers["location"]
        assert "github.com/login/oauth/authorize" in location

    async def test_me_returns_401_when_not_authenticated(self, client: AsyncClient) -> None:
        """Test that /api/auth/me returns 401 when not logged in."""
        response = await client.get("/api/auth/me")
        assert response.status_code == 401
        data = response.json()
        assert data["detail"] == "Not authenticated"

    async def test_logout_clears_session(self, client: AsyncClient) -> None:
        """Test that /api/auth/logout returns success."""
        response = await client.get("/api/auth/logout")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "logged_out"

    async def test_callback_requires_code_and_state(self, client: AsyncClient) -> None:
        """Test that callback without code/state redirects with error."""
        response = await client.get("/api/auth/callback", follow_redirects=False)
        assert response.status_code == 302
        location = response.headers["location"]
        assert "error=missing_params" in location

    async def test_callback_forwards_mcp_flows(self, client: AsyncClient) -> None:
        """Test that callback with mismatched state forwards to MCP OAuth handler.

        When state doesn't match the session, we assume it's an MCP OAuth flow
        and forward to /mcp/auth/callback for FastMCP to handle.
        """
        response = await client.get(
            "/api/auth/callback?code=test_code&state=mcp_state",
            follow_redirects=False,
        )
        assert response.status_code == 302
        location = response.headers["location"]
        # Should forward to MCP OAuth handler, not return error
        assert location == "/mcp/auth/callback?code=test_code&state=mcp_state"

    async def test_callback_error_parameter(self, client: AsyncClient) -> None:
        """Test that callback with error parameter redirects with error."""
        response = await client.get(
            "/api/auth/callback?error=access_denied",
            follow_redirects=False,
        )
        assert response.status_code == 302
        location = response.headers["location"]
        assert "error=access_denied" in location
