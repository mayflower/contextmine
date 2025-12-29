"""Tests for MCP token hashing utilities."""

from contextmine_core import generate_api_token, hash_api_token, verify_api_token


class TestTokenHashing:
    """Tests for API token hashing functions.

    These utilities are used for backwards compatibility with existing
    MCP tokens in the database, even though new authentication uses
    GitHub OAuth via FastMCP.
    """

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
