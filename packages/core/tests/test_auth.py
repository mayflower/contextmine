"""Tests for auth module: token encryption, state generation, SSH key helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.auth import (
    compute_ssh_key_fingerprint,
    decrypt_token,
    encrypt_token,
    generate_state,
    get_fernet,
    get_github_authorize_url,
    validate_ssh_private_key,
)
from cryptography.fernet import InvalidToken
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_settings(**overrides):
    """Build a mock Settings object with sensible defaults."""
    defaults = {
        "token_encryption_key": "test-key-for-fernet-derivation",
        "github_client_id": "test-client-id",
        "github_client_secret": "test-client-secret",
        "public_base_url": "https://example.com",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


@pytest.fixture(autouse=True)
def _patch_settings():
    """Patch get_settings for every test so we never touch real env."""
    with patch("contextmine_core.auth.get_settings", return_value=_mock_settings()):
        yield


# ---------------------------------------------------------------------------
# Fernet / token encryption
# ---------------------------------------------------------------------------


class TestFernetEncryption:
    """Tests for get_fernet, encrypt_token, decrypt_token."""

    def test_get_fernet_returns_fernet_instance(self):
        fernet = get_fernet()
        assert fernet is not None
        # Fernet should be able to encrypt something
        token = fernet.encrypt(b"hello")
        assert fernet.decrypt(token) == b"hello"

    def test_encrypt_decrypt_roundtrip(self):
        original = "ghp_some_oauth_token_12345"
        encrypted = encrypt_token(original)
        assert encrypted != original
        decrypted = decrypt_token(encrypted)
        assert decrypted == original

    def test_encrypt_produces_different_ciphertext_each_time(self):
        """Fernet uses random IV so two encryptions should differ."""
        token = "test-token"
        enc1 = encrypt_token(token)
        enc2 = encrypt_token(token)
        assert enc1 != enc2
        # But both should decrypt to the same value
        assert decrypt_token(enc1) == token
        assert decrypt_token(enc2) == token

    def test_decrypt_invalid_token_raises(self):
        with pytest.raises(InvalidToken):
            decrypt_token("not-valid-fernet-ciphertext")

    def test_encrypt_empty_string(self):
        encrypted = encrypt_token("")
        assert decrypt_token(encrypted) == ""


# ---------------------------------------------------------------------------
# OAuth state
# ---------------------------------------------------------------------------


class TestGenerateState:
    def test_returns_non_empty_string(self):
        state = generate_state()
        assert isinstance(state, str)
        assert len(state) > 0

    def test_returns_url_safe_characters(self):
        state = generate_state()
        # url-safe base64 characters
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=")
        assert set(state).issubset(allowed)

    def test_states_are_unique(self):
        states = {generate_state() for _ in range(50)}
        assert len(states) == 50


# ---------------------------------------------------------------------------
# GitHub authorize URL
# ---------------------------------------------------------------------------


class TestGetGithubAuthorizeUrl:
    def test_basic_url_structure(self):
        url = get_github_authorize_url("random-state")
        assert url.startswith("https://github.com/login/oauth/authorize?")
        assert "client_id=test-client-id" in url
        assert "state=random-state" in url
        assert "scope=read:user" in url

    def test_redirect_uri_uses_public_base_url(self):
        url = get_github_authorize_url("s")
        assert "redirect_uri=https://example.com/api/auth/callback" in url

    def test_raises_when_client_id_missing(self):
        with (
            patch(
                "contextmine_core.auth.get_settings",
                return_value=_mock_settings(github_client_id=None),
            ),
            pytest.raises(ValueError, match="GITHUB_CLIENT_ID is not configured"),
        ):
            get_github_authorize_url("s")


# ---------------------------------------------------------------------------
# exchange_code_for_token (async)
# ---------------------------------------------------------------------------


class TestExchangeCodeForToken:
    @pytest.mark.anyio
    async def test_raises_when_credentials_missing(self):
        from contextmine_core.auth import exchange_code_for_token

        with (
            patch(
                "contextmine_core.auth.get_settings",
                return_value=_mock_settings(github_client_id=None, github_client_secret=None),
            ),
            pytest.raises(ValueError, match="GitHub OAuth credentials"),
        ):
            await exchange_code_for_token("some-code")

    @pytest.mark.anyio
    async def test_returns_access_token_on_success(self):
        from contextmine_core.auth import exchange_code_for_token

        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "gho_abc123"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("contextmine_core.auth.httpx.AsyncClient", return_value=mock_client):
            token = await exchange_code_for_token("code-123")

        assert token == "gho_abc123"

    @pytest.mark.anyio
    async def test_raises_on_github_error_response(self):
        from contextmine_core.auth import exchange_code_for_token

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": "bad_verification_code",
            "error_description": "The code is invalid.",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("contextmine_core.auth.httpx.AsyncClient", return_value=mock_client),
            pytest.raises(ValueError, match="The code is invalid"),
        ):
            await exchange_code_for_token("bad-code")


# ---------------------------------------------------------------------------
# get_github_user (async)
# ---------------------------------------------------------------------------


class TestGetGithubUser:
    @pytest.mark.anyio
    async def test_returns_user_dict(self):
        from contextmine_core.auth import get_github_user

        user_data = {"login": "octocat", "id": 1, "email": "octocat@github.com"}
        mock_response = MagicMock()
        mock_response.json.return_value = user_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("contextmine_core.auth.httpx.AsyncClient", return_value=mock_client):
            result = await get_github_user("gho_token123")

        assert result["login"] == "octocat"
        assert result["id"] == 1

    @pytest.mark.anyio
    async def test_sends_correct_headers(self):
        from contextmine_core.auth import get_github_user

        mock_response = MagicMock()
        mock_response.json.return_value = {"login": "x"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("contextmine_core.auth.httpx.AsyncClient", return_value=mock_client):
            await get_github_user("my-token")

        call_kwargs = mock_client.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "Bearer my-token" in headers["Authorization"]


# ---------------------------------------------------------------------------
# SSH key validation and fingerprinting
# ---------------------------------------------------------------------------


def _generate_rsa_key_pem() -> str:
    """Generate a fresh RSA private key in traditional PEM format for testing."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    # Use TraditionalOpenSSL format so load_pem_private_key can parse it
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


def _generate_ed25519_key_pem() -> str:
    """Generate a fresh Ed25519 private key in PKCS8 PEM format for testing."""
    key = ed25519.Ed25519PrivateKey.generate()
    # Ed25519 does not support TraditionalOpenSSL; use PKCS8
    return key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()).decode()


class TestValidateSshPrivateKey:
    def test_valid_rsa_key(self):
        pem = _generate_rsa_key_pem()
        assert validate_ssh_private_key(pem) is True

    def test_valid_ed25519_key(self):
        pem = _generate_ed25519_key_pem()
        assert validate_ssh_private_key(pem) is True

    def test_invalid_key_returns_false(self):
        assert validate_ssh_private_key("not-a-key") is False

    def test_empty_string_returns_false(self):
        assert validate_ssh_private_key("") is False


class TestComputeSshKeyFingerprint:
    def test_fingerprint_format_sha256(self):
        pem = _generate_rsa_key_pem()
        fp = compute_ssh_key_fingerprint(pem)
        assert fp.startswith("SHA256:")
        # Base64 portion after SHA256: should be non-empty
        assert len(fp) > len("SHA256:")

    def test_same_key_same_fingerprint(self):
        pem = _generate_rsa_key_pem()
        fp1 = compute_ssh_key_fingerprint(pem)
        fp2 = compute_ssh_key_fingerprint(pem)
        assert fp1 == fp2

    def test_different_keys_different_fingerprints(self):
        pem1 = _generate_rsa_key_pem()
        pem2 = _generate_rsa_key_pem()
        assert compute_ssh_key_fingerprint(pem1) != compute_ssh_key_fingerprint(pem2)

    def test_ed25519_fingerprint(self):
        pem = _generate_ed25519_key_pem()
        fp = compute_ssh_key_fingerprint(pem)
        assert fp.startswith("SHA256:")
