"""Authentication services for GitHub OAuth and API tokens."""

import base64
import hashlib
import secrets
from typing import Any

import httpx
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from contextmine_core.settings import get_settings
from cryptography.fernet import Fernet

# Argon2 hasher for API tokens
_ph = PasswordHasher()

# GitHub OAuth endpoints
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def get_fernet() -> Fernet:
    """Get Fernet instance for token encryption."""
    settings = get_settings()
    # Derive a 32-byte key from the encryption key using SHA256
    key_bytes = hashlib.sha256(settings.token_encryption_key.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def encrypt_token(token: str) -> str:
    """Encrypt an OAuth token for storage."""
    fernet = get_fernet()
    return fernet.encrypt(token.encode()).decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt an OAuth token from storage."""
    fernet = get_fernet()
    return fernet.decrypt(encrypted_token.encode()).decode()


def generate_state() -> str:
    """Generate a random state parameter for OAuth."""
    return secrets.token_urlsafe(32)


def get_github_authorize_url(state: str) -> str:
    """Get the GitHub OAuth authorization URL."""
    settings = get_settings()
    if not settings.github_client_id:
        raise ValueError("GITHUB_CLIENT_ID is not configured")

    redirect_uri = f"{settings.public_base_url}/api/auth/callback"
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": redirect_uri,
        "scope": "read:user user:email repo",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{GITHUB_AUTHORIZE_URL}?{query}"


async def exchange_code_for_token(code: str) -> str:
    """Exchange authorization code for access token."""
    settings = get_settings()
    if not settings.github_client_id or not settings.github_client_secret:
        raise ValueError("GitHub OAuth credentials are not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ValueError(f"GitHub OAuth error: {data.get('error_description', data['error'])}")

        return data["access_token"]


async def get_github_user(access_token: str) -> dict[str, Any]:
    """Fetch user profile from GitHub API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        response.raise_for_status()
        return response.json()


def generate_api_token() -> str:
    """Generate a random API token for MCP access."""
    return secrets.token_urlsafe(32)


def hash_api_token(token: str) -> str:
    """Hash an API token for secure storage using Argon2."""
    return _ph.hash(token)


def verify_api_token(token: str, token_hash: str) -> bool:
    """Verify an API token against its hash."""
    try:
        _ph.verify(token_hash, token)
        return True
    except VerifyMismatchError:
        return False


def compute_ssh_key_fingerprint(private_key_pem: str) -> str:
    """Compute SHA256 fingerprint of an SSH private key.

    Args:
        private_key_pem: The private key in PEM format

    Returns:
        SHA256 fingerprint in the format "SHA256:base64..."
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    # Load the private key
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None,
    )

    # Get the public key in OpenSSH format
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.OpenSSH,
        format=PublicFormat.OpenSSH,
    )

    # Compute SHA256 hash of the public key
    # OpenSSH format is "ssh-xxx base64data comment"
    # We hash the base64-decoded key data
    parts = public_key_bytes.decode().split()
    key_data = base64.b64decode(parts[1]) if len(parts) >= 2 else public_key_bytes

    fingerprint = hashlib.sha256(key_data).digest()
    fingerprint_b64 = base64.b64encode(fingerprint).decode().rstrip("=")

    return f"SHA256:{fingerprint_b64}"


def validate_ssh_private_key(key_pem: str) -> bool:
    """Validate that a string is a valid SSH private key.

    Args:
        key_pem: The key in PEM format

    Returns:
        True if valid, False otherwise
    """
    from cryptography.hazmat.primitives import serialization

    try:
        serialization.load_pem_private_key(
            key_pem.encode(),
            password=None,
        )
        return True
    except Exception:
        return False
