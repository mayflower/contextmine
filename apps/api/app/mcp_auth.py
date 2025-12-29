"""Custom GitHub OAuth provider for MCP with ContextMine user mapping.

This module extends FastMCP's GitHubProvider to:
1. Authenticate MCP clients via GitHub OAuth
2. Map GitHub users to ContextMine database users
3. Store user_id in context for tool authorization
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

import httpx
from contextmine_core import User
from contextmine_core import get_session as get_db_session
from contextmine_core.settings import get_settings
from fastmcp.server.auth.providers.github import GitHubProvider
from pydantic import AnyHttpUrl
from sqlalchemy import select
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Context variable to pass user_id to tools (async-safe)
_current_user_id: ContextVar[uuid.UUID | None] = ContextVar("current_user_id", default=None)


def set_current_user_id(user_id: uuid.UUID | None) -> None:
    """Set the current user ID for authorization."""
    _current_user_id.set(user_id)


def get_current_user_id() -> uuid.UUID | None:
    """Get the current user ID."""
    return _current_user_id.get()


class UserMappingMiddleware(BaseHTTPMiddleware):
    """Middleware to map GitHub user to ContextMine database user.

    This middleware runs after FastMCP's authentication middleware.
    It extracts the GitHub token from the authenticated request,
    fetches GitHub user info, and maps to a ContextMine user.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Map GitHub user to database user."""
        # Check if request is authenticated
        if not hasattr(request, "user") or not hasattr(request.user, "access_token"):
            return await call_next(request)

        try:
            # Get GitHub token from authenticated user
            github_token = request.user.access_token.token

            # Fetch GitHub user info
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {github_token}",
                        "Accept": "application/vnd.github+json",
                    },
                    timeout=10.0,
                )

                if resp.status_code != 200:
                    logger.warning("Failed to fetch GitHub user: %s", resp.status_code)
                    return await call_next(request)

                github_user = resp.json()

            # Look up or create user in database
            async with get_db_session() as db:
                result = await db.execute(
                    select(User).where(User.github_user_id == github_user["id"])
                )
                user = result.scalar_one_or_none()

                if not user:
                    # Create new user
                    user = User(
                        id=uuid.uuid4(),
                        github_user_id=github_user["id"],
                        github_login=github_user["login"],
                        name=github_user.get("name"),
                        avatar_url=github_user.get("avatar_url"),
                    )
                    db.add(user)
                    await db.commit()
                    logger.info("Created new user for GitHub: %s", github_user["login"])

                # Store user_id in context for tools
                set_current_user_id(user.id)

        except Exception as e:
            logger.exception("Error mapping GitHub user: %s", e)
            # Continue without user mapping - request is still authenticated

        try:
            return await call_next(request)
        finally:
            # Clear user_id after request
            set_current_user_id(None)


class ContextMineGitHubProvider(GitHubProvider):
    """GitHub OAuth provider that maps to ContextMine users.

    Extends FastMCP's GitHubProvider with additional middleware to:
    - Map authenticated GitHub users to database users
    - Store user_id in context for tool authorization
    """

    def __init__(self) -> None:
        """Initialize with settings from environment."""
        settings = get_settings()

        if not settings.github_client_id or not settings.github_client_secret:
            raise ValueError(
                "GitHub OAuth not configured. "
                "Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET environment variables."
            )

        super().__init__(
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            # Use /api path so OAuth callbacks go to the unified /api/auth/callback
            # The callback handler forwards MCP flows to /mcp/auth/callback
            base_url=AnyHttpUrl(f"{settings.mcp_oauth_base_url}/api"),
            # Enable consent screen for security (protects against confused deputy)
            require_authorization_consent=True,
            # Request scopes needed for user info
            required_scopes=["read:user"],
        )

    def get_middleware(self) -> list:
        """Get middleware including user mapping.

        Returns parent middleware plus our user mapping middleware.
        """
        # Get FastMCP's auth middleware
        middleware = super().get_middleware()

        # Add user mapping middleware (runs after auth)
        middleware.append(Middleware(UserMappingMiddleware))

        return middleware
