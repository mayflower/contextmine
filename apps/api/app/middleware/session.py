"""Session middleware for cookie-based sessions."""

import json
from typing import Any

from contextmine_core import get_settings
from fastapi import Request, Response
from itsdangerous import BadSignature, TimestampSigner
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

SESSION_COOKIE_NAME = "contextmine_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def get_signer() -> TimestampSigner:
    """Get signer for session cookies."""
    settings = get_settings()
    return TimestampSigner(settings.session_secret)


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware that manages signed session cookies."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and manage session."""
        # Load session from cookie
        session_data: dict[str, Any] = {}
        cookie = request.cookies.get(SESSION_COOKIE_NAME)

        if cookie:
            try:
                signer = get_signer()
                unsigned = signer.unsign(cookie, max_age=SESSION_MAX_AGE)
                session_data = json.loads(unsigned.decode())
            except (BadSignature, json.JSONDecodeError):
                # Invalid or expired session, start fresh
                session_data = {}

        # Store session in request state
        request.state.session = session_data
        request.state.session_modified = False

        # Process request
        response = await call_next(request)

        # Save session if modified
        if getattr(request.state, "session_modified", False):
            session_json = json.dumps(request.state.session)
            signer = get_signer()
            signed = signer.sign(session_json.encode()).decode()

            settings = get_settings()
            is_secure = not settings.debug  # Only secure in production

            response.set_cookie(
                SESSION_COOKIE_NAME,
                signed,
                max_age=SESSION_MAX_AGE,
                httponly=True,
                secure=is_secure,
                samesite="lax",
            )

        return response


def get_session(request: Request) -> dict[str, Any]:
    """Get the current session from request."""
    return getattr(request.state, "session", {})


def set_session(request: Request, data: dict[str, Any]) -> None:
    """Set session data and mark as modified."""
    request.state.session = data
    request.state.session_modified = True


def clear_session(request: Request) -> None:
    """Clear the session."""
    request.state.session = {}
    request.state.session_modified = True
