"""Tests for MCP identity mapping from FastMCP's verified claims."""

import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp_auth import UserMappingMiddleware, get_current_user_id
from starlette.requests import Request
from starlette.responses import JSONResponse


def _request(claims: dict) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [],
            "query_string": b"",
            "user": SimpleNamespace(
                access_token=SimpleNamespace(claims=claims),
            ),
        }
    )


@pytest.mark.anyio
async def test_mcp_identity_uses_verified_claims_without_profile_request() -> None:
    user_id = uuid.uuid4()
    db = MagicMock(commit=AsyncMock())

    @asynccontextmanager
    async def db_session():
        yield db

    async def downstream(_request: Request) -> JSONResponse:
        return JSONResponse({"user_id": str(get_current_user_id())})

    middleware = UserMappingMiddleware(MagicMock())
    mapped_user = SimpleNamespace(id=user_id)
    with (
        patch("app.mcp_auth.get_db_session", return_value=db_session()),
        patch(
            "app.mcp_auth.upsert_github_user",
            new=AsyncMock(return_value=mapped_user),
        ) as upsert,
    ):
        response = await middleware.dispatch(
            _request({"sub": "1234", "login": "octocat"}), downstream
        )

    assert json.loads(response.body) == {"user_id": str(user_id)}
    upsert.assert_awaited_once_with(
        db,
        {"id": 1234, "login": "octocat", "name": None, "avatar_url": None},
    )
    db.commit.assert_awaited_once()
    assert get_current_user_id() is None


@pytest.mark.anyio
async def test_mcp_identity_fails_closed_for_unusable_claims() -> None:
    downstream = AsyncMock(return_value=JSONResponse({"ok": True}))
    middleware = UserMappingMiddleware(MagicMock())

    response = await middleware.dispatch(_request({"sub": "1234"}), downstream)

    assert response.status_code == 401
    downstream.assert_not_awaited()
