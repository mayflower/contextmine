"""MCP API token management routes."""

import uuid
from datetime import UTC, datetime

from contextmine_core import (
    MCPApiToken,
    generate_api_token,
    hash_api_token,
)
from contextmine_core import (
    get_session as get_db_session,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from app.middleware import get_session

router = APIRouter(prefix="/mcp-tokens", tags=["mcp-tokens"])


class CreateTokenRequest(BaseModel):
    """Request model for creating an MCP token."""

    name: str


class CreateTokenResponse(BaseModel):
    """Response model for token creation (includes plaintext token once)."""

    id: str
    name: str
    token: str  # Plaintext token, only shown once
    created_at: datetime


class TokenResponse(BaseModel):
    """Response model for token metadata (no plaintext)."""

    id: str
    name: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get the current user ID from session or raise 401."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


@router.post("", response_model=CreateTokenResponse)
async def create_token(request: Request, body: CreateTokenRequest) -> CreateTokenResponse:
    """Create a new MCP API token. Returns the plaintext token once."""
    user_id = get_current_user_id(request)

    # Generate token and hash
    plaintext_token = generate_api_token()
    token_hash = hash_api_token(plaintext_token)

    async with get_db_session() as db:
        token = MCPApiToken(
            id=uuid.uuid4(),
            user_id=user_id,
            name=body.name,
            token_hash=token_hash,
        )
        db.add(token)
        await db.flush()

        return CreateTokenResponse(
            id=str(token.id),
            name=token.name,
            token=plaintext_token,
            created_at=token.created_at,
        )


@router.get("", response_model=list[TokenResponse])
async def list_tokens(request: Request) -> list[TokenResponse]:
    """List all MCP tokens for the current user (metadata only)."""
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        result = await db.execute(
            select(MCPApiToken)
            .where(MCPApiToken.user_id == user_id)
            .order_by(MCPApiToken.created_at.desc())
        )
        tokens = result.scalars().all()

        return [
            TokenResponse(
                id=str(t.id),
                name=t.name,
                created_at=t.created_at,
                last_used_at=t.last_used_at,
                revoked_at=t.revoked_at,
            )
            for t in tokens
        ]


@router.delete("/{token_id}")
async def revoke_token(request: Request, token_id: str) -> dict[str, str]:
    """Revoke an MCP token."""
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        result = await db.execute(
            select(MCPApiToken)
            .where(MCPApiToken.id == uuid.UUID(token_id))
            .where(MCPApiToken.user_id == user_id)
        )
        token = result.scalar_one_or_none()

        if not token:
            raise HTTPException(status_code=404, detail="Token not found")

        if token.revoked_at:
            raise HTTPException(status_code=400, detail="Token already revoked")

        token.revoked_at = datetime.now(UTC)
        await db.flush()

        return {"status": "revoked"}
