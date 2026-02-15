"""Validation dashboard routes."""

from __future__ import annotations

import uuid

from contextmine_core import get_session as get_db_session
from contextmine_core.validation import get_latest_validation_status, refresh_validation_snapshots
from fastapi import APIRouter, HTTPException, Request

from app.middleware import get_session

router = APIRouter(prefix="/validation", tags=["validation"])


def _user_id_or_401(request: Request) -> uuid.UUID:
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


@router.get("/status")
async def validation_status(request: Request, collection_id: str | None = None) -> dict:
    """Get latest validation and orchestration status."""
    _user_id_or_401(request)

    collection_uuid: uuid.UUID | None = None
    if collection_id:
        try:
            collection_uuid = uuid.UUID(collection_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid collection_id") from e

    async with get_db_session() as db:
        # Refresh from live connectors on every request.
        await refresh_validation_snapshots(db, collection_uuid)
        payload = await get_latest_validation_status(db, collection_uuid)
        await db.commit()
        return payload
