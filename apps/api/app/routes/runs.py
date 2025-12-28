"""Sync runs routes."""

import uuid
from datetime import datetime

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Source,
    SyncRun,
)
from contextmine_core import (
    get_session as get_db_session,
)
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select

from app.middleware import get_session

router = APIRouter(tags=["runs"])


class SyncRunResponse(BaseModel):
    """Response model for a sync run."""

    id: str
    source_id: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    stats: dict | None
    error: str | None


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get the current user ID from session or raise 401."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


@router.get("/runs", response_model=list[SyncRunResponse])
async def list_runs(
    request: Request,
    source_id: str = Query(..., description="Source ID to get runs for"),
) -> list[SyncRunResponse]:
    """List sync runs for a source."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        # Get source
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check collection access
        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()

        # Verify access: global, owner, or member
        if (
            collection.visibility == CollectionVisibility.PRIVATE
            and collection.owner_user_id != user_id
        ):
            result = await db.execute(
                select(CollectionMember)
                .where(CollectionMember.collection_id == collection.id)
                .where(CollectionMember.user_id == user_id)
            )
            if not result.scalar_one_or_none():
                raise HTTPException(status_code=403, detail="Access denied to this source")

        # Get runs for source, ordered by started_at desc
        result = await db.execute(
            select(SyncRun)
            .where(SyncRun.source_id == src_uuid)
            .order_by(SyncRun.started_at.desc())
            .limit(100)  # Limit to last 100 runs
        )
        runs = result.scalars().all()

        return [
            SyncRunResponse(
                id=str(run.id),
                source_id=str(run.source_id),
                started_at=run.started_at,
                finished_at=run.finished_at,
                status=run.status.value,
                stats=run.stats,
                error=run.error,
            )
            for run in runs
        ]
