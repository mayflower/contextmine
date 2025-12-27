"""Documents routes."""

import uuid
from datetime import datetime

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Document,
    Source,
)
from contextmine_core import (
    get_session as get_db_session,
)
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from app.middleware import get_session

router = APIRouter(tags=["documents"])


class DocumentResponse(BaseModel):
    """Response model for a document."""

    id: str
    source_id: str
    uri: str
    title: str
    content_hash: str
    meta: dict
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Response model for paginated document list."""

    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get the current user ID from session or raise 401."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


@router.get("/sources/{source_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    source_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
) -> DocumentListResponse:
    """List documents for a source (paginated)."""
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
        result = await db.execute(
            select(Collection).where(Collection.id == source.collection_id)
        )
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
                raise HTTPException(
                    status_code=403, detail="Access denied to this source"
                )

        # Get total count
        result = await db.execute(
            select(func.count()).select_from(Document).where(Document.source_id == src_uuid)
        )
        total = result.scalar() or 0

        # Get paginated documents
        offset = (page - 1) * page_size
        result = await db.execute(
            select(Document)
            .where(Document.source_id == src_uuid)
            .order_by(Document.updated_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        documents = result.scalars().all()

        return DocumentListResponse(
            documents=[
                DocumentResponse(
                    id=str(doc.id),
                    source_id=str(doc.source_id),
                    uri=doc.uri,
                    title=doc.title,
                    content_hash=doc.content_hash,
                    meta=doc.meta,
                    last_seen_at=doc.last_seen_at,
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
                for doc in documents
            ],
            total=total,
            page=page,
            page_size=page_size,
        )


@router.get("/sources/{source_id}/documents/count")
async def get_document_count(request: Request, source_id: str) -> dict:
    """Get document count for a source."""
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
        result = await db.execute(
            select(Collection).where(Collection.id == source.collection_id)
        )
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
                raise HTTPException(
                    status_code=403, detail="Access denied to this source"
                )

        # Get count
        result = await db.execute(
            select(func.count()).select_from(Document).where(Document.source_id == src_uuid)
        )
        count = result.scalar() or 0

        return {"source_id": source_id, "document_count": count}
