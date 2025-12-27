"""Database health check and stats endpoints."""

from contextmine_core import (
    Chunk,
    Collection,
    Document,
    Source,
    SyncRun,
    get_session,
    get_settings,
)
from fastapi import APIRouter
from sqlalchemy import func, select, text

router = APIRouter(tags=["database"])


@router.get("/db/health")
async def db_health_check() -> dict[str, str]:
    """Check database connectivity by running a simple query."""
    settings = get_settings()

    if not settings.database_url:
        return {"db": "not_configured"}

    try:
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            return {"db": "ok"}
    except Exception as e:
        return {"db": "error", "detail": str(e)}


@router.get("/stats")
async def get_stats() -> dict:
    """Get dashboard statistics."""
    settings = get_settings()

    if not settings.database_url:
        return {"error": "database_not_configured"}

    try:
        async with get_session() as session:
            # Count collections
            result = await session.execute(select(func.count(Collection.id)))
            collections_count = result.scalar() or 0

            # Count sources
            result = await session.execute(select(func.count(Source.id)))
            sources_count = result.scalar() or 0

            # Count documents
            result = await session.execute(select(func.count(Document.id)))
            documents_count = result.scalar() or 0

            # Count chunks
            result = await session.execute(select(func.count(Chunk.id)))
            chunks_count = result.scalar() or 0

            # Count embedded chunks (where embedded_at is not null)
            result = await session.execute(
                select(func.count(Chunk.id)).where(Chunk.embedded_at.isnot(None))
            )
            embedded_chunks_count = result.scalar() or 0

            # Count sync runs by status
            result = await session.execute(
                select(SyncRun.status, func.count(SyncRun.id)).group_by(SyncRun.status)
            )
            runs_by_status = {row[0].value: row[1] for row in result.fetchall()}

            # Get recent sync runs (last 10)
            result = await session.execute(
                select(
                    SyncRun.id,
                    SyncRun.status,
                    SyncRun.started_at,
                    SyncRun.finished_at,
                    Source.url.label("source_url"),
                )
                .join(Source, SyncRun.source_id == Source.id)
                .order_by(SyncRun.started_at.desc())
                .limit(10)
            )
            recent_runs = [
                {
                    "id": str(row.id),
                    "status": row.status.value,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                    "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                    "source_url": row.source_url,
                }
                for row in result.fetchall()
            ]

            return {
                "collections": collections_count,
                "sources": sources_count,
                "documents": documents_count,
                "chunks": chunks_count,
                "embedded_chunks": embedded_chunks_count,
                "runs_by_status": runs_by_status,
                "recent_runs": recent_runs,
            }
    except Exception as e:
        return {"error": str(e)}
