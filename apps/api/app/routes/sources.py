"""Sources management routes."""

import hashlib
import re
import secrets
import uuid
from datetime import UTC, datetime

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Document,
    Source,
    SourceIngestToken,
    SourceType,
    compute_ssh_key_fingerprint,
    encrypt_token,
    validate_ssh_private_key,
)
from contextmine_core import (
    get_session as get_db_session,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select

from app.middleware import get_session

router = APIRouter(tags=["sources"])


class CreateSourceRequest(BaseModel):
    """Request model for creating a source."""

    type: str  # "github" or "web"
    url: str
    enabled: bool = True
    schedule_interval_minutes: int = 60
    # Deprecated M2 no-op, kept for compatibility.
    coverage_report_patterns: list[str] | None = None


class SourceResponse(BaseModel):
    """Response model for a source."""

    id: str
    collection_id: str
    type: str
    url: str
    config: dict
    enabled: bool
    schedule_interval_minutes: int
    next_run_at: datetime | None
    last_run_at: datetime | None
    created_at: datetime
    document_count: int = 0
    deploy_key_fingerprint: str | None = None


class UpdateSourceRequest(BaseModel):
    """Request model for updating a source."""

    enabled: bool | None = None
    schedule_interval_minutes: int | None = None
    max_pages: int | None = None  # For web sources only
    # Deprecated M2 no-op, kept for compatibility.
    coverage_report_patterns: list[str] | None = None


class SetDeployKeyRequest(BaseModel):
    """Request model for setting a deploy key."""

    private_key: str  # PEM-formatted private key


class DeployKeyResponse(BaseModel):
    """Response model for deploy key info."""

    fingerprint: str | None
    has_key: bool


class RotateCoverageIngestTokenResponse(BaseModel):
    """Response for ingest token rotation (raw token returned once)."""

    token: str
    token_preview: str
    created_at: datetime
    rotated_at: datetime | None


class CoverageIngestTokenMetadataResponse(BaseModel):
    """Metadata-only view of source ingest token."""

    has_token: bool
    token_preview: str | None
    created_at: datetime | None
    rotated_at: datetime | None
    last_used_at: datetime | None


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get the current user ID from session or raise 401."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


def validate_github_url(url: str) -> dict:
    """Validate and parse a GitHub repository URL."""
    pattern = r"^https://github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+?)(?:\.git)?$"
    match = re.match(pattern, url)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub URL. Must be https://github.com/owner/repo",
        )

    return {
        "owner": match.group(1),
        "repo": match.group(2),
        "branch": "main",
    }


def validate_web_url(url: str) -> dict:
    """Validate a web URL for crawling."""
    from urllib.parse import urlparse, urlunparse

    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Must start with http:// or https://",
        )

    parsed = urlparse(url)
    path = parsed.path
    if path and not path.endswith("/"):
        last_slash = path.rfind("/")
        path = path[: last_slash + 1] if last_slash > 0 else "/"

    base_url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return {"start_url": url, "base_url": base_url}


def mark_coverage_patterns_deprecated(config: dict, patterns: list[str] | None) -> dict:
    """Keep deprecated coverage field as compatibility marker."""
    next_config = dict(config or {})
    metrics_cfg = dict(next_config.get("metrics") or {})
    metrics_cfg["deprecated"] = True
    metrics_cfg["deprecated_field"] = "coverage_report_patterns"
    if patterns is not None:
        metrics_cfg["coverage_report_patterns_ignored"] = list(patterns)
    next_config["metrics"] = metrics_cfg
    return next_config


def hash_ingest_token(token: str) -> str:
    """Hash ingest token for at-rest storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def make_token_preview(token: str) -> str:
    """Build masked preview value."""
    if len(token) < 10:
        return "********"
    return f"{token[:6]}...{token[-4:]}"


async def _get_collection_with_access(
    db, collection_id: str, user_id: uuid.UUID, require_owner: bool = False
) -> Collection:
    """Get a collection and verify access rights."""
    try:
        coll_uuid = uuid.UUID(collection_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid collection ID") from e

    result = await db.execute(select(Collection).where(Collection.id == coll_uuid))
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    if require_owner:
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can perform this action")
    elif (
        collection.visibility == CollectionVisibility.PRIVATE
        and collection.owner_user_id != user_id
    ):
        result = await db.execute(
            select(CollectionMember)
            .where(CollectionMember.collection_id == collection.id)
            .where(CollectionMember.user_id == user_id)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=403, detail="Access denied to this collection")

    return collection


@router.post("/collections/{collection_id}/sources", response_model=SourceResponse)
async def create_source(
    request: Request, collection_id: str, body: CreateSourceRequest
) -> SourceResponse:
    """Create a new source in a collection."""
    user_id = get_current_user_id(request)

    try:
        source_type = SourceType(body.type)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Invalid type. Must be 'github' or 'web'"
        ) from e

    config = (
        validate_github_url(body.url)
        if source_type == SourceType.GITHUB
        else validate_web_url(body.url)
    )
    if body.coverage_report_patterns is not None:
        config = mark_coverage_patterns_deprecated(config, body.coverage_report_patterns)

    async with get_db_session() as db:
        collection = await _get_collection_with_access(
            db, collection_id, user_id, require_owner=True
        )
        source = Source(
            id=uuid.uuid4(),
            collection_id=collection.id,
            type=source_type,
            url=body.url,
            config=config,
            enabled=body.enabled,
            schedule_interval_minutes=body.schedule_interval_minutes,
            next_run_at=datetime.now(UTC),
        )
        db.add(source)
        await db.flush()

        return SourceResponse(
            id=str(source.id),
            collection_id=str(source.collection_id),
            type=source.type.value,
            url=source.url,
            config=source.config,
            enabled=source.enabled,
            schedule_interval_minutes=source.schedule_interval_minutes,
            next_run_at=source.next_run_at,
            last_run_at=source.last_run_at,
            created_at=source.created_at,
            deploy_key_fingerprint=source.deploy_key_fingerprint,
        )


@router.get("/collections/{collection_id}/sources", response_model=list[SourceResponse])
async def list_sources(request: Request, collection_id: str) -> list[SourceResponse]:
    """List sources in a collection."""
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        collection = await _get_collection_with_access(db, collection_id, user_id)
        doc_count_subq = (
            select(Document.source_id, func.count(Document.id).label("doc_count"))
            .group_by(Document.source_id)
            .subquery()
        )

        result = await db.execute(
            select(Source, func.coalesce(doc_count_subq.c.doc_count, 0).label("doc_count"))
            .outerjoin(doc_count_subq, Source.id == doc_count_subq.c.source_id)
            .where(Source.collection_id == collection.id)
            .order_by(Source.created_at.desc())
        )
        rows = result.all()

        return [
            SourceResponse(
                id=str(source.id),
                collection_id=str(source.collection_id),
                type=source.type.value,
                url=source.url,
                config=source.config,
                enabled=source.enabled,
                schedule_interval_minutes=source.schedule_interval_minutes,
                next_run_at=source.next_run_at,
                last_run_at=source.last_run_at,
                created_at=source.created_at,
                document_count=doc_count,
                deploy_key_fingerprint=source.deploy_key_fingerprint,
            )
            for source, doc_count in rows
        ]


@router.delete("/sources/{source_id}")
async def delete_source(request: Request, source_id: str) -> dict[str, str]:
    """Delete a source."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can delete sources")

        await db.delete(source)
        await db.flush()
        return {"status": "deleted"}


@router.patch("/sources/{source_id}", response_model=SourceResponse)
async def update_source(
    request: Request, source_id: str, body: UpdateSourceRequest
) -> SourceResponse:
    """Update a source's settings."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can update sources")

        if body.enabled is not None:
            source.enabled = body.enabled

        if body.schedule_interval_minutes is not None:
            if body.schedule_interval_minutes < 1:
                raise HTTPException(
                    status_code=400, detail="Schedule interval must be at least 1 minute"
                )
            source.schedule_interval_minutes = body.schedule_interval_minutes

        if body.max_pages is not None:
            if source.type != SourceType.WEB:
                raise HTTPException(
                    status_code=400, detail="max_pages is only supported for web sources"
                )
            if body.max_pages < 1 or body.max_pages > 1000:
                raise HTTPException(status_code=400, detail="max_pages must be between 1 and 1000")
            config = dict(source.config or {})
            config["max_pages"] = body.max_pages
            source.config = config

        if body.coverage_report_patterns is not None:
            source.config = mark_coverage_patterns_deprecated(
                source.config or {}, body.coverage_report_patterns
            )

        await db.flush()

        doc_count_result = await db.execute(
            select(func.count(Document.id)).where(Document.source_id == source.id)
        )
        doc_count = doc_count_result.scalar() or 0

        return SourceResponse(
            id=str(source.id),
            collection_id=str(source.collection_id),
            type=source.type.value,
            url=source.url,
            config=source.config,
            enabled=source.enabled,
            schedule_interval_minutes=source.schedule_interval_minutes,
            next_run_at=source.next_run_at,
            last_run_at=source.last_run_at,
            created_at=source.created_at,
            document_count=doc_count,
            deploy_key_fingerprint=source.deploy_key_fingerprint,
        )


@router.post("/sources/{source_id}/sync-now")
async def sync_now(request: Request, source_id: str) -> dict[str, str]:
    """Trigger a sync for a source (sets next_run_at to now)."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        await _get_collection_with_access(db, str(source.collection_id), user_id)
        next_run = datetime.now(UTC)
        source.next_run_at = next_run
        await db.flush()
        return {"status": "sync_scheduled", "next_run_at": next_run.isoformat()}


@router.post(
    "/sources/{source_id}/metrics/coverage-ingest-token/rotate",
    response_model=RotateCoverageIngestTokenResponse,
)
async def rotate_coverage_ingest_token(
    request: Request, source_id: str
) -> RotateCoverageIngestTokenResponse:
    """Rotate source-scoped CI ingest token (owner only)."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        collection = (
            await db.execute(select(Collection).where(Collection.id == source.collection_id))
        ).scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can rotate ingest tokens")

        raw_token = f"cmi_{secrets.token_urlsafe(36)}"
        digest = hash_ingest_token(raw_token)
        preview = make_token_preview(raw_token)
        now = datetime.now(UTC)

        token_row = (
            await db.execute(
                select(SourceIngestToken).where(SourceIngestToken.source_id == source.id)
            )
        ).scalar_one_or_none()
        if token_row:
            token_row.token_hash = digest
            token_row.token_preview = preview
            token_row.rotated_at = now
        else:
            token_row = SourceIngestToken(
                id=uuid.uuid4(),
                source_id=source.id,
                token_hash=digest,
                token_preview=preview,
                created_at=now,
                rotated_at=now,
            )
            db.add(token_row)

        await db.flush()
        return RotateCoverageIngestTokenResponse(
            token=raw_token,
            token_preview=token_row.token_preview,
            created_at=token_row.created_at,
            rotated_at=token_row.rotated_at,
        )


@router.get(
    "/sources/{source_id}/metrics/coverage-ingest-token",
    response_model=CoverageIngestTokenMetadataResponse,
)
async def get_coverage_ingest_token(
    request: Request, source_id: str
) -> CoverageIngestTokenMetadataResponse:
    """Read source ingest token metadata (owner only)."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        collection = (
            await db.execute(select(Collection).where(Collection.id == source.collection_id))
        ).scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(
                status_code=403, detail="Only the owner can view ingest token metadata"
            )

        token_row = (
            await db.execute(
                select(SourceIngestToken).where(SourceIngestToken.source_id == source.id)
            )
        ).scalar_one_or_none()
        if not token_row:
            return CoverageIngestTokenMetadataResponse(
                has_token=False,
                token_preview=None,
                created_at=None,
                rotated_at=None,
                last_used_at=None,
            )

        return CoverageIngestTokenMetadataResponse(
            has_token=True,
            token_preview=token_row.token_preview,
            created_at=token_row.created_at,
            rotated_at=token_row.rotated_at,
            last_used_at=token_row.last_used_at,
        )


@router.get("/sources/{source_id}/deploy-key", response_model=DeployKeyResponse)
async def get_deploy_key(request: Request, source_id: str) -> DeployKeyResponse:
    """Get deploy key info for a source (fingerprint only, not the key itself)."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        collection = (
            await db.execute(select(Collection).where(Collection.id == source.collection_id))
        ).scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can view deploy key info")

        return DeployKeyResponse(
            fingerprint=source.deploy_key_fingerprint,
            has_key=source.deploy_key_encrypted is not None,
        )


@router.put("/sources/{source_id}/deploy-key", response_model=DeployKeyResponse)
async def set_deploy_key(
    request: Request, source_id: str, body: SetDeployKeyRequest
) -> DeployKeyResponse:
    """Set or update the deploy key for a GitHub source."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    private_key = body.private_key.strip()
    if not validate_ssh_private_key(private_key):
        raise HTTPException(
            status_code=400,
            detail="Invalid SSH private key. Must be a valid PEM-formatted private key without a passphrase.",
        )

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        if source.type != SourceType.GITHUB:
            raise HTTPException(
                status_code=400, detail="Deploy keys are only supported for GitHub sources"
            )

        collection = (
            await db.execute(select(Collection).where(Collection.id == source.collection_id))
        ).scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can set deploy keys")

        source.deploy_key_encrypted = encrypt_token(private_key)
        source.deploy_key_fingerprint = compute_ssh_key_fingerprint(private_key)
        await db.flush()

        return DeployKeyResponse(fingerprint=source.deploy_key_fingerprint, has_key=True)


@router.delete("/sources/{source_id}/deploy-key")
async def delete_deploy_key(request: Request, source_id: str) -> dict[str, str]:
    """Remove the deploy key from a source."""
    user_id = get_current_user_id(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source ID") from e

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        collection = (
            await db.execute(select(Collection).where(Collection.id == source.collection_id))
        ).scalar_one()
        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can delete deploy keys")

        source.deploy_key_encrypted = None
        source.deploy_key_fingerprint = None
        await db.flush()
        return {"status": "deleted"}
