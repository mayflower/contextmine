"""Sources management routes."""

import re
import uuid
from datetime import UTC, datetime

from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    Document,
    Source,
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


class SetDeployKeyRequest(BaseModel):
    """Request model for setting a deploy key."""

    private_key: str  # PEM-formatted private key


class DeployKeyResponse(BaseModel):
    """Response model for deploy key info."""

    fingerprint: str | None
    has_key: bool


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get the current user ID from session or raise 401."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


def validate_github_url(url: str) -> dict:
    """Validate and parse a GitHub repository URL.

    Returns config dict with owner, repo, and branch.
    """
    # Match https://github.com/owner/repo or https://github.com/owner/repo.git
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
        "branch": "main",  # Default branch
    }


def validate_web_url(url: str) -> dict:
    """Validate a web URL for crawling.

    Returns config dict with start_url and base_url.
    - start_url: The URL to begin crawling from (user's input)
    - base_url: The path prefix for scoping (derived from start_url)

    The base_url is derived by taking the parent directory of the start_url.
    For example:
    - https://example.com/docs/intro → base_url = https://example.com/docs/
    - https://example.com/docs/ → base_url = https://example.com/docs/
    """
    from urllib.parse import urlparse, urlunparse

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Must start with http:// or https://",
        )

    # Parse URL to derive base_url
    parsed = urlparse(url)
    path = parsed.path

    # Derive base path: if path doesn't end with /, strip the last component
    if path and not path.endswith("/"):
        # Strip last path component (the "page")
        last_slash = path.rfind("/")
        path = path[: last_slash + 1] if last_slash > 0 else "/"

    # Reconstruct base_url with derived path
    base_url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))

    return {
        "start_url": url,  # Original URL to start crawling from
        "base_url": base_url,  # Derived path prefix for scoping
    }


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
    else:
        # Check access: global, owner, or member
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
                raise HTTPException(status_code=403, detail="Access denied to this collection")

    return collection


@router.post("/collections/{collection_id}/sources", response_model=SourceResponse)
async def create_source(
    request: Request, collection_id: str, body: CreateSourceRequest
) -> SourceResponse:
    """Create a new source in a collection."""
    user_id = get_current_user_id(request)

    # Validate source type
    try:
        source_type = SourceType(body.type)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Invalid type. Must be 'github' or 'web'"
        ) from e

    # Validate URL and get config
    if source_type == SourceType.GITHUB:
        config = validate_github_url(body.url)
    else:
        config = validate_web_url(body.url)

    async with get_db_session() as db:
        # Check collection access (only owner can add sources)
        collection = await _get_collection_with_access(
            db, collection_id, user_id, require_owner=True
        )

        # Create source
        source = Source(
            id=uuid.uuid4(),
            collection_id=collection.id,
            type=source_type,
            url=body.url,
            config=config,
            enabled=body.enabled,
            schedule_interval_minutes=body.schedule_interval_minutes,
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
        # Check collection access
        collection = await _get_collection_with_access(db, collection_id, user_id)

        # Get sources with document counts
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
        # Get source
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check collection ownership
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
        # Get source
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check collection ownership
        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()

        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can update sources")

        # Update fields if provided
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
            config = source.config or {}
            config["max_pages"] = body.max_pages
            source.config = config

        await db.flush()

        # Get document count
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
        # Get source
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check collection access (owner or member can trigger sync)
        await _get_collection_with_access(db, str(source.collection_id), user_id)

        # Set next_run_at to now
        next_run = datetime.now(UTC)
        source.next_run_at = next_run
        await db.flush()

        return {"status": "sync_scheduled", "next_run_at": next_run.isoformat()}


@router.get("/sources/{source_id}/deploy-key", response_model=DeployKeyResponse)
async def get_deploy_key(request: Request, source_id: str) -> DeployKeyResponse:
    """Get deploy key info for a source (fingerprint only, not the key itself)."""
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

        # Check collection ownership (only owner can view deploy key info)
        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()

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

    # Validate the private key format
    private_key = body.private_key.strip()
    if not validate_ssh_private_key(private_key):
        raise HTTPException(
            status_code=400,
            detail="Invalid SSH private key. Must be a valid PEM-formatted private key without a passphrase.",
        )

    async with get_db_session() as db:
        # Get source
        result = await db.execute(select(Source).where(Source.id == src_uuid))
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Only GitHub sources can have deploy keys
        if source.type != SourceType.GITHUB:
            raise HTTPException(
                status_code=400, detail="Deploy keys are only supported for GitHub sources"
            )

        # Check collection ownership
        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()

        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can set deploy keys")

        # Compute fingerprint and encrypt the key
        fingerprint = compute_ssh_key_fingerprint(private_key)
        encrypted_key = encrypt_token(private_key)

        # Update source
        source.deploy_key_encrypted = encrypted_key
        source.deploy_key_fingerprint = fingerprint
        await db.flush()

        return DeployKeyResponse(
            fingerprint=fingerprint,
            has_key=True,
        )


@router.delete("/sources/{source_id}/deploy-key")
async def delete_deploy_key(request: Request, source_id: str) -> dict[str, str]:
    """Remove the deploy key from a source."""
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

        # Check collection ownership
        result = await db.execute(select(Collection).where(Collection.id == source.collection_id))
        collection = result.scalar_one()

        if collection.owner_user_id != user_id:
            raise HTTPException(status_code=403, detail="Only the owner can delete deploy keys")

        # Clear deploy key
        source.deploy_key_encrypted = None
        source.deploy_key_fingerprint = None
        await db.flush()

        return {"status": "deleted"}
