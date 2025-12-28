"""Collections management routes."""

import uuid
from datetime import datetime

from contextmine_core import (
    Collection,
    CollectionInvite,
    CollectionMember,
    CollectionVisibility,
    User,
)
from contextmine_core import (
    get_session as get_db_session,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import delete, or_, select

from app.middleware import get_session

router = APIRouter(prefix="/collections", tags=["collections"])


class CreateCollectionRequest(BaseModel):
    """Request model for creating a collection."""

    name: str
    slug: str
    visibility: str  # "global" or "private"


class CollectionResponse(BaseModel):
    """Response model for a collection."""

    id: str
    slug: str
    name: str
    visibility: str
    owner_id: str
    owner_github_login: str
    created_at: datetime
    is_owner: bool
    member_count: int


class ShareRequest(BaseModel):
    """Request model for sharing a collection."""

    github_login: str


class MemberResponse(BaseModel):
    """Response model for a collection member."""

    user_id: str
    github_login: str
    name: str | None
    avatar_url: str | None
    is_owner: bool


class InviteResponse(BaseModel):
    """Response model for a collection invite."""

    github_login: str
    created_at: datetime


def get_current_user_id(request: Request) -> uuid.UUID:
    """Get the current user ID from session or raise 401."""
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


@router.post("", response_model=CollectionResponse)
async def create_collection(request: Request, body: CreateCollectionRequest) -> CollectionResponse:
    """Create a new collection."""
    user_id = get_current_user_id(request)

    # Validate visibility
    try:
        visibility = CollectionVisibility(body.visibility)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail="Invalid visibility. Must be 'global' or 'private'"
        ) from e

    async with get_db_session() as db:
        # Check if slug is already taken
        result = await db.execute(select(Collection).where(Collection.slug == body.slug))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Slug already exists")

        # Get owner info
        result = await db.execute(select(User).where(User.id == user_id))
        owner = result.scalar_one_or_none()
        if not owner:
            raise HTTPException(status_code=401, detail="User not found")

        collection = Collection(
            id=uuid.uuid4(),
            slug=body.slug,
            name=body.name,
            visibility=visibility,
            owner_user_id=user_id,
        )
        db.add(collection)
        await db.flush()

        return CollectionResponse(
            id=str(collection.id),
            slug=collection.slug,
            name=collection.name,
            visibility=collection.visibility.value,
            owner_id=str(collection.owner_user_id),
            owner_github_login=owner.github_login,
            created_at=collection.created_at,
            is_owner=True,
            member_count=0,
        )


@router.get("", response_model=list[CollectionResponse])
async def list_collections(request: Request) -> list[CollectionResponse]:
    """List collections visible to the current user.

    Returns all global collections + private collections where user is owner or member.
    """
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        # Get collections: global OR (private AND (owner OR member))
        result = await db.execute(
            select(Collection, User)
            .join(User, Collection.owner_user_id == User.id)
            .where(
                or_(
                    Collection.visibility == CollectionVisibility.GLOBAL,
                    Collection.owner_user_id == user_id,
                    Collection.id.in_(
                        select(CollectionMember.collection_id).where(
                            CollectionMember.user_id == user_id
                        )
                    ),
                )
            )
            .order_by(Collection.created_at.desc())
        )
        rows = result.all()

        collections = []
        for collection, owner in rows:
            # Count members
            member_result = await db.execute(
                select(CollectionMember).where(CollectionMember.collection_id == collection.id)
            )
            member_count = len(member_result.scalars().all())

            collections.append(
                CollectionResponse(
                    id=str(collection.id),
                    slug=collection.slug,
                    name=collection.name,
                    visibility=collection.visibility.value,
                    owner_id=str(collection.owner_user_id),
                    owner_github_login=owner.github_login,
                    created_at=collection.created_at,
                    is_owner=collection.owner_user_id == user_id,
                    member_count=member_count,
                )
            )

        return collections


@router.get("/{collection_id}/members", response_model=list[MemberResponse])
async def list_members(request: Request, collection_id: str) -> list[MemberResponse]:
    """List members of a collection."""
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        # Get collection and check access
        collection = await _get_collection_with_access(db, collection_id, user_id)

        # Get owner
        result = await db.execute(select(User).where(User.id == collection.owner_user_id))
        owner = result.scalar_one()

        members = [
            MemberResponse(
                user_id=str(owner.id),
                github_login=owner.github_login,
                name=owner.name,
                avatar_url=owner.avatar_url,
                is_owner=True,
            )
        ]

        # Get other members
        result = await db.execute(
            select(CollectionMember, User)
            .join(User, CollectionMember.user_id == User.id)
            .where(CollectionMember.collection_id == collection.id)
        )
        for _membership, user in result.all():
            members.append(
                MemberResponse(
                    user_id=str(user.id),
                    github_login=user.github_login,
                    name=user.name,
                    avatar_url=user.avatar_url,
                    is_owner=False,
                )
            )

        return members


@router.get("/{collection_id}/invites", response_model=list[InviteResponse])
async def list_invites(request: Request, collection_id: str) -> list[InviteResponse]:
    """List pending invites for a collection (owner only)."""
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        # Get collection and check ownership
        collection = await _get_collection_with_access(
            db, collection_id, user_id, require_owner=True
        )

        result = await db.execute(
            select(CollectionInvite).where(CollectionInvite.collection_id == collection.id)
        )
        invites = result.scalars().all()

        return [
            InviteResponse(
                github_login=invite.github_login,
                created_at=invite.created_at,
            )
            for invite in invites
        ]


@router.post("/{collection_id}/share")
async def share_collection(
    request: Request, collection_id: str, body: ShareRequest
) -> dict[str, str]:
    """Share a collection with a GitHub user.

    If user exists, add as member. If not, create invite.
    """
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        # Get collection and check ownership
        collection = await _get_collection_with_access(
            db, collection_id, user_id, require_owner=True
        )

        # Check if user exists
        result = await db.execute(select(User).where(User.github_login == body.github_login))
        target_user = result.scalar_one_or_none()

        if target_user:
            # Check if already owner
            if target_user.id == collection.owner_user_id:
                raise HTTPException(status_code=400, detail="Cannot share with collection owner")

            # Check if already a member
            result = await db.execute(
                select(CollectionMember)
                .where(CollectionMember.collection_id == collection.id)
                .where(CollectionMember.user_id == target_user.id)
            )
            if result.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="User is already a member")

            # Add as member
            member = CollectionMember(
                collection_id=collection.id,
                user_id=target_user.id,
            )
            db.add(member)
            await db.flush()

            return {"status": "member_added", "github_login": body.github_login}
        else:
            # Check if invite already exists
            result = await db.execute(
                select(CollectionInvite)
                .where(CollectionInvite.collection_id == collection.id)
                .where(CollectionInvite.github_login == body.github_login)
            )
            if result.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Invite already exists for this user")

            # Create invite
            invite = CollectionInvite(
                id=uuid.uuid4(),
                collection_id=collection.id,
                github_login=body.github_login,
            )
            db.add(invite)
            await db.flush()

            return {"status": "invite_created", "github_login": body.github_login}


@router.delete("/{collection_id}")
async def delete_collection(request: Request, collection_id: str) -> dict[str, str]:
    """Delete a collection and all its sources/documents/chunks.

    Only the owner can delete a collection.
    """
    from contextmine_core import Chunk, Document, Source

    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        # Get collection and check ownership
        collection = await _get_collection_with_access(
            db, collection_id, user_id, require_owner=True
        )

        # Delete in order: chunks -> documents -> sources -> invites -> members -> collection
        # Get all sources for this collection
        result = await db.execute(select(Source).where(Source.collection_id == collection.id))
        sources = result.scalars().all()

        for source in sources:
            # Delete chunks for all documents in this source
            result = await db.execute(select(Document).where(Document.source_id == source.id))
            documents = result.scalars().all()

            for doc in documents:
                await db.execute(delete(Chunk).where(Chunk.document_id == doc.id))

            # Delete documents
            await db.execute(delete(Document).where(Document.source_id == source.id))

        # Delete sources
        await db.execute(delete(Source).where(Source.collection_id == collection.id))

        # Delete invites
        await db.execute(
            delete(CollectionInvite).where(CollectionInvite.collection_id == collection.id)
        )

        # Delete members
        await db.execute(
            delete(CollectionMember).where(CollectionMember.collection_id == collection.id)
        )

        # Delete collection
        await db.delete(collection)
        await db.flush()

        return {"status": "deleted", "collection_id": collection_id}


@router.delete("/{collection_id}/share/{identifier}")
async def unshare_collection(
    request: Request, collection_id: str, identifier: str
) -> dict[str, str]:
    """Remove a member or invite from a collection.

    Identifier can be a user_id (UUID) or github_login.
    """
    user_id = get_current_user_id(request)

    async with get_db_session() as db:
        # Get collection and check ownership
        collection = await _get_collection_with_access(
            db, collection_id, user_id, require_owner=True
        )

        # Try to parse as UUID (user_id)
        try:
            target_user_id = uuid.UUID(identifier)
            # Remove member
            result = await db.execute(
                select(CollectionMember)
                .where(CollectionMember.collection_id == collection.id)
                .where(CollectionMember.user_id == target_user_id)
            )
            member = result.scalar_one_or_none()
            if member:
                await db.delete(member)
                await db.flush()
                return {"status": "member_removed"}
        except ValueError:
            pass

        # Try as github_login - first check members
        result = await db.execute(
            select(CollectionMember, User)
            .join(User, CollectionMember.user_id == User.id)
            .where(CollectionMember.collection_id == collection.id)
            .where(User.github_login == identifier)
        )
        row = result.first()
        if row:
            member, _ = row
            await db.delete(member)
            await db.flush()
            return {"status": "member_removed"}

        # Check invites
        result = await db.execute(
            select(CollectionInvite)
            .where(CollectionInvite.collection_id == collection.id)
            .where(CollectionInvite.github_login == identifier)
        )
        invite = result.scalar_one_or_none()
        if invite:
            await db.delete(invite)
            await db.flush()
            return {"status": "invite_removed"}

        raise HTTPException(status_code=404, detail="Member or invite not found")


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
