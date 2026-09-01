"""Shared collection access decisions for REST, MCP, and research paths."""

import uuid

from contextmine_core.models import Collection, CollectionMember, CollectionVisibility
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession


def accessible_collections_clause(user_id: uuid.UUID | None):  # noqa: ANN201
    """Build the shared SQL predicate for collections visible to an identity."""
    if user_id is None:
        return Collection.visibility == CollectionVisibility.GLOBAL
    return or_(
        Collection.visibility == CollectionVisibility.GLOBAL,
        Collection.owner_user_id == user_id,
        Collection.id.in_(
            select(CollectionMember.collection_id).where(CollectionMember.user_id == user_id)
        ),
    )


async def user_can_access_collection(
    session: AsyncSession,
    collection: Collection,
    user_id: uuid.UUID | None,
    *,
    allow_global: bool = True,
) -> bool:
    """Return whether the current collection visibility and membership permit access."""
    if allow_global and collection.visibility == CollectionVisibility.GLOBAL:
        return True
    if user_id is None:
        return False
    if collection.owner_user_id == user_id:
        return True
    membership = (
        await session.execute(
            select(CollectionMember).where(
                CollectionMember.collection_id == collection.id,
                CollectionMember.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    return membership is not None


async def get_accessible_collection_ids(
    session: AsyncSession, user_id: uuid.UUID | None
) -> list[uuid.UUID]:
    """Return global, owned, and member collection IDs for a local identity."""
    statement = select(Collection.id).where(accessible_collections_clause(user_id))
    return list((await session.execute(statement)).scalars().all())
