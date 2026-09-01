"""Behavior tests for shared collection access decisions."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from contextmine_core.access import user_can_access_collection
from contextmine_core.models import CollectionVisibility


def _collection(*, owner_id: uuid.UUID, visibility: CollectionVisibility) -> MagicMock:
    return MagicMock(
        id=uuid.uuid4(),
        owner_user_id=owner_id,
        visibility=visibility,
    )


@pytest.mark.anyio
async def test_global_collection_is_accessible_without_local_identity() -> None:
    session = MagicMock(execute=AsyncMock())
    collection = _collection(owner_id=uuid.uuid4(), visibility=CollectionVisibility.GLOBAL)

    assert await user_can_access_collection(session, collection, None)
    session.execute.assert_not_awaited()


@pytest.mark.anyio
async def test_private_collection_requires_owner_or_member() -> None:
    owner_id = uuid.uuid4()
    member_id = uuid.uuid4()
    collection = _collection(owner_id=owner_id, visibility=CollectionVisibility.PRIVATE)
    member_result = MagicMock()
    member_result.scalar_one_or_none.return_value = MagicMock()
    session = MagicMock(execute=AsyncMock(return_value=member_result))

    assert await user_can_access_collection(session, collection, owner_id)
    session.execute.assert_not_awaited()
    assert await user_can_access_collection(session, collection, member_id)


@pytest.mark.anyio
async def test_private_collection_denies_non_member() -> None:
    collection = _collection(owner_id=uuid.uuid4(), visibility=CollectionVisibility.PRIVATE)
    member_result = MagicMock()
    member_result.scalar_one_or_none.return_value = None
    session = MagicMock(execute=AsyncMock(return_value=member_result))

    assert not await user_can_access_collection(session, collection, uuid.uuid4())
