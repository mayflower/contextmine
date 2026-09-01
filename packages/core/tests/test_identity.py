"""Tests for the shared GitHub-to-local identity mapping."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from contextmine_core.identity import (
    github_profile_from_verified_claims,
    upsert_github_user,
)


def test_github_profile_uses_verified_claims() -> None:
    profile = github_profile_from_verified_claims(
        {
            "sub": "1234",
            "login": "octocat",
            "name": "The Octocat",
            "avatar_url": "https://avatars.example/octocat",
        }
    )

    assert profile == {
        "id": 1234,
        "login": "octocat",
        "name": "The Octocat",
        "avatar_url": "https://avatars.example/octocat",
    }


@pytest.mark.parametrize("claims", [{}, {"sub": "not-a-number", "login": "octocat"}])
def test_github_profile_rejects_incomplete_claims(claims: dict) -> None:
    with pytest.raises(ValueError):
        github_profile_from_verified_claims(claims)


@pytest.mark.anyio
async def test_upsert_github_user_creates_local_identity() -> None:
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)

    user = await upsert_github_user(
        session,
        {"id": 1234, "login": "octocat", "name": None, "avatar_url": None},
    )

    assert user.id
    assert user.github_user_id == 1234
    assert user.github_login == "octocat"
    session.add.assert_called_once_with(user)


@pytest.mark.anyio
async def test_upsert_github_user_updates_existing_identity() -> None:
    user = MagicMock(id=uuid.uuid4())
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    session.execute = AsyncMock(return_value=result)

    mapped = await upsert_github_user(
        session,
        {
            "id": 1234,
            "login": "new-login",
            "name": "New Name",
            "avatar_url": "https://avatars.example/new",
        },
    )

    assert mapped is user
    assert user.github_login == "new-login"
    assert user.name == "New Name"
    assert user.avatar_url == "https://avatars.example/new"
    session.add.assert_not_called()
