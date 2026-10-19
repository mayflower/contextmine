"""Shared external identity mapping helpers."""

import uuid
from collections.abc import Mapping
from typing import Any

from contextmine_core.models import User
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


def github_profile_from_verified_claims(claims: Mapping[str, Any]) -> dict[str, Any]:
    """Build GitHub profile fields carried by FastMCP's verified token claims."""
    subject = claims.get("sub")
    login = claims.get("login")
    if not isinstance(subject, (str, int)) or not str(subject).isdigit():
        raise ValueError("Verified GitHub claims are missing a valid subject")
    if not isinstance(login, str) or not login.strip():
        raise ValueError("Verified GitHub claims are missing a login")

    return {
        "id": int(subject),
        "login": login,
        "name": claims.get("name") if isinstance(claims.get("name"), str) else None,
        "avatar_url": (
            claims.get("avatar_url") if isinstance(claims.get("avatar_url"), str) else None
        ),
    }


async def upsert_github_user(session: AsyncSession, github_profile: Mapping[str, Any]) -> User:
    """Map a GitHub profile to the local user record used by REST and MCP."""
    github_user_id = github_profile.get("id")
    github_login = github_profile.get("login")
    if not isinstance(github_user_id, int) or not isinstance(github_login, str):
        raise ValueError("GitHub profile is missing id or login")

    user = (
        await session.execute(select(User).where(User.github_user_id == github_user_id))
    ).scalar_one_or_none()
    if user is None:
        user = User(
            id=uuid.uuid4(),
            github_user_id=github_user_id,
            github_login=github_login,
            name=github_profile.get("name"),
            avatar_url=github_profile.get("avatar_url"),
        )
        session.add(user)
    else:
        user.github_login = github_login
        user.name = github_profile.get("name")
        user.avatar_url = github_profile.get("avatar_url")
    return user


# GitHub user ids start at 1, so 0 can never collide with a real account.
LOCAL_GITHUB_USER_ID = 0


async def upsert_local_user(session: AsyncSession) -> User:
    """Return the single user that AUTH_MODE=local signs everyone in as."""
    return await upsert_github_user(
        session,
        {"id": LOCAL_GITHUB_USER_ID, "login": "local", "name": "Local user"},
    )
