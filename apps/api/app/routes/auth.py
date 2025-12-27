"""Authentication routes for GitHub OAuth."""

import uuid

from contextmine_core import (
    CollectionInvite,
    CollectionMember,
    OAuthToken,
    User,
    encrypt_token,
    exchange_code_for_token,
    generate_state,
    get_github_authorize_url,
    get_github_user,
    get_settings,
)
from contextmine_core import (
    get_session as get_db_session,
)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.middleware import clear_session, get_session, set_session

router = APIRouter(prefix="/auth", tags=["auth"])


class UserResponse(BaseModel):
    """Response model for user data."""

    id: str
    github_login: str
    name: str | None
    avatar_url: str | None


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    """Initiate GitHub OAuth login flow."""
    settings = get_settings()
    if not settings.github_client_id:
        raise HTTPException(status_code=500, detail="GitHub OAuth is not configured")

    # Generate and store state for CSRF protection
    state = generate_state()
    session = get_session(request)
    session["oauth_state"] = state
    set_session(request, session)

    # Redirect to GitHub
    authorize_url = get_github_authorize_url(state)
    return RedirectResponse(url=authorize_url, status_code=302)


@router.get("/callback")
async def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Handle GitHub OAuth callback."""
    frontend_url = "http://localhost:5173"  # Frontend URL for redirect

    # Check for OAuth errors
    if error:
        return RedirectResponse(url=f"{frontend_url}?error={error}", status_code=302)

    if not code or not state:
        return RedirectResponse(
            url=f"{frontend_url}?error=missing_params", status_code=302
        )

    # Verify state matches
    session = get_session(request)
    stored_state = session.get("oauth_state")
    if not stored_state or stored_state != state:
        return RedirectResponse(
            url=f"{frontend_url}?error=invalid_state", status_code=302
        )

    try:
        # Exchange code for access token
        access_token = await exchange_code_for_token(code)

        # Fetch user profile from GitHub
        github_user = await get_github_user(access_token)

        # Upsert user in database
        async with get_db_session() as db:
            # Check if user exists
            result = await db.execute(
                select(User).where(User.github_user_id == github_user["id"])
            )
            user = result.scalar_one_or_none()

            if user:
                # Update existing user
                user.github_login = github_user["login"]
                user.name = github_user.get("name")
                user.avatar_url = github_user.get("avatar_url")
            else:
                # Create new user
                user = User(
                    id=uuid.uuid4(),
                    github_user_id=github_user["id"],
                    github_login=github_user["login"],
                    name=github_user.get("name"),
                    avatar_url=github_user.get("avatar_url"),
                )
                db.add(user)

            # Store encrypted OAuth token
            encrypted_token = encrypt_token(access_token)

            # Remove old tokens for this user/provider
            await db.execute(
                select(OAuthToken)
                .where(OAuthToken.user_id == user.id)
                .where(OAuthToken.provider == "github")
            )
            # Delete existing tokens
            await db.execute(
                delete(OAuthToken)
                .where(OAuthToken.user_id == user.id)
                .where(OAuthToken.provider == "github")
            )

            # Create new token
            oauth_token = OAuthToken(
                id=uuid.uuid4(),
                user_id=user.id,
                provider="github",
                access_token_encrypted=encrypted_token,
            )
            db.add(oauth_token)

            await db.flush()

            # Auto-accept pending invites for this user's github_login
            invite_result = await db.execute(
                select(CollectionInvite).where(
                    CollectionInvite.github_login == user.github_login
                )
            )
            pending_invites = invite_result.scalars().all()

            for invite in pending_invites:
                # Check if already a member (shouldn't happen but be safe)
                member_check = await db.execute(
                    select(CollectionMember)
                    .where(CollectionMember.collection_id == invite.collection_id)
                    .where(CollectionMember.user_id == user.id)
                )
                if not member_check.scalar_one_or_none():
                    # Add as member
                    member = CollectionMember(
                        collection_id=invite.collection_id,
                        user_id=user.id,
                    )
                    db.add(member)

                # Delete the invite
                await db.delete(invite)

            await db.flush()

            # Store user ID in session
            session["user_id"] = str(user.id)
            del session["oauth_state"]
            set_session(request, session)

        return RedirectResponse(url=frontend_url, status_code=302)

    except Exception as e:
        return RedirectResponse(
            url=f"{frontend_url}?error=auth_failed&detail={str(e)}", status_code=302
        )


@router.get("/logout")
async def logout(request: Request) -> dict[str, str]:
    """Log out the current user."""
    clear_session(request)
    return {"status": "logged_out"}


@router.get("/me")
async def get_current_user(request: Request) -> UserResponse:
    """Get the current authenticated user."""
    session = get_session(request)
    user_id = session.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async with get_db_session() as db:
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()

        if not user:
            clear_session(request)
            raise HTTPException(status_code=401, detail="User not found")

        return UserResponse(
            id=str(user.id),
            github_login=user.github_login,
            name=user.name,
            avatar_url=user.avatar_url,
        )
