import pytest
from contextmine_core.database import Base
from contextmine_core.models import (
    Collection,
    CollectionVisibility,
    GUIElement,
    GUIScreen,
    Source,
    SourceType,
    User,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def session() -> AsyncSession:  # type: ignore[misc]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with SessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def test_gui_models_persistence_and_relations(session: AsyncSession):
    # 1. Setup required parent entities (User -> Collection -> Source)
    user = User(github_user_id=12345, github_login="testuser")
    session.add(user)
    await session.flush()

    collection = Collection(
        slug="test-collection",
        name="Test Collection",
        visibility=CollectionVisibility.PRIVATE,
        owner_user_id=user.id,
    )
    session.add(collection)
    await session.flush()

    source = Source(collection_id=collection.id, type=SourceType.WEB, url="https://example.com")
    session.add(source)
    await session.flush()

    # 2. Create GUIScreen
    screen = GUIScreen(
        source_id=source.id,
        url_path="/login",
        screenshot_path="/tmp/screens/123.png",
        state_hash="hash123",
    )

    # 3. Create GUIElements
    btn_login = GUIElement(
        role="button",
        name="Login",
        bounding_box={"x": 10.0, "y": 20.0, "width": 100.0, "height": 40.0},
    )

    input_user = GUIElement(
        role="textbox",
        name="Username",
        bounding_box={"x": 10.0, "y": 70.0, "width": 200.0, "height": 30.0},
    )

    screen.elements.extend([btn_login, input_user])

    session.add(screen)
    await session.commit()

    # 4. Query back with JOIN/selectinload
    stmt = (
        select(GUIScreen).options(selectinload(GUIScreen.elements)).where(GUIScreen.id == screen.id)
    )
    result = await session.execute(stmt)
    retrieved_screen = result.scalar_one()

    assert retrieved_screen is not None
    assert retrieved_screen.url_path == "/login"
    assert retrieved_screen.state_hash == "hash123"
    assert len(retrieved_screen.elements) == 2

    roles = {el.role for el in retrieved_screen.elements}
    assert "button" in roles
    assert "textbox" in roles

    # Validate bounding box
    button_el = next(el for el in retrieved_screen.elements if el.role == "button")
    assert button_el.bounding_box["x"] == 10.0
    assert button_el.bounding_box["width"] == 100.0
