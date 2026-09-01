"""Public FastMCP contracts for architecture tools."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp_domains.architecture import (
    Arc42Result,
    ArchitectureIntentResult,
    create_architecture_intent,
    get_arc42,
)
from app.mcp_server import mcp
from contextmine_core.architecture_intents import IntentAction
from fastmcp.exceptions import ToolError

ARCHITECTURE_TOOL_NAMES = {
    "create_architecture_intent",
    "approve_architecture_intent",
    "get_arc42",
    "arc42_drift_report",
    "list_ports_adapters",
}


def _database_patch(database):  # noqa: ANN001, ANN202
    context = patch("app.mcp_domains.architecture.get_db_session")
    session_factory = context.start()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=database)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.mark.anyio
async def test_architecture_tools_publish_native_bounded_contracts() -> None:
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    assert ARCHITECTURE_TOOL_NAMES.issubset(tools)
    for name in ARCHITECTURE_TOOL_NAMES:
        assert tools[name].output_schema is not None
        assert tools[name].output_schema.get("x-fastmcp-wrap-result") is not True
    assert tools["create_architecture_intent"].parameters["properties"]["action"]["enum"] == [
        item.value for item in IntentAction
    ]
    assert tools["arc42_drift_report"].parameters["properties"]["limit"]["maximum"] == 500
    assert tools["list_ports_adapters"].parameters["properties"]["limit"]["maximum"] == 500


@pytest.mark.anyio
async def test_create_intent_delegates_to_existing_twin_core() -> None:
    user_id = uuid.uuid4()
    scenario = SimpleNamespace(id=uuid.uuid4(), collection_id=uuid.uuid4(), version=4)
    collection = SimpleNamespace(id=scenario.collection_id, owner_user_id=user_id)
    intent = SimpleNamespace(
        id=uuid.uuid4(),
        status=SimpleNamespace(value="executed"),
        risk_level=SimpleNamespace(value="low"),
        requires_approval=False,
    )
    scenario_result = MagicMock()
    scenario_result.scalar_one_or_none.return_value = scenario
    collection_result = MagicMock()
    collection_result.scalar_one_or_none.return_value = collection
    database = AsyncMock()
    database.execute = AsyncMock(side_effect=[scenario_result, collection_result])
    context = _database_patch(database)
    try:
        with (
            patch("app.mcp_domains.architecture.get_current_user_id", return_value=user_id),
            patch(
                "contextmine_core.twin.submit_intent",
                new_callable=AsyncMock,
                return_value=intent,
            ) as submit,
        ):
            result = await create_architecture_intent(
                scenario_id=str(scenario.id),
                action=IntentAction.EXTRACT_DOMAIN,
                target_type="context",
                target_id="payments",
                expected_scenario_version=4,
            )
    finally:
        context.stop()

    assert isinstance(result, ArchitectureIntentResult)
    assert result.intent_id == str(intent.id)
    assert submit.await_args.kwargs["intent"].action == IntentAction.EXTRACT_DOMAIN
    database.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_invalid_intent_json_and_disabled_arc42_use_tool_errors() -> None:
    with pytest.raises(ToolError, match="valid JSON object"):
        await create_architecture_intent(
            scenario_id=str(uuid.uuid4()),
            action=IntentAction.EXTRACT_DOMAIN,
            target_type="context",
            target_id="payments",
            expected_scenario_version=1,
            params_json="not-json",
        )
    with (
        patch(
            "app.mcp_domains.architecture.get_settings",
            return_value=SimpleNamespace(arch_docs_enabled=False),
        ),
        pytest.raises(ToolError, match="Architecture docs are disabled"),
    ):
        await get_arc42()


@pytest.mark.anyio
async def test_cached_arc42_is_structured_and_repeatable() -> None:
    collection = SimpleNamespace(id=uuid.uuid4())
    scenario = SimpleNamespace(id=uuid.uuid4(), name="As Is")
    artifact = SimpleNamespace(
        id=uuid.uuid4(),
        content="# arc42\n\nArchitecture",
        meta={
            "facts_hash": "abc",
            "warnings": [],
            "sections": {"3_system_scope_and_context": "Architecture context"},
        },
    )
    collection_result = MagicMock()
    collection_result.scalar_one_or_none.return_value = collection
    scenario_result = MagicMock()
    scenario_result.scalar_one_or_none.return_value = scenario
    artifact_result = MagicMock()
    artifact_result.scalar_one_or_none.return_value = artifact
    database = AsyncMock()
    database.execute = AsyncMock(
        side_effect=[
            collection_result,
            scenario_result,
            artifact_result,
            collection_result,
            scenario_result,
            artifact_result,
        ]
    )
    context = _database_patch(database)
    settings = SimpleNamespace(arch_docs_enabled=True)
    try:
        with (
            patch("app.mcp_domains.architecture.get_settings", return_value=settings),
            patch("app.mcp_domains.architecture.get_current_user_id", return_value=None),
        ):
            first = await get_arc42(
                collection_id=str(collection.id),
                scenario_id=str(scenario.id),
                section="context",
            )
            second = await get_arc42(
                collection_id=str(collection.id),
                scenario_id=str(scenario.id),
                section="context",
            )
    finally:
        context.stop()

    assert isinstance(first, Arc42Result)
    assert first == second
    assert first.generated is True
    assert first.cached is True
    assert first.sections == {"3_system_scope_and_context": "Architecture context"}
