"""Tests for ERM twin view route."""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_core.models import KnowledgeEdgeKind
from httpx import AsyncClient


@pytest.mark.anyio
class TestTwinErmView:
    async def test_erm_view_requires_auth(self, client: AsyncClient) -> None:
        response = await client.get("/api/twin/collections/some-id/views/erm")
        assert response.status_code == 401

    @patch("app.routes.twin._resolve_view_scenario", new_callable=AsyncMock)
    @patch("app.routes.twin._ensure_member", new_callable=AsyncMock)
    @patch("app.routes.twin.get_db_session")
    @patch("app.routes.twin.get_session")
    async def test_erm_view_returns_tables_foreign_keys_and_mermaid(
        self,
        mock_get_session: Any,
        mock_db_session_factory: Any,
        _mock_ensure_member: Any,
        mock_resolve_view_scenario: Any,
        client: AsyncClient,
    ) -> None:
        collection_id = uuid.uuid4()
        scenario_id = uuid.uuid4()
        mock_get_session.return_value = {"user_id": str(uuid.uuid4())}

        scenario = MagicMock()
        scenario.id = scenario_id
        scenario.collection_id = collection_id
        scenario.name = "AS-IS"
        scenario.version = 1
        scenario.is_as_is = True
        scenario.base_scenario_id = None
        mock_resolve_view_scenario.return_value = scenario

        table_node = MagicMock()
        table_node.id = uuid.uuid4()
        table_node.natural_key = "db:users"
        table_node.name = "users"
        table_node.description = None
        table_node.meta = {"column_count": 2, "primary_keys": ["id"]}

        id_column = MagicMock()
        id_column.id = uuid.uuid4()
        id_column.natural_key = "db:users.id"
        id_column.name = "id"
        id_column.meta = {
            "table": "users",
            "type": "UUID",
            "nullable": False,
            "primary_key": True,
            "foreign_key": None,
        }

        account_id_column = MagicMock()
        account_id_column.id = uuid.uuid4()
        account_id_column.natural_key = "db:users.account_id"
        account_id_column.name = "account_id"
        account_id_column.meta = {
            "table": "users",
            "type": "UUID",
            "nullable": False,
            "primary_key": False,
            "foreign_key": "accounts.id",
        }

        account_pk_column = MagicMock()
        account_pk_column.id = uuid.uuid4()
        account_pk_column.natural_key = "db:accounts.id"
        account_pk_column.name = "id"
        account_pk_column.meta = {
            "table": "accounts",
            "type": "UUID",
            "nullable": False,
            "primary_key": True,
            "foreign_key": None,
        }

        table_has_id = MagicMock()
        table_has_id.kind = KnowledgeEdgeKind.TABLE_HAS_COLUMN
        table_has_id.source_node_id = table_node.id
        table_has_id.target_node_id = id_column.id
        table_has_id.meta = {}

        table_has_account_id = MagicMock()
        table_has_account_id.kind = KnowledgeEdgeKind.TABLE_HAS_COLUMN
        table_has_account_id.source_node_id = table_node.id
        table_has_account_id.target_node_id = account_id_column.id
        table_has_account_id.meta = {}

        fk_edge = MagicMock()
        fk_edge.id = uuid.uuid4()
        fk_edge.kind = KnowledgeEdgeKind.COLUMN_FK_TO_COLUMN
        fk_edge.source_node_id = account_id_column.id
        fk_edge.target_node_id = account_pk_column.id
        fk_edge.meta = {"fk_name": "fk_users_account"}

        erd_artifact = MagicMock()
        erd_artifact.id = uuid.uuid4()
        erd_artifact.name = "Database ERD"
        erd_artifact.content = "erDiagram\n    USERS {\n        uuid id PK\n    }"
        erd_artifact.meta = {"table_count": 2, "fk_count": 1}

        tables_result = MagicMock()
        tables_result.scalars.return_value.all.return_value = [table_node]

        columns_result = MagicMock()
        columns_result.scalars.return_value.all.return_value = [
            id_column,
            account_id_column,
            account_pk_column,
        ]

        edges_result = MagicMock()
        edges_result.scalars.return_value.all.return_value = [
            table_has_id,
            table_has_account_id,
            fk_edge,
        ]

        artifact_result = MagicMock()
        artifact_result.scalar_one_or_none.return_value = erd_artifact

        fake_db = MagicMock()
        fake_db.execute = AsyncMock(
            side_effect=[tables_result, columns_result, edges_result, artifact_result]
        )

        class SessionContext:
            async def __aenter__(self):  # noqa: ANN001
                return fake_db

            async def __aexit__(self, _exc_type, _exc, _tb):  # noqa: ANN001
                return False

        mock_db_session_factory.return_value = SessionContext()

        response = await client.get(f"/api/twin/collections/{collection_id}/views/erm")
        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["tables"] == 1
        assert payload["summary"]["columns"] == 2
        assert payload["summary"]["foreign_keys"] == 1
        assert payload["summary"]["has_mermaid"] is True
        assert payload["mermaid"]["content"].startswith("erDiagram")
        assert payload["tables"][0]["name"] == "users"
        assert payload["foreign_keys"][0]["source_table"] == "users"
        assert payload["foreign_keys"][0]["target_table"] == "accounts"
