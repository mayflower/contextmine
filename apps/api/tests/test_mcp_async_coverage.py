"""Additional async tests for mcp_server.py uncovered lines.

Targets:
- _resolve_arc42_repo_checkout (lines 85-107)
- research_architecture: test/qa topic + flow/journey topic + rebuild topic (lines 1561-1627)
- research_validation: no matches path (line 1232)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import app.mcp_server as mcp_mod
import pytest

pytestmark = pytest.mark.anyio

_resolve_arc42_repo_checkout = mcp_mod._resolve_arc42_repo_checkout
_research_validation = mcp_mod.research_validation.fn
_research_architecture = mcp_mod.research_architecture.fn
_code_references = mcp_mod.code_references.fn
_code_expand = mcp_mod.code_expand.fn


def _mock_db_session(mock_db):
    ctx = patch("app.mcp_server.get_db_session")
    mock_session = ctx.start()
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db)
    mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_settings(**overrides):
    defaults = {
        "arch_docs_enabled": True,
        "repos_root": "/tmp/repos",
        "default_llm_provider": "openai",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_knowledge_node(*, kind_value="business_rule", name="rule1", meta=None, node_id=None):
    node = MagicMock()
    node.id = node_id or uuid.uuid4()
    node.name = name
    node.kind = MagicMock(value=kind_value)
    node.kind.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
    node.meta = meta or {}
    return node


# ---------------------------------------------------------------------------
# _resolve_arc42_repo_checkout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _resolve_collection_for_tool with user_id path (line 205)
# ---------------------------------------------------------------------------


class TestResolveCollectionForToolUserPath:
    async def test_user_with_no_collection_id(self) -> None:
        """When collection_id is None and user_id is set, it queries user collections."""
        from app.mcp_server import _resolve_collection_for_tool

        mock_db = AsyncMock()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.name = "My Collection"
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = coll
        mock_db.execute = AsyncMock(return_value=result_mock)

        collection, error = await _resolve_collection_for_tool(
            mock_db, collection_id=None, user_id=uuid.uuid4()
        )
        assert collection is not None
        assert error is None

    async def test_no_user_no_collection_found(self) -> None:
        """When collection_id is None and no collections found."""
        from app.mcp_server import _resolve_collection_for_tool

        mock_db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result_mock)

        collection, error = await _resolve_collection_for_tool(
            mock_db, collection_id=None, user_id=None
        )
        assert collection is None
        assert "No accessible collection" in error


# ---------------------------------------------------------------------------
# list_documents: owner/member access (lines 390-404)
# ---------------------------------------------------------------------------

_list_documents = mcp_mod.list_documents.fn


# ---------------------------------------------------------------------------
# list_collections with user_id (line 322)
# ---------------------------------------------------------------------------

_list_collections = mcp_mod.list_collections.fn


class TestListCollectionsUser:
    async def test_with_user_id(self) -> None:
        mock_db = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_db.execute = AsyncMock(return_value=result_mock)

        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=uuid.uuid4()):
                result = await _list_collections()
            assert "Collections" in result or "No collections" in result
        finally:
            ctx.stop()

    async def test_without_user_id(self) -> None:
        mock_db = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        mock_db.execute = AsyncMock(return_value=result_mock)

        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _list_collections()
            assert "Collections" in result or "No collections" in result
        finally:
            ctx.stop()


class TestListDocumentsAccess:
    async def test_owner_access_to_private_collection(self) -> None:
        mock_db = AsyncMock()
        user_id = uuid.uuid4()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = MagicMock(value="private")
        coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
        coll.owner_user_id = user_id

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = coll
            else:
                result.fetchall.return_value = []
                result.scalars.return_value.all.return_value = []
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=user_id):
                result = await _list_documents(collection_id=str(coll.id))
            assert "Error" not in result or "Access denied" not in result
        finally:
            ctx.stop()

    async def test_member_access_to_private_collection(self) -> None:
        mock_db = AsyncMock()
        user_id = uuid.uuid4()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = MagicMock(value="private")
        coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
        coll.owner_user_id = uuid.uuid4()  # Different user

        member = MagicMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = coll
            elif call_count["n"] == 2:
                # member check
                result.scalar_one_or_none.return_value = member
            else:
                result.fetchall.return_value = []
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=user_id):
                result = await _list_documents(collection_id=str(coll.id))
            assert "Access denied" not in result
        finally:
            ctx.stop()

    async def test_access_denied_non_member(self) -> None:
        mock_db = AsyncMock()
        user_id = uuid.uuid4()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = MagicMock(value="private")
        coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
        coll.owner_user_id = uuid.uuid4()

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = coll
            else:
                result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=user_id):
                result = await _list_documents(collection_id=str(coll.id))
            assert "Access denied" in result
        finally:
            ctx.stop()


class TestResolveCollectionAccessOwner:
    async def test_owner_has_access(self) -> None:
        """Collection owner gets access even if visibility is private."""
        from app.mcp_server import _resolve_collection_access

        mock_db = AsyncMock()
        user_id = uuid.uuid4()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = MagicMock(value="private")
        coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
        coll.owner_user_id = user_id

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = coll
        mock_db.execute = AsyncMock(return_value=result_mock)

        collection, error = await _resolve_collection_access(
            mock_db, collection_id=str(coll.id), user_id=user_id
        )
        assert collection is not None
        assert error is None

    async def test_member_has_access(self) -> None:
        """Collection member gets access."""
        from app.mcp_server import _resolve_collection_access

        mock_db = AsyncMock()
        user_id = uuid.uuid4()
        other_user = uuid.uuid4()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = MagicMock(value="private")
        coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
        coll.owner_user_id = other_user

        member = MagicMock()
        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = coll
            else:
                result.scalar_one_or_none.return_value = member
            return result

        mock_db.execute = mock_execute

        collection, error = await _resolve_collection_access(
            mock_db, collection_id=str(coll.id), user_id=user_id
        )
        assert collection is not None
        assert error is None

    async def test_access_denied(self) -> None:
        """Non-member, non-owner gets denied."""
        from app.mcp_server import _resolve_collection_access

        mock_db = AsyncMock()
        user_id = uuid.uuid4()
        coll = MagicMock()
        coll.id = uuid.uuid4()
        coll.visibility = MagicMock(value="private")
        coll.visibility.__eq__ = lambda self, other: self.value == getattr(other, "value", other)
        coll.owner_user_id = uuid.uuid4()  # Different user

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = coll
            else:
                result.scalar_one_or_none.return_value = None  # Not a member
            return result

        mock_db.execute = mock_execute

        collection, error = await _resolve_collection_access(
            mock_db, collection_id=str(coll.id), user_id=user_id
        )
        assert error is not None
        assert "Access denied" in error


class TestResolveArc42RepoCheckout:
    async def test_no_source_found(self) -> None:
        mock_db = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result)

        with pytest.raises(RuntimeError, match="No enabled GitHub source"):
            await _resolve_arc42_repo_checkout(mock_db, collection_id=uuid.uuid4())

    async def test_repo_path_missing(self, tmp_path: Path) -> None:
        mock_db = AsyncMock()
        source = MagicMock()
        source.id = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = source
        mock_db.execute = AsyncMock(return_value=result)

        with (
            patch(
                "app.mcp_server.get_settings", return_value=_make_settings(repos_root=str(tmp_path))
            ),
            pytest.raises(RuntimeError, match="Local repository checkout missing"),
        ):
            await _resolve_arc42_repo_checkout(mock_db, collection_id=uuid.uuid4())

    async def test_success(self, tmp_path: Path) -> None:
        mock_db = AsyncMock()
        source = MagicMock()
        source.id = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = source
        mock_db.execute = AsyncMock(return_value=result)

        # Create the repo path
        repo_path = tmp_path / str(source.id)
        repo_path.mkdir()

        with patch(
            "app.mcp_server.get_settings", return_value=_make_settings(repos_root=str(tmp_path))
        ):
            src, path = await _resolve_arc42_repo_checkout(mock_db, collection_id=uuid.uuid4())
            assert src == source
            assert path == repo_path


# ---------------------------------------------------------------------------
# research_validation: no matches
# ---------------------------------------------------------------------------


class TestResearchValidationNoMatches:
    async def test_no_matches_returns_not_found(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            # Both business rules and candidates return empty
            result.scalars.return_value.all.return_value = []
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_validation(
                    code_path="nonexistent_module",
                    collection_id=str(coll_id),
                )
            assert "No Validation Rules Found" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# research_architecture: test topic
# ---------------------------------------------------------------------------


class TestResearchArchitectureTopics:
    async def test_test_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        suite = _make_knowledge_node(kind_value="test_suite", name="TestAuth")
        suite.kind = KnowledgeNodeKind.TEST_SUITE
        case = _make_knowledge_node(kind_value="test_case", name="test_login")
        case.kind = KnowledgeNodeKind.TEST_CASE
        fixture = _make_knowledge_node(kind_value="test_fixture", name="db_session")
        fixture.kind = KnowledgeNodeKind.TEST_FIXTURE

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [suite, case, fixture]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="test qa",
                    collection_id=str(coll_id),
                )
            assert "Test Semantics" in result
            assert "Suites:" in result
            assert "Cases:" in result
            assert "Fixtures:" in result
        finally:
            ctx.stop()

    async def test_flow_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        flow = _make_knowledge_node(kind_value="user_flow", name="Login Flow")
        flow.kind = KnowledgeNodeKind.USER_FLOW
        step = _make_knowledge_node(kind_value="flow_step", name="Enter credentials")
        step.kind = KnowledgeNodeKind.FLOW_STEP

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [flow, step]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="flow journey",
                    collection_id=str(coll_id),
                )
            assert "User Flows" in result
        finally:
            ctx.stop()

    async def test_rebuild_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        contract = _make_knowledge_node(
            kind_value="interface_contract",
            name="UserAPI",
            meta={"provenance": {"mode": "inferred", "confidence": 0.5}},
        )
        contract.kind = KnowledgeNodeKind.INTERFACE_CONTRACT

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                # For flow query that comes first
                result.scalars.return_value.all.return_value = []
                return result
            elif call_count["n"] == 2:
                # For readiness query
                result.scalars.return_value.all.return_value = [contract]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="rebuild readiness",
                    collection_id=str(coll_id),
                )
            assert "Rebuild Readiness" in result
            assert "Critical inferred-only nodes:" in result
        finally:
            ctx.stop()

    async def test_api_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        ep = _make_knowledge_node(
            kind_value="api_endpoint",
            name="GET /users",
            meta={"method": "GET", "path": "/users"},
        )
        ep.kind = KnowledgeNodeKind.API_ENDPOINT

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [ep]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="api endpoints",
                    collection_id=str(coll_id),
                )
            assert "API Endpoints" in result
        finally:
            ctx.stop()

    async def test_deployment_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        job = _make_knowledge_node(
            kind_value="job",
            name="deploy-prod",
            meta={"job_type": "github_actions", "schedule": "0 * * * *"},
        )
        job.kind = KnowledgeNodeKind.JOB

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [job]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="deployment jobs",
                    collection_id=str(coll_id),
                )
            assert "Jobs & Workflows" in result
        finally:
            ctx.stop()

    async def test_database_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        table = _make_knowledge_node(
            kind_value="db_table",
            name="users",
            meta={"column_count": 10},
        )
        table.kind = KnowledgeNodeKind.DB_TABLE

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [table]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="database schema",
                    collection_id=str(coll_id),
                )
            assert "Database Tables" in result
        finally:
            ctx.stop()

    async def test_security_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        rule = _make_knowledge_node(
            kind_value="business_rule",
            name="auth_check",
            meta={"category": "authorization", "natural_language": "Must be admin"},
        )
        rule.kind = KnowledgeNodeKind.BUSINESS_RULE

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [rule]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="security auth",
                    collection_id=str(coll_id),
                )
            assert "Security Rules" in result
        finally:
            ctx.stop()

    async def test_no_specific_topic(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        mock_db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
            )
        )
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_architecture(
                    topic="random topic xyz",
                    collection_id=str(coll_id),
                )
            assert "No specific architecture information found" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# code_references: document not found, symbol not found, refs found, truncated
# ---------------------------------------------------------------------------


class TestCodeReferences:
    async def test_document_not_found(self) -> None:
        mock_db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=result_mock)

        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="nonexistent.py", line=1, column=0)
            assert "Document Not Found" in result
        finally:
            ctx.stop()

    async def test_symbol_not_found(self) -> None:
        mock_db = AsyncMock()
        doc = MagicMock()
        doc.id = uuid.uuid4()
        doc.uri = "src/main.py"

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = doc
            else:
                result.scalars.return_value.first.return_value = None
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="src/main.py", line=10, column=0)
            assert "No Symbol Found" in result
        finally:
            ctx.stop()

    async def test_no_references(self) -> None:
        mock_db = AsyncMock()
        doc = MagicMock()
        doc.id = uuid.uuid4()
        doc.uri = "src/main.py"
        symbol = MagicMock()
        symbol.id = uuid.uuid4()
        symbol.qualified_name = "main.process"

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = doc
            elif call_count["n"] == 2:
                result.scalars.return_value.first.return_value = symbol
            else:
                result.scalars.return_value.all.return_value = []
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="src/main.py", line=10, column=0)
            assert "No References Found" in result
        finally:
            ctx.stop()

    async def test_references_found(self) -> None:
        mock_db = AsyncMock()
        doc = MagicMock()
        doc.id = uuid.uuid4()
        doc.uri = "src/main.py"
        symbol = MagicMock()
        symbol.id = uuid.uuid4()
        symbol.qualified_name = "main.process"

        edge = MagicMock()
        edge.source_symbol = MagicMock()
        edge.source_symbol.qualified_name = "caller.func"
        edge.source_symbol.document = MagicMock()
        edge.source_symbol.document.uri = "src/caller.py"
        edge.source_line = 42
        edge.edge_type = MagicMock(value="calls")

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = doc
            elif call_count["n"] == 2:
                result.scalars.return_value.first.return_value = symbol
            else:
                result.scalars.return_value.all.return_value = [edge]
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(file_path="src/main.py", line=10, column=0)
            assert "References to" in result
            assert "caller.func" in result
            assert "src/caller.py" in result
        finally:
            ctx.stop()

    async def test_references_truncated(self) -> None:
        mock_db = AsyncMock()
        doc = MagicMock()
        doc.id = uuid.uuid4()
        symbol = MagicMock()
        symbol.id = uuid.uuid4()
        symbol.qualified_name = "main.process"

        # Create more edges than the limit
        edges = []
        for i in range(25):
            edge = MagicMock()
            edge.source_symbol = MagicMock()
            edge.source_symbol.qualified_name = f"caller{i}.func"
            edge.source_symbol.document = MagicMock()
            edge.source_symbol.document.uri = f"src/caller{i}.py"
            edge.source_line = i + 1
            edge.edge_type = MagicMock(value="calls")
            edges.append(edge)

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalar_one_or_none.return_value = doc
            elif call_count["n"] == 2:
                result.scalars.return_value.first.return_value = symbol
            else:
                result.scalars.return_value.all.return_value = edges
            return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            result = await _code_references(
                file_path="src/main.py",
                line=10,
                column=0,
                limit=5,
            )
            assert "References to" in result
            assert "Showing 5 of" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# code_expand: edge cases
# ---------------------------------------------------------------------------


class TestCodeExpand:
    async def test_no_seeds(self) -> None:
        result = await _code_expand(seeds=[])
        assert "No seed symbols provided" in result

    async def test_invalid_seed_format(self) -> None:
        result = await _code_expand(seeds=["just_a_name"])
        assert "Invalid seed format" in result


# ---------------------------------------------------------------------------
# research_data_model
# ---------------------------------------------------------------------------

_research_data_model = mcp_mod.research_data_model.fn


class TestResearchDataModel:
    async def test_tables_and_columns_and_endpoints(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        from contextmine_core.models import KnowledgeNodeKind

        table = _make_knowledge_node(
            kind_value="db_table",
            name="users",
            meta={
                "column_count": 3,
                "primary_key": "id",
                "columns": [
                    {"name": "id", "type": "Integer", "nullable": False},
                    {"name": "email", "type": "String", "nullable": False},
                    {"name": "name", "type": "String", "nullable": True},
                ],
            },
        )
        table.kind = KnowledgeNodeKind.DB_TABLE
        table.natural_key = "db_table:users"

        col = _make_knowledge_node(
            kind_value="db_column",
            name="users_email",
            meta={"type": "String"},
        )
        col.kind = KnowledgeNodeKind.DB_COLUMN
        col.natural_key = "db_column:users:email"

        ep = _make_knowledge_node(
            kind_value="api_endpoint",
            name="GET /users",
            meta={"method": "GET", "path": "/users"},
        )
        ep.kind = KnowledgeNodeKind.API_ENDPOINT

        erd = MagicMock()
        erd.content = "erDiagram\n  USERS ||--o{ ORDERS : places\n"

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                # tables
                result.scalars.return_value.all.return_value = [table]
                return result
            elif call_count["n"] == 2:
                # columns
                result.scalars.return_value.all.return_value = [col]
                return result
            elif call_count["n"] == 3:
                # endpoints
                result.scalars.return_value.all.return_value = [ep]
                return result
            elif call_count["n"] == 4:
                # ERD artifact
                result.scalar_one_or_none.return_value = erd
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_data_model(
                    entity="users",
                    collection_id=str(coll_id),
                )
            assert "Data Model: users" in result
            assert "Database Tables" in result
            assert "Related Columns" in result
            assert "Related API Endpoints" in result
            assert "PK: `id`" in result
            assert "NOT NULL" in result
            assert "Entity Relationship Diagram" in result
        finally:
            ctx.stop()

    async def test_no_matches(self) -> None:
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        mock_db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
                scalar_one_or_none=MagicMock(return_value=None),
            )
        )
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_data_model(
                    entity="nonexistent_entity",
                    collection_id=str(coll_id),
                )
            assert "No Data Model Found" in result
        finally:
            ctx.stop()


# ---------------------------------------------------------------------------
# research_validation: evidence file path matching (lines 1201-1214)
# ---------------------------------------------------------------------------


class TestResearchValidationEvidence:
    async def test_rule_matched_by_evidence_path(self) -> None:
        """A rule not matching by name but matching by evidence file_path."""
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        rule = _make_knowledge_node(
            kind_value="business_rule",
            name="generic_check",
            meta={
                "category": "validation",
                "severity": "error",
                "natural_language": "Generic check rule",
            },
        )

        evidence = MagicMock()
        evidence.file_path = "auth/validators.py"

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                # business rules
                result.scalars.return_value.all.return_value = [rule]
                return result
            elif call_count["n"] == 2:
                # candidates
                result.scalars.return_value.all.return_value = []
                return result
            elif call_count["n"] == 3:
                # evidence query for rule
                result.scalars.return_value.all.return_value = [evidence]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_validation(
                    code_path="auth/validators",
                    collection_id=str(coll_id),
                )
            assert "Business Rules" in result
            assert "generic_check" in result
        finally:
            ctx.stop()

    async def test_rule_not_matched_by_evidence_path(self) -> None:
        """Rule not matching by name or evidence path."""
        mock_db = AsyncMock()
        coll_id = uuid.uuid4()

        rule = _make_knowledge_node(
            kind_value="business_rule",
            name="unrelated_check",
            meta={"category": "validation"},
        )

        evidence = MagicMock()
        evidence.file_path = "other/module.py"

        call_count = {"n": 0}

        async def mock_execute(stmt):
            call_count["n"] += 1
            result = MagicMock()
            if call_count["n"] == 1:
                result.scalars.return_value.all.return_value = [rule]
                return result
            elif call_count["n"] == 2:
                result.scalars.return_value.all.return_value = []
                return result
            elif call_count["n"] == 3:
                result.scalars.return_value.all.return_value = [evidence]
                return result
            else:
                result.scalars.return_value.all.return_value = []
                return result

        mock_db.execute = mock_execute
        ctx = _mock_db_session(mock_db)
        try:
            with patch("app.mcp_server.get_current_user_id", return_value=None):
                result = await _research_validation(
                    code_path="auth/validators",
                    collection_id=str(coll_id),
                )
            assert "No Validation Rules Found" in result
        finally:
            ctx.stop()
