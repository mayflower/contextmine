"""Async unit tests for GraphRAG retrieval functions.

Tests the async database-dependent functions in graphrag.py using mocked
sessions and embedders:
- graph_rag_context()
- graph_neighborhood()
- trace_path()
- _map_search_to_nodes()
- _expand_from_seeds()
- _gather_citations()
- _citations_by_node()
- _find_relevant_communities()
- _get_community_member_nodes()
- _map_community()
- _reduce_answers()
- graph_rag_query()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from contextmine_core.graphrag import (
    CommunityContext,
    ContextPack,
    EntityContext,
    _citations_by_node,
    _expand_from_seeds,
    _find_relevant_communities,
    _gather_citations,
    _get_community_member_nodes,
    _map_community,
    _map_search_to_nodes,
    _reduce_answers,
    graph_neighborhood,
    graph_rag_context,
    graph_rag_query,
    trace_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session() -> AsyncMock:
    """Create a reusable async mock session."""
    session = AsyncMock()
    return session


def _make_mock_node(
    node_id: UUID | None = None,
    kind_value: str = "file",
    natural_key: str = "file:src/main.py",
    name: str = "main.py",
    collection_id: UUID | None = None,
) -> MagicMock:
    """Create a mock KnowledgeNode."""
    node = MagicMock()
    node.id = node_id or uuid4()
    node.kind = MagicMock()
    node.kind.value = kind_value
    node.natural_key = natural_key
    node.name = name
    node.collection_id = collection_id or uuid4()
    node.meta = {}
    return node


def _make_mock_edge(
    source_node_id: UUID,
    target_node_id: UUID,
    kind_value: str = "file_defines_symbol",
    collection_id: UUID | None = None,
) -> MagicMock:
    """Create a mock KnowledgeEdge."""
    edge = MagicMock()
    edge.source_node_id = source_node_id
    edge.target_node_id = target_node_id
    edge.kind = MagicMock()
    edge.kind.value = kind_value
    edge.collection_id = collection_id or uuid4()
    return edge


def _make_mock_evidence(
    file_path: str = "src/main.py",
    start_line: int = 10,
    end_line: int = 20,
    snippet: str | None = "code",
) -> MagicMock:
    """Create a mock KnowledgeEvidence."""
    evidence = MagicMock()
    evidence.file_path = file_path
    evidence.start_line = start_line
    evidence.end_line = end_line
    evidence.snippet = snippet
    return evidence


# ===========================================================================
# _gather_citations tests
# ===========================================================================


class TestGatherCitations:
    """Tests for _gather_citations with mocked session."""

    @pytest.mark.anyio
    async def test_empty_node_ids_returns_empty(self) -> None:
        session = _mock_session()
        result = await _gather_citations(session, [])
        assert result == []
        session.execute.assert_not_called()

    @pytest.mark.anyio
    async def test_returns_citations_from_evidence(self) -> None:
        session = _mock_session()
        ev1 = _make_mock_evidence(file_path="a.py", start_line=1, end_line=5)
        ev2 = _make_mock_evidence(file_path="b.py", start_line=10, end_line=20)

        mock_result = MagicMock()
        mock_result.scalars.return_value = [ev1, ev2]
        session.execute.return_value = mock_result

        citations = await _gather_citations(session, [uuid4()])
        assert len(citations) == 2
        assert citations[0].file_path == "a.py"
        assert citations[1].file_path == "b.py"

    @pytest.mark.anyio
    async def test_deduplicates_by_file_and_lines(self) -> None:
        session = _mock_session()
        ev1 = _make_mock_evidence(file_path="a.py", start_line=1, end_line=5)
        ev2 = _make_mock_evidence(file_path="a.py", start_line=1, end_line=5)

        mock_result = MagicMock()
        mock_result.scalars.return_value = [ev1, ev2]
        session.execute.return_value = mock_result

        citations = await _gather_citations(session, [uuid4()])
        assert len(citations) == 1

    @pytest.mark.anyio
    async def test_limits_to_50_citations(self) -> None:
        session = _mock_session()
        evidences = [
            _make_mock_evidence(file_path=f"f{i}.py", start_line=i, end_line=i + 1)
            for i in range(60)
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value = evidences
        session.execute.return_value = mock_result

        citations = await _gather_citations(session, [uuid4()])
        assert len(citations) == 50


# ===========================================================================
# _citations_by_node tests
# ===========================================================================


class TestCitationsByNode:
    """Tests for _citations_by_node with mocked session."""

    @pytest.mark.anyio
    async def test_empty_node_ids_returns_empty_dict(self) -> None:
        session = _mock_session()
        result = await _citations_by_node(session, [])
        assert result == {}
        session.execute.assert_not_called()

    @pytest.mark.anyio
    async def test_groups_by_node_id(self) -> None:
        session = _mock_session()
        nid1 = uuid4()
        nid2 = uuid4()

        ev1 = _make_mock_evidence(file_path="a.py", start_line=1, end_line=5)
        ev2 = _make_mock_evidence(file_path="b.py", start_line=10, end_line=20)
        ev3 = _make_mock_evidence(file_path="c.py", start_line=30, end_line=40)

        mock_result = MagicMock()
        mock_result.all.return_value = [(nid1, ev1), (nid1, ev2), (nid2, ev3)]
        session.execute.return_value = mock_result

        result = await _citations_by_node(session, [nid1, nid2])
        assert len(result[nid1]) == 2
        assert len(result[nid2]) == 1
        assert result[nid1][0].file_path == "a.py"
        assert result[nid2][0].file_path == "c.py"


# ===========================================================================
# _map_search_to_nodes tests
# ===========================================================================


class TestMapSearchToNodes:
    """Tests for _map_search_to_nodes with mocked session."""

    @pytest.mark.anyio
    async def test_empty_search_response_returns_empty(self) -> None:
        session = _mock_session()
        collection_ids = [uuid4()]

        mock_response = MagicMock(spec=[])
        result = await _map_search_to_nodes(session, mock_response, collection_ids)
        assert result == []

    @pytest.mark.anyio
    async def test_no_results_returns_empty(self) -> None:
        session = _mock_session()
        mock_response = MagicMock()
        mock_response.results = []

        result = await _map_search_to_nodes(session, mock_response, [uuid4()])
        assert result == []

    @pytest.mark.anyio
    async def test_maps_search_uris_to_file_nodes(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid = uuid4()

        search_result = MagicMock()
        search_result.uri = "src/main.py"
        mock_response = MagicMock()
        mock_response.results = [search_result]

        mock_db_result = MagicMock()
        mock_db_result.fetchall.return_value = [(nid,)]
        session.execute.return_value = mock_db_result

        result = await _map_search_to_nodes(session, mock_response, [coll_id])
        assert nid in result

    @pytest.mark.anyio
    async def test_maps_search_uris_to_legacy_prefixed_file_nodes(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid = uuid4()

        search_result = MagicMock()
        search_result.uri = "src/main.py"
        mock_response = MagicMock()
        mock_response.results = [search_result]

        mock_db_result = MagicMock()
        mock_db_result.fetchall.return_value = [(nid,)]
        session.execute.return_value = mock_db_result

        result = await _map_search_to_nodes(session, mock_response, [coll_id])
        assert result == [nid]

        stmt = session.execute.call_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "'src/main.py'" in compiled
        assert "'file:src/main.py'" in compiled

    @pytest.mark.anyio
    async def test_limits_to_20_results(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        search_results = []
        for i in range(25):
            sr = MagicMock()
            sr.uri = f"src/file_{i}.py"
            search_results.append(sr)
        mock_response = MagicMock()
        mock_response.results = search_results

        node_ids = [(uuid4(),) for _ in range(25)]
        mock_db_result = MagicMock()
        mock_db_result.fetchall.return_value = node_ids
        session.execute.return_value = mock_db_result

        result = await _map_search_to_nodes(session, mock_response, [coll_id])
        assert len(result) <= 20

    @pytest.mark.anyio
    async def test_skips_results_without_uri(self) -> None:
        session = _mock_session()

        sr1 = MagicMock()
        sr1.uri = ""
        sr2 = MagicMock()
        sr2.uri = None
        mock_response = MagicMock()
        mock_response.results = [sr1, sr2]

        result = await _map_search_to_nodes(session, mock_response, [uuid4()])
        assert result == []


# ===========================================================================
# _get_community_member_nodes tests
# ===========================================================================


class TestGetCommunityMemberNodes:
    """Tests for _get_community_member_nodes with mocked session."""

    @pytest.mark.anyio
    async def test_empty_community_ids_returns_empty(self) -> None:
        session = _mock_session()
        result = await _get_community_member_nodes(session, [])
        assert result == []

    @pytest.mark.anyio
    async def test_returns_member_node_ids(self) -> None:
        session = _mock_session()
        comm_id = uuid4()
        nid1 = uuid4()
        nid2 = uuid4()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(comm_id, nid1, 0.9), (comm_id, nid2, 0.8)]
        session.execute.return_value = mock_result

        result = await _get_community_member_nodes(session, [comm_id])
        assert result == [nid1, nid2]

    @pytest.mark.anyio
    async def test_spreads_member_selection_across_communities(self) -> None:
        session = _mock_session()
        comm_a = uuid4()
        comm_b = uuid4()
        a1 = uuid4()
        a2 = uuid4()
        b1 = uuid4()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (comm_a, a1, 0.95),
            (comm_a, a2, 0.9),
            (comm_b, b1, 0.85),
        ]
        session.execute.return_value = mock_result

        result = await _get_community_member_nodes(session, [comm_a, comm_b], limit=2)
        assert result == [a1, b1]


# ===========================================================================
# _find_relevant_communities tests
# ===========================================================================


class TestFindRelevantCommunities:
    """Tests for _find_relevant_communities with mocked session."""

    @pytest.mark.anyio
    async def test_returns_communities_from_db(self) -> None:
        session = _mock_session()
        comm_id = uuid4()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            (comm_id, 0, "Auth Module", "Handles auth", {"member_count": 5}, 0.85),
        ]
        session.execute.return_value = mock_result

        communities = await _find_relevant_communities(session, [0.1, 0.2], [uuid4()], 5)
        assert len(communities) == 1
        assert communities[0].community_id == comm_id
        assert communities[0].title == "Auth Module"
        assert communities[0].summary == "Handles auth"
        assert communities[0].member_count == 5
        assert communities[0].relevance_score == 0.85

    @pytest.mark.anyio
    async def test_empty_result_returns_empty_list(self) -> None:
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute.return_value = mock_result

        communities = await _find_relevant_communities(session, [0.1], [uuid4()], 5)
        assert communities == []

    @pytest.mark.anyio
    async def test_handles_none_meta(self) -> None:
        session = _mock_session()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            (uuid4(), 1, None, None, None, None),
        ]
        session.execute.return_value = mock_result

        communities = await _find_relevant_communities(session, [0.1], [uuid4()], 5)
        assert len(communities) == 1
        assert communities[0].title == "Untitled"
        assert communities[0].summary == ""
        assert communities[0].member_count == 0
        assert communities[0].relevance_score == 0.0

    @pytest.mark.anyio
    async def test_falls_back_to_size_meta_for_member_count(self) -> None:
        session = _mock_session()
        comm_id = uuid4()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            (comm_id, 0, "Auth Module", "Handles auth", {"size": 7}, 0.9),
        ]
        session.execute.return_value = mock_result

        communities = await _find_relevant_communities(session, [0.1, 0.2], [uuid4()], 5)

        assert len(communities) == 1
        assert communities[0].member_count == 7


# ===========================================================================
# _expand_from_seeds tests
# ===========================================================================


class TestExpandFromSeeds:
    """Tests for _expand_from_seeds with mocked session."""

    @pytest.mark.anyio
    async def test_empty_seeds_returns_empty(self) -> None:
        session = _mock_session()
        entities, edges = await _expand_from_seeds(session, [], [uuid4()], 2, 20)
        assert entities == []
        assert edges == []

    @pytest.mark.anyio
    async def test_expands_single_seed(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        node_id = uuid4()
        node = _make_mock_node(node_id=node_id, collection_id=coll_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # First call: fetch nodes
                result.scalars.return_value = [node]
            elif call_count[0] == 1:
                # Second call: fetch edges (empty)
                result.scalars.return_value = []
            else:
                result.scalars.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        entities, edges = await _expand_from_seeds(session, [node_id], [coll_id], 1, 20)
        assert len(entities) == 1
        assert entities[0].node_id == node_id
        assert entities[0].name == "main.py"

    @pytest.mark.anyio
    async def test_respects_max_entities(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        nodes = [
            _make_mock_node(node_id=uuid4(), collection_id=coll_id, name=f"n{i}") for i in range(5)
        ]
        seed_ids = [n.id for n in nodes]

        async def mock_execute(stmt):
            result = MagicMock()
            result.scalars.return_value = nodes
            return result

        session.execute.side_effect = mock_execute

        entities, _ = await _expand_from_seeds(session, seed_ids, [coll_id], 0, 3)
        assert len(entities) <= 3

    @pytest.mark.anyio
    async def test_relevance_score_decreases_with_depth(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid1 = uuid4()
        nid2 = uuid4()
        node1 = _make_mock_node(node_id=nid1, collection_id=coll_id, name="n1")
        node2 = _make_mock_node(node_id=nid2, collection_id=coll_id, name="n2")

        edge = _make_mock_edge(source_node_id=nid1, target_node_id=nid2, collection_id=coll_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = [node1]
            elif call_count[0] == 1:
                result.scalars.return_value = [edge]
            elif call_count[0] == 2:
                result.scalars.return_value = [node2]
            else:
                result.scalars.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        entities, edges = await _expand_from_seeds(session, [nid1], [coll_id], 2, 20)

        # Depth 0 -> score 1.0, depth 1 -> score 0.8
        depth0_entities = [e for e in entities if e.node_id == nid1]
        depth1_entities = [e for e in entities if e.node_id == nid2]
        if depth0_entities:
            assert depth0_entities[0].relevance_score == 1.0
        if depth1_entities:
            assert depth1_entities[0].relevance_score == 0.8

    @pytest.mark.anyio
    async def test_deduplicates_edges_across_bfs_depths(self) -> None:
        session = _mock_session()
        coll_id = uuid4()
        nid1 = uuid4()
        nid2 = uuid4()
        node1 = _make_mock_node(node_id=nid1, collection_id=coll_id, name="n1")
        node2 = _make_mock_node(node_id=nid2, collection_id=coll_id, name="n2")
        edge = _make_mock_edge(source_node_id=nid1, target_node_id=nid2, collection_id=coll_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = [node1]
            elif call_count[0] == 1:
                result.scalars.return_value = [edge]
            elif call_count[0] == 2:
                result.scalars.return_value = [node2]
            elif call_count[0] == 3:
                result.scalars.return_value = [edge]
            else:
                result.scalars.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        entities, edges = await _expand_from_seeds(session, [nid1], [coll_id], 2, 20)

        assert len(entities) == 2
        assert len(edges) == 1


# ===========================================================================
# graph_neighborhood tests
# ===========================================================================


class TestGraphNeighborhood:
    """Tests for graph_neighborhood with mocked session."""

    @pytest.mark.anyio
    async def test_with_collection_id(self) -> None:
        session = _mock_session()
        nid = uuid4()
        coll_id = uuid4()
        node = _make_mock_node(node_id=nid, collection_id=coll_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = [node]
            else:
                result.scalars.return_value = []
                result.all.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        pack = await graph_neighborhood(session, nid, collection_id=coll_id, depth=0)
        assert isinstance(pack, ContextPack)
        assert len(pack.entities) >= 1

    @pytest.mark.anyio
    async def test_without_collection_id_resolves_from_node(self) -> None:
        session = _mock_session()
        nid = uuid4()
        coll_id = uuid4()
        node = _make_mock_node(node_id=nid, collection_id=coll_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # Resolve collection from node
                result.scalar_one_or_none.return_value = coll_id
            elif call_count[0] == 1:
                result.scalars.return_value = [node]
            else:
                result.scalars.return_value = []
                result.all.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        pack = await graph_neighborhood(session, nid, depth=0)
        assert isinstance(pack, ContextPack)

    @pytest.mark.anyio
    async def test_unknown_node_returns_empty_pack(self) -> None:
        session = _mock_session()
        nid = uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        pack = await graph_neighborhood(session, nid)
        assert pack.entities == []
        assert pack.edges == []

    @pytest.mark.anyio
    async def test_edge_kind_filter(self) -> None:
        session = _mock_session()
        nid = uuid4()
        coll_id = uuid4()
        node = _make_mock_node(node_id=nid, collection_id=coll_id)

        nid2 = uuid4()
        edge_good = _make_mock_edge(
            nid, nid2, kind_value="file_defines_symbol", collection_id=coll_id
        )
        edge_bad = _make_mock_edge(
            nid, uuid4(), kind_value="symbol_calls_symbol", collection_id=coll_id
        )

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.scalars.return_value = [node]
            elif call_count[0] == 1:
                result.scalars.return_value = [edge_good, edge_bad]
            else:
                result.scalars.return_value = []
                result.all.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        pack = await graph_neighborhood(
            session,
            nid,
            collection_id=coll_id,
            depth=1,
            edge_kinds=["file_defines_symbol"],
        )
        for e in pack.edges:
            assert e.kind == "file_defines_symbol"


# ===========================================================================
# trace_path tests
# ===========================================================================


class TestTracePath:
    """Tests for trace_path with mocked session (BFS logic)."""

    @pytest.mark.anyio
    async def test_same_node_returns_single_entity(self) -> None:
        session = _mock_session()
        nid = uuid4()
        coll_id = uuid4()
        node = _make_mock_node(node_id=nid, collection_id=coll_id, name="nodeA")

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # collection_id lookup
                result.fetchall.return_value = [(coll_id,)]
            elif call_count[0] == 1:
                # Node fetch for path
                result.scalars.return_value = MagicMock(all=lambda: [node])
            else:
                # Gather citations
                result.scalars.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        pack = await trace_path(session, nid, nid)
        assert len(pack.entities) == 1
        assert pack.entities[0].name == "nodeA"

    @pytest.mark.anyio
    async def test_no_path_returns_empty_pack(self) -> None:
        session = _mock_session()
        nid1 = uuid4()
        nid2 = uuid4()
        coll_id = uuid4()

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                result.fetchall.return_value = [(coll_id,)]
            else:
                # No edges found - BFS will exhaust
                result.scalars.return_value = MagicMock(all=lambda: [])
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        pack = await trace_path(session, nid1, nid2)
        assert pack.entities == []
        assert pack.paths == []

    @pytest.mark.anyio
    async def test_with_collection_id_skips_lookup(self) -> None:
        session = _mock_session()
        nid1 = uuid4()
        nid2 = uuid4()
        coll_id = uuid4()

        node1 = _make_mock_node(node_id=nid1, name="A")
        node2 = _make_mock_node(node_id=nid2, name="B")
        edge = _make_mock_edge(nid1, nid2, kind_value="calls", collection_id=coll_id)

        call_count = [0]

        async def mock_execute(stmt):
            result = MagicMock()
            if call_count[0] == 0:
                # BFS edges from nid1
                result.scalars.return_value = MagicMock(all=lambda: [edge])
            elif call_count[0] == 1:
                # Fetch full node data
                node_map = {nid1: node1, nid2: node2}
                result.scalars.return_value = MagicMock(all=lambda: list(node_map.values()))
            else:
                result.scalars.return_value = []
            call_count[0] += 1
            return result

        session.execute.side_effect = mock_execute

        pack = await trace_path(session, nid1, nid2, collection_id=coll_id)
        assert len(pack.entities) == 2
        assert len(pack.edges) == 1
        assert len(pack.paths) == 1
        assert "A" in pack.paths[0].description
        assert "B" in pack.paths[0].description

    @pytest.mark.anyio
    async def test_no_collection_ids_returns_empty(self) -> None:
        session = _mock_session()
        nid1 = uuid4()
        nid2 = uuid4()

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        session.execute.return_value = mock_result

        pack = await trace_path(session, nid1, nid2)
        assert pack.entities == []


# ===========================================================================
# _map_community tests
# ===========================================================================


class TestMapCommunity:
    """Tests for _map_community (MAP phase) with mocked LLM provider."""

    @pytest.mark.anyio
    async def test_returns_llm_response(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "  The auth module handles JWT tokens.  "

        comm = CommunityContext(
            community_id=uuid4(),
            level=0,
            title="Auth",
            summary="Handles auth",
            relevance_score=0.9,
            member_count=3,
        )

        result = await _map_community(mock_llm, "How does auth work?", comm)
        assert result == "The auth module handles JWT tokens."
        mock_llm.generate_text.assert_called_once()

    @pytest.mark.anyio
    async def test_uses_temperature_zero(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "answer"

        comm = CommunityContext(
            community_id=uuid4(),
            level=0,
            title="T",
            summary="S",
            relevance_score=0.5,
            member_count=1,
        )

        await _map_community(mock_llm, "query", comm)
        call_kwargs = mock_llm.generate_text.call_args.kwargs
        assert call_kwargs.get("temperature") == 0


# ===========================================================================
# _reduce_answers tests
# ===========================================================================


class TestReduceAnswers:
    """Tests for _reduce_answers (REDUCE phase) with mocked LLM provider."""

    @pytest.mark.anyio
    async def test_returns_combined_answer(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "  Combined answer here.  "

        result = await _reduce_answers(mock_llm, "query", ["partial 1", "partial 2"])
        assert result == "Combined answer here."

    @pytest.mark.anyio
    async def test_prompt_includes_all_partials(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "combined"

        await _reduce_answers(mock_llm, "q", ["answer A", "answer B", "answer C"])
        call_args = mock_llm.generate_text.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "answer A" in prompt
        assert "answer B" in prompt
        assert "answer C" in prompt


# ===========================================================================
# graph_rag_query tests
# ===========================================================================


def _skip_if_trio():
    """Skip test if running under trio (graph_rag_query uses asyncio.gather)."""
    try:
        import sniffio

        if sniffio.current_async_library() == "trio":
            pytest.skip("graph_rag_query uses asyncio.gather, incompatible with trio")
    except Exception:
        pass


class TestGraphRagQuery:
    """Tests for graph_rag_query with mocked context retrieval.

    graph_rag_query internally uses asyncio.gather which is not compatible
    with trio, so these tests skip on the trio backend.
    """

    @pytest.mark.anyio
    async def test_no_communities_returns_fallback_message(self) -> None:
        _skip_if_trio()
        session = _mock_session()
        mock_llm = AsyncMock()

        with patch(
            "contextmine_core.graphrag.graph_rag_context",
            new_callable=AsyncMock,
        ) as mock_ctx:
            mock_ctx.return_value = ContextPack(query="test", communities=[])

            result = await graph_rag_query(session, "test", mock_llm)
            assert "not have been fully indexed" in result.final_answer
            assert result.communities_used == 0

    @pytest.mark.anyio
    async def test_single_partial_returns_directly(self) -> None:
        _skip_if_trio()
        session = _mock_session()
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "The single answer."

        comm = CommunityContext(
            community_id=uuid4(),
            level=0,
            title="Auth",
            summary="Handles auth",
            relevance_score=0.9,
            member_count=3,
        )

        with patch(
            "contextmine_core.graphrag.graph_rag_context",
            new_callable=AsyncMock,
        ) as mock_ctx:
            mock_ctx.return_value = ContextPack(query="test", communities=[comm])

            result = await graph_rag_query(session, "test", mock_llm)
            assert result.final_answer == "The single answer."
            assert result.communities_used == 1

    @pytest.mark.anyio
    async def test_all_not_relevant_returns_no_info_message(self) -> None:
        _skip_if_trio()
        session = _mock_session()
        mock_llm = AsyncMock()
        mock_llm.generate_text.return_value = "NOT_RELEVANT"

        comms = [
            CommunityContext(
                community_id=uuid4(),
                level=0,
                title=f"C{i}",
                summary=f"S{i}",
                relevance_score=0.5,
                member_count=2,
            )
            for i in range(3)
        ]

        with patch(
            "contextmine_core.graphrag.graph_rag_context",
            new_callable=AsyncMock,
        ) as mock_ctx:
            mock_ctx.return_value = ContextPack(query="test", communities=comms)

            result = await graph_rag_query(session, "test", mock_llm)
            assert "No relevant information" in result.final_answer

    @pytest.mark.anyio
    async def test_multiple_partials_triggers_reduce(self) -> None:
        _skip_if_trio()
        session = _mock_session()
        mock_llm = AsyncMock()

        call_count = [0]

        async def mock_generate(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return f"Partial answer {call_count[0]}"
            return "Final combined answer"

        mock_llm.generate_text.side_effect = mock_generate

        comms = [
            CommunityContext(
                community_id=uuid4(),
                level=0,
                title=f"C{i}",
                summary=f"Summary {i}",
                relevance_score=0.9,
                member_count=5,
            )
            for i in range(2)
        ]

        with patch(
            "contextmine_core.graphrag.graph_rag_context",
            new_callable=AsyncMock,
        ) as mock_ctx:
            mock_ctx.return_value = ContextPack(query="test", communities=comms)

            result = await graph_rag_query(session, "test", mock_llm)
            assert result.final_answer == "Final combined answer"
            assert len(result.partial_answers) == 2

    @pytest.mark.anyio
    async def test_map_exception_is_handled(self) -> None:
        _skip_if_trio()
        session = _mock_session()
        mock_llm = AsyncMock()

        call_count = [0]

        async def mock_generate(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM timeout")
            return "Good answer"

        mock_llm.generate_text.side_effect = mock_generate

        comms = [
            CommunityContext(
                community_id=uuid4(),
                level=0,
                title=f"C{i}",
                summary=f"S{i}",
                relevance_score=0.9,
                member_count=5,
            )
            for i in range(2)
        ]

        with patch(
            "contextmine_core.graphrag.graph_rag_context",
            new_callable=AsyncMock,
        ) as mock_ctx:
            mock_ctx.return_value = ContextPack(query="test", communities=comms)

            result = await graph_rag_query(session, "test", mock_llm)
            # Should still have the one good answer
            assert len(result.partial_answers) >= 1


# ===========================================================================
# graph_rag_context integration test (mocked dependencies)
# ===========================================================================


class TestGraphRagContext:
    """Tests for graph_rag_context with all external calls mocked.

    graph_rag_context uses local imports inside the function body, so we
    must patch at the source module, not at contextmine_core.graphrag.
    """

    @pytest.mark.anyio
    async def test_returns_empty_pack_when_no_collection_ids(self) -> None:
        session = _mock_session()

        with (
            patch("contextmine_core.settings.get_settings") as mock_settings,
            patch("contextmine_core.embeddings.parse_embedding_model_spec") as mock_parse,
            patch("contextmine_core.embeddings.get_embedder") as mock_get_emb,
            patch(
                "contextmine_core.search.get_accessible_collection_ids", new_callable=AsyncMock
            ) as mock_acl,
        ):
            mock_settings.return_value = MagicMock(
                default_embedding_model="openai:text-embedding-3-small"
            )
            mock_parse.return_value = ("openai", "text-embedding-3-small")
            mock_embedder = AsyncMock()
            mock_embedder.embed_batch.return_value = MagicMock(embeddings=[[0.1, 0.2]])
            mock_get_emb.return_value = mock_embedder
            mock_acl.return_value = []

            pack = await graph_rag_context(session, "query")
            assert isinstance(pack, ContextPack)
            assert pack.query == "query"
            assert pack.communities == []
            assert pack.entities == []

    @pytest.mark.anyio
    async def test_with_collection_id_skips_acl(self) -> None:
        session = _mock_session()
        coll_id = uuid4()

        with (
            patch("contextmine_core.settings.get_settings") as mock_settings,
            patch("contextmine_core.embeddings.parse_embedding_model_spec") as mock_parse,
            patch("contextmine_core.embeddings.get_embedder") as mock_get_emb,
            patch("contextmine_core.search.hybrid_search", new_callable=AsyncMock) as mock_search,
            patch(
                "contextmine_core.graphrag._find_relevant_communities", new_callable=AsyncMock
            ) as mock_comms,
            patch(
                "contextmine_core.graphrag._map_search_to_nodes", new_callable=AsyncMock
            ) as mock_map,
            patch(
                "contextmine_core.graphrag._get_community_member_nodes", new_callable=AsyncMock
            ) as mock_members,
            patch(
                "contextmine_core.graphrag._expand_from_seeds", new_callable=AsyncMock
            ) as mock_expand,
            patch(
                "contextmine_core.graphrag._gather_citations", new_callable=AsyncMock
            ) as mock_cit,
            patch(
                "contextmine_core.graphrag._citations_by_node", new_callable=AsyncMock
            ) as mock_cit_by,
        ):
            mock_settings.return_value = MagicMock(
                default_embedding_model="openai:text-embedding-3-small"
            )
            mock_parse.return_value = ("openai", "text-embedding-3-small")
            mock_embedder = AsyncMock()
            mock_embedder.embed_batch.return_value = MagicMock(embeddings=[[0.1, 0.2]])
            mock_get_emb.return_value = mock_embedder

            mock_comms.return_value = []
            mock_search.return_value = MagicMock(results=[])
            mock_map.return_value = []
            mock_members.return_value = []

            entity = EntityContext(
                node_id=uuid4(),
                kind="FILE",
                natural_key="file:main.py",
                name="main.py",
            )
            mock_expand.return_value = ([entity], [])
            mock_cit.return_value = []
            mock_cit_by.return_value = {}

            pack = await graph_rag_context(session, "query", collection_id=coll_id)
            assert isinstance(pack, ContextPack)
            assert len(pack.entities) == 1
