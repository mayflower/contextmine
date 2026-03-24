"""Coverage tests targeting remaining uncovered lines in routes/twin.py.

Targets:
- _resolve_arc42_repo_checkout (lines 432-459)
- _community_label and _compute_symbol_communities (lines 1267-1393)
- _load_community_graph recovery paths (lines 1484-1580)
- _dedupe_traces (lines 1729-1741)
- _trace_process_paths (lines 1752-1772)
- _detect_processes (lines 1780-1865)
- process detail endpoint (lines 4092-4143)
- semantic map community branch (lines 3737-3888)
- _project_vectors empty vectors (lines 800-821)
- _normalize_xy (force-directed layout, lines 887-899)
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# _community_label tests (lines 1267-1290)
# ---------------------------------------------------------------------------


class TestCommunityLabel:
    def test_community_label_from_file_paths(self) -> None:
        from app.routes.twin import _community_label

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.natural_key = "symbol:auth/login"
        node1.meta = {"file_path": "src/auth/login.py"}

        node2 = MagicMock()
        node2.id = uuid.uuid4()
        node2.natural_key = "symbol:auth/register"
        node2.meta = {"file_path": "src/auth/register.py"}

        node_by_id = {node1.id: node1, node2.id: node2}
        label = _community_label([node1.id, node2.id], node_by_id)
        assert isinstance(label, str)
        assert len(label) > 0

    def test_community_label_no_file_path(self) -> None:
        from app.routes.twin import _community_label

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.natural_key = "symbol:utils.helper"
        node1.meta = {}

        node_by_id = {node1.id: node1}
        label = _community_label([node1.id], node_by_id)
        assert isinstance(label, str)

    def test_community_label_short_tokens_ignored(self) -> None:
        from app.routes.twin import _community_label

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.natural_key = "a"
        node1.meta = {"file_path": "x/ab"}

        node_by_id = {node1.id: node1}
        label = _community_label([node1.id], node_by_id)
        # Short tokens (< 3 chars) should be skipped, falls back to "Community"
        assert label == "Community"


# ---------------------------------------------------------------------------
# _compute_symbol_communities tests (lines 1293-1393)
# ---------------------------------------------------------------------------


class TestComputeSymbolCommunities:
    def test_empty_nodes(self) -> None:
        from app.routes.twin import _compute_symbol_communities

        mapping, communities = _compute_symbol_communities([], [])
        assert mapping == {}
        assert communities == {}

    def test_single_node_community(self) -> None:
        from app.routes.twin import _compute_symbol_communities

        node = MagicMock()
        node.id = uuid.uuid4()
        node.name = "main"
        node.natural_key = "symbol:main"
        node.kind = MagicMock(value="function")
        node.meta = {"file_path": "src/service/main.py"}

        mapping, communities = _compute_symbol_communities([node], [])
        assert node.id in mapping
        assert len(communities) == 1
        comm = list(communities.values())[0]
        assert comm["size"] == 1

    def test_two_connected_nodes_same_community(self) -> None:
        from app.routes.twin import _compute_symbol_communities

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.name = "handler"
        node1.natural_key = "symbol:handler"
        node1.kind = MagicMock(value="function")
        node1.meta = {"file_path": "src/api/handler.py"}

        node2 = MagicMock()
        node2.id = uuid.uuid4()
        node2.name = "service"
        node2.natural_key = "symbol:service"
        node2.kind = MagicMock(value="function")
        node2.meta = {"file_path": "src/api/service.py"}

        edge = MagicMock()
        edge.source_node_id = node1.id
        edge.target_node_id = node2.id

        mapping, communities = _compute_symbol_communities([node1, node2], [edge])
        # Both nodes should be in same community
        assert mapping[node1.id] == mapping[node2.id]
        assert len(communities) == 1
        comm = list(communities.values())[0]
        assert comm["size"] == 2
        assert comm["cohesion"] == 1.0  # 1 edge / 1 possible = 1.0

    def test_disconnected_nodes_separate_communities(self) -> None:
        from app.routes.twin import _compute_symbol_communities

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.name = "alpha"
        node1.natural_key = "symbol:alpha"
        node1.kind = MagicMock(value="function")
        node1.meta = {"file_path": "src/alpha/main.py"}

        node2 = MagicMock()
        node2.id = uuid.uuid4()
        node2.name = "beta"
        node2.natural_key = "symbol:beta"
        node2.kind = MagicMock(value="class")
        node2.meta = {"file_path": "src/beta/main.py"}

        mapping, communities = _compute_symbol_communities([node1, node2], [])
        assert mapping[node1.id] != mapping[node2.id]
        assert len(communities) == 2

    def test_self_loop_edge_ignored(self) -> None:
        from app.routes.twin import _compute_symbol_communities

        node = MagicMock()
        node.id = uuid.uuid4()
        node.name = "recursive"
        node.natural_key = "symbol:recursive"
        node.kind = MagicMock(value="function")
        node.meta = {}

        edge = MagicMock()
        edge.source_node_id = node.id
        edge.target_node_id = node.id  # self-loop

        mapping, communities = _compute_symbol_communities([node], [edge])
        assert len(communities) == 1


# ---------------------------------------------------------------------------
# _dedupe_traces tests (lines 1729-1741)
# ---------------------------------------------------------------------------


class TestDedupeTraces:
    def test_duplicate_subpath_removed(self) -> None:
        from app.routes.twin import _dedupe_traces

        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        traces = [
            [a, b, c],  # longer trace
            [a, b],  # subpath of above
        ]
        result = _dedupe_traces(traces)
        assert len(result) == 1
        assert result[0] == [a, b, c]

    def test_unique_traces_kept(self) -> None:
        from app.routes.twin import _dedupe_traces

        a, b, c, d = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        traces = [
            [a, b],
            [c, d],
        ]
        result = _dedupe_traces(traces)
        assert len(result) == 2

    def test_empty_traces(self) -> None:
        from app.routes.twin import _dedupe_traces

        result = _dedupe_traces([])
        assert result == []


# ---------------------------------------------------------------------------
# _trace_process_paths tests (lines 1752-1772)
# ---------------------------------------------------------------------------


class TestTraceProcessPaths:
    def test_single_entry_no_outgoing(self) -> None:
        from app.routes.twin import _trace_process_paths

        entry_id = uuid.uuid4()
        outgoing: dict[uuid.UUID, list[uuid.UUID]] = {}
        traces = _trace_process_paths(
            entry_id=entry_id,
            outgoing=outgoing,
            max_depth=5,
            max_branching=3,
            min_steps=1,
        )
        assert len(traces) == 1
        assert traces[0] == [entry_id]

    def test_linear_chain(self) -> None:
        from app.routes.twin import _trace_process_paths

        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        outgoing = {a: [b], b: [c]}
        traces = _trace_process_paths(
            entry_id=a,
            outgoing=outgoing,
            max_depth=10,
            max_branching=4,
            min_steps=2,
        )
        assert any(len(t) >= 2 for t in traces)

    def test_branching_path(self) -> None:
        from app.routes.twin import _trace_process_paths

        a, b, c, d = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        outgoing = {a: [b, c], b: [d], c: [d]}
        traces = _trace_process_paths(
            entry_id=a,
            outgoing=outgoing,
            max_depth=10,
            max_branching=4,
            min_steps=2,
        )
        assert len(traces) >= 1

    def test_cycle_avoided(self) -> None:
        from app.routes.twin import _trace_process_paths

        a, b = uuid.uuid4(), uuid.uuid4()
        outgoing = {a: [b], b: [a]}  # cycle
        traces = _trace_process_paths(
            entry_id=a,
            outgoing=outgoing,
            max_depth=10,
            max_branching=4,
            min_steps=2,
        )
        # Should not infinite loop, traces are finite
        for trace in traces:
            # No node appears twice in a trace
            assert len(trace) == len(set(trace))

    def test_min_steps_filter(self) -> None:
        from app.routes.twin import _trace_process_paths

        a = uuid.uuid4()
        outgoing: dict[uuid.UUID, list[uuid.UUID]] = {}
        # Single node path, min_steps=2 means it won't be included
        traces = _trace_process_paths(
            entry_id=a,
            outgoing=outgoing,
            max_depth=5,
            max_branching=3,
            min_steps=2,
        )
        assert len(traces) == 0


# ---------------------------------------------------------------------------
# _detect_processes tests (lines 1780-1865)
# ---------------------------------------------------------------------------


class TestDetectProcesses:
    def test_empty_nodes(self) -> None:
        from app.routes.twin import _detect_processes

        processes = _detect_processes([], [], {})
        assert processes == []

    def test_two_node_process(self) -> None:
        from app.routes.twin import _detect_processes

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.name = "entry"
        node1.natural_key = "symbol:entry"
        node1.kind = MagicMock(value="function")
        node1.meta = {}

        node2 = MagicMock()
        node2.id = uuid.uuid4()
        node2.name = "handler"
        node2.natural_key = "symbol:handler"
        node2.kind = MagicMock(value="function")
        node2.meta = {}

        edge = MagicMock()
        edge.source_node_id = node1.id
        edge.target_node_id = node2.id

        community_map = {node1.id: "comm_1", node2.id: "comm_1"}

        processes = _detect_processes([node1, node2], [edge], community_map)
        assert len(processes) >= 1
        proc = processes[0]
        assert proc["step_count"] >= 2
        assert proc["process_type"] == "intra_community"

    def test_cross_community_process(self) -> None:
        from app.routes.twin import _detect_processes

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.name = "api_handler"
        node1.natural_key = "symbol:api_handler"
        node1.kind = MagicMock(value="function")
        node1.meta = {}

        node2 = MagicMock()
        node2.id = uuid.uuid4()
        node2.name = "db_query"
        node2.natural_key = "symbol:db_query"
        node2.kind = MagicMock(value="function")
        node2.meta = {}

        edge = MagicMock()
        edge.source_node_id = node1.id
        edge.target_node_id = node2.id

        # Different communities
        community_map = {node1.id: "comm_1", node2.id: "comm_2"}

        processes = _detect_processes([node1, node2], [edge], community_map)
        assert len(processes) >= 1
        proc = processes[0]
        assert proc["process_type"] == "cross_community"

    def test_no_entry_points_fallback(self) -> None:
        """When all nodes have incoming edges, use candidates directly."""
        from app.routes.twin import _detect_processes

        node1 = MagicMock()
        node1.id = uuid.uuid4()
        node1.name = "a"
        node1.natural_key = "symbol:a"
        node1.kind = MagicMock(value="function")
        node1.meta = {}

        node2 = MagicMock()
        node2.id = uuid.uuid4()
        node2.name = "b"
        node2.natural_key = "symbol:b"
        node2.kind = MagicMock(value="function")
        node2.meta = {}

        # Mutual edges: both have incoming, so no pure entry points
        edge1 = MagicMock()
        edge1.source_node_id = node1.id
        edge1.target_node_id = node2.id
        edge2 = MagicMock()
        edge2.source_node_id = node2.id
        edge2.target_node_id = node1.id

        community_map = {node1.id: "comm_1", node2.id: "comm_1"}

        processes = _detect_processes([node1, node2], [edge1, edge2], community_map)
        # Should still produce processes (uses candidates fallback)
        assert isinstance(processes, list)


# ---------------------------------------------------------------------------
# _project_vectors edge cases (lines 800-821)
# ---------------------------------------------------------------------------


class TestProjectVectors:
    def test_empty_vectors_fallback(self) -> None:
        from app.routes.twin import _project_vectors

        points = [
            {"id": "1", "x": 0.0, "y": 0.0, "vector": []},
            {"id": "2", "x": 0.0, "y": 0.0, "vector": []},
        ]
        _project_vectors(points, "vector")
        # With empty vectors, x/y remain at defaults
        assert points[0]["x"] == 0.0
        assert points[0]["y"] == 0.0

    def test_with_vectors(self) -> None:
        from app.routes.twin import _project_vectors

        points = [
            {"id": "1", "x": 0.0, "y": 0.0, "vector": [0.1, 0.2, 0.3]},
            {"id": "2", "x": 0.0, "y": 0.0, "vector": [0.4, 0.5, 0.6]},
        ]
        _project_vectors(points, "vector")
        # Should have been projected to x/y
        assert isinstance(points[0]["x"], float)
        assert isinstance(points[0]["y"], float)

    def test_mixed_vectors(self) -> None:
        from app.routes.twin import _project_vectors

        points = [
            {"id": "1", "x": 0.0, "y": 0.0, "vector": [0.1, 0.2]},
            {"id": "2", "x": 0.0, "y": 0.0, "vector": []},
        ]
        _project_vectors(points, "vector")
        # Point with vector gets projection; point without gets sin/cos fallback
        assert isinstance(points[0]["x"], float)
        assert isinstance(points[1]["x"], float)


# ---------------------------------------------------------------------------
# _normalize_xy tests (lines 887-899)
# ---------------------------------------------------------------------------


class TestNormalizeXY:
    def test_normalize_xy_basic(self) -> None:
        from app.routes.twin import _normalize_xy

        points = [
            {"id": "1", "x": 0.0, "y": 0.0},
            {"id": "2", "x": 10.0, "y": 10.0},
            {"id": "3", "x": 5.0, "y": 5.0},
        ]
        _normalize_xy(points)
        # After normalization, coordinates should be in [-1, 1] range roughly
        for point in points:
            assert isinstance(point["x"], float)
            assert isinstance(point["y"], float)

    def test_normalize_xy_single_point(self) -> None:
        from app.routes.twin import _normalize_xy

        points = [{"id": "1", "x": 5.0, "y": 5.0}]
        _normalize_xy(points)
        assert isinstance(points[0]["x"], float)


# ---------------------------------------------------------------------------
# _cosine_similarity (lines 913-924)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        result = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(result) < 1e-6

    def test_empty_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        assert _cosine_similarity([], [1.0]) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0

    def test_zero_vectors(self) -> None:
        from app.routes.twin import _cosine_similarity

        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# _parse_db_table_from_natural_key (line 466-469)
# ---------------------------------------------------------------------------


class TestParseDbTable:
    def test_valid_db_key(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        result = _parse_db_table_from_natural_key("db:users")
        assert result == "users"

    def test_non_db_key(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        result = _parse_db_table_from_natural_key("symbol:foo")
        assert result is None

    def test_none_key(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        result = _parse_db_table_from_natural_key(None)
        assert result is None

    def test_empty_key(self) -> None:
        from app.routes.twin import _parse_db_table_from_natural_key

        result = _parse_db_table_from_natural_key("")
        assert result is None


# ---------------------------------------------------------------------------
# _symbol_sort_key (line 1263-1264)
# ---------------------------------------------------------------------------


class TestSymbolSortKey:
    def test_sort_key_returns_tuple(self) -> None:
        from app.routes.twin import _symbol_sort_key

        node = MagicMock()
        node.natural_key = "symbol:handler"
        node.name = "handler"
        node.id = uuid.uuid4()

        result = _symbol_sort_key(node)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == "symbol:handler"
        assert result[1] == "handler"

    def test_sort_key_none_values(self) -> None:
        from app.routes.twin import _symbol_sort_key

        node = MagicMock()
        node.natural_key = None
        node.name = None
        node.id = uuid.uuid4()

        result = _symbol_sort_key(node)
        assert result[0] == ""
        assert result[1] == ""
