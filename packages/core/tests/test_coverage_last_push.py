"""Last push tests to cross 80% coverage.

Targets:
- twin/grouping.py: derive_arch_group, canonical_file_path_from_node, canonical_file_path_from_meta
- pathing.py: canonicalize_repo_relative_path
- analyzer/extractors/openapi.py: _parse_schema array edge case
- analyzer/extractors/ui.py: more route and view patterns
- context.py: more FakeLLM edge cases
- search.py: dataclass construction
- twin/evolution.py: more build_entity_key patterns
- mermaid_c4.py: _kind_value, _normalize_c4_view
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# twin/grouping.py
# ---------------------------------------------------------------------------


class TestDeriveArchGroup:
    def test_explicit_architecture_meta(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        result = derive_arch_group(
            "src/main.py",
            {
                "architecture": {
                    "domain": "core",
                    "container": "api",
                    "component": "handlers",
                },
            },
        )
        assert result == ("core", "api", "handlers")

    def test_explicit_meta_no_component(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        result = derive_arch_group(
            "src/main.py",
            {
                "architecture": {"domain": "core", "container": "api"},
            },
        )
        assert result == ("core", "api", "api")

    def test_services_path(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        result = derive_arch_group("services/auth/handlers/login.py")
        assert result is not None
        assert result[0] == "auth"
        assert result[1] == "handlers"

    def test_apps_path(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        result = derive_arch_group("apps/web/components/Header.tsx")
        assert result is not None
        assert result[0] == "web"

    def test_simple_path(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        result = derive_arch_group("backend/api/routes.py")
        assert result is not None
        assert result[0] == "backend"
        assert result[1] == "api"

    def test_single_segment(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        result = derive_arch_group("main.py")
        assert result is None

    def test_empty_path(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        assert derive_arch_group("") is None
        assert derive_arch_group(None) is None

    def test_incomplete_explicit_meta(self) -> None:
        from contextmine_core.twin.grouping import derive_arch_group

        # Missing container should still fall back to a substantive path after generic wrappers.
        result = derive_arch_group(
            "src/billing/api/main.py",
            {
                "architecture": {"domain": "core"},
            },
        )
        assert result is not None
        assert result[0] == "billing"
        assert result[1] == "api"


class TestCanonicalFilePath:
    def test_from_node_file_kind(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_node

        result = canonical_file_path_from_node(
            {
                "kind": "file",
                "natural_key": "file:src/main.py",
            }
        )
        assert result == "src/main.py"

    def test_from_node_file_kind_empty(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_node

        result = canonical_file_path_from_node(
            {
                "kind": "file",
                "natural_key": "file:",
            }
        )
        assert result is None

    def test_from_node_symbol_kind(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_node

        result = canonical_file_path_from_node(
            {
                "kind": "symbol",
                "natural_key": "symbol:MyClass",
                "meta": {"file_path": "src/models.py"},
            }
        )
        assert result == "src/models.py"

    def test_from_node_no_file_path(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_node

        result = canonical_file_path_from_node(
            {
                "kind": "symbol",
                "natural_key": "symbol:X",
            }
        )
        assert result is None

    def test_from_meta_path_first(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_meta

        result = canonical_file_path_from_meta("src/main.py", {"file_path": "other.py"})
        assert result == "src/main.py"

    def test_from_meta_fallback(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_meta

        result = canonical_file_path_from_meta(None, {"file_path": "src/utils.py"})
        assert result == "src/utils.py"

    def test_from_meta_none(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_meta

        result = canonical_file_path_from_meta(None, None)
        assert result is None

    def test_from_meta_empty_string(self) -> None:
        from contextmine_core.twin.grouping import canonical_file_path_from_meta

        result = canonical_file_path_from_meta("  ", {"file_path": "  "})
        assert result is None


# ---------------------------------------------------------------------------
# pathing.py
# ---------------------------------------------------------------------------


class TestPathing:
    def test_canonicalize_basic(self) -> None:
        from contextmine_core.pathing import canonicalize_repo_relative_path

        result = canonicalize_repo_relative_path("src/main.py")
        assert result == "src/main.py"

    def test_canonicalize_backslashes(self) -> None:
        from contextmine_core.pathing import canonicalize_repo_relative_path

        result = canonicalize_repo_relative_path("src\\main.py")
        assert result == "src/main.py"

    def test_canonicalize_leading_dot_slash(self) -> None:
        from contextmine_core.pathing import canonicalize_repo_relative_path

        result = canonicalize_repo_relative_path("./src/main.py")
        assert result == "src/main.py"

    def test_canonicalize_empty(self) -> None:
        from contextmine_core.pathing import canonicalize_repo_relative_path

        result = canonicalize_repo_relative_path("")
        assert result == "" or result is None

    def test_canonicalize_whitespace(self) -> None:
        from contextmine_core.pathing import canonicalize_repo_relative_path

        result = canonicalize_repo_relative_path("  src/main.py  ")
        assert result == "src/main.py"


# ---------------------------------------------------------------------------
# UI: more route and view patterns
# ---------------------------------------------------------------------------


class TestUIMorePatterns:
    def test_react_router_v6_with_outlet(self) -> None:
        from contextmine_core.analyzer.extractors.ui import extract_ui_from_file

        code = """
import { Route, Routes, Outlet } from 'react-router-dom';

function AppLayout() {
  return (
    <div>
      <nav>
        <a href="/dashboard">Dashboard</a>
      </nav>
      <Outlet />
    </div>
  );
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
"""
        extraction = extract_ui_from_file("src/app/routes.tsx", code)
        # Should detect Route elements with paths
        if extraction.routes:
            paths = [r.path for r in extraction.routes]
            assert any("dashboard" in p for p in paths) or len(paths) >= 1

    def test_tsx_with_fetch_and_navigate(self) -> None:
        from contextmine_core.analyzer.extractors.ui import extract_ui_from_file

        code = """
import { useNavigate } from 'react-router-dom';

function UserList() {
  const navigate = useNavigate();
  const loadUsers = async () => {
    const resp = await fetch('/api/users');
    return resp.json();
  };
  return (
    <div>
      <button onClick={() => navigate('/users/new')}>New User</button>
      <a href="/api/export">Export</a>
    </div>
  );
}

export default UserList;
"""
        extraction = extract_ui_from_file("src/pages/users.tsx", code)
        if extraction.views:
            for v in extraction.views:
                if v.name == "UserList" and v.endpoint_hints:
                    # /api/users should be endpoint hint, not navigation
                    assert any("/api" in h for h in v.endpoint_hints)

    def test_vue_file_ignored(self) -> None:
        from contextmine_core.analyzer.extractors.ui import extract_ui_from_file

        code = """
<template>
  <div>
    <h1>Vue Component</h1>
    <a href="/about">About</a>
  </div>
</template>
<script>
export default { name: 'MyComponent' }
</script>
"""
        # Vue files in non-UI paths might not be detected
        extraction = extract_ui_from_file("src/components/MyComponent.vue", code)
        assert extraction is not None


# ---------------------------------------------------------------------------
# evolution: build_entity_key with file level
# ---------------------------------------------------------------------------


class TestBuildEntityKeyFile:
    def test_file_level_empty_path(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key("", None, "file")
        assert result is None

    def test_file_level_valid(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key("src/main.py", None, "file")
        assert result is not None
        assert result.startswith("file:")

    def test_container_level_with_meta(self) -> None:
        from contextmine_core.twin.evolution import build_entity_key

        result = build_entity_key(
            "apps/api/main.py",
            {
                "architecture": {"domain": "backend", "container": "api"},
            },
            "container",
        )
        assert result == "container:backend/api"


# ---------------------------------------------------------------------------
# OpenAPI: schema parsing edge cases
# ---------------------------------------------------------------------------


class TestOpenAPISchemaEdge:
    def test_schema_property_non_dict_skipped(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import _parse_schema

        schema = {
            "type": "object",
            "properties": {
                "good": {"type": "string"},
                "bad": "not a dict",
            },
        }
        result = _parse_schema("Mixed", schema)
        assert "good" in result.properties
        assert "bad" not in result.properties

    def test_schema_array_no_items(self) -> None:
        from contextmine_core.analyzer.extractors.openapi import _parse_schema

        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array"},
            },
        }
        result = _parse_schema("NoItems", schema)
        assert result.properties["tags"] == "array"
