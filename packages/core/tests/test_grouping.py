"""Tests for twin/grouping.py architecture grouping and file-path helpers.

Covers all branches of:
- derive_arch_group
- canonical_file_path_from_node
- canonical_file_path_from_meta
"""

from __future__ import annotations

from contextmine_core.twin.grouping import (
    canonical_file_path_from_meta,
    canonical_file_path_from_node,
    derive_arch_group,
)

# ---------------------------------------------------------------------------
# derive_arch_group
# ---------------------------------------------------------------------------


class TestDeriveArchGroup:
    # --- explicit architecture meta ---

    def test_explicit_architecture_meta_all_three(self) -> None:
        meta = {
            "architecture": {"domain": "payments", "container": "gateway", "component": "stripe"}
        }
        assert derive_arch_group("anything.py", meta) == ("payments", "gateway", "stripe")

    def test_explicit_architecture_meta_missing_component(self) -> None:
        """component defaults to container when absent."""
        meta = {"architecture": {"domain": "payments", "container": "gateway"}}
        assert derive_arch_group("anything.py", meta) == ("payments", "gateway", "gateway")

    def test_explicit_architecture_meta_empty_domain(self) -> None:
        """Falls through to path heuristic when domain is empty."""
        meta = {"architecture": {"domain": "", "container": "gateway"}}
        result = derive_arch_group("apps/web/page.tsx", meta)
        # Falls through to path heuristic
        assert result is not None
        assert result[0] == "web"

    def test_explicit_architecture_meta_empty_container(self) -> None:
        """Falls through to path heuristic when container is empty."""
        meta = {"architecture": {"domain": "payments", "container": ""}}
        result = derive_arch_group("services/payments/api/handler.py", meta)
        assert result is not None
        assert result[0] == "payments"

    # --- path heuristic: services/<domain>/<container>/... ---

    def test_path_services_prefix(self) -> None:
        result = derive_arch_group("services/billing/api/handler.py")
        assert result is not None
        domain, container, component = result
        assert domain == "billing"
        assert container == "api"
        assert component == "handler"

    def test_path_services_domain_file_fallback(self) -> None:
        result = derive_arch_group("services/billing/handler.py")
        assert result is not None
        domain, container, component = result
        assert domain == "billing"
        assert container == "billing"
        assert component == "handler"

    # --- path heuristic: apps/<name>/... ---

    def test_path_apps_prefix(self) -> None:
        result = derive_arch_group("apps/dashboard/index.tsx")
        assert result is not None
        domain, container, component = result
        assert domain == "dashboard"
        assert container == "dashboard"
        assert component == "index"

    # --- path heuristic: generic fallback ---

    def test_path_generic_two_parts(self) -> None:
        result = derive_arch_group("packages/core/service.py")
        assert result is None

    def test_path_generic_single_part(self) -> None:
        assert derive_arch_group("README.md") is None

    def test_path_skips_generic_prefixes(self) -> None:
        result = derive_arch_group("src/billing/api/handler.py")
        assert result is not None
        domain, container, component = result
        assert domain == "billing"
        assert container == "api"
        assert component == "handler"

    def test_path_skips_packages_prefix(self) -> None:
        result = derive_arch_group("packages/core/contextmine_core/models.py")
        assert result is not None
        domain, container, component = result
        assert domain == "core"
        assert container == "contextmine_core"
        assert component == "models"

    def test_path_non_generic_three_parts(self) -> None:
        result = derive_arch_group("billing/api/handler.py")
        assert result is not None
        domain, container, component = result
        assert domain == "billing"
        assert container == "api"
        assert component == "handler"

    # --- None / empty paths ---

    def test_none_path_none_meta(self) -> None:
        assert derive_arch_group(None) is None

    def test_empty_path(self) -> None:
        assert derive_arch_group("") is None

    def test_slash_only(self) -> None:
        assert derive_arch_group("/") is None

    def test_none_path_with_non_matching_meta(self) -> None:
        """No architecture key in meta and no path -> None."""
        assert derive_arch_group(None, {"other": "stuff"}) is None

    # --- path with leading slashes ---

    def test_leading_slashes_stripped(self) -> None:
        result = derive_arch_group("/apps/web/page.tsx")
        assert result is not None
        assert result[0] == "web"

    # --- services with only two parts ---

    def test_services_only_two_parts(self) -> None:
        assert derive_arch_group("services/billing") is None


# ---------------------------------------------------------------------------
# canonical_file_path_from_node
# ---------------------------------------------------------------------------


class TestCanonicalFilePathFromNode:
    def test_file_kind_with_prefix(self) -> None:
        node = {"kind": "file", "natural_key": "file:src/main.py"}
        assert canonical_file_path_from_node(node) == "src/main.py"

    def test_file_kind_empty_value(self) -> None:
        node = {"kind": "file", "natural_key": "file:"}
        assert canonical_file_path_from_node(node) is None

    def test_file_kind_whitespace_value(self) -> None:
        node = {"kind": "file", "natural_key": "file:   "}
        assert canonical_file_path_from_node(node) is None

    def test_non_file_kind_uses_meta(self) -> None:
        node = {
            "kind": "symbol",
            "natural_key": "symbol:foo",
            "meta": {"file_path": "src/lib.py"},
        }
        assert canonical_file_path_from_node(node) == "src/lib.py"

    def test_non_file_kind_no_meta(self) -> None:
        node = {"kind": "symbol", "natural_key": "symbol:foo"}
        assert canonical_file_path_from_node(node) is None

    def test_non_file_kind_meta_empty_file_path(self) -> None:
        node = {"kind": "symbol", "natural_key": "symbol:foo", "meta": {"file_path": "  "}}
        assert canonical_file_path_from_node(node) is None

    def test_non_file_kind_meta_non_string_file_path(self) -> None:
        node = {"kind": "symbol", "natural_key": "symbol:foo", "meta": {"file_path": 42}}
        assert canonical_file_path_from_node(node) is None

    def test_missing_kind_uses_meta_fallback(self) -> None:
        node = {"meta": {"file_path": "fallback.py"}}
        assert canonical_file_path_from_node(node) == "fallback.py"

    def test_file_kind_case_insensitive(self) -> None:
        node = {"kind": "FILE", "natural_key": "file:test.py"}
        assert canonical_file_path_from_node(node) == "test.py"


# ---------------------------------------------------------------------------
# canonical_file_path_from_meta
# ---------------------------------------------------------------------------


class TestCanonicalFilePathFromMeta:
    def test_path_provided(self) -> None:
        assert canonical_file_path_from_meta("src/main.py") == "src/main.py"

    def test_path_whitespace_stripped(self) -> None:
        assert canonical_file_path_from_meta("  src/main.py  ") == "src/main.py"

    def test_path_empty_falls_to_meta(self) -> None:
        assert canonical_file_path_from_meta("", {"file_path": "from_meta.py"}) == "from_meta.py"

    def test_path_none_falls_to_meta(self) -> None:
        assert canonical_file_path_from_meta(None, {"file_path": "from_meta.py"}) == "from_meta.py"

    def test_no_path_no_meta(self) -> None:
        assert canonical_file_path_from_meta(None) is None

    def test_meta_empty_file_path(self) -> None:
        assert canonical_file_path_from_meta(None, {"file_path": "  "}) is None

    def test_meta_non_string_file_path(self) -> None:
        assert canonical_file_path_from_meta(None, {"file_path": 123}) is None

    def test_meta_none(self) -> None:
        assert canonical_file_path_from_meta(None, None) is None

    def test_path_empty_string_and_meta_none(self) -> None:
        assert canonical_file_path_from_meta("", None) is None
