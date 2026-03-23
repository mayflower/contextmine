"""Tests for traceability pure helpers and data classes.

Covers:
- _canonical_path
- _normalize_symbol_token
- _extract_token_variants / symbol_token_variants
- _parse_uuid
- CallSite / ResolvedSymbolRef data classes
- _remember_best (via SymbolTraceResolver)
- _find_knowledge_symbol_by_location (via SymbolTraceResolver)
- resolve_symbol_refs_for_calls normalisation of call sites
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from contextmine_core.analyzer.extractors.traceability import (
    CallSite,
    ResolvedSymbolRef,
    SymbolTraceResolver,
    _canonical_path,
    _extract_token_variants,
    _KnowledgeSymbolRow,
    _normalize_symbol_token,
    _parse_uuid,
    resolve_symbol_refs_for_calls,
    symbol_token_variants,
)

# ---------------------------------------------------------------------------
# _canonical_path
# ---------------------------------------------------------------------------


class TestCanonicalPath:
    def test_strips_whitespace(self) -> None:
        assert _canonical_path("  src/main.py  ") == "src/main.py"

    def test_converts_backslash_to_forward(self) -> None:
        assert _canonical_path("src\\models\\user.py") == "src/models/user.py"

    def test_removes_query_string(self) -> None:
        assert _canonical_path("src/file.py?v=2") == "src/file.py"

    def test_strips_leading_dot_slash(self) -> None:
        assert _canonical_path("./src/file.py") == "src/file.py"

    def test_empty_string(self) -> None:
        assert _canonical_path("") == ""

    def test_none_becomes_empty(self) -> None:
        assert _canonical_path(None) == ""

    def test_dot_slash_and_backslash_combined(self) -> None:
        assert _canonical_path(".\\src\\file.py") == "src/file.py"

    def test_preserves_absolute_path(self) -> None:
        assert _canonical_path("/home/user/project/file.py") == "/home/user/project/file.py"


# ---------------------------------------------------------------------------
# _normalize_symbol_token
# ---------------------------------------------------------------------------


class TestNormalizeSymbolToken:
    def test_simple_name(self) -> None:
        assert _normalize_symbol_token("myFunction") == "myfunction"

    def test_dotted_qualified_name(self) -> None:
        assert _normalize_symbol_token("com.example.MyClass.myMethod") == "mymethod"

    def test_double_colon_separator(self) -> None:
        assert _normalize_symbol_token("std::string::npos") == "npos"

    def test_hash_separator(self) -> None:
        assert _normalize_symbol_token("MyClass#doSomething") == "dosomething"

    def test_slash_separator(self) -> None:
        assert _normalize_symbol_token("pkg/utils/helper") == "helper"

    def test_colon_separator(self) -> None:
        assert _normalize_symbol_token("module:func") == "func"

    def test_empty_returns_empty(self) -> None:
        assert _normalize_symbol_token("") == ""

    def test_none_returns_empty(self) -> None:
        assert _normalize_symbol_token(None) == ""

    def test_whitespace_only(self) -> None:
        assert _normalize_symbol_token("   ") == ""

    def test_strips_surrounding_spaces(self) -> None:
        assert _normalize_symbol_token("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# _extract_token_variants / symbol_token_variants
# ---------------------------------------------------------------------------


class TestExtractTokenVariants:
    def test_simple_name(self) -> None:
        variants = _extract_token_variants("hello")
        assert "hello" in variants

    def test_camel_case_produces_snake_variant(self) -> None:
        variants = _extract_token_variants("getUserById")
        assert "getuserbyid" in variants  # compact
        assert "get_user_by_id" in variants  # snake

    def test_dotted_name(self) -> None:
        variants = _extract_token_variants("MyClass.doWork")
        # base (last segment, lowercased): "dowork"
        assert "dowork" in variants

    def test_empty_returns_empty_set(self) -> None:
        assert _extract_token_variants("") == set()

    def test_none_returns_empty_set(self) -> None:
        assert _extract_token_variants(None) == set()

    def test_whitespace_returns_empty_set(self) -> None:
        assert _extract_token_variants("   ") == set()

    def test_snake_case_name(self) -> None:
        variants = _extract_token_variants("get_user_by_id")
        assert "get_user_by_id" in variants

    def test_symbol_token_variants_delegates(self) -> None:
        """symbol_token_variants is a public wrapper."""
        assert symbol_token_variants("getUserById") == _extract_token_variants("getUserById")

    def test_upper_case(self) -> None:
        variants = _extract_token_variants("CONSTANT_NAME")
        assert "constant_name" in variants

    def test_mixed_separators(self) -> None:
        variants = _extract_token_variants("com.example::MyModule#processData")
        # base = last colon-separated, so "processdata"
        assert "processdata" in variants


# ---------------------------------------------------------------------------
# _parse_uuid
# ---------------------------------------------------------------------------


class TestParseUUID:
    def test_valid_uuid_string(self) -> None:
        uid = uuid4()
        assert _parse_uuid(str(uid)) == uid

    def test_none_returns_none(self) -> None:
        assert _parse_uuid(None) is None

    def test_invalid_string_returns_none(self) -> None:
        assert _parse_uuid("not-a-uuid") is None

    def test_integer_returns_none(self) -> None:
        assert _parse_uuid(12345) is None

    def test_uuid_object(self) -> None:
        uid = uuid4()
        assert _parse_uuid(uid) == uid

    def test_empty_string_returns_none(self) -> None:
        assert _parse_uuid("") is None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestCallSite:
    def test_frozen(self) -> None:
        cs = CallSite(file_path="a.py", line=10, column=5, callee="foo")
        assert cs.file_path == "a.py"
        assert cs.line == 10
        assert cs.column == 5
        assert cs.callee == "foo"
        with pytest.raises(AttributeError):
            cs.line = 20


class TestResolvedSymbolRef:
    def test_frozen(self) -> None:
        uid = uuid4()
        ref = ResolvedSymbolRef(
            symbol_node_id=uid,
            symbol_name="foo",
            engine="scip.calls",
            confidence=0.94,
            natural_key="symbol:foo",
        )
        assert ref.symbol_node_id == uid
        with pytest.raises(AttributeError):
            ref.confidence = 0.5


# ---------------------------------------------------------------------------
# SymbolTraceResolver._remember_best
# ---------------------------------------------------------------------------


class TestRememberBest:
    def _make_resolver(self) -> SymbolTraceResolver:
        return SymbolTraceResolver(
            session=None,
            collection_id=uuid4(),
        )

    def test_first_candidate_is_added(self) -> None:
        resolver = self._make_resolver()
        uid = uuid4()
        bag: dict[UUID, ResolvedSymbolRef] = {}
        ref = ResolvedSymbolRef(
            symbol_node_id=uid,
            symbol_name="foo",
            engine="scip.calls",
            confidence=0.80,
            natural_key="symbol:foo",
        )
        resolver._remember_best(bag, ref)
        assert bag[uid] is ref

    def test_higher_confidence_replaces(self) -> None:
        resolver = self._make_resolver()
        uid = uuid4()
        bag: dict[UUID, ResolvedSymbolRef] = {}

        low = ResolvedSymbolRef(
            symbol_node_id=uid,
            symbol_name="foo",
            engine="name_fallback",
            confidence=0.45,
            natural_key="symbol:foo",
        )
        high = ResolvedSymbolRef(
            symbol_node_id=uid,
            symbol_name="foo",
            engine="scip.calls",
            confidence=0.94,
            natural_key="symbol:foo",
        )
        resolver._remember_best(bag, low)
        resolver._remember_best(bag, high)
        assert bag[uid].confidence == 0.94

    def test_lower_confidence_does_not_replace(self) -> None:
        resolver = self._make_resolver()
        uid = uuid4()
        bag: dict[UUID, ResolvedSymbolRef] = {}

        high = ResolvedSymbolRef(
            symbol_node_id=uid,
            symbol_name="foo",
            engine="scip.calls",
            confidence=0.94,
            natural_key="symbol:foo",
        )
        low = ResolvedSymbolRef(
            symbol_node_id=uid,
            symbol_name="foo",
            engine="name_fallback",
            confidence=0.45,
            natural_key="symbol:foo",
        )
        resolver._remember_best(bag, high)
        resolver._remember_best(bag, low)
        assert bag[uid].confidence == 0.94


# ---------------------------------------------------------------------------
# SymbolTraceResolver._find_knowledge_symbol_by_location
# ---------------------------------------------------------------------------


class TestFindKnowledgeSymbolByLocation:
    def _make_resolver(self) -> SymbolTraceResolver:
        resolver = SymbolTraceResolver(
            session=None,
            collection_id=uuid4(),
        )
        # Prepopulate the internal index
        uid1 = uuid4()
        uid2 = uuid4()
        row1 = _KnowledgeSymbolRow(
            node_id=uid1,
            name="funcA",
            natural_key="symbol:funcA",
            file_path="src/main.py",
            start_line=10,
            end_line=20,
            def_id=None,
        )
        row2 = _KnowledgeSymbolRow(
            node_id=uid2,
            name="innerFunc",
            natural_key="symbol:innerFunc",
            file_path="src/main.py",
            start_line=12,
            end_line=15,
            def_id=None,
        )
        resolver._kg_by_file["src/main.py"] = [row1, row2]
        return resolver

    def test_finds_narrowest_match(self) -> None:
        resolver = self._make_resolver()
        result = resolver._find_knowledge_symbol_by_location(file_path="src/main.py", line=13)
        assert result is not None
        assert result.name == "innerFunc"

    def test_returns_none_for_outside_range(self) -> None:
        resolver = self._make_resolver()
        result = resolver._find_knowledge_symbol_by_location(file_path="src/main.py", line=50)
        assert result is None

    def test_returns_none_for_unknown_file(self) -> None:
        resolver = self._make_resolver()
        result = resolver._find_knowledge_symbol_by_location(file_path="src/other.py", line=13)
        assert result is None

    def test_resolves_absolute_path_to_relative(self, tmp_path: Path) -> None:
        resolver = self._make_resolver()
        resolver._repo_root = tmp_path
        # The resolver uses str(repo_root.resolve()) to strip the prefix.
        # Build the absolute path using the resolved tmp_path so it matches.
        resolved_root = str(tmp_path.resolve()).replace("\\", "/")
        abs_file = f"{resolved_root}/src/main.py"
        result = resolver._find_knowledge_symbol_by_location(file_path=abs_file, line=13)
        assert result is not None
        assert result.name == "innerFunc"


# ---------------------------------------------------------------------------
# resolve_symbol_refs_for_calls (normalisation of input call sites)
# ---------------------------------------------------------------------------


class TestResolveSymbolRefsForCalls:
    @pytest.mark.anyio
    async def test_skips_sites_with_zero_line(self) -> None:
        """Call sites with line <= 0 are dropped before resolution."""
        from unittest.mock import AsyncMock, patch

        mock_session = AsyncMock()
        # Patch resolve_many to capture what it receives
        with patch.object(
            SymbolTraceResolver, "resolve_many", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = []
            await resolve_symbol_refs_for_calls(
                session=mock_session,
                collection_id=uuid4(),
                source_id=None,
                file_path="test.py",
                call_sites=[
                    {"line": 0, "column": 0, "callee": "bad"},
                    {"line": -1, "column": 0, "callee": "worse"},
                    {"line": 10, "column": 5, "callee": "good"},
                ],
                fallback_symbol_hints=[],
            )
            # Only the valid call site (line=10) should reach resolve_many
            call_args = mock_resolve.call_args
            assert len(call_args.kwargs["call_sites"]) == 1
            assert call_args.kwargs["call_sites"][0].line == 10

    @pytest.mark.anyio
    async def test_uses_file_path_from_outer_argument(self) -> None:
        """All normalised call sites get the outer file_path."""
        from unittest.mock import AsyncMock, patch

        mock_session = AsyncMock()
        with patch.object(
            SymbolTraceResolver, "resolve_many", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = []
            await resolve_symbol_refs_for_calls(
                session=mock_session,
                collection_id=uuid4(),
                source_id=None,
                file_path="outer.py",
                call_sites=[{"line": 5, "column": 0, "callee": "fn"}],
                fallback_symbol_hints=[],
            )
            sites = mock_resolve.call_args.kwargs["call_sites"]
            assert sites[0].file_path == "outer.py"

    @pytest.mark.anyio
    async def test_defaults_column_and_callee(self) -> None:
        """Missing column / callee default to 0 / empty."""
        from unittest.mock import AsyncMock, patch

        mock_session = AsyncMock()
        with patch.object(
            SymbolTraceResolver, "resolve_many", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = []
            await resolve_symbol_refs_for_calls(
                session=mock_session,
                collection_id=uuid4(),
                source_id=None,
                file_path="test.py",
                call_sites=[{"line": 7}],
                fallback_symbol_hints=[],
            )
            site = mock_resolve.call_args.kwargs["call_sites"][0]
            assert site.column == 0
            assert site.callee == ""
