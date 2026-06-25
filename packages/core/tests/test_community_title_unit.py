"""Unit tests for community title resolution.

Regression guard for the bug where the LLM-generated community title was computed
and then discarded, leaving the keyword-derived title in KnowledgeCommunity.title.
"""

from __future__ import annotations

from contextmine_core.knowledge.summaries import (
    CommunitySummaryOutput,
    _resolve_community_title,
)


def _summary(title: str) -> CommunitySummaryOutput:
    return CommunitySummaryOutput(
        title=title,
        responsibilities=["does a thing"],
        key_concepts=["concept"],
        key_dependencies=[],
        key_paths=[],
        confidence=0.9,
    )


def test_prefers_llm_title_over_existing() -> None:
    resolved = _resolve_community_title(_summary("Authentication Service"), "user_login, session")
    assert resolved == "Authentication Service"


def test_strips_whitespace() -> None:
    resolved = _resolve_community_title(_summary("  Billing Pipeline  "), "old")
    assert resolved == "Billing Pipeline"


def test_falls_back_to_existing_when_llm_title_blank() -> None:
    resolved = _resolve_community_title(_summary("   "), "user_login, session +2 more")
    assert resolved == "user_login, session +2 more"


def test_empty_when_both_blank() -> None:
    assert _resolve_community_title(_summary(""), None) == ""
