"""Tests for detecting a missing embedding provider key up front."""

from __future__ import annotations

import pytest
from contextmine_core.embeddings import embedding_credential_available
from contextmine_core.models import EmbeddingProvider
from contextmine_core.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


class TestEmbeddingCredentialAvailable:
    def test_reports_openai_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "contextmine_core.embeddings.get_settings",
            lambda: _settings(openai_api_key="sk-test"),
        )
        assert embedding_credential_available(EmbeddingProvider.OPENAI) is True

    def test_reports_openai_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "contextmine_core.embeddings.get_settings",
            lambda: _settings(openai_api_key=None, gemini_api_key="g-test"),
        )
        # A Gemini key does not make the OpenAI embedder usable.
        assert embedding_credential_available(EmbeddingProvider.OPENAI) is False

    def test_reports_gemini_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "contextmine_core.embeddings.get_settings",
            lambda: _settings(gemini_api_key="g-test"),
        )
        assert embedding_credential_available(EmbeddingProvider.GEMINI) is True

    def test_accepts_a_provider_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "contextmine_core.embeddings.get_settings",
            lambda: _settings(openai_api_key="sk-test"),
        )
        assert embedding_credential_available("openai") is True

    def test_rejects_an_unknown_provider_name(self) -> None:
        assert embedding_credential_available("no-such-provider") is False
