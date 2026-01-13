"""Tests for the telemetry module."""

import asyncio
from unittest.mock import patch

import pytest


def run_async(coro):
    """Helper to run async functions in tests."""
    return asyncio.run(coro)


class TestTelemetryDisabled:
    """Tests for telemetry when disabled (default)."""

    def test_get_tracer_returns_noop_when_disabled(self) -> None:
        """get_tracer should return a no-op tracer when telemetry is disabled."""
        from contextmine_core.telemetry.setup import get_tracer

        tracer = get_tracer("test")
        # Should not raise, returns no-op tracer from opentelemetry
        assert tracer is not None

    def test_get_meter_returns_noop_when_disabled(self) -> None:
        """get_meter should return a no-op meter when telemetry is disabled."""
        from contextmine_core.telemetry.setup import get_meter

        meter = get_meter("test")
        # Should not raise, returns no-op meter from opentelemetry
        assert meter is not None


class TestSpanHelpers:
    """Tests for span helper decorators and context managers."""

    def test_trace_embedding_call_creates_span(self) -> None:
        """trace_embedding_call should create a span with correct attributes."""
        from contextmine_core.telemetry.spans import trace_embedding_call

        async def test():
            async with trace_embedding_call("openai", "text-embedding-3-small", 10) as span:
                # Span should be active
                assert span is not None
                # Should be able to set attributes
                span.set_attribute("embedding.tokens_used", 100)

        run_async(test())

    def test_trace_db_operation_creates_span(self) -> None:
        """trace_db_operation should create a span with correct attributes."""
        from contextmine_core.telemetry.spans import trace_db_operation

        async def test():
            async with trace_db_operation("SELECT", "chunks") as span:
                assert span is not None
                span.set_attribute("db.rows_affected", 5)

        run_async(test())

    def test_trace_embedding_call_records_exception(self) -> None:
        """trace_embedding_call should record exceptions on the span."""
        from contextmine_core.telemetry.spans import trace_embedding_call

        async def test():
            with pytest.raises(ValueError):
                async with trace_embedding_call("openai", "test-model", 1):
                    raise ValueError("Test error")

        run_async(test())

    def test_trace_llm_call_decorator(self) -> None:
        """trace_llm_call decorator should wrap async functions."""
        from contextmine_core.telemetry.spans import trace_llm_call

        @trace_llm_call(provider="anthropic", model="claude-3")
        async def mock_llm_call() -> str:
            return "response"

        result = run_async(mock_llm_call())
        assert result == "response"

    def test_trace_sync_llm_call_decorator(self) -> None:
        """trace_sync_llm_call decorator should wrap sync functions."""
        from contextmine_core.telemetry.spans import trace_sync_llm_call

        @trace_sync_llm_call(provider="openai", model="gpt-4")
        def mock_sync_llm_call() -> str:
            return "sync_response"

        result = mock_sync_llm_call()
        assert result == "sync_response"


class TestSettingsIntegration:
    """Tests for OTEL settings integration."""

    def test_otel_settings_have_defaults(self) -> None:
        """OTEL settings should have sensible defaults."""
        from contextmine_core.settings import Settings

        settings = Settings()

        assert settings.otel_enabled is False
        assert settings.otel_service_name == "contextmine"
        assert settings.otel_exporter_otlp_endpoint == "http://localhost:4317"
        assert settings.otel_exporter_otlp_protocol == "grpc"
        assert settings.otel_traces_sampler == "parentbased_traceidratio"
        assert settings.otel_traces_sampler_arg == 1.0
        assert settings.otel_log_level == "INFO"

    def test_otel_settings_from_env(self) -> None:
        """OTEL settings should be configurable via environment variables."""
        import os

        with patch.dict(
            os.environ,
            {
                "OTEL_ENABLED": "true",
                "OTEL_SERVICE_NAME": "my-service",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://tempo:4317",
                "OTEL_TRACES_SAMPLER_ARG": "0.5",
            },
        ):
            from contextmine_core.settings import Settings

            settings = Settings()

            assert settings.otel_enabled is True
            assert settings.otel_service_name == "my-service"
            assert settings.otel_exporter_otlp_endpoint == "http://tempo:4317"
            assert settings.otel_traces_sampler_arg == 0.5
