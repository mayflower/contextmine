"""Behavioral tests for application settings safety."""

import pytest
from contextmine_core.settings import Settings, secret_value
from pydantic import SecretStr, ValidationError


def _settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)


def _safe_production_settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "app_mode": "production",
        "debug": False,
        "session_secret": "production-session-secret",
        "token_encryption_key": "production-encryption-key",
        "mcp_allowed_origins": "https://client.example",
        "cors_allowed_origins": "https://app.example",
        "public_base_url": "https://contextmine.example",
        "mcp_oauth_base_url": "https://contextmine.example",
        "scip_install_deps_mode": "never",
        "sandbox_api_url": "https://agent-sandbox.example",
        "sandbox_api_key": "sandbox-production-token",
        "sandbox_analyzer_snapshot": "contextmine-analyzer-v1",
    }
    values.update(overrides)
    return values


def test_development_defaults_remain_available() -> None:
    settings = _settings()

    assert settings.app_mode == "development"
    assert settings.cors_origins == ["http://localhost:8000"]


def test_secrets_are_masked_and_revealed_only_explicitly() -> None:
    settings = _settings(session_secret="not-for-logs")

    assert isinstance(settings.session_secret, SecretStr)
    assert "not-for-logs" not in repr(settings)
    assert secret_value(settings.session_secret) == "not-for-logs"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"debug": True}, "DEBUG must be disabled"),
        ({"session_secret": "dev-session-secret"}, "SESSION_SECRET"),
        ({"token_encryption_key": "dev-encryption-key"}, "TOKEN_ENCRYPTION_KEY"),
        ({"mcp_allowed_origins": ""}, "MCP_ALLOWED_ORIGINS"),
        ({"cors_allowed_origins": ""}, "CORS_ALLOWED_ORIGINS"),
        ({"public_base_url": "http://localhost:8000"}, "PUBLIC_BASE_URL"),
        ({"mcp_oauth_base_url": "http://localhost:8000"}, "MCP_OAUTH_BASE_URL"),
        ({"scip_install_deps_mode": "auto"}, "SCIP_INSTALL_DEPS_MODE"),
        ({"sandbox_api_url": None}, "SANDBOX_API_URL"),
        ({"sandbox_api_url": "http://localhost:8000"}, "SANDBOX_API_URL"),
        ({"sandbox_api_key": None}, "SANDBOX_API_KEY"),
        ({"sandbox_analyzer_snapshot": None}, "SANDBOX_ANALYZER_SNAPSHOT"),
    ],
)
def test_unsafe_production_settings_fail_closed(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _settings(**_safe_production_settings(**override))


def test_safe_production_settings_are_accepted() -> None:
    settings = _settings(**_safe_production_settings())

    assert settings.app_mode == "production"


def test_production_validation_does_not_echo_secret_values() -> None:
    secret = "dev-secret-that-must-not-leak"

    with pytest.raises(ValidationError) as exc_info:
        _settings(**_safe_production_settings(session_secret=secret, debug=True))

    assert secret not in str(exc_info.value)
