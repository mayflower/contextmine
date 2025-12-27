"""Application settings using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Database
    database_url: str | None = None

    # GitHub OAuth
    github_client_id: str | None = None
    github_client_secret: str | None = None

    # Public URL for OAuth callbacks
    public_base_url: str = "http://localhost:8000"

    # Session management
    session_secret: str = Field(
        default="dev-session-secret-change-in-production",
        description="Secret key for signing session cookies",
    )

    # Token encryption
    token_encryption_key: str = Field(
        default="dev-encryption-key-change-in-prod",
        description="Key for encrypting OAuth tokens (should be 32 bytes for Fernet)",
    )

    # MCP security
    mcp_allowed_origins: str = Field(
        default="",
        description="Comma-separated list of allowed origins for MCP requests. Empty = allow all (dev mode).",
    )

    # Embedding providers
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for embeddings and LLM",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key for embeddings and LLM",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for LLM",
    )
    default_embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        description="Default embedding model in format 'provider:model_name'",
    )
    default_llm_provider: str = Field(
        default="openai",
        description="Default LLM provider for context assembly (openai, anthropic, gemini)",
    )
    default_llm_model: str = Field(
        default="gpt-5-mini",
        description="Default LLM model for context assembly",
    )

    # Prefect
    prefect_api_url: str = Field(
        default="http://prefect-server:4200/api",
        description="Prefect server API URL",
    )


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
