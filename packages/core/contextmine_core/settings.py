"""Application settings using pydantic-settings."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API settings
    app_mode: Literal["development", "test", "production"] = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Database
    database_url: str | None = None
    database_pool_size: int = Field(
        default=20,
        description="SQLAlchemy connection pool size for non-SQLite databases",
    )
    database_max_overflow: int = Field(
        default=40,
        description="Additional overflow connections allowed beyond pool size",
    )
    database_pool_timeout_seconds: int = Field(
        default=120,
        description="Seconds to wait for a DB connection before raising PoolTimeout",
    )

    # GitHub OAuth
    github_client_id: str | None = None
    github_client_secret: SecretStr | None = None

    # Public URL for OAuth callbacks
    public_base_url: str = "http://localhost:8000"

    # Session management
    session_secret: SecretStr = Field(
        default="dev-session-secret-change-in-production",
        description="Secret key for signing session cookies",
    )

    # Token encryption
    token_encryption_key: SecretStr = Field(
        default="dev-encryption-key-change-in-prod",
        description="Key for encrypting OAuth tokens (should be 32 bytes for Fernet)",
    )

    # MCP security
    mcp_allowed_origins: str = Field(
        default="",
        description="Comma-separated list of allowed origins for MCP requests. Empty = allow all (dev mode).",
    )
    cors_allowed_origins: str = Field(
        default="http://localhost:8000",
        description="Comma-separated browser origins allowed to call the API",
    )

    # MCP OAuth (uses same GitHub OAuth app, different callback path)
    mcp_oauth_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for MCP OAuth callbacks. Must match where the server is accessible.",
    )

    # External model policy
    model_calls_enabled: bool = Field(
        default=True,
        description=(
            "Allow external embedding and generative-model calls. When disabled, "
            "sync uses deterministic extraction and retrieval uses full-text search only."
        ),
    )

    # Embedding providers
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for embeddings and LLM",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key for embeddings and LLM",
    )
    anthropic_api_key: SecretStr | None = Field(
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
    openai_llm_model: str = Field(
        default="gpt-5-mini",
        description="Default OpenAI chat model",
    )
    anthropic_llm_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Default Anthropic chat model",
    )
    gemini_llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Default Gemini chat model",
    )

    # Prefect
    prefect_api_url: str = Field(
        default="http://prefect-server:4200/api",
        description="Prefect server API URL",
    )
    prefect_work_pool_name: str = Field(
        default="contextmine-process",
        description="Prefect work pool used by ContextMine deployments",
    )
    prefect_sync_deployment: str = Field(
        default="sync_single_source/default",
        description="Prefect deployment used for source sync runs",
    )
    prefect_due_interval_seconds: int = Field(
        default=60,
        ge=10,
        description="Schedule interval for the due-source dispatcher",
    )
    prefect_worker_limit: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent flow runs for the process worker",
    )

    # Ephemeral repository analyzer
    sandbox_api_url: str | None = Field(
        default=None,
        description="Mayflower Agent Sandbox platform API URL",
    )
    sandbox_api_key: SecretStr | None = Field(
        default=None,
        description="Credential used only by the worker to create analyzer sandboxes",
    )
    sandbox_analyzer_snapshot: str | None = Field(
        default=None,
        description="Published snapshot containing the ContextMine analyzer toolchain",
    )
    sandbox_analysis_timeout_seconds: int = Field(default=3600, ge=60, le=86_400)
    sandbox_result_max_bytes: int = Field(default=64 * 1024 * 1024, ge=1024, le=1024**3)
    sandbox_artifact_max_bytes: int = Field(default=1024**3, ge=1024, le=4 * 1024**3)
    sandbox_analyzer_vcpus: int = Field(default=2, ge=1, le=32)
    sandbox_analyzer_mem_bytes: int = Field(default=8 * 1024**3, ge=256 * 1024**2)
    sandbox_analyzer_fs_bytes: int = Field(default=20 * 1024**3, ge=1024**3)

    # Research Agent
    artifact_store: str = Field(
        default="memory",
        description="Artifact store type: 'memory' or 'file'",
    )
    artifact_dir: str = Field(
        default=".mcp_artifacts",
        description="Directory for file-backed artifact store",
    )
    artifact_ttl_minutes: int = Field(
        default=60,
        description="Time-to-live for artifacts in minutes",
    )
    artifact_max_runs: int = Field(
        default=100,
        description="Maximum number of research runs to keep",
    )
    research_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="LLM model for research agent reasoning",
    )
    research_max_tokens: int = Field(
        default=4096,
        description="Max tokens per research agent LLM call",
    )
    research_budget_steps: int = Field(
        default=10,
        description="Default maximum steps for research agent",
    )

    # LSP Settings
    lsp_idle_timeout_seconds: float = Field(
        default=300.0,
        description="Idle timeout before stopping language servers (seconds)",
    )
    lsp_request_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for individual LSP requests (seconds)",
    )

    # Tree-sitter Settings
    treesitter_cache_size: int = Field(
        default=100,
        description="Maximum number of parsed syntax trees to cache",
    )

    # Graph Settings
    graph_max_depth: int = Field(
        default=3,
        description="Maximum traversal depth for graph expansion",
    )
    graph_max_nodes: int = Field(
        default=100,
        description="Maximum nodes to collect during graph expansion",
    )

    # Verification Settings
    verification_require_citations: bool = Field(
        default=True,
        description="Require citations in answers for verification to pass",
    )
    verification_min_evidence_support: float = Field(
        default=0.5,
        description="Minimum evidence support score (0.0-1.0) for verification to pass",
    )
    verification_confidence_tolerance: float = Field(
        default=0.2,
        description="Tolerance for confidence calibration (|stated - evidence| <= tolerance)",
    )

    # SCIP Polyglot Indexing Settings
    sync_source_timeout_seconds: int = Field(
        default=14400,
        description=(
            "Timeout in seconds for one source sync run in the worker scheduler "
            "(0 disables outer sync timeout)"
        ),
    )
    sync_blocking_step_timeout_seconds: int = Field(
        default=900,
        description=(
            "Timeout in seconds for blocking sync steps (git pull, metrics pipeline, "
            "evolution snapshots) executed in worker thread pool"
        ),
    )
    sync_document_step_timeout_seconds: int = Field(
        default=120,
        description=(
            "Timeout in seconds per document processing step (chunking, symbol extraction, "
            "embedding) to avoid single-document stalls"
        ),
    )
    sync_documents_per_run_limit: int = Field(
        default=400,
        description=(
            "Maximum documents to process per sync run (0 = unlimited). Remaining docs are "
            "recovered automatically in subsequent runs."
        ),
    )
    sync_temporal_coupling_max_files_per_commit: int = Field(
        default=200,
        description=(
            "Maximum files per commit considered for temporal-coupling pair generation "
            "(0 = unlimited)."
        ),
    )
    knowledge_graph_build_timeout_seconds: int = Field(
        default=3600,
        description="Timeout in seconds for knowledge graph build in source sync",
    )
    semantic_extraction_max_chunks: int = Field(
        default=0,
        description=(
            "Optional maximum semantic chunks to extract per run after prioritization "
            "(0 = unlimited)."
        ),
    )
    twin_graph_build_timeout_seconds: int = Field(
        default=3600,
        description="Timeout in seconds for digital twin graph build step",
    )
    embedding_batch_timeout_seconds: int = Field(
        default=120,
        description="Timeout in seconds for a single embedding batch request",
    )
    scip_languages: str = Field(
        default="python,typescript,javascript,java,php",
        description="Comma-separated list of enabled languages for SCIP indexing",
    )
    scip_install_deps_mode: str = Field(
        default="auto",
        description="Dependency installation mode: auto, always, or never",
    )
    scip_timeout_python: int = Field(
        default=300,
        description="Timeout in seconds for Python SCIP indexing",
    )
    scip_timeout_typescript: int = Field(
        default=600,
        description="Timeout in seconds for TypeScript/JavaScript SCIP indexing",
    )
    scip_timeout_java: int = Field(
        default=900,
        description="Timeout in seconds for Java SCIP indexing",
    )
    scip_timeout_php: int = Field(
        default=900,
        description="Timeout in seconds for PHP SCIP indexing",
    )
    scip_node_memory_mb: int = Field(
        default=4096,
        description="Node.js memory limit in MB for TS/JS/Python indexers",
    )
    scip_best_effort: bool = Field(
        default=True,
        description="Continue indexing other projects if one fails",
    )
    scip_require_language_coverage: bool = Field(
        default=True,
        description="Fail sync when any detected supported language has zero indexed files",
    )
    scip_require_relation_coverage: bool = Field(
        default=True,
        description=(
            "Fail sync when indexed languages have no semantic relations "
            "(calls/references/imports/extends/implements)"
        ),
    )
    scip_require_php_relation_coverage: bool = Field(
        default=True,
        description=(
            "Fail sync when PHP is indexed but semantic relations "
            "(calls/references/imports) are missing"
        ),
    )

    # Real metrics pipeline settings
    metrics_strict_mode: bool = Field(
        default=True,
        description="Require real LOC/complexity/coupling/coverage for relevant GitHub files",
    )
    metrics_languages: str = Field(
        default="python,typescript,javascript,java,php",
        description="Comma-separated language scope for real metrics extraction",
    )
    metrics_autodiscovery_enabled: bool = Field(
        default=True,
        description="Enable fallback auto-discovery for coverage reports when no config patterns match",
    )
    coverage_ingest_max_payload_mb: int = Field(
        default=25,
        description="Maximum total multipart payload size for CI coverage ingest endpoint",
    )
    coverage_ingest_prefect_flow_name: str = Field(
        default="ingest_coverage_metrics",
        description="Prefect flow name used for asynchronous coverage ingest processing",
    )
    twin_analysis_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL for cached twin analysis query payloads in seconds",
    )
    twin_event_retention_days: int = Field(
        default=90,
        description="Retention target for twin materialization lifecycle events",
    )
    arch_docs_enabled: bool = Field(
        default=True,
        description="Enable architecture facts, arc42 generation, and ports/adapters outputs",
    )
    arch_docs_llm_enrich: bool = Field(
        default=False,
        description="Allow optional LLM enrichment for unresolved architecture mappings",
    )
    arch_docs_drift_enabled: bool = Field(
        default=True,
        description="Enable advisory architecture drift report generation",
    )
    arch_docs_llm_max_hypotheses: int = Field(
        default=12,
        description="Maximum number of ambiguous architecture hypotheses to adjudicate with the LLM per build",
    )
    arch_docs_agent_sdk_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model used by arc42 agent-sdk generation",
    )
    arch_docs_agent_sdk_max_turns: int = Field(
        default=50,
        description="Maximum turns for arc42 agent-sdk generation",
    )
    arch_docs_generate_on_sync: bool = Field(
        default=True,
        description=(
            "Generate/update arc42 artifacts during sync runs. "
            "Set to false to defer arc42 generation to explicit MCP trigger."
        ),
    )
    twin_evolution_view_enabled: bool = Field(
        default=True,
        description="Enable evolution cockpit view endpoints (investment, ownership, coupling, fitness)",
    )
    twin_evolution_window_days: int = Field(
        default=365,
        description="Time window in days for git-based evolution analytics",
    )
    digital_twin_behavioral_enabled: bool = Field(
        default=True,
        description="Enable behavioral twin extraction (tests, UI, interface contracts)",
    )
    digital_twin_ui_enabled: bool = Field(
        default=True,
        description="Enable UI-map extraction and projection layers",
    )
    digital_twin_flows_enabled: bool = Field(
        default=True,
        description="Enable synthesized user-flow extraction and projection layers",
    )
    joern_server_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for Joern HTTP server used by twin analysis endpoints",
    )
    joern_query_timeout_seconds: int = Field(
        default=120,
        description="Timeout for Joern query execution in seconds",
    )
    joern_parse_timeout_seconds: int = Field(
        default=900,
        description="Timeout for Joern CPG parse execution in seconds",
    )
    joern_required_for_sync: bool = Field(
        default=False,
        description="Require Joern CPG generation for sync success (false = advisory)",
    )
    joern_parse_binary: str = Field(
        default="joern-parse",
        description="Binary name/path for Joern CPG generation",
    )
    joern_cpg_root: str = Field(
        default="/data/joern-cpg",
        description="Filesystem root for generated Joern CPG artifacts",
    )
    repos_root: str = Field(
        default="/data/repos",
        description="Filesystem root where GitHub sources are checked out for LSP analysis",
    )

    # OpenTelemetry Settings (disabled by default - no overhead when disabled)
    otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry instrumentation (traces, metrics, logs)",
    )
    otel_service_name: str = Field(
        default="contextmine",
        description="Service name for OTEL (will be suffixed with -api, -worker)",
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP exporter endpoint (gRPC). E.g., http://tempo:4317",
    )
    otel_exporter_otlp_protocol: str = Field(
        default="grpc",
        description="OTLP protocol: 'grpc' or 'http/protobuf'",
    )
    otel_traces_sampler: str = Field(
        default="parentbased_traceidratio",
        description="Trace sampler: always_on, always_off, traceidratio, parentbased_traceidratio",
    )
    otel_traces_sampler_arg: float = Field(
        default=1.0,
        description="Sampler argument (e.g., 0.1 for 10% sampling)",
    )
    otel_log_level: str = Field(
        default="INFO",
        description="Minimum log level to export via OTEL: DEBUG, INFO, WARNING, ERROR",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Return configured CORS origins without empty entries."""
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def validate_production_safety(self) -> Settings:
        """Reject development-only behavior in production."""
        if self.app_mode != "production":
            return self

        errors: list[str] = []
        if self.debug:
            errors.append("DEBUG must be disabled")
        if secret_value(self.session_secret).startswith("dev-"):
            errors.append("SESSION_SECRET must not use the development default")
        if secret_value(self.token_encryption_key).startswith("dev-"):
            errors.append("TOKEN_ENCRYPTION_KEY must not use the development default")
        if not [origin.strip() for origin in self.mcp_allowed_origins.split(",") if origin.strip()]:
            errors.append("MCP_ALLOWED_ORIGINS must not be empty")
        if not self.cors_origins:
            errors.append("CORS_ALLOWED_ORIGINS must not be empty")
        if self.scip_install_deps_mode != "never":
            errors.append("SCIP_INSTALL_DEPS_MODE must be 'never'")
        if not self.sandbox_api_url:
            errors.append("SANDBOX_API_URL is required")
        else:
            parsed_sandbox_url = urlsplit(self.sandbox_api_url)
            if parsed_sandbox_url.scheme != "https" or parsed_sandbox_url.hostname in {
                None,
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
            }:
                errors.append("SANDBOX_API_URL must be an external HTTPS URL")
        if not secret_value(self.sandbox_api_key):
            errors.append("SANDBOX_API_KEY is required")
        if not self.sandbox_analyzer_snapshot:
            errors.append("SANDBOX_ANALYZER_SNAPSHOT is required")
        for name, value in (
            ("PUBLIC_BASE_URL", self.public_base_url),
            ("MCP_OAUTH_BASE_URL", self.mcp_oauth_base_url),
        ):
            parsed = urlsplit(value)
            if parsed.scheme != "https" or parsed.hostname in {
                None,
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
            }:
                errors.append(f"{name} must be an external HTTPS URL")
        if errors:
            raise ValueError("Unsafe production configuration: " + "; ".join(errors))
        return self


def secret_value(value: SecretStr | str | None) -> str | None:
    """Reveal a secret only at the boundary that consumes it."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
