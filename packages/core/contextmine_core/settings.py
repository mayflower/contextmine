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

    # MCP OAuth (uses same GitHub OAuth app, different callback path)
    mcp_oauth_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for MCP OAuth callbacks. Must match where the server is accessible.",
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
        default=True,
        description="Allow optional LLM enrichment for unresolved architecture mappings",
    )
    arch_docs_drift_enabled: bool = Field(
        default=True,
        description="Enable advisory architecture drift report generation",
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


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
