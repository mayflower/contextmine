"""Prefect flows for syncing sources.

Features:
- Native Prefect scheduling (no custom scheduler)
- Automatic retries with exponential backoff
- Concurrency limits via tags (github-api, embedding-api, web-crawl)
- Task result caching for idempotent operations
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from contextmine_core import (
    Chunk,
    Collection,
    CoverageIngestJob,
    CoverageIngestReport,
    Document,
    EmbeddingModel,
    EmbeddingProvider,
    OAuthToken,
    Source,
    SourceType,
    SyncRun,
    SyncRunStatus,
    decrypt_token,
    get_session,
    get_settings,
)
from prefect import flow, task
from prefect.artifacts import create_progress_artifact, update_progress_artifact
from prefect.cache_policies import INPUTS
from prefect.tasks import exponential_backoff
from sqlalchemy import delete, select, text

from contextmine_worker.chunking import chunk_document
from contextmine_worker.embeddings import FakeEmbedder, get_embedder, parse_embedding_model_spec
from contextmine_worker.github_sync import (
    SyncStats,
    build_uri,
    clone_or_pull_repo,
    compute_content_hash,
    compute_git_change_metrics,
    compute_git_evolution_snapshots,
    ensure_repos_dir,
    get_changed_files,
    get_file_title,
    get_repo_path,
    is_eligible_file,
    read_file_content,
)
from contextmine_worker.symbol_indexing import maintain_symbols_for_document
from contextmine_worker.telemetry import traced_flow, traced_task
from contextmine_worker.web_sync import (
    DEFAULT_DELAY_MS,
    DEFAULT_MAX_PAGES,
    WebSyncStats,
    get_page_title,
    run_spider_md,
)

logger = logging.getLogger(__name__)

# SCIP Polyglot Indexing Tag
TAG_SCIP_INDEX = "scip-index"

# Retry configuration
DEFAULT_RETRIES = 2
GITHUB_API_RETRIES = 3
EMBEDDING_API_RETRIES = 2

# Concurrency limit tags (set limits via: prefect concurrency-limit create <tag> <limit>)
# Example: prefect concurrency-limit create github-api 5
TAG_GITHUB_API = "github-api"
TAG_EMBEDDING_API = "embedding-api"
TAG_WEB_CRAWL = "web-crawl"
TAG_DB_HEAVY = "db-heavy"
SYNC_RUN_STALE_AFTER = timedelta(hours=6)
_FILE_NODE_PREFIX = "file:"
IGNORED_REPO_PATH_PARTS = frozenset(
    {
        "node_modules",
        "vendor",
        "dist",
        "build",
        "__pycache__",
        ".git",
        "venv",
        ".venv",
    }
)


def _sync_source_timeout_seconds() -> int:
    configured = int(get_settings().sync_source_timeout_seconds)
    # 0 disables the outer flow timeout and relies on step-level guards.
    return max(0, configured)


def _embedding_batch_timeout_seconds() -> int:
    configured = int(get_settings().embedding_batch_timeout_seconds)
    return max(10, configured)


def _knowledge_graph_build_timeout_seconds() -> int:
    configured = int(get_settings().knowledge_graph_build_timeout_seconds)
    return max(120, configured)


def _twin_graph_build_timeout_seconds() -> int:
    configured = int(get_settings().twin_graph_build_timeout_seconds)
    return max(120, configured)


def _sync_blocking_step_timeout_seconds() -> int:
    configured = int(get_settings().sync_blocking_step_timeout_seconds)
    return max(30, configured)


def _sync_document_step_timeout_seconds() -> int:
    configured = int(get_settings().sync_document_step_timeout_seconds)
    return max(10, configured)


def _sync_documents_per_run_limit() -> int:
    configured = int(get_settings().sync_documents_per_run_limit)
    return max(0, configured)


def _sync_temporal_coupling_max_files_per_commit() -> int:
    configured = int(get_settings().sync_temporal_coupling_max_files_per_commit)
    return max(0, configured)


def _joern_parse_timeout_seconds() -> int:
    configured = int(get_settings().joern_parse_timeout_seconds)
    return max(30, configured)


async def _run_blocking_with_timeout(
    step_name: str,
    timeout_seconds: int,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=max(1, int(timeout_seconds)),
        )
    except TimeoutError as exc:
        raise RuntimeError(f"STEP_TIMEOUT: {step_name} exceeded {int(timeout_seconds)}s") from exc


def _log_background_task_failure(task: "asyncio.Task[object]") -> None:
    if task.cancelled():
        logger.warning("Behavioral layer background extraction task was cancelled.")
        return
    exc = task.exception()
    if exc:
        logger.warning("Behavioral layer background extraction failed: %s", exc)


def _uri_to_file_path(uri: str) -> str:
    """Normalize document URI to repo-relative file path."""
    return uri.split("?")[0].split("/", 5)[-1] if "/" in uri else uri


def _is_ignored_repo_path(file_path: str) -> bool:
    """Return True when a repo-relative path belongs to generated/dependency dirs."""
    normalized = file_path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if any(part in IGNORED_REPO_PATH_PARTS for part in parts):
        return True
    marker_path = f"/{normalized}/"
    return "/src/libs/" in marker_path


@dataclass
class _DocProcessingAccumulator:
    """Accumulates stats from the per-document chunk/symbol/embed loop."""

    chunks_created: int = 0
    chunks_deleted: int = 0
    chunks_embedded: int = 0
    chunks_deduplicated: int = 0
    tokens_used: int = 0
    symbols_created: int = 0
    symbols_deleted: int = 0
    processing_failures: int = 0
    processing_timeouts: int = 0
    error_samples: list = field(default_factory=list)  # type: ignore[assignment]

    def record_error(self, value: str) -> None:
        if len(self.error_samples) < 100:
            self.error_samples.append(value)


async def _process_single_document(
    acc: _DocProcessingAccumulator,
    doc_id: str,
    content: str,
    file_path: str | None,
    collection_id_str: str,
    per_doc_timeout_seconds: int,
) -> None:
    """Run chunk, symbol, and embed steps for a single document."""
    import uuid as uuid_module

    # Chunk
    try:
        chunk_stats = await asyncio.wait_for(
            maintain_chunks_for_document(doc_id, content, file_path),
            timeout=per_doc_timeout_seconds,
        )
    except TimeoutError:
        acc.processing_timeouts += 1
        acc.processing_failures += 1
        acc.record_error(f"{doc_id}:chunk_timeout")
        return
    except Exception as exc:  # noqa: BLE001
        acc.processing_failures += 1
        acc.record_error(f"{doc_id}:chunk_error:{exc}")
        return
    acc.chunks_created += chunk_stats["chunks_created"]
    acc.chunks_deleted += chunk_stats["chunks_deleted"]

    # Symbol extraction
    try:
        async with get_session() as session:
            sym_created, sym_deleted = await asyncio.wait_for(
                maintain_symbols_for_document(session, uuid_module.UUID(doc_id)),
                timeout=per_doc_timeout_seconds,
            )
            acc.symbols_created += sym_created
            acc.symbols_deleted += sym_deleted
            await session.commit()
    except TimeoutError:
        acc.processing_timeouts += 1
        acc.record_error(f"{doc_id}:symbol_timeout")
    except Exception as exc:  # noqa: BLE001
        acc.processing_failures += 1
        acc.record_error(f"{doc_id}:symbol_error:{exc}")

    # Embed
    try:
        embed_stats = await asyncio.wait_for(
            embed_document(doc_id, collection_id_str),
            timeout=per_doc_timeout_seconds,
        )
    except TimeoutError:
        acc.processing_timeouts += 1
        acc.record_error(f"{doc_id}:embed_timeout")
        return
    except Exception as exc:  # noqa: BLE001
        acc.processing_failures += 1
        acc.record_error(f"{doc_id}:embed_error:{exc}")
        return
    acc.chunks_embedded += embed_stats["chunks_embedded"]
    acc.chunks_deduplicated += embed_stats.get("chunks_deduplicated", 0)
    acc.tokens_used += embed_stats["tokens_used"]


async def _process_document_batch(
    docs_to_chunk: list[tuple[str, str, str | None]],
    collection_id_str: str,
    progress_id: object,
) -> _DocProcessingAccumulator:
    """Process a batch of documents: chunk, extract symbols, embed."""
    acc = _DocProcessingAccumulator()
    docs_limit = _sync_documents_per_run_limit()
    total_candidate = len(docs_to_chunk)
    docs_deferred = 0
    if docs_limit and total_candidate > docs_limit:
        docs_to_chunk = docs_to_chunk[:docs_limit]
        docs_deferred = total_candidate - len(docs_to_chunk)

    total_docs = len(docs_to_chunk)
    per_doc_timeout = _sync_document_step_timeout_seconds()

    if docs_deferred > 0:
        await update_progress_artifact(  # type: ignore[misc]
            progress_id,
            progress=49,
            description=f"Deferring {docs_deferred} documents to later runs (limit={docs_limit} docs/run).",
        )
    if total_docs > 0:
        await update_progress_artifact(  # type: ignore[misc]
            progress_id,
            progress=50,
            description=f"Chunking and embedding {total_docs} documents...",
        )

    for i, (doc_id, content, file_path) in enumerate(docs_to_chunk):
        if total_docs > 0 and (i % 5 == 0 or i == total_docs - 1):
            pct = 50 + int((i + 1) / total_docs * 45)
            await update_progress_artifact(  # type: ignore[misc]
                progress_id,
                progress=pct,
                description=f"Processing document {i + 1}/{total_docs}...",
            )
        await _process_single_document(
            acc,
            doc_id,
            content,
            file_path,
            collection_id_str,
            per_doc_timeout,
        )

    # Stash batch metadata for callers
    acc._docs_deferred = docs_deferred  # type: ignore[attr-defined]
    acc._total_candidate = total_candidate  # type: ignore[attr-defined]
    acc._changed_doc_ids = [doc_id for doc_id, _, _ in docs_to_chunk]  # type: ignore[attr-defined]
    return acc


def _try_surface_extract(
    uri: str, content: str | None, extractor: Any, stats: dict[str, int]
) -> None:
    """Attempt to extract surface data from one document."""
    if not content:
        return
    file_path = _uri_to_file_path(uri)
    if _is_ignored_repo_path(file_path):
        return
    stats["surface_files_scanned"] += 1
    try:
        if extractor.add_file(file_path, content):
            stats["surface_files_recognized"] += 1
    except Exception as exc:
        stats["surface_parse_errors"] += 1
        logger.debug("Surface extraction failed for %s: %s", file_path, exc)


async def materialize_surface_catalog_for_source(
    *,
    source_id: str,
    collection_id: str,
) -> dict[str, int]:
    """Deterministically materialize spec-driven surfaces for one source."""
    import uuid as uuid_module

    from contextmine_core.analyzer.extractors.surface import (
        SurfaceCatalogExtractor,
        build_surface_graph,
    )
    from contextmine_core.models import Document

    source_uuid = uuid_module.UUID(source_id)
    collection_uuid = uuid_module.UUID(collection_id)
    stats: dict[str, int] = {
        "surface_files_scanned": 0,
        "surface_files_recognized": 0,
        "surface_parse_errors": 0,
        "endpoint_nodes": 0,
        "endpoint_handler_links": 0,
        "graphql_nodes": 0,
        "proto_nodes": 0,
        "job_nodes": 0,
        "edges_created": 0,
        "evidence_created": 0,
    }

    async with get_session() as session:
        result = await session.execute(
            select(Document.uri, Document.content_markdown).where(Document.source_id == source_uuid)
        )
        docs = result.all()
        extractor = SurfaceCatalogExtractor()

        for uri, content in docs:
            _try_surface_extract(uri, content, extractor, stats)

        catalog = extractor.catalog
        has_surfaces = (
            catalog.openapi_specs
            or catalog.graphql_schemas
            or catalog.protobuf_files
            or catalog.job_definitions
        )
        if has_surfaces:
            graph_stats = await build_surface_graph(session, collection_uuid, catalog)
            for key, value in graph_stats.items():
                if isinstance(value, int):
                    stats[key] = value
        await session.commit()

    return stats


@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=1,
    tags=[TAG_DB_HEAVY],
)
async def maintain_chunks_for_document(
    document_id: str,
    content: str,
    file_path: str | None = None,
) -> dict:
    """Maintain chunks for a document incrementally.

    - Computes new chunks from content
    - Upserts chunks by hash (only new hashes are inserted)
    - Hard deletes chunks that no longer exist

    Returns stats dict with chunks_created, chunks_kept, chunks_deleted.
    """
    import uuid as uuid_module

    stats = {"chunks_created": 0, "chunks_kept": 0, "chunks_deleted": 0}

    # Generate new chunks
    new_chunks = chunk_document(content, file_path)
    new_hashes = {c.chunk_hash for c in new_chunks}

    async with get_session() as session:
        doc_uuid = uuid_module.UUID(document_id)

        # Get existing chunk hashes for this document
        result = await session.execute(
            select(Chunk.id, Chunk.chunk_hash).where(Chunk.document_id == doc_uuid)
        )
        existing = {row.chunk_hash: row.id for row in result.all()}
        existing_hashes = set(existing.keys())

        # Determine what to add and delete
        hashes_to_add = new_hashes - existing_hashes
        hashes_to_delete = existing_hashes - new_hashes
        hashes_to_keep = existing_hashes & new_hashes

        stats["chunks_kept"] = len(hashes_to_keep)

        # Delete removed chunks
        if hashes_to_delete:
            ids_to_delete = [existing[h] for h in hashes_to_delete]
            await session.execute(delete(Chunk).where(Chunk.id.in_(ids_to_delete)))
            stats["chunks_deleted"] = len(ids_to_delete)

        # Add new chunks (track added hashes to avoid duplicates)
        added_hashes: set[str] = set()
        for chunk_result in new_chunks:
            if (
                chunk_result.chunk_hash in hashes_to_add
                and chunk_result.chunk_hash not in added_hashes
            ):
                new_chunk = Chunk(
                    document_id=doc_uuid,
                    chunk_index=chunk_result.chunk_index,
                    chunk_hash=chunk_result.chunk_hash,
                    content=chunk_result.content,
                    meta=chunk_result.meta,
                )
                session.add(new_chunk)
                added_hashes.add(chunk_result.chunk_hash)
                stats["chunks_created"] += 1

        await session.commit()

    return stats


@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=1,
    cache_policy=INPUTS,
    cache_expiration=timedelta(hours=24),
    tags=[TAG_DB_HEAVY],
)
async def get_or_create_embedding_model(
    provider: EmbeddingProvider,
    model_name: str,
    dimension: int,
) -> EmbeddingModel:
    """Get or create an EmbeddingModel record.

    Returns the EmbeddingModel for the given provider/model, creating it if needed.

    Cached for 24 hours since embedding models rarely change.
    """
    async with get_session() as session:
        # Try to find existing model
        result = await session.execute(
            select(EmbeddingModel).where(
                EmbeddingModel.provider == provider,
                EmbeddingModel.model_name == model_name,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            return existing

        # Create new model record
        new_model = EmbeddingModel(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
        )
        session.add(new_model)
        await session.commit()
        await session.refresh(new_model)
        return new_model


async def _embed_batch_with_fallback(
    embedder: Any,
    texts: list[str],
    dimension: int,
    document_id: str,
) -> Any:
    """Embed a batch of texts, falling back to FakeEmbedder on failure."""
    try:
        return await asyncio.wait_for(
            embedder.embed_batch(texts),
            timeout=_embedding_batch_timeout_seconds(),
        )
    except TimeoutError:
        logger.warning(
            "Embedding batch timed out after %ss for document %s; using deterministic fallback.",
            _embedding_batch_timeout_seconds(),
            document_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Embedding batch failed for document %s: %s; using deterministic fallback.",
            document_id,
            exc,
        )
    return await FakeEmbedder(dimension=dimension).embed_batch(texts)


@task(
    retries=EMBEDDING_API_RETRIES,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5,
    tags=[TAG_EMBEDDING_API],
)
async def embed_chunks_for_document(
    document_id: str,
    embedding_model: EmbeddingModel,
    batch_size: int = 100,
) -> dict:
    """Embed chunks for a document that don't have embeddings yet.

    Uses deduplication: if another chunk with the same hash already has an
    embedding for this model, copies that embedding instead of calling the API.

    Tagged with embedding-api for concurrency limiting to prevent rate limits.
    Retries on transient API failures.

    Args:
        document_id: The document UUID
        embedding_model: The EmbeddingModel to use
        batch_size: Number of chunks to embed per batch

    Returns:
        Stats dict with chunks_embedded, chunks_deduplicated, and tokens_used
    """
    import uuid as uuid_module

    stats = {"chunks_embedded": 0, "chunks_deduplicated": 0, "tokens_used": 0}

    doc_uuid = uuid_module.UUID(document_id)
    model_uuid = embedding_model.id

    # Find chunks without embeddings for this model (include chunk_hash for dedup)
    async with get_session() as session:
        result = await session.execute(
            select(Chunk.id, Chunk.chunk_hash, Chunk.content).where(
                Chunk.document_id == doc_uuid,
                (Chunk.embedding_model_id != model_uuid) | (Chunk.embedding_model_id.is_(None)),
            )
        )
        chunks_to_process = result.all()

    if not chunks_to_process:
        return stats

    # Try to deduplicate: find existing embeddings for chunks with same hash
    chunk_hashes = list({row.chunk_hash for row in chunks_to_process})

    async with get_session() as session:
        # Find chunks with same hashes that already have embeddings for this model
        result = await session.execute(
            text("""
                SELECT DISTINCT ON (chunk_hash) chunk_hash, embedding::text
                FROM chunks
                WHERE chunk_hash = ANY(:hashes)
                  AND embedding_model_id = :model_id
                  AND embedding IS NOT NULL
            """),
            {"hashes": chunk_hashes, "model_id": model_uuid},
        )
        existing_embeddings = {row[0]: row[1] for row in result.all()}

    # Separate chunks into those we can deduplicate vs those needing API call
    chunks_to_copy = []
    chunks_to_embed = []

    for chunk in chunks_to_process:
        if chunk.chunk_hash in existing_embeddings:
            chunks_to_copy.append((chunk.id, existing_embeddings[chunk.chunk_hash]))
        else:
            chunks_to_embed.append(chunk)

    # Copy embeddings for deduplicated chunks
    if chunks_to_copy:
        async with get_session() as session:
            for chunk_id, embedding_str in chunks_to_copy:
                await session.execute(
                    text("""
                        UPDATE chunks
                        SET embedding = CAST(:embedding AS vector),
                            embedding_model_id = :model_id,
                            embedded_at = :embedded_at
                        WHERE id = :chunk_id
                    """),
                    {
                        "embedding": embedding_str,
                        "model_id": model_uuid,
                        "embedded_at": datetime.now(UTC),
                        "chunk_id": chunk_id,
                    },
                )
                stats["chunks_deduplicated"] += 1

            await session.commit()

    # Embed remaining chunks via API
    if not chunks_to_embed:
        return stats

    # Get embedder for this model
    embedder = get_embedder(
        provider=embedding_model.provider,
        model_name=embedding_model.model_name,
    )

    # Process in batches
    for i in range(0, len(chunks_to_embed), batch_size):
        batch = chunks_to_embed[i : i + batch_size]
        chunk_ids = [row.id for row in batch]
        texts = [row.content for row in batch]

        result = await _embed_batch_with_fallback(
            embedder, texts, int(embedding_model.dimension), document_id
        )
        stats["tokens_used"] += result.tokens_used

        # Update chunks with embeddings via raw SQL (for pgvector)
        async with get_session() as session:
            for chunk_id, embedding in zip(chunk_ids, result.embeddings, strict=True):
                # Convert embedding to pgvector format
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                await session.execute(
                    text("""
                        UPDATE chunks
                        SET embedding = CAST(:embedding AS vector),
                            embedding_model_id = :model_id,
                            embedded_at = :embedded_at
                        WHERE id = :chunk_id
                    """),
                    {
                        "embedding": embedding_str,
                        "model_id": model_uuid,
                        "embedded_at": datetime.now(UTC),
                        "chunk_id": chunk_id,
                    },
                )
                stats["chunks_embedded"] += 1

            await session.commit()

    return stats


async def get_embedding_model_for_collection(collection_id: str) -> str:
    """Get the embedding model spec for a collection.

    Checks collection.config['embedding_model'] first, falls back to global default.

    Returns:
        Model spec string like 'openai:text-embedding-3-small'
    """
    import uuid as uuid_module

    async with get_session() as session:
        result = await session.execute(
            select(Collection.config).where(Collection.id == uuid_module.UUID(collection_id))
        )
        config = result.scalar_one_or_none()

        if config and config.get("embedding_model"):
            return config["embedding_model"]

    # Fall back to global default
    settings = get_settings()
    return settings.default_embedding_model


async def embed_document(document_id: str, collection_id: str | None = None) -> dict:
    """Embed all chunks for a document using the collection's or default embedding model.

    This is the main entry point for embedding after chunking. Uses deduplication
    to copy embeddings from chunks with identical hashes instead of calling the API.

    Args:
        document_id: The document UUID
        collection_id: Optional collection UUID for per-collection embedding config

    Returns:
        Stats dict with chunks_embedded, chunks_deduplicated, and tokens_used
    """
    # Get embedding model spec from collection config or global default
    if collection_id:
        model_spec = await get_embedding_model_for_collection(collection_id)
    else:
        settings = get_settings()
        model_spec = settings.default_embedding_model

    provider, model_name = parse_embedding_model_spec(model_spec)

    # Get embedder to determine dimension
    embedder = get_embedder(provider, model_name)
    dimension = embedder.dimension

    # Get or create the embedding model record
    embedding_model = await get_or_create_embedding_model(
        provider=provider,
        model_name=model_name,
        dimension=dimension,
    )

    # Embed chunks
    return await embed_chunks_for_document(document_id, embedding_model)


_SCHEMA_EXTENSIONS = (".sql", ".ddl", ".prisma", ".php", ".py", ".ts", ".js", ".java", ".rb")
_SCHEMA_TOKENS = (
    "schema",
    "migration",
    "migrate",
    "database",
    "db",
    "sql",
    "doctrine",
    "entity",
    "model",
    "prisma",
)


def _is_schema_candidate(file_path: str) -> bool:
    """Check if a file path is a potential schema file."""
    normalized = file_path.lower()
    return normalized.endswith(_SCHEMA_EXTENSIONS) and any(
        token in normalized for token in _SCHEMA_TOKENS
    )


def _try_alembic_extraction(
    file_path: str, content: str, erm_extractor: Any, extract_fn: Any
) -> None:
    """Try to extract ERM from an Alembic migration file."""
    if "alembic/versions" not in file_path or not file_path.endswith(".py"):
        return
    try:
        extraction = extract_fn(file_path, content)
        erm_extractor.add_alembic_extraction(extraction)
    except Exception as e:
        logger.debug("ERM extraction failed for %s: %s", file_path, e)


async def _kg_extract_erm(
    source_uuid: object,
    collection_uuid: object,
    research_llm: object,
) -> int:
    """Extract ERM tables from Alembic migrations or generic schema files. Returns tables found."""
    from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
    from contextmine_core.analyzer.extractors.erm import (
        ERMExtractor,
        build_erm_graph,
        save_erd_artifact,
    )
    from contextmine_core.analyzer.extractors.schema import (
        build_schema_graph,
    )
    from contextmine_core.analyzer.extractors.schema import (
        save_erd_artifact as save_generic_erd_artifact,
    )
    from contextmine_core.models import Document

    erm_extractor = ERMExtractor()
    schema_candidates: list[tuple[str, str]] = []

    async with get_session() as session:
        result = await session.execute(
            select(Document.uri, Document.content_markdown).where(Document.source_id == source_uuid)
        )
        docs = result.all()

        for uri, content in docs:
            if not content:
                continue
            file_path = _uri_to_file_path(uri)
            if _is_ignored_repo_path(file_path):
                continue
            if _is_schema_candidate(file_path):
                schema_candidates.append((file_path, content))
            _try_alembic_extraction(file_path, content, erm_extractor, extract_from_alembic)

        if erm_extractor.schema.tables:
            erm_stats = await build_erm_graph(session, collection_uuid, erm_extractor.schema)
            await save_erd_artifact(session, collection_uuid, erm_extractor.schema)
            await session.commit()
            tables = erm_stats.get("table_nodes_created", 0)
            logger.info("Extracted %d database tables", tables)
            return tables

        if not schema_candidates:
            return 0

        aggregated = await _extract_schema_fallback(schema_candidates, research_llm)
        if aggregated and aggregated.tables:
            schema_stats = await build_schema_graph(session, collection_uuid, aggregated)
            await save_generic_erd_artifact(session, collection_uuid, aggregated)
            await session.commit()
            tables = schema_stats.get("table_nodes_created", 0)
            logger.info("Extracted %d database tables via generic schema extraction", tables)
            return tables
    return 0


async def _extract_schema_fallback(
    schema_candidates: list[tuple[str, str]],
    research_llm: object,
) -> object | None:
    """Try deterministic SQL extraction, then LLM-based fallback."""
    from contextmine_core.analyzer.extractors.schema import (
        aggregate_schema_extractions,
        extract_schema_from_file,
        extract_schema_from_files,
    )

    deterministic_extractions = []
    for candidate_path, candidate_content in schema_candidates[:250]:
        if not candidate_path.lower().endswith((".sql", ".ddl")):
            continue
        try:
            extraction = await extract_schema_from_file(
                candidate_path, candidate_content, research_llm
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Deterministic SQL schema extraction failed for %s: %s", candidate_path, e)
            continue
        if extraction.tables or extraction.foreign_keys:
            deterministic_extractions.append(extraction)

    if deterministic_extractions:
        return aggregate_schema_extractions(deterministic_extractions)

    try:
        extractions = await extract_schema_from_files(
            files=schema_candidates[:250], provider=research_llm
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Generic schema extraction failed: %s", e)
        extractions = []
    return aggregate_schema_extractions(extractions)


async def _kg_extract_business_rules(
    source_uuid: object,
    collection_uuid: object,
    changed_doc_ids: list[str] | None,
    research_llm: object,
) -> int:
    """Extract business rules from code files using LLM. Returns rules created count."""
    import uuid as uuid_module

    from contextmine_core.analyzer.extractors.rules import (
        build_rules_graph,
        extract_rules_from_file,
    )
    from contextmine_core.models import Document
    from contextmine_core.treesitter.languages import detect_language

    if changed_doc_ids is not None and len(changed_doc_ids) == 0:
        logger.info("No changed documents - skipping business rule extraction")
        return 0

    all_extractions = []
    async with get_session() as session:
        if changed_doc_ids:
            result = await session.execute(
                select(Document.id, Document.uri, Document.content_markdown).where(
                    Document.id.in_([uuid_module.UUID(d) for d in changed_doc_ids])
                )
            )
        else:
            result = await session.execute(
                select(Document.id, Document.uri, Document.content_markdown).where(
                    Document.source_id == source_uuid
                )
            )
        docs = result.all()
        logger.info("Extracting business rules from %d documents", len(docs))

        for _doc_id, uri, content in docs:
            if not content:
                continue
            file_path = _uri_to_file_path(uri)
            if _is_ignored_repo_path(file_path) or detect_language(file_path) is None:
                continue
            try:
                rule_result = await extract_rules_from_file(file_path, content, research_llm)
                if rule_result.rules:
                    all_extractions.append(rule_result)
            except Exception as e:
                logger.debug("Rule extraction failed for %s: %s", file_path, e)

        if all_extractions:
            rule_stats = await build_rules_graph(session, collection_uuid, all_extractions)
            await session.commit()
            rules_created = rule_stats.get("rules_created", 0)
            logger.info("Extracted %d business rules", rules_created)
            return rules_created
    return 0


async def _kg_extract_surfaces(source_uuid: object, collection_uuid: object) -> dict[str, int]:
    """Extract surface catalog (OpenAPI, GraphQL, Protobuf, Jobs) into the KG."""
    from contextmine_core.analyzer.extractors.surface import (
        SurfaceCatalogExtractor,
        build_surface_graph,
    )
    from contextmine_core.models import Document

    result_stats: dict[str, int] = {"kg_endpoints": 0, "kg_jobs": 0}
    surface_extractor = SurfaceCatalogExtractor()
    async with get_session() as session:
        result = await session.execute(
            select(Document.uri, Document.content_markdown).where(Document.source_id == source_uuid)
        )
        for uri, content in result.all():
            _try_surface_extract(
                uri,
                content,
                surface_extractor,
                {
                    "surface_files_scanned": 0,
                    "surface_files_recognized": 0,
                    "surface_parse_errors": 0,
                },
            )
        catalog = surface_extractor.catalog
        has_surfaces = (
            catalog.openapi_specs
            or catalog.graphql_schemas
            or catalog.protobuf_files
            or catalog.job_definitions
        )
        if has_surfaces:
            surface_stats = await build_surface_graph(session, collection_uuid, catalog)
            result_stats["kg_endpoints"] = surface_stats.get("endpoint_nodes", 0)
            result_stats["kg_jobs"] = surface_stats.get("job_nodes", 0)
            await session.commit()
            logger.info(
                "Extracted %d endpoints, %d jobs",
                result_stats["kg_endpoints"],
                result_stats["kg_jobs"],
            )
    return result_stats


@traced_task()
@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=1,
    tags=[TAG_DB_HEAVY],
)
async def build_knowledge_graph(
    source_id: str,
    collection_id: str,
    changed_doc_ids: list[str] | None = None,
) -> dict:
    """Build Knowledge Graph from indexed documents.

    Extracts:
    - FILE and SYMBOL nodes from documents/symbols
    - RULE_CANDIDATE nodes from code validation patterns
    - DB_TABLE/DB_COLUMN nodes from Alembic migrations
    - API_ENDPOINT/JOB/etc. nodes from spec files
    - BUSINESS_RULE nodes via LLM extraction
    - Semantic entities via LLM extraction
    - Hierarchical Leiden communities
    - Community summaries and embeddings

    REQUIRES: LLM provider and embedder must be configured.
    GraphRAG features require both for proper entity resolution and community summaries.

    Returns:
        Stats dict with KG extraction metrics

    Raises:
        ValueError: If LLM provider or embedder is not configured
    """
    import uuid as uuid_module

    from contextmine_core.research.llm import get_llm_provider, get_research_llm_provider

    stats = {
        "kg_file_nodes": 0,
        "kg_symbol_nodes": 0,
        "kg_business_rules": 0,
        "kg_tables": 0,
        "kg_endpoints": 0,
        "kg_jobs": 0,
        "kg_errors": [],
    }

    source_uuid = uuid_module.UUID(source_id)
    collection_uuid = uuid_module.UUID(collection_id)

    # REQUIRED: Get LLM provider and embedder upfront
    # GraphRAG is the core feature - no point running without these
    settings = get_settings()

    # Get LLM provider (required for entity extraction and community summaries)
    llm_provider = None
    if settings.default_llm_provider:
        try:
            llm_provider = get_llm_provider(settings.default_llm_provider)
        except Exception as e:
            raise ValueError(f"LLM provider configured but failed to initialize: {e}") from e
    else:
        raise ValueError(
            "LLM provider required for Knowledge Graph. "
            "Set DEFAULT_LLM_PROVIDER (e.g., 'openai', 'anthropic', 'gemini')."
        )

    research_llm = get_research_llm_provider()
    if not research_llm:
        raise ValueError(
            "Research LLM provider required for business rule extraction. "
            "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY."
        )

    # Get embedder (required for entity resolution and community retrieval)
    try:
        provider_name, model_name = parse_embedding_model_spec(settings.default_embedding_model)
        embedder = get_embedder(provider_name, model_name)
    except Exception as e:
        raise ValueError(
            f"Embedder required for Knowledge Graph but failed to initialize: {e}. "
            f"Set DEFAULT_EMBEDDING_MODEL (e.g., 'openai:text-embedding-3-small')."
        ) from e

    logger.info(
        "Knowledge Graph build starting with LLM=%s, embedder=%s",
        settings.default_llm_provider,
        settings.default_embedding_model,
    )

    # Step 1: Build FILE and SYMBOL nodes from indexed documents
    try:
        from contextmine_core.knowledge.builder import build_knowledge_graph_for_source

        async with get_session() as session:
            kg_build_stats = await build_knowledge_graph_for_source(session, source_uuid)
            stats["kg_file_nodes"] = kg_build_stats.file_nodes_created
            stats["kg_symbol_nodes"] = kg_build_stats.symbol_nodes_created
            await session.commit()
            logger.info(
                "Built KG: %d file nodes, %d symbol nodes",
                stats["kg_file_nodes"],
                stats["kg_symbol_nodes"],
            )
    except Exception as e:
        logger.warning("Failed to build FILE/SYMBOL nodes: %s", e)
        stats["kg_errors"].append(f"file_symbol: {e}")

    # Step 2: Extract business rules from code files using LLM (INCREMENTAL)
    try:
        rules_count = await _kg_extract_business_rules(
            source_uuid,
            collection_uuid,
            changed_doc_ids,
            research_llm,
        )
        stats["kg_business_rules"] = rules_count
    except Exception as e:
        logger.warning("Failed to extract business rules: %s", e)
        stats["kg_errors"].append(f"rules: {e}")

    # Step 3: Extract ERM from Alembic migrations
    try:
        tables_found = await _kg_extract_erm(
            source_uuid,
            collection_uuid,
            research_llm,
        )
        stats["kg_tables"] = tables_found
    except Exception as e:
        logger.warning("Failed to extract ERM: %s", e)
        stats["kg_errors"].append(f"erm: {e}")

    # Step 4: Extract Surface Catalog (OpenAPI, GraphQL, Protobuf, Jobs)
    try:
        surface_result = await _kg_extract_surfaces(source_uuid, collection_uuid)
        stats["kg_endpoints"] = surface_result.get("kg_endpoints", 0)
        stats["kg_jobs"] = surface_result.get("kg_jobs", 0)
    except Exception as e:
        logger.warning("Failed to extract surfaces: %s", e)
        stats["kg_errors"].append(f"surface: {e}")

    # Step 5: Extract semantic entities using LLM (for proper GraphRAG)
    try:
        from contextmine_core.knowledge.extraction import (
            extract_from_documents,
            persist_semantic_entities,
        )

        if changed_doc_ids is not None and len(changed_doc_ids) == 0:
            logger.info("No changed documents - skipping semantic entity extraction")
        else:
            async with get_session() as session:
                # Extract semantic entities from documents
                # Uses embedding similarity for cross-language entity resolution
                extraction_batch = await extract_from_documents(
                    collection_id=collection_uuid,
                    llm_provider=llm_provider,
                    embedder=embedder,
                    max_chunks=50,  # Limit for cost control
                )

                # Persist to knowledge graph
                extraction_stats = await persist_semantic_entities(
                    session=session,
                    collection_id=collection_uuid,
                    batch=extraction_batch,
                )
                await session.commit()

                stats["kg_semantic_entities"] = extraction_stats.get("entities_created", 0)
                stats["kg_semantic_relationships"] = extraction_stats.get(
                    "relationships_created", 0
                )
                logger.info(
                    "Extracted %d semantic entities, %d relationships",
                    stats["kg_semantic_entities"],
                    stats["kg_semantic_relationships"],
                )

    except Exception as e:
        logger.warning("Failed to extract semantic entities: %s", e)
        stats["kg_errors"].append(f"semantic_extraction: {e}")

    # Step 6: Detect communities using Leiden algorithm (GraphRAG)
    try:
        from contextmine_core.knowledge.communities import detect_communities, persist_communities

        async with get_session() as session:
            # Leiden with resolution parameters: [1.0, 0.5, 0.1] for levels 0, 1, 2
            community_result = await detect_communities(session, collection_uuid)
            await persist_communities(session, collection_uuid, community_result)
            await session.commit()

            # Report communities at each hierarchical level
            stats["kg_communities_l0"] = community_result.community_count(level=0)
            stats["kg_communities_l1"] = community_result.community_count(level=1)
            stats["kg_communities_l2"] = community_result.community_count(level=2)
            logger.info(
                "Leiden communities: L0=%d, L1=%d, L2=%d (modularity: %.3f, %.3f, %.3f)",
                stats["kg_communities_l0"],
                stats["kg_communities_l1"],
                stats["kg_communities_l2"],
                community_result.modularity.get(0, 0),
                community_result.modularity.get(1, 0),
                community_result.modularity.get(2, 0),
            )

    except Exception as e:
        logger.warning("Failed to detect communities: %s", e)
        stats["kg_errors"].append(f"communities: {e}")

    # Step 7: Generate community summaries and embeddings
    try:
        from contextmine_core.knowledge.summaries import generate_community_summaries

        if changed_doc_ids is not None and len(changed_doc_ids) == 0:
            logger.info("No changed documents - skipping community summary regeneration")
        else:
            async with get_session() as session:
                summary_stats = await generate_community_summaries(
                    session,
                    collection_uuid,
                    provider=research_llm,
                    embed_provider=embedder,
                )
                await session.commit()

                stats["kg_summaries_created"] = summary_stats.communities_summarized
                stats["kg_embeddings_created"] = summary_stats.embeddings_created
                logger.info(
                    "Generated %d community summaries, %d embeddings",
                    stats["kg_summaries_created"],
                    stats["kg_embeddings_created"],
                )

    except Exception as e:
        logger.warning("Failed to generate community summaries: %s", e)
        stats["kg_errors"].append(f"summaries: {e}")

    return stats


async def _generate_arch_docs_on_sync(
    session: Any,
    settings: Any,
    collection_uuid: Any,
    as_is: Any,
) -> dict[str, Any]:
    """Generate arc42 architecture docs during sync. Returns stats dict."""
    from dataclasses import asdict

    from contextmine_core.architecture import (
        build_architecture_facts,
        compute_arc42_drift,
        generate_arc42_from_facts,
    )
    from contextmine_core.models import KnowledgeArtifact, KnowledgeArtifactKind, TwinScenario

    stats: dict[str, Any] = {"arch_facts_count": 0, "arch_ports_adapters_count": 0}
    llm_provider = None
    if settings.arch_docs_llm_enrich and settings.default_llm_provider:
        try:
            from contextmine_core.research.llm import get_llm_provider

            llm_provider = get_llm_provider(settings.default_llm_provider)
        except Exception:
            pass

    facts_bundle = await build_architecture_facts(
        session,
        collection_id=collection_uuid,
        scenario_id=as_is.id,
        enable_llm_enrich=settings.arch_docs_llm_enrich,
        llm_provider=llm_provider,
    )
    arc42_doc = generate_arc42_from_facts(facts_bundle, as_is, options={})
    artifact_name = f"{as_is.id}.arc42.md"

    artifact = (
        await session.execute(
            select(KnowledgeArtifact).where(
                KnowledgeArtifact.collection_id == collection_uuid,
                KnowledgeArtifact.kind == KnowledgeArtifactKind.ARC42,
                KnowledgeArtifact.name == artifact_name,
            )
        )
    ).scalar_one_or_none()

    drift_meta: dict[str, object] | None = None
    if settings.arch_docs_drift_enabled:
        baseline = (
            await session.execute(
                select(TwinScenario)
                .where(TwinScenario.collection_id == collection_uuid, TwinScenario.id != as_is.id)
                .order_by(TwinScenario.version.desc(), TwinScenario.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if baseline:
            baseline_bundle = await build_architecture_facts(
                session,
                collection_id=collection_uuid,
                scenario_id=baseline.id,
                enable_llm_enrich=settings.arch_docs_llm_enrich,
                llm_provider=llm_provider,
            )
            drift_report = compute_arc42_drift(
                facts_bundle, baseline_bundle, baseline_scenario_id=baseline.id
            )
            drift_meta = {
                "generated_at": drift_report.generated_at.isoformat(),
                "baseline_scenario_id": str(baseline.id),
                "current_hash": drift_report.current_hash,
                "baseline_hash": drift_report.baseline_hash,
                "deltas": [asdict(delta) for delta in drift_report.deltas],
                "warnings": drift_report.warnings,
            }
            stats["arch_drift_deltas"] = len(drift_report.deltas)

    artifact_meta = {
        "scenario_id": str(as_is.id),
        "generated_at": arc42_doc.generated_at.isoformat(),
        "facts_hash": facts_bundle.facts_hash(),
        "confidence_summary": arc42_doc.confidence_summary,
        "section_coverage": arc42_doc.section_coverage,
        "warnings": arc42_doc.warnings,
        "sections": arc42_doc.sections,
    }
    if drift_meta is not None:
        artifact_meta["drift"] = drift_meta

    if artifact:
        artifact.content = arc42_doc.markdown
        artifact.meta = artifact_meta
    else:
        session.add(
            KnowledgeArtifact(
                collection_id=collection_uuid,
                kind=KnowledgeArtifactKind.ARC42,
                name=artifact_name,
                content=arc42_doc.markdown,
                meta=artifact_meta,
            )
        )

    stats["arch_facts_count"] = len(facts_bundle.facts)
    stats["arch_ports_adapters_count"] = len(facts_bundle.ports_adapters)
    return stats


@traced_task()
@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=1,
    tags=[TAG_DB_HEAVY],
)
async def build_twin_graph(
    source_id: str,
    collection_id: str,
    snapshot_dicts: list[dict] | None = None,
    changed_doc_ids: list[str] | None = None,
    file_metrics: list[dict] | None = None,
    evolution_payload: dict[str, object] | None = None,
) -> dict:
    """Build digital twin graph from semantic snapshots (SCIP/LSIF) and existing KG."""
    del changed_doc_ids
    import uuid as uuid_module

    from contextmine_core.graph.age import sync_scenario_to_age
    from contextmine_core.models import (
        TwinNode,
    )
    from contextmine_core.pathing import canonicalize_repo_relative_path
    from contextmine_core.semantic_snapshot.models import Snapshot
    from contextmine_core.twin import (
        apply_file_metrics_to_scenario,
        evaluate_and_store_fitness_findings,
        get_or_create_as_is_scenario,
        ingest_snapshot_into_as_is,
        refresh_metric_snapshots,
        replace_evolution_snapshots,
        seed_scenario_from_knowledge_graph,
    )
    from contextmine_core.validation import refresh_validation_snapshots

    collection_uuid = uuid_module.UUID(collection_id)
    source_uuid = uuid_module.UUID(source_id)
    stats: dict[str, Any] = {
        "twin_nodes_upserted": 0,
        "twin_edges_upserted": 0,
        "twin_metric_nodes_enriched": 0,
        "metrics_requested_files": 0,
        "metrics_mapped_files": 0,
        "twin_metrics_snapshots": 0,
        "twin_validation_snapshots": 0,
        "arch_facts_count": 0,
        "arch_ports_adapters_count": 0,
        "arch_drift_deltas": 0,
    }

    async with get_session() as session:
        as_is = await get_or_create_as_is_scenario(session, collection_uuid, user_id=None)
        stats["twin_asis_scenario_id"] = str(as_is.id)

        if snapshot_dicts:
            for snapshot_dict in snapshot_dicts:
                snapshot = Snapshot.from_dict(snapshot_dict)
                _, ingest_stats = await ingest_snapshot_into_as_is(
                    session,
                    collection_uuid,
                    snapshot,
                    source_id=source_uuid,
                    user_id=None,
                )
                stats["twin_nodes_upserted"] += int(ingest_stats.get("nodes_upserted", 0))
                stats["twin_edges_upserted"] += int(ingest_stats.get("edges_upserted", 0))
            # Always enrich SCIP snapshots with the full document-based knowledge graph.
            # This closes language/indexer gaps (for example script-style PHP entrypoints)
            # without requiring sync failure gates.
            kg_nodes, kg_edges = await seed_scenario_from_knowledge_graph(
                session,
                as_is.id,
                collection_uuid,
                clear_existing=False,
            )
            stats["twin_nodes_upserted"] += int(kg_nodes)
            stats["twin_edges_upserted"] += int(kg_edges)
            stats["twin_kg_supplement_nodes"] = int(kg_nodes)
            stats["twin_kg_supplement_edges"] = int(kg_edges)
        else:
            nodes, edges = await seed_scenario_from_knowledge_graph(
                session,
                as_is.id,
                collection_uuid,
                clear_existing=False,
            )
            stats["twin_nodes_upserted"] += int(nodes)
            stats["twin_edges_upserted"] += int(edges)

        settings = get_settings()
        if file_metrics:
            requested_metric_paths = {
                canonicalize_repo_relative_path(str(metric.get("file_path", "")).strip())
                for metric in file_metrics
                if canonicalize_repo_relative_path(str(metric.get("file_path", "")).strip())
            }
            requested_metric_files = len(requested_metric_paths)
            available_nodes = (
                (
                    await session.execute(
                        select(TwinNode.natural_key).where(
                            TwinNode.scenario_id == as_is.id,
                            TwinNode.kind == "file",
                        )
                    )
                )
                .scalars()
                .all()
            )
            available_paths = {
                canonicalize_repo_relative_path(str(node_key).removeprefix(_FILE_NODE_PREFIX))
                for node_key in available_nodes
                if str(node_key).startswith(_FILE_NODE_PREFIX)
            }
            metrics_unmapped = sorted(
                path for path in requested_metric_paths if path not in available_paths
            )
            enriched = await apply_file_metrics_to_scenario(session, as_is.id, file_metrics)
            stats["metrics_requested_files"] = requested_metric_files
            stats["metrics_mapped_files"] = enriched
            stats["metrics_unmapped_sample"] = metrics_unmapped[:25]
            if enriched < requested_metric_files:
                stats["metrics_gate"] = "fail"
                gate_error = (
                    "METRICS_GATE_FAILED: twin_node_mapping_incomplete "
                    f"(mapped={enriched}, metrics={requested_metric_files})"
                )
                if settings.metrics_strict_mode:
                    raise RuntimeError(gate_error)
                logger.warning("%s", gate_error)
            else:
                stats["metrics_gate"] = "pass"
            stats["twin_metric_nodes_enriched"] = enriched
            stats["twin_metric_nodes_requested"] = requested_metric_files

        stats["twin_metrics_snapshots"] = await refresh_metric_snapshots(session, as_is.id)
        if evolution_payload:
            ownership_rows = list(evolution_payload.get("ownership_rows") or [])
            coupling_rows = list(evolution_payload.get("coupling_rows") or [])
            persist_stats = await replace_evolution_snapshots(
                session,
                scenario_id=as_is.id,
                ownership_rows=ownership_rows,
                coupling_rows=coupling_rows,
            )
            stats["evolution_ownership_rows"] = int(persist_stats.get("ownership_rows", 0))
            stats["evolution_coupling_rows"] = int(persist_stats.get("coupling_rows", 0))

            window_days = int(
                evolution_payload.get("window_days") or get_settings().twin_evolution_window_days
            )
            fitness_stats = await evaluate_and_store_fitness_findings(
                session,
                collection_id=collection_uuid,
                scenario_id=as_is.id,
                window_days=window_days,
            )
            stats["fitness_findings_written"] = int(fitness_stats.get("created", 0))
            stats["fitness_findings_by_type"] = fitness_stats.get("by_type", {})
            stats["fitness_findings_warnings"] = fitness_stats.get("warnings", [])
        stats["twin_validation_snapshots"] = await refresh_validation_snapshots(
            session, collection_uuid
        )

        if settings.arch_docs_enabled and settings.arch_docs_generate_on_sync:
            try:
                arch_result = await _generate_arch_docs_on_sync(
                    session, settings, collection_uuid, as_is
                )
                stats.update(arch_result)
            except Exception as exc:
                logger.warning("Architecture docs generation failed (advisory): %s", exc)
                stats["arch_docs_error"] = str(exc)
        elif settings.arch_docs_enabled:
            stats["arch_docs_sync_generation"] = "skipped_requires_explicit_trigger"

        # Keep AGE in sync as a mandatory M1 requirement.
        await sync_scenario_to_age(session, as_is.id)
        await session.commit()

    return stats


async def _materialize_behavioral_layers_impl(
    *,
    source_id: str,
    collection_id: str,
    scenario_id: str | None,
    source_version_id: str | None,
) -> dict[str, object]:
    """Build deep behavioral layers (tests/ui/flows) and append twin status metadata."""
    import uuid as uuid_module

    from contextmine_core.analyzer.extractors.flows import build_flows_graph, synthesize_user_flows
    from contextmine_core.analyzer.extractors.tests import (
        build_tests_graph,
        extract_tests_from_files,
    )
    from contextmine_core.analyzer.extractors.ui import build_ui_graph, extract_ui_from_files
    from contextmine_core.graph.age import sync_scenario_to_age
    from contextmine_core.models import Document, TwinSourceVersion
    from contextmine_core.twin import record_twin_event, seed_scenario_from_knowledge_graph

    settings = get_settings()
    now_iso = datetime.now(UTC).isoformat()

    if not settings.digital_twin_behavioral_enabled:
        return {
            "behavioral_layers_status": "disabled",
            "last_behavioral_materialized_at": None,
            "deep_warnings": ["DIGITAL_TWIN_BEHAVIORAL_ENABLED=false"],
        }

    collection_uuid = uuid_module.UUID(collection_id)
    source_uuid = uuid_module.UUID(source_id)
    scenario_uuid = uuid_module.UUID(scenario_id) if scenario_id else None
    source_version_uuid = uuid_module.UUID(source_version_id) if source_version_id else None

    async with get_session() as session:
        if source_version_uuid:
            await record_twin_event(
                session,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                source_id=source_uuid,
                source_version_id=source_version_uuid,
                event_type="behavioral_extract_started",
                status="materializing",
                payload={"started_at": now_iso},
                idempotency_key=f"behavioral_extract_started:{source_id}:{source_version_id}",
            )
            await session.commit()

        result = await session.execute(
            select(Document.uri, Document.content_markdown).where(Document.source_id == source_uuid)
        )
        docs = result.all()

        files: list[tuple[str, str]] = []
        for uri, content in docs:
            if not content:
                continue
            file_path = _uri_to_file_path(uri)
            files.append((file_path, content))

        deep_warnings: list[str] = []
        test_extractions = (
            extract_tests_from_files(files) if settings.digital_twin_behavioral_enabled else []
        )
        ui_extractions = extract_ui_from_files(files) if settings.digital_twin_ui_enabled else []

        behavioral_stats: dict[str, int] = {}
        if test_extractions:
            behavioral_stats.update(
                await build_tests_graph(
                    session,
                    collection_uuid,
                    test_extractions,
                    source_id=source_uuid,
                )
            )
        if ui_extractions:
            behavioral_stats.update(
                await build_ui_graph(
                    session,
                    collection_uuid,
                    ui_extractions,
                    source_id=source_uuid,
                )
            )
        if settings.digital_twin_flows_enabled:
            synthesis = synthesize_user_flows(ui_extractions, test_extractions)
            behavioral_stats.update(
                await build_flows_graph(
                    session,
                    collection_uuid,
                    synthesis,
                    source_id=source_uuid,
                )
            )

        if not test_extractions:
            deep_warnings.append("No test semantics extracted from current source.")
        if settings.digital_twin_ui_enabled and not ui_extractions:
            deep_warnings.append("No UI semantics extracted from current source.")

        if scenario_uuid:
            await seed_scenario_from_knowledge_graph(
                session,
                scenario_uuid,
                collection_uuid,
                clear_existing=False,
            )
            await sync_scenario_to_age(session, scenario_uuid)

        source_version = None
        if source_version_uuid:
            source_version = (
                await session.execute(
                    select(TwinSourceVersion).where(TwinSourceVersion.id == source_version_uuid)
                )
            ).scalar_one_or_none()
        if source_version:
            merged = dict(source_version.stats or {})
            merged.update(
                {
                    "behavioral_layers_status": "ready",
                    "last_behavioral_materialized_at": now_iso,
                    "deep_warnings": deep_warnings,
                    "behavioral_extract": behavioral_stats,
                }
            )
            source_version.stats = merged

        if source_version_uuid:
            await record_twin_event(
                session,
                collection_id=collection_uuid,
                scenario_id=scenario_uuid,
                source_id=source_uuid,
                source_version_id=source_version_uuid,
                event_type="behavioral_extract_ready",
                status="ready",
                payload={
                    "finished_at": now_iso,
                    "deep_warnings": deep_warnings,
                    "stats": behavioral_stats,
                },
                idempotency_key=f"behavioral_extract_ready:{source_id}:{source_version_id}",
            )
        await session.commit()

        return {
            "behavioral_layers_status": "ready",
            "last_behavioral_materialized_at": now_iso,
            "deep_warnings": deep_warnings,
            "behavioral_extract": behavioral_stats,
        }


@traced_task()
@task(
    retries=0,
    tags=[TAG_DB_HEAVY],
)
async def materialize_behavioral_layers(
    source_id: str,
    collection_id: str,
    scenario_id: str | None = None,
    source_version_id: str | None = None,
) -> dict[str, object]:
    """Task wrapper for behavioral layer extraction."""
    try:
        return await _materialize_behavioral_layers_impl(
            source_id=source_id,
            collection_id=collection_id,
            scenario_id=scenario_id,
            source_version_id=source_version_id,
        )
    except Exception as exc:
        import uuid as uuid_module

        from contextmine_core.models import TwinSourceVersion
        from contextmine_core.twin import record_twin_event

        collection_uuid = uuid_module.UUID(collection_id)
        source_uuid = uuid_module.UUID(source_id)
        scenario_uuid = uuid_module.UUID(scenario_id) if scenario_id else None
        source_version_uuid = uuid_module.UUID(source_version_id) if source_version_id else None

        async with get_session() as session:
            if source_version_uuid:
                source_version = (
                    await session.execute(
                        select(TwinSourceVersion).where(TwinSourceVersion.id == source_version_uuid)
                    )
                ).scalar_one_or_none()
                if source_version:
                    merged = dict(source_version.stats or {})
                    merged.update(
                        {
                            "behavioral_layers_status": "failed",
                            "last_behavioral_materialized_at": None,
                            "deep_warnings": [str(exc)],
                        }
                    )
                    source_version.stats = merged
                await record_twin_event(
                    session,
                    collection_id=collection_uuid,
                    scenario_id=scenario_uuid,
                    source_id=source_uuid,
                    source_version_id=source_version_uuid,
                    event_type="behavioral_extract_failed",
                    status="failed",
                    payload={},
                    idempotency_key=f"behavioral_extract_failed:{source_id}:{source_version_id}",
                    error=str(exc),
                )
                await session.commit()
        raise


def _build_scip_index_config():
    """Build IndexConfig from settings.

    Returns:
        IndexConfig instance configured from settings.
    """
    from contextmine_core.semantic_snapshot.models import (
        IndexConfig,
        InstallDepsMode,
        Language,
    )

    settings = get_settings()

    # Parse enabled languages
    enabled_languages: set[Language] = set()
    for lang_str in settings.scip_languages.split(","):
        lang_str = lang_str.strip().lower()
        try:
            enabled_languages.add(Language(lang_str))
        except ValueError:
            logger.warning("Unknown SCIP language: %s", lang_str)

    # Parse install deps mode
    try:
        install_mode = InstallDepsMode(settings.scip_install_deps_mode)
    except ValueError:
        install_mode = InstallDepsMode.AUTO

    return IndexConfig(
        enabled_languages=enabled_languages,
        timeout_s_by_language={
            Language.PYTHON: settings.scip_timeout_python,
            Language.TYPESCRIPT: settings.scip_timeout_typescript,
            Language.JAVASCRIPT: settings.scip_timeout_typescript,
            Language.JAVA: settings.scip_timeout_java,
            Language.PHP: settings.scip_timeout_php,
        },
        install_deps_mode=install_mode,
        node_memory_mb=settings.scip_node_memory_mb,
        best_effort=settings.scip_best_effort,
    )


@task(
    retries=0,  # Don't retry - indexing is expensive
    tags=[TAG_SCIP_INDEX],
)
async def task_detect_scip_projects(repo_path: Path) -> dict:
    """Detect projects suitable for SCIP indexing in a repository.

    Returns:
        Dict with:
        - projects: list of ProjectTarget dicts
        - diagnostics: census and detection diagnostics
    """
    from contextmine_core.semantic_snapshot.indexers.detection import (
        detect_projects_with_diagnostics,
    )

    projects, diagnostics = detect_projects_with_diagnostics(repo_path)
    return {
        "projects": [p.to_dict() for p in projects],
        "diagnostics": diagnostics.to_dict(),
    }


@task(
    retries=0,  # Don't retry - indexing is expensive
    timeout_seconds=900,  # 15 minute max
    tags=[TAG_SCIP_INDEX],
)
async def task_index_scip_project(project_dict: dict, output_dir: Path) -> dict | None:
    """Run SCIP indexer on a single project.

    Args:
        project_dict: ProjectTarget as dict
        output_dir: Directory for SCIP output files

    Returns:
        IndexArtifact as dict, or None if indexing failed
    """
    from contextmine_core.semantic_snapshot.indexers import BACKENDS
    from contextmine_core.semantic_snapshot.models import ProjectTarget

    target = ProjectTarget.from_dict(project_dict)
    cfg = _build_scip_index_config()
    cfg.output_dir = output_dir

    backend = next((b for b in BACKENDS if b.can_handle(target)), None)
    if backend is None:
        return None
    try:
        artifact = backend.index(target, cfg)
    except Exception as e:
        logger.warning("SCIP backend error for %s: %s", target.root_path, e)
        if not cfg.best_effort:
            raise
        return None
    if artifact.success:
        logger.info(
            "SCIP indexed %s project at %s in %.1fs",
            target.language.value,
            target.root_path,
            artifact.duration_s,
        )
        return artifact.to_dict()
    logger.warning(
        "SCIP indexing failed for %s: %s",
        target.root_path,
        artifact.error_message,
    )
    return None if cfg.best_effort else None


@task(
    retries=0,
    tags=[TAG_SCIP_INDEX],
)
async def task_parse_scip_snapshot(scip_path: str) -> dict | None:
    """Parse a SCIP file into a Snapshot.

    Args:
        scip_path: Path to the .scip file

    Returns:
        Snapshot as dict, or None if parsing failed
    """
    from contextmine_core.semantic_snapshot import build_snapshot

    try:
        snapshot = build_snapshot(scip_path)
        logger.info(
            "Parsed SCIP: %d symbols, %d relations",
            len(snapshot.symbols),
            len(snapshot.relations),
        )
        return snapshot.to_dict()
    except Exception as e:
        logger.warning("Failed to parse SCIP file %s: %s", scip_path, e)
        return None


@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    tags=[TAG_DB_HEAVY],
)
async def get_due_sources() -> list[Source]:
    """Get all sources that are due for syncing."""
    async with get_session() as session:
        now = datetime.now(UTC)
        stmt = select(Source).where(
            Source.enabled == True,  # noqa: E712
            Source.next_run_at <= now,
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def get_github_token_for_source(_source_id: str, collection_id: str) -> str | None:
    """Get the GitHub OAuth token for a source's collection owner."""
    import uuid as uuid_module

    async with get_session() as session:
        # Get collection owner
        result = await session.execute(
            select(Collection.owner_user_id).where(Collection.id == uuid_module.UUID(collection_id))
        )
        owner_id = result.scalar_one_or_none()
        if not owner_id:
            return None

        # Get OAuth token for owner
        result = await session.execute(
            select(OAuthToken)
            .where(
                OAuthToken.user_id == owner_id,
                OAuthToken.provider == "github",
            )
            .order_by(OAuthToken.created_at.desc())
        )
        token_rows = result.scalars().all()
        token_record = token_rows[0] if token_rows else None
        if len(token_rows) > 1:
            logger.warning(
                "Found %d GitHub OAuth tokens for user %s; using most recent token.",
                len(token_rows),
                owner_id,
            )
        if not token_record:
            return None

        # Decrypt and return token
        return decrypt_token(token_record.access_token_encrypted)


async def get_deploy_key_for_source(source_id: str) -> str | None:
    """Get the decrypted deploy key for a source, if configured."""
    import uuid as uuid_module

    async with get_session() as session:
        result = await session.execute(
            select(Source.deploy_key_encrypted).where(Source.id == uuid_module.UUID(source_id))
        )
        encrypted_key = result.scalar_one_or_none()

        if not encrypted_key:
            return None

        # Decrypt and return key
        return decrypt_token(encrypted_key)


@traced_task()
@task(
    retries=GITHUB_API_RETRIES,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5,
    tags=[TAG_GITHUB_API],
    task_run_name="sync-github-{source.url}",
)
async def sync_github_source(
    source: Source,
    sync_run: SyncRun,
    run_started_at: datetime,
) -> SyncStats:
    """Sync a GitHub source, creating/updating/deleting documents.

    Tagged with github-api for concurrency limiting.
    Retries on transient GitHub API failures.
    """
    stats = SyncStats()

    # Get config
    config = source.config or {}
    owner = config.get("owner", "")
    repo = config.get("repo", "")
    branch = config.get("branch", "main")

    if not owner or not repo:
        raise ValueError("GitHub source missing owner/repo in config")

    # Create progress artifact
    progress_id = await create_progress_artifact(  # type: ignore[misc]
        progress=0.0,
        description=f"Starting sync for {owner}/{repo}...",
    )

    # Get deploy key (preferred) or OAuth token (fallback)
    deploy_key = await get_deploy_key_for_source(str(source.id))
    token = None
    if not deploy_key:
        token = await get_github_token_for_source(str(source.id), str(source.collection_id))

    # Prepare repo path
    ensure_repos_dir()
    repo_path = get_repo_path(str(source.id))

    await update_progress_artifact(progress_id, progress=5, description="Fetching repository...")  # type: ignore[misc]

    # Clone or pull (deploy key takes priority over token)
    clone_url = f"https://github.com/{owner}/{repo}.git"
    git_repo = await _run_blocking_with_timeout(
        "git_clone_or_pull",
        _sync_blocking_step_timeout_seconds(),
        clone_or_pull_repo,
        repo_path,
        clone_url,
        branch,
        token=token,
        ssh_private_key=deploy_key,
    )

    # Get current commit SHA
    new_sha = git_repo.head.commit.hexsha
    old_sha = source.cursor
    source_version_id = None

    from contextmine_core.twin import (
        get_or_create_source_version,
        record_twin_event,
        set_source_version_status,
    )

    async with get_session() as session:
        source_version = await get_or_create_source_version(
            session,
            collection_id=source.collection_id,
            source_id=source.id,
            revision_key=new_sha,
            extractor_version="scip-kg-v1",
            status="materializing",
        )
        await set_source_version_status(
            session,
            source_version_id=source_version.id,
            status="materializing",
            stats={
                "sync_run_id": str(sync_run.id),
                "sync_started_at": datetime.now(UTC).isoformat(),
            },
            started=True,
        )
        await record_twin_event(
            session,
            collection_id=source.collection_id,
            scenario_id=None,
            source_id=source.id,
            source_version_id=source_version.id,
            event_type="sync_started",
            status="materializing",
            payload={"sync_run_id": str(sync_run.id), "revision_key": new_sha},
            idempotency_key=f"sync_started:{source.id}:{new_sha}",
        )
        await session.commit()
        source_version_id = source_version.id

    # Joern CPG generation is mandatory for GitHub twin materialization.
    settings = get_settings()
    cpg_path = Path(settings.joern_cpg_root) / str(source.id) / f"{new_sha}.cpg.bin"
    cpg_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    parse_command = [
        settings.joern_parse_binary,
        str(repo_path),
        "--output",
        str(cpg_path),
    ]
    joern_ok = False
    joern_error = ""
    joern_parse_timeout = _joern_parse_timeout_seconds()
    try:
        parse_result = await _run_blocking_with_timeout(
            "joern_parse",
            joern_parse_timeout,
            subprocess.run,
            parse_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=joern_parse_timeout,
        )
        if parse_result.returncode != 0:
            raise RuntimeError(
                "JOERN_PARSE_FAILED: "
                f"binary={settings.joern_parse_binary} "
                f"code={parse_result.returncode} "
                f"stderr={parse_result.stderr.strip()}"
            )
        if not cpg_path.exists():
            raise RuntimeError(f"JOERN_PARSE_FAILED: expected CPG artifact missing at {cpg_path}")
        joern_ok = True
    except Exception as exc:  # noqa: BLE001
        joern_error = str(exc)
        if settings.joern_required_for_sync:
            raise RuntimeError(f"JOERN_PARSE_FAILED: {joern_error}") from exc
        logger.warning(
            "Joern CPG generation failed for %s/%s in advisory mode: %s",
            owner,
            repo,
            joern_error,
        )

    async with get_session() as session:
        from contextmine_core.models import TwinSourceVersion

        twin_source_version = (
            await session.execute(
                select(TwinSourceVersion).where(TwinSourceVersion.id == source_version_id)
            )
        ).scalar_one_or_none()
        if twin_source_version:
            twin_source_version.joern_status = "ready" if joern_ok else "failed"
            twin_source_version.joern_project = f"{owner}/{repo}"
            twin_source_version.joern_cpg_path = str(cpg_path) if joern_ok else None
            twin_source_version.joern_server_url = settings.joern_server_url
        await record_twin_event(
            session,
            collection_id=source.collection_id,
            scenario_id=None,
            source_id=source.id,
            source_version_id=source_version_id,
            event_type="joern_cpg_generated" if joern_ok else "joern_cpg_failed",
            status="ready" if joern_ok else "failed",
            payload={"cpg_path": str(cpg_path)} if joern_ok else {"error": joern_error},
            idempotency_key=(
                f"joern_cpg:{source.id}:{new_sha}"
                if joern_ok
                else f"joern_cpg_failed:{source.id}:{new_sha}"
            ),
            error=None if joern_ok else joern_error,
        )
        await session.commit()

    # SCIP Polyglot Indexing
    scip_stats: dict[str, Any] = {
        "scip_projects_detected": 0,
        "scip_projects_indexed": 0,
        "scip_snapshots_parsed": 0,
        "scip_projects_failed": 0,
        "scip_symbols": 0,
        "scip_relations": 0,
        "scip_degraded": False,
        "scip_failed_projects": [],
        "scip_languages_detected": [],
        "scip_projects_by_language": {},
        "scip_detected_files_by_language": {},
        "scip_detected_code_by_language": {},
        "scip_indexed_files_by_language": {},
        "scip_missing_languages": [],
        "scip_coverage_complete": True,
        "scip_relation_counts_by_language": {},
        "scip_relation_kinds_by_language": {},
        "scip_missing_relation_languages": [],
        "scip_relation_coverage_complete": True,
        "scip_recovery_attempts": 0,
        "scip_recovery_successes": 0,
        "scip_relation_recovery_attempts": 0,
        "scip_relation_recovery_successes": 0,
        "evolution_commits_considered": 0,
        "scip_detection_warnings": [],
        "scip_census_tool": "",
        "scip_census_tool_version": "",
    }
    kg_stats: dict[str, Any] = {}

    def _scip_failed_projects() -> list[dict[str, str]]:
        failed = scip_stats.get("scip_failed_projects")
        if not isinstance(failed, list):
            failed = []
            scip_stats["scip_failed_projects"] = failed
        return cast(list[dict[str, str]], failed)

    project_dicts: list[dict] = []
    snapshot_dicts: list[dict] = []
    file_metric_dicts: list[dict] = []
    evolution_payload: dict[str, object] | None = None
    await update_progress_artifact(
        progress_id, progress=10, description="Running SCIP polyglot indexing..."
    )  # type: ignore[misc]

    try:
        import tempfile

        from contextmine_core.semantic_snapshot.indexers.language_census import (
            EXTENSION_TO_LANGUAGE,
            build_language_census,
        )
        from contextmine_core.semantic_snapshot.models import Language

        def _normalize_language(value: object) -> str:
            return str(value or "").strip().lower()

        supported_languages = {language.value for language in Language}
        attempted_targets: set[tuple[str, str, str]] = set()

        def _project_key_for(proj_dict: dict) -> tuple[str, str, str]:
            language = _normalize_language(proj_dict.get("language"))
            root = str(Path(str(proj_dict.get("root_path", repo_path))).resolve())
            metadata = dict(proj_dict.get("metadata") or {})
            mode = "default"
            if metadata.get("relation_recovery"):
                mode = "relation_recovery"
            elif metadata.get("recovery_pass"):
                mode = "recovery"
            return language, root, mode

        def _snapshot_repo_file_path(file_info: dict[str, object], snapshot_meta: dict) -> str:
            raw_path = str(file_info.get("path") or "").strip().replace("\\", "/")
            if not raw_path:
                return ""

            path_obj = Path(raw_path)
            if path_obj.is_absolute():
                try:
                    return path_obj.resolve().relative_to(repo_path.resolve()).as_posix()
                except ValueError:
                    return raw_path.lstrip("./")

            repo_relative_root = str(snapshot_meta.get("repo_relative_root") or "").strip()
            normalized = raw_path.lstrip("./")
            if repo_relative_root:
                repo_relative_root = repo_relative_root.replace("\\", "/").strip("/")
                if normalized != repo_relative_root and not normalized.startswith(
                    f"{repo_relative_root}/"
                ):
                    normalized = f"{repo_relative_root}/{normalized}".strip("/")
            return normalized

        def _snapshot_file_language(
            *,
            repo_relative_path: str,
            file_info: dict[str, object],
            snapshot_language: str,
        ) -> str | None:
            explicit = _normalize_language(file_info.get("language"))
            if explicit in supported_languages:
                return explicit
            extension_language = EXTENSION_TO_LANGUAGE.get(Path(repo_relative_path).suffix.lower())
            if extension_language:
                return extension_language.value
            if snapshot_language in supported_languages:
                return snapshot_language
            return None

        def _collect_indexed_paths_by_language(
            snapshots: list[dict],
        ) -> dict[str, set[str]]:
            indexed: dict[str, set[str]] = {}
            for snapshot_dict in snapshots:
                snapshot_meta = dict(snapshot_dict.get("meta") or {})
                snapshot_language = _normalize_language(snapshot_meta.get("language"))
                files = snapshot_dict.get("files") or []
                if not isinstance(files, list):
                    continue
                for item in files:
                    if not isinstance(item, dict):
                        continue
                    repo_relative_path = _snapshot_repo_file_path(item, snapshot_meta)
                    if not repo_relative_path:
                        continue
                    language = _snapshot_file_language(
                        repo_relative_path=repo_relative_path,
                        file_info=item,
                        snapshot_language=snapshot_language,
                    )
                    if not language:
                        continue
                    indexed.setdefault(language, set()).add(repo_relative_path)
            return indexed

        def _collect_indexed_files_by_language(
            snapshots: list[dict],
        ) -> dict[str, int]:
            indexed_paths = _collect_indexed_paths_by_language(snapshots)
            return {language: len(paths) for language, paths in indexed_paths.items()}

        def _append_file_coverage_completion_snapshot(
            snapshots: list[dict],
            *,
            census_report: object,
        ) -> dict[str, int]:
            if not snapshots:
                return {}

            indexed_paths = _collect_indexed_paths_by_language(snapshots)
            missing_by_language: dict[str, set[str]] = {}

            file_stats = list(getattr(census_report, "file_stats", []) or [])
            for item in file_stats:
                language_obj = getattr(item, "language", None)
                language = _normalize_language(getattr(language_obj, "value", language_obj))
                if language not in supported_languages:
                    continue
                code_lines = int(getattr(item, "code", 0) or 0)
                if code_lines <= 0:
                    continue
                raw_path = getattr(item, "path", None)
                if not isinstance(raw_path, Path):
                    try:
                        raw_path = Path(str(raw_path))
                    except Exception:  # noqa: BLE001  # nosec B112
                        continue
                try:
                    repo_relative = raw_path.resolve().relative_to(repo_path.resolve()).as_posix()
                except Exception:  # noqa: BLE001  # nosec B112
                    continue
                if not repo_relative:
                    continue
                if repo_relative in indexed_paths.get(language, set()):
                    continue
                missing_by_language.setdefault(language, set()).add(repo_relative)

            if not missing_by_language:
                return {}

            for language, missing_paths in sorted(missing_by_language.items()):
                snapshot_meta = {
                    "language": language,
                    "repo_relative_root": "",
                    "completion_pass": "file_coverage",
                }
                snapshots.append(
                    {
                        "files": [
                            {"path": path, "language": language} for path in sorted(missing_paths)
                        ],
                        "symbols": [],
                        "occurrences": [],
                        "relations": [],
                        "meta": snapshot_meta,
                    }
                )

            return {language: len(paths) for language, paths in missing_by_language.items()}

        def _collect_relation_coverage_by_language(
            snapshots: list[dict],
        ) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
            totals: dict[str, int] = defaultdict(int)
            kind_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for snapshot_dict in snapshots:
                snapshot_meta = dict(snapshot_dict.get("meta") or {})
                language = _normalize_language(snapshot_meta.get("language"))
                if language not in supported_languages:
                    continue
                relations = snapshot_dict.get("relations") or []
                if not isinstance(relations, list):
                    continue
                for relation in relations:
                    if not isinstance(relation, dict):
                        continue
                    kind = _normalize_language(relation.get("kind"))
                    totals[language] += 1
                    if kind:
                        kind_totals[language][kind] += 1
            return dict(totals), {
                language: dict(counts) for language, counts in kind_totals.items()
            }

        semantic_relation_kinds = {"calls", "references", "imports", "extends", "implements"}

        def _missing_relation_languages(
            indexed_files_by_language: dict[str, int],
            relation_kinds_by_language: dict[str, dict[str, int]],
        ) -> list[str]:
            missing: list[str] = []
            for language, indexed_count in indexed_files_by_language.items():
                if int(indexed_count or 0) <= 0:
                    continue
                relation_kinds = relation_kinds_by_language.get(language) or {}
                semantic_edges = sum(
                    int(relation_kinds.get(kind, 0) or 0) for kind in semantic_relation_kinds
                )
                if semantic_edges <= 0:
                    missing.append(language)
            return sorted(set(missing))

        async def _index_project_target(proj_dict: dict) -> bool:
            language = _normalize_language(proj_dict.get("language"))
            project_root = str(Path(str(proj_dict.get("root_path", repo_path))).resolve())
            artifact_dict = await task_index_scip_project(proj_dict, scip_output_dir)
            if artifact_dict and artifact_dict.get("success"):
                scip_stats["scip_projects_indexed"] += 1
                scip_path = artifact_dict.get("scip_path")
                if scip_path:
                    snapshot_dict = await task_parse_scip_snapshot(str(scip_path))
                    if snapshot_dict:
                        project_root_path = Path(project_root)
                        try:
                            repo_relative_root = (
                                project_root_path.resolve()
                                .relative_to(repo_path.resolve())
                                .as_posix()
                            )
                        except ValueError:
                            repo_relative_root = ""

                        snapshot_meta = dict(snapshot_dict.get("meta") or {})
                        snapshot_meta.update(
                            {
                                "project_root": str(project_root_path.resolve()),
                                "repo_relative_root": repo_relative_root,
                                "language": language,
                            }
                        )
                        snapshot_dict["meta"] = snapshot_meta
                        snapshot_dicts.append(snapshot_dict)
                        scip_stats["scip_snapshots_parsed"] += 1
                        scip_stats["scip_symbols"] += len(snapshot_dict.get("symbols", []))
                        scip_stats["scip_relations"] += len(snapshot_dict.get("relations", []))
                        return True

                scip_stats["scip_projects_failed"] += 1
                _scip_failed_projects().append(
                    {
                        "language": language,
                        "project_root": project_root,
                        "error": "snapshot_parse_failed",
                    }
                )
                return False

            scip_stats["scip_projects_failed"] += 1
            _scip_failed_projects().append(
                {
                    "language": language,
                    "project_root": project_root,
                    "error": (
                        str(artifact_dict.get("error_message", "")).strip()
                        if artifact_dict
                        else "indexer_returned_no_artifact"
                    ),
                }
            )
            return False

        # Detect projects + language census diagnostics
        detect_result = await task_detect_scip_projects(repo_path)
        project_dicts = list(detect_result.get("projects") or [])
        diagnostics = detect_result.get("diagnostics") or {}
        scip_stats["scip_languages_detected"] = list(diagnostics.get("languages_detected") or [])
        scip_stats["scip_projects_by_language"] = dict(
            diagnostics.get("projects_by_language") or {}
        )
        scip_stats["scip_detection_warnings"] = list(diagnostics.get("warnings") or [])
        scip_stats["scip_census_tool"] = str(diagnostics.get("census_tool") or "")
        scip_stats["scip_census_tool_version"] = str(diagnostics.get("census_tool_version") or "")
        scip_stats["scip_projects_detected"] = len(project_dicts)

        census = build_language_census(repo_path)
        detected_files_by_language = {
            language.value: int(entry.files)
            for language, entry in census.entries.items()
            if int(entry.files) > 0
        }
        detected_code_by_language = {
            language.value: int(entry.code)
            for language, entry in census.entries.items()
            if int(entry.code) > 0
        }
        scip_stats["scip_detected_files_by_language"] = detected_files_by_language
        scip_stats["scip_detected_code_by_language"] = detected_code_by_language

        if project_dicts:
            # Create temp output directory for SCIP files
            scip_output_dir = Path(tempfile.mkdtemp(prefix="scip_"))

            # Index each project
            for proj_dict in project_dicts:
                attempted_targets.add(_project_key_for(proj_dict))
                await _index_project_target(proj_dict)

            indexed_files_by_language = _collect_indexed_files_by_language(snapshot_dicts)
            missing_languages = sorted(
                language
                for language, file_count in detected_files_by_language.items()
                if file_count > 0 and indexed_files_by_language.get(language, 0) == 0
            )

            if missing_languages:
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [
                    f"missing_language_index_coverage_initial:{language}"
                    for language in missing_languages
                ]

            for language in missing_languages:
                candidates = [
                    proj
                    for proj in project_dicts
                    if _normalize_language(proj.get("language")) == language
                ]
                fallback_target = dict(candidates[0]) if candidates else {}
                fallback_target["language"] = language
                fallback_target["root_path"] = str(repo_path.resolve())
                fallback_metadata = dict(fallback_target.get("metadata") or {})
                fallback_metadata["recovery_pass"] = True
                fallback_target["metadata"] = fallback_metadata

                key = _project_key_for(fallback_target)
                if key in attempted_targets:
                    continue

                attempted_targets.add(key)
                scip_stats["scip_recovery_attempts"] += 1
                recovered = await _index_project_target(fallback_target)
                if recovered:
                    scip_stats["scip_recovery_successes"] += 1

            indexed_files_by_language = _collect_indexed_files_by_language(snapshot_dicts)
            relation_counts_by_language, relation_kinds_by_language = (
                _collect_relation_coverage_by_language(snapshot_dicts)
            )
            missing_relation_languages = _missing_relation_languages(
                indexed_files_by_language,
                relation_kinds_by_language,
            )

            for language in missing_relation_languages:
                candidates = [
                    proj
                    for proj in project_dicts
                    if _normalize_language(proj.get("language")) == language
                ]
                relation_recovery_target = dict(candidates[0]) if candidates else {}
                relation_recovery_target["language"] = language
                relation_recovery_target["root_path"] = str(
                    Path(str(relation_recovery_target.get("root_path", repo_path))).resolve()
                )
                relation_recovery_metadata = dict(relation_recovery_target.get("metadata") or {})
                relation_recovery_metadata["recovery_pass"] = True
                relation_recovery_metadata["relation_recovery"] = True
                if language == "php":
                    relation_recovery_metadata["force_install_deps"] = True
                relation_recovery_target["metadata"] = relation_recovery_metadata

                relation_recovery_key = _project_key_for(relation_recovery_target)
                if relation_recovery_key in attempted_targets:
                    continue

                attempted_targets.add(relation_recovery_key)
                scip_stats["scip_relation_recovery_attempts"] += 1
                recovered = await _index_project_target(relation_recovery_target)
                if recovered:
                    scip_stats["scip_relation_recovery_successes"] += 1

            indexed_files_by_language = _collect_indexed_files_by_language(snapshot_dicts)
            relation_counts_by_language, relation_kinds_by_language = (
                _collect_relation_coverage_by_language(snapshot_dicts)
            )
            missing_relation_languages = _missing_relation_languages(
                indexed_files_by_language,
                relation_kinds_by_language,
            )

            completion_added_by_language = _append_file_coverage_completion_snapshot(
                snapshot_dicts,
                census_report=census,
            )
            if completion_added_by_language:
                scip_stats["scip_completion_added_files_by_language"] = completion_added_by_language
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [
                    f"scip_completion_files_added:{language}:{count}"
                    for language, count in sorted(completion_added_by_language.items())
                ]
                indexed_files_by_language = _collect_indexed_files_by_language(snapshot_dicts)

            missing_languages = sorted(
                language
                for language, file_count in detected_files_by_language.items()
                if file_count > 0 and indexed_files_by_language.get(language, 0) == 0
            )

            scip_stats["scip_indexed_files_by_language"] = indexed_files_by_language
            scip_stats["scip_missing_languages"] = missing_languages
            scip_stats["scip_coverage_complete"] = len(missing_languages) == 0
            scip_stats["scip_relation_counts_by_language"] = relation_counts_by_language
            scip_stats["scip_relation_kinds_by_language"] = relation_kinds_by_language
            scip_stats["scip_missing_relation_languages"] = missing_relation_languages
            scip_stats["scip_relation_coverage_complete"] = len(missing_relation_languages) == 0
            if missing_languages:
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [
                    f"missing_language_index_coverage:{language}" for language in missing_languages
                ]
            if missing_relation_languages:
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [
                    f"missing_relation_coverage:{language}"
                    for language in missing_relation_languages
                ]

            scip_stats["scip_degraded"] = bool(scip_stats["scip_projects_failed"])
            if not scip_stats["scip_coverage_complete"]:
                scip_stats["scip_degraded"] = True
            if not scip_stats["scip_relation_coverage_complete"]:
                scip_stats["scip_degraded"] = True

            logger.info(
                "SCIP indexing complete: %d/%d projects (%d failed), %d snapshots, %d symbols, %d relations",
                scip_stats["scip_projects_indexed"],
                scip_stats["scip_projects_detected"],
                scip_stats["scip_projects_failed"],
                scip_stats["scip_snapshots_parsed"],
                scip_stats["scip_symbols"],
                scip_stats["scip_relations"],
            )
        else:
            missing_languages = sorted(detected_files_by_language.keys())
            scip_stats["scip_indexed_files_by_language"] = {}
            scip_stats["scip_missing_languages"] = missing_languages
            scip_stats["scip_coverage_complete"] = len(missing_languages) == 0
            scip_stats["scip_relation_counts_by_language"] = {}
            scip_stats["scip_relation_kinds_by_language"] = {}
            scip_stats["scip_missing_relation_languages"] = []
            scip_stats["scip_relation_coverage_complete"] = True
            if missing_languages:
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [
                    f"missing_language_index_coverage:{language}" for language in missing_languages
                ]
                scip_stats["scip_degraded"] = True
    except Exception as e:
        logger.warning("SCIP indexing failed: %s", e)
        scip_stats["scip_degraded"] = True
        scip_stats["scip_detection_warnings"] = list(scip_stats["scip_detection_warnings"]) + [
            f"scip_indexing_exception:{e}"
        ]

    settings = get_settings()
    missing_language_coverage = list(scip_stats.get("scip_missing_languages") or [])
    if settings.scip_require_language_coverage and missing_language_coverage:
        logger.warning(
            "SCIP language coverage incomplete; continuing with completion-pass data (missing=%s)",
            ",".join(missing_language_coverage),
        )
        scip_stats["scip_language_coverage_gate"] = "warn"
    missing_relation_coverage = list(scip_stats.get("scip_missing_relation_languages") or [])
    if settings.scip_require_relation_coverage and missing_relation_coverage:
        logger.warning(
            "SCIP relation coverage incomplete; continuing with available semantic edges "
            "(missing_relations=%s)",
            ",".join(missing_relation_coverage),
        )
        scip_stats["scip_relation_coverage_gate"] = "warn"
    if settings.scip_require_php_relation_coverage and "php" in missing_relation_coverage:
        logger.warning(
            "SCIP PHP relation coverage incomplete; continuing and supplementing from "
            "document symbol graph (missing_relations=%s)",
            ",".join(missing_relation_coverage),
        )
        scip_stats["scip_php_relation_coverage_gate"] = "warn"

    await update_progress_artifact(
        progress_id, progress=14, description="Extracting structural code metrics..."
    )  # type: ignore[misc]
    if snapshot_dicts and project_dicts:
        from contextmine_core.metrics import flatten_metric_bundles, run_polyglot_metrics_pipeline

        evolution_window_days = int(settings.twin_evolution_window_days)
        step_timeout_seconds = _sync_blocking_step_timeout_seconds()
        scip_stats["evolution_window_days"] = evolution_window_days
        try:
            bundles = await _run_blocking_with_timeout(
                "metrics_pipeline",
                step_timeout_seconds,
                run_polyglot_metrics_pipeline,
                repo_root=repo_path,
                project_dicts=project_dicts,
                snapshot_dicts=snapshot_dicts,
                strict_mode=settings.metrics_strict_mode,
                metrics_languages=settings.metrics_languages,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structural metrics pipeline failed for source %s: %s", source.id, exc)
            scip_stats["scip_degraded"] = True
            scip_stats["scip_detection_warnings"] = list(
                scip_stats.get("scip_detection_warnings") or []
            ) + [f"metrics_pipeline_exception:{exc}"]
            bundles = []

        file_metric_dicts = [record.to_dict() for record in flatten_metric_bundles(bundles or [])]
        scip_stats["structural_metric_files"] = len(file_metric_dicts)
        if file_metric_dicts:
            target_files = {
                str(metric.get("file_path", "")).strip()
                for metric in file_metric_dicts
                if str(metric.get("file_path", "")).strip()
            }
            try:
                git_metrics_by_file = await _run_blocking_with_timeout(
                    "git_change_metrics",
                    step_timeout_seconds,
                    compute_git_change_metrics,
                    git_repo,
                    target_files,
                    since_days=evolution_window_days,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Git change metrics failed for source %s: %s", source.id, exc)
                scip_stats["scip_degraded"] = True
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [f"git_change_metrics_exception:{exc}"]
                git_metrics_by_file = {
                    path: {"change_frequency": 0, "insertions": 0, "deletions": 0, "churn": 0}
                    for path in target_files
                }
            files_with_history = 0
            total_change_frequency = 0.0
            total_churn = 0.0

            for metric in file_metric_dicts:
                file_path = str(metric.get("file_path", "")).strip()
                git_values = git_metrics_by_file.get(
                    file_path,
                    {"change_frequency": 0, "insertions": 0, "deletions": 0, "churn": 0},
                )
                change_frequency = float(git_values.get("change_frequency", 0) or 0.0)
                churn = float(git_values.get("churn", 0) or 0.0)
                if change_frequency > 0:
                    files_with_history += 1
                total_change_frequency += change_frequency
                total_churn += churn

                sources = dict(metric.get("sources") or {})
                sources["change_frequency"] = {
                    "provider": "git",
                    "window": f"{evolution_window_days}d",
                    "no_merges": True,
                    "renames_followed": False,
                    "unit": "commits",
                }
                sources["churn"] = {
                    "provider": "git",
                    "window": f"{evolution_window_days}d",
                    "unit": "lines_changed",
                    "formula": "insertions+deletions",
                }
                metric["sources"] = sources
                metric["change_frequency"] = change_frequency
                metric["churn"] = churn

            scip_stats["git_metric_files_targeted"] = len(target_files)
            scip_stats["git_metric_files_with_history"] = files_with_history
            scip_stats["git_metric_total_change_frequency"] = total_change_frequency
            scip_stats["git_metric_total_churn"] = total_churn
            try:
                temporal_pair_cap = _sync_temporal_coupling_max_files_per_commit()
                evolution_payload = await _run_blocking_with_timeout(
                    "git_evolution_snapshots",
                    step_timeout_seconds,
                    compute_git_evolution_snapshots,
                    git_repo,
                    target_files,
                    window_days=evolution_window_days,
                    max_files_per_commit=temporal_pair_cap,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Git evolution snapshots failed for source %s: %s", source.id, exc)
                scip_stats["scip_degraded"] = True
                scip_stats["scip_detection_warnings"] = list(
                    scip_stats.get("scip_detection_warnings") or []
                ) + [f"git_evolution_snapshots_exception:{exc}"]
                evolution_payload = {
                    "ownership_rows": [],
                    "coupling_rows": [],
                    "stats": {
                        "window_days": evolution_window_days,
                        "max_files_per_commit": _sync_temporal_coupling_max_files_per_commit(),
                        "pairing_truncated_commits": 0,
                        "commits_scanned": 0,
                        "commits_considered": 0,
                        "files_seen": 0,
                        "ownership_rows": 0,
                        "coupling_rows": 0,
                    },
                    "warnings": [f"git_evolution_snapshots_exception:{exc}"],
                }
            evolution_stats = dict(evolution_payload.get("stats") or {})
            scip_stats["evolution_commits_scanned"] = int(evolution_stats.get("commits_scanned", 0))
            scip_stats["evolution_commits_considered"] = int(
                evolution_stats.get("commits_considered", 0)
            )
            scip_stats["evolution_files_seen"] = int(evolution_stats.get("files_seen", 0))
            scip_stats["evolution_ownership_rows"] = int(evolution_stats.get("ownership_rows", 0))
            scip_stats["evolution_coupling_rows"] = int(evolution_stats.get("coupling_rows", 0))
            scip_stats["evolution_warnings"] = list(evolution_payload.get("warnings") or [])

    await update_progress_artifact(
        progress_id, progress=15, description="Detecting changed files..."
    )  # type: ignore[misc]

    # Get changed and deleted files
    changed_files, deleted_files = get_changed_files(git_repo, old_sha, new_sha)

    # Track documents to chunk (doc_id, content, file_path)
    docs_to_chunk: list[tuple[str, str, str | None]] = []

    # Chunk, symbol, and embedding stats
    total_chunks_created = 0
    total_chunks_deleted = 0
    total_chunks_embedded = 0
    total_chunks_deduplicated = 0
    total_tokens_used = 0
    total_symbols_created = 0
    total_symbols_deleted = 0

    total_files = len(changed_files) + len(deleted_files)
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=20, description=f"Processing {total_files} files..."
    )

    async with get_session() as session:
        # Process deleted files
        if deleted_files:
            for file_path in deleted_files:
                uri = build_uri(owner, repo, file_path, branch)
                result = await session.execute(
                    delete(Document).where(
                        Document.source_id == source.id,
                        Document.uri == uri,
                    )
                )
                if result.rowcount > 0:  # type: ignore[union-attr]
                    stats.docs_deleted += 1
                    stats.files_deleted += 1

        # Process changed files
        for file_path in changed_files:
            stats.files_scanned += 1

            # Check eligibility
            if not is_eligible_file(Path(file_path), repo_path):
                stats.files_skipped += 1
                continue

            # Read content
            content = read_file_content(repo_path, file_path)
            if content is None:
                stats.files_skipped += 1
                continue

            stats.files_indexed += 1

            # Compute hash and URI
            content_hash = compute_content_hash(content)
            uri = build_uri(owner, repo, file_path, branch)
            title = get_file_title(Path(file_path))

            # Check if document exists
            result = await session.execute(select(Document).where(Document.uri == uri))
            existing_doc = result.scalar_one_or_none()

            if existing_doc:
                # Update if content changed
                if existing_doc.content_hash != content_hash:
                    existing_doc.content_markdown = content
                    existing_doc.content_hash = content_hash
                    existing_doc.title = title
                    existing_doc.updated_at = datetime.now(UTC)
                    stats.docs_updated += 1
                    # Mark for re-chunking
                    docs_to_chunk.append((str(existing_doc.id), content, file_path))
                existing_doc.last_seen_at = run_started_at
            else:
                # Create new document
                new_doc = Document(
                    source_id=source.id,
                    uri=uri,
                    title=title,
                    content_markdown=content,
                    content_hash=content_hash,
                    meta={
                        "file_path": file_path,
                        "owner": owner,
                        "repo": repo,
                        "branch": branch,
                    },
                    last_seen_at=run_started_at,
                )
                session.add(new_doc)
                await session.flush()  # Get the ID
                stats.docs_created += 1
                # Mark for chunking
                docs_to_chunk.append((str(new_doc.id), content, file_path))

        # Hard delete documents not seen in this run (for full index)
        if old_sha is None:
            result = await session.execute(
                delete(Document).where(
                    Document.source_id == source.id,
                    Document.last_seen_at < run_started_at,
                )
            )
            stats.docs_deleted += result.rowcount  # type: ignore[union-attr]

        # Update source cursor
        result = await session.execute(select(Source).where(Source.id == source.id))
        db_source = result.scalar_one()
        db_source.cursor = new_sha
        db_source.last_run_at = datetime.now(UTC)
        db_source.next_run_at = datetime.now(UTC) + timedelta(
            minutes=db_source.schedule_interval_minutes
        )

        await session.commit()

    # Find documents missing chunks (fault tolerance for interrupted syncs)
    async with get_session() as session:
        # Subquery for documents that have at least one chunk
        docs_with_chunks = select(Chunk.document_id).distinct().subquery()

        # Find documents for this source that have no chunks
        result = await session.execute(
            select(Document.id, Document.content_markdown, Document.uri)
            .outerjoin(docs_with_chunks, Document.id == docs_with_chunks.c.document_id)
            .where(
                Document.source_id == source.id,
                docs_with_chunks.c.document_id.is_(None),
            )
        )
        unchunked_docs = result.all()

        # Add unchunked docs to the processing queue
        for doc_id, content, uri in unchunked_docs:
            if content:
                # Extract file path from URI (format: git://github.com/owner/repo/path?ref=branch)
                file_path = uri.split("?")[0].split("/", 5)[-1] if "/" in uri else None
                docs_to_chunk.append((str(doc_id), content, file_path))

    # Run chunk maintenance and embedding for changed/new/unchunked documents
    collection_id_str = str(source.collection_id)
    doc_acc = await _process_document_batch(docs_to_chunk, collection_id_str, progress_id)
    total_chunks_created = doc_acc.chunks_created
    total_chunks_deleted = doc_acc.chunks_deleted
    total_chunks_embedded = doc_acc.chunks_embedded
    total_chunks_deduplicated = doc_acc.chunks_deduplicated
    total_tokens_used = doc_acc.tokens_used
    total_symbols_created = doc_acc.symbols_created
    total_symbols_deleted = doc_acc.symbols_deleted
    docs_processing_failures = doc_acc.processing_failures
    docs_processing_timeouts = doc_acc.processing_timeouts
    docs_processing_error_samples = doc_acc.error_samples
    total_docs_candidate = doc_acc._total_candidate  # type: ignore[attr-defined]
    docs_deferred = doc_acc._docs_deferred  # type: ignore[attr-defined]

    # Build Knowledge Graph (FILE/SYMBOL + semantic entities + communities).
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=94, description="Building knowledge graph..."
    )
    changed_doc_ids = doc_acc._changed_doc_ids  # type: ignore[attr-defined]
    try:
        kg_timeout_seconds = _knowledge_graph_build_timeout_seconds()
        kg_stats = await asyncio.wait_for(
            build_knowledge_graph(
                source_id=str(source.id),
                collection_id=collection_id_str,
                changed_doc_ids=changed_doc_ids,
            ),
            timeout=kg_timeout_seconds,
        )
    except TimeoutError:
        kg_timeout_seconds = _knowledge_graph_build_timeout_seconds()
        logger.warning(
            "Knowledge graph build timed out for source %s after %ss",
            source.id,
            kg_timeout_seconds,
        )
        kg_stats = {"kg_errors": [f"pipeline: timeout_after_{kg_timeout_seconds}s"]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Knowledge graph build failed for source %s: %s", source.id, exc)
        kg_errors = list((kg_stats or {}).get("kg_errors") or [])
        kg_errors.append(f"pipeline: {exc}")
        kg_stats = {"kg_errors": kg_errors}

    # Materialize deterministic interface/spec surfaces (OpenAPI/GraphQL/Protobuf/jobs)
    surface_stats: dict[str, int] = {}
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=95, description="Materializing interface/spec surfaces..."
    )
    try:
        surface_stats = await asyncio.wait_for(
            materialize_surface_catalog_for_source(
                source_id=str(source.id),
                collection_id=collection_id_str,
            ),
            timeout=_sync_blocking_step_timeout_seconds(),
        )
    except Exception as exc:
        logger.warning("Surface materialization failed for source %s: %s", source.id, exc)
        surface_stats = {}

    # Build digital twin from indexed documents and semantic snapshots
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=96, description="Building digital twin..."
    )
    twin_timeout_seconds = _twin_graph_build_timeout_seconds()
    try:
        twin_stats = await asyncio.wait_for(
            build_twin_graph(
                str(source.id),
                collection_id_str,
                snapshot_dicts=snapshot_dicts,
                changed_doc_ids=changed_doc_ids,
                file_metrics=file_metric_dicts,
                evolution_payload=evolution_payload,
            ),
            timeout=twin_timeout_seconds,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"TWIN_BUILD_TIMEOUT: source={source.id} timeout={twin_timeout_seconds}s"
        ) from exc

    await update_progress_artifact(progress_id, progress=98, description="Saving results...")  # type: ignore[misc]

    async with get_session() as session:
        import uuid as uuid_module

        from contextmine_core.models import TwinScenario, TwinSourceVersion

        # Update sync run
        result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
        db_run = result.scalar_one()
        db_run.status = SyncRunStatus.SUCCESS
        db_run.finished_at = datetime.now(UTC)
        scenario_version = None
        scenario_id_raw = twin_stats.get("twin_asis_scenario_id")
        if scenario_id_raw:
            scenario = (
                await session.execute(
                    select(TwinScenario).where(
                        TwinScenario.id == uuid_module.UUID(str(scenario_id_raw))
                    )
                )
            ).scalar_one_or_none()
            scenario_version = int(scenario.version) if scenario else None

        if source_version_id:
            await set_source_version_status(
                session,
                source_version_id=source_version_id,
                status="ready",
                stats={
                    "commit_sha": new_sha,
                    "sync_run_id": str(sync_run.id),
                    "surface_extract": surface_stats,
                    "knowledge_extract": kg_stats,
                    "docs_chunk_queue_total": int(total_docs_candidate),
                    "docs_chunk_deferred": int(docs_deferred),
                    "docs_processing_failures": int(docs_processing_failures),
                    "docs_processing_timeouts": int(docs_processing_timeouts),
                    "docs_processing_error_samples": list(docs_processing_error_samples)[:25],
                    "joern_status": "ready" if joern_ok else "failed",
                    "joern_error": joern_error if not joern_ok else "",
                    "scip_projects_detected": scip_stats.get("scip_projects_detected", 0),
                    "scip_projects_indexed": scip_stats.get("scip_projects_indexed", 0),
                    "scip_snapshots_parsed": scip_stats.get("scip_snapshots_parsed", 0),
                    "scip_projects_failed": scip_stats.get("scip_projects_failed", 0),
                    "scip_failed_projects": scip_stats.get("scip_failed_projects", []),
                    "scip_projects_by_language": scip_stats.get("scip_projects_by_language", {}),
                    "scip_detected_files_by_language": scip_stats.get(
                        "scip_detected_files_by_language", {}
                    ),
                    "scip_detected_code_by_language": scip_stats.get(
                        "scip_detected_code_by_language", {}
                    ),
                    "scip_indexed_files_by_language": scip_stats.get(
                        "scip_indexed_files_by_language", {}
                    ),
                    "scip_missing_languages": scip_stats.get("scip_missing_languages", []),
                    "scip_coverage_complete": bool(scip_stats.get("scip_coverage_complete", True)),
                    "scip_relation_counts_by_language": scip_stats.get(
                        "scip_relation_counts_by_language", {}
                    ),
                    "scip_relation_kinds_by_language": scip_stats.get(
                        "scip_relation_kinds_by_language", {}
                    ),
                    "scip_missing_relation_languages": scip_stats.get(
                        "scip_missing_relation_languages", []
                    ),
                    "scip_relation_coverage_complete": bool(
                        scip_stats.get("scip_relation_coverage_complete", True)
                    ),
                    "scip_recovery_attempts": scip_stats.get("scip_recovery_attempts", 0),
                    "scip_recovery_successes": scip_stats.get("scip_recovery_successes", 0),
                    "scip_relation_recovery_attempts": scip_stats.get(
                        "scip_relation_recovery_attempts", 0
                    ),
                    "scip_relation_recovery_successes": scip_stats.get(
                        "scip_relation_recovery_successes", 0
                    ),
                    "scip_completion_added_files_by_language": scip_stats.get(
                        "scip_completion_added_files_by_language", {}
                    ),
                    "scip_language_coverage_gate": scip_stats.get(
                        "scip_language_coverage_gate", "n/a"
                    ),
                    "scip_relation_coverage_gate": scip_stats.get(
                        "scip_relation_coverage_gate", "n/a"
                    ),
                    "scip_php_relation_coverage_gate": scip_stats.get(
                        "scip_php_relation_coverage_gate", "n/a"
                    ),
                    "scip_degraded": bool(scip_stats.get("scip_degraded", False)),
                    "evolution_window_days": int(scip_stats.get("evolution_window_days", 0)),
                    "evolution_commits_scanned": int(
                        scip_stats.get("evolution_commits_scanned", 0)
                    ),
                    "evolution_commits_considered": int(
                        scip_stats.get("evolution_commits_considered", 0)
                    ),
                    "evolution_files_seen": int(scip_stats.get("evolution_files_seen", 0)),
                    "evolution_warnings": list(scip_stats.get("evolution_warnings", []) or []),
                    "twin_nodes_upserted": int(twin_stats.get("twin_nodes_upserted", 0)),
                    "twin_edges_upserted": int(twin_stats.get("twin_edges_upserted", 0)),
                    "evolution_ownership_rows": int(twin_stats.get("evolution_ownership_rows", 0)),
                    "evolution_coupling_rows": int(twin_stats.get("evolution_coupling_rows", 0)),
                    "fitness_findings_written": int(twin_stats.get("fitness_findings_written", 0)),
                    "fitness_findings_by_type": dict(
                        twin_stats.get("fitness_findings_by_type", {}) or {}
                    ),
                    "metrics_requested_files": int(twin_stats.get("metrics_requested_files", 0)),
                    "metrics_mapped_files": int(twin_stats.get("metrics_mapped_files", 0)),
                    "metrics_unmapped_sample": list(
                        twin_stats.get("metrics_unmapped_sample", []) or []
                    )[:25],
                    "metrics_gate": (
                        "pass"
                        if int(twin_stats.get("metrics_mapped_files", 0))
                        >= int(twin_stats.get("metrics_requested_files", 0))
                        else "fail"
                    ),
                    "scenario_id": scenario_id_raw,
                    "scenario_version": scenario_version,
                },
                finished=True,
            )
            stale_rows = (
                (
                    await session.execute(
                        select(TwinSourceVersion).where(
                            TwinSourceVersion.source_id == source.id,
                            TwinSourceVersion.id != source_version_id,
                            TwinSourceVersion.status == "ready",
                        )
                    )
                )
                .scalars()
                .all()
            )
            for stale in stale_rows:
                stale.status = "stale"
                stale.updated_at = datetime.now(UTC)

            await record_twin_event(
                session,
                collection_id=source.collection_id,
                scenario_id=uuid_module.UUID(str(scenario_id_raw)) if scenario_id_raw else None,
                source_id=source.id,
                source_version_id=source_version_id,
                event_type="materialization_complete",
                status="ready",
                payload={
                    "scenario_version": scenario_version,
                    "nodes_upserted": int(twin_stats.get("twin_nodes_upserted", 0)),
                    "edges_upserted": int(twin_stats.get("twin_edges_upserted", 0)),
                    "nodes_deactivated": int(twin_stats.get("twin_nodes_deactivated", 0)),
                    "edges_deactivated": int(twin_stats.get("twin_edges_deactivated", 0)),
                    "sample_node_keys": list(twin_stats.get("sample_node_keys", []))[:20],
                },
                idempotency_key=f"materialization_complete:{source.id}:{new_sha}",
            )

        db_run.stats = {
            "files_scanned": stats.files_scanned,
            "files_indexed": stats.files_indexed,
            "files_skipped": stats.files_skipped,
            "files_deleted": stats.files_deleted,
            "docs_created": stats.docs_created,
            "docs_updated": stats.docs_updated,
            "docs_deleted": stats.docs_deleted,
            "docs_unchunked_recovered": len(unchunked_docs),
            "docs_chunk_queue_total": total_docs_candidate,
            "docs_chunk_deferred": docs_deferred,
            "docs_processing_failures": docs_processing_failures,
            "docs_processing_timeouts": docs_processing_timeouts,
            "docs_processing_error_samples": list(docs_processing_error_samples)[:25],
            "chunks_created": total_chunks_created,
            "chunks_deleted": total_chunks_deleted,
            "chunks_embedded": total_chunks_embedded,
            "chunks_deduplicated": total_chunks_deduplicated,
            "embedding_tokens_used": total_tokens_used,
            "symbols_created": total_symbols_created,
            "symbols_deleted": total_symbols_deleted,
            "commit_sha": new_sha,
            "previous_sha": old_sha,
            # SCIP Polyglot Indexing stats
            "scip_projects_detected": scip_stats.get("scip_projects_detected", 0),
            "scip_projects_indexed": scip_stats.get("scip_projects_indexed", 0),
            "scip_snapshots_parsed": scip_stats.get("scip_snapshots_parsed", 0),
            "scip_projects_failed": scip_stats.get("scip_projects_failed", 0),
            "scip_symbols": scip_stats.get("scip_symbols", 0),
            "scip_relations": scip_stats.get("scip_relations", 0),
            "scip_degraded": bool(scip_stats.get("scip_degraded", False)),
            "scip_failed_projects": scip_stats.get("scip_failed_projects", []),
            "scip_languages_detected": scip_stats.get("scip_languages_detected", []),
            "scip_projects_by_language": scip_stats.get("scip_projects_by_language", {}),
            "scip_detected_files_by_language": scip_stats.get(
                "scip_detected_files_by_language", {}
            ),
            "scip_detected_code_by_language": scip_stats.get("scip_detected_code_by_language", {}),
            "scip_indexed_files_by_language": scip_stats.get("scip_indexed_files_by_language", {}),
            "scip_missing_languages": scip_stats.get("scip_missing_languages", []),
            "scip_coverage_complete": bool(scip_stats.get("scip_coverage_complete", True)),
            "scip_relation_counts_by_language": scip_stats.get(
                "scip_relation_counts_by_language", {}
            ),
            "scip_relation_kinds_by_language": scip_stats.get(
                "scip_relation_kinds_by_language", {}
            ),
            "scip_missing_relation_languages": scip_stats.get(
                "scip_missing_relation_languages", []
            ),
            "scip_relation_coverage_complete": bool(
                scip_stats.get("scip_relation_coverage_complete", True)
            ),
            "scip_recovery_attempts": scip_stats.get("scip_recovery_attempts", 0),
            "scip_recovery_successes": scip_stats.get("scip_recovery_successes", 0),
            "scip_relation_recovery_attempts": scip_stats.get("scip_relation_recovery_attempts", 0),
            "scip_relation_recovery_successes": scip_stats.get(
                "scip_relation_recovery_successes", 0
            ),
            "scip_completion_added_files_by_language": scip_stats.get(
                "scip_completion_added_files_by_language", {}
            ),
            "scip_language_coverage_gate": scip_stats.get("scip_language_coverage_gate", "n/a"),
            "scip_relation_coverage_gate": scip_stats.get("scip_relation_coverage_gate", "n/a"),
            "scip_php_relation_coverage_gate": scip_stats.get(
                "scip_php_relation_coverage_gate", "n/a"
            ),
            "scip_detection_warnings": scip_stats.get("scip_detection_warnings", []),
            "scip_census_tool": scip_stats.get("scip_census_tool", ""),
            "scip_census_tool_version": scip_stats.get("scip_census_tool_version", ""),
            # Twin stats
            "twin_nodes_upserted": twin_stats.get("twin_nodes_upserted", 0),
            "twin_edges_upserted": twin_stats.get("twin_edges_upserted", 0),
            "twin_metric_nodes_enriched": twin_stats.get("twin_metric_nodes_enriched", 0),
            "metrics_requested_files": twin_stats.get("metrics_requested_files", 0),
            "metrics_mapped_files": twin_stats.get("metrics_mapped_files", 0),
            "metrics_unmapped_sample": twin_stats.get("metrics_unmapped_sample", []),
            "metrics_gate": (
                "pass"
                if int(twin_stats.get("metrics_mapped_files", 0))
                >= int(twin_stats.get("metrics_requested_files", 0))
                else "fail"
            ),
            "twin_metrics_snapshots": twin_stats.get("twin_metrics_snapshots", 0),
            "twin_validation_snapshots": twin_stats.get("twin_validation_snapshots", 0),
            "twin_asis_scenario_id": twin_stats.get("twin_asis_scenario_id"),
            "structural_metric_files": scip_stats.get("structural_metric_files", 0),
            "git_metric_files_targeted": scip_stats.get("git_metric_files_targeted", 0),
            "git_metric_files_with_history": scip_stats.get("git_metric_files_with_history", 0),
            "git_metric_total_change_frequency": scip_stats.get(
                "git_metric_total_change_frequency", 0.0
            ),
            "git_metric_total_churn": scip_stats.get("git_metric_total_churn", 0.0),
            "evolution_window_days": scip_stats.get("evolution_window_days", 0),
            "evolution_commits_scanned": scip_stats.get("evolution_commits_scanned", 0),
            "evolution_commits_considered": scip_stats.get("evolution_commits_considered", 0),
            "evolution_files_seen": scip_stats.get("evolution_files_seen", 0),
            "evolution_warnings": scip_stats.get("evolution_warnings", []),
            "evolution_ownership_rows": twin_stats.get("evolution_ownership_rows", 0),
            "evolution_coupling_rows": twin_stats.get("evolution_coupling_rows", 0),
            "fitness_findings_written": twin_stats.get("fitness_findings_written", 0),
            "fitness_findings_by_type": twin_stats.get("fitness_findings_by_type", {}),
            "fitness_findings_warnings": twin_stats.get("fitness_findings_warnings", []),
            "source_version_id": str(source_version_id) if source_version_id else None,
            "joern_status": "ready" if joern_ok else "failed",
            "joern_error": joern_error if not joern_ok else "",
            "joern_cpg_path": str(cpg_path) if joern_ok else None,
            "joern_server_url": settings.joern_server_url,
            # Knowledge Graph extraction stats
            "kg_file_nodes": kg_stats.get("kg_file_nodes", 0),
            "kg_symbol_nodes": kg_stats.get("kg_symbol_nodes", 0),
            "kg_business_rules": kg_stats.get("kg_business_rules", 0),
            "kg_tables": kg_stats.get("kg_tables", 0),
            "kg_endpoints": kg_stats.get("kg_endpoints", 0),
            "kg_jobs": kg_stats.get("kg_jobs", 0),
            "kg_semantic_entities": kg_stats.get("kg_semantic_entities", 0),
            "kg_semantic_relationships": kg_stats.get("kg_semantic_relationships", 0),
            "kg_communities_l0": kg_stats.get("kg_communities_l0", 0),
            "kg_communities_l1": kg_stats.get("kg_communities_l1", 0),
            "kg_communities_l2": kg_stats.get("kg_communities_l2", 0),
            "kg_summaries_created": kg_stats.get("kg_summaries_created", 0),
            "kg_embeddings_created": kg_stats.get("kg_embeddings_created", 0),
            "kg_errors": list(kg_stats.get("kg_errors", []) or []),
            # Surface extraction stats
            "surface_files_scanned": surface_stats.get("surface_files_scanned", 0),
            "surface_files_recognized": surface_stats.get("surface_files_recognized", 0),
            "surface_parse_errors": surface_stats.get("surface_parse_errors", 0),
            "surface_endpoint_nodes": surface_stats.get("endpoint_nodes", 0),
            "surface_endpoint_handler_links": surface_stats.get("endpoint_handler_links", 0),
            "surface_graphql_nodes": surface_stats.get("graphql_nodes", 0),
            "surface_proto_nodes": surface_stats.get("proto_nodes", 0),
            "surface_job_nodes": surface_stats.get("job_nodes", 0),
            "surface_edges_created": surface_stats.get("edges_created", 0),
        }

        await session.commit()

    scenario_for_behavioral = str(twin_stats.get("twin_asis_scenario_id") or "")
    if scenario_for_behavioral:
        background = asyncio.create_task(
            _materialize_behavioral_layers_impl(
                source_id=str(source.id),
                collection_id=collection_id_str,
                scenario_id=scenario_for_behavioral,
                source_version_id=str(source_version_id) if source_version_id else None,
            )
        )
        background.add_done_callback(_log_background_task_failure)

    await update_progress_artifact(progress_id, progress=100, description="Sync complete!")  # type: ignore[misc]

    return stats


def _build_page_meta(page: Any, base_meta: dict | None = None) -> dict:
    """Build meta dict from page HTTP cache headers."""
    meta = dict(base_meta or {})
    if page.etag:
        meta["etag"] = page.etag
    if page.last_modified:
        meta["last_modified"] = page.last_modified
    return meta


async def _process_web_page(
    session: Any,
    source: Source,
    page: Any,
    base_url: str,
    run_started_at: datetime,
    stats: Any,
) -> tuple[str, str, str | None] | None:
    """Process a single web page: upsert document and return chunk tuple if content changed."""
    uri = page.url
    title = get_page_title(page)
    result = await session.execute(select(Document).where(Document.uri == uri))
    existing_doc = result.scalar_one_or_none()

    if existing_doc:
        if existing_doc.content_hash != page.content_hash:
            existing_doc.content_markdown = page.markdown
            existing_doc.content_hash = page.content_hash
            existing_doc.title = title
            existing_doc.updated_at = datetime.now(UTC)
            existing_doc.meta = _build_page_meta(page, existing_doc.meta)
            stats.docs_updated += 1
            existing_doc.last_seen_at = run_started_at
            return (str(existing_doc.id), page.markdown, None)
        existing_doc.last_seen_at = run_started_at
        return None

    new_doc = Document(
        source_id=source.id,
        uri=uri,
        title=title,
        content_markdown=page.markdown,
        content_hash=page.content_hash,
        meta=_build_page_meta(page, {"base_url": base_url}),
        last_seen_at=run_started_at,
    )
    session.add(new_doc)
    await session.flush()
    stats.docs_created += 1
    return (str(new_doc.id), page.markdown, None)


@traced_task()
@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=exponential_backoff(backoff_factor=3),
    retry_jitter_factor=0.5,
    tags=[TAG_WEB_CRAWL],
    task_run_name="sync-web-{source.url}",
)
async def sync_web_source(
    source: Source,
    sync_run: SyncRun,
    run_started_at: datetime,
) -> WebSyncStats:
    """Sync a web source, creating/updating/deleting documents.

    Tagged with web-crawl for concurrency limiting.
    Retries on transient network failures.
    """
    stats = WebSyncStats()

    # Get config with sensible defaults
    config = source.config or {}
    # start_url: where to begin crawling (user's original URL)
    # base_url: path prefix for scoping (derived from start_url)
    start_url = config.get("start_url", source.url)
    base_url = config.get("base_url", start_url)  # Fall back to start_url for old sources
    max_pages = config.get("max_pages", DEFAULT_MAX_PAGES)
    delay_ms = config.get("delay_ms", DEFAULT_DELAY_MS)

    if not base_url:
        raise ValueError("Web source missing base_url in config")

    # Create progress artifact
    progress_id = await create_progress_artifact(  # type: ignore[misc]
        progress=0.0,
        description=f"Starting crawl of {start_url}...",
    )

    await update_progress_artifact(
        progress_id, progress=5, description=f"Crawling up to {max_pages} pages..."
    )  # type: ignore[misc]

    # Run the spider with rate limiting
    pages = run_spider_md(
        base_url=base_url,
        start_url=start_url,
        max_pages=max_pages,
        delay_ms=delay_ms,
    )

    stats.pages_crawled = len(pages)
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=40, description=f"Crawled {len(pages)} pages, processing..."
    )

    # Track documents to chunk (doc_id, content, file_path)
    docs_to_chunk: list[tuple[str, str, str | None]] = []

    # Chunk, symbol, and embedding stats
    total_chunks_created = 0
    total_chunks_deleted = 0
    total_chunks_embedded = 0
    total_chunks_deduplicated = 0
    total_tokens_used = 0
    total_symbols_created = 0
    total_symbols_deleted = 0

    async with get_session() as session:
        for page in pages:
            doc_pair = await _process_web_page(
                session, source, page, base_url, run_started_at, stats
            )
            if doc_pair is not None:
                docs_to_chunk.append(doc_pair)

        # Hard delete documents not seen in this run (pages that disappeared)
        result = await session.execute(
            delete(Document).where(
                Document.source_id == source.id,
                Document.last_seen_at < run_started_at,
            )
        )
        stats.docs_deleted = result.rowcount or 0  # type: ignore[union-attr]

        # Update source timestamps (no cursor for web sources)
        result = await session.execute(select(Source).where(Source.id == source.id))
        db_source = result.scalar_one()
        db_source.last_run_at = datetime.now(UTC)
        db_source.next_run_at = datetime.now(UTC) + timedelta(
            minutes=db_source.schedule_interval_minutes
        )

        await session.commit()

    # Find documents missing chunks (fault tolerance for interrupted syncs)
    async with get_session() as session:
        docs_with_chunks = select(Chunk.document_id).distinct().subquery()

        result = await session.execute(
            select(Document.id, Document.content_markdown, Document.uri)
            .outerjoin(docs_with_chunks, Document.id == docs_with_chunks.c.document_id)
            .where(
                Document.source_id == source.id,
                docs_with_chunks.c.document_id.is_(None),
            )
        )
        unchunked_docs = result.all()

        for doc_id, content, _uri in unchunked_docs:
            if content:
                docs_to_chunk.append((str(doc_id), content, None))

    # Run chunk maintenance and embedding for changed/new/unchunked documents
    collection_id_str = str(source.collection_id)
    doc_acc = await _process_document_batch(docs_to_chunk, collection_id_str, progress_id)
    total_chunks_created = doc_acc.chunks_created
    total_chunks_deleted = doc_acc.chunks_deleted
    total_chunks_embedded = doc_acc.chunks_embedded
    total_chunks_deduplicated = doc_acc.chunks_deduplicated
    total_tokens_used = doc_acc.tokens_used
    total_symbols_created = doc_acc.symbols_created
    total_symbols_deleted = doc_acc.symbols_deleted
    docs_processing_failures = doc_acc.processing_failures
    docs_processing_timeouts = doc_acc.processing_timeouts
    docs_processing_error_samples = doc_acc.error_samples
    total_docs_candidate = doc_acc._total_candidate  # type: ignore[attr-defined]
    docs_deferred = doc_acc._docs_deferred  # type: ignore[attr-defined]

    # Materialize deterministic interface/spec surfaces (OpenAPI/GraphQL/Protobuf/jobs)
    surface_stats: dict[str, int] = {}
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=95, description="Materializing interface/spec surfaces..."
    )
    try:
        surface_stats = await asyncio.wait_for(
            materialize_surface_catalog_for_source(
                source_id=str(source.id),
                collection_id=collection_id_str,
            ),
            timeout=_sync_blocking_step_timeout_seconds(),
        )
    except Exception as exc:
        logger.warning("Surface materialization failed for source %s: %s", source.id, exc)
        surface_stats = {}

    # Build Twin graph from indexed documents
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=96, description="Building digital twin..."
    )
    changed_doc_ids = doc_acc._changed_doc_ids  # type: ignore[attr-defined]
    twin_timeout_seconds = _twin_graph_build_timeout_seconds()
    try:
        twin_stats = await asyncio.wait_for(
            build_twin_graph(
                str(source.id),
                collection_id_str,
                snapshot_dicts=[],
                changed_doc_ids=changed_doc_ids,
            ),
            timeout=twin_timeout_seconds,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"TWIN_BUILD_TIMEOUT: source={source.id} timeout={twin_timeout_seconds}s"
        ) from exc

    await update_progress_artifact(progress_id, progress=98, description="Saving results...")  # type: ignore[misc]

    async with get_session() as session:
        # Update sync run
        result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
        db_run = result.scalar_one()
        db_run.status = SyncRunStatus.SUCCESS
        db_run.finished_at = datetime.now(UTC)
        db_run.stats = {
            "pages_crawled": stats.pages_crawled,
            "pages_skipped": stats.pages_skipped,
            "docs_created": stats.docs_created,
            "docs_updated": stats.docs_updated,
            "docs_deleted": stats.docs_deleted,
            "docs_unchunked_recovered": len(unchunked_docs),
            "docs_chunk_queue_total": total_docs_candidate,
            "docs_chunk_deferred": docs_deferred,
            "docs_processing_failures": docs_processing_failures,
            "docs_processing_timeouts": docs_processing_timeouts,
            "docs_processing_error_samples": list(docs_processing_error_samples)[:25],
            "chunks_created": total_chunks_created,
            "chunks_deleted": total_chunks_deleted,
            "chunks_embedded": total_chunks_embedded,
            "chunks_deduplicated": total_chunks_deduplicated,
            "embedding_tokens_used": total_tokens_used,
            "symbols_created": total_symbols_created,
            "symbols_deleted": total_symbols_deleted,
            # Twin stats
            "twin_nodes_upserted": twin_stats.get("twin_nodes_upserted", 0),
            "twin_edges_upserted": twin_stats.get("twin_edges_upserted", 0),
            "twin_metrics_snapshots": twin_stats.get("twin_metrics_snapshots", 0),
            "twin_validation_snapshots": twin_stats.get("twin_validation_snapshots", 0),
            "twin_asis_scenario_id": twin_stats.get("twin_asis_scenario_id"),
            # Surface extraction stats
            "surface_files_scanned": surface_stats.get("surface_files_scanned", 0),
            "surface_files_recognized": surface_stats.get("surface_files_recognized", 0),
            "surface_parse_errors": surface_stats.get("surface_parse_errors", 0),
            "surface_endpoint_nodes": surface_stats.get("endpoint_nodes", 0),
            "surface_endpoint_handler_links": surface_stats.get("endpoint_handler_links", 0),
            "surface_graphql_nodes": surface_stats.get("graphql_nodes", 0),
            "surface_proto_nodes": surface_stats.get("proto_nodes", 0),
            "surface_job_nodes": surface_stats.get("job_nodes", 0),
            "surface_edges_created": surface_stats.get("edges_created", 0),
        }

        await session.commit()

    scenario_for_behavioral = str(twin_stats.get("twin_asis_scenario_id") or "")
    if scenario_for_behavioral:
        background = asyncio.create_task(
            _materialize_behavioral_layers_impl(
                source_id=str(source.id),
                collection_id=collection_id_str,
                scenario_id=scenario_for_behavioral,
                source_version_id=None,
            )
        )
        background.add_done_callback(_log_background_task_failure)

    await update_progress_artifact(progress_id, progress=100, description="Sync complete!")  # type: ignore[misc]

    return stats


def _build_failure_stats(sync_run_id: str, error_message: str) -> dict[str, object]:
    """Parse gate-failure error messages into structured stats."""
    failure_stats: dict[str, object] = {
        "sync_run_id": sync_run_id,
        "error": error_message,
    }
    if "METRICS_GATE_FAILED" in error_message:
        failure_stats["metrics_gate"] = "fail"
        _parse_metrics_gate_details(failure_stats, error_message)
    if "SCIP_GATE_FAILED" in error_message:
        failure_stats["scip_gate"] = "fail"
        _parse_scip_gate_details(failure_stats, error_message)
    return failure_stats


def _parse_metrics_gate_details(stats: dict[str, object], text: str) -> None:
    if "(mapped=" not in text or ", metrics=" not in text:
        return
    try:
        mapped_part = text.split("(mapped=", 1)[1]
        stats["metrics_mapped_files"] = int(mapped_part.split(",", 1)[0])
        stats["metrics_requested_files"] = int(mapped_part.split("metrics=", 1)[1].split(")", 1)[0])
    except Exception:  # noqa: BLE001
        pass


def _parse_scip_gate_details(stats: dict[str, object], text: str) -> None:
    if "(missing=" in text:
        try:
            missing_raw = text.split("(missing=", 1)[1].split(")", 1)[0]
            stats["scip_missing_languages"] = [lang for lang in missing_raw.split(",") if lang]
        except Exception:  # noqa: BLE001
            pass
    if "(missing_relations=" in text:
        try:
            missing_rel_raw = text.split("(missing_relations=", 1)[1].split(")", 1)[0]
            stats["scip_missing_relation_languages"] = [
                lang for lang in missing_rel_raw.split(",") if lang
            ]
        except Exception:  # noqa: BLE001
            pass


async def _handle_sync_failure(source: Source, sync_run: SyncRun, exc: Exception) -> SyncRun:
    """Handle a sync failure: log, mark run as failed, update source timestamps."""
    from contextmine_core.models import TwinSourceVersion
    from contextmine_core.twin import record_twin_event, set_source_version_status

    error_message = str(exc).strip() or repr(exc)
    if error_message == repr(exc):
        logger.exception("Source sync failed without explicit message for source %s", source.id)
    else:
        logger.exception("Source sync failed for source %s: %s", source.id, error_message)

    async with get_session() as session:
        result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
        db_run = result.scalar_one()
        db_run.status = SyncRunStatus.FAILED
        db_run.finished_at = datetime.now(UTC)
        db_run.error = error_message

        failed_source_version = (
            await session.execute(
                select(TwinSourceVersion)
                .where(
                    TwinSourceVersion.source_id == source.id,
                    TwinSourceVersion.status.in_(["queued", "materializing"]),
                )
                .order_by(TwinSourceVersion.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if failed_source_version:
            failure_stats = _build_failure_stats(str(sync_run.id), error_message)
            await set_source_version_status(
                session,
                source_version_id=failed_source_version.id,
                status="failed",
                stats=failure_stats,
                finished=True,
            )
            await record_twin_event(
                session,
                collection_id=source.collection_id,
                scenario_id=None,
                source_id=source.id,
                source_version_id=failed_source_version.id,
                event_type="materialization_failed",
                status="failed",
                payload={"sync_run_id": str(sync_run.id)},
                idempotency_key=f"materialization_failed:{source.id}:{sync_run.id}",
                error=error_message,
            )

        result = await session.execute(select(Source).where(Source.id == source.id))
        db_source = result.scalar_one()
        db_source.last_run_at = datetime.now(UTC)
        db_source.next_run_at = datetime.now(UTC) + timedelta(
            minutes=db_source.schedule_interval_minutes
        )

        await session.commit()
        await session.refresh(db_run)
        return db_run


@traced_task()
@task(
    retries=DEFAULT_RETRIES,
    retry_delay_seconds=exponential_backoff(backoff_factor=2),
    retry_jitter_factor=0.5,
    tags=[TAG_DB_HEAVY],
)
async def sync_source(source: Source) -> SyncRun | None:
    """Sync a single source.

    Creates a sync run record and executes the appropriate sync logic.
    Checks for existing running syncs to prevent concurrent syncs for the same source.

    Retries automatically on transient failures (network issues, DB timeouts).
    """
    run_started_at = datetime.now(UTC)

    async with get_session() as session:
        # Lock the source row to prevent race conditions
        # Use FOR UPDATE SKIP LOCKED so concurrent attempts just skip
        source_lock = await session.execute(
            select(Source).where(Source.id == source.id).with_for_update(skip_locked=True)
        )
        locked_source = source_lock.scalar_one_or_none()
        if not locked_source:
            # Another sync already has the lock
            return None

        # Recover stale running rows so new syncs are not blocked indefinitely.
        running_runs = select(SyncRun).where(
            SyncRun.source_id == source.id,
            SyncRun.status == SyncRunStatus.RUNNING,
        )
        running_rows = (await session.execute(running_runs)).scalars().all()
        now = datetime.now(UTC)
        stale_cutoff = now - SYNC_RUN_STALE_AFTER
        stale_recovered = 0

        for candidate in running_rows:
            started = candidate.started_at
            if started and started < stale_cutoff:
                candidate.status = SyncRunStatus.FAILED
                candidate.finished_at = now
                candidate.error = (
                    "AUTO_RECOVERED_STALE_RUN: worker did not finish within "
                    f"{int(SYNC_RUN_STALE_AFTER.total_seconds())}s"
                )
                stale_recovered += 1

        if stale_recovered:
            await session.commit()

        # Check if there's still a fresh running sync for this source.
        existing_run = await session.execute(
            select(SyncRun).where(
                SyncRun.source_id == source.id,
                SyncRun.status == SyncRunStatus.RUNNING,
            )
        )
        if existing_run.scalar_one_or_none():
            # Another sync is already running for this source
            return None

        # Create sync run record
        sync_run = SyncRun(
            source_id=source.id,
            status=SyncRunStatus.RUNNING,
        )
        session.add(sync_run)
        await session.commit()
        await session.refresh(sync_run)

    try:
        if source.type == SourceType.GITHUB:
            await sync_github_source(source, sync_run, run_started_at)
        else:
            # Web sources
            await sync_web_source(source, sync_run, run_started_at)

    except Exception as e:
        return await _handle_sync_failure(source, sync_run, e)

    # Refresh and return
    async with get_session() as session:
        result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
        return result.scalar_one()


async def _fail_running_sync_runs_for_source(source_id: str, reason: str) -> int:
    """Mark currently running sync rows for a source as failed."""
    import uuid as uuid_module

    source_uuid = uuid_module.UUID(source_id)
    now = datetime.now(UTC)
    async with get_session() as session:
        running_rows = (
            (
                await session.execute(
                    select(SyncRun).where(
                        SyncRun.source_id == source_uuid,
                        SyncRun.status == SyncRunStatus.RUNNING,
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in running_rows:
            row.status = SyncRunStatus.FAILED
            row.finished_at = now
            row.error = reason
        if running_rows:
            await session.commit()
        return len(running_rows)


async def _fail_coverage_ingest_job(
    job_id: str,
    *,
    status: str = "failed",
    error_code: str,
    error_detail: str,
    stats: dict | None = None,
) -> dict:
    """Persist a terminal ingest job state."""
    import uuid as uuid_module

    async with get_session() as session:
        result = await session.execute(
            select(CoverageIngestJob).where(CoverageIngestJob.id == uuid_module.UUID(job_id))
        )
        job = result.scalar_one_or_none()
        if not job:
            return {"status": "missing_job"}

        job.status = status
        job.error_code = error_code
        job.error_detail = error_detail
        job.stats = stats or {}
        await session.commit()

    return {"status": status, "error_code": error_code, "error_detail": error_detail}


def _find_relevant_metric_files(file_nodes: list, source_id_str: str) -> set[str]:
    """Filter twin file nodes to those with structural metrics from the given source."""
    from contextmine_core.pathing import canonicalize_repo_relative_path

    relevant: set[str] = set()
    for node in file_nodes:
        if node.kind != "file":
            continue
        if not node.natural_key.startswith(_FILE_NODE_PREFIX):
            continue
        meta = dict(node.meta or {})
        if str(meta.get("source_id") or "") != source_id_str:
            continue
        if not meta.get("metrics_structural_ready"):
            continue
        file_path = canonicalize_repo_relative_path(
            node.natural_key.removeprefix(_FILE_NODE_PREFIX)
        )
        if file_path:
            relevant.add(file_path)
    return relevant


@traced_flow()
@flow(name="ingest_coverage_metrics", retries=2, retry_delay_seconds=5)
async def ingest_coverage_metrics(job_id: str) -> dict:
    """Ingest CI-pushed coverage reports and apply coverage to Twin metrics."""
    import tempfile
    import uuid as uuid_module

    from contextmine_core.metrics.coverage_reports import parse_coverage_reports
    from contextmine_core.models import TwinNode
    from contextmine_core.twin import (
        apply_coverage_metrics_to_scenario,
        get_or_create_as_is_scenario,
        refresh_metric_snapshots,
    )

    from contextmine_worker.github_sync import get_repo_path

    try:
        job_uuid = uuid_module.UUID(job_id)
    except ValueError:
        return {
            "status": "failed",
            "error_code": "INGEST_APPLY_FAILED",
            "error_detail": f"Invalid job_id: {job_id}",
        }

    async with get_session() as session:
        locked = await session.execute(
            select(CoverageIngestJob)
            .where(CoverageIngestJob.id == job_uuid)
            .with_for_update(skip_locked=True)
        )
        job = locked.scalar_one_or_none()
        if not job:
            return {"status": "missing"}
        if job.status in {"applied", "failed", "rejected"}:
            return {
                "status": job.status,
                "error_code": job.error_code,
                "error_detail": job.error_detail,
            }
        job.status = "processing"
        job.error_code = None
        job.error_detail = None
        await session.commit()

    async with get_session() as session:
        result = await session.execute(
            select(CoverageIngestJob).where(CoverageIngestJob.id == job_uuid)
        )
        job = result.scalar_one_or_none()
        if not job:
            return {"status": "missing"}

        source = (
            await session.execute(select(Source).where(Source.id == job.source_id))
        ).scalar_one_or_none()
        if not source:
            return await _fail_coverage_ingest_job(
                job_id,
                status="rejected",
                error_code="INGEST_APPLY_FAILED",
                error_detail="Source not found",
            )
        if source.type != SourceType.GITHUB:
            return await _fail_coverage_ingest_job(
                job_id,
                status="rejected",
                error_code="INGEST_APPLY_FAILED",
                error_detail="Coverage ingest is only supported for GitHub sources",
            )
        if not source.cursor or str(source.cursor) != str(job.commit_sha):
            return await _fail_coverage_ingest_job(
                job_id,
                status="rejected",
                error_code="INGEST_SHA_MISMATCH",
                error_detail=f"Expected source cursor {source.cursor}, got {job.commit_sha}",
            )

        scenario = await get_or_create_as_is_scenario(session, source.collection_id, user_id=None)
        job.scenario_id = scenario.id
        await session.flush()

        file_nodes = (
            (await session.execute(select(TwinNode).where(TwinNode.scenario_id == scenario.id)))
            .scalars()
            .all()
        )
        relevant_files = _find_relevant_metric_files(file_nodes, str(source.id))

        if not relevant_files:
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                error_code="INGEST_NO_RELEVANT_FILES",
                error_detail="No structural metric files found for source in current AS-IS scenario",
            )

        repo_path = get_repo_path(str(source.id))
        if not repo_path.exists():
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                error_code="INGEST_APPLY_FAILED",
                error_detail=f"Repository path not found: {repo_path}",
            )

        reports = (
            (
                await session.execute(
                    select(CoverageIngestReport).where(CoverageIngestReport.job_id == job.id)
                )
            )
            .scalars()
            .all()
        )
        if not reports:
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                error_code="INGEST_PARSE_FAILED",
                error_detail="No report files attached to ingest job",
            )

        settings = get_settings()
        max_payload = settings.coverage_ingest_max_payload_mb * 1024 * 1024
        total_bytes = sum(len(report.report_bytes or b"") for report in reports)
        if total_bytes > max_payload:
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                status="rejected",
                error_code="INGEST_PAYLOAD_TOO_LARGE",
                error_detail=(
                    f"Payload size {total_bytes} exceeds limit {settings.coverage_ingest_max_payload_mb}MB"
                ),
            )

        with tempfile.TemporaryDirectory(prefix=f"coverage_ingest_{job.id}_") as tmp_dir:
            report_paths: list[Path] = []
            for idx, report in enumerate(reports):
                safe_name = Path(report.filename or f"report_{idx}").name
                temp_path = Path(tmp_dir) / safe_name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_bytes(report.report_bytes or b"")
                report_paths.append(temp_path)

            coverage_map, coverage_sources, protocol_by_report = parse_coverage_reports(
                report_paths=report_paths,
                repo_root=repo_path,
                project_root=repo_path,
            )

            for report, report_path in zip(reports, report_paths, strict=True):
                report.protocol_detected = protocol_by_report.get(str(report_path))
                diagnostics = dict(report.diagnostics or {})
                diagnostics["file_size"] = len(report.report_bytes or b"")
                if report.protocol_detected is None:
                    diagnostics["warning"] = "unsupported_or_unrecognized_protocol"
                report.diagnostics = diagnostics

        if not coverage_map:
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                error_code="INGEST_PROTOCOL_UNSUPPORTED",
                error_detail="No supported coverage protocol detected in uploaded reports",
                stats={"reports_total": len(reports)},
            )

        matched_coverage = {k: v for k, v in coverage_map.items() if k in relevant_files}
        if not matched_coverage:
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                error_code="INGEST_COVERAGE_PATH_MISMATCH",
                error_detail=(
                    "Coverage parsed successfully, but no files matched relevant Twin file nodes"
                ),
                stats={
                    "reports_total": len(reports),
                    "coverage_files": len(coverage_map),
                    "relevant_files": len(relevant_files),
                },
            )

        matched_sources = {
            file_path: coverage_sources.get(file_path, {}) for file_path in matched_coverage
        }
        applied_files = await apply_coverage_metrics_to_scenario(
            session=session,
            scenario_id=scenario.id,
            source_id=source.id,
            coverage_map=matched_coverage,
            coverage_sources=matched_sources,
            commit_sha=job.commit_sha,
            ingest_job_id=job.id,
        )
        if applied_files == 0:
            await session.commit()
            return await _fail_coverage_ingest_job(
                job_id,
                error_code="INGEST_NO_RELEVANT_FILES",
                error_detail="Coverage matched files but no Twin nodes were updated",
            )

        snapshot_rows = await refresh_metric_snapshots(session, scenario.id)
        job.status = "applied"
        job.error_code = None
        job.error_detail = None
        job.stats = {
            "reports_total": len(reports),
            "files_parsed": len(coverage_map),
            "files_matched": len(matched_coverage),
            "files_applied": applied_files,
            "metric_snapshots": snapshot_rows,
        }
        await session.commit()

        return {
            "status": "applied",
            "job_id": str(job.id),
            "scenario_id": str(scenario.id),
            "files_applied": applied_files,
            "metric_snapshots": snapshot_rows,
        }


@traced_flow()
@flow(name="sync_due_sources")
async def sync_due_sources() -> dict:
    """Flow to sync all sources that are due.

    This flow:
    1. Queries for sources where next_run_at <= now and enabled = true
    2. For each source, creates a SyncRun record and executes sync
    3. Updates source timestamps after completion
    4. Uses advisory locks to prevent concurrent syncs for the same source
    """
    sources = await get_due_sources()

    if not sources:
        return {"synced": 0, "skipped": 0, "sources": []}

    results = []
    skipped = 0
    timeout_seconds = _sync_source_timeout_seconds()
    for source in sources:
        try:
            if timeout_seconds > 0:
                sync_run = await asyncio.wait_for(sync_source(source), timeout=timeout_seconds)
            else:
                sync_run = await sync_source(source)
            if sync_run is None:
                skipped += 1
                continue
            results.append(
                {
                    "source_id": str(source.id),
                    "sync_run_id": str(sync_run.id),
                    "status": sync_run.status.value,
                }
            )
        except TimeoutError:
            reason = f"AUTO_TIMEOUT_SYNC_SOURCE: exceeded {timeout_seconds}s in scheduler"
            recovered = await _fail_running_sync_runs_for_source(str(source.id), reason)
            results.append(
                {
                    "source_id": str(source.id),
                    "error": reason,
                    "recovered_running_rows": recovered,
                }
            )
        except Exception as e:
            results.append(
                {
                    "source_id": str(source.id),
                    "error": str(e),
                }
            )

    return {"synced": len(results), "skipped": skipped, "sources": results}


@traced_flow()
@flow(name="sync_single_source")
async def sync_single_source(source_id: str, source_url: str | None = None) -> dict:
    """Flow to sync a single source by ID.

    This is the preferred flow for on-demand syncing, triggered by:
    - The scheduler when a source is due
    - The API when user clicks "Sync Now"
    - Automations or webhooks

    Args:
        source_id: UUID of the source to sync
        source_url: Optional URL for display purposes in Prefect UI

    Returns:
        Dict with sync results
    """
    import uuid as uuid_module

    # Load the source from database
    async with get_session() as session:
        result = await session.execute(
            select(Source).where(Source.id == uuid_module.UUID(source_id))
        )
        source = result.scalar_one_or_none()

    if not source:
        return {"error": f"Source {source_id} not found"}

    if not source.enabled:
        return {"error": f"Source {source_id} is disabled", "skipped": True}

    timeout_seconds = _sync_source_timeout_seconds()
    try:
        if timeout_seconds > 0:
            sync_run = await asyncio.wait_for(sync_source(source), timeout=timeout_seconds)
        else:
            sync_run = await sync_source(source)
        if sync_run is None:
            return {"source_id": source_id, "skipped": True, "reason": "lock_not_acquired"}

        return {
            "source_id": source_id,
            "sync_run_id": str(sync_run.id),
            "status": sync_run.status.value,
            "stats": sync_run.stats,
        }
    except TimeoutError:
        reason = f"AUTO_TIMEOUT_SYNC_SOURCE: exceeded {timeout_seconds}s in sync_single_source"
        recovered = await _fail_running_sync_runs_for_source(source_id, reason)
        return {"source_id": source_id, "error": reason, "recovered_running_rows": recovered}
    except Exception as e:
        return {"source_id": source_id, "error": str(e)}


@traced_task()
@task(retries=0, tags=[TAG_DB_HEAVY])
async def task_repair_twin_file_path_canonicalization(
    collection_id: str | None = None,
    scenario_id: str | None = None,
) -> dict:
    """Repair legacy twin file keys and refresh snapshots for affected scenarios."""
    import uuid as uuid_module

    from contextmine_core.twin import (
        record_twin_event,
        refresh_metric_snapshots,
        repair_twin_file_path_canonicalization,
    )

    collection_uuid = uuid_module.UUID(collection_id) if collection_id else None
    scenario_uuid = uuid_module.UUID(scenario_id) if scenario_id else None

    async with get_session() as session:
        repair_stats = await repair_twin_file_path_canonicalization(
            session,
            collection_id=collection_uuid,
            scenario_id=scenario_uuid,
        )

        scenario_snapshot_counts: dict[str, int] = {}
        for scenario_id_raw in repair_stats.get("scenarios_changed", []):
            scenario_uuid_local = uuid_module.UUID(str(scenario_id_raw))
            snapshot_count = await refresh_metric_snapshots(session, scenario_uuid_local)
            scenario_snapshot_counts[str(scenario_uuid_local)] = snapshot_count

        repair_stats["metric_snapshots_refreshed"] = int(sum(scenario_snapshot_counts.values()))
        repair_stats["metric_snapshots_by_scenario"] = scenario_snapshot_counts

        for collection_id_raw in repair_stats.get("collections_changed", []):
            collection_uuid_local = uuid_module.UUID(str(collection_id_raw))
            await record_twin_event(
                session=session,
                collection_id=collection_uuid_local,
                scenario_id=None,
                source_id=None,
                source_version_id=None,
                event_type="file_path_canonicalization_repair",
                status="ready",
                payload={
                    "legacy_candidates": int(repair_stats.get("legacy_candidates", 0)),
                    "updated_in_place": int(repair_stats.get("updated_in_place", 0)),
                    "duplicates_deactivated": int(repair_stats.get("duplicates_deactivated", 0)),
                    "edges_rewired": int(repair_stats.get("edges_rewired", 0)),
                    "meta_paths_updated": int(repair_stats.get("meta_paths_updated", 0)),
                    "metric_snapshots_refreshed": int(
                        repair_stats.get("metric_snapshots_refreshed", 0)
                    ),
                    "scenarios_changed": list(repair_stats.get("scenarios_changed", [])),
                },
                idempotency_key=f"file_path_repair:{collection_uuid_local}:{uuid_module.uuid4()}",
            )

        await session.commit()
        return repair_stats


@traced_flow()
@flow(name="repair_twin_file_paths")
async def repair_twin_file_paths(
    collection_id: str | None = None,
    scenario_id: str | None = None,
) -> dict:
    """Admin flow to repair legacy twin file-path canonicalization defects."""
    return await task_repair_twin_file_path_canonicalization(
        collection_id=collection_id,
        scenario_id=scenario_id,
    )
