"""Prefect flows for syncing sources.

Features:
- Native Prefect scheduling (no custom scheduler)
- Automatic retries with exponential backoff
- Concurrency limits via tags (github-api, embedding-api, web-crawl)
- Task result caching for idempotent operations
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from contextmine_core import (
    Chunk,
    Collection,
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
from contextmine_worker.embeddings import get_embedder, parse_embedding_model_spec
from contextmine_worker.github_sync import (
    SyncStats,
    build_uri,
    clone_or_pull_repo,
    compute_content_hash,
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

        # Get embeddings from API
        result = await embedder.embed_batch(texts)
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

    from contextmine_core.models import Document
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
            kg_stats = await build_knowledge_graph_for_source(session, source_uuid)
            stats["kg_file_nodes"] = kg_stats.get("file_nodes_created", 0)
            stats["kg_symbol_nodes"] = kg_stats.get("symbol_nodes_created", 0)
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
    # Only processes documents that were created/updated in this sync run
    try:
        from contextmine_core.analyzer.extractors.rules import (
            build_rules_graph,
            extract_rules_from_file,
        )
        from contextmine_core.treesitter.languages import detect_language

        all_extractions = []

        if changed_doc_ids is not None and len(changed_doc_ids) == 0:
            logger.info("No changed documents - skipping business rule extraction")
        else:
            async with get_session() as session:
                # Get documents that need rule extraction
                if changed_doc_ids:
                    # Incremental: only changed documents
                    result = await session.execute(
                        select(Document.id, Document.uri, Document.content_markdown).where(
                            Document.id.in_([uuid_module.UUID(d) for d in changed_doc_ids])
                        )
                    )
                else:
                    # First run: all documents for this source
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
                    # Extract file path from URI
                    file_path = uri.split("?")[0].split("/", 5)[-1] if "/" in uri else uri
                    # Process all files with supported Tree-sitter languages
                    if detect_language(file_path) is not None:
                        try:
                            rule_result = await extract_rules_from_file(
                                file_path, content, research_llm
                            )
                            if rule_result.rules:
                                all_extractions.append(rule_result)
                        except Exception as e:
                            logger.debug("Rule extraction failed for %s: %s", file_path, e)

                # Build graph nodes for all business rules
                if all_extractions:
                    rule_stats = await build_rules_graph(session, collection_uuid, all_extractions)
                    stats["kg_business_rules"] = rule_stats.get("rules_created", 0)
                    await session.commit()
                    logger.info("Extracted %d business rules", stats["kg_business_rules"])

    except Exception as e:
        logger.warning("Failed to extract business rules: %s", e)
        stats["kg_errors"].append(f"rules: {e}")

    # Step 3: Extract ERM from Alembic migrations
    try:
        from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
        from contextmine_core.analyzer.extractors.erm import (
            ERMExtractor,
            build_erm_graph,
            save_erd_artifact,
        )

        erm_extractor = ERMExtractor()

        async with get_session() as session:
            result = await session.execute(
                select(Document.uri, Document.content_markdown).where(
                    Document.source_id == source_uuid
                )
            )
            docs = result.all()

            for uri, content in docs:
                if not content:
                    continue
                file_path = uri.split("?")[0].split("/", 5)[-1] if "/" in uri else uri
                # Detect Alembic migrations
                if "alembic/versions" in file_path and file_path.endswith(".py"):
                    try:
                        extraction = extract_from_alembic(file_path, content)
                        erm_extractor.add_alembic_extraction(extraction)
                    except Exception as e:
                        logger.debug("ERM extraction failed for %s: %s", file_path, e)

            # Build graph and save ERD if we found tables
            if erm_extractor.schema.tables:
                erm_stats = await build_erm_graph(session, collection_uuid, erm_extractor.schema)
                stats["kg_tables"] = erm_stats.get("table_nodes_created", 0)
                await save_erd_artifact(session, collection_uuid, erm_extractor.schema)
                await session.commit()
                logger.info("Extracted %d database tables", stats["kg_tables"])

    except Exception as e:
        logger.warning("Failed to extract ERM: %s", e)
        stats["kg_errors"].append(f"erm: {e}")

    # Step 4: Extract Surface Catalog (OpenAPI, GraphQL, Protobuf, Jobs)
    try:
        from contextmine_core.analyzer.extractors.surface import (
            SurfaceCatalogExtractor,
            build_surface_graph,
        )

        surface_extractor = SurfaceCatalogExtractor()

        async with get_session() as session:
            result = await session.execute(
                select(Document.uri, Document.content_markdown).where(
                    Document.source_id == source_uuid
                )
            )
            docs = result.all()

            for uri, content in docs:
                if not content:
                    continue
                file_path = uri.split("?")[0].split("/", 5)[-1] if "/" in uri else uri
                try:
                    surface_extractor.add_file(file_path, content)
                except Exception as e:
                    logger.debug("Surface extraction failed for %s: %s", file_path, e)

            # Build graph if we found anything
            catalog = surface_extractor.catalog
            has_surfaces = (
                catalog.openapi_specs
                or catalog.graphql_schemas
                or catalog.protobuf_files
                or catalog.job_definitions
            )
            if has_surfaces:
                surface_stats = await build_surface_graph(session, collection_uuid, catalog)
                stats["kg_endpoints"] = surface_stats.get("endpoint_nodes_created", 0)
                stats["kg_jobs"] = surface_stats.get("job_nodes_created", 0)
                await session.commit()
                logger.info(
                    "Extracted %d endpoints, %d jobs",
                    stats["kg_endpoints"],
                    stats["kg_jobs"],
                )

    except Exception as e:
        logger.warning("Failed to extract surfaces: %s", e)
        stats["kg_errors"].append(f"surface: {e}")

    # Step 5: Extract semantic entities using LLM (for proper GraphRAG)
    try:
        from contextmine_core.knowledge.extraction import (
            extract_from_documents,
            persist_semantic_entities,
        )

        async with get_session() as session:
            # Extract semantic entities from documents
            # Uses embedding similarity for cross-language entity resolution
            extraction_batch = await extract_from_documents(
                session=session,
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
            stats["kg_semantic_relationships"] = extraction_stats.get("relationships_created", 0)
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
) -> dict:
    """Build digital twin graph from semantic snapshots (SCIP/LSIF) and existing KG."""
    del source_id, changed_doc_ids
    import uuid as uuid_module

    from contextmine_core.graph.age import sync_scenario_to_age
    from contextmine_core.semantic_snapshot.models import Snapshot
    from contextmine_core.twin import (
        get_or_create_as_is_scenario,
        ingest_snapshot_into_as_is,
        refresh_metric_snapshots,
        seed_scenario_from_knowledge_graph,
    )
    from contextmine_core.validation import refresh_validation_snapshots

    collection_uuid = uuid_module.UUID(collection_id)
    stats: dict[str, int | str] = {
        "twin_nodes_upserted": 0,
        "twin_edges_upserted": 0,
        "twin_metrics_snapshots": 0,
        "twin_validation_snapshots": 0,
    }

    async with get_session() as session:
        as_is = await get_or_create_as_is_scenario(session, collection_uuid, user_id=None)
        stats["twin_asis_scenario_id"] = str(as_is.id)

        if snapshot_dicts:
            for snapshot_dict in snapshot_dicts:
                snapshot = Snapshot.from_dict(snapshot_dict)
                _, ingest_stats = await ingest_snapshot_into_as_is(
                    session, collection_uuid, snapshot, user_id=None
                )
                stats["twin_nodes_upserted"] += int(ingest_stats.get("nodes_upserted", 0))
                stats["twin_edges_upserted"] += int(ingest_stats.get("edges_upserted", 0))
        else:
            nodes, edges = await seed_scenario_from_knowledge_graph(
                session,
                as_is.id,
                collection_uuid,
                clear_existing=False,
            )
            stats["twin_nodes_upserted"] += int(nodes)
            stats["twin_edges_upserted"] += int(edges)

        stats["twin_metrics_snapshots"] = await refresh_metric_snapshots(session, as_is.id)
        stats["twin_validation_snapshots"] = await refresh_validation_snapshots(
            session, collection_uuid
        )

        # Keep AGE in sync as a mandatory M1 requirement.
        await sync_scenario_to_age(session, as_is.id)
        await session.commit()

    return stats


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
async def task_detect_scip_projects(repo_path: Path) -> list[dict]:
    """Detect projects suitable for SCIP indexing in a repository.

    Returns list of ProjectTarget dicts for serialization.
    """
    from contextmine_core.semantic_snapshot import detect_projects

    projects = detect_projects(repo_path)
    return [p.to_dict() for p in projects]


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

    # Find appropriate backend
    for backend in BACKENDS:
        if backend.can_handle(target):
            try:
                artifact = backend.index(target, cfg)
                if artifact.success:
                    logger.info(
                        "SCIP indexed %s project at %s in %.1fs",
                        target.language.value,
                        target.root_path,
                        artifact.duration_s,
                    )
                    return artifact.to_dict()
                else:
                    logger.warning(
                        "SCIP indexing failed for %s: %s",
                        target.root_path,
                        artifact.error_message,
                    )
                    if not cfg.best_effort:
                        return None
            except Exception as e:
                logger.warning("SCIP backend error for %s: %s", target.root_path, e)
                if not cfg.best_effort:
                    raise
            break

    return None


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


async def get_github_token_for_source(source_id: str, collection_id: str) -> str | None:
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
            select(OAuthToken).where(
                OAuthToken.user_id == owner_id,
                OAuthToken.provider == "github",
            )
        )
        token_record = result.scalar_one_or_none()
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
    git_repo = clone_or_pull_repo(
        repo_path, clone_url, branch, token=token, ssh_private_key=deploy_key
    )

    # Get current commit SHA
    new_sha = git_repo.head.commit.hexsha
    old_sha = source.cursor

    # SCIP Polyglot Indexing
    scip_stats = {
        "scip_projects_detected": 0,
        "scip_projects_indexed": 0,
        "scip_symbols": 0,
        "scip_relations": 0,
    }
    snapshot_dicts: list[dict] = []
    await update_progress_artifact(
        progress_id, progress=10, description="Running SCIP polyglot indexing..."
    )  # type: ignore[misc]

    try:
        import tempfile

        # Detect projects
        project_dicts = await task_detect_scip_projects(repo_path)
        scip_stats["scip_projects_detected"] = len(project_dicts)

        if project_dicts:
            # Create temp output directory for SCIP files
            scip_output_dir = Path(tempfile.mkdtemp(prefix="scip_"))

            # Index each project
            for proj_dict in project_dicts:
                artifact_dict = await task_index_scip_project(proj_dict, scip_output_dir)
                if artifact_dict and artifact_dict.get("success"):
                    scip_stats["scip_projects_indexed"] += 1

                    # Parse SCIP snapshot
                    scip_path = artifact_dict.get("scip_path")
                    if scip_path:
                        snapshot_dict = await task_parse_scip_snapshot(scip_path)
                        if snapshot_dict:
                            snapshot_dicts.append(snapshot_dict)
                            scip_stats["scip_symbols"] += len(snapshot_dict.get("symbols", []))
                            scip_stats["scip_relations"] += len(snapshot_dict.get("relations", []))

            logger.info(
                "SCIP indexing complete: %d/%d projects, %d symbols, %d relations",
                scip_stats["scip_projects_indexed"],
                scip_stats["scip_projects_detected"],
                scip_stats["scip_symbols"],
                scip_stats["scip_relations"],
            )
    except Exception as e:
        logger.warning("SCIP indexing failed: %s", e)

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
    total_docs = len(docs_to_chunk)
    if total_docs > 0:
        await update_progress_artifact(  # type: ignore[misc]
            progress_id,
            progress=50,
            description=f"Chunking and embedding {total_docs} documents...",
        )

    for i, (doc_id, content, file_path) in enumerate(docs_to_chunk):
        # Update progress every 5 documents or on last document
        if total_docs > 0 and (i % 5 == 0 or i == total_docs - 1):
            pct = 50 + int((i + 1) / total_docs * 45)  # 50% to 95%
            await update_progress_artifact(  # type: ignore[misc]
                progress_id,
                progress=pct,
                description=f"Processing document {i + 1}/{total_docs}...",
            )

        chunk_stats = await maintain_chunks_for_document(doc_id, content, file_path)
        total_chunks_created += chunk_stats["chunks_created"]
        total_chunks_deleted += chunk_stats["chunks_deleted"]

        # Extract symbols for code files (using tree-sitter)
        import uuid as uuid_module

        async with get_session() as session:
            sym_created, sym_deleted = await maintain_symbols_for_document(
                session, uuid_module.UUID(doc_id)
            )
            total_symbols_created += sym_created
            total_symbols_deleted += sym_deleted
            await session.commit()

        # Embed new/updated chunks (using collection's embedding config)
        embed_stats = await embed_document(doc_id, collection_id_str)
        total_chunks_embedded += embed_stats["chunks_embedded"]
        total_chunks_deduplicated += embed_stats.get("chunks_deduplicated", 0)
        total_tokens_used += embed_stats["tokens_used"]

    # Build digital twin from indexed documents and semantic snapshots
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=96, description="Building digital twin..."
    )
    changed_doc_ids = [doc_id for doc_id, _, _ in docs_to_chunk]
    twin_stats = await build_twin_graph(
        str(source.id),
        collection_id_str,
        snapshot_dicts=snapshot_dicts,
        changed_doc_ids=changed_doc_ids,
    )

    await update_progress_artifact(progress_id, progress=98, description="Saving results...")  # type: ignore[misc]

    async with get_session() as session:
        # Update sync run
        result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
        db_run = result.scalar_one()
        db_run.status = SyncRunStatus.SUCCESS
        db_run.finished_at = datetime.now(UTC)
        db_run.stats = {
            "files_scanned": stats.files_scanned,
            "files_indexed": stats.files_indexed,
            "files_skipped": stats.files_skipped,
            "files_deleted": stats.files_deleted,
            "docs_created": stats.docs_created,
            "docs_updated": stats.docs_updated,
            "docs_deleted": stats.docs_deleted,
            "docs_unchunked_recovered": len(unchunked_docs),
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
            "scip_symbols": scip_stats.get("scip_symbols", 0),
            "scip_relations": scip_stats.get("scip_relations", 0),
            # Twin stats
            "twin_nodes_upserted": twin_stats.get("twin_nodes_upserted", 0),
            "twin_edges_upserted": twin_stats.get("twin_edges_upserted", 0),
            "twin_metrics_snapshots": twin_stats.get("twin_metrics_snapshots", 0),
            "twin_validation_snapshots": twin_stats.get("twin_validation_snapshots", 0),
            "twin_asis_scenario_id": twin_stats.get("twin_asis_scenario_id"),
        }

        await session.commit()

    await update_progress_artifact(progress_id, progress=100, description="Sync complete!")  # type: ignore[misc]

    return stats


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
        # Process each page
        for page in pages:
            # Use URL as URI
            uri = page.url
            title = get_page_title(page)

            # Check if document exists
            result = await session.execute(select(Document).where(Document.uri == uri))
            existing_doc = result.scalar_one_or_none()

            if existing_doc:
                # Update only if content changed (incremental)
                if existing_doc.content_hash != page.content_hash:
                    existing_doc.content_markdown = page.markdown
                    existing_doc.content_hash = page.content_hash
                    existing_doc.title = title
                    existing_doc.updated_at = datetime.now(UTC)
                    # Update HTTP cache headers in meta
                    meta = existing_doc.meta or {}
                    if page.etag:
                        meta["etag"] = page.etag
                    if page.last_modified:
                        meta["last_modified"] = page.last_modified
                    existing_doc.meta = meta
                    stats.docs_updated += 1
                    # Mark for re-chunking
                    docs_to_chunk.append((str(existing_doc.id), page.markdown, None))
                existing_doc.last_seen_at = run_started_at
            else:
                # Create new document with HTTP cache headers in meta
                meta = {"base_url": base_url}
                if page.etag:
                    meta["etag"] = page.etag
                if page.last_modified:
                    meta["last_modified"] = page.last_modified
                new_doc = Document(
                    source_id=source.id,
                    uri=uri,
                    title=title,
                    content_markdown=page.markdown,
                    content_hash=page.content_hash,
                    meta=meta,
                    last_seen_at=run_started_at,
                )
                session.add(new_doc)
                await session.flush()  # Get the ID
                stats.docs_created += 1
                # Mark for chunking
                docs_to_chunk.append((str(new_doc.id), page.markdown, None))

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
    total_docs = len(docs_to_chunk)
    if total_docs > 0:
        await update_progress_artifact(  # type: ignore[misc]
            progress_id,
            progress=50,
            description=f"Chunking and embedding {total_docs} documents...",
        )

    import uuid as uuid_module

    for i, (doc_id, content, file_path) in enumerate(docs_to_chunk):
        # Update progress every 5 documents or on last document
        if total_docs > 0 and (i % 5 == 0 or i == total_docs - 1):
            pct = 50 + int((i + 1) / total_docs * 45)  # 50% to 95%
            await update_progress_artifact(  # type: ignore[misc]
                progress_id,
                progress=pct,
                description=f"Processing document {i + 1}/{total_docs}...",
            )

        chunk_stats = await maintain_chunks_for_document(doc_id, content, file_path)
        total_chunks_created += chunk_stats["chunks_created"]
        total_chunks_deleted += chunk_stats["chunks_deleted"]

        # Extract symbols for code files (using tree-sitter)
        async with get_session() as session:
            sym_created, sym_deleted = await maintain_symbols_for_document(
                session, uuid_module.UUID(doc_id)
            )
            total_symbols_created += sym_created
            total_symbols_deleted += sym_deleted
            await session.commit()

        # Embed new/updated chunks (using collection's embedding config)
        embed_stats = await embed_document(doc_id, collection_id_str)
        total_chunks_embedded += embed_stats["chunks_embedded"]
        total_chunks_deduplicated += embed_stats.get("chunks_deduplicated", 0)
        total_tokens_used += embed_stats["tokens_used"]

    # Build Twin graph from indexed documents
    await update_progress_artifact(  # type: ignore[misc]
        progress_id, progress=96, description="Building digital twin..."
    )
    changed_doc_ids = [doc_id for doc_id, _, _ in docs_to_chunk]
    twin_stats = await build_twin_graph(
        str(source.id),
        collection_id_str,
        snapshot_dicts=[],
        changed_doc_ids=changed_doc_ids,
    )

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
        }

        await session.commit()

    await update_progress_artifact(progress_id, progress=100, description="Sync complete!")  # type: ignore[misc]

    return stats


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

        # Check if there's already a running sync for this source
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
        # Mark run as failed
        async with get_session() as session:
            result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
            db_run = result.scalar_one()
            db_run.status = SyncRunStatus.FAILED
            db_run.finished_at = datetime.now(UTC)
            db_run.error = str(e)

            # Still update source timestamps
            result = await session.execute(select(Source).where(Source.id == source.id))
            db_source = result.scalar_one()
            db_source.last_run_at = datetime.now(UTC)
            db_source.next_run_at = datetime.now(UTC) + timedelta(
                minutes=db_source.schedule_interval_minutes
            )

            await session.commit()
            await session.refresh(db_run)
            return db_run

    # Refresh and return
    async with get_session() as session:
        result = await session.execute(select(SyncRun).where(SyncRun.id == sync_run.id))
        return result.scalar_one()


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
    for source in sources:
        try:
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

    try:
        sync_run = await sync_source(source)
        if sync_run is None:
            return {"source_id": source_id, "skipped": True, "reason": "lock_not_acquired"}

        return {
            "source_id": source_id,
            "sync_run_id": str(sync_run.id),
            "status": sync_run.status.value,
            "stats": sync_run.stats,
        }
    except Exception as e:
        return {"source_id": source_id, "error": str(e)}
