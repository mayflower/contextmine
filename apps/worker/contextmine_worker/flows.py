"""Prefect flows for syncing sources.

Features:
- Native Prefect scheduling (no custom scheduler)
- Automatic retries with exponential backoff
- Concurrency limits via tags (github-api, embedding-api, web-crawl)
- Task result caching for idempotent operations
"""

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
from contextmine_worker.web_sync import (
    DEFAULT_DELAY_MS,
    DEFAULT_MAX_PAGES,
    WebSyncStats,
    get_page_title,
    run_spider_md,
)

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
        }

        await session.commit()

    await update_progress_artifact(progress_id, progress=100, description="Sync complete!")  # type: ignore[misc]

    return stats


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
    base_url = config.get("base_url", source.url)
    max_pages = config.get("max_pages", DEFAULT_MAX_PAGES)
    delay_ms = config.get("delay_ms", DEFAULT_DELAY_MS)

    if not base_url:
        raise ValueError("Web source missing base_url in config")

    # Create progress artifact
    progress_id = await create_progress_artifact(  # type: ignore[misc]
        progress=0.0,
        description=f"Starting crawl of {base_url}...",
    )

    await update_progress_artifact(
        progress_id, progress=5, description=f"Crawling up to {max_pages} pages..."
    )  # type: ignore[misc]

    # Run the spider with rate limiting
    pages = run_spider_md(
        base_url=base_url,
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
        }

        await session.commit()

    await update_progress_artifact(progress_id, progress=100, description="Sync complete!")  # type: ignore[misc]

    return stats


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
