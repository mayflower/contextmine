"""Main entry point for the ContextMine worker.

Uses Prefect's native scheduling instead of a custom scheduler loop.
Deployments are configured with intervals and work pools for scalability.
"""

import asyncio
import logging
from datetime import timedelta

from prefect import serve

from contextmine_worker.flows import sync_due_sources, sync_single_source
from contextmine_worker.init_prefect import init_prefect

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the worker with Prefect-native scheduling.

    Deployments:
    - sync-due-sources: Runs every minute to check for due sources
    - sync-single-source: On-demand, triggered by API or scheduler

    Concurrency limits (initialized at startup):
    - github-api: 5 concurrent GitHub API calls
    - embedding-api: 10 concurrent embedding API calls
    - web-crawl: 3 concurrent web crawlers
    - db-heavy: 20 concurrent heavy DB operations
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize Prefect infrastructure (concurrency limits, etc.)
    logger.info("Initializing Prefect infrastructure...")
    asyncio.run(init_prefect())

    # Main scheduler deployment - checks for due sources every minute
    # This replaces the custom check_and_trigger_due_sources() loop
    due_sources_deployment = sync_due_sources.to_deployment(
        name="sync-due-sources-deployment",
        interval=timedelta(minutes=1),
        description="Checks for sources due for syncing and triggers individual syncs",
        tags=["scheduler"],
    )

    # Single source sync deployment - triggered on-demand by scheduler or API
    single_source_deployment = sync_single_source.to_deployment(
        name="sync-single-source-deployment",
        description="Syncs a single source (GitHub or web)",
        tags=["sync"],
    )

    # Serve deployments (this blocks)
    # Prefect handles scheduling, retries, and concurrency automatically
    logger.info("Starting Prefect worker with deployments...")
    serve(
        due_sources_deployment,  # type: ignore[arg-type]
        single_source_deployment,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
