"""Initialize Prefect concurrency limits and work pools.

Run this once at startup to configure Prefect infrastructure.
"""

import asyncio
import logging

from prefect.client.orchestration import get_client

logger = logging.getLogger(__name__)

# Concurrency limits for external API calls
CONCURRENCY_LIMITS = {
    "github-api": 5,  # Max concurrent GitHub API calls
    "embedding-api": 10,  # Max concurrent embedding API calls
    "web-crawl": 3,  # Max concurrent web crawlers
    "db-heavy": 20,  # Max concurrent heavy DB operations
}


async def init_concurrency_limits() -> None:
    """Create or update concurrency limits."""
    async with get_client() as client:
        for tag, limit in CONCURRENCY_LIMITS.items():
            try:
                # Try to get existing limit
                existing = await client.read_concurrency_limit_by_tag(tag)
                if existing and existing.concurrency_limit != limit:
                    # Update existing limit
                    await client.delete_concurrency_limit_by_tag(tag)
                    await client.create_concurrency_limit(tag=tag, concurrency_limit=limit)
                    logger.info(f"Updated concurrency limit: {tag}={limit}")
                elif existing:
                    logger.info(f"Concurrency limit exists: {tag}={existing.concurrency_limit}")
            except Exception:
                # Create new limit
                try:
                    await client.create_concurrency_limit(tag=tag, concurrency_limit=limit)
                    logger.info(f"Created concurrency limit: {tag}={limit}")
                except Exception as e:
                    logger.warning(f"Could not create concurrency limit {tag}: {e}")


async def init_prefect() -> None:
    """Initialize all Prefect infrastructure."""
    logger.info("Initializing Prefect infrastructure...")
    await init_concurrency_limits()
    logger.info("Prefect initialization complete")


def main() -> None:
    """Entry point for manual initialization."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_prefect())


if __name__ == "__main__":
    main()
