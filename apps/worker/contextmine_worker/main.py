"""Main entry point for the ContextMine worker."""

import asyncio
import threading
from datetime import UTC, datetime

from prefect import serve
from prefect.client.orchestration import get_client

from contextmine_worker.flows import sync_due_sources, sync_single_source


async def check_and_trigger_due_sources() -> None:
    """Check for due sources and trigger syncs on-demand.

    This runs in a loop but only creates flow runs when there are
    actually sources that need syncing, avoiding empty scheduled runs.
    """
    # Wait for deployments to be ready
    await asyncio.sleep(10)

    while True:
        try:
            from contextmine_core import Source, get_session
            from sqlalchemy import select

            # Check if any sources are due
            async with get_session() as session:
                now = datetime.now(UTC)
                stmt = select(Source).where(
                    Source.enabled == True,  # noqa: E712
                    Source.next_run_at <= now,
                )
                result = await session.execute(stmt)
                due_sources = list(result.scalars().all())

            if due_sources:
                # Trigger a flow run for each due source
                async with get_client() as client:
                    for source in due_sources:
                        try:
                            # Find the deployment (format: flow_name/deployment_name)
                            deployment = await client.read_deployment_by_name(
                                "sync_single_source/sync-single-source-deployment"
                            )
                            # Create a flow run with the source_id parameter
                            await client.create_flow_run_from_deployment(
                                deployment_id=deployment.id,
                                parameters={
                                    "source_id": str(source.id),
                                    "source_url": source.url,
                                },
                            )
                        except Exception as e:
                            print(f"Error triggering sync for source {source.id}: {e}")

        except Exception as e:
            print(f"Error checking for due sources: {e}")

        # Wait before checking again
        await asyncio.sleep(60)


def run_scheduler_in_thread() -> None:
    """Run the scheduler loop in a background thread."""
    asyncio.run(check_and_trigger_due_sources())


def main() -> None:
    """Start the worker with on-demand flow triggering."""
    # Create deployments without interval scheduling
    # The sync_due_sources flow is kept for backward compatibility / manual triggers
    due_sources_deployment = sync_due_sources.to_deployment(
        name="sync-due-sources-deployment",
        # No interval - only triggered manually or by API
    )

    # Single source sync deployment - triggered on-demand
    single_source_deployment = sync_single_source.to_deployment(
        name="sync-single-source-deployment",
        # No interval - triggered by the scheduler loop or API
    )

    # Start the scheduler loop in a daemon thread
    scheduler_thread = threading.Thread(target=run_scheduler_in_thread, daemon=True)
    scheduler_thread.start()

    # Serve deployments (this blocks) - type: ignore for Prefect's complex types
    serve(
        due_sources_deployment,  # type: ignore[arg-type]
        single_source_deployment,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
