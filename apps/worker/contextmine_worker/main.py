"""Main entry point for the ContextMine worker.

Uses a simple asyncio scheduler instead of Prefect's subprocess-based serve().
This avoids OOM issues from zombie subprocesses that don't terminate properly.
"""

import asyncio
import logging
import signal
from datetime import timedelta

from contextmine_core.telemetry import init_telemetry

from contextmine_worker.flows import sync_due_sources
from contextmine_worker.init_prefect import init_prefect

logger = logging.getLogger(__name__)

# Scheduler interval
SCHEDULER_INTERVAL = timedelta(minutes=1)


async def scheduler_loop() -> None:
    """Run the scheduler loop that checks for due sources.

    This replaces Prefect's subprocess-based serve() to avoid OOM issues
    from zombie processes that don't terminate due to OTEL/asyncio cleanup issues.

    Flows are executed directly in the main process.
    """
    logger.info("Starting scheduler loop (interval: %s)", SCHEDULER_INTERVAL)

    while True:
        try:
            logger.info("Checking for due sources...")
            # Call the flow function directly (bypasses subprocess isolation)
            # This runs the flow in the current process
            result = await sync_due_sources.fn()
            logger.info("Scheduler run complete: %s", result)
        except Exception as e:
            logger.exception("Scheduler error: %s", e)

        # Wait for next interval
        await asyncio.sleep(SCHEDULER_INTERVAL.total_seconds())


async def run_worker() -> None:
    """Run the worker with graceful shutdown support."""
    # Create the scheduler task
    scheduler_task = asyncio.create_task(scheduler_loop())

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler() -> None:
        logger.info("Received shutdown signal, stopping...")
        stop_event.set()
        scheduler_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Wait for either the scheduler to complete (shouldn't happen)
        # or the stop event to be set
        await asyncio.gather(scheduler_task, stop_event.wait(), return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Worker cancelled")
    finally:
        logger.info("Worker shutdown complete")


def main() -> None:
    """Start the worker with a simple scheduler.

    This worker:
    - Runs sync_due_sources every minute to check for due sources
    - Executes flows directly in the main process (no subprocess isolation)
    - Properly handles shutdown signals

    Note: sync_single_source is still available via Prefect API for on-demand syncs.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize telemetry FIRST (before any other setup)
    init_telemetry(service_suffix="-worker")

    # Initialize Prefect infrastructure (concurrency limits, etc.)
    logger.info("Initializing Prefect infrastructure...")
    asyncio.run(init_prefect())

    # Run the scheduler
    logger.info("Starting ContextMine worker...")
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
