"""LLM-based file triage for extraction.

Instead of hardcoded path patterns, we ask the LLM to identify which files
are likely to contain specific types of content (jobs, schemas, etc.).

This approach:
1. Sends file listing to LLM (cheap - just paths and small snippets)
2. LLM identifies relevant files
3. Detailed extraction runs only on identified files
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class FileTriageResult(BaseModel):
    """Result of file triage - which files to analyze."""

    relevant_files: list[str] = Field(
        default_factory=list,
        description="List of file paths that should be analyzed",
    )
    reasoning: str = Field(
        description="Brief explanation of why these files were selected",
    )


# Triage prompt for job/workflow files
JOBS_TRIAGE_PROMPT = """You are analyzing a codebase to identify files that likely contain job, task, or workflow definitions.

Job definitions include:
- CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins, CircleCI, Azure Pipelines, etc.)
- Workflow engines (Airflow DAGs, Prefect flows, Dagster jobs, Luigi tasks, Temporal workflows)
- Schedulers (Kubernetes CronJobs, systemd timers, cron configurations)
- Task queues (Celery tasks, RQ jobs, Dramatiq actors)
- Serverless functions (Lambda, Cloud Functions, Azure Functions)
- Any scheduled or triggered automation

From the file list below, identify which files are likely to contain job/workflow definitions.
Consider file paths, names, and the code snippets provided.

Files in this codebase:
{file_listing}

Return ONLY the file paths that likely contain job definitions. Be inclusive - if uncertain, include the file."""


# Triage prompt for schema/database files
SCHEMA_TRIAGE_PROMPT = """You are analyzing a codebase to identify files that likely contain database schema definitions.

Schema definitions include:
- Migration files (Alembic, Django, Rails, Knex, Flyway, Liquibase, Prisma Migrate)
- ORM models (SQLAlchemy, Django models, TypeORM entities, Sequelize, ActiveRecord, Hibernate)
- Schema files (Prisma schema, SQL DDL, GraphQL with database types)
- Any file that defines database tables, columns, or relationships

From the file list below, identify which files are likely to contain schema definitions.
Consider file paths, names, and the code snippets provided.

Files in this codebase:
{file_listing}

Return ONLY the file paths that likely contain schema definitions. Be inclusive - if uncertain, include the file."""

TRIAGE_SYSTEM_PROMPT = "You are a precise codebase analyst. Return only structured results grounded in the provided file list."


def _format_file_listing(files: list[tuple[str, str]], max_snippet_lines: int = 5) -> str:
    """Format file listing for triage prompt.

    Args:
        files: List of (file_path, content) tuples
        max_snippet_lines: Maximum lines to include from each file

    Returns:
        Formatted string for the prompt
    """
    lines = []
    for file_path, content in files:
        # Get first few lines as snippet
        snippet_lines = content.split("\n")[:max_snippet_lines]
        snippet = "\n".join(f"  {line}" for line in snippet_lines)
        if len(content.split("\n")) > max_snippet_lines:
            snippet += "\n  ..."

        lines.append(f"--- {file_path} ---")
        lines.append(snippet)
        lines.append("")

    return "\n".join(lines)


async def triage_files_for_jobs(
    files: list[tuple[str, str]],
    provider: LLMProvider,
) -> list[str]:
    """Identify which files likely contain job definitions.

    Args:
        files: List of (file_path, content) tuples
        provider: LLM provider for triage

    Returns:
        List of file paths to analyze for jobs
    """
    if not files:
        return []

    # Skip triage for small file sets - just analyze all
    if len(files) <= 10:
        return [f[0] for f in files]

    try:
        file_listing = _format_file_listing(files)
        prompt = JOBS_TRIAGE_PROMPT.format(file_listing=file_listing)

        result = await provider.generate_structured(
            system=TRIAGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_schema=FileTriageResult,
            temperature=0.0,
            max_tokens=1500,
        )

        logger.debug(
            "Jobs triage: %d/%d files selected - %s",
            len(result.relevant_files),
            len(files),
            result.reasoning,
        )

        # Validate returned paths exist in input
        valid_paths = {f[0] for f in files}
        return [p for p in result.relevant_files if p in valid_paths]

    except Exception as e:
        logger.warning("Jobs triage failed, falling back to all files: %s", e)
        return [f[0] for f in files]


async def triage_files_for_schema(
    files: list[tuple[str, str]],
    provider: LLMProvider,
) -> list[str]:
    """Identify which files likely contain schema definitions.

    Args:
        files: List of (file_path, content) tuples
        provider: LLM provider for triage

    Returns:
        List of file paths to analyze for schemas
    """
    if not files:
        return []

    # Skip triage for small file sets - just analyze all
    if len(files) <= 10:
        return [f[0] for f in files]

    try:
        file_listing = _format_file_listing(files)
        prompt = SCHEMA_TRIAGE_PROMPT.format(file_listing=file_listing)

        result = await provider.generate_structured(
            system=TRIAGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_schema=FileTriageResult,
            temperature=0.0,
            max_tokens=1500,
        )

        logger.debug(
            "Schema triage: %d/%d files selected - %s",
            len(result.relevant_files),
            len(files),
            result.reasoning,
        )

        # Validate returned paths exist in input
        valid_paths = {f[0] for f in files}
        return [p for p in result.relevant_files if p in valid_paths]

    except Exception as e:
        logger.warning("Schema triage failed, falling back to all files: %s", e)
        return [f[0] for f in files]


async def triage_files(
    files: list[tuple[str, str]],
    provider: LLMProvider,
    triage_type: str,
) -> list[str]:
    """Generic triage function.

    Args:
        files: List of (file_path, content) tuples
        provider: LLM provider for triage
        triage_type: Type of content to look for ("jobs" or "schema")

    Returns:
        List of file paths to analyze
    """
    if triage_type == "jobs":
        return await triage_files_for_jobs(files, provider)
    elif triage_type == "schema":
        return await triage_files_for_schema(files, provider)
    else:
        raise ValueError(f"Unknown triage type: {triage_type}")
