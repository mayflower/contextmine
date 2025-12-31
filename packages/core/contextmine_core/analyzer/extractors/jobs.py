"""Job/workflow definition extractor using LLM.

This module extracts job and workflow definitions from ANY orchestration framework:
- CI/CD: GitHub Actions, GitLab CI, Jenkins, CircleCI, Azure Pipelines, Bitbucket Pipelines
- Workflow engines: Airflow, Prefect, Dagster, Luigi, Temporal, AWS Step Functions
- Schedulers: Kubernetes CronJobs, systemd timers, cron files
- Any other job/task/workflow definition format

Uses LLM for semantic analysis - no framework-specific hardcoding.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.llm.base import LLMProvider
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================================
# Data models for extraction output
# ============================================================================


@dataclass
class JobTriggerDef:
    """Extracted job trigger definition."""

    trigger_type: str  # schedule, push, pull_request, webhook, manual, dependency, etc.
    cron: str | None = None
    description: str | None = None  # Human-readable trigger description


@dataclass
class JobStepDef:
    """Extracted job step."""

    name: str | None
    description: str | None = None
    action: str | None = None  # What this step does


@dataclass
class JobDef:
    """Extracted job definition."""

    name: str
    framework: str  # Detected framework (e.g., "github_actions", "airflow", "k8s_cronjob")
    file_path: str
    description: str | None = None
    triggers: list[JobTriggerDef] = field(default_factory=list)
    steps: list[JobStepDef] = field(default_factory=list)
    schedule: str | None = None  # Cron expression if applicable
    container_image: str | None = None
    dependencies: list[str] = field(default_factory=list)  # Other jobs this depends on
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobsExtraction:
    """Result of extracting job definitions from a file."""

    file_path: str
    jobs: list[JobDef] = field(default_factory=list)


# ============================================================================
# Pydantic models for LLM structured output
# ============================================================================


class JobTriggerOutput(BaseModel):
    """A trigger that starts a job."""

    trigger_type: str = Field(
        description="Type of trigger (schedule, push, webhook, manual, dependency, event, etc.)"
    )
    cron: str | None = Field(
        default=None, description="Cron expression if this is a scheduled trigger"
    )
    description: str | None = Field(
        default=None, description="Human-readable description of when this triggers"
    )


class JobStepOutput(BaseModel):
    """A step within a job."""

    name: str | None = Field(default=None, description="Name of the step")
    description: str | None = Field(default=None, description="What this step does")
    action: str | None = Field(
        default=None, description="The action being performed (e.g., 'run tests', 'deploy to prod')"
    )


class JobOutput(BaseModel):
    """A job or task definition."""

    name: str = Field(description="Name of the job/task/workflow")
    framework: str = Field(
        description="Framework or tool (e.g., 'github_actions', 'airflow', 'prefect', 'k8s_cronjob', 'jenkins', 'gitlab_ci')"
    )
    description: str | None = Field(default=None, description="What this job does")
    schedule: str | None = Field(default=None, description="Cron schedule if applicable")
    container_image: str | None = Field(default=None, description="Container image if specified")
    triggers: list[JobTriggerOutput] = Field(
        default_factory=list, description="What triggers this job"
    )
    steps: list[JobStepOutput] = Field(default_factory=list, description="Steps in the job")
    dependencies: list[str] = Field(
        default_factory=list, description="Names of other jobs this depends on"
    )


class JobsExtractionOutput(BaseModel):
    """LLM output for job extraction."""

    jobs: list[JobOutput] = Field(default_factory=list, description="Extracted job definitions")


# ============================================================================
# LLM-based extraction
# ============================================================================


JOBS_EXTRACTION_PROMPT = """Analyze this configuration file and extract any job, task, or workflow definitions.

Look for definitions from ANY framework including but not limited to:
- CI/CD: GitHub Actions, GitLab CI, Jenkins, CircleCI, Azure Pipelines, Bitbucket Pipelines, Travis CI
- Workflow engines: Apache Airflow (DAGs), Prefect (flows), Dagster (jobs), Luigi (tasks), Temporal (workflows), AWS Step Functions
- Schedulers: Kubernetes CronJobs, systemd timers, crontab entries
- Task queues: Celery tasks, RQ jobs, Dramatiq actors
- Serverless: AWS Lambda, Google Cloud Functions, Azure Functions

For each job/task/workflow found, extract:
1. Name and description
2. Framework/tool being used
3. Triggers (schedules, events, webhooks, dependencies)
4. Steps or tasks within the job
5. Container images if specified
6. Dependencies on other jobs

File: {file_path}

Content:
```
{content}
```

Return an empty jobs list if no job/workflow definitions are found."""


def _is_config_file(file_path: str) -> bool:
    """Check if a file is a config file that could contain jobs.

    This is a basic filter to skip obvious non-config files (binaries, images, etc.)
    The actual relevance check is done by LLM triage.
    """
    path_lower = file_path.lower()

    # Skip binary/media files
    skip_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".pyc",
        ".pyo",
        ".class",
        ".lock",
        ".sum",
    )
    if path_lower.endswith(skip_extensions):
        return False

    # Accept config files and code files that might define jobs
    config_extensions = (
        ".yml",
        ".yaml",
        ".json",
        ".toml",
        ".py",
        ".js",
        ".ts",
        ".rb",
        ".java",
        ".go",
        ".xml",
        ".hcl",
        ".tf",
    )
    return path_lower.endswith(config_extensions) or "jenkinsfile" in path_lower


async def extract_jobs_from_files(
    files: list[tuple[str, str]],
    provider: LLMProvider,
) -> list[JobsExtraction]:
    """Extract job definitions from multiple files using LLM triage.

    This function:
    1. Uses LLM to identify which files likely contain jobs (triage)
    2. Extracts jobs only from identified files

    Args:
        files: List of (file_path, content) tuples
        provider: LLM provider for triage and extraction

    Returns:
        List of JobsExtraction results
    """
    from contextmine_core.analyzer.extractors.triage import triage_files_for_jobs

    # Filter to config files only
    config_files = [(p, c) for p, c in files if _is_config_file(p)]

    if not config_files:
        return []

    # Use LLM triage to identify relevant files
    relevant_paths = await triage_files_for_jobs(config_files, provider)
    relevant_files = {p: c for p, c in config_files if p in relevant_paths}

    logger.info(
        "Jobs triage: %d/%d files selected for extraction",
        len(relevant_files),
        len(config_files),
    )

    # Extract from relevant files
    results = []
    for file_path, content in relevant_files.items():
        result = await _extract_jobs_from_single_file(file_path, content, provider)
        if result.jobs:
            results.append(result)

    return results


async def extract_jobs_from_file(
    file_path: str,
    content: str,
    provider: LLMProvider,
) -> JobsExtraction:
    """Extract job definitions from a single file using LLM.

    Use this for single-file extraction when you know the file contains jobs.
    For batch extraction with automatic file selection, use extract_jobs_from_files().

    Args:
        file_path: Path to the file
        content: File content
        provider: LLM provider for analysis

    Returns:
        JobsExtraction with extracted job definitions
    """
    return await _extract_jobs_from_single_file(file_path, content, provider)


async def _extract_jobs_from_single_file(
    file_path: str,
    content: str,
    provider: LLMProvider,
) -> JobsExtraction:
    """Internal: Extract job definitions from a single file.

    No filtering - assumes caller has already determined this file should be analyzed.
    """
    result = JobsExtraction(file_path=file_path)

    # Skip very large files (likely not config)
    if len(content) > 50000:
        logger.debug("Skipping large file for job extraction: %s", file_path)
        return result

    # Skip binary or non-text content
    if "\x00" in content[:1000]:
        return result

    try:
        prompt = JOBS_EXTRACTION_PROMPT.format(
            file_path=file_path,
            content=content[:30000],  # Truncate very long files
        )

        llm_result = await provider.generate_structured(
            prompt=prompt,
            output_schema=JobsExtractionOutput,
            temperature=0.0,
        )

        # Convert to dataclass format
        for job_output in llm_result.jobs:
            job = JobDef(
                name=job_output.name,
                framework=job_output.framework,
                file_path=file_path,
                description=job_output.description,
                schedule=job_output.schedule,
                container_image=job_output.container_image,
                dependencies=job_output.dependencies,
            )

            for trigger_output in job_output.triggers:
                job.triggers.append(
                    JobTriggerDef(
                        trigger_type=trigger_output.trigger_type,
                        cron=trigger_output.cron,
                        description=trigger_output.description,
                    )
                )

            for step_output in job_output.steps:
                job.steps.append(
                    JobStepDef(
                        name=step_output.name,
                        description=step_output.description,
                        action=step_output.action,
                    )
                )

            result.jobs.append(job)

    except Exception as e:
        logger.warning("Failed to extract jobs from %s: %s", file_path, e)

    return result


# ============================================================================
# Knowledge graph building
# ============================================================================


def get_job_natural_key(job: JobDef) -> str:
    """Generate a stable natural key for a job definition.

    The key is based on the file path, framework, and job name.
    """
    key_parts = [
        job.file_path,
        job.framework,
        job.name,
    ]
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


async def build_jobs_graph(
    session: AsyncSession,
    collection_id: UUID,
    extractions: list[JobsExtraction],
) -> dict:
    """Build knowledge graph nodes and edges from extracted jobs.

    Creates:
    - JOB nodes for each job definition
    - JOB_DEPENDS_ON edges for job dependencies

    Args:
        session: Database session
        collection_id: Collection UUID
        extractions: List of job extractions

    Returns:
        Stats dict with counts
    """
    from contextmine_core.models import (
        KnowledgeEdge,
        KnowledgeEdgeKind,
        KnowledgeEvidence,
        KnowledgeNode,
        KnowledgeNodeEvidence,
        KnowledgeNodeKind,
    )
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stats = {
        "job_nodes_created": 0,
        "edges_created": 0,
        "evidence_created": 0,
    }

    job_node_ids: dict[str, UUID] = {}  # job_name -> node_id

    for extraction in extractions:
        for job in extraction.jobs:
            natural_key = f"job:{get_job_natural_key(job)}"

            # Build metadata
            meta: dict[str, Any] = {
                "framework": job.framework,
                "schedule": job.schedule,
                "container_image": job.container_image,
                "trigger_count": len(job.triggers),
                "step_count": len(job.steps),
            }

            if job.triggers:
                meta["triggers"] = [
                    {
                        "type": t.trigger_type,
                        "cron": t.cron,
                        "description": t.description,
                    }
                    for t in job.triggers
                ]

            stmt = pg_insert(KnowledgeNode).values(
                collection_id=collection_id,
                kind=KnowledgeNodeKind.JOB,
                natural_key=natural_key,
                name=job.name,
                description=job.description,
                meta=meta,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_knowledge_node_natural",
                set_={
                    "name": stmt.excluded.name,
                    "description": stmt.excluded.description,
                    "meta": stmt.excluded.meta,
                },
            ).returning(KnowledgeNode.id)

            result = await session.execute(stmt)
            node_id = result.scalar_one()
            job_node_ids[job.name] = node_id
            stats["job_nodes_created"] += 1

            # Create evidence linking to source file
            evidence = KnowledgeEvidence(
                file_path=job.file_path,
                start_line=1,
                end_line=1,
            )
            session.add(evidence)
            await session.flush()

            # Check if evidence link already exists
            existing_link = await session.execute(
                select(KnowledgeNodeEvidence.evidence_id).where(
                    KnowledgeNodeEvidence.node_id == node_id,
                    KnowledgeNodeEvidence.evidence_id == evidence.id,
                )
            )
            if not existing_link.scalar_one_or_none():
                session.add(
                    KnowledgeNodeEvidence(
                        node_id=node_id,
                        evidence_id=evidence.id,
                    )
                )
                stats["evidence_created"] += 1

    # Create dependency edges
    for extraction in extractions:
        for job in extraction.jobs:
            source_id = job_node_ids.get(job.name)
            if not source_id:
                continue

            for dep_name in job.dependencies:
                target_id = job_node_ids.get(dep_name)
                if not target_id:
                    continue

                # Check if edge already exists
                edge_exists = await session.execute(
                    select(KnowledgeEdge.id).where(
                        KnowledgeEdge.collection_id == collection_id,
                        KnowledgeEdge.source_node_id == source_id,
                        KnowledgeEdge.target_node_id == target_id,
                        KnowledgeEdge.kind == KnowledgeEdgeKind.JOB_DEPENDS_ON,
                    )
                )
                if not edge_exists.scalar_one_or_none():
                    session.add(
                        KnowledgeEdge(
                            collection_id=collection_id,
                            source_node_id=source_id,
                            target_node_id=target_id,
                            kind=KnowledgeEdgeKind.JOB_DEPENDS_ON,
                            meta={},
                        )
                    )
                    stats["edges_created"] += 1

    return stats


# ============================================================================
# Backwards compatibility - sync extraction without LLM
# ============================================================================


class JobKind:
    """Job framework types for backwards compatibility."""

    GITHUB_WORKFLOW = "github_workflow"
    GITHUB_JOB = "github_job"
    K8S_CRONJOB = "k8s_cronjob"
    PREFECT_DEPLOYMENT = "prefect_deployment"


def extract_jobs(file_path: str, content: str) -> JobsExtraction:
    """Extract job definitions using structural parsing (sync, no LLM).

    This is the legacy sync extraction function for backwards compatibility.
    For comprehensive extraction, use extract_jobs_from_file() with an LLM provider.

    Supports:
    - GitHub Actions workflows
    - Kubernetes CronJob manifests
    - Prefect deployment specs

    Args:
        file_path: Path to the job definition file
        content: File content

    Returns:
        JobsExtraction with extracted job definitions
    """
    import yaml

    result = JobsExtraction(file_path=file_path)

    # Detect file type based on path and content
    if ".github/workflows" in file_path and file_path.endswith((".yml", ".yaml")):
        _extract_github_workflow_sync(file_path, content, result)
    elif file_path.endswith((".yml", ".yaml")):
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                if data.get("kind") == "CronJob":
                    _extract_k8s_cronjob_sync(file_path, data, result)
                elif "deployments" in data or data.get("name") and "flow" in data:
                    _extract_prefect_deployment_sync(file_path, data, result)
        except yaml.YAMLError:
            pass

    return result


def _extract_github_workflow_sync(file_path: str, content: str, result: JobsExtraction) -> None:
    """Extract GitHub Actions workflow jobs (sync)."""
    from pathlib import Path

    import yaml

    try:
        workflow = yaml.safe_load(content)
        if not isinstance(workflow, dict):
            return

        workflow_name = workflow.get("name", Path(file_path).stem)

        # Extract triggers
        triggers: list[JobTriggerDef] = []
        on_config = workflow.get("on") or workflow.get(True) or {}

        if isinstance(on_config, str):
            triggers.append(JobTriggerDef(trigger_type=on_config))
        elif isinstance(on_config, list):
            for trigger in on_config:
                triggers.append(JobTriggerDef(trigger_type=trigger))
        elif isinstance(on_config, dict):
            for trigger_type, config in on_config.items():
                trigger = JobTriggerDef(trigger_type=trigger_type)

                if isinstance(config, dict) and "cron" in config:
                    trigger.cron = config["cron"]

                # Handle schedule array
                if trigger_type == "schedule" and isinstance(config, list):
                    for sched in config:
                        if isinstance(sched, dict) and "cron" in sched:
                            t = JobTriggerDef(trigger_type="schedule", cron=sched["cron"])
                            triggers.append(t)
                    continue

                triggers.append(trigger)

        # Extract jobs
        jobs = workflow.get("jobs", {})
        for job_id, job_config in jobs.items():
            if not isinstance(job_config, dict):
                continue

            job = JobDef(
                name=job_config.get("name", job_id),
                framework=JobKind.GITHUB_JOB,
                file_path=file_path,
                triggers=list(triggers),
                meta={"workflow": workflow_name, "job_id": job_id},
            )

            # Extract steps
            steps = job_config.get("steps", [])
            for step in steps:
                if isinstance(step, dict):
                    job.steps.append(
                        JobStepDef(
                            name=step.get("name"),
                            action=step.get("uses") or step.get("run"),
                        )
                    )

            result.jobs.append(job)

    except yaml.YAMLError as e:
        logger.warning("Failed to parse GitHub workflow %s: %s", file_path, e)


def _extract_k8s_cronjob_sync(file_path: str, data: dict[str, Any], result: JobsExtraction) -> None:
    """Extract Kubernetes CronJob definition (sync)."""
    metadata = data.get("metadata", {})
    spec = data.get("spec", {})

    job = JobDef(
        name=metadata.get("name", "unnamed"),
        framework=JobKind.K8S_CRONJOB,
        file_path=file_path,
        schedule=spec.get("schedule"),
        meta={
            "namespace": metadata.get("namespace"),
            "suspend": spec.get("suspend", False),
        },
    )

    # Extract container image
    job_template = spec.get("jobTemplate", {})
    pod_spec = job_template.get("spec", {}).get("template", {}).get("spec", {})
    containers = pod_spec.get("containers", [])
    if containers and isinstance(containers[0], dict):
        job.container_image = containers[0].get("image")

    if job.schedule:
        job.triggers.append(JobTriggerDef(trigger_type="schedule", cron=job.schedule))

    result.jobs.append(job)


def _extract_prefect_deployment_sync(
    file_path: str, data: dict[str, Any], result: JobsExtraction
) -> None:
    """Extract Prefect deployment definitions (sync)."""
    deployments = data.get("deployments", [])

    if not deployments and data.get("name"):
        deployments = [data]

    for deployment in deployments:
        if not isinstance(deployment, dict):
            continue

        job = JobDef(
            name=deployment.get("name", "unnamed"),
            framework=JobKind.PREFECT_DEPLOYMENT,
            file_path=file_path,
            meta={
                "flow_name": deployment.get("flow_name"),
                "entrypoint": deployment.get("entrypoint"),
                "work_pool": deployment.get("work_pool", {}).get("name")
                if isinstance(deployment.get("work_pool"), dict)
                else None,
            },
        )

        # Extract schedule
        schedules = deployment.get("schedules", [])
        for sched in schedules:
            if isinstance(sched, dict) and "cron" in sched:
                job.schedule = sched["cron"]
                job.triggers.append(JobTriggerDef(trigger_type="schedule", cron=sched["cron"]))

        # Legacy schedule format
        if not job.schedule and "schedule" in deployment:
            sched = deployment["schedule"]
            if isinstance(sched, dict) and "cron" in sched:
                job.schedule = sched["cron"]
                job.triggers.append(JobTriggerDef(trigger_type="schedule", cron=sched["cron"]))

        result.jobs.append(job)
