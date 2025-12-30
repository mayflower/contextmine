"""Job definition extractor.

Parses job/workflow definition files to extract:
- GitHub Actions workflows
- Kubernetes CronJob manifests
- Prefect deployment specs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class JobKind(str, Enum):
    """Types of jobs that can be extracted."""

    GITHUB_WORKFLOW = "github_workflow"
    GITHUB_JOB = "github_job"
    K8S_CRONJOB = "k8s_cronjob"
    PREFECT_DEPLOYMENT = "prefect_deployment"


@dataclass
class JobTriggerDef:
    """Extracted job trigger definition."""

    trigger_type: str  # schedule, push, pull_request, workflow_dispatch, etc.
    cron: str | None = None
    branches: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


@dataclass
class JobStepDef:
    """Extracted job step (GitHub Actions)."""

    name: str | None
    uses: str | None = None
    run: str | None = None


@dataclass
class JobDef:
    """Extracted job definition."""

    name: str
    kind: JobKind
    file_path: str
    triggers: list[JobTriggerDef] = field(default_factory=list)
    steps: list[JobStepDef] = field(default_factory=list)
    schedule: str | None = None  # Cron expression
    container_image: str | None = None
    runs_on: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobsExtraction:
    """Result of parsing job definition files."""

    file_path: str
    jobs: list[JobDef] = field(default_factory=list)


def extract_jobs(file_path: str, content: str) -> JobsExtraction:
    """Extract job definitions from a file based on its path and content.

    Automatically detects the job type based on:
    - .github/workflows/*.yml -> GitHub Actions
    - Files containing kind: CronJob -> Kubernetes CronJob
    - prefect.yaml or deployment specs -> Prefect

    Args:
        file_path: Path to the job definition file
        content: File content

    Returns:
        JobsExtraction with extracted job definitions
    """
    result = JobsExtraction(file_path=file_path)

    # Detect file type based on path and content
    if ".github/workflows" in file_path and file_path.endswith((".yml", ".yaml")):
        _extract_github_workflow(file_path, content, result)
    elif file_path.endswith((".yml", ".yaml")):
        # Try to parse and detect type
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                if data.get("kind") == "CronJob":
                    _extract_k8s_cronjob(file_path, data, result)
                elif "deployments" in data or data.get("name") and "flow" in data:
                    _extract_prefect_deployment(file_path, data, result)
        except yaml.YAMLError:
            pass

    return result


def _extract_github_workflow(file_path: str, content: str, result: JobsExtraction) -> None:
    """Extract GitHub Actions workflow jobs."""
    try:
        workflow = yaml.safe_load(content)
        if not isinstance(workflow, dict):
            return

        workflow_name = workflow.get("name", Path(file_path).stem)

        # Extract triggers
        # Note: YAML 1.1 parses bare 'on' as boolean True, so check both
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

                if isinstance(config, dict):
                    if "cron" in config:
                        # schedule trigger
                        trigger.cron = config["cron"]
                    if "branches" in config:
                        trigger.branches = config["branches"]
                    if "paths" in config:
                        trigger.paths = config["paths"]

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
                kind=JobKind.GITHUB_JOB,
                file_path=file_path,
                triggers=list(triggers),  # Copy to avoid shared reference
                runs_on=job_config.get("runs-on"),
                meta={"workflow": workflow_name, "job_id": job_id},
            )

            # Extract steps
            steps = job_config.get("steps", [])
            for step in steps:
                if isinstance(step, dict):
                    job.steps.append(
                        JobStepDef(
                            name=step.get("name"),
                            uses=step.get("uses"),
                            run=step.get("run"),
                        )
                    )

            result.jobs.append(job)

    except yaml.YAMLError as e:
        logger.warning("Failed to parse GitHub workflow %s: %s", file_path, e)


def _extract_k8s_cronjob(file_path: str, data: dict[str, Any], result: JobsExtraction) -> None:
    """Extract Kubernetes CronJob definition."""
    metadata = data.get("metadata", {})
    spec = data.get("spec", {})

    job = JobDef(
        name=metadata.get("name", "unnamed"),
        kind=JobKind.K8S_CRONJOB,
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


def _extract_prefect_deployment(
    file_path: str, data: dict[str, Any], result: JobsExtraction
) -> None:
    """Extract Prefect deployment definitions."""
    deployments = data.get("deployments", [])

    if not deployments and data.get("name"):
        # Single deployment format
        deployments = [data]

    for deployment in deployments:
        if not isinstance(deployment, dict):
            continue

        job = JobDef(
            name=deployment.get("name", "unnamed"),
            kind=JobKind.PREFECT_DEPLOYMENT,
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


def extract_jobs_from_file(file_path: Path | str) -> JobsExtraction:
    """Extract from a job definition file on disk.

    Args:
        file_path: Path to the job definition file

    Returns:
        JobsExtraction with extracted definitions
    """
    path = Path(file_path)
    if not path.exists():
        return JobsExtraction(file_path=str(file_path))

    content = path.read_text(encoding="utf-8", errors="replace")
    return extract_jobs(str(file_path), content)
