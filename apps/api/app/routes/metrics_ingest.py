"""Coverage ingest routes for CI-pushed report uploads."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
from contextmine_core import (
    Collection,
    CollectionMember,
    CollectionVisibility,
    CoverageIngestJob,
    CoverageIngestReport,
    Source,
    SourceIngestToken,
    SourceType,
    get_settings,
)
from contextmine_core import get_session as get_db_session
from contextmine_core.metrics import detect_coverage_protocol
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel
from slowapi.util import get_remote_address
from sqlalchemy import select

from app.middleware import get_session
from app.rate_limit import limiter

router = APIRouter(tags=["metrics-ingest"])

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class CoverageIngestJobResponse(BaseModel):
    """Coverage ingest job status response."""

    id: str
    source_id: str
    collection_id: str
    scenario_id: str | None
    commit_sha: str
    branch: str | None
    provider: str
    workflow_run_id: str | None
    status: str
    error_code: str | None
    error_detail: str | None
    stats: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    reports: list[dict[str, Any]]


def _user_id_or_401(request: Request) -> uuid.UUID:
    session = get_session(request)
    user_id = session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uuid.UUID(user_id)


def _hash_ingest_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _ingest_source_key(request: Request) -> str:
    source_id = request.path_params.get("source_id", "unknown")
    return f"{get_remote_address(request)}:{source_id}"


async def _ensure_collection_member(db, collection_id: uuid.UUID, user_id: uuid.UUID) -> None:
    collection = (
        await db.execute(select(Collection).where(Collection.id == collection_id))
    ).scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    if collection.visibility == CollectionVisibility.GLOBAL:
        return
    if collection.owner_user_id == user_id:
        return

    membership = (
        await db.execute(
            select(CollectionMember).where(
                CollectionMember.collection_id == collection_id,
                CollectionMember.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if not membership:
        raise HTTPException(status_code=403, detail="Access denied")


async def _trigger_prefect_ingest_flow(job_id: uuid.UUID) -> dict[str, Any]:
    settings = get_settings()
    prefect_url = settings.prefect_api_url.rstrip("/")
    flow_name = settings.coverage_ingest_prefect_flow_name

    async with httpx.AsyncClient(timeout=10.0) as client:
        flow_response = await client.post(f"{prefect_url}/flows/filter", json={"limit": 200})
        flow_response.raise_for_status()
        flows = flow_response.json()
        flow = next((item for item in flows if item.get("name") == flow_name), None)
        if not flow:
            raise RuntimeError(f"Prefect flow '{flow_name}' not found")

        run_response = await client.post(
            f"{prefect_url}/flow_runs/",
            json={
                "flow_id": flow["id"],
                "name": f"coverage-ingest-{job_id}",
                "parameters": {"job_id": str(job_id)},
            },
        )
        run_response.raise_for_status()
        return run_response.json()


def _detect_protocol_from_bytes(filename: str, payload: bytes) -> str | None:
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(payload)
        tmp.flush()
        return detect_coverage_protocol(Path(tmp.name))


def _serialize_job(
    job: CoverageIngestJob, reports: list[CoverageIngestReport]
) -> CoverageIngestJobResponse:
    return CoverageIngestJobResponse(
        id=str(job.id),
        source_id=str(job.source_id),
        collection_id=str(job.collection_id),
        scenario_id=str(job.scenario_id) if job.scenario_id else None,
        commit_sha=job.commit_sha,
        branch=job.branch,
        provider=job.provider,
        workflow_run_id=job.workflow_run_id,
        status=job.status,
        error_code=job.error_code,
        error_detail=job.error_detail,
        stats=dict(job.stats or {}),
        created_at=job.created_at,
        updated_at=job.updated_at,
        reports=[
            {
                "id": str(report.id),
                "filename": report.filename,
                "protocol_detected": report.protocol_detected,
                "diagnostics": dict(report.diagnostics or {}),
                "created_at": report.created_at,
            }
            for report in reports
        ],
    )


@router.post("/sources/{source_id}/metrics/coverage-ingest")
@limiter.limit("30/minute")
@limiter.limit("10/minute", key_func=_ingest_source_key)
async def upload_coverage_ingest(
    request: Request,
    source_id: str,
    commit_sha: Annotated[str, Form(...)],
    reports: Annotated[list[UploadFile], File(...)],
    branch: str | None = Form(default=None),
    workflow_run_id: str | None = Form(default=None),
    provider: str = Form(default="github_actions"),
    manifest: str | None = Form(default=None),
    x_contextmine_ingest_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Upload CI coverage reports and enqueue async ingest processing."""
    del request

    if x_contextmine_ingest_token is None:
        raise HTTPException(status_code=401, detail="INGEST_AUTH_INVALID")
    if not SHA1_RE.match(commit_sha.lower()):
        raise HTTPException(status_code=400, detail="commit_sha must be 40-char lowercase hex")

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source_id") from e

    if not reports:
        raise HTTPException(status_code=400, detail="At least one report file is required")

    settings = get_settings()
    max_payload_bytes = settings.coverage_ingest_max_payload_mb * 1024 * 1024

    manifest_payload: dict[str, Any] | None = None
    if manifest:
        try:
            loaded = json.loads(manifest)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid manifest JSON: {e}") from e
        if not isinstance(loaded, dict):
            raise HTTPException(status_code=400, detail="manifest must be a JSON object")
        manifest_payload = loaded

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        if source.type != SourceType.GITHUB:
            raise HTTPException(
                status_code=400, detail="Coverage ingest is only supported for GitHub sources"
            )

        token_row = (
            await db.execute(
                select(SourceIngestToken).where(SourceIngestToken.source_id == source.id)
            )
        ).scalar_one_or_none()
        if not token_row:
            raise HTTPException(status_code=401, detail="INGEST_AUTH_INVALID")

        digest = _hash_ingest_token(x_contextmine_ingest_token)
        if not hmac.compare_digest(token_row.token_hash, digest):
            raise HTTPException(status_code=401, detail="INGEST_AUTH_INVALID")

        if not source.cursor or source.cursor != commit_sha:
            raise HTTPException(
                status_code=409,
                detail=f"INGEST_SHA_MISMATCH: source cursor is {source.cursor}, payload is {commit_sha}",
            )

        total_bytes = 0
        report_payloads: list[tuple[UploadFile, bytes]] = []
        for upload in reports:
            payload = await upload.read()
            total_bytes += len(payload)
            if total_bytes > max_payload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "INGEST_PAYLOAD_TOO_LARGE: total report payload exceeds "
                        f"{settings.coverage_ingest_max_payload_mb} MB"
                    ),
                )
            report_payloads.append((upload, payload))

        if not report_payloads:
            raise HTTPException(status_code=400, detail="No report payload bytes received")

        job = CoverageIngestJob(
            id=uuid.uuid4(),
            source_id=source.id,
            collection_id=source.collection_id,
            scenario_id=None,
            commit_sha=commit_sha,
            branch=branch,
            provider=provider,
            workflow_run_id=workflow_run_id,
            status="queued",
            stats={
                "reports_total": len(report_payloads),
                "payload_bytes": total_bytes,
                "manifest": manifest_payload or {},
            },
        )
        db.add(job)
        await db.flush()

        for upload, payload in report_payloads:
            filename = upload.filename or "report"
            protocol = _detect_protocol_from_bytes(filename, payload)
            report = CoverageIngestReport(
                id=uuid.uuid4(),
                job_id=job.id,
                filename=filename,
                protocol_detected=protocol,
                report_bytes=payload,
                diagnostics={
                    "content_type": upload.content_type,
                    "size_bytes": len(payload),
                    "detected_at_ingest": protocol is not None,
                },
            )
            db.add(report)

        token_row.last_used_at = datetime.now(UTC)
        await db.commit()

    try:
        flow_run = await _trigger_prefect_ingest_flow(job.id)
    except Exception as e:  # noqa: BLE001
        async with get_db_session() as db:
            job_row = (
                await db.execute(select(CoverageIngestJob).where(CoverageIngestJob.id == job.id))
            ).scalar_one_or_none()
            if job_row:
                job_row.status = "failed"
                job_row.error_code = "INGEST_APPLY_FAILED"
                job_row.error_detail = f"Failed to trigger Prefect ingest flow: {e}"
                await db.commit()
        raise HTTPException(status_code=500, detail=f"INGEST_APPLY_FAILED: {e}") from e

    return {
        "job_id": str(job.id),
        "status": "queued",
        "flow_run_id": flow_run.get("id"),
    }


@router.get(
    "/sources/{source_id}/metrics/coverage-ingest/{job_id}",
    response_model=CoverageIngestJobResponse,
)
async def get_coverage_ingest_job(
    request: Request, source_id: str, job_id: str
) -> CoverageIngestJobResponse:
    """Get coverage ingest job status and diagnostics."""
    user_id = _user_id_or_401(request)

    try:
        src_uuid = uuid.UUID(source_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid source_id") from e

    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid job_id") from e

    async with get_db_session() as db:
        source = (
            await db.execute(select(Source).where(Source.id == src_uuid))
        ).scalar_one_or_none()
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        await _ensure_collection_member(db, source.collection_id, user_id)

        job = (
            await db.execute(
                select(CoverageIngestJob).where(
                    CoverageIngestJob.id == job_uuid,
                    CoverageIngestJob.source_id == source.id,
                )
            )
        ).scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Coverage ingest job not found")

        reports = (
            (
                await db.execute(
                    select(CoverageIngestReport).where(CoverageIngestReport.job_id == job.id)
                )
            )
            .scalars()
            .all()
        )

        return _serialize_job(job, reports)
