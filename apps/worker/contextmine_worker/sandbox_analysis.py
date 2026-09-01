"""Run repository analysis through the Mayflower Agent Sandbox platform."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import re
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from langsmith.sandbox import AsyncSandboxClient, opaque_secret, proxy_config
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

_RESULT_MANIFEST = "/workspace/result/manifest.json"
_ANALYZER_COMMAND = (
    "/app/.venv/bin/python -m contextmine_worker.sandbox_analyzer "
    "/workspace/request.json /workspace/result"
)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class SandboxAnalysisRequest(BaseModel):
    """The complete, non-secret input passed into one analyzer sandbox."""

    model_config = ConfigDict(extra="forbid")

    source_id: uuid.UUID
    owner: str = Field(min_length=1, max_length=100)
    repository: str = Field(min_length=1, max_length=100)
    branch: str | None = Field(default=None, max_length=255)
    previous_commit: str | None = None
    analyzer_profile: str = Field(default="scip-kg-v1", pattern=r"^[a-z0-9._-]+$")
    scip_languages: str = Field(max_length=128)
    scip_timeout_python: int = Field(ge=1, le=86_400)
    scip_timeout_typescript: int = Field(ge=1, le=86_400)
    scip_timeout_java: int = Field(ge=1, le=86_400)
    scip_timeout_php: int = Field(ge=1, le=86_400)
    scip_node_memory_mb: int = Field(ge=128, le=131_072)
    scip_best_effort: bool
    metrics_strict_mode: bool
    metrics_languages: str = Field(max_length=128)
    evolution_window_days: int = Field(ge=1, le=3650)
    temporal_coupling_max_files_per_commit: int = Field(ge=0, le=100_000)
    lsp_request_timeout_seconds: float = Field(ge=1, le=300)
    joern_parse_timeout_seconds: int = Field(ge=1, le=86_400)
    joern_required: bool

    @field_validator("owner", "repository")
    @classmethod
    def validate_repository_component(cls, value: str) -> str:
        if not _REPOSITORY_RE.fullmatch(value):
            raise ValueError("must be a GitHub owner or repository name")
        return value

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value.startswith("-") or ".." in value or value.endswith("."):
            raise ValueError("invalid Git branch")
        if any(ord(char) < 32 or char in " ~^:?*[\\" for char in value):
            raise ValueError("invalid Git branch")
        return value

    @field_validator("previous_commit")
    @classmethod
    def validate_previous_commit(cls, value: str | None) -> str | None:
        if value is not None and not _SHA_RE.fullmatch(value):
            raise ValueError("previous_commit must be a full lowercase Git SHA")
        return value


class AnalyzedFile(BaseModel):
    """One bounded UTF-8 repository file returned by the analyzer."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096)
    content: str = Field(max_length=1_048_576)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or normalized != value:
            raise ValueError("path must be a normalized relative POSIX path")
        parts = normalized.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("path traversal is not allowed")
        return value


class LspSymbol(BaseModel):
    """One validated symbol produced by the optional sandboxed language server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(max_length=512)
    kind_id: int = Field(ge=1, le=256)
    kind: str = Field(min_length=1, max_length=64)
    file_path: str = Field(min_length=1, max_length=4096)
    line_number: int = Field(ge=1)
    column: int = Field(ge=0)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        return AnalyzedFile.validate_relative_path(value)


class LspAnalysisResult(BaseModel):
    """Bounded optional LSP result persisted with the source version."""

    model_config = ConfigDict(extra="forbid")

    files_scanned: int = Field(ge=0, le=30)
    errors: list[str] = Field(default_factory=list, max_length=30)
    symbols: list[LspSymbol] = Field(default_factory=list, max_length=100_000)


class SandboxAnalysisResult(BaseModel):
    """Validated output imported from an untrusted analyzer sandbox."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source_id: uuid.UUID
    analyzer_profile: str
    commit: str
    previous_commit: str | None = None
    files: list[AnalyzedFile] = Field(default_factory=list, max_length=100_000)
    deleted_paths: list[str] = Field(default_factory=list, max_length=100_000)
    scip_stats: dict[str, Any] = Field(default_factory=dict)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    snapshots: list[dict[str, Any]] = Field(default_factory=list)
    file_metrics: list[dict[str, Any]] = Field(default_factory=list)
    evolution: dict[str, Any] | None = None
    lsp: LspAnalysisResult | None = None
    joern_status: Literal["ready", "failed", "disabled"] = "disabled"
    joern_error: str = Field(default="", max_length=4096)

    @field_validator("commit")
    @classmethod
    def validate_commit(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(value):
            raise ValueError("commit must be a full lowercase Git SHA")
        return value

    @field_validator("previous_commit")
    @classmethod
    def validate_result_previous_commit(cls, value: str | None) -> str | None:
        if value is not None and not _SHA_RE.fullmatch(value):
            raise ValueError("previous_commit must be a full lowercase Git SHA")
        return value

    @field_validator("deleted_paths")
    @classmethod
    def validate_deleted_paths(cls, values: list[str]) -> list[str]:
        for value in values:
            AnalyzedFile.validate_relative_path(value)
        return values

    @model_validator(mode="after")
    def validate_unique_paths(self) -> SandboxAnalysisResult:
        file_paths = [item.path for item in self.files]
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("result contains duplicate file paths")
        if set(file_paths) & set(self.deleted_paths):
            raise ValueError("a path cannot be both changed and deleted")
        for snapshot in self.snapshots:
            for file_info in snapshot.get("files") or []:
                if isinstance(file_info, dict) and file_info.get("path"):
                    AnalyzedFile.validate_relative_path(str(file_info["path"]))
            for key in ("symbols", "occurrences"):
                for item in snapshot.get(key) or []:
                    if isinstance(item, dict) and item.get("file_path"):
                        AnalyzedFile.validate_relative_path(str(item["file_path"]))
        for metric in self.file_metrics:
            if metric.get("file_path"):
                AnalyzedFile.validate_relative_path(str(metric["file_path"]))
        return self


class SandboxArtifactManifest(BaseModel):
    """One bounded binary analyzer artifact returned in canonical chunks."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["repository.cpg.bin"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1)
    parts: list[str] = Field(min_length=1, max_length=512)

    @field_validator("parts")
    @classmethod
    def validate_parts(cls, values: list[str]) -> list[str]:
        expected = [f"cpg-part-{index:04d}.bin" for index in range(len(values))]
        if values != expected:
            raise ValueError("artifact parts must be ordered canonical filenames")
        return values


class SandboxResultManifest(BaseModel):
    """Small manifest used to retrieve and authenticate chunked result bytes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    encoding: Literal["gzip"] = "gzip"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compressed_size: int = Field(ge=1)
    uncompressed_size: int = Field(ge=1)
    parts: list[str] = Field(min_length=1, max_length=128)
    artifacts: list[SandboxArtifactManifest] = Field(default_factory=list, max_length=1)

    @field_validator("parts")
    @classmethod
    def validate_parts(cls, values: list[str]) -> list[str]:
        expected = [f"part-{index:04d}.bin" for index in range(len(values))]
        if values != expected:
            raise ValueError("manifest parts must be ordered canonical filenames")
        return values


def github_proxy_config(token: str | None) -> dict[str, Any]:
    """Allow only GitHub HTTPS egress, keeping credentials outside the sandbox."""
    if token:
        headers = [
            {
                "name": "Authorization",
                "type": "opaque",
                "value": opaque_secret(f"Bearer {token}")["value"],
            }
        ]
    else:
        headers = [
            {
                "name": "User-Agent",
                "type": "plaintext",
                "value": "contextmine-sandbox-analyzer",
            }
        ]
    return proxy_config(
        rules=[
            {
                "name": "github",
                "match_hosts": ["github.com"],
                "match_paths": ["*"],
                "headers": headers,
            }
        ]
    )


def _decompress_bounded(payload: bytes, max_bytes: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as source:
        result = source.read(max_bytes + 1)
    if len(result) > max_bytes:
        raise ValueError("sandbox result exceeds configured size limit")
    return result


async def run_sandbox_analysis(
    request: SandboxAnalysisRequest,
    *,
    api_endpoint: str,
    api_key: SecretStr,
    snapshot_name: str,
    github_token: str | None,
    timeout_seconds: int,
    max_result_bytes: int,
    max_artifact_bytes: int,
    joern_cpg_root: Path,
    vcpus: int,
    mem_bytes: int,
    fs_capacity_bytes: int,
) -> SandboxAnalysisResult:
    """Run one analyzer and validate its bounded, chunked result before import."""
    sandbox = None
    handle = None
    async with AsyncSandboxClient(
        api_endpoint=api_endpoint,
        api_key=api_key.get_secret_value(),
        timeout=30,
    ) as client:
        try:
            sandbox = await client.create_sandbox(
                snapshot_name=snapshot_name,
                name=f"contextmine-{uuid.uuid4().hex[:20]}",
                timeout=min(timeout_seconds, 300),
                idle_ttl_seconds=min(timeout_seconds + 300, 86_400),
                delete_after_stop_seconds=300,
                vcpus=vcpus,
                mem_bytes=mem_bytes,
                fs_capacity_bytes=fs_capacity_bytes,
                proxy_config=github_proxy_config(github_token),
            )
            await sandbox.write("/workspace/request.json", request.model_dump_json())
            handle = await sandbox.run(
                _ANALYZER_COMMAND,
                cwd="/workspace",
                timeout=timeout_seconds,
                idle_timeout=min(timeout_seconds, 300),
                kill_on_disconnect=True,
                ttl_seconds=60,
                wait=False,
            )
            execution = await asyncio.wait_for(handle.result, timeout=timeout_seconds + 30)
            if execution.exit_code != 0:
                stderr = execution.stderr[-4096:]
                raise RuntimeError(
                    f"sandbox analyzer failed with exit {execution.exit_code}: {stderr}"
                )

            manifest_bytes = await sandbox.read(_RESULT_MANIFEST)
            if len(manifest_bytes) > 64 * 1024:
                raise ValueError("sandbox result manifest exceeds 64 KiB")
            manifest = SandboxResultManifest.model_validate_json(manifest_bytes)
            if manifest.uncompressed_size > max_result_bytes:
                raise ValueError("sandbox result exceeds configured size limit")
            if manifest.compressed_size > max_result_bytes:
                raise ValueError("compressed sandbox result exceeds configured size limit")

            compressed_parts: list[bytes] = []
            received = 0
            for part_name in manifest.parts:
                part = await sandbox.read(f"/workspace/result/{part_name}")
                received += len(part)
                if received > manifest.compressed_size or received > max_result_bytes:
                    raise ValueError("sandbox result part sizes do not match manifest")
                compressed_parts.append(part)
            compressed = b"".join(compressed_parts)
            if len(compressed) != manifest.compressed_size:
                raise ValueError("sandbox result compressed size does not match manifest")
            if hashlib.sha256(compressed).hexdigest() != manifest.sha256:
                raise ValueError("sandbox result checksum mismatch")
            payload = _decompress_bounded(compressed, max_result_bytes)
            if len(payload) != manifest.uncompressed_size or len(payload) > max_result_bytes:
                raise ValueError("sandbox result uncompressed size does not match manifest")
            result = SandboxAnalysisResult.model_validate_json(payload)
            if (
                result.source_id != request.source_id
                or result.analyzer_profile != request.analyzer_profile
                or result.previous_commit != request.previous_commit
            ):
                raise ValueError("sandbox result does not match its request")
            if (result.joern_status == "ready") != bool(manifest.artifacts):
                raise ValueError("Joern status and returned CPG artifact do not match")
            for artifact in manifest.artifacts:
                joern_cpg_path = joern_cpg_root / str(result.source_id) / f"{result.commit}.cpg.bin"
                received = 0
                digest = hashlib.sha256()
                joern_cpg_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=joern_cpg_path.parent,
                        prefix=f".{joern_cpg_path.name}.",
                        suffix=".partial",
                        delete=False,
                    ) as destination:
                        temporary_path = Path(destination.name)
                        for part_name in artifact.parts:
                            part = await sandbox.read(f"/workspace/result/{part_name}")
                            received += len(part)
                            if received > artifact.size or received > max_artifact_bytes:
                                raise ValueError("sandbox artifact exceeds configured size limit")
                            digest.update(part)
                            destination.write(part)
                    if received != artifact.size:
                        raise ValueError("sandbox artifact size does not match manifest")
                    if digest.hexdigest() != artifact.sha256:
                        raise ValueError("sandbox artifact checksum mismatch")
                    temporary_path.replace(joern_cpg_path)
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
            return result
        except asyncio.CancelledError:
            if handle is not None:
                with suppress(Exception):
                    await handle.kill()
            raise
        except TimeoutError:
            if handle is not None:
                with suppress(Exception):
                    await handle.kill()
            raise
        finally:
            if sandbox is not None:
                with suppress(Exception):
                    await sandbox.delete()
