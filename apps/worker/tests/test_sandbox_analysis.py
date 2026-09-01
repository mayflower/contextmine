"""Behavioral tests for the repository sandbox trust boundary."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextmine_worker.sandbox_analysis import (
    AnalyzedFile,
    SandboxAnalysisRequest,
    SandboxAnalysisResult,
    SandboxArtifactManifest,
    SandboxResultManifest,
    _decompress_bounded,
    github_proxy_config,
    run_sandbox_analysis,
)
from contextmine_worker.sandbox_analyzer import _safe_repository_file, write_result
from pydantic import SecretStr, ValidationError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _request() -> SandboxAnalysisRequest:
    return SandboxAnalysisRequest(
        source_id=uuid.uuid4(),
        owner="mayflower",
        repository="contextmine",
        branch="main",
        previous_commit="a" * 40,
        scip_languages="python,typescript,javascript,java,php",
        scip_timeout_python=300,
        scip_timeout_typescript=600,
        scip_timeout_java=900,
        scip_timeout_php=900,
        scip_node_memory_mb=4096,
        scip_best_effort=True,
        metrics_strict_mode=True,
        metrics_languages="python,typescript,javascript,java,php",
        evolution_window_days=365,
        temporal_coupling_max_files_per_commit=200,
        lsp_request_timeout_seconds=30,
        joern_parse_timeout_seconds=900,
        joern_required=False,
    )


def test_request_rejects_command_injection_and_invalid_sha() -> None:
    with pytest.raises(ValidationError):
        SandboxAnalysisRequest.model_validate(
            _request().model_dump() | {"branch": "main; touch /tmp/pwned"}
        )
    with pytest.raises(ValidationError):
        SandboxAnalysisRequest.model_validate(_request().model_dump() | {"previous_commit": "HEAD"})


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "a\\b.py", "a//b.py"])
def test_result_rejects_traversal_and_noncanonical_paths(path: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        AnalyzedFile(path=path, content="x")


def test_safe_repository_file_rejects_symlink_escape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("secret", encoding="utf-8")
    (repo / "escape.py").symlink_to(outside)
    assert _safe_repository_file(repo, "escape.py") is None


def test_result_rejects_invalid_encoding() -> None:
    with pytest.raises(ValidationError):
        SandboxAnalysisResult.model_validate_json(b"\xff")


def test_result_decompression_is_bounded() -> None:
    compressed = gzip.compress(b"x" * 4096, mtime=0)
    with pytest.raises(ValueError, match="size limit"):
        _decompress_bounded(compressed, 1024)


def test_write_result_round_trips_and_is_deterministic(tmp_path: Path) -> None:
    result = SandboxAnalysisResult(
        source_id=uuid.uuid4(),
        analyzer_profile="scip-kg-v1",
        commit="b" * 40,
        files=[AnalyzedFile(path="src/app.py", content="print('ok')")],
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_result(result, first)
    write_result(result, second)
    first_manifest = SandboxResultManifest.model_validate_json(
        (first / "manifest.json").read_bytes()
    )
    second_manifest = SandboxResultManifest.model_validate_json(
        (second / "manifest.json").read_bytes()
    )
    assert first_manifest == second_manifest
    compressed = b"".join((first / name).read_bytes() for name in first_manifest.parts)
    assert hashlib.sha256(compressed).hexdigest() == first_manifest.sha256
    assert SandboxAnalysisResult.model_validate_json(gzip.decompress(compressed)) == result


def test_write_result_chunks_joern_artifact(tmp_path: Path) -> None:
    result = SandboxAnalysisResult(
        source_id=uuid.uuid4(),
        analyzer_profile="scip-kg-v1",
        commit="b" * 40,
        joern_status="ready",
    )
    output = tmp_path / "result"
    output.mkdir()
    (output / "repository.cpg.bin").write_bytes(b"validated-cpg")

    write_result(result, output)

    manifest = SandboxResultManifest.model_validate_json((output / "manifest.json").read_bytes())
    assert manifest.artifacts == [
        SandboxArtifactManifest(
            name="repository.cpg.bin",
            sha256=hashlib.sha256(b"validated-cpg").hexdigest(),
            size=len(b"validated-cpg"),
            parts=["cpg-part-0000.bin"],
        )
    ]


def test_github_token_is_opaque_and_public_rule_has_no_credential() -> None:
    private = github_proxy_config("top-secret")
    header = private["rules"][0]["headers"][0]
    assert header == {"name": "Authorization", "type": "opaque", "value": "Bearer top-secret"}
    public = github_proxy_config(None)
    assert public["rules"][0]["match_hosts"] == ["github.com"]
    assert public["rules"][0]["headers"][0]["type"] == "plaintext"


def _sandbox_payload(request: SandboxAnalysisRequest) -> tuple[bytes, bytes]:
    result = SandboxAnalysisResult(
        source_id=request.source_id,
        analyzer_profile=request.analyzer_profile,
        commit="b" * 40,
        previous_commit=request.previous_commit,
        files=[AnalyzedFile(path="README.md", content="hello")],
    )
    payload = result.model_dump_json().encode()
    compressed = gzip.compress(payload, mtime=0)
    manifest = SandboxResultManifest(
        sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_size=len(compressed),
        uncompressed_size=len(payload),
        parts=["part-0000.bin"],
    )
    return manifest.model_dump_json().encode(), compressed


@pytest.mark.anyio
async def test_client_validates_result_and_always_deletes_sandbox() -> None:
    request = _request()
    manifest, compressed = _sandbox_payload(request)
    sandbox = AsyncMock()
    sandbox.run.return_value.result = asyncio.sleep(0, result=MagicMock(exit_code=0, stderr=""))
    sandbox.read.side_effect = [manifest, compressed]
    client = AsyncMock()
    client.create_sandbox.return_value = sandbox
    client.__aenter__.return_value = client
    with patch("contextmine_worker.sandbox_analysis.AsyncSandboxClient", return_value=client):
        result = await run_sandbox_analysis(
            request,
            api_endpoint="https://sandbox.example.test",
            api_key=SecretStr("platform-token"),
            snapshot_name="contextmine-analyzer",
            github_token=None,
            timeout_seconds=60,
            max_result_bytes=1024 * 1024,
            max_artifact_bytes=1024 * 1024,
            joern_cpg_root=Path("/tmp/contextmine-test-cpg"),
            vcpus=2,
            mem_bytes=2 * 1024**3,
            fs_capacity_bytes=4 * 1024**3,
        )
    assert result.commit == "b" * 40
    sandbox.delete.assert_awaited_once()


@pytest.mark.anyio
async def test_client_validates_and_imports_joern_artifact(tmp_path: Path) -> None:
    request = _request()
    result = SandboxAnalysisResult(
        source_id=request.source_id,
        analyzer_profile=request.analyzer_profile,
        commit="c" * 40,
        previous_commit=request.previous_commit,
        joern_status="ready",
    )
    payload = result.model_dump_json().encode()
    compressed = gzip.compress(payload, mtime=0)
    cpg = b"trusted-only-after-validation"
    manifest = SandboxResultManifest(
        sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_size=len(compressed),
        uncompressed_size=len(payload),
        parts=["part-0000.bin"],
        artifacts=[
            SandboxArtifactManifest(
                name="repository.cpg.bin",
                sha256=hashlib.sha256(cpg).hexdigest(),
                size=len(cpg),
                parts=["cpg-part-0000.bin"],
            )
        ],
    )
    sandbox = AsyncMock()
    sandbox.run.return_value.result = asyncio.sleep(0, result=MagicMock(exit_code=0, stderr=""))
    sandbox.read.side_effect = [manifest.model_dump_json().encode(), compressed, cpg]
    client = AsyncMock()
    client.create_sandbox.return_value = sandbox
    client.__aenter__.return_value = client

    with patch("contextmine_worker.sandbox_analysis.AsyncSandboxClient", return_value=client):
        await run_sandbox_analysis(
            request,
            api_endpoint="https://sandbox.example.test",
            api_key=SecretStr("platform-token"),
            snapshot_name="contextmine-analyzer",
            github_token=None,
            timeout_seconds=60,
            max_result_bytes=1024 * 1024,
            max_artifact_bytes=1024 * 1024,
            joern_cpg_root=tmp_path,
            vcpus=2,
            mem_bytes=2 * 1024**3,
            fs_capacity_bytes=4 * 1024**3,
        )

    assert (tmp_path / str(request.source_id) / f"{'c' * 40}.cpg.bin").read_bytes() == cpg


@pytest.mark.anyio
async def test_client_rejects_tampered_result_and_deletes_sandbox() -> None:
    request = _request()
    manifest, compressed = _sandbox_payload(request)
    sandbox = AsyncMock()
    sandbox.run.return_value.result = asyncio.sleep(0, result=MagicMock(exit_code=0, stderr=""))
    sandbox.read.side_effect = [manifest, compressed + b"tampered"]
    client = AsyncMock()
    client.create_sandbox.return_value = sandbox
    client.__aenter__.return_value = client
    with (
        patch("contextmine_worker.sandbox_analysis.AsyncSandboxClient", return_value=client),
        pytest.raises(ValueError, match="part sizes"),
    ):
        await run_sandbox_analysis(
            request,
            api_endpoint="https://sandbox.example.test",
            api_key=SecretStr("token"),
            snapshot_name="contextmine-analyzer",
            github_token=None,
            timeout_seconds=60,
            max_result_bytes=1024 * 1024,
            max_artifact_bytes=1024 * 1024,
            joern_cpg_root=Path("/tmp/contextmine-test-cpg"),
            vcpus=1,
            mem_bytes=1024**3,
            fs_capacity_bytes=2 * 1024**3,
        )
    sandbox.delete.assert_awaited_once()


@pytest.mark.anyio
async def test_client_kills_and_cleans_up_on_cancel() -> None:
    request = _request()
    never = asyncio.Future()
    sandbox = AsyncMock()
    sandbox.run.return_value.result = never
    client = AsyncMock()
    client.create_sandbox.return_value = sandbox
    client.__aenter__.return_value = client
    with patch("contextmine_worker.sandbox_analysis.AsyncSandboxClient", return_value=client):
        running = asyncio.create_task(
            run_sandbox_analysis(
                request,
                api_endpoint="https://sandbox.example.test",
                api_key=SecretStr("token"),
                snapshot_name="contextmine-analyzer",
                github_token=None,
                timeout_seconds=60,
                max_result_bytes=1024,
                max_artifact_bytes=1024,
                joern_cpg_root=Path("/tmp/contextmine-test-cpg"),
                vcpus=1,
                mem_bytes=1024,
                fs_capacity_bytes=2048,
            )
        )
        await asyncio.sleep(0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    sandbox.run.return_value.kill.assert_awaited_once()
    sandbox.delete.assert_awaited_once()


@pytest.mark.anyio
async def test_client_kills_and_cleans_up_on_timeout() -> None:
    request = _request()
    sandbox = AsyncMock()
    sandbox.run.return_value.result = asyncio.Future()
    client = AsyncMock()
    client.create_sandbox.return_value = sandbox
    client.__aenter__.return_value = client
    with (
        patch("contextmine_worker.sandbox_analysis.AsyncSandboxClient", return_value=client),
        patch(
            "contextmine_worker.sandbox_analysis.asyncio.wait_for",
            AsyncMock(side_effect=TimeoutError),
        ),
        pytest.raises(TimeoutError),
    ):
        await run_sandbox_analysis(
            request,
            api_endpoint="https://sandbox.example.test",
            api_key=SecretStr("token"),
            snapshot_name="contextmine-analyzer",
            github_token=None,
            timeout_seconds=60,
            max_result_bytes=1024,
            max_artifact_bytes=1024,
            joern_cpg_root=Path("/tmp/contextmine-test-cpg"),
            vcpus=1,
            mem_bytes=1024,
            fs_capacity_bytes=2048,
        )
    sandbox.run.return_value.kill.assert_awaited_once()
    sandbox.delete.assert_awaited_once()


@pytest.mark.anyio
async def test_analyzer_refuses_root_execution(tmp_path: Path) -> None:
    from contextmine_worker.sandbox_analyzer import analyze

    with (
        patch("contextmine_worker.sandbox_analyzer.os.geteuid", return_value=0),
        pytest.raises(RuntimeError, match="refuses to run as root"),
    ):
        await analyze(_request(), tmp_path)


@pytest.mark.anyio
async def test_github_sync_uses_sandbox_without_local_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextmine_worker import flows

    request = _request()
    source = MagicMock(
        id=request.source_id,
        collection_id=uuid.uuid4(),
        cursor=request.previous_commit,
        config={"owner": request.owner, "repo": request.repository, "branch": request.branch},
    )
    ctx = flows._SyncGitHubCtx(source=source, sync_run=MagicMock(), run_started_at=MagicMock())
    result = SandboxAnalysisResult(
        source_id=request.source_id,
        analyzer_profile=request.analyzer_profile,
        commit="b" * 40,
        previous_commit=request.previous_commit,
    )
    sandbox_call = AsyncMock(return_value=result)
    monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="progress"))
    monkeypatch.setattr(flows, "get_deploy_key_for_source", AsyncMock(return_value=None))
    monkeypatch.setattr(flows, "get_github_token_for_source", AsyncMock(return_value="token"))
    monkeypatch.setattr(flows, "run_sandbox_analysis", sandbox_call)
    monkeypatch.setattr(
        flows,
        "get_settings",
        lambda: SimpleNamespace(
            app_mode="production",
            sandbox_api_url="https://sandbox.example.test",
            sandbox_api_key=SecretStr("platform-token"),
            sandbox_analyzer_snapshot="contextmine-analyzer",
            sandbox_analysis_timeout_seconds=600,
            sandbox_result_max_bytes=1024 * 1024,
            sandbox_artifact_max_bytes=1024 * 1024,
            sandbox_analyzer_vcpus=2,
            sandbox_analyzer_mem_bytes=2 * 1024**3,
            sandbox_analyzer_fs_bytes=4 * 1024**3,
            joern_cpg_root="/tmp/contextmine-test-cpg",
            scip_languages="python,typescript,javascript,java,php",
            scip_timeout_python=300,
            scip_timeout_typescript=600,
            scip_timeout_java=900,
            scip_timeout_php=900,
            scip_node_memory_mb=4096,
            scip_best_effort=True,
            metrics_strict_mode=True,
            metrics_languages="python,typescript,javascript,java,php",
            twin_evolution_window_days=365,
            sync_temporal_coupling_max_files_per_commit=200,
            lsp_request_timeout_seconds=30,
            joern_parse_timeout_seconds=900,
            joern_required_for_sync=False,
        ),
    )
    monkeypatch.setattr(
        flows, "clone_or_pull_repo", MagicMock(side_effect=AssertionError("local clone used"))
    )

    await flows._gh_phase_auth_and_clone(ctx)

    assert ctx.sandbox_result is result
    assert ctx.new_sha == "b" * 40
    sandbox_call.assert_awaited_once()


@pytest.mark.anyio
async def test_production_refuses_unsandboxed_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextmine_worker import flows

    source = MagicMock(
        id=uuid.uuid4(),
        collection_id=uuid.uuid4(),
        cursor=None,
        config={"owner": "mayflower", "repo": "contextmine"},
    )
    ctx = flows._SyncGitHubCtx(source=source, sync_run=MagicMock(), run_started_at=MagicMock())
    monkeypatch.setattr(flows, "create_progress_artifact", AsyncMock(return_value="progress"))
    monkeypatch.setattr(flows, "get_deploy_key_for_source", AsyncMock(return_value=None))
    monkeypatch.setattr(flows, "get_github_token_for_source", AsyncMock(return_value=None))
    monkeypatch.setattr(
        flows,
        "get_settings",
        lambda: SimpleNamespace(app_mode="production", sandbox_api_url=None),
    )

    with pytest.raises(RuntimeError, match="no local analyzer fallback"):
        await flows._gh_phase_auth_and_clone(ctx)
