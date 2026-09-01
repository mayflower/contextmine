"""Entrypoint executed only inside an ephemeral repository analyzer sandbox."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from git import Repo

from contextmine_worker.github_sync import (
    get_changed_files,
    is_eligible_file,
    read_file_content,
)
from contextmine_worker.sandbox_analysis import (
    AnalyzedFile,
    LspAnalysisResult,
    LspSymbol,
    SandboxAnalysisRequest,
    SandboxAnalysisResult,
    SandboxArtifactManifest,
    SandboxResultManifest,
)

_PART_SIZE = 8 * 1024 * 1024


def _safe_repository_file(repo_root: Path, relative_path: str) -> Path | None:
    """Return a regular in-repository file without following symlinks out."""
    try:
        normalized = AnalyzedFile.validate_relative_path(relative_path)
        candidate = repo_root / normalized
        if candidate.is_symlink() or not candidate.is_file():
            return None
        candidate.resolve(strict=True).relative_to(repo_root.resolve(strict=True))
        return candidate
    except OSError, ValueError:
        return None


async def _run_scip_analysis(
    repo: Repo, repo_root: Path, request: SandboxAnalysisRequest
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Run the existing SCIP and metrics implementation inside the sandbox."""
    from contextmine_core.metrics import flatten_metric_bundles, run_polyglot_metrics_pipeline
    from contextmine_core.semantic_snapshot.indexers.language_census import build_language_census
    from contextmine_core.semantic_snapshot.models import Language

    from contextmine_worker import flows
    from contextmine_worker.github_sync import (
        compute_git_change_metrics,
        compute_git_evolution_snapshots,
    )

    stats = flows._default_scip_stats()
    snapshots: list[dict[str, Any]] = []
    detection = await flows._detect_scip_projects_impl(repo_root)
    projects = list(detection.get("projects") or [])
    diagnostics = dict(detection.get("diagnostics") or {})
    stats.update(
        {
            "scip_languages_detected": list(diagnostics.get("languages_detected") or []),
            "scip_projects_by_language": dict(diagnostics.get("projects_by_language") or {}),
            "scip_detection_warnings": list(diagnostics.get("warnings") or []),
            "scip_census_tool": str(diagnostics.get("census_tool") or ""),
            "scip_census_tool_version": str(diagnostics.get("census_tool_version") or ""),
            "scip_projects_detected": len(projects),
        }
    )
    census = build_language_census(repo_root)
    detected_files = {
        language.value: int(entry.files)
        for language, entry in census.entries.items()
        if int(entry.files) > 0
    }
    stats["scip_detected_files_by_language"] = detected_files
    stats["scip_detected_code_by_language"] = {
        language.value: int(entry.code)
        for language, entry in census.entries.items()
        if int(entry.code) > 0
    }

    output_dir = Path(tempfile.mkdtemp(prefix="scip-", dir=repo_root.parent))
    for project in projects:
        language = flows._scip_normalize_language(project.get("language"))
        artifact = await flows._index_scip_project_impl(project, output_dir)
        if not artifact or not artifact.get("success") or not artifact.get("scip_path"):
            stats["scip_projects_failed"] += 1
            flows._scip_get_failed_projects(stats).append(
                {
                    "language": language,
                    "project_root": str(project.get("root_path") or ""),
                    "error": str((artifact or {}).get("error_message") or "index_failed"),
                }
            )
            continue
        snapshot = await flows._parse_scip_snapshot_impl(str(artifact["scip_path"]))
        if not snapshot:
            stats["scip_projects_failed"] += 1
            continue
        project_root = Path(str(project.get("root_path") or repo_root)).resolve()
        try:
            relative_root = project_root.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            relative_root = ""
        meta = dict(snapshot.get("meta") or {})
        meta.update({"repo_relative_root": relative_root, "language": language})
        snapshot["meta"] = meta
        snapshots.append(snapshot)
        stats["scip_projects_indexed"] += 1
        stats["scip_snapshots_parsed"] += 1
        stats["scip_symbols"] += len(snapshot.get("symbols") or [])
        stats["scip_relations"] += len(snapshot.get("relations") or [])

    supported = {language.value for language in Language}
    indexed_files = flows._scip_collect_indexed_files_by_language(snapshots, repo_root, supported)
    relation_counts, relation_kinds = flows._scip_collect_relation_coverage_by_language(
        snapshots, supported
    )
    missing_relations = flows._scip_missing_relation_languages(indexed_files, relation_kinds)
    flows._scip_append_file_coverage_completion_snapshot(
        snapshots,
        census_report=census,
        repo_path=repo_root,
        supported_languages=supported,
    )
    indexed_files = flows._scip_collect_indexed_files_by_language(snapshots, repo_root, supported)
    flows._scip_finalize_coverage_stats(
        stats,
        detected_files,
        indexed_files,
        relation_counts,
        relation_kinds,
        missing_relations,
    )

    bundles = run_polyglot_metrics_pipeline(
        repo_root=repo_root,
        project_dicts=projects,
        snapshot_dicts=snapshots,
        strict_mode=request.metrics_strict_mode,
        metrics_languages=request.metrics_languages,
    )
    file_metrics = [record.to_dict() for record in flatten_metric_bundles(bundles or [])]
    target_files = {
        str(metric.get("file_path") or "") for metric in file_metrics if metric.get("file_path")
    }
    git_metrics = compute_git_change_metrics(
        repo, target_files, since_days=request.evolution_window_days
    )
    flows._enrich_file_metrics_with_git(file_metrics, git_metrics, request.evolution_window_days)
    evolution = compute_git_evolution_snapshots(
        repo,
        target_files,
        window_days=request.evolution_window_days,
        max_files_per_commit=request.temporal_coupling_max_files_per_commit,
    )
    for snapshot in snapshots:
        meta = dict(snapshot.get("meta") or {})
        for file_info in snapshot.get("files") or []:
            if isinstance(file_info, dict):
                path = flows._scip_snapshot_repo_file_path(file_info, meta, repo_root)
                if path:
                    file_info["path"] = path
        for key in ("symbols", "occurrences"):
            for item in snapshot.get(key) or []:
                if not isinstance(item, dict) or not item.get("file_path"):
                    continue
                path = Path(str(item["file_path"]))
                if path.is_absolute():
                    try:
                        item["file_path"] = (
                            path.resolve().relative_to(repo_root.resolve()).as_posix()
                        )
                    except ValueError:
                        item["file_path"] = path.name
                else:
                    item["file_path"] = str(item["file_path"]).replace("\\", "/").lstrip("./")
    stats["structural_metric_files"] = len(file_metrics)
    stats["evolution_window_days"] = request.evolution_window_days
    return stats, projects, snapshots, file_metrics, evolution


def _lsp_kind_name(kind: int) -> str:
    return {
        1: "file",
        2: "module",
        3: "namespace",
        4: "package",
        5: "class",
        6: "method",
        7: "property",
        8: "field",
        9: "constructor",
        10: "enum",
        11: "interface",
        12: "function",
        13: "variable",
    }.get(kind, f"kind_{kind}")


def _flatten_lsp_result(file_path: str, raw_symbols: list[dict[str, Any]]) -> list[LspSymbol]:
    symbols: list[LspSymbol] = []
    stack = list(raw_symbols)
    while stack:
        item = stack.pop()
        if not isinstance(item, dict):
            continue
        symbol_range = item.get("selectionRange") or item.get("range") or {}
        start = symbol_range.get("start", {}) if isinstance(symbol_range, dict) else {}
        try:
            kind = int(item.get("kind") or 0)
            symbols.append(
                LspSymbol(
                    name=str(item.get("name") or ""),
                    kind_id=kind,
                    kind=_lsp_kind_name(kind),
                    file_path=file_path,
                    line_number=int(start.get("line") or 0) + 1,
                    column=int(start.get("character") or 0),
                )
            )
        except TypeError, ValueError:
            pass
        children = item.get("children")
        if isinstance(children, list):
            stack.extend(children)
    return symbols


async def _run_lsp_analysis(repo_root: Path) -> LspAnalysisResult:
    """Run pre-installed language servers in the ephemeral analyzer only."""
    from contextmine_core.lsp.manager import get_lsp_manager, shutdown_lsp_manager

    candidates = sorted(
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}
    )[:30]
    manager = get_lsp_manager()
    symbols: list[LspSymbol] = []
    errors: list[str] = []
    files_scanned = 0
    try:
        for file_path in candidates:
            relative_path = file_path.relative_to(repo_root).as_posix()
            try:
                client = await manager.get_client(file_path=file_path, project_root=repo_root)
                raw_symbols = await client.get_document_symbols(str(file_path))
                symbols.extend(_flatten_lsp_result(relative_path, raw_symbols))
                files_scanned += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{relative_path}: {exc}"[:4096])
    finally:
        await shutdown_lsp_manager()
    return LspAnalysisResult(
        files_scanned=files_scanned,
        errors=errors[:30],
        symbols=symbols[:100_000],
    )


async def analyze(request: SandboxAnalysisRequest, workspace: Path) -> SandboxAnalysisResult:
    """Clone and execute repository tooling without installing dependencies."""
    if os.geteuid() == 0:
        raise RuntimeError("sandbox analyzer refuses to run as root")
    os.environ.update(
        {
            "SCIP_INSTALL_DEPS_MODE": "never",
            "SCIP_LANGUAGES": request.scip_languages,
            "SCIP_TIMEOUT_PYTHON": str(request.scip_timeout_python),
            "SCIP_TIMEOUT_TYPESCRIPT": str(request.scip_timeout_typescript),
            "SCIP_TIMEOUT_JAVA": str(request.scip_timeout_java),
            "SCIP_TIMEOUT_PHP": str(request.scip_timeout_php),
            "SCIP_NODE_MEMORY_MB": str(request.scip_node_memory_mb),
            "SCIP_BEST_EFFORT": str(request.scip_best_effort).lower(),
            "LSP_REQUEST_TIMEOUT_SECONDS": str(request.lsp_request_timeout_seconds),
        }
    )
    repo_root = workspace / "repository"
    clone_url = f"https://github.com/{request.owner}/{request.repository}.git"
    clone_options: dict[str, object] = {"no_single_branch": True}
    if request.branch:
        clone_options["branch"] = request.branch
    repo = Repo.clone_from(clone_url, repo_root, **clone_options)
    commit = repo.head.commit.hexsha
    changed_paths, deleted_paths = get_changed_files(repo, request.previous_commit, commit)

    files: list[AnalyzedFile] = []
    for relative_path in changed_paths:
        file_path = _safe_repository_file(repo_root, relative_path)
        if file_path is None or not is_eligible_file(Path(relative_path), repo_root):
            continue
        content = read_file_content(repo_root, relative_path)
        if content is not None:
            files.append(AnalyzedFile(path=relative_path, content=content))

    safe_deleted = [
        path
        for path in deleted_paths
        if not path.startswith("/") and ".." not in Path(path).parts and "\\" not in path
    ]
    try:
        scip_stats, projects, snapshots, file_metrics, evolution = await _run_scip_analysis(
            repo, repo_root, request
        )
    except Exception as exc:  # noqa: BLE001
        scip_stats = {"scip_degraded": True, "scip_detection_warnings": [str(exc)[:4096]]}
        projects, snapshots, file_metrics, evolution = [], [], [], {}

    try:
        lsp = await _run_lsp_analysis(repo_root)
    except Exception as exc:  # noqa: BLE001
        lsp = LspAnalysisResult(files_scanned=0, errors=[str(exc)[:4096]], symbols=[])

    joern_status = "disabled"
    joern_error = ""
    if shutil.which("joern-parse"):
        (workspace / "result").mkdir(parents=True, exist_ok=True)
        cpg_path = workspace / "result" / "repository.cpg.bin"
        try:
            completed = subprocess.run(
                ["joern-parse", str(repo_root), "--output", str(cpg_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=request.joern_parse_timeout_seconds,
            )
            if completed.returncode == 0 and cpg_path.is_file() and cpg_path.stat().st_size > 0:
                joern_status = "ready"
            else:
                joern_status = "failed"
                joern_error = completed.stderr[-4096:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            joern_status = "failed"
            joern_error = str(exc)[:4096]
    elif request.joern_required:
        joern_status = "failed"
        joern_error = "joern-parse is not installed in the analyzer snapshot"
    return SandboxAnalysisResult(
        source_id=request.source_id,
        analyzer_profile=request.analyzer_profile,
        commit=commit,
        previous_commit=request.previous_commit,
        files=files,
        deleted_paths=safe_deleted,
        scip_stats=scip_stats,
        projects=projects,
        snapshots=snapshots,
        file_metrics=file_metrics,
        evolution=evolution,
        lsp=lsp,
        joern_status=joern_status,
        joern_error=joern_error,
    )


def write_result(result: SandboxAnalysisResult, output_dir: Path) -> None:
    """Write a deterministic, chunked gzip payload for the platform file API."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump_json().encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    parts: list[str] = []
    for index, offset in enumerate(range(0, len(compressed), _PART_SIZE)):
        name = f"part-{index:04d}.bin"
        (output_dir / name).write_bytes(compressed[offset : offset + _PART_SIZE])
        parts.append(name)
    artifacts: list[SandboxArtifactManifest] = []
    cpg_path = output_dir / "repository.cpg.bin"
    if cpg_path.is_file():
        artifact_parts: list[str] = []
        artifact_digest = hashlib.sha256()
        artifact_size = 0
        with cpg_path.open("rb") as source:
            for index, chunk in enumerate(iter(lambda: source.read(_PART_SIZE), b"")):
                name = f"cpg-part-{index:04d}.bin"
                (output_dir / name).write_bytes(chunk)
                artifact_parts.append(name)
                artifact_digest.update(chunk)
                artifact_size += len(chunk)
        artifacts.append(
            SandboxArtifactManifest(
                name="repository.cpg.bin",
                sha256=artifact_digest.hexdigest(),
                size=artifact_size,
                parts=artifact_parts,
            )
        )
    manifest = SandboxResultManifest(
        sha256=hashlib.sha256(compressed).hexdigest(),
        compressed_size=len(compressed),
        uncompressed_size=len(payload),
        parts=parts,
        artifacts=artifacts,
    )
    (output_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: sandbox_analyzer REQUEST_JSON OUTPUT_DIR")
    request_path = Path(sys.argv[1]).resolve(strict=True)
    output_dir = Path(sys.argv[2]).resolve()
    workspace = request_path.parent.resolve(strict=True)
    output_dir.relative_to(workspace)
    request = SandboxAnalysisRequest.model_validate_json(request_path.read_bytes())
    write_result(asyncio.run(analyze(request, workspace)), output_dir)


if __name__ == "__main__":
    main()
