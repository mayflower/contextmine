"""Repo-wide artifact inventory helpers for architecture recovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .schemas import EvidenceRef

_TEXT_EXTENSIONS = {
    ".md": ("documentation", "text/markdown", "markdown"),
    ".mdx": ("documentation", "text/markdown", "mdx"),
    ".rst": ("documentation", "text/x-rst", "rst"),
    ".txt": ("documentation", "text/plain", "plain_text"),
    ".puml": ("diagram", "text/plain", "plantuml"),
    ".plantuml": ("diagram", "text/plain", "plantuml"),
    ".mmd": ("diagram", "text/vnd.mermaid", "mermaid"),
    ".mermaid": ("diagram", "text/vnd.mermaid", "mermaid"),
    ".sql": ("sql", "application/sql", "sql"),
    ".tf": ("infrastructure", "text/plain", "terraform"),
    ".tfvars": ("infrastructure", "text/plain", "terraform"),
}
_YAML_EXTENSIONS = {".yaml", ".yml"}
_JSON_EXTENSIONS = {".json"}


@dataclass(frozen=True)
class ArtifactInventoryEntry:
    """One typed artifact discovered from repo files or persisted documents."""

    artifact_id: str
    artifact_kind: str
    repo_path: str
    media_type: str
    parser_hint: str
    raw_text: str | None = None
    raw_data: dict[str, Any] | list[Any] | None = None
    evidence: tuple[EvidenceRef, ...] = ()


def _normalize_repo_path(repo_path: str) -> str:
    return PurePosixPath(str(repo_path or "").strip()).as_posix()


def _artifact_id(repo_path: str) -> str:
    normalized = _normalize_repo_path(repo_path)
    return f"artifact:{normalized}" if normalized else "artifact:unknown"


def _dedupe_evidence(evidence: Iterable[EvidenceRef]) -> tuple[EvidenceRef, ...]:
    seen: set[tuple[str, str, int | None, int | None]] = set()
    rows: list[EvidenceRef] = []
    for ref in evidence:
        key = (ref.kind, ref.ref, ref.start_line, ref.end_line)
        if key in seen:
            continue
        seen.add(key)
        rows.append(ref)
    return tuple(rows)


def _classify_text_artifact(repo_path: str, raw_text: str) -> tuple[str, str, str]:
    normalized = _normalize_repo_path(repo_path).lower()
    path = PurePosixPath(normalized)
    name = path.name
    suffix = path.suffix.lower()
    lowered = raw_text.lower()

    if normalized.endswith(".c4.dsl"):
        return ("diagram", "text/plain", "c4_dsl")
    if suffix in _TEXT_EXTENSIONS:
        return _TEXT_EXTENSIONS[suffix]

    if ".github/workflows/" in normalized and suffix in _YAML_EXTENSIONS:
        return ("ci_workflow", "application/yaml", "github_actions_workflow")
    if name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
        return ("deployment", "application/yaml", "docker_compose")
    if name == "chart.yaml" or "/charts/" in normalized:
        return ("deployment", "application/yaml", "helm_chart")
    if suffix in _YAML_EXTENSIONS | _JSON_EXTENSIONS:
        if "asyncapi:" in lowered:
            media_type = "application/json" if suffix in _JSON_EXTENSIONS else "application/yaml"
            return ("api_spec", media_type, "asyncapi")
        if "openapi:" in lowered or "swagger:" in lowered:
            media_type = "application/json" if suffix in _JSON_EXTENSIONS else "application/yaml"
            return ("api_spec", media_type, "openapi")
        if "apiversion:" in lowered and "\nkind:" in lowered:
            return ("deployment", "application/yaml", "kubernetes_manifest")

    return ("documentation", "text/plain", "plain_text")


def _build_entry(
    *,
    repo_path: str,
    raw_text: str | None,
    raw_data: dict[str, Any] | list[Any] | None,
    evidence: Iterable[EvidenceRef],
) -> ArtifactInventoryEntry:
    artifact_kind, media_type, parser_hint = _classify_text_artifact(repo_path, raw_text or "")
    return ArtifactInventoryEntry(
        artifact_id=_artifact_id(repo_path),
        artifact_kind=artifact_kind,
        repo_path=_normalize_repo_path(repo_path),
        media_type=media_type,
        parser_hint=parser_hint,
        raw_text=raw_text,
        raw_data=raw_data,
        evidence=_dedupe_evidence(evidence),
    )


def build_repo_artifact_inventory(
    *,
    repo_root: str | Path,
    repo_paths: list[str] | None = None,
) -> list[ArtifactInventoryEntry]:
    """Build inventory entries from files in a repository checkout."""

    root = Path(repo_root)
    selected_paths = repo_paths
    if selected_paths is None:
        selected_paths = [
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        ]

    artifacts: list[ArtifactInventoryEntry] = []
    for repo_path in selected_paths:
        file_path = root / repo_path
        if not file_path.is_file():
            continue
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        artifacts.append(
            _build_entry(
                repo_path=repo_path,
                raw_text=raw_text,
                raw_data=None,
                evidence=(EvidenceRef(kind="file", ref=_normalize_repo_path(repo_path)),),
            )
        )
    return sorted(artifacts, key=lambda row: row.repo_path)


def build_document_artifact_inventory(documents: Iterable[Any]) -> list[ArtifactInventoryEntry]:
    """Build inventory entries from persisted document rows."""

    artifacts: list[ArtifactInventoryEntry] = []
    for doc in documents:
        meta = doc.meta if isinstance(getattr(doc, "meta", None), dict) else {}
        repo_path = _normalize_repo_path(
            str(meta.get("file_path") or getattr(doc, "uri", "") or getattr(doc, "title", ""))
        )
        raw_text = str(getattr(doc, "content_markdown", "") or "")
        evidence: list[EvidenceRef] = []
        if repo_path:
            evidence.append(EvidenceRef(kind="file", ref=repo_path))
        doc_id = str(getattr(doc, "id", "") or "").strip()
        if doc_id:
            evidence.append(EvidenceRef(kind="artifact", ref=f"document:{doc_id}"))
        artifacts.append(
            _build_entry(
                repo_path=repo_path,
                raw_text=raw_text,
                raw_data={
                    "title": str(getattr(doc, "title", "") or "").strip() or None,
                    "uri": str(getattr(doc, "uri", "") or "").strip() or None,
                },
                evidence=evidence,
            )
        )
    return sorted(artifacts, key=lambda row: row.repo_path)


def merge_document_and_repo_artifacts(
    document_artifacts: Iterable[ArtifactInventoryEntry],
    repo_artifacts: Iterable[ArtifactInventoryEntry],
) -> list[ArtifactInventoryEntry]:
    """Merge artifacts discovered from repo files and persisted documents."""

    merged: dict[str, ArtifactInventoryEntry] = {
        artifact.repo_path: artifact for artifact in repo_artifacts if artifact.repo_path
    }
    for artifact in document_artifacts:
        existing = merged.get(artifact.repo_path)
        if existing is None:
            merged[artifact.repo_path] = artifact
            continue
        preferred = (
            artifact if len(artifact.raw_text or "") >= len(existing.raw_text or "") else existing
        )
        merged[artifact.repo_path] = replace(
            preferred,
            evidence=_dedupe_evidence(list(existing.evidence) + list(artifact.evidence)),
            raw_data=preferred.raw_data or existing.raw_data,
        )
    return [merged[key] for key in sorted(merged)]
