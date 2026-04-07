"""Helpers for loading ADR-like documents into architecture recovery."""

from __future__ import annotations

from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from contextmine_core.models import Document, KnowledgeNode, KnowledgeNodeKind
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .artifact_inventory import (
    ArtifactInventoryEntry,
    build_document_artifact_inventory,
    merge_document_and_repo_artifacts,
)
from .artifact_parsers import parse_artifact

_TEXT_DOC_SUFFIXES = {".md", ".mdx", ".rst", ".txt"}
_ARCH_PATH_TOKENS = (
    "/adr/",
    "/adrs/",
    "/architecture/",
    "/architectural/",
    "/decision/",
    "/decisions/",
    "/design/",
    "/rfc/",
    "/rfcs/",
    "/notes/",
)
_ARCH_TEXT_TOKENS = (
    "adr-",
    "architecture decision",
    "design decision",
    "architectural decision",
    "status:",
    "\n# decision",
    "\n## decision",
    "\n# context",
    "\n## context",
    "\n# consequences",
    "\n## consequences",
    "\n# rationale",
    "\n## rationale",
)


def _document_path(doc: Any) -> str:
    meta = doc.meta if isinstance(getattr(doc, "meta", None), dict) else {}
    raw_path = str(meta.get("file_path") or getattr(doc, "uri", "") or "").strip()
    return raw_path


def _document_affected_entity_ids(doc: Any) -> list[str]:
    meta = doc.meta if isinstance(getattr(doc, "meta", None), dict) else {}
    affected = meta.get("affected_entity_ids")
    if not isinstance(affected, list):
        return []
    return [value for value in affected if isinstance(value, str) and value.strip()]


def _frontmatter_block(content: str) -> str | None:
    text = content.lstrip()
    if not text.startswith("---\n"):
        return None
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    collected: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return "\n".join(collected)
        collected.append(line)
    return None


def _frontmatter_affected_entity_ids(content: str) -> list[str]:
    block = _frontmatter_block(content)
    if not block:
        return []

    affected: list[str] = []
    in_list = False
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("affected_entity_ids:"):
            remainder = stripped.split(":", 1)[1].strip()
            if remainder.startswith("[") and remainder.endswith("]"):
                values = remainder[1:-1].split(",")
                return [value.strip().strip("\"'") for value in values if value.strip()]
            in_list = remainder == ""
            continue

        if in_list and stripped.startswith("- "):
            value = stripped[2:].strip().strip("\"'")
            if value:
                affected.append(value)
            continue

        if in_list and not raw_line.startswith((" ", "\t")):
            break

    return affected


def _looks_like_architecture_doc(doc: Any) -> bool:
    file_path = _document_path(doc)
    suffix = PurePosixPath(file_path or str(getattr(doc, "uri", ""))).suffix.lower()
    if suffix and suffix not in _TEXT_DOC_SUFFIXES:
        return False

    meta = doc.meta if isinstance(getattr(doc, "meta", None), dict) else {}
    if _document_affected_entity_ids(doc):
        return True
    if meta.get("doc_kind") in {"adr", "architecture_decision", "architecture_note"}:
        return True
    if meta.get("artifact_kind") in {"architecture_decision", "architecture_note"}:
        return True
    if meta.get("parser") in {"markdown_adr", "adr_frontmatter_v1"}:
        return True
    if meta.get("parser_hint") in {"markdown_adr", "adr_frontmatter_v1"}:
        return True

    lowered_path = file_path.lower()
    title = str(getattr(doc, "title", "") or "").strip().lower()
    content = str(getattr(doc, "content_markdown", "") or "")
    lowered_content = content.lower()

    if any(token in lowered_path for token in _ARCH_PATH_TOKENS):
        return True
    if any(token in title for token in ("adr", "decision", "architecture", "design", "rfc")):
        return True
    if any(token in lowered_content for token in _ARCH_TEXT_TOKENS):
        return True
    return bool(_frontmatter_affected_entity_ids(content))


def _document_to_recovery_doc(doc: Any) -> dict[str, Any]:
    file_path = _document_path(doc)
    affected_entity_ids = _document_affected_entity_ids(doc)
    if not affected_entity_ids:
        affected_entity_ids = _frontmatter_affected_entity_ids(
            str(getattr(doc, "content_markdown", "") or "")
        )

    meta: dict[str, Any] = {}
    if file_path:
        meta["file_path"] = file_path
    if affected_entity_ids:
        meta["affected_entity_ids"] = affected_entity_ids

    content = str(getattr(doc, "content_markdown", "") or "")
    title = str(getattr(doc, "title", "") or getattr(doc, "uri", "") or getattr(doc, "id", ""))
    return {
        "id": f"document:{getattr(doc, 'id', '')}",
        "title": title,
        "name": title,
        "text": content,
        "summary": content.strip() or title,
        "meta": meta,
    }


def _artifact_to_document_like(artifact: ArtifactInventoryEntry) -> Any:
    meta: dict[str, Any] = {
        "file_path": artifact.repo_path,
        "artifact_kind": artifact.artifact_kind,
        "parser_hint": artifact.parser_hint,
    }
    raw_data = artifact.raw_data if isinstance(artifact.raw_data, dict) else {}
    title = str(raw_data.get("title") or PurePosixPath(artifact.repo_path).name).strip()
    uri = str(raw_data.get("uri") or artifact.repo_path).strip()
    return type(
        "ArtifactDoc",
        (),
        {
            "id": artifact.artifact_id,
            "uri": uri,
            "title": title,
            "content_markdown": artifact.raw_text or "",
            "meta": meta,
        },
    )()


def artifact_inventory_to_recovery_docs(
    artifacts: list[ArtifactInventoryEntry],
) -> list[dict[str, Any]]:
    """Convert typed artifact inventory rows into legacy recovery docs."""

    docs: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.parser_hint not in {"markdown", "mdx", "rst", "plain_text"}:
            continue
        parsed = parse_artifact(artifact)
        document_like = _artifact_to_document_like(artifact)
        if parsed.parser_name == "markdown_adr":
            affected_entity_ids = parsed.structured_data.get("affected_entity_ids") or []
            if affected_entity_ids:
                document_like.meta["affected_entity_ids"] = affected_entity_ids
            document_like.meta["parser"] = parsed.parser_name
            document_like.meta["artifact_kind"] = "architecture_decision"
            title = str(parsed.structured_data.get("title") or document_like.title).strip()
            if title:
                document_like.title = title
            docs.append(
                {
                    "id": artifact.artifact_id,
                    "title": document_like.title,
                    "name": document_like.title,
                    "text": artifact.raw_text or "",
                    "summary": str(
                        parsed.structured_data.get("decision")
                        or parsed.structured_data.get("summary")
                        or artifact.raw_text
                        or document_like.title
                    ).strip(),
                    "meta": dict(document_like.meta),
                    "structured_data": dict(parsed.structured_data),
                }
            )
            continue
        elif not _looks_like_architecture_doc(document_like):
            continue
        docs.append(_document_to_recovery_doc(document_like))
    return docs


def _scenario_document_refs(kg_nodes: list[KnowledgeNode]) -> tuple[set[UUID], set[str]]:
    document_ids: set[UUID] = set()
    document_uris: set[str] = set()
    for node in kg_nodes:
        if node.kind != KnowledgeNodeKind.FILE:
            continue
        meta = node.meta or {}
        raw_document_id = meta.get("document_id")
        if isinstance(raw_document_id, str) and raw_document_id.strip():
            with suppress(ValueError):
                document_ids.add(UUID(raw_document_id))
        uri = str(meta.get("uri") or node.natural_key or "").strip()
        if uri:
            document_uris.add(uri)
    return document_ids, document_uris


def _scenario_repo_artifacts(kg_nodes: list[KnowledgeNode]) -> list[ArtifactInventoryEntry]:
    artifacts: list[ArtifactInventoryEntry] = []
    for node in kg_nodes:
        if node.kind != KnowledgeNodeKind.FILE:
            continue
        meta = node.meta or {}
        repo_path = str(meta.get("file_path") or meta.get("uri") or node.natural_key or "").strip()
        if not repo_path:
            continue
        raw_text = str(
            meta.get("content_markdown") or meta.get("content") or meta.get("raw_text") or ""
        )
        artifact = ArtifactInventoryEntry(
            artifact_id=f"artifact:{repo_path}",
            artifact_kind="documentation",
            repo_path=repo_path,
            media_type="text/markdown"
            if PurePosixPath(repo_path).suffix.lower() in _TEXT_DOC_SUFFIXES
            else "text/plain",
            parser_hint="markdown"
            if PurePosixPath(repo_path).suffix.lower() in {".md", ".mdx"}
            else "rst"
            if PurePosixPath(repo_path).suffix.lower() == ".rst"
            else "plain_text",
            raw_text=raw_text,
            raw_data=None,
            evidence=(),
        )
        artifacts.append(artifact)
    return sorted(artifacts, key=lambda row: row.repo_path)


async def load_recovery_docs(
    session: AsyncSession,
    kg_nodes: list[KnowledgeNode],
) -> list[dict[str, Any]]:
    """Load ADR-like document payloads using typed artifact inventory inputs."""
    document_ids, document_uris = _scenario_document_refs(kg_nodes)
    repo_artifacts = _scenario_repo_artifacts(kg_nodes)

    filters = []
    if document_ids:
        filters.append(Document.id.in_(document_ids))
    if document_uris:
        filters.append(Document.uri.in_(document_uris))

    docs = []
    if filters:
        docs = (await session.execute(select(Document).where(or_(*filters)))).scalars().all()

    document_artifacts = build_document_artifact_inventory(docs)
    merged_artifacts = merge_document_and_repo_artifacts(document_artifacts, repo_artifacts)
    return artifact_inventory_to_recovery_docs(merged_artifacts)
