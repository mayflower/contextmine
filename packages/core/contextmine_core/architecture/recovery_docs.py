"""Helpers for loading ADR-like documents into architecture recovery."""

from __future__ import annotations

from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from contextmine_core.models import Document, KnowledgeNode, KnowledgeNodeKind
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

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


async def load_recovery_docs(
    session: AsyncSession,
    kg_nodes: list[KnowledgeNode],
) -> list[dict[str, Any]]:
    """Load ADR-like document payloads that correspond to scenario FILE nodes."""
    document_ids, document_uris = _scenario_document_refs(kg_nodes)
    if not document_ids and not document_uris:
        return []

    filters = []
    if document_ids:
        filters.append(Document.id.in_(document_ids))
    if document_uris:
        filters.append(Document.uri.in_(document_uris))

    docs = (await session.execute(select(Document).where(or_(*filters)))).scalars().all()

    recovery_docs = [
        _document_to_recovery_doc(doc)
        for doc in sorted(
            docs, key=lambda row: str(getattr(row, "uri", "") or getattr(row, "id", ""))
        )
        if _looks_like_architecture_doc(doc)
    ]
    return recovery_docs
