"""Shared architecture grouping and file-path canonicalization helpers.

Centralises the domain/container/component derivation logic that was previously
duplicated across projections, mermaid_c4, codecharta, facts, and evolution.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any


def _arch_group_from_meta(meta: dict[str, Any]) -> tuple[str, str, str] | None:
    """Try to resolve arch group from explicit architecture metadata."""
    architecture_meta = meta.get("architecture")
    if not isinstance(architecture_meta, dict):
        return None
    explicit_domain = str(architecture_meta.get("domain") or "").strip()
    explicit_container = str(architecture_meta.get("container") or "").strip()
    if not explicit_domain or not explicit_container:
        return None
    explicit_component = str(architecture_meta.get("component") or "").strip()
    return explicit_domain, explicit_container, explicit_component or explicit_container


def _arch_group_from_path(path: str) -> tuple[str, str, str] | None:
    """Derive arch group from file path heuristics."""
    normalized = path.strip("/")
    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return None

    if parts[0] == "services" and len(parts) >= 3:
        domain, container = parts[1], parts[2]
    elif parts[0] == "apps" and len(parts) >= 2:
        domain, container = parts[1], parts[1]
    else:
        domain = parts[0]
        container = parts[1] if len(parts) > 1 else parts[0]

    component = PurePosixPath(normalized).stem or container
    return domain, container, component


def derive_arch_group(
    path: str | None, meta: dict[str, Any] | None = None
) -> tuple[str, str, str] | None:
    """Resolve (domain, container, component) from architecture meta or path heuristics.

    Returns ``None`` when the group cannot be determined.
    """
    result = _arch_group_from_meta(meta or {})
    if result is not None:
        return result
    if not path:
        return None
    return _arch_group_from_path(path)


def canonical_file_path_from_node(node: dict[str, Any]) -> str | None:
    """Extract a canonical file path from a graph node dict.

    Handles both ``file:`` natural-key prefixes and ``meta.file_path`` fallback.
    """
    kind = str(node.get("kind") or "").lower()
    natural_key = str(node.get("natural_key") or "")
    if kind == "file" and natural_key.startswith("file:"):
        value = natural_key.split(":", 1)[1].strip()
        return value or None

    meta = node.get("meta") or {}
    file_path = meta.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None


def canonical_file_path_from_meta(
    path: str | None, meta: dict[str, Any] | None = None
) -> str | None:
    """Resolve a canonical file path from a raw path string or meta dict.

    Used in contexts where you already have a separate ``path`` value (e.g.
    evidence rows) rather than a full graph node dict.
    """
    if path:
        cleaned = str(path).strip()
        if cleaned:
            return cleaned
    payload = meta or {}
    file_path = payload.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return file_path.strip()
    return None
