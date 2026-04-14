"""Shared architecture grouping and file-path canonicalization helpers.

Centralises the domain/container/component derivation logic that was previously
duplicated across projections, mermaid_c4, codecharta, facts, and evolution.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

_GENERIC_ARCH_SEGMENTS = {
    ".github",
    ".vscode",
    "__pycache__",
    "alembic",
    "assets",
    "build",
    "ci",
    "config",
    "configs",
    "coverage",
    "dist",
    "doc",
    "docs",
    "documentation",
    "example",
    "examples",
    "fixture",
    "fixtures",
    "infra",
    "lib",
    "libs",
    "migration",
    "migrations",
    "node_modules",
    "ops",
    "out",
    "package",
    "packages",
    "public",
    "sample",
    "samples",
    "script",
    "scripts",
    "shared",
    "spec",
    "specs",
    "src",
    "static",
    "support",
    "target",
    "temp",
    "test",
    "tests",
    "tmp",
    "tool",
    "tooling",
    "tools",
    "vendor",
}


def _is_generic_arch_segment(segment: str) -> bool:
    normalized = segment.strip().lower()
    return not normalized or normalized in _GENERIC_ARCH_SEGMENTS


def _component_from_path(normalized: str, fallback: str) -> str:
    component = PurePosixPath(normalized).stem or fallback
    return fallback if _is_generic_arch_segment(component) else component


def _has_path_suffix(segment: str) -> bool:
    return bool(PurePosixPath(segment).suffix)


def _scan_path_pair(parts: list[str]) -> tuple[str, str] | None:
    for index, segment in enumerate(parts):
        if _is_generic_arch_segment(segment) or _has_path_suffix(segment):
            continue
        if segment == "services" and index + 2 < len(parts):
            domain = parts[index + 1]
            container = parts[index + 2]
            if not _is_generic_arch_segment(domain) and not _has_path_suffix(domain):
                if not _is_generic_arch_segment(container) and not _has_path_suffix(container):
                    return domain, container
                # Preserve a usable group for short service paths like
                # ``services/billing/a.py`` where the file sits directly under the domain.
                if _has_path_suffix(container):
                    return domain, domain
            continue
        if segment == "apps" and index + 1 < len(parts):
            app_name = parts[index + 1]
            if not _is_generic_arch_segment(app_name) and not _has_path_suffix(app_name):
                return app_name, app_name
            continue
        if index + 1 >= len(parts):
            break
        next_segment = parts[index + 1]
        if _is_generic_arch_segment(next_segment) or _has_path_suffix(next_segment):
            continue
        return segment, next_segment
    return None


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

    if len(parts) < 3:
        return None

    pair = _scan_path_pair(parts)
    if pair is None:
        return None
    domain, container = pair

    if _is_generic_arch_segment(domain) or _is_generic_arch_segment(container):
        return None

    component = _component_from_path(normalized, container)
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
