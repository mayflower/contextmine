"""Data models for real code metrics extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileMetricRecord:
    """Canonical file-level metric record used for Twin/City."""

    file_path: str
    language: str
    loc: int
    complexity: float
    coupling_in: int
    coupling_out: int
    coupling: float
    coverage: float | None = None
    sources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "loc": self.loc,
            "complexity": self.complexity,
            "coupling_in": self.coupling_in,
            "coupling_out": self.coupling_out,
            "coupling": self.coupling,
            "coverage": self.coverage,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FileMetricRecord:
        return cls(
            file_path=str(payload["file_path"]),
            language=str(payload.get("language", "unknown")),
            loc=int(payload["loc"]),
            complexity=float(payload["complexity"]),
            coupling_in=int(payload["coupling_in"]),
            coupling_out=int(payload["coupling_out"]),
            coupling=float(payload["coupling"]),
            coverage=(
                float(payload["coverage"])
                if payload.get("coverage") is not None
                else None
            ),
            sources=dict(payload.get("sources", {})),
        )


@dataclass
class ProjectMetricBundle:
    """Aggregated metric records for one project root."""

    project_root: str
    language: str
    files: list[FileMetricRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "language": self.language,
            "files": [record.to_dict() for record in self.files],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectMetricBundle:
        return cls(
            project_root=str(payload["project_root"]),
            language=str(payload.get("language", "unknown")),
            files=[FileMetricRecord.from_dict(item) for item in payload.get("files", [])],
        )


class MetricsGateError(RuntimeError):
    """Raised when strict metrics gate fails for a project or source."""

    def __init__(self, code: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(f"METRICS_GATE_FAILED: {code}")

    def to_payload(self) -> dict[str, Any]:
        return {"code": self.code, "details": self.details}
