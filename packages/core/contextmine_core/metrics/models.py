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
    cohesion: float = 1.0
    instability: float = 0.0
    fan_in: int = 0
    fan_out: int = 0
    cycle_participation: bool = False
    cycle_size: int = 0
    duplication_ratio: float = 0.0
    crap_score: float | None = None
    change_frequency: float = 0.0
    churn: float = 0.0
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
            "cohesion": self.cohesion,
            "instability": self.instability,
            "fan_in": self.fan_in,
            "fan_out": self.fan_out,
            "cycle_participation": self.cycle_participation,
            "cycle_size": self.cycle_size,
            "duplication_ratio": self.duplication_ratio,
            "crap_score": self.crap_score,
            "change_frequency": self.change_frequency,
            "churn": self.churn,
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
            cohesion=float(payload.get("cohesion", 1.0) or 1.0),
            instability=float(payload.get("instability", 0.0) or 0.0),
            fan_in=int(payload.get("fan_in", 0) or 0),
            fan_out=int(payload.get("fan_out", 0) or 0),
            cycle_participation=bool(payload.get("cycle_participation", False)),
            cycle_size=int(payload.get("cycle_size", 0) or 0),
            duplication_ratio=float(payload.get("duplication_ratio", 0.0) or 0.0),
            crap_score=(
                float(payload["crap_score"]) if payload.get("crap_score") is not None else None
            ),
            change_frequency=float(payload.get("change_frequency", 0.0) or 0.0),
            churn=float(payload.get("churn", 0.0) or 0.0),
            coverage=(float(payload["coverage"]) if payload.get("coverage") is not None else None),
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
