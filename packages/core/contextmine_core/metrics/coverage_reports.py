"""Coverage report parsers and protocol detection for polyglot projects."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextmine_core.metrics.discovery import to_repo_relative_path

PROTOCOL_LCOV = "lcov"
PROTOCOL_COBERTURA = "cobertura_xml"
PROTOCOL_JACOCO = "jacoco_xml"
PROTOCOL_CLOVER = "clover_xml"
PROTOCOL_OPENCOVER = "opencover_xml"
PROTOCOL_GENERIC_JSON = "generic_file_coverage_json"


@dataclass
class CoverageAggregate:
    """Aggregated coverage for a single file."""

    sum_coverage: float = 0.0
    sample_count: int = 0
    reports: set[str] = None

    def __post_init__(self) -> None:
        if self.reports is None:
            self.reports = set()

    def add(self, coverage: float, report: Path) -> None:
        clamped = max(0.0, min(100.0, float(coverage)))
        self.sum_coverage += clamped
        self.sample_count += 1
        self.reports.add(str(report))

    def avg(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.sum_coverage / self.sample_count


def _tag_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _iter_nodes(root: ET.Element, local_name: str) -> Iterable[ET.Element]:
    for node in root.iter():
        if _tag_local_name(node.tag) == local_name:
            yield node


def _coverage_from_counts(covered: int, total: int) -> float | None:
    if total <= 0:
        return None
    return (covered / total) * 100.0


def parse_lcov_report(
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> dict[str, float]:
    """Parse `lcov.info` into file->coverage percentage."""
    per_file: dict[str, list[int]] = {}
    current_file: str | None = None
    covered = 0
    total = 0

    def flush() -> None:
        nonlocal current_file, covered, total
        if current_file:
            pct = _coverage_from_counts(covered, total)
            if pct is not None:
                per_file[current_file] = [covered, total]
        current_file = None
        covered = 0
        total = 0

    for raw_line in report_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.startswith("SF:"):
            flush()
            raw_file = line[3:]
            current_file = to_repo_relative_path(
                raw_file,
                repo_root=repo_root,
                project_root=project_root,
                base_dir=report_path.parent,
            )
            continue
        if line.startswith("DA:"):
            try:
                _, counts = line.split(":", 1)
                _, hits = counts.split(",", 1)
                total += 1
                if int(hits) > 0:
                    covered += 1
            except ValueError:
                continue
            continue
        if line == "end_of_record":
            flush()

    flush()

    result: dict[str, float] = {}
    for file_path, (covered_lines, total_lines) in per_file.items():
        pct = _coverage_from_counts(covered_lines, total_lines)
        if pct is not None:
            result[file_path] = pct
    return result


def parse_cobertura_xml(
    root: ET.Element,
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> dict[str, float]:
    """Parse Cobertura XML coverage into file->percentage."""
    result: dict[str, float] = {}

    for class_node in _iter_nodes(root, "class"):
        filename = class_node.attrib.get("filename")
        if not filename:
            continue

        file_path = to_repo_relative_path(
            filename,
            repo_root=repo_root,
            project_root=project_root,
            base_dir=report_path.parent,
        )
        if not file_path:
            continue

        line_nodes = [node for node in _iter_nodes(class_node, "line")]
        covered = 0
        total = 0
        for line_node in line_nodes:
            hits_raw = line_node.attrib.get("hits")
            if hits_raw is None:
                continue
            total += 1
            try:
                if int(hits_raw) > 0:
                    covered += 1
            except ValueError:
                continue

        if total > 0:
            result[file_path] = (covered / total) * 100.0
            continue

        line_rate_raw = class_node.attrib.get("line-rate")
        if line_rate_raw is None:
            continue

        try:
            result[file_path] = float(line_rate_raw) * 100.0
        except ValueError:
            continue

    return result


def parse_jacoco_xml(
    root: ET.Element,
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> dict[str, float]:
    """Parse JaCoCo XML coverage into file->percentage."""
    del report_path
    result: dict[str, float] = {}

    for package_node in _iter_nodes(root, "package"):
        package_name = package_node.attrib.get("name", "").strip("/")

        for source_node in _iter_nodes(package_node, "sourcefile"):
            source_name = source_node.attrib.get("name")
            if not source_name:
                continue

            relative = f"{package_name}/{source_name}" if package_name else source_name
            file_path = to_repo_relative_path(
                relative,
                repo_root=repo_root,
                project_root=project_root,
            )
            if not file_path:
                continue

            covered = 0
            missed = 0
            for counter_node in _iter_nodes(source_node, "counter"):
                if counter_node.attrib.get("type") != "LINE":
                    continue
                try:
                    covered = int(counter_node.attrib.get("covered", "0"))
                    missed = int(counter_node.attrib.get("missed", "0"))
                except ValueError:
                    covered = 0
                    missed = 0
                break

            pct = _coverage_from_counts(covered, covered + missed)
            if pct is not None:
                result[file_path] = pct

    return result


def parse_clover_xml(
    root: ET.Element,
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> dict[str, float]:
    """Parse Clover/PHPUnit XML coverage into file->percentage."""
    result: dict[str, float] = {}

    for file_node in _iter_nodes(root, "file"):
        raw_name = file_node.attrib.get("name")
        if not raw_name:
            continue

        file_path = to_repo_relative_path(
            raw_name,
            repo_root=repo_root,
            project_root=project_root,
            base_dir=report_path.parent,
        )
        if not file_path:
            continue

        covered = 0
        total = 0
        line_nodes = [node for node in _iter_nodes(file_node, "line")]
        for line_node in line_nodes:
            count_raw = line_node.attrib.get("count")
            if count_raw is None:
                continue
            total += 1
            try:
                if int(count_raw) > 0:
                    covered += 1
            except ValueError:
                continue

        if total > 0:
            result[file_path] = (covered / total) * 100.0
            continue

        metric_node = next(_iter_nodes(file_node, "metrics"), None)
        if metric_node is not None:
            try:
                covered_total = (
                    int(metric_node.attrib.get("coveredstatements", "0"))
                    + int(metric_node.attrib.get("coveredconditionals", "0"))
                    + int(metric_node.attrib.get("coveredmethods", "0"))
                )
                total_all = (
                    int(metric_node.attrib.get("statements", "0"))
                    + int(metric_node.attrib.get("conditionals", "0"))
                    + int(metric_node.attrib.get("methods", "0"))
                )
            except ValueError:
                covered_total = 0
                total_all = 0

            pct = _coverage_from_counts(covered_total, total_all)
            if pct is not None:
                result[file_path] = pct

    return result


def parse_opencover_xml(
    root: ET.Element,
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> dict[str, float]:
    """Parse OpenCover XML coverage into file->percentage."""
    file_id_to_path: dict[str, str] = {}
    for file_node in _iter_nodes(root, "File"):
        file_id = file_node.attrib.get("uid")
        full_path = file_node.attrib.get("fullPath")
        if not file_id or not full_path:
            continue
        normalized = to_repo_relative_path(
            full_path,
            repo_root=repo_root,
            project_root=project_root,
            base_dir=report_path.parent,
        )
        if normalized:
            file_id_to_path[file_id] = normalized

    coverage_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for seq_point in _iter_nodes(root, "SequencePoint"):
        file_id = seq_point.attrib.get("fileid") or seq_point.attrib.get("fileId")
        visits_raw = seq_point.attrib.get("vc") or seq_point.attrib.get("visitcount")
        if not file_id or visits_raw is None:
            continue

        file_path = file_id_to_path.get(file_id)
        if not file_path:
            continue

        counts = coverage_counts[file_path]
        counts[1] += 1
        try:
            if int(visits_raw) > 0:
                counts[0] += 1
        except ValueError:
            continue

    result: dict[str, float] = {}
    for file_path, (covered, total) in coverage_counts.items():
        pct = _coverage_from_counts(covered, total)
        if pct is not None:
            result[file_path] = pct
    return result


def parse_generic_file_coverage_json(
    payload: dict[str, Any],
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> dict[str, float]:
    """Parse generic-file-coverage-v1 into file->percentage."""
    if payload.get("schema") != "generic-file-coverage-v1":
        return {}

    files = payload.get("files")
    if not isinstance(files, list):
        return {}

    result: dict[str, float] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            continue
        try:
            coverage = float(entry.get("coverage"))
        except (TypeError, ValueError):
            continue

        file_path = to_repo_relative_path(
            raw_path,
            repo_root=repo_root,
            project_root=project_root,
            base_dir=report_path.parent,
        )
        if not file_path:
            continue
        result[file_path] = max(0.0, min(100.0, coverage))

    return result


def detect_coverage_protocol(report_path: Path) -> str | None:
    """Detect coverage report protocol by filename/signatures."""
    lower_name = report_path.name.lower()

    if lower_name.endswith(".info"):
        return PROTOCOL_LCOV

    if lower_name.endswith(".json"):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("schema") == "generic-file-coverage-v1":
            return PROTOCOL_GENERIC_JSON

    try:
        root = ET.parse(report_path).getroot()
    except ET.ParseError:
        return None

    root_tag = _tag_local_name(root.tag).lower()
    all_tags = {_tag_local_name(node.tag) for node in root.iter()}

    if root_tag == "coveragesession":
        return PROTOCOL_OPENCOVER

    if root_tag == "report" and "sourcefile" in all_tags:
        return PROTOCOL_JACOCO

    if "class" in all_tags and "line" in all_tags:
        return PROTOCOL_COBERTURA

    if "file" in all_tags and ("metrics" in all_tags or root_tag in {"coverage", "project"}):
        return PROTOCOL_CLOVER

    return None


def parse_coverage_report(
    report_path: Path,
    repo_root: Path,
    project_root: Path,
) -> tuple[str | None, dict[str, float]]:
    """Parse one coverage report into protocol + file->coverage percentage."""
    protocol = detect_coverage_protocol(report_path)
    if protocol == PROTOCOL_LCOV:
        return protocol, parse_lcov_report(report_path, repo_root, project_root)

    if protocol == PROTOCOL_GENERIC_JSON:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return protocol, {}
        return protocol, parse_generic_file_coverage_json(
            payload=payload,
            report_path=report_path,
            repo_root=repo_root,
            project_root=project_root,
        )

    if protocol is None:
        return None, {}

    try:
        root = ET.parse(report_path).getroot()
    except ET.ParseError:
        return protocol, {}

    if protocol == PROTOCOL_JACOCO:
        return protocol, parse_jacoco_xml(root, report_path, repo_root, project_root)
    if protocol == PROTOCOL_COBERTURA:
        return protocol, parse_cobertura_xml(root, report_path, repo_root, project_root)
    if protocol == PROTOCOL_CLOVER:
        return protocol, parse_clover_xml(root, report_path, repo_root, project_root)
    if protocol == PROTOCOL_OPENCOVER:
        return protocol, parse_opencover_xml(root, report_path, repo_root, project_root)

    return protocol, {}


def parse_coverage_reports(
    report_paths: list[Path],
    repo_root: Path,
    project_root: Path,
) -> tuple[dict[str, float], dict[str, dict[str, object]], dict[str, str]]:
    """Parse and aggregate multiple coverage reports.

    Returns:
        tuple(file->coverage_percent, file->provenance, report->protocol)
    """
    aggregates: dict[str, CoverageAggregate] = defaultdict(CoverageAggregate)
    protocol_by_report: dict[str, str] = {}

    for report_path in report_paths:
        protocol, parsed = parse_coverage_report(
            report_path, repo_root=repo_root, project_root=project_root
        )
        if protocol:
            protocol_by_report[str(report_path)] = protocol
        for file_path, coverage in parsed.items():
            aggregates[file_path].add(coverage, report_path)

    coverage_map: dict[str, float] = {}
    provenance_map: dict[str, dict[str, object]] = {}

    for file_path, aggregate in aggregates.items():
        coverage_map[file_path] = aggregate.avg()
        provenance_map[file_path] = {
            "reports": sorted(aggregate.reports),
            "samples": aggregate.sample_count,
        }

    return coverage_map, provenance_map, protocol_by_report
