"""Unit tests for the real polyglot metrics pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextmine_core.metrics.complexity_loc_lizard import aggregate_lizard_metrics
from contextmine_core.metrics.coupling_from_snapshot import compute_file_coupling_from_snapshots
from contextmine_core.metrics.coverage_reports import parse_coverage_reports
from contextmine_core.metrics.discovery import to_repo_relative_path
from contextmine_core.metrics.models import MetricsGateError
from contextmine_core.metrics.pipeline import run_polyglot_metrics_pipeline
from contextmine_core.semantic_snapshot.models import (
    FileInfo,
    Range,
    Relation,
    RelationKind,
    Snapshot,
    Symbol,
    SymbolKind,
)


def test_parse_lcov_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.py").write_text("print('x')\n", encoding="utf-8")
    report = repo_root / "lcov.info"
    report.write_text(
        "\n".join(
            [
                "TN:",
                "SF:src/main.py",
                "DA:1,1",
                "DA:2,0",
                "end_of_record",
            ]
        ),
        encoding="utf-8",
    )

    coverage, _, _ = parse_coverage_reports([report], repo_root=repo_root, project_root=repo_root)
    assert coverage["src/main.py"] == pytest.approx(50.0)


def test_parse_cobertura_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.py").write_text("print('x')\n", encoding="utf-8")
    report = repo_root / "coverage.xml"
    report.write_text(
        """<coverage>
  <packages>
    <package name="src">
      <classes>
        <class filename="src/main.py" line-rate="0.5">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""",
        encoding="utf-8",
    )

    coverage, _, _ = parse_coverage_reports([report], repo_root=repo_root, project_root=repo_root)

    assert coverage["src/main.py"] == pytest.approx(50.0)


def test_parse_jacoco_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "com" / "example").mkdir(parents=True)
    (repo_root / "com" / "example" / "App.java").write_text("class App {}\n", encoding="utf-8")
    report = repo_root / "jacoco.xml"
    report.write_text(
        """<report name="demo">
  <package name="com/example">
    <sourcefile name="App.java">
      <counter type="LINE" missed="3" covered="7"/>
    </sourcefile>
  </package>
</report>""",
        encoding="utf-8",
    )

    coverage, _, _ = parse_coverage_reports([report], repo_root=repo_root, project_root=repo_root)

    assert coverage["com/example/App.java"] == pytest.approx(70.0)


def test_parse_clover_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.php").write_text("<?php echo 'x';\n", encoding="utf-8")
    report = repo_root / "clover.xml"
    report.write_text(
        """<coverage>
  <project>
    <file name="src/main.php">
      <line num="1" count="1" type="stmt"/>
      <line num="2" count="0" type="stmt"/>
    </file>
  </project>
</coverage>""",
        encoding="utf-8",
    )

    coverage, _, _ = parse_coverage_reports([report], repo_root=repo_root, project_root=repo_root)


def test_parse_opencover_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.cs").write_text("class App {}\n", encoding="utf-8")
    report = repo_root / "coverage.opencover.xml"
    report.write_text(
        """<CoverageSession>
  <Modules>
    <Module>
      <Files>
        <File uid="1" fullPath="src/main.cs" />
      </Files>
      <Classes>
        <Class>
          <Methods>
            <Method>
              <SequencePoints>
                <SequencePoint vc="1" fileid="1" />
                <SequencePoint vc="0" fileid="1" />
              </SequencePoints>
            </Method>
          </Methods>
        </Class>
      </Classes>
    </Module>
  </Modules>
</CoverageSession>""",
        encoding="utf-8",
    )

    coverage, _, _ = parse_coverage_reports([report], repo_root=repo_root, project_root=repo_root)
    assert coverage["src/main.cs"] == pytest.approx(50.0)


def test_parse_generic_json_coverage(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.py").write_text("print('x')\n", encoding="utf-8")
    report = repo_root / "coverage.json"
    report.write_text(
        """{
  "schema": "generic-file-coverage-v1",
  "files": [
    {"path": "src/main.py", "coverage": 87.5}
  ]
}""",
        encoding="utf-8",
    )

    coverage, _, _ = parse_coverage_reports([report], repo_root=repo_root, project_root=repo_root)
    assert coverage["src/main.py"] == pytest.approx(87.5)


def test_aggregate_lizard_metrics() -> None:
    class FakeFunction:
        def __init__(self, cyclomatic_complexity: int) -> None:
            self.cyclomatic_complexity = cyclomatic_complexity

    class FakeFileInfo:
        def __init__(self, filename: str, nloc: int, cc_values: list[int]) -> None:
            self.filename = filename
            self.nloc = nloc
            self.function_list = [FakeFunction(value) for value in cc_values]

    repo_root = Path("/repo")
    project_root = Path("/repo")
    file_infos = [FakeFileInfo("/repo/src/main.py", 42, [3, 4, 5])]

    metrics = aggregate_lizard_metrics(
        file_infos=file_infos,
        repo_root=repo_root,
        project_root=project_root,
        relevant_files={"src/main.py"},
    )

    assert metrics["src/main.py"]["loc"] == 42
    assert metrics["src/main.py"]["complexity"] == pytest.approx(12.0)


def test_compute_file_coupling_bidirectional() -> None:
    snapshot = Snapshot(
        files=[],
        symbols=[
            Symbol("a", SymbolKind.FUNCTION, "src/a.py", Range(1, 0, 1, 1), "a"),
            Symbol("b", SymbolKind.FUNCTION, "src/b.py", Range(1, 0, 1, 1), "b"),
        ],
        occurrences=[],
        relations=[
            Relation(
                src_def_id="a",
                kind=RelationKind.CALLS,
                dst_def_id="b",
                resolved=True,
                weight=1.0,
                meta={},
            )
        ],
        meta={"project_root": "/repo"},
    ).to_dict()

    coupling, _ = compute_file_coupling_from_snapshots(
        snapshot_dicts=[snapshot],
        repo_root=Path("/repo"),
        project_root=Path("/repo"),
        relevant_files={"src/a.py", "src/b.py"},
    )

    assert coupling["src/a.py"]["coupling_out"] == 1
    assert coupling["src/a.py"]["coupling_in"] == 0
    assert coupling["src/b.py"]["coupling_in"] == 1
    assert coupling["src/b.py"]["coupling_out"] == 0


def test_coupling_mapping_incomplete_raises() -> None:
    snapshot = {
        "files": [],
        "symbols": [],
        "occurrences": [],
        "relations": [
            {
                "src_def_id": "a",
                "kind": RelationKind.CALLS.value,
                "dst_def_id": "b",
                "resolved": True,
                "weight": 1.0,
                "meta": {},
            }
        ],
        "meta": {"project_root": "/repo"},
    }

    with pytest.raises(MetricsGateError) as exc_info:
        compute_file_coupling_from_snapshots(
            snapshot_dicts=[snapshot],
            repo_root=Path("/repo"),
            project_root=Path("/repo"),
            relevant_files={"src/a.py"},
        )

    assert exc_info.value.code == "coupling_mapping_incomplete"


def test_strict_gate_missing_metric_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src").mkdir()
    (repo_root / "src" / "main.py").write_text("print('x')\n", encoding="utf-8")

    snapshot = Snapshot(
        files=[FileInfo("src/main.py", "python")],
        symbols=[Symbol("main", SymbolKind.FUNCTION, "src/main.py", Range(1, 0, 1, 1), "main")],
        occurrences=[],
        relations=[],
        meta={
            "project_root": str(repo_root),
            "repo_relative_root": "",
            "language": "python",
        },
    ).to_dict()

    project_dicts = [{"language": "python", "root_path": str(repo_root)}]

    import contextmine_core.metrics.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "extract_complexity_loc_metrics", lambda **_: {})
    monkeypatch.setattr(
        pipeline_mod,
        "compute_file_coupling_from_snapshots",
        lambda **_: (
            {"src/main.py": {"coupling_in": 0, "coupling_out": 0, "coupling": 0.0}},
            {},
        ),
    )

    with pytest.raises(MetricsGateError) as exc_info:
        run_polyglot_metrics_pipeline(
            repo_root=repo_root,
            project_dicts=project_dicts,
            snapshot_dicts=[snapshot],
            strict_mode=True,
            metrics_languages="python",
        )

    assert exc_info.value.code == "missing_required_metrics"


def test_to_repo_relative_path_handles_windows_style() -> None:
    repo_root = Path("/repo")
    normalized = to_repo_relative_path("src\\module\\main.py", repo_root=repo_root)
    assert normalized == "src/module/main.py"
