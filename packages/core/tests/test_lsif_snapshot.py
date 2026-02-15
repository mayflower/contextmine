"""Tests for LSIF semantic snapshot provider."""

from __future__ import annotations

from pathlib import Path

from contextmine_core.semantic_snapshot.lsif import build_snapshot_lsif

MINIMAL_LSIF = """
{"id":1,"type":"vertex","label":"document","uri":"file:///repo/src/app.ts"}
{"id":2,"type":"vertex","label":"range","start":{"line":0,"character":0},"end":{"line":0,"character":5}}
{"id":3,"type":"vertex","label":"resultSet"}
{"id":4,"type":"vertex","label":"definitionResult"}
{"id":5,"type":"vertex","label":"referenceResult"}
{"id":6,"type":"edge","label":"contains","outV":1,"inVs":[2]}
{"id":7,"type":"edge","label":"next","outV":2,"inV":3}
{"id":8,"type":"edge","label":"textDocument/definition","outV":2,"inV":4}
{"id":9,"type":"edge","label":"textDocument/references","outV":2,"inV":5}
{"id":10,"type":"edge","label":"item","outV":4,"inVs":[2],"property":"definitions"}
{"id":11,"type":"edge","label":"item","outV":5,"inVs":[2],"property":"references"}
""".strip()


def test_build_snapshot_lsif(tmp_path: Path) -> None:
    lsif_file = tmp_path / "index.lsif"
    lsif_file.write_text(MINIMAL_LSIF, encoding="utf-8")

    snapshot = build_snapshot_lsif(lsif_file)

    assert len(snapshot.files) == 1
    assert snapshot.files[0].path.endswith("src/app.ts")
    assert len(snapshot.occurrences) >= 1
    assert snapshot.meta.get("provider") == "lsif"
