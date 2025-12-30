"""Tests for arc42 Architecture Twin generator."""

from datetime import datetime

from contextmine_core.analyzer.arc42 import (
    Arc42Document,
    Arc42Section,
    DriftItem,
    DriftReport,
)


class TestArc42Section:
    """Tests for Arc42Section dataclass."""

    def test_create_section(self) -> None:
        """Test creating a section."""
        section = Arc42Section(
            id="context",
            title="1. Context",
            content="System boundary content",
            evidence_count=5,
        )
        assert section.id == "context"
        assert section.title == "1. Context"
        assert section.evidence_count == 5

    def test_section_default_evidence_count(self) -> None:
        """Test section with default evidence count."""
        section = Arc42Section(
            id="glossary",
            title="7. Glossary",
            content="Domain terms",
        )
        assert section.evidence_count == 0


class TestArc42Document:
    """Tests for Arc42Document dataclass."""

    def test_create_empty_document(self) -> None:
        """Test creating an empty document."""
        doc = Arc42Document()
        assert doc.sections == []
        assert isinstance(doc.generated_at, datetime)

    def test_to_markdown(self) -> None:
        """Test markdown generation."""
        doc = Arc42Document()
        doc.sections = [
            Arc42Section(
                id="context",
                title="1. Context",
                content="This is the context section.",
                evidence_count=3,
            ),
            Arc42Section(
                id="building-blocks",
                title="2. Building Blocks",
                content="Components listed here.",
                evidence_count=0,
            ),
        ]

        md = doc.to_markdown()

        assert "# Architecture Documentation (arc42)" in md
        assert "## 1. Context" in md
        assert "This is the context section" in md
        assert "*Based on 3 extracted facts*" in md
        assert "## 2. Building Blocks" in md

    def test_get_section(self) -> None:
        """Test getting a specific section."""
        doc = Arc42Document()
        context = Arc42Section(id="context", title="1. Context", content="...")
        runtime = Arc42Section(id="runtime", title="3. Runtime", content="...")
        doc.sections = [context, runtime]

        assert doc.get_section("context") == context
        assert doc.get_section("runtime") == runtime
        assert doc.get_section("nonexistent") is None


class TestDriftReport:
    """Tests for DriftReport dataclass."""

    def test_empty_report(self) -> None:
        """Test empty drift report."""
        report = DriftReport()
        md = report.to_markdown()

        assert "# Architecture Drift Report" in md
        assert "No changes detected" in md

    def test_report_with_items(self) -> None:
        """Test drift report with items."""
        report = DriftReport(
            items=[
                DriftItem(
                    kind="added",
                    category="API",
                    name="New Endpoint",
                    details="POST /api/users",
                ),
                DriftItem(
                    kind="removed",
                    category="Database",
                    name="Old Table",
                    details="Deprecated table removed",
                ),
                DriftItem(
                    kind="changed",
                    category="Job",
                    name="Sync Job",
                    details="Schedule changed from daily to hourly",
                ),
            ]
        )

        md = report.to_markdown()

        assert "## Added" in md
        assert "New Endpoint" in md
        assert "## Removed" in md
        assert "Old Table" in md
        assert "## Changed" in md
        assert "Sync Job" in md
        assert "1 added, 1 removed, 1 changed" in md


class TestDriftItem:
    """Tests for DriftItem dataclass."""

    def test_create_drift_item(self) -> None:
        """Test creating a drift item."""
        item = DriftItem(
            kind="added",
            category="Endpoint",
            name="POST /api/test",
            details="New test endpoint",
        )
        assert item.kind == "added"
        assert item.category == "Endpoint"
        assert item.name == "POST /api/test"


class TestArc42Sections:
    """Tests for arc42 section structure."""

    def test_all_section_ids(self) -> None:
        """Test that all expected section IDs are valid."""
        expected_ids = [
            "context",
            "building-blocks",
            "runtime",
            "deployment",
            "crosscutting",
            "risks",
            "glossary",
        ]

        doc = Arc42Document()
        for section_id in expected_ids:
            doc.sections.append(
                Arc42Section(
                    id=section_id,
                    title=section_id.replace("-", " ").title(),
                    content=f"Content for {section_id}",
                )
            )

        assert len(doc.sections) == 7
        for section_id in expected_ids:
            assert doc.get_section(section_id) is not None

    def test_markdown_has_separator(self) -> None:
        """Test that markdown has section separators."""
        doc = Arc42Document()
        doc.sections = [
            Arc42Section(id="context", title="Context", content="..."),
            Arc42Section(id="blocks", title="Blocks", content="..."),
        ]

        md = doc.to_markdown()
        assert md.count("---") >= 2  # At least header separator and between sections
