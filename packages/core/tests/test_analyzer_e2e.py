"""End-to-end tests for the Analyzer / Knowledge Graph subsystem.

These tests verify that all extractors work together and produce
well-formed outputs without requiring a database connection.
"""

import pytest

# Fixture: Small Python codebase for testing
FIXTURE_PYTHON_FILE = '''
"""User authentication module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """User entity."""
    id: int
    email: str
    age: int
    is_admin: bool = False


def validate_user(user: User) -> bool:
    """Validate user data."""
    if user.age < 18:
        raise ValueError("User must be at least 18 years old")

    if not user.email:
        raise ValueError("Email is required")

    return True


def authenticate(email: str, password: str) -> Optional[User]:
    """Authenticate user by email and password."""
    if not email or not password:
        return None

    # Mock authentication
    return User(id=1, email=email, age=25)


def delete_user(user: User, requester: User) -> bool:
    """Delete a user. Only admins can delete."""
    if not requester.is_admin:
        raise PermissionError("Only admins can delete users")

    return True
'''

FIXTURE_TYPESCRIPT_FILE = """
interface User {
    id: number;
    email: string;
    age: number;
}

function validateAge(age: number): void {
    if (age < 18) {
        throw new Error("Must be at least 18 years old");
    }
}

function processPayment(amount: number): boolean {
    if (amount <= 0) {
        throw new Error("Amount must be positive");
    }
    return true;
}
"""

FIXTURE_OPENAPI = """
openapi: "3.0.0"
info:
  title: Test API
  version: "1.0"
paths:
  /users:
    get:
      summary: List users
      responses:
        "200":
          description: Success
    post:
      summary: Create user
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/CreateUser"
      responses:
        "201":
          description: Created
  /users/{id}:
    get:
      summary: Get user
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
      responses:
        "200":
          description: Success
components:
  schemas:
    CreateUser:
      type: object
      properties:
        email:
          type: string
        age:
          type: integer
"""

FIXTURE_GITHUB_WORKFLOW = """
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint
        run: ruff check .
"""

FIXTURE_ALEMBIC_MIGRATION = '''
"""Create users table."""

from alembic import op
import sqlalchemy as sa


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("age", sa.Integer()),
        sa.Column("created_at", sa.DateTime()),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("total", sa.Numeric(10, 2)),
    )

    op.create_foreign_key(
        "fk_orders_user",
        "orders", "users",
        ["user_id"], ["id"],
    )


def downgrade():
    op.drop_table("orders")
    op.drop_table("users")
'''


class TestRuleExtractorE2E:
    """End-to-end tests for rule candidate extraction."""

    def test_extract_python_rules(self) -> None:
        """Test extracting rules from Python code."""
        from contextmine_core.analyzer.extractors.rules import (
            FailureKind,
            extract_rule_candidates,
        )

        result = extract_rule_candidates("auth.py", FIXTURE_PYTHON_FILE)

        # Should find multiple rule candidates
        assert len(result.candidates) >= 3

        # Check specific rules
        age_rule = next(
            (c for c in result.candidates if "age" in c.predicate_text.lower()),
            None,
        )
        assert age_rule is not None
        assert age_rule.failure_kind == FailureKind.RAISE_EXCEPTION
        assert "18" in age_rule.predicate_text or "18" in age_rule.failure_text

        admin_rule = next(
            (c for c in result.candidates if "admin" in c.predicate_text.lower()),
            None,
        )
        assert admin_rule is not None
        assert "PermissionError" in admin_rule.failure_text

    def test_extract_typescript_rules(self) -> None:
        """Test extracting rules from TypeScript code."""
        from contextmine_core.analyzer.extractors.rules import (
            FailureKind,
            extract_rule_candidates,
        )

        result = extract_rule_candidates("validation.ts", FIXTURE_TYPESCRIPT_FILE)

        assert len(result.candidates) >= 2

        # Check age validation
        age_rule = next(
            (c for c in result.candidates if "age" in c.predicate_text.lower()),
            None,
        )
        assert age_rule is not None
        assert age_rule.failure_kind == FailureKind.THROW_ERROR

        # Check payment validation
        payment_rule = next(
            (c for c in result.candidates if "amount" in c.predicate_text.lower()),
            None,
        )
        assert payment_rule is not None

    def test_rule_candidates_have_evidence(self) -> None:
        """Test that extracted rules have proper evidence."""
        from contextmine_core.analyzer.extractors.rules import extract_rule_candidates

        result = extract_rule_candidates("auth.py", FIXTURE_PYTHON_FILE)

        for candidate in result.candidates:
            assert candidate.start_line > 0
            assert candidate.end_line >= candidate.start_line
            assert candidate.predicate_text
            assert candidate.failure_text
            assert candidate.language == "python"


class TestSurfaceExtractorE2E:
    """End-to-end tests for surface catalog extraction."""

    def test_extract_openapi_endpoints(self) -> None:
        """Test extracting API endpoints from OpenAPI spec."""
        from contextmine_core.analyzer.extractors.openapi import extract_from_openapi

        result = extract_from_openapi("api.yaml", FIXTURE_OPENAPI)

        assert len(result.endpoints) >= 3

        # Check GET /users
        get_users = next(
            (e for e in result.endpoints if e.path == "/users" and e.method == "GET"),
            None,
        )
        assert get_users is not None
        assert get_users.summary == "List users"

        # Check POST /users
        post_users = next(
            (e for e in result.endpoints if e.path == "/users" and e.method == "POST"),
            None,
        )
        assert post_users is not None

        # Check schemas
        assert len(result.schemas) >= 1
        assert "CreateUser" in result.schemas

    def test_extract_github_jobs(self) -> None:
        """Test extracting jobs from GitHub Actions workflow."""
        from contextmine_core.analyzer.extractors.jobs import JobKind, extract_jobs

        result = extract_jobs(".github/workflows/ci.yml", FIXTURE_GITHUB_WORKFLOW)

        assert len(result.jobs) >= 2

        test_job = next((j for j in result.jobs if j.name == "test"), None)
        assert test_job is not None
        assert test_job.kind == JobKind.GITHUB_JOB

        lint_job = next((j for j in result.jobs if j.name == "lint"), None)
        assert lint_job is not None

    def test_surface_catalog_integration(self) -> None:
        """Test the unified surface catalog extractor."""
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        extractor = SurfaceCatalogExtractor()
        extractor.add_file("api.yaml", FIXTURE_OPENAPI)
        extractor.add_file(".github/workflows/ci.yml", FIXTURE_GITHUB_WORKFLOW)

        catalog = extractor.catalog

        # Should have OpenAPI specs with endpoints
        assert len(catalog.openapi_specs) >= 1
        total_endpoints = sum(len(spec.endpoints) for spec in catalog.openapi_specs)
        assert total_endpoints >= 3

        # Should have job definitions with jobs
        assert len(catalog.job_definitions) >= 1
        total_jobs = sum(len(job_def.jobs) for job_def in catalog.job_definitions)
        assert total_jobs >= 2


class TestERMExtractorE2E:
    """End-to-end tests for ERM extraction."""

    def test_extract_tables_from_migration(self) -> None:
        """Test extracting tables from Alembic migration."""
        from contextmine_core.analyzer.extractors.alembic import extract_from_alembic

        result = extract_from_alembic("001_create_users.py", FIXTURE_ALEMBIC_MIGRATION)

        assert len(result.tables) >= 2

        # Check users table
        users = next((t for t in result.tables if t.name == "users"), None)
        assert users is not None
        assert len(users.columns) >= 4

        email_col = next((c for c in users.columns if c.name == "email"), None)
        assert email_col is not None
        assert not email_col.nullable

        # Check orders table
        orders = next((t for t in result.tables if t.name == "orders"), None)
        assert orders is not None

    def test_extract_foreign_keys(self) -> None:
        """Test extracting foreign key relationships."""
        from contextmine_core.analyzer.extractors.alembic import extract_from_alembic

        result = extract_from_alembic("001_create_users.py", FIXTURE_ALEMBIC_MIGRATION)

        assert len(result.foreign_keys) >= 1

        fk = result.foreign_keys[0]
        assert fk.source_table == "orders"
        assert fk.target_table == "users"

    def test_generate_mermaid_erd(self) -> None:
        """Test generating Mermaid ERD from schema."""
        from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
        from contextmine_core.analyzer.extractors.erm import ERMExtractor, generate_mermaid_erd

        migration_result = extract_from_alembic("001.py", FIXTURE_ALEMBIC_MIGRATION)

        extractor = ERMExtractor()
        extractor.add_alembic_extraction(migration_result)

        mermaid = generate_mermaid_erd(extractor.schema)

        assert "erDiagram" in mermaid
        assert "users" in mermaid
        assert "orders" in mermaid
        assert "||--o{" in mermaid  # Relationship notation


class TestArc42GeneratorE2E:
    """End-to-end tests for arc42 generation."""

    def test_arc42_document_structure(self) -> None:
        """Test that arc42 document has all required sections."""
        from contextmine_core.analyzer.arc42 import Arc42Document, Arc42Section

        # Create a document manually (no DB needed)
        doc = Arc42Document()
        doc.sections = [
            Arc42Section(id="context", title="1. Context", content="Context content"),
            Arc42Section(id="building-blocks", title="2. Building Blocks", content="..."),
            Arc42Section(id="runtime", title="3. Runtime View", content="..."),
            Arc42Section(id="deployment", title="4. Deployment View", content="..."),
            Arc42Section(id="crosscutting", title="5. Crosscutting Concepts", content="..."),
            Arc42Section(id="risks", title="6. Risks & Technical Debt", content="..."),
            Arc42Section(id="glossary", title="7. Glossary", content="..."),
        ]

        md = doc.to_markdown()

        # Should have all sections
        assert "# Architecture Documentation (arc42)" in md
        assert "## 1. Context" in md
        assert "## 2. Building Blocks" in md
        assert "## 7. Glossary" in md

    def test_drift_report_output(self) -> None:
        """Test drift report formatting."""
        from contextmine_core.analyzer.arc42 import DriftItem, DriftReport

        report = DriftReport(
            items=[
                DriftItem(kind="added", category="API", name="POST /users", details="New endpoint"),
                DriftItem(
                    kind="removed", category="Table", name="legacy_users", details="Deprecated"
                ),
            ]
        )

        md = report.to_markdown()

        assert "# Architecture Drift Report" in md
        assert "## Added" in md
        assert "POST /users" in md
        assert "## Removed" in md
        assert "legacy_users" in md


class TestGraphRAGDataStructures:
    """Tests for GraphRAG data structures without DB."""

    def test_graphrag_result_structure(self) -> None:
        """Test GraphRAGResult can be created and serialized."""
        from contextmine_core.graphrag import Evidence, GraphEdge, GraphNode, GraphRAGResult

        result = GraphRAGResult(
            query="test query",
            nodes=[
                GraphNode(
                    id="n1",
                    kind="file",
                    name="auth.py",
                    natural_key="file:auth.py",
                    meta={"path": "/src/auth.py"},
                ),
                GraphNode(
                    id="n2",
                    kind="symbol",
                    name="authenticate",
                    natural_key="symbol:auth.py:authenticate",
                ),
            ],
            edges=[
                GraphEdge(
                    source_id="n1",
                    target_id="n2",
                    kind="file_defines_symbol",
                ),
            ],
            evidence=[
                Evidence(
                    file_path="/src/auth.py",
                    start_line=10,
                    end_line=20,
                    snippet="def authenticate(...):",
                ),
            ],
            summary_markdown="# Results\n\nFound authentication code.",
        )

        # Test serialization
        data = result.to_dict()
        assert data["query"] == "test query"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert len(data["evidence"]) == 1

        # Test JSON serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0


class TestLabelingSchemas:
    """Tests for LLM labeling schemas."""

    def test_business_rule_output_validation(self) -> None:
        """Test that BusinessRuleOutput validates correctly."""
        from contextmine_core.analyzer.labeling import BusinessRuleOutput

        # Valid rule
        rule = BusinessRuleOutput(
            name="Age Validation",
            description="Validates user age",
            category="validation",
            severity="error",
            natural_language="Users must be at least 18 years old",
            confidence=0.9,
            is_valid_rule=True,
            reasoning="Clear business requirement",
        )

        assert rule.name == "Age Validation"
        assert rule.is_valid_rule is True
        assert 0 <= rule.confidence <= 1

    def test_business_rule_output_rejects_invalid(self) -> None:
        """Test that invalid data is rejected."""
        from contextmine_core.analyzer.labeling import BusinessRuleOutput

        # Invalid confidence (too high)
        with pytest.raises(ValueError):
            BusinessRuleOutput(
                name="Test",
                description="Test",
                category="validation",
                severity="error",
                natural_language="Test",
                confidence=1.5,  # Invalid
                is_valid_rule=True,
                reasoning="Test",
            )


class TestIntegrationWorkflow:
    """Tests verifying the full extraction workflow."""

    def test_full_extraction_pipeline(self) -> None:
        """Test running all extractors on fixture data."""
        from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
        from contextmine_core.analyzer.extractors.erm import ERMExtractor, generate_mermaid_erd
        from contextmine_core.analyzer.extractors.rules import extract_rule_candidates
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        # 1. Extract rule candidates
        python_rules = extract_rule_candidates("auth.py", FIXTURE_PYTHON_FILE)
        ts_rules = extract_rule_candidates("validation.ts", FIXTURE_TYPESCRIPT_FILE)

        assert len(python_rules.candidates) >= 3
        assert len(ts_rules.candidates) >= 2

        # 2. Extract surface catalog
        surface_extractor = SurfaceCatalogExtractor()
        surface_extractor.add_file("api.yaml", FIXTURE_OPENAPI)
        surface_extractor.add_file(".github/workflows/ci.yml", FIXTURE_GITHUB_WORKFLOW)

        catalog = surface_extractor.catalog
        total_endpoints = sum(len(spec.endpoints) for spec in catalog.openapi_specs)
        total_jobs = sum(len(job_def.jobs) for job_def in catalog.job_definitions)
        assert total_endpoints >= 3
        assert total_jobs >= 2

        # 3. Extract ERM
        migration_result = extract_from_alembic("001.py", FIXTURE_ALEMBIC_MIGRATION)
        erm_extractor = ERMExtractor()
        erm_extractor.add_alembic_extraction(migration_result)

        schema = erm_extractor.schema
        assert len(schema.tables) >= 2

        mermaid = generate_mermaid_erd(schema)
        assert "erDiagram" in mermaid

        # 4. Verify all outputs are well-formed
        for candidate in python_rules.candidates + ts_rules.candidates:
            assert candidate.start_line > 0
            assert candidate.predicate_text
            assert candidate.failure_text

        for spec in catalog.openapi_specs:
            for endpoint in spec.endpoints:
                assert endpoint.path
                assert endpoint.method

        for job_def in catalog.job_definitions:
            for job in job_def.jobs:
                assert job.name
                assert job.kind

        for table in schema.tables.values():
            assert table.name
            assert len(table.columns) > 0

    def test_outputs_are_idempotent(self) -> None:
        """Test that extractors produce same output on repeated calls."""
        from contextmine_core.analyzer.extractors.rules import (
            extract_rule_candidates,
            get_candidate_natural_key,
        )

        # Run extraction twice
        result1 = extract_rule_candidates("auth.py", FIXTURE_PYTHON_FILE)
        result2 = extract_rule_candidates("auth.py", FIXTURE_PYTHON_FILE)

        # Same number of candidates
        assert len(result1.candidates) == len(result2.candidates)

        # Same natural keys (stable across runs)
        keys1 = {get_candidate_natural_key(c) for c in result1.candidates}
        keys2 = {get_candidate_natural_key(c) for c in result2.candidates}
        assert keys1 == keys2
