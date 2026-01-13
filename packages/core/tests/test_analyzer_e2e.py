"""End-to-end tests for the Analyzer / Knowledge Graph subsystem.

These tests verify that all extractors work together and produce
well-formed outputs without requiring a database connection.
"""

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
        assert test_job.framework == JobKind.GITHUB_JOB

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


class TestGraphRAGDataStructures:
    """Tests for GraphRAG data structures without DB."""

    def test_graphrag_result_structure(self) -> None:
        """Test ContextPack can be created and serialized."""
        from uuid import uuid4

        from contextmine_core.graphrag import (
            Citation,
            ContextPack,
            EdgeContext,
            EntityContext,
        )

        result = ContextPack(
            query="test query",
            entities=[
                EntityContext(
                    node_id=uuid4(),
                    kind="file",
                    name="auth.py",
                    natural_key="file:auth.py",
                ),
                EntityContext(
                    node_id=uuid4(),
                    kind="symbol",
                    name="authenticate",
                    natural_key="symbol:auth.py:authenticate",
                ),
            ],
            edges=[
                EdgeContext(
                    source_id="n1",
                    target_id="n2",
                    kind="file_defines_symbol",
                ),
            ],
            citations=[
                Citation(
                    file_path="/src/auth.py",
                    start_line=10,
                    end_line=20,
                    snippet="def authenticate(...):",
                ),
            ],
        )

        # Test serialization
        data = result.to_dict()
        assert data["query"] == "test query"
        assert len(data["entities"]) == 2
        assert len(data["edges"]) == 1
        assert len(data["citations"]) == 1

        # Test JSON serializable
        import json

        json_str = json.dumps(data)
        assert len(json_str) > 0


class TestLabelingSchemas:
    """Tests for LLM labeling schemas."""

    def test_business_rule_def_validation(self) -> None:
        """Test that BusinessRuleDef validates correctly."""
        from contextmine_core.analyzer.extractors.rules import BusinessRuleDef

        # Valid rule
        rule = BusinessRuleDef(
            name="Age Validation",
            description="Validates user age",
            category="validation",
            severity="error",
            natural_language="Users must be at least 18 years old",
            evidence_snippet="if user.age < 18: raise ValueError",
            start_line=10,
            end_line=12,
        )

        assert rule.name == "Age Validation"
        assert rule.category == "validation"

    def test_extraction_output_structure(self) -> None:
        """Test that ExtractionOutput can hold multiple rules."""
        from contextmine_core.analyzer.extractors.rules import (
            BusinessRuleDef,
            ExtractionOutput,
        )

        output = ExtractionOutput(
            rules=[
                BusinessRuleDef(
                    name="Rule 1",
                    description="First rule",
                    category="validation",
                    severity="error",
                    natural_language="Must be positive",
                    evidence_snippet="if x < 0: raise",
                    start_line=1,
                    end_line=2,
                ),
                BusinessRuleDef(
                    name="Rule 2",
                    description="Second rule",
                    category="authorization",
                    severity="error",
                    natural_language="Must be admin",
                    evidence_snippet="if not admin: raise",
                    start_line=5,
                    end_line=6,
                ),
            ],
            reasoning="Found two business rules",
        )

        assert len(output.rules) == 2
        assert output.reasoning == "Found two business rules"


class TestIntegrationWorkflow:
    """Tests verifying the full extraction workflow.

    Note: LLM-based rule extraction requires an API key.
    These tests verify the non-LLM extractors work together.
    """

    def test_full_extraction_pipeline(self) -> None:
        """Test running all non-LLM extractors on fixture data."""
        from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
        from contextmine_core.analyzer.extractors.erm import ERMExtractor, generate_mermaid_erd
        from contextmine_core.analyzer.extractors.surface import SurfaceCatalogExtractor

        # 1. Extract surface catalog
        surface_extractor = SurfaceCatalogExtractor()
        surface_extractor.add_file("api.yaml", FIXTURE_OPENAPI)
        surface_extractor.add_file(".github/workflows/ci.yml", FIXTURE_GITHUB_WORKFLOW)

        catalog = surface_extractor.catalog
        total_endpoints = sum(len(spec.endpoints) for spec in catalog.openapi_specs)
        total_jobs = sum(len(job_def.jobs) for job_def in catalog.job_definitions)
        assert total_endpoints >= 3
        assert total_jobs >= 2

        # 2. Extract ERM
        migration_result = extract_from_alembic("001.py", FIXTURE_ALEMBIC_MIGRATION)
        erm_extractor = ERMExtractor()
        erm_extractor.add_alembic_extraction(migration_result)

        schema = erm_extractor.schema
        assert len(schema.tables) >= 2

        mermaid = generate_mermaid_erd(schema)
        assert "erDiagram" in mermaid

        # 3. Verify all outputs are well-formed
        for spec in catalog.openapi_specs:
            for endpoint in spec.endpoints:
                assert endpoint.path
                assert endpoint.method

        for job_def in catalog.job_definitions:
            for job in job_def.jobs:
                assert job.name
                assert job.framework

        for table in schema.tables.values():
            assert table.name
            assert len(table.columns) > 0
