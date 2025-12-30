"""Tests for ERM extraction from Alembic migrations."""

from contextmine_core.analyzer.extractors.alembic import extract_from_alembic
from contextmine_core.analyzer.extractors.erm import (
    ERMExtractor,
    generate_mermaid_erd,
)


class TestAlembicExtractor:
    """Tests for Alembic migration parsing."""

    def test_extract_create_table(self) -> None:
        """Test extracting CREATE TABLE from Alembic migration."""
        content = '''
"""Create users table."""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
'''
        result = extract_from_alembic("001_users.py", content)

        assert len(result.tables) == 1
        assert result.tables[0].name == "users"
        assert len(result.tables[0].columns) == 3

        cols = {c.name: c for c in result.tables[0].columns}
        assert "id" in cols
        assert "name" in cols
        assert "email" in cols
        assert cols["id"].type_name == "UUID"
        assert cols["name"].type_name == "String"

    def test_extract_add_column(self) -> None:
        """Test extracting ADD COLUMN from Alembic migration."""
        content = '''
"""Add avatar column to users."""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column("users", sa.Column("avatar_url", sa.String(1024), nullable=True))
'''
        result = extract_from_alembic("002_avatar.py", content)

        assert len(result.added_columns) == 1
        table, col = result.added_columns[0]
        assert table == "users"
        assert col.name == "avatar_url"
        assert col.type_name == "String"
        assert col.nullable is True

    def test_extract_foreign_key(self) -> None:
        """Test extracting foreign keys."""
        content = '''
"""Create posts table with FK to users."""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "posts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
'''
        result = extract_from_alembic("003_posts.py", content)

        assert len(result.tables) == 1
        cols = {c.name: c for c in result.tables[0].columns}
        assert cols["user_id"].foreign_key == "users.id"

    def test_extract_explicit_foreign_key(self) -> None:
        """Test extracting explicit create_foreign_key calls."""
        content = '''
"""Add FK constraint."""

from alembic import op

def upgrade():
    op.create_foreign_key(
        "fk_posts_user",
        "posts",
        "users",
        ["user_id"],
        ["id"]
    )
'''
        result = extract_from_alembic("004_fk.py", content)

        assert len(result.foreign_keys) == 1
        fk = result.foreign_keys[0]
        assert fk.name == "fk_posts_user"
        assert fk.source_table == "posts"
        assert fk.target_table == "users"
        assert fk.source_columns == ["user_id"]
        assert fk.target_columns == ["id"]

    def test_extract_primary_key(self) -> None:
        """Test extracting primary key columns."""
        content = '''
"""Create table with PK."""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "items",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
    )
'''
        result = extract_from_alembic("005_items.py", content)

        assert len(result.tables) == 1
        cols = {c.name: c for c in result.tables[0].columns}
        assert cols["id"].primary_key is True
        assert cols["name"].primary_key is False


class TestERMExtractor:
    """Tests for ERMExtractor consolidation."""

    def test_merge_tables_from_multiple_migrations(self) -> None:
        """Test merging tables from multiple migration files."""
        migration1 = """
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
    )
"""
        migration2 = """
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column("users", sa.Column("email", sa.String(255), nullable=True))
"""
        extractor = ERMExtractor()
        extractor.add_alembic_extraction(extract_from_alembic("001.py", migration1))
        extractor.add_alembic_extraction(extract_from_alembic("002.py", migration2))

        assert "users" in extractor.schema.tables
        cols = {c.name: c for c in extractor.schema.tables["users"].columns}
        assert "id" in cols
        assert "name" in cols
        assert "email" in cols

    def test_collect_sources(self) -> None:
        """Test that source files are tracked."""
        migration = """
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table("test", sa.Column("id", sa.UUID()))
"""
        extractor = ERMExtractor()
        extractor.add_alembic_extraction(extract_from_alembic("001_test.py", migration))

        assert "001_test.py" in extractor.schema.sources


class TestMermaidERD:
    """Tests for Mermaid ERD generation."""

    def test_generate_simple_erd(self) -> None:
        """Test generating a simple ERD."""
        migration = """
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
    )
"""
        extractor = ERMExtractor()
        extractor.add_alembic_extraction(extract_from_alembic("001.py", migration))

        mermaid = generate_mermaid_erd(extractor.schema)

        assert "erDiagram" in mermaid
        assert "users {" in mermaid
        assert "uuid id PK" in mermaid
        assert "string name" in mermaid

    def test_generate_erd_with_relationships(self) -> None:
        """Test generating ERD with foreign key relationships."""
        migration = """
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("name", sa.String(255)),
    )
    op.create_table(
        "posts",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id")),
        sa.Column("title", sa.String(255)),
    )
"""
        extractor = ERMExtractor()
        extractor.add_alembic_extraction(extract_from_alembic("001.py", migration))

        mermaid = generate_mermaid_erd(extractor.schema)

        assert "erDiagram" in mermaid
        assert "users {" in mermaid
        assert "posts {" in mermaid
        assert "uuid user_id FK" in mermaid
        # Should have relationship line
        assert "||--o{" in mermaid

    def test_mermaid_safe_names(self) -> None:
        """Test that names with special chars are escaped."""
        migration = """
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "user-roles",
        sa.Column("id", sa.UUID(), primary_key=True),
    )
"""
        extractor = ERMExtractor()
        extractor.add_alembic_extraction(extract_from_alembic("001.py", migration))

        mermaid = generate_mermaid_erd(extractor.schema)

        # Hyphen should be replaced with underscore
        assert "user_roles {" in mermaid
