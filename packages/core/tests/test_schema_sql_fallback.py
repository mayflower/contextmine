from __future__ import annotations

import pytest
from contextmine_core.analyzer.extractors.schema import extract_schema_from_file
from contextmine_core.research.llm.mock import MockLLMProvider


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_sql_fallback_extracts_tables_columns_and_inline_fk_without_llm() -> None:
    provider = MockLLMProvider()
    sql = """
    CREATE TABLE users (
      id UUID PRIMARY KEY,
      email VARCHAR(255) NOT NULL
    );

    CREATE TABLE posts (
      id BIGINT PRIMARY KEY,
      user_id UUID NOT NULL REFERENCES users(id),
      body TEXT
    );
    """

    result = await extract_schema_from_file("db/schema.sql", sql, provider)

    assert provider.call_history == []
    assert len(result.tables) == 2
    by_name = {table.name: table for table in result.tables}
    assert "users" in by_name
    assert "posts" in by_name
    assert [col.name for col in by_name["users"].columns] == ["id", "email"]
    assert [col.name for col in by_name["posts"].columns] == ["id", "user_id", "body"]
    user_id = next(col for col in by_name["posts"].columns if col.name == "user_id")
    assert user_id.foreign_key == "users.id"
    assert user_id.nullable is False


@pytest.mark.anyio
async def test_sql_fallback_extracts_table_level_fk_constraint() -> None:
    provider = MockLLMProvider()
    sql = """
    CREATE TABLE posts (
      id INT PRIMARY KEY
    );

    CREATE TABLE comments (
      id INT PRIMARY KEY,
      post_id INT NOT NULL,
      CONSTRAINT fk_comments_post FOREIGN KEY (post_id) REFERENCES posts(id)
    );
    """

    result = await extract_schema_from_file("db/schema.ddl", sql, provider)

    assert provider.call_history == []
    assert len(result.foreign_keys) == 1
    fk = result.foreign_keys[0]
    assert fk.source_table == "comments"
    assert fk.source_columns == ["post_id"]
    assert fk.target_table == "posts"
    assert fk.target_columns == ["id"]
