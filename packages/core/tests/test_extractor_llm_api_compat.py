"""Regression tests for extractor integration with the current LLM provider API."""

import pytest
from contextmine_core.analyzer.extractors.jobs import extract_jobs_from_file
from contextmine_core.analyzer.extractors.schema import extract_schema_from_file
from contextmine_core.analyzer.extractors.triage import (
    triage_files_for_jobs,
    triage_files_for_schema,
)
from contextmine_core.research.llm.mock import MockLLMProvider


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_triage_files_for_schema_uses_structured_llm_call() -> None:
    files = [(f"file_{idx}.py", "print('x')") for idx in range(11)]
    provider = MockLLMProvider(
        structured_responses={
            "FileTriageResult": {
                "relevant_files": [files[3][0]],
                "reasoning": "Contains schema-like definitions.",
            }
        }
    )

    selected = await triage_files_for_schema(files, provider)

    assert selected == [files[3][0]]
    call = provider.call_history[-1]
    assert call["method"] == "generate_structured"
    assert str(call["system"]).strip()
    assert call["messages"] and call["messages"][0]["role"] == "user"


@pytest.mark.anyio
async def test_triage_files_for_jobs_uses_structured_llm_call() -> None:
    files = [(f"config_{idx}.yml", "name: test") for idx in range(11)]
    provider = MockLLMProvider(
        structured_responses={
            "FileTriageResult": {
                "relevant_files": [files[6][0]],
                "reasoning": "Contains workflow configuration.",
            }
        }
    )

    selected = await triage_files_for_jobs(files, provider)

    assert selected == [files[6][0]]
    call = provider.call_history[-1]
    assert call["method"] == "generate_structured"
    assert str(call["system"]).strip()
    assert call["messages"] and call["messages"][0]["role"] == "user"


@pytest.mark.anyio
async def test_extract_schema_from_file_uses_current_provider_signature() -> None:
    provider = MockLLMProvider(
        structured_responses={
            "SchemaExtractionOutput": {
                "framework": "sql",
                "tables": [
                    {
                        "name": "users",
                        "description": "User table",
                        "columns": [
                            {
                                "name": "id",
                                "type_name": "UUID",
                                "nullable": False,
                                "primary_key": True,
                                "foreign_key": None,
                                "description": "Primary key",
                            }
                        ],
                    }
                ],
                "foreign_keys": [],
            }
        }
    )

    result = await extract_schema_from_file(
        "db/schema.sql",
        "CREATE TABLE users (id UUID PRIMARY KEY);",
        provider,
    )

    assert result.framework == "sql"
    assert len(result.tables) == 1
    assert result.tables[0].name == "users"
    assert len(result.tables[0].columns) == 1
    assert result.tables[0].columns[0].name == "id"


@pytest.mark.anyio
async def test_extract_jobs_from_file_uses_current_provider_signature() -> None:
    provider = MockLLMProvider(
        structured_responses={
            "JobsExtractionOutput": {
                "jobs": [
                    {
                        "name": "nightly-build",
                        "framework": "github_actions",
                        "description": "Nightly build pipeline",
                        "schedule": "0 2 * * *",
                        "container_image": "python:3.12",
                        "dependencies": [],
                        "triggers": [
                            {
                                "trigger_type": "schedule",
                                "cron": "0 2 * * *",
                                "description": "Nightly",
                            }
                        ],
                        "steps": [
                            {
                                "name": "test",
                                "description": "Run tests",
                                "action": "pytest",
                            }
                        ],
                    }
                ]
            }
        }
    )

    result = await extract_jobs_from_file(
        ".github/workflows/ci.yml",
        "name: CI",
        provider,
    )

    assert len(result.jobs) == 1
    assert result.jobs[0].name == "nightly-build"
    assert result.jobs[0].framework == "github_actions"
    assert result.jobs[0].triggers and result.jobs[0].triggers[0].trigger_type == "schedule"
