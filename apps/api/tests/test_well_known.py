"""Tests for root OAuth discovery forwarding to the embedded MCP app."""

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_root_well_known_matches_mounted_mcp_metadata(client: AsyncClient) -> None:
    mounted = await client.get("/mcp/.well-known/oauth-authorization-server")
    root = await client.get("/.well-known/oauth-authorization-server")

    assert root.status_code == mounted.status_code
    assert root.content == mounted.content
    assert root.headers["content-type"] == mounted.headers["content-type"]


@pytest.mark.anyio
async def test_root_well_known_preserves_resource_suffix(client: AsyncClient) -> None:
    mounted = await client.get("/mcp/.well-known/oauth-protected-resource/mcp/")
    root = await client.get("/.well-known/oauth-protected-resource/mcp/")

    assert root.status_code == mounted.status_code
    assert root.content == mounted.content
    assert root.headers["content-type"] == mounted.headers["content-type"]
