"""Joern integration primitives used by twin analysis and worker materialization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class JoernResponse:
    success: bool
    stdout: str
    stderr: str


class JoernClient:
    """Minimal async client for the Joern HTTP API."""

    def __init__(self, base_url: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(self.base_url)
            return response.status_code in {200, 404}
        except Exception:
            return False

    async def execute_query(self, query: str, timeout_seconds: int | None = None) -> JoernResponse:
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        payload = {"query": query}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/query-sync", json=payload)
            if response.status_code != 200:
                return JoernResponse(
                    success=False,
                    stdout="",
                    stderr=f"HTTP {response.status_code}: {response.text}",
                )
            data = response.json()
            return JoernResponse(
                success=bool(data.get("success")),
                stdout=str(data.get("stdout", "")),
                stderr=str(data.get("stderr", "")),
            )
        except Exception as exc:  # noqa: BLE001
            return JoernResponse(success=False, stdout="", stderr=str(exc))

    async def load_cpg(self, cpg_path: str, timeout_seconds: int = 600) -> JoernResponse:
        return await self.execute_query(
            f'workspace.reset; importCpg("{cpg_path}")',
            timeout_seconds=timeout_seconds,
        )


def parse_joern_output(output: str) -> Any:
    """Parse Joern output into JSON/primitive values where possible."""
    if not output or not output.strip():
        return []

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    cleaned = ansi_escape.sub("", output)

    marker_match = re.search(
        r"<contextmine_result>\s*(.*?)\s*</contextmine_result>",
        cleaned,
        re.DOTALL,
    )
    if marker_match:
        return marker_match.group(1).strip()

    triple_json = re.search(r'"""(\[.*?\]|\{.*?\})"""', cleaned, re.DOTALL)
    if triple_json:
        try:
            parsed = json.loads(triple_json.group(1))
            return parsed
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    value = cleaned.strip()
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
