"""arc42 generation via Anthropic Claude Agent SDK with shared sessions."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from .arc42 import SECTION_TITLES
from .schemas import Arc42Document


class ClaudeAgentSdkUnavailableError(RuntimeError):
    """Raised when Claude Agent SDK is not available in the runtime."""


@dataclass
class _ClientEntry:
    """One shared SDK client instance bound to one repository checkout."""

    client: Any
    lock: asyncio.Lock
    connected: bool = False
    session_ids: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.session_ids is None:
            self.session_ids = {}


class ClaudeSDKSessionManager:
    """Shared ClaudeSDKClient manager for arc42 generation."""

    def __init__(self) -> None:
        self._entries: dict[str, _ClientEntry] = {}
        self._entries_lock = asyncio.Lock()

    async def _get_entry(
        self,
        *,
        repo_path: Path,
        model: str,
        max_turns: int,
        permission_mode: str,
    ) -> _ClientEntry:
        key = f"{repo_path.resolve()}|{model}|{max_turns}|{permission_mode}"
        async with self._entries_lock:
            existing = self._entries.get(key)
            if existing is not None:
                return existing

            try:
                from claude_code_sdk import ClaudeCodeOptions
                from claude_code_sdk import ClaudeSDKClient as RawClaudeSDKClient
            except Exception as exc:  # noqa: BLE001
                raise ClaudeAgentSdkUnavailableError(
                    "claude-code-sdk is not installed. Add dependency and redeploy."
                ) from exc

            options = ClaudeCodeOptions(
                cwd=str(repo_path.resolve()),
                model=model,
                max_turns=max(1, int(max_turns)),
                permission_mode=permission_mode,
            )
            entry = _ClientEntry(client=RawClaudeSDKClient(options=options), lock=asyncio.Lock())
            self._entries[key] = entry
            return entry

    async def run_prompt(
        self,
        *,
        repo_path: Path,
        scope_key: str,
        prompt: str,
        model: str,
        max_turns: int,
        permission_mode: str,
    ) -> tuple[str, dict[str, Any]]:
        entry = await self._get_entry(
            repo_path=repo_path,
            model=model,
            max_turns=max_turns,
            permission_mode=permission_mode,
        )

        async with entry.lock:
            if not entry.connected:
                await entry.client.connect()
                entry.connected = True

            session_id = (entry.session_ids or {}).get(scope_key, scope_key)
            await entry.client.query(prompt, session_id=session_id)

            result_text = ""
            raw_messages: list[Any] = []
            usage: dict[str, Any] | None = None
            total_cost_usd: float | None = None
            returned_session_id: str | None = None

            from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

            async for message in entry.client.receive_response():
                raw_messages.append(message)
                if isinstance(message, ResultMessage):
                    returned_session_id = str(message.session_id or "")
                    usage = message.usage
                    total_cost_usd = message.total_cost_usd
                    if message.is_error:
                        raise RuntimeError(message.result or "Claude SDK returned an error")
                    if message.result:
                        result_text = str(message.result)

            if not result_text:
                parts: list[str] = []
                for msg in raw_messages:
                    if not isinstance(msg, AssistantMessage):
                        continue
                    for block in getattr(msg, "content", []) or []:
                        if isinstance(block, TextBlock):
                            parts.append(str(getattr(block, "text", "")))
                result_text = "\n".join(p for p in parts if p.strip()).strip()

            if returned_session_id:
                (entry.session_ids or {})[scope_key] = returned_session_id

            return result_text, {
                "session_id": returned_session_id or session_id,
                "usage": usage,
                "total_cost_usd": total_cost_usd,
            }


_SESSION_MANAGER = ClaudeSDKSessionManager()


def _extract_json_blob(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Claude SDK returned empty response.")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Claude SDK response is not valid JSON.")


def _render_markdown(title: str, sections: dict[str, str]) -> str:
    lines = [f"# {title}", ""]
    for key in SECTION_TITLES:
        content = (sections.get(key) or "").strip()
        lines.append(f"## {SECTION_TITLES[key]}")
        lines.append(content or "UNKNOWN: insufficient evidence")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _arc42_prompt(*, scenario_name: str, section: str | None) -> str:
    section_instruction = (
        f"Focus section: {section}. Still return all 12 section keys."
        if section
        else "No section filter. Return all 12 sections."
    )
    section_keys = ", ".join(SECTION_TITLES.keys())
    return (
        "Generate a real arc42 document from repository evidence using tools. "
        "Do not invent facts. If evidence is missing, write exactly "
        "'UNKNOWN: insufficient evidence'. "
        f"Scenario name: {scenario_name}. {section_instruction}\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "title": "arc42 - <scenario>",\n'
        '  "warnings": ["..."],\n'
        '  "sections": {\n'
        f'    "{list(SECTION_TITLES.keys())[0]}": "...",\n'
        '    "...": "..."\n'
        "  }\n"
        "}\n\n"
        f"Mandatory section keys: {section_keys}\n"
        "No Markdown fences. JSON only."
    )


async def generate_arc42_with_claude_sdk(
    *,
    collection_id: UUID,
    scenario_id: UUID,
    scenario_name: str,
    repo_path: Path,
    section: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    max_turns: int = 50,
    permission_mode: str = "bypassPermissions",
) -> tuple[Arc42Document, dict[str, Any]]:
    """Generate an arc42 document via shared ClaudeSDKClient session."""

    if not repo_path.exists() or not repo_path.is_dir():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    scope_key = f"arc42:{collection_id}:{scenario_id}"
    prompt = _arc42_prompt(scenario_name=scenario_name, section=section)
    raw_output, runtime_meta = await _SESSION_MANAGER.run_prompt(
        repo_path=repo_path,
        scope_key=scope_key,
        prompt=prompt,
        model=model,
        max_turns=max_turns,
        permission_mode=permission_mode,
    )
    payload = _extract_json_blob(raw_output)

    incoming_sections = payload.get("sections")
    sections: dict[str, str] = {}
    if isinstance(incoming_sections, dict):
        for key in SECTION_TITLES:
            value = incoming_sections.get(key)
            sections[key] = str(value).strip() if isinstance(value, str) else ""
    else:
        sections = dict.fromkeys(SECTION_TITLES, "")

    title = (
        str(payload.get("title") or f"arc42 - {scenario_name}").strip()
        or f"arc42 - {scenario_name}"
    )
    warnings = [
        str(item).strip()
        for item in (payload.get("warnings") or [])
        if isinstance(item, str) and str(item).strip()
    ]
    section_coverage = {key: bool((sections.get(key) or "").strip()) for key in SECTION_TITLES}
    markdown = _render_markdown(title, sections)

    document = Arc42Document(
        collection_id=collection_id,
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        title=title,
        generated_at=datetime.now(UTC),
        sections=sections,
        markdown=markdown,
        warnings=warnings,
        confidence_summary={
            "engine": "claude_agent_sdk",
            "model": model,
            "runtime": runtime_meta,
        },
        section_coverage=section_coverage,
    )
    meta = {
        "engine": "claude_agent_sdk",
        "model": model,
        "session_id": runtime_meta.get("session_id"),
        "usage": runtime_meta.get("usage"),
        "total_cost_usd": runtime_meta.get("total_cost_usd"),
        "raw_length": len(raw_output),
    }
    return document, meta
