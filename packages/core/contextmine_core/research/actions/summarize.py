"""Summarize evidence action for the research agent.

Uses LLM to compress collected evidence into a concise memo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextmine_core.research.actions.registry import Action, ActionResult
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contextmine_core.research.llm.provider import LLMProvider
    from contextmine_core.research.run import ResearchRun


class EvidenceSummary(BaseModel):
    """Structured output for evidence summarization."""

    summary: str = Field(description="Concise summary of the evidence")
    key_findings: list[str] = Field(description="List of key findings from the evidence")
    relevant_files: list[str] = Field(description="Most relevant files from the evidence")
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps that need more investigation",
    )


SUMMARIZE_SYSTEM_PROMPT = """You are a code research assistant summarizing evidence for a research question.

Your task is to:
1. Summarize the collected evidence concisely
2. Highlight key findings that answer the research question
3. List the most relevant files
4. Identify any gaps in the evidence

Be factual and cite specific evidence. Do not make assumptions beyond what the evidence shows.
"""


class SummarizeEvidenceAction(Action):
    """Summarize collected evidence into a concise memo."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        """Initialize with optional LLM provider.

        Args:
            llm_provider: LLM provider for summarization. If None, will use default.
        """
        self._llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "summarize_evidence"

    @property
    def description(self) -> str:
        return "Compress collected evidence into a concise memo. Use when you have enough evidence but need to organize it."

    def _get_provider(self) -> LLMProvider:
        """Get the LLM provider, creating default if needed."""
        if self._llm_provider is None:
            from contextmine_core.research.llm import get_research_llm_provider

            self._llm_provider = get_research_llm_provider()
        return self._llm_provider

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Summarize evidence.

        Args:
            run: Current research run
            params: Should contain 'goal' describing what to summarize

        Returns:
            ActionResult with summary
        """
        goal = params.get("goal", "summarize all evidence")

        if not run.evidence:
            return ActionResult(
                success=True,
                output_summary="No evidence to summarize",
                data={"summary": "No evidence collected yet.", "evidence_count": 0},
            )

        try:
            provider = self._get_provider()

            # Build evidence context
            evidence_text = self._format_evidence_for_llm(run)

            # Create prompt
            user_message = f"""## Research Question
{run.question}

## Summarization Goal
{goal}

## Collected Evidence ({len(run.evidence)} items)

{evidence_text}

Please summarize this evidence."""

            # Get structured summary
            summary = await provider.generate_structured(
                system=SUMMARIZE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_schema=EvidenceSummary,
                max_tokens=1024,
            )

            return ActionResult(
                success=True,
                output_summary=f"Summarized {len(run.evidence)} evidence items",
                data={
                    "summary": summary.summary,
                    "key_findings": summary.key_findings,
                    "relevant_files": summary.relevant_files,
                    "gaps": summary.gaps,
                    "evidence_count": len(run.evidence),
                },
            )

        except Exception as e:
            return ActionResult(
                success=False,
                output_summary=f"Summarization failed: {e}",
                error=str(e),
            )

    def _format_evidence_for_llm(self, run: ResearchRun) -> str:
        """Format evidence items for LLM consumption."""
        parts = []
        for e in run.evidence:
            # Truncate long content
            content = e.content
            if len(content) > 1000:
                content = content[:1000] + "\n... [truncated]"

            parts.append(f"""### [{e.id}] {e.file_path}:{e.start_line}-{e.end_line}
**Reason:** {e.reason}
**Provenance:** {e.provenance}
**Score:** {e.score or "N/A"}

```
{content}
```
""")
        return "\n".join(parts)


class MockSummarizeAction(Action):
    """Mock summarize action for testing."""

    def __init__(self, mock_summary: str = "Mock summary of evidence."):
        self._mock_summary = mock_summary

    @property
    def name(self) -> str:
        return "summarize_evidence"

    @property
    def description(self) -> str:
        return "Compress collected evidence into a concise memo."

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock summary."""
        goal = params.get("goal", "summarize")

        if not run.evidence:
            return ActionResult(
                success=True,
                output_summary="No evidence to summarize",
                data={"summary": "No evidence collected yet.", "evidence_count": 0},
            )

        # Extract unique files from evidence
        files = list({e.file_path for e in run.evidence})

        return ActionResult(
            success=True,
            output_summary=f"Summarized {len(run.evidence)} evidence items",
            data={
                "summary": self._mock_summary,
                "key_findings": [f"Finding from {goal}"],
                "relevant_files": files[:5],
                "gaps": [],
                "evidence_count": len(run.evidence),
            },
        )
