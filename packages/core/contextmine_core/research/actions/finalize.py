"""Finalize action for the research agent.

Produces the final answer with citations to evidence.
Note: Verification is handled by a separate LangGraph node, not this action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextmine_core.research.actions.registry import Action, ActionResult

if TYPE_CHECKING:
    from contextmine_core.research.run import ResearchRun


class FinalizeAction(Action):
    """Finalize the research with an answer and citations.

    This action marks the run as ready for verification. The actual
    verification happens in the verify_answer LangGraph node.
    """

    @property
    def name(self) -> str:
        return "finalize"

    @property
    def description(self) -> str:
        return (
            "Produce the final answer with citations. Use when you have sufficient "
            "evidence to answer the question. The answer will be verified against "
            "the evidence before being accepted."
        )

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Finalize the research.

        Args:
            run: Current research run
            params: Must contain 'answer', optionally 'confidence'

        Returns:
            ActionResult with success=True to signal verification needed
        """
        answer = params.get("answer", "")
        confidence = params.get("confidence", 0.8)

        if not answer:
            return ActionResult(
                success=False,
                output_summary="No answer provided",
                error="answer parameter is required",
            )

        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            confidence = max(0.0, min(1.0, confidence))

        # Build citations from evidence
        citations = self._build_citations(run)

        # Return success - the LangGraph workflow will handle verification
        # and call run.complete() if verification passes
        return ActionResult(
            success=True,
            output_summary=f"Answer ready for verification ({len(citations)} citations)",
            should_stop=False,  # Don't stop yet - verification needed
            data={
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "evidence_count": len(run.evidence),
            },
        )

    def _build_citations(self, run: ResearchRun) -> list[dict[str, Any]]:
        """Build citation list from evidence."""
        citations = []
        for e in run.evidence:
            citations.append(
                {
                    "id": e.id,
                    "file": e.file_path,
                    "lines": f"{e.start_line}-{e.end_line}",
                    "provenance": e.provenance,
                }
            )
        return citations


def format_answer_with_citations(
    answer: str,
    citations: list[dict[str, Any]],
    max_length: int = 800,
) -> str:
    """Format answer with citations for MCP tool output.

    Args:
        answer: The answer text
        citations: List of citation dicts
        max_length: Maximum length for the answer portion

    Returns:
        Formatted string with answer and citations
    """
    # Truncate answer if too long
    if len(answer) > max_length:
        answer = answer[:max_length] + "..."

    parts = [answer, "", "**Citations:**"]

    for c in citations[:10]:  # Limit to 10 citations in output
        parts.append(f"- [{c['id']}] {c['file']}:{c['lines']} ({c['provenance']})")

    if len(citations) > 10:
        parts.append(f"  ... and {len(citations) - 10} more")

    return "\n".join(parts)
