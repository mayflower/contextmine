"""Hybrid search action for the research agent.

Wraps the existing hybrid_search functionality to be used as an agent action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from contextmine_core.research.actions.registry import Action, ActionResult
from contextmine_core.research.run import Evidence

if TYPE_CHECKING:
    from contextmine_core.research.run import ResearchRun


class HybridSearchAction(Action):
    """Search the codebase using hybrid BM25 + vector search."""

    @property
    def name(self) -> str:
        return "hybrid_search"

    @property
    def description(self) -> str:
        return "Search the codebase using keywords and semantic similarity. Returns relevant code snippets."

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Execute hybrid search.

        Args:
            run: Current research run
            params: Must contain 'query' and optionally 'k'

        Returns:
            ActionResult with search results as evidence
        """
        query = params.get("query", "")
        k = params.get("k", 10)

        if not query:
            return ActionResult(
                success=False,
                output_summary="No query provided",
                error="Query parameter is required",
            )

        try:
            # Import here to avoid circular imports and allow mocking
            from contextmine_core.embeddings import EmbeddingProvider, get_embedder
            from contextmine_core.search import hybrid_search
            from contextmine_core.settings import get_settings

            settings = get_settings()

            # Parse provider from default_embedding_model (format: 'provider:model_name')
            embedding_model = settings.default_embedding_model
            if ":" in embedding_model:
                provider_str, model_name = embedding_model.split(":", 1)
            else:
                provider_str = "openai"
                model_name = embedding_model

            # Get embedder and create query embedding
            embedder = get_embedder(
                provider=EmbeddingProvider(provider_str),
                model_name=model_name,
                api_key=settings.openai_api_key
                if provider_str == "openai"
                else settings.gemini_api_key,
            )
            embedding_result = await embedder.embed_batch([query])
            query_embedding = embedding_result.embeddings[0]

            # Perform hybrid search
            # Note: We search across all accessible collections (no user_id = global only)
            response = await hybrid_search(
                query=query,
                query_embedding=query_embedding,
                user_id=None,  # Research agent uses global access
                collection_id=None,  # Search all collections
                top_k=k,
            )

            # Convert results to evidence
            evidence_items: list[Evidence] = []
            for i, result in enumerate(response.results):
                evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + i + 1:03d}"

                # Parse line numbers from content if available (chunk metadata)
                # For now, use placeholder line numbers - will be enhanced with actual metadata
                start_line = 1
                end_line = len(result.content.split("\n"))

                # Determine provenance based on ranking
                if result.fts_rank and result.vector_rank:
                    provenance = "hybrid"
                elif result.fts_rank:
                    provenance = "bm25"
                else:
                    provenance = "vector"

                evidence = Evidence(
                    id=evidence_id,
                    file_path=result.uri or result.title,
                    start_line=start_line,
                    end_line=end_line,
                    content=result.content[:2000],  # Limit content size
                    reason=f"Matched query '{query}' with score {result.score:.3f}",
                    provenance=provenance,
                    score=result.score,
                )
                evidence_items.append(evidence)

            return ActionResult(
                success=True,
                output_summary=f"Found {len(response.results)} results for '{query}'",
                evidence=evidence_items,
                data={
                    "query": query,
                    "total_results": len(response.results),
                    "total_fts_matches": response.total_fts_matches,
                    "total_vector_matches": response.total_vector_matches,
                },
            )

        except Exception as e:
            return ActionResult(
                success=False,
                output_summary=f"Search failed: {e}",
                error=str(e),
            )


class MockHybridSearchAction(Action):
    """Mock hybrid search for testing without database access."""

    def __init__(self, mock_results: list[dict[str, Any]] | None = None):
        self._mock_results = mock_results or []

    @property
    def name(self) -> str:
        return "hybrid_search"

    @property
    def description(self) -> str:
        return "Search the codebase using keywords and semantic similarity."

    def set_results(self, results: list[dict[str, Any]]) -> None:
        """Set mock results for testing."""
        self._mock_results = results

    async def execute(
        self,
        run: ResearchRun,
        params: dict[str, Any],
    ) -> ActionResult:
        """Return mock search results."""
        query = params.get("query", "")
        k = params.get("k", 10)

        if not query:
            return ActionResult(
                success=False,
                output_summary="No query provided",
                error="Query parameter is required",
            )

        # Generate evidence from mock results
        evidence_items: list[Evidence] = []
        for i, result in enumerate(self._mock_results[:k]):
            evidence_id = f"ev-{run.run_id[:8]}-{len(run.evidence) + i + 1:03d}"
            evidence = Evidence(
                id=evidence_id,
                file_path=result.get("file_path", f"mock/file_{i}.py"),
                start_line=result.get("start_line", 1),
                end_line=result.get("end_line", 10),
                content=result.get("content", f"# Mock content for {query}"),
                reason=f"Matched query '{query}'",
                provenance=result.get("provenance", "bm25"),
                score=result.get("score", 0.9 - i * 0.1),
            )
            evidence_items.append(evidence)

        return ActionResult(
            success=True,
            output_summary=f"Found {len(evidence_items)} results for '{query}'",
            evidence=evidence_items,
            data={
                "query": query,
                "total_results": len(evidence_items),
            },
        )
