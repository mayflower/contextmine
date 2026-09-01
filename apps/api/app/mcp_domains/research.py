"""Research and knowledge-graph MCP tools."""

import json
import uuid
from typing import Annotated, Any, Literal

from contextmine_core import Collection, user_can_access_collection
from contextmine_core import get_session as get_db_session
from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.mcp_auth import get_current_user_id


class ResearchResult(BaseModel):
    """Structured, bounded research result."""

    kind: Literal[
        "deep_research",
        "validation",
        "data_model",
        "architecture",
        "graph_neighborhood",
        "trace_path",
        "graph_rag",
    ]
    query: str
    markdown: str
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    status: str | None = None
    truncated: bool = False


def _uuid(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise ToolError(f"Invalid {field_name}: {value}") from exc


async def _accessible_collection_ids(db, collection_id: str | None) -> list[uuid.UUID]:
    from contextmine_core.search import get_accessible_collection_ids

    user_id = get_current_user_id()
    accessible_ids = await get_accessible_collection_ids(db, user_id)
    if collection_id is None:
        if not accessible_ids:
            raise ToolError("No accessible collections found.")
        return accessible_ids
    requested = _uuid(collection_id, "collection_id")
    if requested not in accessible_ids:
        raise ToolError("Collection not found or access denied.")
    return [requested]


async def _require_graph_node(db, node_id: str):  # noqa: ANN202
    from contextmine_core.models import KnowledgeNode

    node_uuid = _uuid(node_id, "node_id")
    node = (
        await db.execute(select(KnowledgeNode).where(KnowledgeNode.id == node_uuid))
    ).scalar_one_or_none()
    if node is None:
        raise ToolError("Knowledge node not found.")
    collection = (
        await db.execute(select(Collection).where(Collection.id == node.collection_id))
    ).scalar_one_or_none()
    if collection is None or not await user_can_access_collection(
        db, collection, get_current_user_id()
    ):
        raise ToolError("Knowledge node not found or access denied.")
    return node


async def deep_research(
    question: Annotated[
        str,
        Field(
            description="Complex question requiring multi-step investigation",
            min_length=1,
            max_length=4000,
        ),
    ],
    scope: Annotated[
        str | None, Field(description="Optional path-pattern scope", max_length=1000)
    ] = None,
    budget: Annotated[int, Field(description="Maximum investigation steps", ge=1, le=20)] = 10,
    debug: Annotated[bool, Field(description="Include compact trace metadata")] = False,
) -> ResearchResult:
    """Run the existing LangGraph research agent and return its ContextMine run ID."""
    from contextmine_core.research import AgentConfig, ResearchAgent
    from contextmine_core.research.llm import get_research_llm_provider

    try:
        agent = ResearchAgent(
            llm_provider=get_research_llm_provider(),
            config=AgentConfig(max_steps=budget, store_artifacts=True),
        )
        run = await agent.research(question=question, scope=scope)
    except Exception as exc:
        raise ToolError(f"Research failed to start or execute: {exc}") from exc

    evidence = [
        {
            "id": item.id,
            "file": item.file_path,
            "start_line": item.start_line,
            "end_line": item.end_line,
            "provenance": item.provenance,
        }
        for item in run.evidence[:50]
    ]
    data: dict[str, Any] = {
        "citations": evidence,
        "steps_used": run.budget_used,
        "steps_budget": run.budget_steps,
        "error": run.error_message,
    }
    if debug:
        data["duration_ms"] = run.total_duration_ms
        data["artifact_report_uri"] = f"research://runs/{run.run_id}/report.md"
    answer = run.answer or ""
    truncated = len(answer) > 12000 or len(run.evidence) > 50
    markdown = answer[:12000]
    return ResearchResult(
        kind="deep_research",
        query=question,
        markdown=markdown,
        data=data,
        run_id=run.run_id,
        status=run.status.value,
        truncated=truncated,
    )


async def list_research_runs() -> str:
    """List the latest persisted research runs as a JSON resource."""
    from contextmine_core.research import get_artifact_store

    runs = get_artifact_store().list_runs(limit=20)
    return json.dumps(
        [
            {
                "run_id": item.run_id,
                "question": item.question,
                "status": item.status,
                "created_at": item.created_at.isoformat(),
                "completed_at": item.completed_at.isoformat() if item.completed_at else None,
            }
            for item in runs
        ],
        indent=2,
    )


def _research_artifact(run_id: str, artifact: str) -> str:
    from contextmine_core.research import get_artifact_store

    store = get_artifact_store()
    readers = {
        "trace": store.get_trace,
        "evidence": store.get_evidence,
        "report": store.get_report,
    }
    value = readers[artifact](run_id)
    if value is None:
        raise ResourceError(f"Research run not found: {run_id}")
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2)


async def get_research_trace(run_id: str) -> str:
    """Read one persisted research trace."""
    return _research_artifact(run_id, "trace")


async def get_research_evidence(run_id: str) -> str:
    """Read persisted evidence for one research run."""
    return _research_artifact(run_id, "evidence")


async def get_research_report(run_id: str) -> str:
    """Read the persisted Markdown report for one research run."""
    return _research_artifact(run_id, "report")


def _matches(value: str, words: set[str]) -> bool:
    normalized = value.lower().replace("_", " ")
    return any(word in normalized for word in words if len(word) > 2)


async def research_validation(
    code_path: Annotated[
        str, Field(description="File path or function name", min_length=1, max_length=1000)
    ],
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
) -> ResearchResult:
    """Find extracted business rules and validation candidates."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind

    words = set(code_path.lower().replace("/", " ").replace("_", " ").split())
    try:
        async with get_db_session() as db:
            collection_ids = await _accessible_collection_ids(db, collection_id)
            rows = list(
                (
                    await db.execute(
                        select(KnowledgeNode).where(
                            KnowledgeNode.collection_id.in_(collection_ids),
                            KnowledgeNode.kind.in_(
                                [
                                    KnowledgeNodeKind.BUSINESS_RULE,
                                    KnowledgeNodeKind.RULE_CANDIDATE,
                                ]
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to research validation: {exc}") from exc

    matched = []
    for row in rows:
        meta = row.meta or {}
        searchable = " ".join(
            [
                row.name,
                str(meta.get("natural_language") or ""),
                str(meta.get("container_name") or ""),
                str(meta.get("file_path") or ""),
            ]
        )
        if _matches(searchable, words):
            matched.append(row)
    matched = matched[:20]
    items = [
        {
            "id": str(row.id),
            "kind": row.kind.value,
            "name": row.name,
            "meta": row.meta or {},
        }
        for row in matched
    ]
    lines = [f"# Validation Rules for: {code_path}"]
    for item in items:
        lines.append(f"- **{item['name']}** ({item['kind']})")
    if not items:
        lines.append("No matching validation rules found.")
    return ResearchResult(
        kind="validation",
        query=code_path,
        markdown="\n".join(lines),
        data={"items": items, "count": len(items)},
        truncated=len(matched) == 20,
    )


async def research_data_model(
    entity: Annotated[
        str, Field(description="Table, entity, or data concept", min_length=1, max_length=500)
    ],
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
) -> ResearchResult:
    """Research extracted tables, columns, and related API endpoints."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind

    words = set(entity.lower().replace("_", " ").split())
    kinds = [
        KnowledgeNodeKind.DB_TABLE,
        KnowledgeNodeKind.DB_COLUMN,
        KnowledgeNodeKind.API_ENDPOINT,
    ]
    try:
        async with get_db_session() as db:
            collection_ids = await _accessible_collection_ids(db, collection_id)
            rows = list(
                (
                    await db.execute(
                        select(KnowledgeNode).where(
                            KnowledgeNode.collection_id.in_(collection_ids),
                            KnowledgeNode.kind.in_(kinds),
                        )
                    )
                )
                .scalars()
                .all()
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to research data model: {exc}") from exc

    matched = [
        row
        for row in rows
        if _matches(
            " ".join([row.name, str((row.meta or {}).get("path") or "")]),
            words,
        )
    ][:50]
    items = [
        {
            "id": str(row.id),
            "kind": row.kind.value,
            "name": row.name,
            "meta": row.meta or {},
        }
        for row in matched
    ]
    lines = [f"# Data Model: {entity}"] + [
        f"- **{item['name']}** ({item['kind']})" for item in items
    ]
    if not items:
        lines.append("No matching data-model entities found.")
    return ResearchResult(
        kind="data_model",
        query=entity,
        markdown="\n".join(lines),
        data={"items": items, "count": len(items)},
        truncated=len(matched) == 50,
    )


_ARCHITECTURE_KINDS = {
    "api": {"api_endpoint", "service_rpc", "interface_contract"},
    "deployment": {"job"},
    "database": {"db_table", "db_column"},
    "security": {"business_rule"},
    "ui": {"ui_route", "ui_view", "ui_component"},
    "tests": {"test_suite", "test_case", "test_fixture"},
    "flows": {"user_flow", "flow_step"},
    "rebuild": {
        "interface_contract",
        "user_flow",
        "flow_step",
        "test_case",
        "ui_route",
    },
}


async def research_architecture(
    topic: Annotated[
        Literal["api", "deployment", "database", "security", "ui", "tests", "flows", "rebuild"],
        Field(description="Architecture topic"),
    ],
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
) -> ResearchResult:
    """Research architecture entities already extracted into the knowledge graph."""
    from contextmine_core.models import KnowledgeNode, KnowledgeNodeKind

    enum_kinds = [KnowledgeNodeKind(value) for value in _ARCHITECTURE_KINDS[topic]]
    try:
        async with get_db_session() as db:
            collection_ids = await _accessible_collection_ids(db, collection_id)
            rows = list(
                (
                    await db.execute(
                        select(KnowledgeNode)
                        .where(
                            KnowledgeNode.collection_id.in_(collection_ids),
                            KnowledgeNode.kind.in_(enum_kinds),
                        )
                        .limit(101)
                    )
                )
                .scalars()
                .all()
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Failed to research architecture: {exc}") from exc

    truncated = len(rows) > 100
    rows = rows[:100]
    items = [
        {
            "id": str(row.id),
            "kind": row.kind.value,
            "name": row.name,
            "natural_key": row.natural_key,
            "meta": row.meta or {},
        }
        for row in rows
    ]
    lines = [f"# Architecture: {topic}"] + [
        f"- **{item['name']}** ({item['kind']})" for item in items
    ]
    if not items:
        lines.append("No matching architecture entities found.")
    return ResearchResult(
        kind="architecture",
        query=topic,
        markdown="\n".join(lines),
        data={"items": items, "count": len(items)},
        truncated=truncated,
    )


async def graph_neighborhood(
    node_id: Annotated[str, Field(description="Starting knowledge-node UUID")],
    depth: Annotated[int, Field(description="Expansion depth", ge=1, le=3)] = 1,
    edge_kinds: Annotated[
        list[str] | None, Field(description="Optional edge-kind filters", max_length=50)
    ] = None,
    limit: Annotated[int, Field(description="Maximum nodes", ge=1, le=200)] = 30,
) -> ResearchResult:
    """Explore an access-checked knowledge-graph neighborhood."""
    from contextmine_core.graphrag import graph_neighborhood as query_neighborhood

    async with get_db_session() as db:
        node = await _require_graph_node(db, node_id)
        try:
            result = await query_neighborhood(
                session=db,
                node_id=node.id,
                collection_id=node.collection_id,
                depth=depth,
                edge_kinds=edge_kinds,
                max_nodes=limit,
            )
        except Exception as exc:
            raise ToolError(f"Failed to get graph neighborhood: {exc}") from exc
    return ResearchResult(
        kind="graph_neighborhood",
        query=node_id,
        markdown=result.to_markdown(),
        data=result.to_dict(),
        truncated=len(result.entities) >= limit,
    )


async def trace_path(
    from_node_id: Annotated[str, Field(description="Starting node UUID")],
    to_node_id: Annotated[str, Field(description="Target node UUID")],
    max_hops: Annotated[int, Field(description="Maximum path length", ge=1, le=10)] = 6,
) -> ResearchResult:
    """Find a path between access-checked nodes in the same collection."""
    from contextmine_core.graphrag import trace_path as query_path

    async with get_db_session() as db:
        from_node = await _require_graph_node(db, from_node_id)
        to_node = await _require_graph_node(db, to_node_id)
        if from_node.collection_id != to_node.collection_id:
            raise ToolError("Nodes must belong to the same accessible collection.")
        try:
            result = await query_path(
                session=db,
                from_node_id=from_node.id,
                to_node_id=to_node.id,
                collection_id=from_node.collection_id,
                max_hops=max_hops,
            )
        except Exception as exc:
            raise ToolError(f"Failed to trace path: {exc}") from exc
    return ResearchResult(
        kind="trace_path",
        query=f"{from_node_id}->{to_node_id}",
        markdown=result.to_markdown(),
        data=result.to_dict(),
    )


def _node_kind_in_scope(kind: str, scope: str) -> bool:
    normalized = kind.strip().lower()
    test_kinds = {"test_suite", "test_case", "test_fixture"}
    ui_kinds = {"ui_route", "ui_view", "ui_component", "interface_contract"}
    flow_kinds = {"user_flow", "flow_step"}
    if scope == "all":
        return True
    if scope == "tests":
        return normalized in test_kinds
    if scope == "ui":
        return normalized in ui_kinds
    if scope == "flows":
        return normalized in flow_kinds
    return normalized not in (test_kinds | ui_kinds | flow_kinds)


def _filter_context(result, scope: str):  # noqa: ANN001, ANN202
    if scope == "all":
        return result
    result.entities = [
        entity for entity in result.entities if _node_kind_in_scope(entity.kind, scope)
    ]
    allowed_ids = {str(entity.node_id) for entity in result.entities}
    result.edges = [
        edge
        for edge in result.edges
        if edge.source_id in allowed_ids and edge.target_id in allowed_ids
    ]
    return result


async def graph_rag(
    query: Annotated[
        str, Field(description="Natural-language query", min_length=1, max_length=4000)
    ],
    collection_id: Annotated[str | None, Field(description="Optional collection UUID")] = None,
    max_communities: Annotated[
        int, Field(description="Maximum community summaries", ge=1, le=10)
    ] = 5,
    max_entities: Annotated[int, Field(description="Maximum local entities", ge=1, le=50)] = 20,
    max_depth: Annotated[int, Field(description="Graph expansion depth", ge=1, le=3)] = 2,
    format: Annotated[
        Literal["markdown", "json"], Field(description="Preferred presentation")
    ] = "markdown",
    answer: Annotated[
        bool, Field(description="Synthesize an answer with the configured LLM")
    ] = False,
    twin_scope: Annotated[
        Literal["code", "tests", "ui", "flows", "all"], Field(description="Semantic scope")
    ] = "all",
    rebuild_mode: Annotated[bool, Field(description="Include rebuild-oriented counts")] = False,
) -> ResearchResult:
    """Run the existing GraphRAG context or map-reduce answer path."""
    from contextmine_core.graphrag import graph_rag_context, graph_rag_query

    collection_uuid = _uuid(collection_id, "collection_id") if collection_id else None
    user_id = get_current_user_id()
    try:
        async with get_db_session() as db:
            if answer:
                from contextmine_core.research.llm import get_llm_provider
                from contextmine_core.settings import get_settings

                settings = get_settings()
                if not settings.default_llm_provider:
                    raise ToolError("No LLM provider configured.")
                result = await graph_rag_query(
                    session=db,
                    query=query,
                    llm_provider=get_llm_provider(settings.default_llm_provider),
                    collection_id=collection_uuid,
                    user_id=user_id,
                    max_communities=max_communities,
                    max_entities=max_entities,
                )
                context = _filter_context(result.context, twin_scope) if result.context else None
                data = {
                    "communities_used": result.communities_used,
                    "partial_answer_count": len(result.partial_answers),
                    "context": context.to_dict() if context else None,
                }
                markdown = result.final_answer
            else:
                context = await graph_rag_context(
                    session=db,
                    query=query,
                    collection_id=collection_uuid,
                    user_id=user_id,
                    max_communities=max_communities,
                    max_entities=max_entities,
                    max_depth=max_depth,
                )
                context = _filter_context(context, twin_scope)
                data = context.to_dict()
                markdown = context.to_markdown()
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"GraphRAG query failed: {exc}") from exc

    if rebuild_mode:
        entities = data.get("entities")
        kinds = [item.get("kind") for item in entities] if isinstance(entities, list) else []
        data["rebuild_counts"] = {
            "interfaces": sum(
                kind in {"api_endpoint", "service_rpc", "interface_contract"} for kind in kinds
            ),
            "ui": sum(kind in {"ui_route", "ui_view", "ui_component"} for kind in kinds),
            "flows": sum(kind in {"user_flow", "flow_step"} for kind in kinds),
            "tests": sum(kind in {"test_suite", "test_case", "test_fixture"} for kind in kinds),
        }
    if format == "json":
        markdown = ""
    return ResearchResult(
        kind="graph_rag",
        query=query,
        markdown=markdown[:20000],
        data=data,
        truncated=len(markdown) > 20000,
    )


def register_research_tools(mcp: FastMCP) -> None:
    """Register research tools and persisted-run resources."""
    mcp.tool(name="deep_research")(deep_research)
    mcp.tool(name="research_validation")(research_validation)
    mcp.tool(name="research_data_model")(research_data_model)
    mcp.tool(name="research_architecture")(research_architecture)
    mcp.tool(name="graph_neighborhood")(graph_neighborhood)
    mcp.tool(name="trace_path")(trace_path)
    mcp.tool(name="graph_rag")(graph_rag)
    mcp.resource("research://runs")(list_research_runs)
    mcp.resource("research://runs/{run_id}/trace.json")(get_research_trace)
    mcp.resource("research://runs/{run_id}/evidence.json")(get_research_evidence)
    mcp.resource("research://runs/{run_id}/report.md")(get_research_report)
