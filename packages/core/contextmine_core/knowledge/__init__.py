"""Knowledge graph module for ContextMine.

This module provides:
- Pydantic schemas for knowledge graph entities
- Builder functions for populating the graph from indexed data
- Service functions for graph operations
"""

from contextmine_core.knowledge.builder import (
    GraphBuildStats,
    build_knowledge_graph_for_source,
    cleanup_orphan_nodes,
)
from contextmine_core.knowledge.communities import (
    Community,
    HierarchicalCommunities,
    detect_communities,
    persist_communities,
)
from contextmine_core.knowledge.schemas import (
    BusinessRuleMetadata,
    EvidenceCreate,
    EvidenceSchema,
    GraphNeighborhoodRequest,
    GraphNeighborhoodResponse,
    GraphPathRequest,
    GraphPathResponse,
    GraphRAGBundle,
    GraphRAGRequest,
    KnowledgeArtifactCreate,
    KnowledgeArtifactSchema,
    KnowledgeArtifactWithEvidenceSchema,
    KnowledgeEdgeCreate,
    KnowledgeEdgeSchema,
    KnowledgeEdgeWithNodesSchema,
    KnowledgeNodeCreate,
    KnowledgeNodeSchema,
    KnowledgeNodeWithEvidenceSchema,
    RuleCandidateMetadata,
)
from contextmine_core.knowledge.summaries import (
    CommunitySummaryOutput,
    SummaryStats,
    generate_community_summaries,
)

__all__ = [
    # Builder functions
    "GraphBuildStats",
    "build_knowledge_graph_for_source",
    "cleanup_orphan_nodes",
    # Community detection
    "Community",
    "HierarchicalCommunities",
    "detect_communities",
    "persist_communities",
    # Community summaries
    "CommunitySummaryOutput",
    "SummaryStats",
    "generate_community_summaries",
    # Evidence
    "EvidenceSchema",
    "EvidenceCreate",
    # Nodes
    "KnowledgeNodeSchema",
    "KnowledgeNodeWithEvidenceSchema",
    "KnowledgeNodeCreate",
    # Edges
    "KnowledgeEdgeSchema",
    "KnowledgeEdgeWithNodesSchema",
    "KnowledgeEdgeCreate",
    # Artifacts
    "KnowledgeArtifactSchema",
    "KnowledgeArtifactWithEvidenceSchema",
    "KnowledgeArtifactCreate",
    # Graph queries
    "GraphNeighborhoodRequest",
    "GraphNeighborhoodResponse",
    "GraphPathRequest",
    "GraphPathResponse",
    # Business rules
    "RuleCandidateMetadata",
    "BusinessRuleMetadata",
    # GraphRAG
    "GraphRAGRequest",
    "GraphRAGBundle",
]
