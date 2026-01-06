"""Hierarchical community detection using the Leiden algorithm.

This module implements the community detection component of GraphRAG
as described in the Microsoft paper "From Local to Global: A Graph RAG
Approach to Query-Focused Summarization".

Key features:
- Leiden algorithm for high-quality community detection
- Hierarchical communities via resolution parameter
- Deterministic results with fixed seed
- Multi-level community structure (C0, C1, C2, ...)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import igraph as ig
import leidenalg
from contextmine_core.models import (
    CommunityMember,
    KnowledgeCommunity,
    KnowledgeEdge,
    KnowledgeEdgeKind,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from sqlalchemy import delete, select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Edge kinds for community detection - semantic relationships only
# No fallback to code relationships (that defeats the purpose of GraphRAG)
SEMANTIC_EDGE_KINDS = {
    KnowledgeEdgeKind.SEMANTIC_RELATIONSHIP,
}

# Resolution parameters for hierarchical levels
# Higher resolution = more smaller communities
# Lower resolution = fewer larger communities
DEFAULT_RESOLUTIONS = [1.0, 0.5, 0.1]  # Level 0, 1, 2


@dataclass
class Community:
    """A single community at a specific level."""

    id: int  # Community index within level
    level: int
    node_ids: list[UUID] = field(default_factory=list)
    node_keys: list[str] = field(default_factory=list)
    size: int = 0

    @property
    def natural_key(self) -> str:
        """Generate natural key for this community."""
        return f"community:L{self.level}:C{self.id}"


@dataclass
class HierarchicalCommunities:
    """Result of hierarchical Leiden community detection."""

    # Communities by level: level -> list of Community
    levels: dict[int, list[Community]] = field(default_factory=dict)

    # Node to community mapping: node_id -> {level: community_id}
    node_membership: dict[UUID, dict[int, int]] = field(default_factory=dict)

    # Modularity scores per level
    modularity: dict[int, float] = field(default_factory=dict)

    def community_count(self, level: int = 0) -> int:
        """Get number of communities at a level."""
        return len(self.levels.get(level, []))

    def total_communities(self) -> int:
        """Get total communities across all levels."""
        return sum(len(comms) for comms in self.levels.values())

    def get_community(self, level: int, community_id: int) -> Community | None:
        """Get a specific community."""
        communities = self.levels.get(level, [])
        for comm in communities:
            if comm.id == community_id:
                return comm
        return None


async def detect_communities(
    session: AsyncSession,
    collection_id: UUID,
    resolutions: list[float] | None = None,
    seed: int = 42,
) -> HierarchicalCommunities:
    """Detect hierarchical communities using Leiden algorithm.

    This implements the community detection described in the GraphRAG paper:
    1. Build graph from semantic entities (LLM-extracted) - REQUIRED
    2. Run Leiden at multiple resolution levels
    3. Return hierarchical community structure

    IMPORTANT: This requires LLM-extracted semantic entities to exist.
    Without semantic entities, this returns empty (no fallback to code symbols,
    as that would defeat the purpose of semantic clustering).

    Args:
        session: Database session
        collection_id: Collection UUID
        resolutions: Resolution parameters for each level (default: [1.0, 0.5, 0.1])
        seed: Random seed for reproducibility

    Returns:
        HierarchicalCommunities with multi-level structure (empty if no semantic entities)
    """
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTIONS

    result = HierarchicalCommunities()

    # Load semantic entities (LLM-extracted) - required for proper GraphRAG
    node_result = await session.execute(
        select(
            KnowledgeNode.id, KnowledgeNode.natural_key, KnowledgeNode.kind, KnowledgeNode.name
        ).where(
            KnowledgeNode.collection_id == collection_id,
            KnowledgeNode.kind == KnowledgeNodeKind.SEMANTIC_ENTITY,
        )
    )
    nodes = node_result.all()
    edge_kinds = SEMANTIC_EDGE_KINDS

    if not nodes:
        # No semantic entities = no GraphRAG community detection
        # Do NOT fall back to code symbols - that defeats the purpose
        logger.warning(
            "No semantic entities found for collection %s. "
            "Run LLM entity extraction first. Skipping community detection.",
            collection_id,
        )
        return result

    logger.info("Using %d semantic entities for Leiden community detection", len(nodes))

    # Build node mappings
    node_id_to_idx: dict[UUID, int] = {}
    idx_to_node_id: dict[int, UUID] = {}
    idx_to_node_key: dict[int, str] = {}
    idx_to_node_name: dict[int, str] = {}

    for idx, (node_id, natural_key, _kind, name) in enumerate(nodes):
        node_id_to_idx[node_id] = idx
        idx_to_node_id[idx] = node_id
        idx_to_node_key[idx] = natural_key
        idx_to_node_name[idx] = name or natural_key

    # Load edges
    edge_result = await session.execute(
        select(
            KnowledgeEdge.source_node_id,
            KnowledgeEdge.target_node_id,
            KnowledgeEdge.kind,
            KnowledgeEdge.meta,
        ).where(
            KnowledgeEdge.collection_id == collection_id,
            KnowledgeEdge.kind.in_(edge_kinds),
        )
    )
    edges = edge_result.all()

    # Build igraph Graph
    edge_list: list[tuple[int, int]] = []
    edge_weights: list[float] = []

    for src_id, dst_id, _kind, meta in edges:
        src_idx = node_id_to_idx.get(src_id)
        dst_idx = node_id_to_idx.get(dst_id)

        if src_idx is None or dst_idx is None:
            continue

        # Weight: use meta.strength for semantic relationships, default to 1.0
        weight = meta.get("strength", 1.0) if meta else 1.0

        edge_list.append((src_idx, dst_idx))
        edge_weights.append(weight)

    if not edge_list:
        logger.debug("No edges found for collection %s", collection_id)
        # Create single community with all nodes
        single_comm = Community(id=0, level=0, size=len(nodes))
        single_comm.node_ids = [n[0] for n in nodes]
        single_comm.node_keys = [n[1] for n in nodes]
        result.levels[0] = [single_comm]
        for node_id, _, _, _ in nodes:
            result.node_membership[node_id] = {0: 0}
        return result

    # Create igraph Graph
    g = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
    g.es["weight"] = edge_weights

    # Run Leiden at each resolution level
    for level, resolution in enumerate(resolutions):
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )

        # Record modularity
        result.modularity[level] = partition.modularity

        # Build communities for this level
        communities: list[Community] = []
        community_nodes: dict[int, list[int]] = {}

        for node_idx, comm_id in enumerate(partition.membership):
            if comm_id not in community_nodes:
                community_nodes[comm_id] = []
            community_nodes[comm_id].append(node_idx)

        # Create Community objects
        for comm_id, node_indices in sorted(community_nodes.items()):
            comm = Community(
                id=comm_id,
                level=level,
                size=len(node_indices),
            )
            comm.node_ids = [idx_to_node_id[idx] for idx in node_indices]
            comm.node_keys = [idx_to_node_key[idx] for idx in node_indices]
            communities.append(comm)

            # Update node membership
            for node_idx in node_indices:
                node_id = idx_to_node_id[node_idx]
                if node_id not in result.node_membership:
                    result.node_membership[node_id] = {}
                result.node_membership[node_id][level] = comm_id

        result.levels[level] = communities
        logger.info(
            "Level %d (resolution=%.2f): %d communities, modularity=%.4f",
            level,
            resolution,
            len(communities),
            partition.modularity,
        )

    return result


async def persist_communities(
    session: AsyncSession,
    collection_id: UUID,
    result: HierarchicalCommunities,
) -> dict:
    """Persist hierarchical community structure to database.

    Replaces all existing communities for the collection.

    Args:
        session: Database session
        collection_id: Collection UUID
        result: Hierarchical community detection result

    Returns:
        Stats dict with counts
    """
    stats = {"communities_created": 0, "members_created": 0}

    # Delete existing communities for this collection
    await session.execute(
        delete(KnowledgeCommunity).where(KnowledgeCommunity.collection_id == collection_id)
    )

    # Create communities at each level
    for level, communities in result.levels.items():
        for comm in communities:
            # Generate title from top members
            title = _generate_title(comm.node_keys, comm.size)

            db_community = KnowledgeCommunity(
                collection_id=collection_id,
                level=level,
                natural_key=comm.natural_key,
                title=title,
                meta={
                    "size": comm.size,
                    "modularity": result.modularity.get(level, 0.0),
                    "top_members": comm.node_keys[:10],
                },
            )
            session.add(db_community)
            await session.flush()
            stats["communities_created"] += 1

            # Create memberships
            for node_id in comm.node_ids:
                # Compute membership score based on node's internal connectivity
                # For now, use 1.0 (Leiden doesn't provide soft membership)
                member = CommunityMember(
                    community_id=db_community.id,
                    node_id=node_id,
                    score=1.0,
                )
                session.add(member)
                stats["members_created"] += 1

    return stats


def _generate_title(node_keys: list[str], size: int) -> str:
    """Generate a descriptive title for a community.

    Args:
        node_keys: Natural keys of member nodes (entity:name format for GraphRAG)
        size: Total community size

    Returns:
        Human-readable title
    """
    if not node_keys:
        return f"Community ({size} members)"

    # Extract readable names from natural keys
    names: list[str] = []
    for key in node_keys[:3]:
        parts = key.split(":", 1)  # Split only on first colon
        if len(parts) >= 2:
            prefix, value = parts[0], parts[1]

            if prefix == "entity":
                # Semantic entity: entity:user_authentication -> "User Authentication"
                # Convert snake_case to Title Case
                name = value.replace("_", " ").title()
                if name and name not in names:
                    names.append(name)

    if names:
        title = ", ".join(names[:3])
        if size > 3:
            title += f" +{size - 3} more"
        return title

    return f"Community ({size} members)"
