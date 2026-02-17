import type { GraphFilters, TwinGraphEdge, TwinGraphNode, TwinGraphResponse } from './types'

function nodeSearchText(node: TwinGraphNode): string {
  const metaText = Object.values(node.meta || {})
    .map((value) => String(value))
    .join(' ')
  return `${node.id} ${node.natural_key} ${node.kind} ${node.name} ${metaText}`.toLowerCase()
}

export function filterGraph(graph: TwinGraphResponse, filters: GraphFilters): TwinGraphResponse {
  const query = filters.query.trim().toLowerCase()
  const edgeKindSet = new Set(filters.edgeKinds.map((entry) => entry.toLowerCase()))
  const filteredEdges: TwinGraphEdge[] =
    edgeKindSet.size > 0
      ? graph.edges.filter((edge) => edgeKindSet.has(edge.kind.toLowerCase()))
      : graph.edges

  const connectedNodeIds = new Set<string>()
  for (const edge of filteredEdges) {
    connectedNodeIds.add(edge.source_node_id)
    connectedNodeIds.add(edge.target_node_id)
  }

  let nodes = graph.nodes
  if (query) {
    nodes = nodes.filter((node) => nodeSearchText(node).includes(query))
  }

  if (filters.hideIsolated) {
    nodes = nodes.filter((node) => connectedNodeIds.has(node.id))
  }

  const nodeIds = new Set(nodes.map((node) => node.id))
  const edges = filteredEdges.filter(
    (edge) => nodeIds.has(edge.source_node_id) && nodeIds.has(edge.target_node_id),
  )

  return {
    ...graph,
    nodes,
    edges,
    total_nodes: graph.total_nodes,
  }
}

export function graphKinds(graph: TwinGraphResponse): { nodeKinds: string[]; edgeKinds: string[] } {
  const nodeKinds = [...new Set(graph.nodes.map((node) => node.kind))].sort()
  const edgeKinds = [...new Set(graph.edges.map((edge) => edge.kind))].sort()
  return { nodeKinds, edgeKinds }
}

export function resolveNodeId(graph: TwinGraphResponse, raw: string): string {
  if (!raw.trim()) return ''
  const trimmed = raw.trim().toLowerCase()
  const exact = graph.nodes.find((node) => node.id.toLowerCase() === trimmed)
  if (exact) return exact.id
  const byKey = graph.nodes.find((node) => node.natural_key.toLowerCase() === trimmed)
  if (byKey) return byKey.id
  const fuzzy = graph.nodes.find((node) => nodeSearchText(node).includes(trimmed))
  return fuzzy?.id || ''
}
