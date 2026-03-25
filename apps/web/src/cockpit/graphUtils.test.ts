import { describe, expect, it } from 'vitest'

import { filterGraph, graphKinds, resolveNodeId } from './graphUtils'
import type { GraphFilters, TwinGraphResponse } from './types'

function makeGraph(overrides?: Partial<TwinGraphResponse>): TwinGraphResponse {
  return {
    nodes: [
      { id: 'n1', natural_key: 'service:auth', kind: 'CONTAINER', name: 'AuthService', meta: {} },
      { id: 'n2', natural_key: 'service:billing', kind: 'CONTAINER', name: 'BillingService', meta: {} },
      { id: 'n3', natural_key: 'db:users', kind: 'DB_TABLE', name: 'Users', meta: {} },
    ],
    edges: [
      { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', kind: 'CALLS', meta: {} },
      { id: 'e2', source_node_id: 'n1', target_node_id: 'n3', kind: 'READS', meta: {} },
    ],
    page: 0,
    limit: 100,
    total_nodes: 3,
    ...overrides,
  }
}

const defaultFilters: GraphFilters = {
  query: '',
  hideIsolated: false,
  edgeKinds: [],
  includeKinds: [],
  excludeKinds: [],
}

describe('filterGraph', () => {
  it('returns all nodes and edges with default filters', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, defaultFilters)
    expect(result.nodes).toHaveLength(3)
    expect(result.edges).toHaveLength(2)
  })

  it('filters nodes by query (case-insensitive)', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, query: 'billing' })
    expect(result.nodes).toHaveLength(1)
    expect(result.nodes[0].id).toBe('n2')
  })

  it('filters nodes by query matching natural_key', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, query: 'service:auth' })
    expect(result.nodes).toHaveLength(1)
    expect(result.nodes[0].id).toBe('n1')
  })

  it('filters edges by edgeKinds', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, edgeKinds: ['CALLS'] })
    expect(result.edges).toHaveLength(1)
    expect(result.edges[0].kind).toBe('CALLS')
  })

  it('hides isolated nodes when hideIsolated is true', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, edgeKinds: ['CALLS'], hideIsolated: true })
    // Only n1 and n2 are connected via CALLS
    expect(result.nodes).toHaveLength(2)
    expect(result.nodes.map((n) => n.id).sort()).toEqual(['n1', 'n2'])
  })

  it('preserves total_nodes from original graph', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, query: 'billing' })
    expect(result.total_nodes).toBe(3)
  })

  it('removes edges whose nodes are filtered out', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, query: 'users' })
    expect(result.nodes).toHaveLength(1)
    expect(result.edges).toHaveLength(0)
  })

  it('edgeKinds filter is case-insensitive', () => {
    const graph = makeGraph()
    const result = filterGraph(graph, { ...defaultFilters, edgeKinds: ['calls'] })
    expect(result.edges).toHaveLength(1)
  })
})

describe('graphKinds', () => {
  it('returns sorted unique node and edge kinds', () => {
    const graph = makeGraph()
    const result = graphKinds(graph)
    expect(result.nodeKinds).toEqual(['CONTAINER', 'DB_TABLE'])
    expect(result.edgeKinds).toEqual(['CALLS', 'READS'])
  })

  it('returns empty arrays for empty graph', () => {
    const graph = makeGraph({ nodes: [], edges: [] })
    const result = graphKinds(graph)
    expect(result.nodeKinds).toEqual([])
    expect(result.edgeKinds).toEqual([])
  })
})

describe('resolveNodeId', () => {
  it('returns empty string for empty input', () => {
    const graph = makeGraph()
    expect(resolveNodeId(graph, '')).toBe('')
    expect(resolveNodeId(graph, '   ')).toBe('')
  })

  it('resolves by exact id match', () => {
    const graph = makeGraph()
    expect(resolveNodeId(graph, 'n1')).toBe('n1')
  })

  it('resolves by exact id match case-insensitive', () => {
    const graph = makeGraph()
    expect(resolveNodeId(graph, 'N1')).toBe('n1')
  })

  it('resolves by natural_key', () => {
    const graph = makeGraph()
    expect(resolveNodeId(graph, 'service:billing')).toBe('n2')
  })

  it('resolves by fuzzy name match', () => {
    const graph = makeGraph()
    expect(resolveNodeId(graph, 'authservice')).toBe('n1')
  })

  it('returns empty string when no match is found', () => {
    const graph = makeGraph()
    expect(resolveNodeId(graph, 'nonexistent')).toBe('')
  })
})
