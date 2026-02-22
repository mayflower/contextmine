import { useEffect, useMemo, useRef, useState } from 'react'
import cytoscape, { type Core } from 'cytoscape'

import GraphRagProcessModal from '../components/GraphRagProcessModal'
import type {
  CockpitLoadState,
  GraphRagCommunity,
  GraphRagCommunityMode,
  GraphRagEvidenceItem,
  GraphRagPathPayload,
  GraphRagProcessDetailPayload,
  GraphRagProcessSummary,
  TwinGraphResponse,
} from '../types'

interface GraphRagViewProps {
  graph: TwinGraphResponse
  state: CockpitLoadState
  error: string
  status: 'ready' | 'unavailable'
  reason: 'ok' | 'no_knowledge_graph'
  selectedNodeId: string
  communityMode: GraphRagCommunityMode
  communityId: string
  communities: GraphRagCommunity[]
  communitiesState: CockpitLoadState
  communitiesError: string
  path: GraphRagPathPayload | null
  pathState: CockpitLoadState
  pathError: string
  processes: GraphRagProcessSummary[]
  processesState: CockpitLoadState
  processesError: string
  processDetail: GraphRagProcessDetailPayload | null
  processDetailState: CockpitLoadState
  processDetailError: string
  evidenceItems: GraphRagEvidenceItem[]
  evidenceTotal: number
  evidenceNodeName: string
  evidenceState: CockpitLoadState
  evidenceError: string
  onSelectNodeId: (nodeId: string) => void
  onTracePath: (fromNodeId: string, toNodeId: string, maxHops: number) => Promise<GraphRagPathPayload | null>
  onLoadProcessDetail: (processId: string) => Promise<GraphRagProcessDetailPayload | null>
  onRetry: () => void
}

const COMMUNITY_COLORS = [
  '#ef4444',
  '#f97316',
  '#eab308',
  '#22c55e',
  '#06b6d4',
  '#3b82f6',
  '#8b5cf6',
  '#d946ef',
  '#ec4899',
  '#14b8a6',
]

function colorForKind(kind: string): string {
  const normalized = kind.toLowerCase()
  if (normalized.includes('file')) return '#2563eb'
  if (normalized.includes('symbol')) return '#16a34a'
  if (normalized.includes('rule')) return '#d97706'
  if (normalized.includes('db')) return '#9333ea'
  if (normalized.includes('api')) return '#0f766e'
  if (normalized.includes('semantic')) return '#0ea5e9'
  return '#475569'
}

function pairKey(source: string, target: string): string {
  return `${source}->${target}`
}

export default function GraphRagView({
  graph,
  state,
  error,
  status,
  reason,
  selectedNodeId,
  communityMode,
  communityId,
  communities,
  communitiesState,
  communitiesError,
  path,
  pathState,
  pathError,
  processes,
  processesState,
  processesError,
  processDetail,
  processDetailState,
  processDetailError,
  evidenceItems,
  evidenceTotal,
  evidenceNodeName,
  evidenceState,
  evidenceError,
  onSelectNodeId,
  onTracePath,
  onLoadProcessDetail,
  onRetry,
}: GraphRagViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Core | null>(null)
  const [showLabels, setShowLabels] = useState(true)
  const [pathFrom, setPathFrom] = useState('')
  const [pathTo, setPathTo] = useState('')
  const [pathHops, setPathHops] = useState(6)
  const [focusedProcessId, setFocusedProcessId] = useState('')
  const [focusedProcessNodeIds, setFocusedProcessNodeIds] = useState<Set<string>>(new Set())
  const [modalOpen, setModalOpen] = useState(false)

  useEffect(() => {
    if (!pathFrom && selectedNodeId) {
      setPathFrom(selectedNodeId)
    }
  }, [pathFrom, selectedNodeId])

  const communityColorById = useMemo(() => {
    const map = new Map<string, string>()
    communities.forEach((community, index) => {
      map.set(community.id, COMMUNITY_COLORS[index % COMMUNITY_COLORS.length])
    })
    return map
  }, [communities])

  const pathNodeIds = useMemo(
    () => new Set(path?.path.nodes.map((node) => node.id) || []),
    [path],
  )
  const pathEdgeKeys = useMemo(() => {
    const keys = new Set<string>()
    ;(path?.path.edges || []).forEach((edge) => {
      keys.add(pairKey(edge.source_node_id, edge.target_node_id))
      keys.add(pairKey(edge.target_node_id, edge.source_node_id))
    })
    return keys
  }, [path])

  useEffect(() => {
    if (!containerRef.current || graph.nodes.length === 0 || state === 'loading') {
      return
    }

    if (graphRef.current) {
      graphRef.current.destroy()
      graphRef.current = null
    }

    const neighborIds = new Set<string>()
    if (selectedNodeId) {
      graph.edges.forEach((edge) => {
        if (edge.source_node_id === selectedNodeId) neighborIds.add(edge.target_node_id)
        if (edge.target_node_id === selectedNodeId) neighborIds.add(edge.source_node_id)
      })
    }

    const hasPathFocus = pathNodeIds.size > 0
    const hasProcessFocus = focusedProcessNodeIds.size > 0
    const hasCommunityFocus = communityMode === 'focus' && communityId.trim().length > 0
    const hasNodeFocus = selectedNodeId.trim().length > 0

    const next = cytoscape({
      container: containerRef.current,
      elements: [
        ...graph.nodes.map((node) => {
          const community = String((node.meta?.community_id as string) || '')
          const inPath = pathNodeIds.has(node.id)
          const inProcess = focusedProcessNodeIds.has(node.id)
          const inFocusedCommunity = Boolean(hasCommunityFocus && community && community === communityId)
          const isSelected = selectedNodeId === node.id
          const isNeighbor = neighborIds.has(node.id)

          let dimmed = false
          if (hasPathFocus) {
            dimmed = !inPath
          } else if (hasProcessFocus) {
            dimmed = !inProcess
          } else if (hasCommunityFocus) {
            dimmed = !inFocusedCommunity
          } else if (hasNodeFocus) {
            dimmed = !isSelected && !isNeighbor
          }

          const classes: string[] = []
          if (isSelected) classes.push('selected-focus')
          if (isNeighbor) classes.push('focus-neighbor')
          if (inPath) classes.push('path-node')
          if (inProcess) classes.push('process-node')
          if (dimmed) classes.push('dimmed')

          const color =
            communityMode !== 'none' && community
              ? communityColorById.get(community) || colorForKind(node.kind)
              : colorForKind(node.kind)

          return {
            data: {
              id: node.id,
              label: node.name,
              kind: node.kind,
              color,
              community,
            },
            classes: classes.join(' '),
          }
        }),
        ...graph.edges.map((edge) => {
          const edgeInPath = pathEdgeKeys.has(pairKey(edge.source_node_id, edge.target_node_id))
          const edgeInProcess =
            focusedProcessNodeIds.has(edge.source_node_id) && focusedProcessNodeIds.has(edge.target_node_id)
          const edgeInCommunity =
            !hasCommunityFocus
              ? true
              : graph.nodes.some(
                  (node) =>
                    node.id === edge.source_node_id &&
                    String(node.meta?.community_id || '') === communityId,
                ) &&
                graph.nodes.some(
                  (node) =>
                    node.id === edge.target_node_id &&
                    String(node.meta?.community_id || '') === communityId,
                )
          const edgeTouchesSelected =
            selectedNodeId &&
            (edge.source_node_id === selectedNodeId || edge.target_node_id === selectedNodeId)

          let dimmed = false
          if (hasPathFocus) {
            dimmed = !edgeInPath
          } else if (hasProcessFocus) {
            dimmed = !edgeInProcess
          } else if (hasCommunityFocus) {
            dimmed = !edgeInCommunity
          } else if (hasNodeFocus) {
            dimmed = !edgeTouchesSelected
          }

          const classes: string[] = []
          if (edgeInPath) classes.push('path-edge')
          if (edgeInProcess) classes.push('process-edge')
          if (edgeTouchesSelected) classes.push('focus-edge')
          if (dimmed) classes.push('dimmed')

          return {
            data: {
              id: edge.id,
              source: edge.source_node_id,
              target: edge.target_node_id,
              kind: edge.kind,
            },
            classes: classes.join(' '),
          }
        }),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            color: '#0f172a',
            label: showLabels ? 'data(label)' : '',
            'font-size': 9,
            width: 20,
            height: 20,
            'text-wrap': 'ellipsis',
            'text-max-width': '140px',
            'border-width': 1,
            'border-color': '#0f172a',
            opacity: 0.96,
          },
        },
        {
          selector: 'node.selected-focus',
          style: {
            'border-width': 4,
            'border-color': '#f59e0b',
            width: 28,
            height: 28,
            opacity: 1,
          },
        },
        {
          selector: 'node.focus-neighbor',
          style: {
            'border-width': 2,
            'border-color': '#3b82f6',
            width: 22,
            height: 22,
            opacity: 0.96,
          },
        },
        {
          selector: 'node.path-node',
          style: {
            'border-width': 3,
            'border-color': '#059669',
            opacity: 1,
          },
        },
        {
          selector: 'node.process-node',
          style: {
            'border-width': 3,
            'border-color': '#7c3aed',
            opacity: 1,
          },
        },
        {
          selector: 'node.dimmed',
          style: {
            opacity: 0.22,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1,
            'line-color': '#94a3b8',
            'target-arrow-color': '#94a3b8',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            opacity: 0.74,
          },
        },
        {
          selector: 'edge.focus-edge',
          style: {
            width: 2.5,
            'line-color': '#2563eb',
            'target-arrow-color': '#2563eb',
            opacity: 0.95,
          },
        },
        {
          selector: 'edge.path-edge',
          style: {
            width: 3,
            'line-color': '#059669',
            'target-arrow-color': '#059669',
            opacity: 0.98,
          },
        },
        {
          selector: 'edge.process-edge',
          style: {
            width: 3,
            'line-color': '#7c3aed',
            'target-arrow-color': '#7c3aed',
            opacity: 0.98,
          },
        },
        {
          selector: 'edge.dimmed',
          style: {
            opacity: 0.14,
            width: 0.8,
          },
        },
      ],
      layout: {
        name: graph.nodes.length > 1200 ? 'breadthfirst' : 'cose',
        animate: false,
      },
    })

    next.on('tap', 'node', (event) => {
      onSelectNodeId(String(event.target.id()))
    })

    graphRef.current = next

    return () => {
      next.destroy()
      graphRef.current = null
    }
  }, [
    graph,
    onSelectNodeId,
    selectedNodeId,
    showLabels,
    state,
    communityColorById,
    communityId,
    communityMode,
    focusedProcessNodeIds,
    pathEdgeKeys,
    pathNodeIds,
  ])

  const processGroups = useMemo(() => {
    const cross = processes.filter((process) => process.process_type === 'cross_community')
    const intra = processes.filter((process) => process.process_type === 'intra_community')
    return {
      cross: cross.sort((a, b) => b.step_count - a.step_count),
      intra: intra.sort((a, b) => b.step_count - a.step_count),
    }
  }, [processes])

  const activeModalProcessId = processDetail?.process.id || ''

  if (state === 'loading' && graph.nodes.length === 0) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-graphrag" role="tabpanel">
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && graph.nodes.length === 0) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-graphrag" role="tabpanel">
        <h3>GraphRAG request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  const showGuidedEmpty = status === 'unavailable' || graph.total_nodes === 0

  return (
    <section className="cockpit2-panel cockpit2-graphrag-panel" id="cockpit-panel-graphrag" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>GraphRAG graph</h3>
        <p className="muted">
          Nodes: {graph.nodes.length} / Total: {graph.total_nodes} • Edges: {graph.edges.length}
        </p>
      </div>

      {!showGuidedEmpty ? (
        <section className="cockpit2-graphrag-community-bar">
          <h4>Communities</h4>
          {communitiesState === 'loading' ? <p className="muted">Loading communities…</p> : null}
          {communitiesError ? <p className="muted">{communitiesError}</p> : null}
          {communitiesState !== 'loading' && communities.length === 0 ? (
            <p className="muted">No communities detected yet.</p>
          ) : null}
          {communities.length > 0 ? (
            <div className="cockpit2-chip-row">
              {communities.slice(0, 24).map((community) => {
                const isActive = communityId === community.id
                return (
                  <button
                    key={community.id}
                    type="button"
                    className={`secondary cockpit2-community-chip ${isActive ? 'active' : ''}`}
                    onClick={() => onSelectNodeId(community.sample_nodes[0]?.id || '')}
                    title={`${community.label} • Cohesion ${community.cohesion.toFixed(2)}`}
                  >
                    <span
                      className="cockpit2-community-dot"
                      style={{ backgroundColor: communityColorById.get(community.id) || '#94a3b8' }}
                    />
                    {community.label}
                  </button>
                )
              })}
            </div>
          ) : null}
        </section>
      ) : null}

      {showGuidedEmpty ? (
        <section className="cockpit2-empty">
          <h3>No knowledge graph available yet</h3>
          <p>
            {reason === 'no_knowledge_graph'
              ? 'Knowledge graph data has not been generated for this collection. Run sync/build pipeline to populate GraphRAG nodes and evidence.'
              : 'No graph nodes are currently available for this selection.'}
          </p>
        </section>
      ) : (
        <>
          <div className="cockpit2-graph-toolbar">
            <button type="button" className="secondary" onClick={() => graphRef.current?.fit()}>
              Fit view
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => graphRef.current?.layout({ name: 'cose', animate: false }).run()}
            >
              Reset layout
            </button>
            <button type="button" className="secondary" onClick={() => setShowLabels((prev) => !prev)}>
              {showLabels ? 'Hide labels' : 'Show labels'}
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setFocusedProcessId('')
                setFocusedProcessNodeIds(new Set())
                setPathFrom('')
                setPathTo('')
              }}
            >
              Reset focus
            </button>
          </div>

          <div ref={containerRef} className="cockpit2-canvas cockpit2-graphrag-graph" aria-label="GraphRAG graph" />
        </>
      )}

      <section className="cockpit2-graphrag-path">
        <div className="cockpit2-panel-header-row">
          <h4>Trace path</h4>
          {path?.status ? <p className="muted">Status: {path.status}</p> : null}
        </div>
        <div className="cockpit2-graph-toolbar">
          <label>
            From
            <input value={pathFrom} onChange={(event) => setPathFrom(event.target.value)} />
          </label>
          <label>
            To
            <input value={pathTo} onChange={(event) => setPathTo(event.target.value)} />
          </label>
          <label>
            Max hops
            <input
              type="number"
              min={1}
              max={20}
              value={pathHops}
              onChange={(event) => setPathHops(Math.max(1, Math.min(20, Number(event.target.value) || 1)))}
            />
          </label>
          <button type="button" onClick={() => onTracePath(pathFrom, pathTo, pathHops)}>
            {pathState === 'loading' ? 'Tracing…' : 'Trace'}
          </button>
        </div>
        {pathError ? <p className="muted">{pathError}</p> : null}
        {path?.path.nodes && path.path.nodes.length > 0 ? (
          <div className="cockpit2-process-step-list">
            {path.path.nodes.map((node, index) => (
              <button
                key={`${node.id}-${index}`}
                type="button"
                className="secondary"
                onClick={() => onSelectNodeId(node.id)}
              >
                {index + 1}. {node.name}
              </button>
            ))}
          </div>
        ) : null}
      </section>

      <section className="cockpit2-graphrag-processes">
        <div className="cockpit2-panel-header-row">
          <h4>Processes</h4>
          <p className="muted">{processes.length} detected</p>
        </div>
        {processesState === 'loading' ? <p className="muted">Loading processes…</p> : null}
        {processesError ? <p className="muted">{processesError}</p> : null}
        {processes.length > 0 ? (
          <>
            <details open>
              <summary>Cross-community ({processGroups.cross.length})</summary>
              <div className="cockpit2-process-list">
                {processGroups.cross.map((process) => (
                  <article key={process.id}>
                    <strong>{process.label}</strong>
                    <span className="muted">{process.step_count} steps</span>
                    <div className="actions">
                      <button
                        type="button"
                        className="secondary"
                        onClick={async () => {
                          const detail = await onLoadProcessDetail(process.id)
                          if (detail) setModalOpen(true)
                        }}
                      >
                        View flow
                      </button>
                      <button
                        type="button"
                        className="secondary"
                        onClick={async () => {
                          if (focusedProcessId === process.id) {
                            setFocusedProcessId('')
                            setFocusedProcessNodeIds(new Set())
                            return
                          }
                          const detail = await onLoadProcessDetail(process.id)
                          if (!detail) return
                          setFocusedProcessId(process.id)
                          setFocusedProcessNodeIds(new Set(detail.steps.map((step) => step.node_id)))
                        }}
                      >
                        {focusedProcessId === process.id ? 'Clear focus' : 'Focus'}
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </details>
            <details>
              <summary>Intra-community ({processGroups.intra.length})</summary>
              <div className="cockpit2-process-list">
                {processGroups.intra.map((process) => (
                  <article key={process.id}>
                    <strong>{process.label}</strong>
                    <span className="muted">{process.step_count} steps</span>
                    <div className="actions">
                      <button
                        type="button"
                        className="secondary"
                        onClick={async () => {
                          const detail = await onLoadProcessDetail(process.id)
                          if (detail) setModalOpen(true)
                        }}
                      >
                        View flow
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </details>
          </>
        ) : null}
        {processDetailError ? <p className="muted">{processDetailError}</p> : null}
        {processDetailState === 'loading' ? <p className="muted">Loading process detail…</p> : null}
      </section>

      <section className="cockpit2-graphrag-evidence">
        <div className="cockpit2-panel-header-row">
          <h4>Indexed text</h4>
          <p className="muted">
            {evidenceNodeName ? `${evidenceNodeName} • ` : ''}
            {evidenceTotal} items
          </p>
        </div>

        {!selectedNodeId ? (
          <div className="cockpit2-empty">
            <p>Select a graph node to inspect indexed evidence text.</p>
          </div>
        ) : null}

        {selectedNodeId && evidenceState === 'loading' ? (
          <div className="cockpit2-skeleton-grid">
            <div className="cockpit2-skeleton-card" />
            <div className="cockpit2-skeleton-card" />
          </div>
        ) : null}

        {selectedNodeId && evidenceState === 'error' ? (
          <div className="cockpit2-alert error inline">
            <p>{evidenceError}</p>
            <button type="button" onClick={onRetry}>Retry</button>
          </div>
        ) : null}

        {selectedNodeId && evidenceState !== 'loading' && evidenceState !== 'error' && evidenceItems.length === 0 ? (
          <div className="cockpit2-empty">
            <p>No indexed evidence was found for this node.</p>
          </div>
        ) : null}

        {selectedNodeId && evidenceItems.length > 0 ? (
          <div className="cockpit2-graphrag-evidence-list">
            {evidenceItems.map((item) => (
              <article key={item.evidence_id} className="cockpit2-graphrag-evidence-item">
                <header>
                  <strong>{item.file_path}</strong>
                  <span>
                    L{item.start_line}-L{item.end_line} • {item.text_source}
                  </span>
                </header>
                {item.text ? <pre>{item.text}</pre> : <p className="muted">No text available.</p>}
              </article>
            ))}
          </div>
        ) : null}
      </section>

      {modalOpen && processDetail ? (
        <GraphRagProcessModal
          detail={processDetail}
          focused={focusedProcessId === activeModalProcessId}
          onSelectNodeId={onSelectNodeId}
          onToggleFocus={() => {
            if (focusedProcessId === activeModalProcessId) {
              setFocusedProcessId('')
              setFocusedProcessNodeIds(new Set())
              return
            }
            setFocusedProcessId(activeModalProcessId)
            setFocusedProcessNodeIds(new Set(processDetail.steps.map((step) => step.node_id)))
          }}
          onClose={() => setModalOpen(false)}
        />
      ) : null}
    </section>
  )
}
