import { useEffect, useRef, useState } from 'react'
import cytoscape, { type Core } from 'cytoscape'

import type {
  CockpitLoadState,
  GraphRagEvidenceItem,
  TwinGraphResponse,
} from '../types'

interface GraphRagViewProps {
  graph: TwinGraphResponse
  state: CockpitLoadState
  error: string
  status: 'ready' | 'unavailable'
  reason: 'ok' | 'no_knowledge_graph'
  selectedNodeId: string
  evidenceItems: GraphRagEvidenceItem[]
  evidenceTotal: number
  evidenceNodeName: string
  evidenceState: CockpitLoadState
  evidenceError: string
  onSelectNodeId: (nodeId: string) => void
  onRetry: () => void
}

function colorForKind(kind: string): string {
  const normalized = kind.toLowerCase()
  if (normalized.includes('file')) return '#2563eb'
  if (normalized.includes('symbol')) return '#16a34a'
  if (normalized.includes('rule')) return '#d97706'
  if (normalized.includes('db')) return '#9333ea'
  if (normalized.includes('api')) return '#0f766e'
  return '#475569'
}

export default function GraphRagView({
  graph,
  state,
  error,
  status,
  reason,
  selectedNodeId,
  evidenceItems,
  evidenceTotal,
  evidenceNodeName,
  evidenceState,
  evidenceError,
  onSelectNodeId,
  onRetry,
}: GraphRagViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Core | null>(null)
  const [showLabels, setShowLabels] = useState(true)

  useEffect(() => {
    if (!containerRef.current || graph.nodes.length === 0 || state === 'loading') {
      return
    }

    if (graphRef.current) {
      graphRef.current.destroy()
      graphRef.current = null
    }

    const next = cytoscape({
      container: containerRef.current,
      elements: [
        ...graph.nodes.map((node) => ({
          data: {
            id: node.id,
            label: node.name,
            kind: node.kind,
            selected: selectedNodeId === node.id ? 1 : 0,
            color: colorForKind(node.kind),
          },
        })),
        ...graph.edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source_node_id,
            target: edge.target_node_id,
            kind: edge.kind,
          },
        })),
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
          },
        },
        {
          selector: 'node[selected = 1]',
          style: {
            'border-width': 3,
            'border-color': '#f59e0b',
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
  }, [graph, onSelectNodeId, selectedNodeId, showLabels, state])

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
            <button
              type="button"
              className="secondary"
              onClick={() => setShowLabels((prev) => !prev)}
            >
              {showLabels ? 'Hide labels' : 'Show labels'}
            </button>
          </div>

          <div ref={containerRef} className="cockpit2-canvas cockpit2-graphrag-graph" aria-label="GraphRAG graph" />
        </>
      )}

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
    </section>
  )
}
