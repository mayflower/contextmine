import { useEffect, useRef, useState } from 'react'
import cytoscape, { type Core } from 'cytoscape'

import type { CockpitLoadState, TwinGraphResponse } from '../types'

interface DeepDiveViewProps {
  graph: TwinGraphResponse
  state: CockpitLoadState
  error: string
  density: number
  onDensityChange: (density: number) => void
  onRetry: () => void
}

function getLayoutName(density: number): 'cose' | 'breadthfirst' {
  return density > 5000 ? 'breadthfirst' : 'cose'
}

export default function DeepDiveView({
  graph,
  state,
  error,
  density,
  onDensityChange,
  onRetry,
}: DeepDiveViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Core | null>(null)
  const [showLabels, setShowLabels] = useState(true)

  useEffect(() => {
    if (!containerRef.current || state === 'loading' || graph.nodes.length === 0) {
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
          },
        })),
        ...graph.edges.map((edge) => ({
          data: {
            id: edge.id,
            source: edge.source_node_id,
            target: edge.target_node_id,
            label: edge.kind,
          },
        })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#1d4ed8',
            color: '#0f172a',
            label: showLabels ? 'data(label)' : '',
            'font-size': 9,
            width: 20,
            height: 20,
            'text-valign': 'center',
            'text-halign': 'center',
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
        name: getLayoutName(density),
        animate: false,
      },
    })

    graphRef.current = next

    return () => {
      next.destroy()
      graphRef.current = null
    }
  }, [graph, state, showLabels, density])

  if (state === 'loading' && graph.nodes.length === 0) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-deep_dive" role="tabpanel">
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && graph.nodes.length === 0) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-deep_dive" role="tabpanel">
        <h3>Deep dive request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  return (
    <section className="cockpit2-panel" id="cockpit-panel-deep_dive" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Deep dive graph</h3>
        <p className="muted">Nodes: {graph.nodes.length} / Total: {graph.total_nodes} • Edges: {graph.edges.length}</p>
      </div>

      <div className="cockpit2-graph-toolbar">
        <button type="button" className="secondary" onClick={() => graphRef.current?.fit()}>
          Fit view
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => graphRef.current?.layout({ name: getLayoutName(density), animate: false }).run()}
        >
          Reset layout
        </button>
        <button type="button" className="secondary" onClick={() => setShowLabels((prev) => !prev)}>
          {showLabels ? 'Hide labels' : 'Show labels'}
        </button>

        <label>
          Density
          <select value={density} onChange={(event) => onDensityChange(Number(event.target.value))}>
            <option value={3000}>Focused</option>
            <option value={5000}>Balanced</option>
            <option value={8000}>Dense</option>
          </select>
        </label>
      </div>

      <div ref={containerRef} className="cockpit2-canvas deep" aria-label="Deep dive graph" />
    </section>
  )
}
