import { useMemo, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Edge as RFEdge,
  type Node as RFNode,
  type ReactFlowInstance,
} from 'reactflow'

import type { CockpitLayer, CockpitLoadState, TwinGraphResponse } from '../types'

interface TopologyViewProps {
  graph: TwinGraphResponse
  state: CockpitLoadState
  error: string
  layer: CockpitLayer
  density: number
  onDensityChange: (density: number) => void
  onSwitchToCodeLayer: () => void
  onRetry: () => void
}

function layerLabel(layer: CockpitLayer): string {
  if (layer === 'portfolio_system') return 'Portfolio / System'
  if (layer === 'domain_container') return 'Domain / Container'
  if (layer === 'component_interface') return 'Component / Interface'
  return 'Code / Controlflow'
}

export default function TopologyView({
  graph,
  state,
  error,
  layer,
  density,
  onDensityChange,
  onSwitchToCodeLayer,
  onRetry,
}: TopologyViewProps) {
  const [layoutTick, setLayoutTick] = useState(0)
  const [showLabels, setShowLabels] = useState(true)
  const [instance, setInstance] = useState<ReactFlowInstance | null>(null)

  const columns = density === 800 ? 4 : density === 1200 ? 6 : 9
  const isArchitectureProjection = graph.projection === 'architecture'

  const nodes = useMemo<RFNode[]>(() => {
    if (!isArchitectureProjection) {
      return graph.nodes.map((node, index) => ({
        id: node.id,
        data: { label: showLabels ? `${node.name} (${node.kind})` : node.kind },
        position: {
          x: (index % columns) * 260,
          y: Math.floor(index / columns) * 130 + layoutTick,
        },
        style: {
          width: 230,
          fontSize: 12,
          borderRadius: 10,
          border: '1px solid #d7dbe7',
          background: '#ffffff',
        },
      }))
    }

    const domainNames = [...new Set(graph.nodes.map((node) => String(node.meta?.domain || 'Unmapped')))]
    const domainNodes: RFNode[] = domainNames.map((domain, index) => ({
      id: `domain:${domain}`,
      data: { label: domain },
      position: {
        x: (index % 3) * 640,
        y: Math.floor(index / 3) * 440 + layoutTick,
      },
      draggable: false,
      selectable: false,
      style: {
        width: 600,
        height: 400,
        borderRadius: 14,
        border: '1px solid #c9d5f3',
        background: '#f1f5ff',
        color: '#233876',
        fontWeight: 700,
        fontSize: 13,
      },
    }))

    const domainSlots = new Map<string, number>()
    const childNodes: RFNode[] = graph.nodes.map((node) => {
      const domain = String(node.meta?.domain || 'Unmapped')
      const domainId = `domain:${domain}`
      const localIndex = domainSlots.get(domain) || 0
      domainSlots.set(domain, localIndex + 1)
      const members = Number(node.meta?.member_count || 0)
      const label = showLabels ? `${node.name}${members ? ` (${members})` : ''}` : node.kind
      return {
        id: node.id,
        parentNode: domainId,
        extent: 'parent',
        data: { label },
        position: {
          x: (localIndex % 2) * 280 + 24,
          y: Math.floor(localIndex / 2) * 96 + 48,
        },
        style: {
          width: 250,
          fontSize: 12,
          borderRadius: 10,
          border: '1px solid #9db4eb',
          background: '#ffffff',
        },
      }
    })

    return [...domainNodes, ...childNodes]
  }, [graph.nodes, showLabels, columns, layoutTick, isArchitectureProjection])

  const edges = useMemo<RFEdge[]>(() => {
    return graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source_node_id,
      target: edge.target_node_id,
      label: showLabels
        ? isArchitectureProjection
          ? `depends_on (${edge.meta?.weight || 1})`
          : edge.kind
        : undefined,
      type: 'smoothstep',
    }))
  }, [graph.edges, showLabels, isArchitectureProjection])

  if (state === 'loading' && graph.nodes.length === 0) {
    return (
      <div className="cockpit2-skeleton-grid" id="cockpit-panel-topology" role="tabpanel">
        <div className="cockpit2-skeleton-card tall" />
      </div>
    )
  }

  if (state === 'error' && graph.nodes.length === 0) {
    return (
      <section className="cockpit2-alert error" id="cockpit-panel-topology" role="tabpanel">
        <h3>Topology request failed</h3>
        <p>{error}</p>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    )
  }

  return (
    <section className="cockpit2-panel" id="cockpit-panel-topology" role="tabpanel">
      {error ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      ) : null}

      <div className="cockpit2-panel-header-row">
        <h3>Topology graph</h3>
        <p className="muted">
          Nodes: {graph.nodes.length} / Total: {graph.total_nodes} • Edges: {graph.edges.length}
          {graph.projection ? ` • Projection: ${graph.projection}` : ''}
          {graph.entity_level ? ` • Level: ${graph.entity_level}` : ''}
        </p>
      </div>

      <div className="cockpit2-graph-toolbar">
        <button type="button" className="secondary" onClick={() => instance?.fitView({ duration: 300 })}>
          Fit view
        </button>
        <button type="button" className="secondary" onClick={() => setLayoutTick((prev) => prev + 1)}>
          Reset layout
        </button>
        <button type="button" className="secondary" onClick={() => setShowLabels((prev) => !prev)}>
          {showLabels ? 'Hide labels' : 'Show labels'}
        </button>

        <label>
          Density
          <select value={density} onChange={(event) => onDensityChange(Number(event.target.value))}>
            <option value={800}>Focused</option>
            <option value={1200}>Balanced</option>
            <option value={2500}>Dense</option>
          </select>
        </label>
      </div>

      <div className="cockpit2-canvas">
        {graph.nodes.length > 0 ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            onInit={setInstance}
            nodesDraggable={false}
            panOnDrag
            zoomOnScroll
            aria-label="Topology graph"
          >
            <Controls />
            <MiniMap pannable zoomable />
            <Background gap={20} size={1} />
          </ReactFlow>
        ) : (
          <section className="cockpit2-empty">
            <h3>No nodes for this layer</h3>
            {graph.total_nodes > 0 ? (
              <>
                <p>
                  The selected layer (<strong>{layerLabel(layer)}</strong>) is currently empty for this scenario.
                </p>
                {layer !== 'code_controlflow' ? (
                  <button type="button" onClick={onSwitchToCodeLayer}>
                    Switch to Code / Controlflow
                  </button>
                ) : null}
              </>
            ) : (
              <p>
                This scenario has no extracted twin nodes yet. Run sync again and verify SCIP indexing produced
                symbols.
              </p>
            )}
          </section>
        )}
      </div>
    </section>
  )
}
