import { useEffect, useMemo, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Edge as RFEdge,
  type Node as RFNode,
  type ReactFlowInstance,
} from 'reactflow'

import { runElkLayout, runGridLayout, type LayoutEngine } from '../layout/layoutCore'
import type { CockpitLayer, CockpitLoadState, OverlayState, TwinGraphResponse } from '../types'

interface TopologyViewProps {
  graph: TwinGraphResponse
  state: CockpitLoadState
  error: string
  layer: CockpitLayer
  density: number
  layoutEngine: LayoutEngine
  elkEnabled: boolean
  overlay: OverlayState
  selectedNodeId: string
  onDensityChange: (density: number) => void
  onLayoutEngineChange: (engine: LayoutEngine) => void
  onSwitchToCodeLayer: () => void
  onSelectNodeId: (nodeId: string) => void
  onLayoutCompleted: (engine: LayoutEngine, durationMs: number, nodeCount: number) => void
  onRetry: () => void
}

function layerLabel(layer: CockpitLayer): string {
  if (layer === 'portfolio_system') return 'Portfolio / System'
  if (layer === 'domain_container') return 'Domain / Container'
  if (layer === 'component_interface') return 'Component / Interface'
  return 'Code / Controlflow'
}

function overlayColorForNode(overlay: OverlayState, naturalKey: string, fallbackName: string): string {
  if (overlay.mode === 'runtime') {
    const runtime = overlay.runtimeByNodeKey[naturalKey] || overlay.runtimeByNodeKey[fallbackName]
    const errorRate = Number(runtime?.error_rate || 0)
    if (errorRate >= 0.1) return '#dc2626'
    if (errorRate >= 0.03) return '#f59e0b'
    return '#2563eb'
  }
  if (overlay.mode === 'risk') {
    const risk = overlay.riskByNodeKey[naturalKey] || overlay.riskByNodeKey[fallbackName]
    const severity = Number(risk?.severity_score || 0)
    if (severity >= 8) return '#b91c1c'
    if (severity >= 4) return '#d97706'
    return '#1d4ed8'
  }
  return '#1d4ed8'
}

export default function TopologyView({
  graph,
  state,
  error,
  layer,
  density,
  layoutEngine,
  elkEnabled,
  overlay,
  selectedNodeId,
  onDensityChange,
  onLayoutEngineChange,
  onSwitchToCodeLayer,
  onSelectNodeId,
  onLayoutCompleted,
  onRetry,
}: TopologyViewProps) {
  const [showLabels, setShowLabels] = useState(true)
  const [instance, setInstance] = useState<ReactFlowInstance | null>(null)
  const [layoutStatus, setLayoutStatus] = useState<'idle' | 'coarse' | 'refined'>('idle')
  const [layoutPositions, setLayoutPositions] = useState<Record<string, { x: number; y: number }>>({})

  const columns = density === 800 ? 4 : density === 1200 ? 6 : 9
  const preferredEngine: LayoutEngine =
    graph.projection === 'architecture'
      ? graph.nodes.length < 100
        ? 'grid'
        : elkEnabled
          ? layoutEngine
          : 'grid'
      : 'grid'

  useEffect(() => {
    if (graph.nodes.length === 0) {
      setLayoutPositions({})
      setLayoutStatus('idle')
      return
    }

    const coarse = runGridLayout(graph.nodes.map((node) => ({ id: node.id })), columns)
    setLayoutPositions(coarse)
    setLayoutStatus(preferredEngine === 'grid' ? 'refined' : 'coarse')
    if (preferredEngine === 'grid') {
      return
    }

    let cancelled = false
    const startedAt = performance.now()

    const apply = async () => {
      try {
        if (graph.nodes.length > 1000) {
          const worker = new Worker(new URL('../layout/layoutWorker.ts', import.meta.url), {
            type: 'module',
          })
          worker.onmessage = (event: MessageEvent<{ ok: boolean; positions?: Record<string, { x: number; y: number }>; durationMs: number }>) => {
            worker.terminate()
            if (cancelled || !event.data.ok || !event.data.positions) {
              return
            }
            setLayoutPositions(event.data.positions)
            setLayoutStatus('refined')
            onLayoutCompleted(preferredEngine, event.data.durationMs, graph.nodes.length)
          }
          worker.postMessage({
            nodes: graph.nodes.map((node) => ({ id: node.id })),
            edges: graph.edges.map((edge) => ({ source: edge.source_node_id, target: edge.target_node_id })),
            engine: preferredEngine,
            columns,
          })
          return
        }

        const positions = await runElkLayout(
          graph.nodes.map((node) => ({ id: node.id })),
          graph.edges.map((edge) => ({ source: edge.source_node_id, target: edge.target_node_id })),
          preferredEngine,
        )
        if (cancelled) return
        const durationMs = performance.now() - startedAt
        setLayoutPositions(positions)
        setLayoutStatus('refined')
        onLayoutCompleted(preferredEngine, durationMs, graph.nodes.length)
      } catch {
        if (cancelled) return
        setLayoutStatus('refined')
      }
    }

    apply()
    return () => {
      cancelled = true
    }
  }, [graph.edges, graph.nodes, columns, preferredEngine, onLayoutCompleted])

  const nodes = useMemo<RFNode[]>(() => {
    return graph.nodes.map((node, index) => {
      const runtime = overlay.runtimeByNodeKey[node.natural_key] || overlay.runtimeByNodeKey[node.name]
      const isSelected = selectedNodeId === node.id
      const position = layoutPositions[node.id] || { x: (index % columns) * 260, y: Math.floor(index / columns) * 130 }
      return {
        id: node.id,
        data: {
          label: showLabels ? `${node.name} (${node.kind})` : node.kind,
        },
        position,
        style: {
          width: 230,
          fontSize: 12,
          borderRadius: 10,
          border: isSelected ? '2px solid #f59e0b' : `1px solid ${overlayColorForNode(overlay, node.natural_key, node.name)}`,
          background: '#ffffff',
          boxShadow: isSelected ? '0 0 0 3px rgba(245, 158, 11, 0.22)' : 'none',
          opacity: overlay.mode === 'runtime' && runtime?.error_rate === undefined ? 0.84 : 1,
        },
      }
    })
  }, [graph.nodes, showLabels, layoutPositions, columns, overlay, selectedNodeId])

  const edges = useMemo<RFEdge[]>(() => {
    const runtimeById = new Map<string, number>()
    for (const node of graph.nodes) {
      const runtime = overlay.runtimeByNodeKey[node.natural_key] || overlay.runtimeByNodeKey[node.name]
      runtimeById.set(node.id, Number(runtime?.latency_p95 || 0))
    }

    return graph.edges.map((edge) => {
      const latency = runtimeById.get(edge.source_node_id) || 0
      const width = overlay.mode === 'runtime' && latency > 0 ? Math.min(6, 1 + latency / 120) : 1
      return {
        id: edge.id,
        source: edge.source_node_id,
        target: edge.target_node_id,
        label: showLabels ? `${edge.kind}${edge.meta?.weight ? ` (${edge.meta.weight})` : ''}` : undefined,
        type: 'smoothstep',
        style: { strokeWidth: width },
      }
    })
  }, [graph.edges, graph.nodes, showLabels, overlay])

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
          {layoutStatus === 'coarse' ? ' • Refining layout…' : ''}
        </p>
      </div>

      <div className="cockpit2-graph-toolbar">
        <button type="button" className="secondary" onClick={() => instance?.fitView({ duration: 300 })}>
          Fit view
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

        {elkEnabled ? (
          <label>
            Layout
            <select value={layoutEngine} onChange={(event) => onLayoutEngineChange(event.target.value as LayoutEngine)}>
              <option value="grid">Grid</option>
              <option value="elk_layered">ELK layered</option>
              <option value="elk_force_like">ELK force-like</option>
            </select>
          </label>
        ) : null}
      </div>

      {overlay.mode !== 'none' ? (
        <div className="cockpit2-overlay-legend">
          <span>Overlay:</span>
          <strong>{overlay.mode === 'runtime' ? 'Runtime health' : 'Dependency risk'}</strong>
        </div>
      ) : null}

      <div className="cockpit2-canvas">
        {graph.nodes.length > 0 ? (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            onInit={setInstance}
            onNodeClick={(_, node) => onSelectNodeId(node.id)}
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
