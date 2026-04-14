import { useEffect, useRef, useState } from 'react'
import cytoscape, { type Core } from 'cytoscape'

import ViewShell from '../components/ViewShell'
import { layerLabel } from '../types'
import type { CockpitLayer, CockpitLoadState, DeepDiveMode, OverlayState, TwinGraphResponse } from '../types'

interface DeepDiveViewProps {
  graph: TwinGraphResponse
  state: CockpitLoadState
  error: string
  layer: CockpitLayer
  mode: DeepDiveMode
  density: number
  overlay: OverlayState
  selectedNodeId: string
  onModeChange: (mode: DeepDiveMode) => void
  onDensityChange: (density: number) => void
  onSelectNodeId: (nodeId: string) => void
  onSwitchToCodeLayer: () => void
  onRetry: () => void
}

function getLayoutName(density: number): 'cose' | 'breadthfirst' {
  return density > 5000 ? 'breadthfirst' : 'cose'
}

function humanizeSliceStrategy(strategy: string | undefined): string {
  if (strategy === 'edge_aware_seed_window') return 'edge-aware seed paging'
  if (strategy === 'sorted_page_slice') return 'sorted page slicing'
  return 'graph paging'
}

export default function DeepDiveView({
  graph,
  state,
  error,
  layer,
  mode,
  density,
  overlay,
  selectedNodeId,
  onModeChange,
  onDensityChange,
  onSelectNodeId,
  onSwitchToCodeLayer,
  onRetry,
}: Readonly<DeepDiveViewProps>) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const graphRef = useRef<Core | null>(null)
  const [showLabels, setShowLabels] = useState(true)
  const [labelsPinned, setLabelsPinned] = useState(false)

  useEffect(() => {
    if (density > 5000 && !labelsPinned) {
      setShowLabels(false)
    }
  }, [density, labelsPinned])

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
        ...graph.nodes.map((node) => {
          const runtime = overlay.runtimeByNodeKey[node.natural_key] || overlay.runtimeByNodeKey[node.name]
          const risk = overlay.riskByNodeKey[node.natural_key] || overlay.riskByNodeKey[node.name]
          return {
            data: {
              id: node.id,
              label: node.name,
              kind: node.kind,
              selected: selectedNodeId === node.id ? 1 : 0,
              runtime_error: Number(runtime?.error_rate || 0),
              risk_score: Number(risk?.severity_score || 0),
            },
          }
        }),
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
          selector: 'node[runtime_error >= 0.1]',
          style: {
            'background-color': '#dc2626',
          },
        },
        {
          selector: 'node[runtime_error >= 0.03][runtime_error < 0.1]',
          style: {
            'background-color': '#f59e0b',
          },
        },
        {
          selector: 'node[risk_score >= 8]',
          style: {
            'background-color': '#b91c1c',
          },
        },
        {
          selector: 'node[risk_score >= 4][risk_score < 8]',
          style: {
            'background-color': '#d97706',
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

    next.on('tap', 'node', (event) => {
      const id = String(event.target.id())
      onSelectNodeId(id)
    })

    graphRef.current = next

    return () => {
      next.destroy()
      graphRef.current = null
    }
  }, [graph, state, showLabels, density, overlay, selectedNodeId, onSelectNodeId])
  const warnings = graph.warnings || []
  const provenanceNotes: string[] = []
  if (graph.provenance?.source === 'knowledge_recovery') {
    provenanceNotes.push('This view is being recovered from the knowledge graph while scenario extraction catches up.')
  }
  if (mode === 'symbol_callgraph' && graph.slice_strategy) {
    provenanceNotes.push(`Callgraph pages now use ${humanizeSliceStrategy(graph.slice_strategy)} to preserve local connectivity.`)
  }

  return (
    <ViewShell
      state={state}
      error={error || null}
      panelId="cockpit-panel-deep_dive"
      title="Deep dive"
      hasData={graph.nodes.length > 0}
      onRetry={onRetry}
      skeletonCount={1}
      skeletonTall
    >
    <section className="cockpit2-panel" id="cockpit-panel-deep_dive" role="tabpanel">
      <div className="cockpit2-panel-header-row">
        <h3>Deep dive graph</h3>
        <p className="muted">
          Nodes: {graph.visible_nodes ?? graph.nodes.length} / Total: {graph.candidate_nodes ?? graph.total_nodes} • Edges: {graph.visible_edges ?? graph.edges.length}
          {graph.projection ? ` • Projection: ${graph.projection}` : ''}
          {mode ? ` • Mode: ${mode}` : ''}
          {graph.slice_strategy ? ` • Slice: ${humanizeSliceStrategy(graph.slice_strategy)}` : ''}
        </p>
      </div>

      {provenanceNotes.length > 0 ? (
        <div className="cockpit2-alert inline">
          {provenanceNotes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      ) : null}

      {warnings.length > 0 ? (
        <div className="cockpit2-alert inline">
          {warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}

      {density > 5000 ? (
        <div className="cockpit2-overlay-legend">
          <p>Dense mode can be expensive. Labels are auto-disabled unless manually re-enabled.</p>
        </div>
      ) : null}

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
        <button
          type="button"
          className="secondary"
          onClick={() => {
            setLabelsPinned(true)
            setShowLabels((prev) => !prev)
          }}
        >
          {showLabels ? 'Hide labels' : 'Show labels'}
        </button>

        <label>
          Mode{' '}
          <select value={mode} onChange={(event) => onModeChange(event.target.value as DeepDiveMode)}>
            <option value="file_dependency">File dependency</option>
            <option value="symbol_callgraph">Symbol callgraph</option>
            <option value="contains_hierarchy">Contains hierarchy</option>
          </select>
        </label>

        <label>
          Density{' '}
          <select value={density} onChange={(event) => onDensityChange(Number(event.target.value))}>
            <option value={3000}>Focused</option>
            <option value={5000}>Balanced</option>
            <option value={8000}>Dense</option>
          </select>
        </label>
      </div>

      {graph.nodes.length > 0 ? (
        <div ref={containerRef} className="cockpit2-canvas deep" aria-label="Deep dive graph" />
      ) : (
        <section className="cockpit2-empty">
          <h3>No nodes for this layer</h3>
          {graph.total_nodes > 0 ? (
            <>
              <p>
                The selected layer (<strong>{layerLabel(layer)}</strong>) has no nodes in this scenario.
              </p>
              {layer === 'code_controlflow' ? null : (
                <button type="button" onClick={onSwitchToCodeLayer}>
                  Switch to Code / Controlflow
                </button>
              )}
            </>
          ) : (
            <p>
              No twin nodes are available yet. Run source sync and ensure semantic snapshots were generated.
            </p>
          )}
        </section>
      )}
    </section>
    </ViewShell>
  )
}
