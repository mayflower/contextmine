import { useEffect, useMemo, useRef, useState } from 'react'
import ReactFlow, { Background, Controls, MiniMap, type Edge as RFEdge, type Node as RFNode } from 'reactflow'
import cytoscape, { type Core } from 'cytoscape'
import 'reactflow/dist/style.css'

type Layer = 'portfolio_system' | 'domain_container' | 'component_interface' | 'code_controlflow'
type ViewTab = 'city' | 'topology' | 'deep_dive' | 'architecture' | 'exports'
type ExportFormat = 'lpg_jsonl' | 'cc_json' | 'cx2' | 'jgf' | 'mermaid_c4'

type CollectionLite = {
  id: string
  name: string
}

type ScenarioLite = {
  id: string
  name: string
  version: number
  is_as_is: boolean
}

type TwinGraphNode = {
  id: string
  natural_key: string
  kind: string
  name: string
  meta: Record<string, unknown>
}

type TwinGraphEdge = {
  id: string
  source_node_id: string
  target_node_id: string
  kind: string
  meta: Record<string, unknown>
}

type TwinGraphResponse = {
  nodes: TwinGraphNode[]
  edges: TwinGraphEdge[]
  page: number
  limit: number
  total_nodes: number
}

type CityHotspot = {
  node_natural_key: string
  loc: number
  symbol_count: number
  coverage: number
  complexity: number
  coupling: number
}

type CityPayload = {
  collection_id: string
  scenario: {
    id: string
    collection_id: string
    name: string
    version: number
    is_as_is: boolean
    base_scenario_id: string | null
  }
  summary: {
    metric_nodes: number
    coverage_avg: number
    complexity_avg: number
    coupling_avg: number
  }
  hotspots: CityHotspot[]
  cc_json: Record<string, unknown>
}

type MermaidPayload = {
  collection_id: string
  scenario: {
    id: string
    collection_id: string
    name: string
    version: number
    is_as_is: boolean
    base_scenario_id: string | null
  }
  mode: 'single' | 'compare'
  content?: string
  as_is?: string
  to_be?: string
  as_is_scenario_id?: string
}

interface CockpitPageProps {
  collections: CollectionLite[]
}

export default function CockpitPage({ collections }: CockpitPageProps) {
  const [collectionId, setCollectionId] = useState<string>('')
  const [scenarios, setScenarios] = useState<ScenarioLite[]>([])
  const [scenarioId, setScenarioId] = useState<string>('')
  const [layer, setLayer] = useState<Layer>('domain_container')
  const [activeTab, setActiveTab] = useState<ViewTab>('city')
  const [graph, setGraph] = useState<TwinGraphResponse>({ nodes: [], edges: [], page: 0, limit: 0, total_nodes: 0 })
  const [city, setCity] = useState<CityPayload | null>(null)
  const [mermaid, setMermaid] = useState<MermaidPayload | null>(null)
  const [exportContent, setExportContent] = useState<string>('')
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cc_json')
  const [error, setError] = useState<string>('')
  const cyDivRef = useRef<HTMLDivElement | null>(null)
  const cyRef = useRef<Core | null>(null)

  useEffect(() => {
    const run = async () => {
      if (!collectionId) {
        setScenarios([])
        setScenarioId('')
        return
      }
      const response = await fetch(`/api/twin/scenarios?collection_id=${collectionId}`, { credentials: 'include' })
      if (!response.ok) {
        setError('Konnte Szenarien nicht laden.')
        return
      }
      const data = await response.json()
      setScenarios(data.scenarios || [])
      if (data.scenarios?.length > 0) {
        const asIs = data.scenarios.find((scenario: ScenarioLite) => scenario.is_as_is)
        setScenarioId(asIs?.id || data.scenarios[0].id)
      } else {
        setScenarioId('')
      }
    }
    run()
  }, [collectionId])

  useEffect(() => {
    const run = async () => {
      if (!scenarioId || !collectionId || (activeTab !== 'topology' && activeTab !== 'deep_dive')) {
        return
      }
      setError('')
      const endpoint = activeTab === 'topology' ? 'topology' : 'deep-dive'
      const limit = activeTab === 'topology' ? 1200 : 3000
      const response = await fetch(
        `/api/twin/collections/${collectionId}/views/${endpoint}?scenario_id=${scenarioId}&layer=${layer}&limit=${limit}`,
        { credentials: 'include' },
      )
      if (!response.ok) {
        setError('Konnte Graph-Sicht nicht laden.')
        return
      }
      const data = await response.json()
      setGraph(data.graph || { nodes: [], edges: [], page: 0, limit: 0, total_nodes: 0 })
    }
    run()
  }, [scenarioId, collectionId, layer, activeTab])

  useEffect(() => {
    const run = async () => {
      if (!scenarioId || !collectionId || activeTab !== 'city') {
        return
      }
      setError('')
      const response = await fetch(
        `/api/twin/collections/${collectionId}/views/city?scenario_id=${scenarioId}&hotspots_limit=40`,
        { credentials: 'include' },
      )
      if (!response.ok) {
        setError('Konnte City-Sicht nicht laden.')
        return
      }
      const data = await response.json()
      setCity(data)
    }
    run()
  }, [scenarioId, collectionId, activeTab])

  useEffect(() => {
    const run = async () => {
      if (!scenarioId || !collectionId || activeTab !== 'architecture') {
        return
      }
      setError('')
      const response = await fetch(
        `/api/twin/collections/${collectionId}/views/mermaid?scenario_id=${scenarioId}&compare_with_base=true`,
        { credentials: 'include' },
      )
      if (!response.ok) {
        setError('Konnte Mermaid-Sicht nicht laden.')
        return
      }
      const data = await response.json()
      setMermaid(data)
    }
    run()
  }, [scenarioId, collectionId, activeTab])

  const rfNodes = useMemo<RFNode[]>(() => {
    return graph.nodes.map((node, index) => ({
      id: node.id,
      data: { label: `${node.name} (${node.kind})` },
      position: { x: (index % 8) * 280, y: Math.floor(index / 8) * 120 },
      style: { width: 240, fontSize: 12 },
    }))
  }, [graph.nodes])

  const rfEdges = useMemo<RFEdge[]>(() => {
    return graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source_node_id,
      target: edge.target_node_id,
      label: edge.kind,
      type: 'smoothstep',
    }))
  }, [graph.edges])

  useEffect(() => {
    if (activeTab !== 'deep_dive' || !cyDivRef.current) {
      return
    }

    if (cyRef.current) {
      cyRef.current.destroy()
      cyRef.current = null
    }

    cyRef.current = cytoscape({
      container: cyDivRef.current,
      elements: [
        ...graph.nodes.map((node) => ({ data: { id: node.id, label: node.name, kind: node.kind } })),
        ...graph.edges.map((edge) => ({ data: { id: edge.id, source: edge.source_node_id, target: edge.target_node_id, label: edge.kind } })),
      ],
      style: [
        { selector: 'node', style: { label: 'data(label)', 'font-size': 9, 'background-color': '#155eef', color: '#0f172a', 'text-valign': 'center', 'text-halign': 'center', width: 18, height: 18 } },
        { selector: 'edge', style: { width: 1, 'line-color': '#94a3b8', 'target-arrow-color': '#94a3b8', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier' } },
      ],
      layout: { name: 'cose', animate: false },
    })

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy()
        cyRef.current = null
      }
    }
  }, [graph])

  const generateExport = async () => {
    if (!scenarioId) {
      return
    }
    setError('')
    const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ format: exportFormat }),
    })
    if (!exportResponse.ok) {
      setError('Export konnte nicht erzeugt werden.')
      return
    }
    const exportData = await exportResponse.json()
    const exportId = exportData.id || exportData.exports?.[0]?.id || ''
    if (!exportId) {
      setError('Export-ID fehlt in der API-Antwort.')
      return
    }

    const artifactResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports/${exportId}`, { credentials: 'include' })
    if (!artifactResponse.ok) {
      setError('Export-Artefakt konnte nicht gelesen werden.')
      return
    }
    const artifact = await artifactResponse.json()
    setExportContent(artifact.content || '')
  }

  return (
    <section className="card cockpit-card">
      <h2>Extracted Views Cockpit (Readonly)</h2>
      <div className="cockpit-toolbar">
        <select value={collectionId} onChange={(e) => setCollectionId(e.target.value)}>
          <option value="">Projekt (Collection) wählen...</option>
          {collections.map((collection) => (
            <option key={collection.id} value={collection.id}>{collection.name}</option>
          ))}
        </select>
        <select value={scenarioId} onChange={(e) => setScenarioId(e.target.value)}>
          <option value="">Szenario wählen...</option>
          {scenarios.map((scenario) => (
            <option key={scenario.id} value={scenario.id}>
              {scenario.name} (v{scenario.version}) {scenario.is_as_is ? '· AS-IS' : '· TO-BE'}
            </option>
          ))}
        </select>
        <select value={layer} onChange={(e) => setLayer(e.target.value as Layer)}>
          <option value="portfolio_system">Portfolio/System</option>
          <option value="domain_container">Domain/Container</option>
          <option value="component_interface">Component/Interface</option>
          <option value="code_controlflow">Code/Controlflow</option>
        </select>
      </div>

      <div className="cockpit-tabs">
        <button className={activeTab === 'city' ? 'active' : ''} onClick={() => setActiveTab('city')}>City</button>
        <button className={activeTab === 'topology' ? 'active' : ''} onClick={() => setActiveTab('topology')}>Topology</button>
        <button className={activeTab === 'deep_dive' ? 'active' : ''} onClick={() => setActiveTab('deep_dive')}>Deep Dive</button>
        <button className={activeTab === 'architecture' ? 'active' : ''} onClick={() => setActiveTab('architecture')}>Mermaid C4</button>
        <button className={activeTab === 'exports' ? 'active' : ''} onClick={() => setActiveTab('exports')}>Exporte</button>
      </div>

      {error ? <p className="cockpit-error">{error}</p> : null}

      {activeTab === 'city' && (
        <div className="cockpit-grid">
          <div className="cockpit-panel">
            <h3>City Summary</h3>
            <div className="cockpit-summary-grid">
              <div><strong>{city?.summary.metric_nodes ?? 0}</strong><span>Metric Nodes</span></div>
              <div><strong>{(city?.summary.coverage_avg ?? 0).toFixed(2)}</strong><span>Avg Coverage</span></div>
              <div><strong>{(city?.summary.complexity_avg ?? 0).toFixed(2)}</strong><span>Avg Complexity</span></div>
              <div><strong>{(city?.summary.coupling_avg ?? 0).toFixed(2)}</strong><span>Avg Coupling</span></div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'city' && (
        <div className="cockpit-grid">
          <div className="cockpit-panel">
            <h3>Top Hotspots</h3>
            <div className="cockpit-table-wrap">
              <table className="cockpit-table">
                <thead>
                  <tr>
                    <th>Node</th>
                    <th>Complexity</th>
                    <th>Coupling</th>
                    <th>Coverage</th>
                    <th>LOC</th>
                  </tr>
                </thead>
                <tbody>
                  {(city?.hotspots || []).slice(0, 15).map((spot) => (
                    <tr key={spot.node_natural_key}>
                      <td>{spot.node_natural_key}</td>
                      <td>{spot.complexity?.toFixed(2) || '0.00'}</td>
                      <td>{spot.coupling?.toFixed(2) || '0.00'}</td>
                      <td>{spot.coverage?.toFixed(2) || '0.00'}</td>
                      <td>{spot.loc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="cockpit-panel">
            <h3>cc.json Preview</h3>
            <pre>{JSON.stringify(city?.cc_json || {}, null, 2)}</pre>
          </div>
        </div>
      )}

      {activeTab === 'topology' && (
        <div className="cockpit-panel">
          <h3>Topology (React Flow)</h3>
          <p className="cockpit-meta">Nodes: {graph.nodes.length} / Total: {graph.total_nodes}</p>
          <div style={{ height: 560, border: '1px solid #d0d5dd', borderRadius: 8 }}>
            <ReactFlow nodes={rfNodes} edges={rfEdges} fitView>
              <Controls />
              <MiniMap />
              <Background />
            </ReactFlow>
          </div>
        </div>
      )}

      {activeTab === 'deep_dive' && (
        <div className="cockpit-panel">
          <h3>Deep Dive (Cytoscape)</h3>
          <p className="cockpit-meta">Nodes: {graph.nodes.length} / Total: {graph.total_nodes}</p>
          <div ref={cyDivRef} style={{ height: 620, border: '1px solid #d0d5dd', borderRadius: 8 }} />
        </div>
      )}

      {activeTab === 'architecture' && (
        <div className="cockpit-panel">
          <h3>Mermaid C4</h3>
          {mermaid?.mode === 'compare' ? (
            <div className="cockpit-mermaid-compare">
              <article>
                <h4>AS-IS</h4>
                <pre>{mermaid.as_is || ''}</pre>
              </article>
              <article>
                <h4>TO-BE</h4>
                <pre>{mermaid.to_be || ''}</pre>
              </article>
            </div>
          ) : (
            <pre>{mermaid?.content || ''}</pre>
          )}
        </div>
      )}

      {activeTab === 'exports' && (
        <div className="cockpit-panel">
          <h3>Visualisierungs-Exporte</h3>
          <div className="cockpit-export-controls">
            <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value as ExportFormat)}>
              <option value="cc_json">cc.json (CodeCharta)</option>
              <option value="cx2">CX2</option>
              <option value="jgf">JGF</option>
              <option value="lpg_jsonl">LPG JSONL</option>
              <option value="mermaid_c4">Mermaid C4</option>
            </select>
            <button onClick={generateExport}>Export generieren</button>
          </div>
          <pre>{exportContent}</pre>
        </div>
      )}
    </section>
  )
}
