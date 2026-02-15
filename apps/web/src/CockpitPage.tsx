import { useEffect, useMemo, useRef, useState } from 'react'
import ReactFlow, { Background, Controls, MiniMap, type Edge as RFEdge, type Node as RFNode } from 'reactflow'
import cytoscape, { type Core } from 'cytoscape'
import 'reactflow/dist/style.css'

type Layer = 'portfolio_system' | 'domain_container' | 'component_interface' | 'code_controlflow'

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
}

type ValidationSource = {
  source: string
  metrics: Record<string, { value: number; status: string; captured_at: string }>
}

type ValidationPayload = {
  sources: ValidationSource[]
}

interface CockpitPageProps {
  collections: CollectionLite[]
}

export default function CockpitPage({ collections }: CockpitPageProps) {
  const [collectionId, setCollectionId] = useState<string>('')
  const [scenarios, setScenarios] = useState<ScenarioLite[]>([])
  const [scenarioId, setScenarioId] = useState<string>('')
  const [layer, setLayer] = useState<Layer | ''>('')
  const [graph, setGraph] = useState<TwinGraphResponse>({ nodes: [], edges: [] })
  const [validation, setValidation] = useState<ValidationPayload | null>(null)
  const [cypherQuery, setCypherQuery] = useState('MATCH (n:Node) RETURN n LIMIT 25')
  const [cypherResult, setCypherResult] = useState<string>('')
  const [ccJson, setCcJson] = useState<string>('')
  const [mermaid, setMermaid] = useState<string>('')
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
      if (!response.ok) return
      const data = await response.json()
      setScenarios(data.scenarios || [])
      if (data.scenarios?.length > 0) {
        setScenarioId(data.scenarios[0].id)
      }
    }
    run()
  }, [collectionId])

  useEffect(() => {
    const run = async () => {
      if (!scenarioId) {
        setGraph({ nodes: [], edges: [] })
        return
      }
      const layerQuery = layer ? `&layer=${layer}` : ''
      const response = await fetch(`/api/twin/scenarios/${scenarioId}/graph?limit=5000${layerQuery}`, { credentials: 'include' })
      if (!response.ok) return
      const data = await response.json()
      setGraph(data)
    }
    run()
  }, [scenarioId, layer])

  useEffect(() => {
    const run = async () => {
      const collectionQuery = collectionId ? `?collection_id=${collectionId}` : ''
      const response = await fetch(`/api/validation/status${collectionQuery}`, { credentials: 'include' })
      if (!response.ok) return
      const data = await response.json()
      setValidation(data)
    }
    run()
  }, [collectionId])

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
    if (!cyDivRef.current) return

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

  const runCypher = async () => {
    if (!scenarioId) return
    const response = await fetch(`/api/twin/scenarios/${scenarioId}/cypher`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ query: cypherQuery }),
    })
    const data = await response.json()
    setCypherResult(JSON.stringify(data, null, 2))
  }

  const generateExport = async (format: 'cc_json' | 'mermaid_c4') => {
    if (!scenarioId) return
    const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ format }),
    })
    if (!exportResponse.ok) return
    const exportData = await exportResponse.json()
    const exportId = exportData.id || exportData.exports?.[0]?.id
    if (!exportId) return

    const artifactResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports/${exportId}`, { credentials: 'include' })
    if (!artifactResponse.ok) return
    const artifact = await artifactResponse.json()
    if (format === 'cc_json') {
      setCcJson(artifact.content || '')
    }
    if (format === 'mermaid_c4') {
      setMermaid(artifact.content || '')
    }
  }

  return (
    <section className="card cockpit-card">
      <h2>Architecture Cockpit (Readonly)</h2>
      <div className="cockpit-toolbar">
        <select value={collectionId} onChange={(e) => setCollectionId(e.target.value)}>
          <option value="">Select collection...</option>
          {collections.map((collection) => (
            <option key={collection.id} value={collection.id}>{collection.name}</option>
          ))}
        </select>
        <select value={scenarioId} onChange={(e) => setScenarioId(e.target.value)}>
          <option value="">Select scenario...</option>
          {scenarios.map((scenario) => (
            <option key={scenario.id} value={scenario.id}>{scenario.name} (v{scenario.version})</option>
          ))}
        </select>
        <select value={layer} onChange={(e) => setLayer((e.target.value as Layer) || '')}>
          <option value="">All layers</option>
          <option value="portfolio_system">Portfolio/System</option>
          <option value="domain_container">Domain/Container</option>
          <option value="component_interface">Component/Interface</option>
          <option value="code_controlflow">Code/Controlflow</option>
        </select>
      </div>

      <div className="cockpit-grid">
        <div className="cockpit-panel">
          <h3>Topology (React Flow)</h3>
          <div style={{ height: 320, border: '1px solid #d0d5dd', borderRadius: 8 }}>
            <ReactFlow nodes={rfNodes} edges={rfEdges} fitView>
              <Controls />
              <MiniMap />
              <Background />
            </ReactFlow>
          </div>
        </div>

        <div className="cockpit-panel">
          <h3>Deep Dive (Cytoscape)</h3>
          <div ref={cyDivRef} style={{ height: 320, border: '1px solid #d0d5dd', borderRadius: 8 }} />
        </div>

        <div className="cockpit-panel">
          <h3>Cypher (AGE, Read-only)</h3>
          <textarea value={cypherQuery} onChange={(e) => setCypherQuery(e.target.value)} rows={4} />
          <button onClick={runCypher}>Run Query</button>
          <pre>{cypherResult}</pre>
        </div>

        <div className="cockpit-panel">
          <h3>Validation Dashboard</h3>
          <pre>{JSON.stringify(validation, null, 2)}</pre>
        </div>

        <div className="cockpit-panel">
          <h3>CodeCharta (cc.json)</h3>
          <button onClick={() => generateExport('cc_json')}>Generate cc.json</button>
          <pre>{ccJson}</pre>
        </div>

        <div className="cockpit-panel">
          <h3>Mermaid C4</h3>
          <button onClick={() => generateExport('mermaid_c4')}>Generate Mermaid</button>
          <pre>{mermaid}</pre>
        </div>
      </div>
    </section>
  )
}
