import { useCallback, useEffect, useMemo, useState } from 'react'
import 'reactflow/dist/style.css'

import { getFaro } from '../faro'
import './cockpit.css'
import CockpitCommandBar from './components/CockpitCommandBar'
import CockpitHeader from './components/CockpitHeader'
import NodeInspector from './components/NodeInspector'
import CockpitTabs from './components/CockpitTabs'
import { cockpitFlags } from './flags'
import { filterGraph, graphKinds, resolveNodeId } from './graphUtils'
import { useCockpitData } from './hooks/useCockpitData'
import { useCockpitState } from './hooks/useCockpitState'
import C4DiffView from './views/C4DiffView'
import CityView from './views/CityView'
import DeepDiveView from './views/DeepDiveView'
import ExportsView from './views/ExportsView'
import GraphRagView from './views/GraphRagView'
import OverviewView from './views/OverviewView'
import TopologyView from './views/TopologyView'
import {
  type C4ViewMode,
  type CockpitToast,
  type CollectionLite,
  type CockpitView,
  type GraphRagCommunityMode,
  type LayoutEngine,
  type OverlayState,
  EXPORT_FORMATS,
} from './types'

interface CockpitPageProps {
  collections: CollectionLite[]
  onOpenCollections?: () => void
  onOpenRuns?: () => void
}

function downloadTextFile(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function parseCsv(content: string): Array<Record<string, string>> {
  const lines = content.split('\n').map((line) => line.trim()).filter(Boolean)
  if (lines.length < 2) return []
  const headers = lines[0].split(',').map((entry) => entry.trim())
  return lines.slice(1).map((line) => {
    const cols = line.split(',').map((entry) => entry.trim())
    const row: Record<string, string> = {}
    headers.forEach((header, index) => {
      row[header] = cols[index] || ''
    })
    return row
  })
}

async function parseOverlayFile(file: File): Promise<Pick<OverlayState, 'runtimeByNodeKey' | 'riskByNodeKey'>> {
  const content = await file.text()
  const ext = file.name.toLowerCase()
  const runtimeByNodeKey: OverlayState['runtimeByNodeKey'] = {}
  const riskByNodeKey: OverlayState['riskByNodeKey'] = {}

  let rows: Array<Record<string, unknown>> = []
  if (ext.endsWith('.json')) {
    const parsed = JSON.parse(content)
    if (Array.isArray(parsed)) {
      rows = parsed
    } else if (Array.isArray(parsed.rows)) {
      rows = parsed.rows
    }
  } else {
    rows = parseCsv(content)
  }

  for (const row of rows) {
    const service = String((row.service as string) || '')
    const node = String((row.node as string) || '')
    if (service) {
      runtimeByNodeKey[service] = {
        service,
        latency_p95: Number(row.latency_p95 || 0),
        error_rate: Number(row.error_rate || 0),
      }
    }
    if (node) {
      riskByNodeKey[node] = {
        node,
        vuln_count: Number(row.vuln_count || 0),
        severity_score: Number(row.severity_score || 0),
      }
    }
  }

  return { runtimeByNodeKey, riskByNodeKey }
}

export default function CockpitPage({
  collections,
  onOpenCollections,
  onOpenRuns,
}: CockpitPageProps) {
  const {
    selection,
    hotspotFilter,
    graphQuery,
    selectedNodeId,
    graphPage,
    graphLimit,
    includeKinds,
    excludeKinds,
    edgeKinds,
    hideIsolated,
    overlayMode,
    setHotspotFilter,
    setGraphQuery,
    setSelectedNodeId,
    setGraphPage,
    setGraphLimit,
    setIncludeKinds,
    setExcludeKinds,
    setEdgeKinds,
    setHideIsolated,
    setOverlayMode,
    setCollectionId,
    setScenarioId,
    setLayer,
    setView,
  } = useCockpitState()

  const [topologyDensity, setTopologyDensity] = useState(1200)
  const [topologyLayoutEngine, setTopologyLayoutEngine] = useState<LayoutEngine>(
    cockpitFlags.elkLayout ? 'elk_layered' : 'grid',
  )
  const [deepDiveDensity, setDeepDiveDensity] = useState(5000)
  const [deepDiveMode, setDeepDiveMode] = useState<'file_dependency' | 'symbol_callgraph' | 'contains_hierarchy'>('file_dependency')
  const [c4View, setC4View] = useState<C4ViewMode>('container')
  const [c4Scope, setC4Scope] = useState('')
  const [c4MaxNodes, setC4MaxNodes] = useState(120)
  const [graphRagCommunityMode, setGraphRagCommunityMode] = useState<GraphRagCommunityMode>('color')
  const [graphRagCommunityId, setGraphRagCommunityId] = useState('')
  const [toast, setToast] = useState<CockpitToast | null>(null)
  const [overlayData, setOverlayData] = useState<OverlayState>({
    mode: overlayMode,
    runtimeByNodeKey: {},
    riskByNodeKey: {},
    loadedAt: null,
  })

  const pushToast = useCallback((kind: CockpitToast['kind'], message: string) => {
    setToast({ id: Date.now(), kind, message })
  }, [])

  useEffect(() => {
    setOverlayData((prev) => ({ ...prev, mode: overlayMode }))
  }, [overlayMode])

  const graphFilters = useMemo(
    () => ({
      query: graphQuery,
      hideIsolated,
      edgeKinds,
      includeKinds,
      excludeKinds,
    }),
    [graphQuery, hideIsolated, edgeKinds, includeKinds, excludeKinds],
  )

  const graphPaging = useMemo(
    () => ({
      page: graphPage,
      limit: graphLimit,
    }),
    [graphPage, graphLimit],
  )

  const onViewError = useCallback(
    (view: CockpitView, message: string) => {
      getFaro()?.api.pushEvent('cockpit_error_shown', {
        view,
        message,
      })
    },
    [],
  )

  const {
    scenarios,
    scenariosState,
    city,
    graph,
    mermaid,
    activeState,
    activeError,
    activeUpdatedAt,
    errors,
    cityProjection,
    setCityProjection,
    cityEntityLevel,
    setCityEntityLevel,
    cityEmbedUrl,
    exportFormat,
    setExportFormat,
    exportProjection,
    setExportProjection,
    exportContent,
    neighborhood,
    neighborhoodState,
    neighborhoodError,
    graphRagStatus,
    graphRagReason,
    graphRagEvidenceItems,
    graphRagEvidenceTotal,
    graphRagEvidenceNodeName,
    graphRagEvidenceState,
    graphRagEvidenceError,
    graphRagCommunities,
    graphRagCommunitiesState,
    graphRagCommunitiesError,
    graphRagPath,
    graphRagPathState,
    graphRagPathError,
    graphRagProcesses,
    graphRagProcessesState,
    graphRagProcessesError,
    graphRagProcessDetail,
    graphRagProcessDetailState,
    graphRagProcessDetailError,
    traceGraphRagPath,
    loadGraphRagProcessDetail,
    generateExport,
    refreshActiveView,
  } = useCockpitData({
    selection,
    topologyLimit: topologyDensity,
    deepDiveLimit: deepDiveDensity,
    deepDiveMode,
    c4View,
    c4Scope,
    c4MaxNodes,
    graphFilters,
    graphPaging,
    graphRagCommunityMode,
    graphRagCommunityId,
    selectedNodeId,
    onScenarioAutoSelect: setScenarioId,
    onViewError,
  })

  const selectedScenario = useMemo(
    () => scenarios.find((scenario) => scenario.id === selection.scenarioId) ?? null,
    [scenarios, selection.scenarioId],
  )
  const filteredGraph = useMemo(() => filterGraph(graph, graphFilters), [graph, graphFilters])
  const { nodeKinds, edgeKinds: availableEdgeKinds } = useMemo(() => graphKinds(graph), [graph])
  const resolvedNodeId = useMemo(() => resolveNodeId(graph, selectedNodeId), [graph, selectedNodeId])

  useEffect(() => {
    if (selectedNodeId && resolvedNodeId && selectedNodeId !== resolvedNodeId) {
      setSelectedNodeId(resolvedNodeId)
    }
  }, [selectedNodeId, resolvedNodeId, setSelectedNodeId])

  useEffect(() => {
    if (collections.length > 0 && !selection.collectionId) {
      setCollectionId(collections[0].id)
    }
  }, [collections, selection.collectionId, setCollectionId])

  useEffect(() => {
    getFaro()?.api.pushEvent('cockpit_opened', {
      collection_count: String(collections.length),
    })
  }, [collections.length])

  useEffect(() => {
    getFaro()?.api.pushEvent('cockpit_tab_changed', {
      view: selection.view,
      layer: selection.layer,
    })
  }, [selection.view, selection.layer])

  useEffect(() => {
    const isGraphView = selection.view === 'topology' || selection.view === 'deep_dive'
    if (!isGraphView) {
      return
    }
    if (activeState !== 'ready') {
      return
    }
    if (selection.layer === 'code_controlflow') {
      return
    }
    if (graph.total_nodes === 0 || graph.nodes.length > 0) {
      return
    }

    setLayer('code_controlflow')
    pushToast('info', 'No nodes in this layer. Switched to Code / Controlflow.')
  }, [activeState, graph.nodes.length, graph.total_nodes, pushToast, selection.layer, selection.view, setLayer])

  useEffect(() => {
    if (!toast) {
      return
    }

    const timeoutId = window.setTimeout(() => setToast(null), 2200)
    return () => window.clearTimeout(timeoutId)
  }, [toast])

  const handleOpenTopologyFromOverview = () => {
    setLayer('code_controlflow')
    setView('topology')
  }

  const handleSelectHotspot = (nodeNaturalKey: string) => {
    setLayer('code_controlflow')
    setView('topology')
    setSelectedNodeId(nodeNaturalKey)
    setGraphQuery(nodeNaturalKey)
  }

  const handleGenerateExport = async () => {
    const result = await generateExport()
    if (!result) {
      pushToast('error', errors.exports || 'Export generation failed.')
      return
    }

    pushToast('success', `Generated ${result.name}`)
    getFaro()?.api.pushEvent('cockpit_export_generated', {
      format: exportFormat,
    })
  }

  const handleCopyExport = async () => {
    if (!exportContent) return
    try {
      await navigator.clipboard.writeText(exportContent)
      pushToast('success', 'Copied export content to clipboard.')
    } catch {
      pushToast('error', 'Could not copy export content.')
    }
  }

  const handleDownloadExport = () => {
    if (!exportContent) {
      return
    }

    const format = EXPORT_FORMATS.find((entry) => entry.key === exportFormat)
    const extension = format?.extension || 'txt'
    downloadTextFile(`cockpit-export.${extension}`, exportContent)
    pushToast('info', 'Downloaded export artifact.')
  }

  const openCollections = () => {
    if (onOpenCollections) {
      onOpenCollections()
      return
    }
    window.location.href = '/?page=collections'
  }

  const openRuns = () => {
    if (onOpenRuns) {
      onOpenRuns()
      return
    }
    window.location.href = '/?page=runs'
  }

  const trackFilterChange = () => {
    getFaro()?.api.pushEvent('cockpit_filter_changed', {
      query: graphQuery,
      include: includeKinds.join(','),
      exclude: excludeKinds.join(','),
      edge_kinds: edgeKinds.join(','),
    })
  }

  const handleLoadOverlayFile = async (file: File) => {
    try {
      const parsed = await parseOverlayFile(file)
      setOverlayData((prev) => ({
        ...prev,
        ...parsed,
        mode: overlayMode,
        loadedAt: new Date().toISOString(),
      }))
      pushToast('success', `Loaded overlay file: ${file.name}`)
      getFaro()?.api.pushEvent('cockpit_overlay_enabled', {
        mode: overlayMode,
        file: file.name,
      })
    } catch (error) {
      pushToast('error', error instanceof Error ? error.message : 'Could not parse overlay file.')
    }
  }

  const handleSelectNodeId = (nodeId: string) => {
    setSelectedNodeId(nodeId)
    getFaro()?.api.pushEvent('cockpit_node_selected', {
      node_id: nodeId,
      view: selection.view,
    })
  }

  if (collections.length === 0) {
    return (
      <section className="card cockpit2-shell">
        <header className="cockpit2-header">
          <div>
            <h2>Architecture Cockpit</h2>
            <p>No projects are available yet.</p>
          </div>
        </header>
        <section className="cockpit2-empty onboarding">
          <h3>Start by creating a collection</h3>
          <p>
            Cockpit views are generated from project sync data. Create a collection and add sources to
            unlock AS-IS and TO-BE architecture views.
          </p>
          <div className="actions">
            <button type="button" onClick={openCollections}>Go to Collections</button>
            <button type="button" className="secondary" onClick={openRuns}>Go to Runs</button>
          </div>
        </section>
      </section>
    )
  }

  if (selection.collectionId && scenariosState === 'empty') {
    return (
      <section className="card cockpit2-shell">
        <CockpitHeader
          selection={selection}
          scenario={selectedScenario}
          activeState={activeState}
          activeUpdatedAt={activeUpdatedAt}
        />
        <CockpitCommandBar
          collections={collections}
          scenarios={[]}
          collectionId={selection.collectionId}
          scenarioId=""
          layer={selection.layer}
          activeView={selection.view}
          hotspotFilter={hotspotFilter}
          graphQuery={graphQuery}
          hideIsolated={hideIsolated}
          graphPage={graphPage}
          graphLimit={graphLimit}
          includeKinds={includeKinds}
          excludeKinds={excludeKinds}
          edgeKinds={edgeKinds}
          overlayMode={overlayMode}
          graphRagCommunityMode={graphRagCommunityMode}
          graphRagCommunityId={graphRagCommunityId}
          graphRagCommunities={graphRagCommunities}
          c4View={c4View}
          c4Scope={c4Scope}
          c4MaxNodes={c4MaxNodes}
          availableNodeKinds={[]}
          availableEdgeKinds={[]}
          onCollectionChange={setCollectionId}
          onScenarioChange={setScenarioId}
          onLayerChange={setLayer}
          onFilterChange={setHotspotFilter}
          onGraphQueryChange={(value) => {
            setGraphQuery(value)
            trackFilterChange()
          }}
          onHideIsolatedChange={(next) => {
            setHideIsolated(next)
            trackFilterChange()
          }}
          onGraphPageChange={setGraphPage}
          onGraphLimitChange={setGraphLimit}
          onIncludeKindsChange={(value) => {
            setIncludeKinds(value)
            trackFilterChange()
          }}
          onExcludeKindsChange={(value) => {
            setExcludeKinds(value)
            trackFilterChange()
          }}
          onEdgeKindsChange={(value) => {
            setEdgeKinds(value)
            trackFilterChange()
          }}
          onOverlayModeChange={setOverlayMode}
          onGraphRagCommunityModeChange={setGraphRagCommunityMode}
          onGraphRagCommunityIdChange={setGraphRagCommunityId}
          onC4ViewChange={setC4View}
          onC4ScopeChange={setC4Scope}
          onC4MaxNodesChange={(value) => setC4MaxNodes(Math.max(10, Math.min(5000, value)))}
          onLoadOverlayFile={handleLoadOverlayFile}
          onRefresh={refreshActiveView}
          onOpenCollections={openCollections}
          onOpenRuns={openRuns}
        />
        <section className="cockpit2-empty onboarding">
          <h3>No scenarios found for this project</h3>
          <p>
            Run a sync to generate the AS-IS digital twin. Once the sync is complete, the Cockpit will
            populate Overview, Topology, and C4 views automatically.
          </p>
          <div className="actions">
            <button type="button" onClick={openRuns}>Run sync from Runs</button>
            <button type="button" className="secondary" onClick={openCollections}>Manage sources</button>
          </div>
        </section>
      </section>
    )
  }

  return (
    <section className="card cockpit2-shell">
      <CockpitHeader
        selection={selection}
        scenario={selectedScenario}
        activeState={activeState}
        activeUpdatedAt={activeUpdatedAt}
      />

      <CockpitCommandBar
        collections={collections}
        scenarios={scenarios}
        collectionId={selection.collectionId}
        scenarioId={selection.scenarioId}
        layer={selection.layer}
        activeView={selection.view}
        hotspotFilter={hotspotFilter}
        graphQuery={graphQuery}
        hideIsolated={hideIsolated}
        graphPage={graphPage}
        graphLimit={graphLimit}
        includeKinds={includeKinds}
        excludeKinds={excludeKinds}
        edgeKinds={edgeKinds}
        overlayMode={overlayMode}
        graphRagCommunityMode={graphRagCommunityMode}
        graphRagCommunityId={graphRagCommunityId}
        graphRagCommunities={graphRagCommunities}
        c4View={c4View}
        c4Scope={c4Scope}
        c4MaxNodes={c4MaxNodes}
        availableNodeKinds={nodeKinds}
        availableEdgeKinds={availableEdgeKinds}
        onCollectionChange={setCollectionId}
        onScenarioChange={setScenarioId}
        onLayerChange={setLayer}
        onFilterChange={setHotspotFilter}
        onGraphQueryChange={(value) => {
          setGraphQuery(value)
          trackFilterChange()
        }}
        onHideIsolatedChange={(next) => {
          setHideIsolated(next)
          trackFilterChange()
        }}
        onGraphPageChange={(value) => setGraphPage(Math.max(0, value))}
        onGraphLimitChange={(value) => {
          const next = Math.min(10000, Math.max(1, value))
          setGraphLimit(next)
          if (next > 5000 && selection.view === 'deep_dive') {
            pushToast('info', 'Deep Dive limit > 5000 may reduce interactivity.')
          }
        }}
        onIncludeKindsChange={(value) => {
          setIncludeKinds(value)
          trackFilterChange()
        }}
        onExcludeKindsChange={(value) => {
          setExcludeKinds(value)
          trackFilterChange()
        }}
        onEdgeKindsChange={(value) => {
          setEdgeKinds(value)
          trackFilterChange()
        }}
        onOverlayModeChange={setOverlayMode}
        onGraphRagCommunityModeChange={setGraphRagCommunityMode}
        onGraphRagCommunityIdChange={setGraphRagCommunityId}
        onC4ViewChange={setC4View}
        onC4ScopeChange={setC4Scope}
        onC4MaxNodesChange={(value) => setC4MaxNodes(Math.max(10, Math.min(5000, value)))}
        onLoadOverlayFile={handleLoadOverlayFile}
        onRefresh={refreshActiveView}
        onOpenCollections={openCollections}
        onOpenRuns={openRuns}
      />

      <CockpitTabs activeView={selection.view} onViewChange={setView} />

      {selection.view === 'overview' ? (
        <OverviewView
          city={city}
          state={activeState}
          error={activeError}
          filter={hotspotFilter}
          onRetry={refreshActiveView}
          onOpenTopology={handleOpenTopologyFromOverview}
          onSelectHotspot={handleSelectHotspot}
        />
      ) : null}

      {selection.view === 'topology' ? (
        <section className="cockpit2-workspace">
          <div className="cockpit2-main">
            <TopologyView
              graph={filteredGraph}
              state={activeState}
              error={activeError}
              layer={selection.layer}
              density={topologyDensity}
              layoutEngine={topologyLayoutEngine}
              elkEnabled={cockpitFlags.elkLayout}
              overlay={overlayData}
              selectedNodeId={resolvedNodeId}
              onDensityChange={setTopologyDensity}
              onLayoutEngineChange={setTopologyLayoutEngine}
              onSwitchToCodeLayer={() => setLayer('code_controlflow')}
              onSelectNodeId={handleSelectNodeId}
              onLayoutCompleted={(engine, durationMs, nodeCount) => {
                getFaro()?.api.pushEvent('cockpit_layout_completed', {
                  engine,
                  duration_ms: String(Math.round(durationMs)),
                  node_count: String(nodeCount),
                })
              }}
              onRetry={refreshActiveView}
            />
          </div>
          {cockpitFlags.inspector ? (
            <div className="cockpit2-rail">
              <NodeInspector
                selectedNodeId={resolvedNodeId}
                graph={graph}
                neighborhood={neighborhood}
                neighborhoodState={neighborhoodState}
                neighborhoodError={neighborhoodError}
                overlay={overlayData}
                onClearSelection={() => setSelectedNodeId('')}
              />
            </div>
          ) : null}
        </section>
      ) : null}

      {selection.view === 'deep_dive' ? (
        <section className="cockpit2-workspace">
          <div className="cockpit2-main">
            <DeepDiveView
              graph={filteredGraph}
              state={activeState}
              error={activeError}
              layer={selection.layer}
              density={deepDiveDensity}
              mode={deepDiveMode}
              overlay={overlayData}
              selectedNodeId={resolvedNodeId}
              onModeChange={setDeepDiveMode}
              onDensityChange={setDeepDiveDensity}
              onSelectNodeId={handleSelectNodeId}
              onSwitchToCodeLayer={() => setLayer('code_controlflow')}
              onRetry={refreshActiveView}
            />
          </div>
          {cockpitFlags.inspector ? (
            <div className="cockpit2-rail">
              <NodeInspector
                selectedNodeId={resolvedNodeId}
                graph={graph}
                neighborhood={neighborhood}
                neighborhoodState={neighborhoodState}
                neighborhoodError={neighborhoodError}
                overlay={overlayData}
                onClearSelection={() => setSelectedNodeId('')}
              />
            </div>
          ) : null}
        </section>
      ) : null}

      {selection.view === 'c4_diff' ? (
        <C4DiffView
          mermaid={mermaid}
          state={activeState}
          error={activeError}
          onRetry={refreshActiveView}
        />
      ) : null}

      {selection.view === 'city' ? (
        <CityView
          state={activeState}
          error={activeError}
          embedUrl={cityEmbedUrl}
          projection={cityProjection}
          entityLevel={cityEntityLevel}
          onProjectionChange={setCityProjection}
          onEntityLevelChange={setCityEntityLevel}
          onReload={refreshActiveView}
        />
      ) : null}

      {selection.view === 'graphrag' ? (
        <GraphRagView
          graph={filteredGraph}
          state={activeState}
          error={activeError}
          status={graphRagStatus}
          reason={graphRagReason}
          selectedNodeId={resolvedNodeId}
          communityMode={graphRagCommunityMode}
          communityId={graphRagCommunityId}
          communities={graphRagCommunities}
          communitiesState={graphRagCommunitiesState}
          communitiesError={graphRagCommunitiesError}
          path={graphRagPath}
          pathState={graphRagPathState}
          pathError={graphRagPathError}
          processes={graphRagProcesses}
          processesState={graphRagProcessesState}
          processesError={graphRagProcessesError}
          processDetail={graphRagProcessDetail}
          processDetailState={graphRagProcessDetailState}
          processDetailError={graphRagProcessDetailError}
          evidenceItems={graphRagEvidenceItems}
          evidenceTotal={graphRagEvidenceTotal}
          evidenceNodeName={graphRagEvidenceNodeName}
          evidenceState={graphRagEvidenceState}
          evidenceError={graphRagEvidenceError}
          onSelectNodeId={handleSelectNodeId}
          onTracePath={traceGraphRagPath}
          onLoadProcessDetail={loadGraphRagProcessDetail}
          onRetry={refreshActiveView}
        />
      ) : null}

      {selection.view === 'exports' ? (
        <ExportsView
          exportFormat={exportFormat}
          exportProjection={exportProjection}
          exportState={activeState}
          exportError={errors.exports}
          exportContent={exportContent}
          onFormatChange={setExportFormat}
          onProjectionChange={setExportProjection}
          onGenerate={handleGenerateExport}
          onCopy={handleCopyExport}
          onDownload={handleDownloadExport}
        />
      ) : null}

      {toast ? (
        <div className={`cockpit2-toast ${toast.kind}`} role="status" aria-live="polite">
          {toast.message}
        </div>
      ) : null}
    </section>
  )
}
