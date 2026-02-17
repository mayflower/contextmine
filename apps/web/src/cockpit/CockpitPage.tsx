import { useCallback, useEffect, useMemo, useState } from 'react'
import 'reactflow/dist/style.css'

import { getFaro } from '../faro'
import './cockpit.css'
import CockpitCommandBar from './components/CockpitCommandBar'
import CockpitHeader from './components/CockpitHeader'
import CockpitTabs from './components/CockpitTabs'
import { useCockpitData } from './hooks/useCockpitData'
import { useCockpitState } from './hooks/useCockpitState'
import C4DiffView from './views/C4DiffView'
import DeepDiveView from './views/DeepDiveView'
import ExportsView from './views/ExportsView'
import OverviewView from './views/OverviewView'
import TopologyView from './views/TopologyView'
import {
  type CockpitToast,
  type CollectionLite,
  type CockpitView,
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

export default function CockpitPage({
  collections,
  onOpenCollections,
  onOpenRuns,
}: CockpitPageProps) {
  const {
    selection,
    hotspotFilter,
    setHotspotFilter,
    setCollectionId,
    setScenarioId,
    setLayer,
    setView,
  } = useCockpitState()

  const [topologyDensity, setTopologyDensity] = useState(1200)
  const [deepDiveDensity, setDeepDiveDensity] = useState(5000)
  const [deepDiveMode, setDeepDiveMode] = useState<'file_dependency' | 'symbol_callgraph' | 'contains_hierarchy'>('file_dependency')
  const [toast, setToast] = useState<CockpitToast | null>(null)

  const topologyLimit = topologyDensity
  const deepDiveLimit = deepDiveDensity

  const pushToast = useCallback((kind: CockpitToast['kind'], message: string) => {
    setToast({ id: Date.now(), kind, message })
  }, [])

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
    exportFormat,
    setExportFormat,
    exportProjection,
    setExportProjection,
    exportContent,
    generateExport,
    refreshActiveView,
  } = useCockpitData({
    selection,
    topologyLimit,
    deepDiveLimit,
    deepDiveMode,
    onScenarioAutoSelect: setScenarioId,
    onViewError,
  })

  const selectedScenario = useMemo(
    () => scenarios.find((scenario) => scenario.id === selection.scenarioId) ?? null,
    [scenarios, selection.scenarioId],
  )

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

  const handleCopyJson = async () => {
    const payload = JSON.stringify(city?.cc_json || {}, null, 2)
    try {
      await navigator.clipboard.writeText(payload)
      pushToast('success', 'Copied cc.json preview to clipboard.')
    } catch {
      pushToast('error', 'Could not copy to clipboard.')
    }
  }

  const handleDownloadJson = () => {
    const payload = JSON.stringify(city?.cc_json || {}, null, 2)
    downloadTextFile('cockpit-cc-preview.json', payload)
    pushToast('info', 'Downloaded cc.json preview.')
  }

  const handleOpenTopologyFromOverview = () => {
    setLayer('code_controlflow')
    setView('topology')
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
          onCollectionChange={setCollectionId}
          onScenarioChange={setScenarioId}
          onLayerChange={setLayer}
          onFilterChange={setHotspotFilter}
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
        onCollectionChange={setCollectionId}
        onScenarioChange={setScenarioId}
        onLayerChange={setLayer}
        onFilterChange={setHotspotFilter}
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
          onCopyJson={handleCopyJson}
          onDownloadJson={handleDownloadJson}
        />
      ) : null}

      {selection.view === 'topology' ? (
        <TopologyView
          graph={graph}
          state={activeState}
          error={activeError}
          layer={selection.layer}
          density={topologyDensity}
          onDensityChange={setTopologyDensity}
          onSwitchToCodeLayer={() => setLayer('code_controlflow')}
          onRetry={refreshActiveView}
        />
      ) : null}

      {selection.view === 'deep_dive' ? (
        <DeepDiveView
          graph={graph}
          state={activeState}
          error={activeError}
          layer={selection.layer}
          density={deepDiveDensity}
          mode={deepDiveMode}
          onModeChange={setDeepDiveMode}
          onDensityChange={setDeepDiveDensity}
          onSwitchToCodeLayer={() => setLayer('code_controlflow')}
          onRetry={refreshActiveView}
        />
      ) : null}

      {selection.view === 'c4_diff' ? (
        <C4DiffView
          mermaid={mermaid}
          state={activeState}
          error={activeError}
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
