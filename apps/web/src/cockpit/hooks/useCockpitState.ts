import { useCallback, useEffect, useState } from 'react'

import {
  type CockpitLayer,
  type CockpitSelection,
  type OverlayMode,
  type CockpitView,
  COCKPIT_LAYERS,
  COCKPIT_VIEWS,
  DEFAULT_LAYER,
  DEFAULT_VIEW,
} from '../types'

const VALID_LAYERS: CockpitLayer[] = COCKPIT_LAYERS.map((l) => l.key)
const VALID_VIEWS: CockpitView[] = COCKPIT_VIEWS.map((v) => v.key)
const VALID_OVERLAYS: OverlayMode[] = ['none', 'runtime', 'risk']
const DEFAULT_PAGE = 0
const DEFAULT_LIMIT = 1200

function parseCsvParam(value: string | null): string[] {
  if (!value) return []
  return value.split(',').map((entry) => entry.trim()).filter(Boolean)
}

function parseInitialSelection(): CockpitSelection {
  const params = new URLSearchParams(globalThis.location.search)
  const rawLayer = params.get('layer')
  const rawView = params.get('view')
  const normalizedRawView = rawView === 'user_flows' ? 'ui_map' : rawView

  return {
    collectionId: params.get('collection') ?? '',
    scenarioId: params.get('scenario') ?? '',
    layer: VALID_LAYERS.includes(rawLayer as CockpitLayer)
      ? (rawLayer as CockpitLayer)
      : DEFAULT_LAYER,
    view: VALID_VIEWS.includes(normalizedRawView as CockpitView)
      ? (normalizedRawView as CockpitView)
      : DEFAULT_VIEW,
  }
}

function writeSelectionToUrl(args: {
  selection: CockpitSelection
  graphQuery: string
  selectedNodeId: string
  graphPage: number
  graphLimit: number
  includeKinds: string[]
  excludeKinds: string[]
  overlayMode: OverlayMode
  hideIsolated: boolean
  edgeKinds: string[]
}): void {
  const {
    selection,
    graphQuery,
    selectedNodeId,
    graphPage,
    graphLimit,
    includeKinds,
    excludeKinds,
    overlayMode,
    hideIsolated,
    edgeKinds,
  } = args

  const params = new URLSearchParams(globalThis.location.search)
  params.set('page', 'cockpit')

  if (selection.collectionId) {
    params.set('collection', selection.collectionId)
  } else {
    params.delete('collection')
  }

  if (selection.scenarioId) {
    params.set('scenario', selection.scenarioId)
  } else {
    params.delete('scenario')
  }

  if (selection.view !== DEFAULT_VIEW) {
    params.set('view', selection.view)
  } else {
    params.delete('view')
  }

  if (selection.layer !== DEFAULT_LAYER) {
    params.set('layer', selection.layer)
  } else {
    params.delete('layer')
  }

  if (graphQuery.trim()) {
    params.set('query', graphQuery.trim())
  } else {
    params.delete('query')
  }

  if (selectedNodeId.trim()) {
    params.set('node', selectedNodeId.trim())
  } else {
    params.delete('node')
  }

  if (graphPage > DEFAULT_PAGE) {
    params.set('pageIndex', String(graphPage))
  } else {
    params.delete('pageIndex')
  }

  if (graphLimit !== DEFAULT_LIMIT) {
    params.set('limit', String(graphLimit))
  } else {
    params.delete('limit')
  }

  if (includeKinds.length > 0) {
    params.set('includeKinds', includeKinds.join(','))
  } else {
    params.delete('includeKinds')
  }

  if (excludeKinds.length > 0) {
    params.set('excludeKinds', excludeKinds.join(','))
  } else {
    params.delete('excludeKinds')
  }

  if (overlayMode !== 'none') {
    params.set('overlay', overlayMode)
  } else {
    params.delete('overlay')
  }

  if (hideIsolated) {
    params.set('hideIsolated', '1')
  } else {
    params.delete('hideIsolated')
  }

  if (edgeKinds.length > 0) {
    params.set('edgeKinds', edgeKinds.join(','))
  } else {
    params.delete('edgeKinds')
  }

  const nextQuery = params.toString()
  const nextUrl = nextQuery ? `${globalThis.location.pathname}?${nextQuery}` : globalThis.location.pathname
  globalThis.history.replaceState({}, '', nextUrl)
}

export function useCockpitState() {
  const [selection, setSelection] = useState<CockpitSelection>(() => parseInitialSelection())
  const [hotspotFilter, setHotspotFilter] = useState('')
  const [graphQuery, setGraphQuery] = useState(() => {
    return new URLSearchParams(globalThis.location.search).get('query') ?? ''
  })
  const [selectedNodeId, setSelectedNodeId] = useState(() => {
    return new URLSearchParams(globalThis.location.search).get('node') ?? ''
  })
  const [graphPage, setGraphPage] = useState(() => {
    const raw = Number(new URLSearchParams(globalThis.location.search).get('pageIndex') ?? DEFAULT_PAGE)
    return Number.isFinite(raw) && raw >= 0 ? raw : DEFAULT_PAGE
  })
  const [graphLimit, setGraphLimit] = useState(() => {
    const raw = Number(new URLSearchParams(globalThis.location.search).get('limit') ?? DEFAULT_LIMIT)
    return Number.isFinite(raw) && raw >= 1 ? raw : DEFAULT_LIMIT
  })
  const [includeKinds, setIncludeKinds] = useState<string[]>(() => {
    return parseCsvParam(new URLSearchParams(globalThis.location.search).get('includeKinds'))
  })
  const [excludeKinds, setExcludeKinds] = useState<string[]>(() => {
    return parseCsvParam(new URLSearchParams(globalThis.location.search).get('excludeKinds'))
  })
  const [edgeKinds, setEdgeKinds] = useState<string[]>(() => {
    return parseCsvParam(new URLSearchParams(globalThis.location.search).get('edgeKinds'))
  })
  const [hideIsolated, setHideIsolated] = useState(() => {
    return (new URLSearchParams(globalThis.location.search).get('hideIsolated') ?? '') === '1'
  })
  const [overlayMode, setOverlayMode] = useState<OverlayMode>(() => {
    const raw = (new URLSearchParams(globalThis.location.search).get('overlay') ?? 'none') as OverlayMode
    return VALID_OVERLAYS.includes(raw) ? raw : 'none'
  })

  useEffect(() => {
    writeSelectionToUrl({
      selection,
      graphQuery,
      selectedNodeId,
      graphPage,
      graphLimit,
      includeKinds,
      excludeKinds,
      overlayMode,
      hideIsolated,
      edgeKinds,
    })
  }, [
    selection,
    graphQuery,
    selectedNodeId,
    graphPage,
    graphLimit,
    includeKinds,
    excludeKinds,
    overlayMode,
    hideIsolated,
    edgeKinds,
  ])

  const updateSelection = useCallback((patch: Partial<CockpitSelection>) => {
    setSelection((prev) => ({ ...prev, ...patch }))
  }, [])

  const setCollectionId = useCallback(
    (collectionId: string) => {
      setSelection((prev) => ({
        ...prev,
        collectionId,
        scenarioId: '',
      }))
      setSelectedNodeId('')
      setGraphPage(DEFAULT_PAGE)
    },
    [],
  )

  const setScenarioId = useCallback(
    (scenarioId: string) => {
      updateSelection({ scenarioId })
      setSelectedNodeId('')
      setGraphPage(DEFAULT_PAGE)
    },
    [updateSelection],
  )

  const setLayer = useCallback(
    (layer: CockpitLayer) => {
      updateSelection({ layer })
      setSelectedNodeId('')
      setGraphPage(DEFAULT_PAGE)
    },
    [updateSelection],
  )

  const setView = useCallback(
    (view: CockpitView) => {
      updateSelection({ view: view === 'user_flows' ? 'ui_map' : view })
      setSelectedNodeId('')
      setGraphPage(DEFAULT_PAGE)
    },
    [updateSelection],
  )

  return {
    selection,
    hotspotFilter,
    setHotspotFilter,
    graphQuery,
    setGraphQuery,
    selectedNodeId,
    setSelectedNodeId,
    graphPage,
    setGraphPage,
    graphLimit,
    setGraphLimit,
    includeKinds,
    setIncludeKinds,
    excludeKinds,
    setExcludeKinds,
    edgeKinds,
    setEdgeKinds,
    hideIsolated,
    setHideIsolated,
    overlayMode,
    setOverlayMode,
    setCollectionId,
    setScenarioId,
    setLayer,
    setView,
    updateSelection,
  }
}
