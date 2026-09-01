import { useCallback, useState } from 'react'
import { useSearchParams } from 'react-router'

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

const VALID_LAYERS = new Set<CockpitLayer>(COCKPIT_LAYERS.map((layer) => layer.key))
const VALID_VIEWS = new Set<CockpitView>(COCKPIT_VIEWS.map((view) => view.key))
const VALID_OVERLAYS = new Set<OverlayMode>(['none', 'runtime', 'risk'])
const DEFAULT_PAGE = 0
const DEFAULT_LIMIT = 1200

function parseCsvParam(value: string | null): string[] {
  if (!value) return []
  return value.split(',').map((entry) => entry.trim()).filter(Boolean)
}

function setOrDelete(params: URLSearchParams, key: string, value: string | false): void {
  if (value) params.set(key, value)
  else params.delete(key)
}

function parseSelection(params: URLSearchParams): CockpitSelection {
  const rawLayer = params.get('layer')
  const rawView = params.get('view')
  const normalizedView = rawView === 'user_flows' ? 'ui_map' : rawView
  return {
    collectionId: params.get('collection') ?? '',
    scenarioId: params.get('scenario') ?? '',
    layer: VALID_LAYERS.has(rawLayer as CockpitLayer) ? rawLayer as CockpitLayer : DEFAULT_LAYER,
    view: VALID_VIEWS.has(normalizedView as CockpitView) ? normalizedView as CockpitView : DEFAULT_VIEW,
  }
}

function parseNumber(value: string | null, fallback: number, minimum: number): number {
  const parsed = Number(value ?? fallback)
  return Number.isFinite(parsed) && parsed >= minimum ? parsed : fallback
}

export function useCockpitState() {
  const [params, setParams] = useSearchParams()
  const [hotspotFilter, setHotspotFilter] = useState('')

  const updateParams = useCallback((mutate: (next: URLSearchParams) => void) => {
    setParams((current) => {
      const next = new URLSearchParams(current)
      next.delete('page')
      mutate(next)
      return next
    }, { replace: true })
  }, [setParams])

  const selection = parseSelection(params)
  const graphQuery = params.get('query') ?? ''
  const selectedNodeId = params.get('node') ?? ''
  const graphPage = parseNumber(params.get('pageIndex'), DEFAULT_PAGE, 0)
  const graphLimit = parseNumber(params.get('limit'), DEFAULT_LIMIT, 1)
  const includeKinds = parseCsvParam(params.get('includeKinds'))
  const excludeKinds = parseCsvParam(params.get('excludeKinds'))
  const edgeKinds = parseCsvParam(params.get('edgeKinds'))
  const hideIsolated = params.get('hideIsolated') === '1'
  const rawOverlay = (params.get('overlay') ?? 'none') as OverlayMode
  const overlayMode = VALID_OVERLAYS.has(rawOverlay) ? rawOverlay : 'none'

  const resetGraphSelection = useCallback((next: URLSearchParams) => {
    next.delete('node')
    next.delete('pageIndex')
  }, [])

  const updateSelection = useCallback((patch: Partial<CockpitSelection>) => {
    updateParams((next) => {
      const updated = { ...parseSelection(next), ...patch }
      setOrDelete(next, 'collection', updated.collectionId || false)
      setOrDelete(next, 'scenario', updated.scenarioId || false)
      setOrDelete(next, 'view', updated.view !== DEFAULT_VIEW && updated.view)
      setOrDelete(next, 'layer', updated.layer !== DEFAULT_LAYER && updated.layer)
    })
  }, [updateParams])

  const setCollectionId = useCallback((collectionId: string) => {
    updateParams((next) => {
      setOrDelete(next, 'collection', collectionId || false)
      next.delete('scenario')
      resetGraphSelection(next)
    })
  }, [resetGraphSelection, updateParams])

  const setScenarioId = useCallback((scenarioId: string) => {
    updateParams((next) => {
      setOrDelete(next, 'scenario', scenarioId || false)
      resetGraphSelection(next)
    })
  }, [resetGraphSelection, updateParams])

  const setLayer = useCallback((layer: CockpitLayer) => {
    updateParams((next) => {
      setOrDelete(next, 'layer', layer !== DEFAULT_LAYER && layer)
      resetGraphSelection(next)
    })
  }, [resetGraphSelection, updateParams])

  const setView = useCallback((view: CockpitView) => {
    const normalized = view === 'user_flows' ? 'ui_map' : view
    updateParams((next) => {
      setOrDelete(next, 'view', normalized !== DEFAULT_VIEW && normalized)
      resetGraphSelection(next)
    })
  }, [resetGraphSelection, updateParams])

  const setStringParam = useCallback((key: string, value: string) => {
    updateParams((next) => setOrDelete(next, key, value.trim() || false))
  }, [updateParams])

  return {
    selection,
    hotspotFilter,
    setHotspotFilter,
    graphQuery,
    setGraphQuery: (value: string) => setStringParam('query', value),
    selectedNodeId,
    setSelectedNodeId: (value: string) => setStringParam('node', value),
    graphPage,
    setGraphPage: (value: number) => updateParams((next) => setOrDelete(next, 'pageIndex', value > DEFAULT_PAGE && String(value))),
    graphLimit,
    setGraphLimit: (value: number) => updateParams((next) => setOrDelete(next, 'limit', value !== DEFAULT_LIMIT && String(value))),
    includeKinds,
    setIncludeKinds: (values: string[]) => updateParams((next) => setOrDelete(next, 'includeKinds', values.length > 0 && values.join(','))),
    excludeKinds,
    setExcludeKinds: (values: string[]) => updateParams((next) => setOrDelete(next, 'excludeKinds', values.length > 0 && values.join(','))),
    edgeKinds,
    setEdgeKinds: (values: string[]) => updateParams((next) => setOrDelete(next, 'edgeKinds', values.length > 0 && values.join(','))),
    hideIsolated,
    setHideIsolated: (value: boolean) => updateParams((next) => setOrDelete(next, 'hideIsolated', value && '1')),
    overlayMode,
    setOverlayMode: (value: OverlayMode) => updateParams((next) => setOrDelete(next, 'overlay', value !== 'none' && value)),
    setCollectionId,
    setScenarioId,
    setLayer,
    setView,
    updateSelection,
  }
}
