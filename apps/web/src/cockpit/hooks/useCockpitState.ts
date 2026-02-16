import { useCallback, useEffect, useState } from 'react'

import {
  type CockpitLayer,
  type CockpitSelection,
  type CockpitView,
  DEFAULT_LAYER,
  DEFAULT_VIEW,
} from '../types'

const VALID_LAYERS: CockpitLayer[] = [
  'portfolio_system',
  'domain_container',
  'component_interface',
  'code_controlflow',
]

const VALID_VIEWS: CockpitView[] = ['overview', 'topology', 'deep_dive', 'c4_diff', 'exports']

function parseInitialSelection(): CockpitSelection {
  if (typeof window === 'undefined') {
    return {
      collectionId: '',
      scenarioId: '',
      layer: DEFAULT_LAYER,
      view: DEFAULT_VIEW,
    }
  }

  const params = new URLSearchParams(window.location.search)
  const rawLayer = params.get('layer')
  const rawView = params.get('view')

  return {
    collectionId: params.get('collection') ?? '',
    scenarioId: params.get('scenario') ?? '',
    layer: VALID_LAYERS.includes(rawLayer as CockpitLayer)
      ? (rawLayer as CockpitLayer)
      : DEFAULT_LAYER,
    view: VALID_VIEWS.includes(rawView as CockpitView) ? (rawView as CockpitView) : DEFAULT_VIEW,
  }
}

function writeSelectionToUrl(selection: CockpitSelection): void {
  if (typeof window === 'undefined') {
    return
  }

  const params = new URLSearchParams(window.location.search)
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

  const nextQuery = params.toString()
  const nextUrl = nextQuery ? `${window.location.pathname}?${nextQuery}` : window.location.pathname
  window.history.replaceState({}, '', nextUrl)
}

export function useCockpitState() {
  const [selection, setSelection] = useState<CockpitSelection>(() => parseInitialSelection())
  const [hotspotFilter, setHotspotFilter] = useState('')

  useEffect(() => {
    writeSelectionToUrl(selection)
  }, [selection])

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
    },
    [],
  )

  const setScenarioId = useCallback(
    (scenarioId: string) => {
      updateSelection({ scenarioId })
    },
    [updateSelection],
  )

  const setLayer = useCallback(
    (layer: CockpitLayer) => {
      updateSelection({ layer })
    },
    [updateSelection],
  )

  const setView = useCallback(
    (view: CockpitView) => {
      updateSelection({ view })
    },
    [updateSelection],
  )

  return {
    selection,
    hotspotFilter,
    setHotspotFilter,
    setCollectionId,
    setScenarioId,
    setLayer,
    setView,
    updateSelection,
  }
}
