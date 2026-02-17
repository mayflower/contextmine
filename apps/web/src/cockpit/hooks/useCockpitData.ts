import { useCallback, useEffect, useMemo, useState } from 'react'

import type {
  CityPayload,
  CockpitLayer,
  CockpitLoadState,
  CockpitProjection,
  CockpitSelection,
  CockpitView,
  DeepDiveMode,
  ExportFormat,
  MermaidPayload,
  ScenarioLite,
  TwinGraphResponse,
} from '../types'

type DataStates = Record<CockpitView, CockpitLoadState>
type DataErrors = Record<CockpitView, string>
type DataUpdated = Partial<Record<CockpitView, string>>

const DEFAULT_GRAPH: TwinGraphResponse = {
  nodes: [],
  edges: [],
  page: 0,
  limit: 0,
  total_nodes: 0,
}

const DEFAULT_STATES: DataStates = {
  overview: 'idle',
  topology: 'idle',
  deep_dive: 'idle',
  c4_diff: 'idle',
  exports: 'idle',
}

const DEFAULT_ERRORS: DataErrors = {
  overview: '',
  topology: '',
  deep_dive: '',
  c4_diff: '',
  exports: '',
}

interface UseCockpitDataArgs {
  selection: CockpitSelection
  topologyLimit: number
  deepDiveLimit: number
  deepDiveMode: DeepDiveMode
  onScenarioAutoSelect: (scenarioId: string) => void
  onViewError?: (view: CockpitView, message: string) => void
}

function topologyEntityLevel(layer: CockpitLayer): 'domain' | 'container' | 'component' {
  if (layer === 'portfolio_system') return 'domain'
  if (layer === 'component_interface') return 'component'
  return 'container'
}

export function useCockpitData({
  selection,
  topologyLimit,
  deepDiveLimit,
  deepDiveMode,
  onScenarioAutoSelect,
  onViewError,
}: UseCockpitDataArgs) {
  const [scenarios, setScenarios] = useState<ScenarioLite[]>([])
  const [scenariosState, setScenariosState] = useState<CockpitLoadState>('idle')
  const [city, setCity] = useState<CityPayload | null>(null)
  const [graph, setGraph] = useState<TwinGraphResponse>(DEFAULT_GRAPH)
  const [mermaid, setMermaid] = useState<MermaidPayload | null>(null)
  const [states, setStates] = useState<DataStates>(DEFAULT_STATES)
  const [errors, setErrors] = useState<DataErrors>(DEFAULT_ERRORS)
  const [updatedAt, setUpdatedAt] = useState<DataUpdated>({})
  const [refreshNonce, setRefreshNonce] = useState(0)
  const [exportFormat, setExportFormat] = useState<ExportFormat>('cc_json')
  const [exportProjection, setExportProjection] = useState<CockpitProjection>('architecture')
  const [exportContent, setExportContent] = useState('')

  const setViewState = useCallback((view: CockpitView, nextState: CockpitLoadState) => {
    setStates((prev) => ({ ...prev, [view]: nextState }))
  }, [])

  const setViewError = useCallback(
    (view: CockpitView, message: string) => {
      setErrors((prev) => ({ ...prev, [view]: message }))
      if (message && onViewError) {
        onViewError(view, message)
      }
    },
    [onViewError],
  )

  const markUpdated = useCallback((view: CockpitView) => {
    setUpdatedAt((prev) => ({ ...prev, [view]: new Date().toISOString() }))
  }, [])

  const refreshActiveView = useCallback(() => {
    setRefreshNonce((prev) => prev + 1)
  }, [])

  useEffect(() => {
    if (!selection.collectionId) {
      setScenarios([])
      setScenariosState('empty')
      return
    }

    const controller = new AbortController()

    const run = async () => {
      setScenariosState('loading')
      try {
        const response = await fetch(`/api/twin/scenarios?collection_id=${selection.collectionId}`, {
          credentials: 'include',
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`Could not load scenarios (${response.status})`)
        }

        const payload = await response.json()
        const nextScenarios: ScenarioLite[] = payload.scenarios || []
        setScenarios(nextScenarios)

        if (nextScenarios.length === 0) {
          setScenariosState('empty')
          return
        }

        const selectedScenario = nextScenarios.find((scenario) => scenario.id === selection.scenarioId)
        if (!selectedScenario) {
          const asIs = nextScenarios.find((scenario) => scenario.is_as_is)
          onScenarioAutoSelect(asIs?.id || nextScenarios[0].id)
        }

        setScenariosState('ready')
      } catch (error) {
        if (controller.signal.aborted) {
          return
        }
        setScenariosState('error')
        setScenarios([])
        setViewError(selection.view, error instanceof Error ? error.message : 'Could not load scenarios')
      }
    }

    run()

    return () => {
      controller.abort()
    }
  }, [selection.collectionId, selection.scenarioId, selection.view, onScenarioAutoSelect, setViewError])

  useEffect(() => {
    const { collectionId, scenarioId, view, layer } = selection
    if (!collectionId || !scenarioId) {
      setViewState(view, 'empty')
      return
    }

    if (view === 'exports') {
      setViewState('exports', 'ready')
      return
    }

    const controller = new AbortController()

    const run = async () => {
      setViewState(view, 'loading')
      setViewError(view, '')

      try {
        if (view === 'overview') {
          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/city?scenario_id=${scenarioId}&hotspots_limit=60`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load overview (${response.status})`)
          }
          const payload: CityPayload = await response.json()
          setCity(payload)
          setViewState('overview', 'ready')
          markUpdated('overview')
          return
        }

        if (view === 'topology' || view === 'deep_dive') {
          const endpoint = view === 'topology' ? 'topology' : 'deep-dive'
          const limit = view === 'topology' ? topologyLimit : deepDiveLimit
          const query = new URLSearchParams({
            scenario_id: scenarioId,
            layer,
            limit: String(limit),
          })

          if (view === 'topology') {
            query.set('projection', 'architecture')
            query.set('entity_level', topologyEntityLevel(layer))
          } else {
            if (deepDiveMode === 'file_dependency') {
              query.set('projection', 'code_file')
              query.set('entity_level', 'file')
            } else if (deepDiveMode === 'symbol_callgraph') {
              query.set('projection', 'code_symbol')
              query.set('entity_level', 'symbol')
              query.set('mode', 'symbol_callgraph')
            } else {
              query.set('projection', 'code_symbol')
              query.set('entity_level', 'symbol')
              query.set('mode', 'contains_hierarchy')
            }
          }

          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/${endpoint}?${query.toString()}`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load ${endpoint} (${response.status})`)
          }
          const payload = await response.json()
          setGraph({
            ...(payload.graph || DEFAULT_GRAPH),
            projection: payload.projection ?? payload.graph?.projection,
            entity_level: payload.entity_level ?? payload.graph?.entity_level,
            grouping_strategy: payload.grouping_strategy ?? payload.graph?.grouping_strategy,
            excluded_kinds: payload.excluded_kinds ?? payload.graph?.excluded_kinds,
          })
          setViewState(view, 'ready')
          markUpdated(view)
          return
        }

        if (view === 'c4_diff') {
          const response = await fetch(
            `/api/twin/collections/${collectionId}/views/mermaid?scenario_id=${scenarioId}&compare_with_base=true`,
            {
              credentials: 'include',
              signal: controller.signal,
            },
          )
          if (!response.ok) {
            throw new Error(`Could not load C4 view (${response.status})`)
          }
          const payload: MermaidPayload = await response.json()
          setMermaid(payload)
          setViewState('c4_diff', 'ready')
          markUpdated('c4_diff')
        }
      } catch (error) {
        if (controller.signal.aborted) {
          return
        }
        const message = error instanceof Error ? error.message : 'Unexpected Cockpit request error'
        setViewError(view, message)
        setViewState(view, 'error')
      }
    }

    run()

    return () => {
      controller.abort()
    }
  }, [
    selection,
    topologyLimit,
    deepDiveLimit,
    deepDiveMode,
    refreshNonce,
    markUpdated,
    setViewError,
    setViewState,
  ])

  const generateExport = useCallback(async () => {
    const scenarioId = selection.scenarioId
    if (!scenarioId) {
      setViewError('exports', 'Select a scenario before generating an export.')
      setViewState('exports', 'error')
      return null
    }

    setViewError('exports', '')
    setViewState('exports', 'loading')

    try {
      const exportResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          format: exportFormat,
          projection: exportProjection,
          entity_level:
            exportProjection === 'architecture'
              ? topologyEntityLevel(selection.layer)
              : exportProjection === 'code_file'
                ? 'file'
                : 'symbol',
        }),
      })

      if (!exportResponse.ok) {
        throw new Error(`Could not generate export (${exportResponse.status})`)
      }

      const exportData = await exportResponse.json()
      const exportId = exportData.id || exportData.exports?.[0]?.id
      if (!exportId) {
        throw new Error('Missing export id from API response')
      }

      const artifactResponse = await fetch(`/api/twin/scenarios/${scenarioId}/exports/${exportId}`, {
        credentials: 'include',
      })

      if (!artifactResponse.ok) {
        throw new Error(`Could not fetch export artifact (${artifactResponse.status})`)
      }

      const artifact = await artifactResponse.json()
      const content = artifact.content || ''
      setExportContent(content)
      setViewState('exports', 'ready')
      markUpdated('exports')
      return {
        content,
        name: artifact.name || `${exportFormat}.txt`,
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Export generation failed'
      setViewError('exports', message)
      setViewState('exports', 'error')
      return null
    }
  }, [
    exportFormat,
    exportProjection,
    markUpdated,
    selection.layer,
    selection.scenarioId,
    setViewError,
    setViewState,
  ])

  const activeState = useMemo(() => states[selection.view], [states, selection.view])
  const activeError = useMemo(() => errors[selection.view], [errors, selection.view])
  const activeUpdatedAt = useMemo(
    () => updatedAt[selection.view] ?? null,
    [updatedAt, selection.view],
  )

  return {
    scenarios,
    scenariosState,
    city,
    graph,
    mermaid,
    states,
    errors,
    activeState,
    activeError,
    activeUpdatedAt,
    exportFormat,
    setExportFormat,
    exportProjection,
    setExportProjection,
    exportContent,
    setExportContent,
    generateExport,
    refreshActiveView,
  }
}
