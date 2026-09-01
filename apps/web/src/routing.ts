import type { CockpitLayer, CockpitView } from './cockpit/types'

export type Page = 'dashboard' | 'collections' | 'runs' | 'cockpit'

export const PAGE_PATHS: Record<Page, string> = {
  dashboard: '/',
  collections: '/collections',
  runs: '/runs',
  cockpit: '/cockpit',
}

export const DEFAULT_COCKPIT_VIEW: CockpitView = 'overview'
export const DEFAULT_COCKPIT_LAYER: CockpitLayer = 'code_controlflow'

export interface CockpitNavigationOptions {
  collectionId?: string
  scenarioId?: string
  view?: CockpitView
  layer?: CockpitLayer
}

const COCKPIT_PARAM_KEYS = [
  'collection', 'scenario', 'view', 'layer', 'query', 'node',
  'pageIndex', 'limit', 'includeKinds', 'excludeKinds', 'overlay', 'hideIsolated', 'edgeKinds',
]

export function routeLocation(
  page: Page,
  currentSearch: string,
  cockpitOptions?: CockpitNavigationOptions,
): { pathname: string; search: string } {
  const params = new URLSearchParams(currentSearch)
  params.delete('page')

  if (page === 'cockpit') {
    if (cockpitOptions?.collectionId) params.set('collection', cockpitOptions.collectionId)
    else if (cockpitOptions && !cockpitOptions.collectionId) params.delete('collection')

    if (cockpitOptions?.scenarioId) params.set('scenario', cockpitOptions.scenarioId)
    else if (cockpitOptions && !cockpitOptions.scenarioId) params.delete('scenario')

    params.set('view', cockpitOptions?.view || params.get('view') || DEFAULT_COCKPIT_VIEW)
    params.set('layer', cockpitOptions?.layer || params.get('layer') || DEFAULT_COCKPIT_LAYER)
  } else {
    COCKPIT_PARAM_KEYS.forEach((key) => params.delete(key))
  }

  return { pathname: PAGE_PATHS[page], search: params.toString() }
}

export function legacyPageLocation(pathname: string, search: string) {
  if (pathname !== '/') return null
  const params = new URLSearchParams(search)
  const page = params.get('page') as Page | null
  if (!page || !PAGE_PATHS[page]) return null
  params.delete('page')
  return { pathname: PAGE_PATHS[page], search: params.toString() }
}
