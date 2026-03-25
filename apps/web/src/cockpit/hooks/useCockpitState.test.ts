import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { useCockpitState } from './useCockpitState'

describe('useCockpitState', () => {
  beforeEach(() => {
    // Reset URL between tests
    globalThis.history.replaceState({}, '', '/')
  })

  it('returns default selection when URL has no params', () => {
    const { result } = renderHook(() => useCockpitState())

    expect(result.current.selection.collectionId).toBe('')
    expect(result.current.selection.scenarioId).toBe('')
    expect(result.current.selection.layer).toBe('code_controlflow')
    expect(result.current.selection.view).toBe('overview')
  })

  it('parses collection and scenario from URL', () => {
    globalThis.history.replaceState({}, '', '/?collection=c1&scenario=s1')

    const { result } = renderHook(() => useCockpitState())

    expect(result.current.selection.collectionId).toBe('c1')
    expect(result.current.selection.scenarioId).toBe('s1')
  })

  it('parses layer from URL', () => {
    globalThis.history.replaceState({}, '', '/?layer=domain_container')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.selection.layer).toBe('domain_container')
  })

  it('defaults to code_controlflow for invalid layer', () => {
    globalThis.history.replaceState({}, '', '/?layer=invalid_layer')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.selection.layer).toBe('code_controlflow')
  })

  it('parses view from URL', () => {
    globalThis.history.replaceState({}, '', '/?view=topology')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.selection.view).toBe('topology')
  })

  it('defaults to overview for invalid view', () => {
    globalThis.history.replaceState({}, '', '/?view=nonexistent')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.selection.view).toBe('overview')
  })

  it('normalizes user_flows to ui_map', () => {
    globalThis.history.replaceState({}, '', '/?view=user_flows')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.selection.view).toBe('ui_map')
  })

  it('parses graphQuery from URL', () => {
    globalThis.history.replaceState({}, '', '/?query=test+search')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.graphQuery).toBe('test search')
  })

  it('parses selectedNodeId from URL', () => {
    globalThis.history.replaceState({}, '', '/?node=n123')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.selectedNodeId).toBe('n123')
  })

  it('parses graphPage from URL', () => {
    globalThis.history.replaceState({}, '', '/?pageIndex=5')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.graphPage).toBe(5)
  })

  it('defaults graphPage to 0 for invalid value', () => {
    globalThis.history.replaceState({}, '', '/?pageIndex=notanumber')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.graphPage).toBe(0)
  })

  it('parses graphLimit from URL', () => {
    globalThis.history.replaceState({}, '', '/?limit=500')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.graphLimit).toBe(500)
  })

  it('defaults graphLimit to 1200 for invalid value', () => {
    globalThis.history.replaceState({}, '', '/?limit=0')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.graphLimit).toBe(1200)
  })

  it('parses includeKinds from comma-separated URL param', () => {
    globalThis.history.replaceState({}, '', '/?includeKinds=CONTAINER,COMPONENT')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.includeKinds).toEqual(['CONTAINER', 'COMPONENT'])
  })

  it('parses excludeKinds from comma-separated URL param', () => {
    globalThis.history.replaceState({}, '', '/?excludeKinds=DB_TABLE')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.excludeKinds).toEqual(['DB_TABLE'])
  })

  it('parses edgeKinds from comma-separated URL param', () => {
    globalThis.history.replaceState({}, '', '/?edgeKinds=CALLS,READS')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.edgeKinds).toEqual(['CALLS', 'READS'])
  })

  it('parses hideIsolated from URL', () => {
    globalThis.history.replaceState({}, '', '/?hideIsolated=1')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.hideIsolated).toBe(true)
  })

  it('defaults hideIsolated to false', () => {
    const { result } = renderHook(() => useCockpitState())
    expect(result.current.hideIsolated).toBe(false)
  })

  it('parses overlayMode from URL', () => {
    globalThis.history.replaceState({}, '', '/?overlay=runtime')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.overlayMode).toBe('runtime')
  })

  it('defaults overlay to none for invalid value', () => {
    globalThis.history.replaceState({}, '', '/?overlay=invalid')

    const { result } = renderHook(() => useCockpitState())
    expect(result.current.overlayMode).toBe('none')
  })

  it('setCollectionId resets scenarioId and selectedNodeId', () => {
    globalThis.history.replaceState({}, '', '/?collection=c1&scenario=s1&node=n1')

    const { result } = renderHook(() => useCockpitState())

    act(() => {
      result.current.setCollectionId('c2')
    })

    expect(result.current.selection.collectionId).toBe('c2')
    expect(result.current.selection.scenarioId).toBe('')
    expect(result.current.selectedNodeId).toBe('')
  })

  it('setScenarioId resets selectedNodeId', () => {
    globalThis.history.replaceState({}, '', '/?collection=c1&scenario=s1&node=n1')

    const { result } = renderHook(() => useCockpitState())

    act(() => {
      result.current.setScenarioId('s2')
    })

    expect(result.current.selection.scenarioId).toBe('s2')
    expect(result.current.selectedNodeId).toBe('')
  })

  it('setLayer resets selectedNodeId', () => {
    globalThis.history.replaceState({}, '', '/?node=n1')

    const { result } = renderHook(() => useCockpitState())

    act(() => {
      result.current.setLayer('domain_container')
    })

    expect(result.current.selection.layer).toBe('domain_container')
    expect(result.current.selectedNodeId).toBe('')
  })

  it('setView normalizes user_flows to ui_map', () => {
    const { result } = renderHook(() => useCockpitState())

    act(() => {
      result.current.setView('user_flows' as 'ui_map')
    })

    expect(result.current.selection.view).toBe('ui_map')
  })

  it('setView resets selectedNodeId', () => {
    globalThis.history.replaceState({}, '', '/?node=n1')

    const { result } = renderHook(() => useCockpitState())

    act(() => {
      result.current.setView('topology')
    })

    expect(result.current.selectedNodeId).toBe('')
  })

  it('updateSelection patches selection', () => {
    const { result } = renderHook(() => useCockpitState())

    act(() => {
      result.current.updateSelection({ view: 'city' })
    })

    expect(result.current.selection.view).toBe('city')
    // Other fields unchanged
    expect(result.current.selection.layer).toBe('code_controlflow')
  })

  it('returns setter functions for all state', () => {
    const { result } = renderHook(() => useCockpitState())

    expect(typeof result.current.setGraphQuery).toBe('function')
    expect(typeof result.current.setSelectedNodeId).toBe('function')
    expect(typeof result.current.setGraphPage).toBe('function')
    expect(typeof result.current.setGraphLimit).toBe('function')
    expect(typeof result.current.setIncludeKinds).toBe('function')
    expect(typeof result.current.setExcludeKinds).toBe('function')
    expect(typeof result.current.setEdgeKinds).toBe('function')
    expect(typeof result.current.setHideIsolated).toBe('function')
    expect(typeof result.current.setOverlayMode).toBe('function')
    expect(typeof result.current.setHotspotFilter).toBe('function')
  })
})
