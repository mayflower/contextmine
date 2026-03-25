import { describe, expect, it } from 'vitest'

import { runGridLayout, type LayoutNodeInput } from './layoutCore'

describe('runGridLayout', () => {
  it('returns empty map for empty nodes', () => {
    const result = runGridLayout([], 3)
    expect(result).toEqual({})
  })

  it('positions a single node at origin', () => {
    const nodes: LayoutNodeInput[] = [{ id: 'a' }]
    const result = runGridLayout(nodes, 3)
    expect(result).toEqual({ a: { x: 0, y: 0 } })
  })

  it('lays out nodes in a grid with correct columns', () => {
    const nodes: LayoutNodeInput[] = [
      { id: 'a' },
      { id: 'b' },
      { id: 'c' },
      { id: 'd' },
      { id: 'e' },
    ]
    const result = runGridLayout(nodes, 3)

    // Row 0: a(0,0), b(260,0), c(520,0)
    expect(result['a']).toEqual({ x: 0, y: 0 })
    expect(result['b']).toEqual({ x: 260, y: 0 })
    expect(result['c']).toEqual({ x: 520, y: 0 })
    // Row 1: d(0,130), e(260,130)
    expect(result['d']).toEqual({ x: 0, y: 130 })
    expect(result['e']).toEqual({ x: 260, y: 130 })
  })

  it('handles columns=0 by treating it as 1', () => {
    const nodes: LayoutNodeInput[] = [{ id: 'a' }, { id: 'b' }]
    const result = runGridLayout(nodes, 0)
    // safeColumns = max(0, 1) = 1, so all items in a single column
    expect(result['a']).toEqual({ x: 0, y: 0 })
    expect(result['b']).toEqual({ x: 0, y: 130 })
  })

  it('handles columns=1 (single column)', () => {
    const nodes: LayoutNodeInput[] = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    const result = runGridLayout(nodes, 1)
    expect(result['a']).toEqual({ x: 0, y: 0 })
    expect(result['b']).toEqual({ x: 0, y: 130 })
    expect(result['c']).toEqual({ x: 0, y: 260 })
  })

  it('handles more columns than nodes', () => {
    const nodes: LayoutNodeInput[] = [{ id: 'a' }, { id: 'b' }]
    const result = runGridLayout(nodes, 10)
    expect(result['a']).toEqual({ x: 0, y: 0 })
    expect(result['b']).toEqual({ x: 260, y: 0 })
  })
})
