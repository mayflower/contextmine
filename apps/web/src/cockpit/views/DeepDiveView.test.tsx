/**
 * Tests for DeepDiveView pure helper functions.
 * The component depends on cytoscape, so we only test pure logic.
 */
import { describe, expect, it } from 'vitest'

// Replicated from DeepDiveView.tsx (line 24-26)
function getLayoutName(density: number): 'cose' | 'breadthfirst' {
  return density > 5000 ? 'breadthfirst' : 'cose'
}

describe('getLayoutName', () => {
  it('returns cose for density <= 5000', () => {
    expect(getLayoutName(5000)).toBe('cose')
    expect(getLayoutName(1000)).toBe('cose')
    expect(getLayoutName(0)).toBe('cose')
  })

  it('returns breadthfirst for density > 5000', () => {
    expect(getLayoutName(5001)).toBe('breadthfirst')
    expect(getLayoutName(10000)).toBe('breadthfirst')
  })
})
