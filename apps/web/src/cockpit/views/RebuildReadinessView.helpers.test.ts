/**
 * Tests for RebuildReadinessView pure helper function (percentage).
 */
import { describe, expect, it } from 'vitest'

// Replicated from RebuildReadinessView.tsx (line 11-13)
function percentage(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`
}

describe('percentage', () => {
  it('formats 0 as 0%', () => {
    expect(percentage(0)).toBe('0%')
  })

  it('formats 1 as 100%', () => {
    expect(percentage(1)).toBe('100%')
  })

  it('formats 0.5 as 50%', () => {
    expect(percentage(0.5)).toBe('50%')
  })

  it('formats 0.85 as 85%', () => {
    expect(percentage(0.85)).toBe('85%')
  })

  it('clamps values above 1 to 100%', () => {
    expect(percentage(1.5)).toBe('100%')
    expect(percentage(2)).toBe('100%')
  })

  it('clamps values below 0 to 0%', () => {
    expect(percentage(-0.5)).toBe('0%')
    expect(percentage(-1)).toBe('0%')
  })

  it('rounds to nearest integer percentage', () => {
    expect(percentage(0.333)).toBe('33%')
    expect(percentage(0.666)).toBe('67%')
    expect(percentage(0.995)).toBe('100%')
  })
})
