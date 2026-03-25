/**
 * Tests for NodeInspector pure helper functions.
 */
import { describe, expect, it } from 'vitest'

// Replicated from NodeInspector.tsx (lines 13-23)
function pretty(value: unknown): string {
  if (value === null || value === undefined) return 'N/A'
  if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(2) : 'N/A'
  return String(value)
}

function riskLevel(score: number, count: number): 'high' | 'medium' | 'low' {
  if (score >= 8 || count >= 10) return 'high'
  if (score >= 4 || count >= 1) return 'medium'
  return 'low'
}

describe('pretty', () => {
  it('returns N/A for null', () => {
    expect(pretty(null)).toBe('N/A')
  })

  it('returns N/A for undefined', () => {
    expect(pretty(undefined)).toBe('N/A')
  })

  it('formats finite numbers with 2 decimal places', () => {
    expect(pretty(3.14159)).toBe('3.14')
    expect(pretty(0)).toBe('0.00')
    expect(pretty(100)).toBe('100.00')
  })

  it('returns N/A for Infinity', () => {
    expect(pretty(Infinity)).toBe('N/A')
    expect(pretty(-Infinity)).toBe('N/A')
  })

  it('returns N/A for NaN', () => {
    expect(pretty(NaN)).toBe('N/A')
  })

  it('converts strings to string', () => {
    expect(pretty('hello')).toBe('hello')
  })

  it('converts booleans to string', () => {
    expect(pretty(true)).toBe('true')
    expect(pretty(false)).toBe('false')
  })
})

describe('riskLevel', () => {
  it('returns high for score >= 8', () => {
    expect(riskLevel(8, 0)).toBe('high')
    expect(riskLevel(10, 0)).toBe('high')
  })

  it('returns high for count >= 10', () => {
    expect(riskLevel(0, 10)).toBe('high')
    expect(riskLevel(0, 15)).toBe('high')
  })

  it('returns medium for score >= 4', () => {
    expect(riskLevel(4, 0)).toBe('medium')
    expect(riskLevel(7, 0)).toBe('medium')
  })

  it('returns medium for count >= 1', () => {
    expect(riskLevel(0, 1)).toBe('medium')
    expect(riskLevel(0, 9)).toBe('medium')
  })

  it('returns low for score < 4 and count < 1', () => {
    expect(riskLevel(0, 0)).toBe('low')
    expect(riskLevel(3, 0)).toBe('low')
  })
})
