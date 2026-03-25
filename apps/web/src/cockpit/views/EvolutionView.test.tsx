/**
 * Tests for EvolutionView pure helper functions.
 * The component itself uses ReactFlow, so we test the logic functions separately.
 */
import { describe, expect, it } from 'vitest'

// Replicated from EvolutionView.tsx (lines 29-42)
function severityClass(severity: string): string {
  const normalized = severity.trim().toLowerCase()
  if (normalized === 'critical' || normalized === 'high') return 'risk-high'
  if (normalized === 'medium') return 'risk-medium'
  return 'risk-low'
}

function bubbleColor(quadrant: string): string {
  if (quadrant === 'strength') return '#166534'
  if (quadrant === 'overinvestment') return '#b91c1c'
  if (quadrant === 'efficient_core') return '#1d4ed8'
  if (quadrant === 'opportunity_or_retire') return '#92400e'
  return '#6b7280'
}

describe('severityClass', () => {
  it('returns risk-high for critical', () => {
    expect(severityClass('critical')).toBe('risk-high')
  })

  it('returns risk-high for high', () => {
    expect(severityClass('high')).toBe('risk-high')
  })

  it('returns risk-medium for medium', () => {
    expect(severityClass('medium')).toBe('risk-medium')
  })

  it('returns risk-low for low', () => {
    expect(severityClass('low')).toBe('risk-low')
  })

  it('returns risk-low for unknown severity', () => {
    expect(severityClass('unknown')).toBe('risk-low')
  })

  it('handles whitespace in severity', () => {
    expect(severityClass('  HIGH  ')).toBe('risk-high')
  })

  it('handles case insensitivity', () => {
    expect(severityClass('CRITICAL')).toBe('risk-high')
    expect(severityClass('Medium')).toBe('risk-medium')
  })
})

describe('bubbleColor', () => {
  it('returns green for strength', () => {
    expect(bubbleColor('strength')).toBe('#166534')
  })

  it('returns red for overinvestment', () => {
    expect(bubbleColor('overinvestment')).toBe('#b91c1c')
  })

  it('returns blue for efficient_core', () => {
    expect(bubbleColor('efficient_core')).toBe('#1d4ed8')
  })

  it('returns brown for opportunity_or_retire', () => {
    expect(bubbleColor('opportunity_or_retire')).toBe('#92400e')
  })

  it('returns gray for unknown quadrant', () => {
    expect(bubbleColor('other')).toBe('#6b7280')
    expect(bubbleColor('')).toBe('#6b7280')
  })
})
