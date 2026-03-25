import { describe, expect, it } from 'vitest'

import { formatDuration } from './formatters'

describe('formatDuration', () => {
  it('returns fallback for null', () => {
    expect(formatDuration(null)).toBe('-')
  })

  it('returns fallback for undefined', () => {
    expect(formatDuration(undefined)).toBe('-')
  })

  it('returns custom fallback when provided', () => {
    expect(formatDuration(null, 'N/A')).toBe('N/A')
  })

  it('formats sub-second durations in milliseconds', () => {
    expect(formatDuration(0)).toBe('0ms')
    expect(formatDuration(1)).toBe('1ms')
    expect(formatDuration(500)).toBe('500ms')
    expect(formatDuration(999)).toBe('999ms')
  })

  it('formats seconds with one decimal', () => {
    expect(formatDuration(1000)).toBe('1.0s')
    expect(formatDuration(4200)).toBe('4.2s')
    expect(formatDuration(59999)).toBe('60.0s')
  })

  it('formats minutes and seconds', () => {
    expect(formatDuration(60000)).toBe('1m 0s')
    expect(formatDuration(133000)).toBe('2m 13s')
    expect(formatDuration(3600000)).toBe('60m 0s')
  })
})
