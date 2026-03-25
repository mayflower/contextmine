import { describe, expect, it } from 'vitest'

import { cockpitFlags } from './flags'

describe('cockpitFlags', () => {
  it('has inspector flag', () => {
    expect(typeof cockpitFlags.inspector).toBe('boolean')
  })

  it('has elkLayout flag', () => {
    expect(typeof cockpitFlags.elkLayout).toBe('boolean')
  })

  it('has c4RenderedDiff flag', () => {
    expect(typeof cockpitFlags.c4RenderedDiff).toBe('boolean')
  })

  it('defaults inspector to true when env var is not false', () => {
    // In test environment, VITE_COCKPIT_INSPECTOR is not set,
    // so import.meta.env.VITE_COCKPIT_INSPECTOR is undefined,
    // and undefined !== 'false' is true
    expect(cockpitFlags.inspector).toBe(true)
  })
})
