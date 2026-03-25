import { describe, expect, it } from 'vitest'

import {
  COCKPIT_LAYERS,
  COCKPIT_VIEWS,
  DEFAULT_LAYER,
  DEFAULT_VIEW,
  EXPORT_FORMATS,
  layerLabel,
} from './types'

describe('COCKPIT_VIEWS', () => {
  it('is a non-empty array', () => {
    expect(COCKPIT_VIEWS.length).toBeGreaterThan(0)
  })

  it('each entry has key and label', () => {
    for (const view of COCKPIT_VIEWS) {
      expect(view.key).toBeTruthy()
      expect(view.label).toBeTruthy()
    }
  })

  it('contains overview, topology, deep_dive, city, and exports', () => {
    const keys = COCKPIT_VIEWS.map((v) => v.key)
    expect(keys).toContain('overview')
    expect(keys).toContain('topology')
    expect(keys).toContain('deep_dive')
    expect(keys).toContain('city')
    expect(keys).toContain('exports')
  })
})

describe('COCKPIT_LAYERS', () => {
  it('is a non-empty array', () => {
    expect(COCKPIT_LAYERS.length).toBeGreaterThan(0)
  })

  it('each entry has key and label', () => {
    for (const layer of COCKPIT_LAYERS) {
      expect(layer.key).toBeTruthy()
      expect(layer.label).toBeTruthy()
    }
  })

  it('contains code_controlflow', () => {
    const keys = COCKPIT_LAYERS.map((l) => l.key)
    expect(keys).toContain('code_controlflow')
  })
})

describe('DEFAULT_LAYER', () => {
  it('is code_controlflow', () => {
    expect(DEFAULT_LAYER).toBe('code_controlflow')
  })
})

describe('DEFAULT_VIEW', () => {
  it('is overview', () => {
    expect(DEFAULT_VIEW).toBe('overview')
  })
})

describe('layerLabel', () => {
  it('returns label for code_controlflow', () => {
    expect(layerLabel('code_controlflow')).toBe('Code / Controlflow')
  })

  it('returns label for portfolio_system', () => {
    expect(layerLabel('portfolio_system')).toBe('Portfolio / System')
  })

  it('returns label for domain_container', () => {
    expect(layerLabel('domain_container')).toBe('Domain / Container')
  })

  it('returns label for component_interface', () => {
    expect(layerLabel('component_interface')).toBe('Component / Interface')
  })
})

describe('EXPORT_FORMATS', () => {
  it('is a non-empty array', () => {
    expect(EXPORT_FORMATS.length).toBeGreaterThan(0)
  })

  it('each entry has key, label, and extension', () => {
    for (const format of EXPORT_FORMATS) {
      expect(format.key).toBeTruthy()
      expect(format.label).toBeTruthy()
      expect(format.extension).toBeTruthy()
    }
  })

  it('contains cc_json format', () => {
    const keys = EXPORT_FORMATS.map((f) => f.key)
    expect(keys).toContain('cc_json')
  })

  it('contains mermaid_c4 format', () => {
    const keys = EXPORT_FORMATS.map((f) => f.key)
    expect(keys).toContain('mermaid_c4')
  })
})
