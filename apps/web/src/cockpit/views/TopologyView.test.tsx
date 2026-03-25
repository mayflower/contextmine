/**
 * Tests for TopologyView pure helper functions.
 * The component itself depends on ReactFlow and layout engines,
 * so we test the extractable pure logic.
 */
import { describe, expect, it } from 'vitest'

import type { OverlayState } from '../types'

// Replicated from TopologyView.tsx (lines 34-50)
function overlayColorForNode(overlay: OverlayState, naturalKey: string, fallbackName: string): string {
  if (overlay.mode === 'runtime') {
    const runtime = overlay.runtimeByNodeKey[naturalKey] || overlay.runtimeByNodeKey[fallbackName]
    const errorRate = Number(runtime?.error_rate || 0)
    if (errorRate >= 0.1) return '#dc2626'
    if (errorRate >= 0.03) return '#f59e0b'
    return '#2563eb'
  }
  if (overlay.mode === 'risk') {
    const risk = overlay.riskByNodeKey[naturalKey] || overlay.riskByNodeKey[fallbackName]
    const severity = Number(risk?.severity_score || 0)
    if (severity >= 8) return '#b91c1c'
    if (severity >= 4) return '#d97706'
    return '#1d4ed8'
  }
  return '#1d4ed8'
}

const noOverlay: OverlayState = {
  mode: 'none',
  runtimeByNodeKey: {},
  riskByNodeKey: {},
  loadedAt: null,
}

describe('overlayColorForNode', () => {
  it('returns default blue when mode is none', () => {
    expect(overlayColorForNode(noOverlay, 'key', 'name')).toBe('#1d4ed8')
  })

  describe('runtime mode', () => {
    const runtimeOverlay: OverlayState = {
      mode: 'runtime',
      runtimeByNodeKey: {
        'service:auth': { service: 'auth', error_rate: 0.15 },
        'service:billing': { service: 'billing', error_rate: 0.05 },
        'service:healthy': { service: 'healthy', error_rate: 0.01 },
      },
      riskByNodeKey: {},
      loadedAt: null,
    }

    it('returns red for high error rate (>=0.1)', () => {
      expect(overlayColorForNode(runtimeOverlay, 'service:auth', '')).toBe('#dc2626')
    })

    it('returns amber for medium error rate (>=0.03)', () => {
      expect(overlayColorForNode(runtimeOverlay, 'service:billing', '')).toBe('#f59e0b')
    })

    it('returns blue for low error rate', () => {
      expect(overlayColorForNode(runtimeOverlay, 'service:healthy', '')).toBe('#2563eb')
    })

    it('uses fallback name when natural key not found', () => {
      expect(overlayColorForNode(runtimeOverlay, 'nonexistent', 'service:auth')).toBe('#dc2626')
    })

    it('returns blue when no runtime data exists', () => {
      expect(overlayColorForNode(runtimeOverlay, 'unknown', 'also_unknown')).toBe('#2563eb')
    })
  })

  describe('risk mode', () => {
    const riskOverlay: OverlayState = {
      mode: 'risk',
      runtimeByNodeKey: {},
      riskByNodeKey: {
        'api-gateway': { node: 'api-gateway', severity_score: 9 },
        'user-service': { node: 'user-service', severity_score: 5 },
        'static-assets': { node: 'static-assets', severity_score: 1 },
      },
      loadedAt: null,
    }

    it('returns dark red for high severity (>=8)', () => {
      expect(overlayColorForNode(riskOverlay, 'api-gateway', '')).toBe('#b91c1c')
    })

    it('returns amber for medium severity (>=4)', () => {
      expect(overlayColorForNode(riskOverlay, 'user-service', '')).toBe('#d97706')
    })

    it('returns blue for low severity', () => {
      expect(overlayColorForNode(riskOverlay, 'static-assets', '')).toBe('#1d4ed8')
    })
  })
})
