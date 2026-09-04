import { describe, expect, it } from 'vitest'

import { buildCityEmbedUrl } from './cityEmbed'

describe('buildCityEmbedUrl', () => {
  it('asks the viewer to show the exported dependency edges', () => {
    // Without an edge metric the embedded viewer leaves isEdgeMetricVisible
    // off and never draws the edges the export carries.
    const params = new URLSearchParams(
      buildCityEmbedUrl('/api/twin/scenarios/s1/exports/e1/raw').split('?')[1],
    )

    expect(params.get('edge')).toBe('dependency_weight')
    expect(params.get('file')).toBe('/api/twin/scenarios/s1/exports/e1/raw')
    expect(params.get('area')).toBe('loc')
    expect(params.get('height')).toBe('coupling')
    expect(params.get('color')).toBe('complexity')
    expect(params.get('mode')).toBe('Single')
  })

  it('points at the proxied viewer', () => {
    expect(buildCityEmbedUrl('/raw')).toMatch(/^\/codecharta\/index\.html\?/)
  })
})
