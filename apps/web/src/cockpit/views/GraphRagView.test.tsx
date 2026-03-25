/**
 * Tests for GraphRagView - we import and verify the module loads.
 * The component depends on ReactFlow, mermaid, and many other components.
 */
import { describe, expect, it, vi } from 'vitest'

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

describe('GraphRagView module', () => {
  it('can be imported without errors', async () => {
    const module = await import('./GraphRagView')
    expect(module.default).toBeDefined()
    expect(typeof module.default).toBe('function')
  })
})
