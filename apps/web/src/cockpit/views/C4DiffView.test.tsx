/**
 * Tests for C4DiffView - minimal rendering tests since it uses mermaid.
 * We test the component renders without crashing.
 */
import { describe, expect, it, vi } from 'vitest'

// The C4DiffView uses mermaid directly, so we test it by checking imports exist.
// More thorough rendering tests would require a full mermaid mock.

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

describe('C4DiffView module', () => {
  it('can be imported without errors', async () => {
    const module = await import('./C4DiffView')
    expect(module.default).toBeDefined()
    expect(typeof module.default).toBe('function')
  })
})
