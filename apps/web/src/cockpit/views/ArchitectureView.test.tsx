/**
 * Tests for ArchitectureView module import.
 * The component uses mermaid for ERD rendering.
 */
import { describe, expect, it, vi } from 'vitest'

vi.mock('mermaid', () => ({
  default: {
    initialize: vi.fn(),
    render: vi.fn().mockResolvedValue({ svg: '<svg></svg>', bindFunctions: undefined }),
  },
}))

describe('ArchitectureView module', () => {
  it('can be imported without errors', async () => {
    const module = await import('./ArchitectureView')
    expect(module.default).toBeDefined()
    expect(typeof module.default).toBe('function')
  })
})
