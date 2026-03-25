import { describe, expect, it, vi, beforeEach } from 'vitest'

// Mock the mermaid library before importing the module under test
vi.mock('mermaid', () => ({
  default: {
    render: vi.fn(),
  },
}))

import mermaidLib from 'mermaid'

import { renderMermaid, renderMermaidSvg } from './mermaidUtils'

describe('renderMermaid', () => {
  let container: HTMLDivElement

  beforeEach(() => {
    container = document.createElement('div')
    vi.mocked(mermaidLib.render).mockReset()
  })

  it('shows empty pre element for blank content', async () => {
    await renderMermaid(container, 'test-id', '   ')
    expect(container.children).toHaveLength(1)
    expect(container.children[0].tagName).toBe('PRE')
    expect(container.children[0].textContent).toBe('')
  })

  it('shows empty pre element for empty string', async () => {
    await renderMermaid(container, 'test-id', '')
    expect(container.children).toHaveLength(1)
    expect(container.children[0].tagName).toBe('PRE')
  })

  it('renders valid SVG into container', async () => {
    vi.mocked(mermaidLib.render).mockResolvedValue({
      svg: '<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100"/></svg>',
      bindFunctions: undefined,
    })

    await renderMermaid(container, 'test-id', 'graph TD\n  A-->B')

    expect(mermaidLib.render).toHaveBeenCalledWith('test-id', 'graph TD\n  A-->B')
    expect(container.children).toHaveLength(1)
    expect(container.children[0].tagName.toLowerCase()).toBe('svg')
  })

  it('falls back to pre element when SVG has parse error', async () => {
    vi.mocked(mermaidLib.render).mockResolvedValue({
      svg: '<html><body><parsererror>bad</parsererror></body></html>',
      bindFunctions: undefined,
    })

    await renderMermaid(container, 'test-id', 'invalid mermaid')

    expect(container.children).toHaveLength(1)
    expect(container.children[0].tagName).toBe('PRE')
    expect(container.children[0].textContent).toBe('invalid mermaid')
  })
})

describe('renderMermaidSvg', () => {
  let container: HTMLDivElement

  beforeEach(() => {
    container = document.createElement('div')
  })

  it('inserts valid SVG into container', () => {
    const svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="10" r="5"/></svg>'
    renderMermaidSvg(container, svg, 'fallback text')

    expect(container.children).toHaveLength(1)
    expect(container.children[0].tagName.toLowerCase()).toBe('svg')
  })

  it('falls back to pre element for invalid SVG', () => {
    const badSvg = '<html><body><parsererror>parse failed</parsererror></body></html>'
    renderMermaidSvg(container, badSvg, 'original mermaid code')

    expect(container.children).toHaveLength(1)
    expect(container.children[0].tagName).toBe('PRE')
    expect(container.children[0].textContent).toBe('original mermaid code')
  })
})
