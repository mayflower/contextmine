import mermaidLib from 'mermaid'

/**
 * Render a Mermaid diagram into a container element.
 *
 * If the content is empty, an empty `<pre>` placeholder is shown.
 * If the rendered SVG contains a parse error the raw source is displayed
 * as a `<pre>` fallback so the user can still inspect the diagram text.
 */
export async function renderMermaid(container: HTMLElement, id: string, content: string): Promise<void> {
  if (!content.trim()) {
    const pre = document.createElement('pre')
    container.replaceChildren(pre)
    return
  }
  const rendered = await mermaidLib.render(id, content)
  const parsed = new DOMParser().parseFromString(rendered.svg, 'image/svg+xml')
  if (parsed.querySelector('parsererror')) {
    const pre = document.createElement('pre')
    pre.textContent = content
    container.replaceChildren(pre)
    return
  }
  container.replaceChildren(document.importNode(parsed.documentElement, true))
}

/**
 * Insert a pre-rendered SVG string into a container element.
 *
 * Used when the caller already has the SVG output (e.g. from `mermaid.render`)
 * and only needs the DOM insertion + parse-error fallback.
 */
export function renderMermaidSvg(container: HTMLElement, svg: string, fallbackText: string): void {
  const parsed = new DOMParser().parseFromString(svg, 'image/svg+xml')
  if (parsed.querySelector('parsererror')) {
    const pre = document.createElement('pre')
    pre.textContent = fallbackText
    container.replaceChildren(pre)
    return
  }
  const svgElement = parsed.documentElement
  container.replaceChildren(document.importNode(svgElement, true))
}
