/**
 * Query string for the embedded CodeCharta viewer.
 *
 * `edge` matters: without an edge metric the viewer never sets
 * isEdgeMetricVisible, so the dependency edges carried in the export are
 * never drawn.
 */
export function buildCityEmbedUrl(rawPath: string): string {
  const embed = new URLSearchParams({
    file: rawPath,
    area: 'loc',
    height: 'coupling',
    color: 'complexity',
    edge: 'dependency_weight',
    mode: 'Single',
  })
  return `/codecharta/index.html?${embed.toString()}`
}
