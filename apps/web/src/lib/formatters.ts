/**
 * Format a duration in milliseconds into a human-readable string.
 *
 * - < 1 000 ms  -> "123ms"
 * - < 60 000 ms -> "4.2s"
 * - >= 60 000 ms -> "2m 13s"
 *
 * Returns `fallback` (default `"-"`) when `ms` is `null` or `undefined`.
 */
export function formatDuration(ms: number | null | undefined, fallback = '-'): string {
  if (ms === null || ms === undefined) return fallback
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
}
