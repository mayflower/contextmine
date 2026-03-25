import type { ReactNode } from 'react'

import type { CockpitLoadState } from '../types'

interface ViewShellProps {
  state: CockpitLoadState
  error: string | null
  panelId: string
  title: string
  hasData: boolean
  onRetry?: () => void
  skeletonCount?: number
  skeletonTall?: boolean
  skeleton?: ReactNode
  children: ReactNode
}

export default function ViewShell({
  state,
  error,
  panelId,
  title,
  hasData,
  onRetry,
  skeletonCount = 3,
  skeletonTall = false,
  skeleton,
  children,
}: Readonly<ViewShellProps>) {
  if (state === 'loading' && !hasData) {
    return (
      <div className="cockpit2-skeleton-grid" id={panelId} role="tabpanel">
        {skeleton ??
          Array.from({ length: skeletonCount }, (_, i) => (
            <div
              key={i}
              className={`cockpit2-skeleton-card${skeletonTall ? ' tall' : ''}`}
            />
          ))}
      </div>
    )
  }

  if (state === 'error' && !hasData) {
    return (
      <section className="cockpit2-alert error" id={panelId} role="tabpanel">
        <h3>{title} request failed</h3>
        <p>{error}</p>
        {onRetry ? (
          <button type="button" onClick={onRetry}>Retry</button>
        ) : null}
      </section>
    )
  }

  return (
    <>
      {error && hasData ? (
        <div className="cockpit2-alert error inline">
          <p>{error}</p>
          {onRetry ? (
            <button type="button" onClick={onRetry}>Retry</button>
          ) : null}
        </div>
      ) : null}
      {children}
    </>
  )
}
