import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'

import { api, apiData } from '../../api/client'
import type { Collection, Source } from '../../api/types'
import { formatDuration } from '../../lib/formatters'

function formatSourceUrl(url: string): string {
  if (url.startsWith('https://github.com/')) return url.replace('https://github.com/', '').split('/').slice(0, 2).join('/')
  try {
    const parsed = new URL(url)
    const path = parsed.pathname.replace(/\/$/, '')
    return path && path !== '/' ? parsed.hostname + path : parsed.hostname
  } catch {
    return url
  }
}

export default function RunsPage() {
  const [runsCollection, setRunsCollection] = useState<Collection | null>(null)
  const [selectedRunSource, setSelectedRunSource] = useState<Source | null>(null)
  const collectionsQuery = useQuery({
    queryKey: ['collections'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections', { signal })),
  })
  const sourcesQuery = useQuery({
    queryKey: ['collections', runsCollection?.id, 'sources'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections/{collection_id}/sources', {
      params: { path: { collection_id: runsCollection!.id } }, signal,
    })),
    enabled: Boolean(runsCollection),
  })
  const runsQuery = useQuery({
    queryKey: ['runs', selectedRunSource?.id],
    queryFn: ({ signal }) => apiData(api.GET('/api/runs', {
      params: { query: { source_id: selectedRunSource!.id } }, signal,
    })),
    enabled: Boolean(selectedRunSource),
    refetchInterval: 5_000,
  })
  const prefectQuery = useQuery({
    queryKey: ['prefect', 'flow-runs'],
    queryFn: ({ signal }) => apiData(api.GET('/api/prefect/flow-runs', { signal })),
    refetchInterval: 5_000,
  })
  const collections = collectionsQuery.data ?? []
  const sources = sourcesQuery.data ?? []
  const runs = runsQuery.data ?? []
  const prefect = prefectQuery.data

  return (
    <>
      <section className="card active-runs-section">
        <h2>Active Runs</h2>
        {prefect?.error && <p className="note error">Prefect connection error: {prefect.error}</p>}
        {prefect?.active && prefect.active.length > 0 ? (
          <div className="active-runs-grid">
            {prefect.active.map((run) => (
              <div key={run.id} className={`active-run-card state-${run.state_type.toLowerCase()}`}>
                <div className="run-header"><span className={`state-badge ${run.state_type.toLowerCase()}`}>{run.state_name}</span><span className="run-name">{run.name}</span></div>
                {run.parameters.source_url && <div className="run-source">{formatSourceUrl(run.parameters.source_url)}</div>}
                {run.progress && <div className="run-progress"><div className="progress-bar"><div className="progress-fill" style={{ width: `${run.progress.percent}%` }} /></div><div className="progress-stats"><span>{run.progress.completed}/{run.progress.total} tasks</span>{run.progress.current_task && <span className="current-task">{run.progress.current_task}</span>}</div></div>}
                {run.start_time && <div className="run-started">Started {new Date(run.start_time).toLocaleTimeString()}</div>}
              </div>
            ))}
          </div>
        ) : !prefect?.error && <p className="note">No active runs. Runs will appear here when syncs are in progress.</p>}
      </section>

      {prefect?.recent && prefect.recent.length > 0 && (
        <section className="card">
          <h2>Recent Runs (Prefect)</h2>
          <table className="runs-table"><thead><tr><th>Name</th><th>Source</th><th>Status</th><th>Started</th><th>Duration</th></tr></thead><tbody>
            {prefect.recent.slice(0, 10).map((run) => {
              const start = run.start_time ? new Date(run.start_time) : null
              const end = run.end_time ? new Date(run.end_time) : null
              return <tr key={run.id} className={`state-${run.state_type.toLowerCase()}`}><td>{run.name}</td><td className="source-cell">{run.parameters.source_url ? formatSourceUrl(run.parameters.source_url) : '-'}</td><td><span className={`state-badge ${run.state_type.toLowerCase()}`}>{run.state_name}</span></td><td>{start ? start.toLocaleString() : '-'}</td><td>{formatDuration(start && end ? end.getTime() - start.getTime() : null)}</td></tr>
            })}
          </tbody></table>
        </section>
      )}

      <section className="card">
        <h2>Run History</h2>
        <div className="run-filters">
          <div className="filter-group"><label htmlFor="runs-collection-filter">Collection:</label><select id="runs-collection-filter" value={runsCollection?.id || ''} onChange={(event) => { const collection = collections.find((item) => item.id === event.target.value) ?? null; setRunsCollection(collection); setSelectedRunSource(null) }} className="filter-select"><option value="">Select collection...</option>{collections.map((collection) => <option key={collection.id} value={collection.id}>{collection.name}</option>)}</select></div>
          {runsCollection && <div className="filter-group"><label htmlFor="runs-source-filter">Source:</label><select id="runs-source-filter" value={selectedRunSource?.id || ''} onChange={(event) => setSelectedRunSource(sources.find((source) => source.id === event.target.value) ?? null)} className="filter-select"><option value="">Select source...</option>{sources.map((source) => <option key={source.id} value={source.id}>[{source.type}] {formatSourceUrl(source.url)}</option>)}</select></div>}
        </div>
        {!runsCollection && <p className="note">Select a collection and source to view run history.</p>}
        {runsCollection && !selectedRunSource && sources.length > 0 && <p className="note">Select a source to view its run history.</p>}
        {runsCollection && !sourcesQuery.isLoading && sources.length === 0 && <p className="note">No sources in this collection.</p>}
      </section>

      {selectedRunSource && (
        <section className="card">
          <h2>Runs: {formatSourceUrl(selectedRunSource.url)}</h2>
          {runsQuery.isLoading && <p>Loading runs...</p>}
          {!runsQuery.isLoading && runs.length === 0 && <p className="note">No runs yet for this source.</p>}
          {runs.length > 0 && <table className="runs-table"><thead><tr><th>Started</th><th>Finished</th><th>Duration</th><th>Status</th><th>Stats</th><th>Error</th></tr></thead><tbody>
            {runs.map((run) => {
              const start = new Date(run.started_at)
              const end = run.finished_at ? new Date(run.finished_at) : null
              const statusClass = run.status === 'success' ? 'ok' : run.status === 'failed' ? 'error' : 'loading'
              return <tr key={run.id} className={`run-${run.status}`}><td>{start.toLocaleString()}</td><td>{end ? end.toLocaleString() : '-'}</td><td>{formatDuration(end ? end.getTime() - start.getTime() : null)}</td><td><span className={`status ${statusClass}`}>{run.status}</span></td><td className="stats-cell">{run.stats ? <div className="stats-grid">
                {run.stats.docs_created !== undefined && <span className="stat-item"><span className="stat-value created">+{run.stats.docs_created}</span><span className="stat-label">docs</span></span>}
                {(run.stats.docs_updated ?? 0) > 0 && <span className="stat-item"><span className="stat-value updated">~{run.stats.docs_updated}</span><span className="stat-label">updated</span></span>}
                {(run.stats.docs_deleted ?? 0) > 0 && <span className="stat-item"><span className="stat-value deleted">-{run.stats.docs_deleted}</span><span className="stat-label">deleted</span></span>}
                {run.stats.chunks_embedded !== undefined && <span className="stat-item"><span className="stat-value">{run.stats.chunks_embedded}</span><span className="stat-label">chunks</span></span>}
                {run.stats.files_scanned !== undefined && <span className="stat-item" title={`Indexed: ${run.stats.files_indexed}, Skipped: ${run.stats.files_skipped}`}><span className="stat-value">{run.stats.files_scanned}</span><span className="stat-label">files</span></span>}
                {run.stats.pages_crawled !== undefined && <span className="stat-item"><span className="stat-value">{run.stats.pages_crawled}</span><span className="stat-label">pages</span></span>}
              </div> : '-'}</td><td className="error-cell">{run.error ? <span className="error-text" title={run.error}>{run.error.length > 50 ? `${run.error.substring(0, 50)}...` : run.error}</span> : '-'}</td></tr>
            })}
          </tbody></table>}
        </section>
      )}
    </>
  )
}
