import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation, useNavigate } from 'react-router'

import { api, apiData, apiErrorMessage } from '../../api/client'
import type { Collection, Source } from '../../api/types'
import { DEFAULT_COCKPIT_LAYER, DEFAULT_COCKPIT_VIEW, routeLocation } from '../../routing'

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

function SourceRow({ source, collection }: Readonly<{ source: Source; collection: Collection }>) {
  const queryClient = useQueryClient()
  const [mode, setMode] = useState<'display' | 'edit' | 'key'>('display')
  const [enabled, setEnabled] = useState(source.enabled)
  const [interval, setIntervalValue] = useState(source.schedule_interval_minutes)
  const [maxPages, setMaxPages] = useState(Number(source.config?.max_pages) || 100)
  const [privateKey, setPrivateKey] = useState('')
  const [syncRunId, setSyncRunId] = useState<string | null>(null)
  const completedRunRef = useRef<string | null>(null)

  const invalidateSources = () => queryClient.invalidateQueries({ queryKey: ['collections', collection.id, 'sources'] })
  const updateMutation = useMutation({
    mutationFn: () => apiData(api.PATCH('/api/sources/{source_id}', {
      params: { path: { source_id: source.id } },
      body: { enabled, schedule_interval_minutes: interval, ...(source.type === 'web' ? { max_pages: maxPages } : {}) },
    })),
    onSuccess: async () => { await invalidateSources(); setMode('display') },
  })
  const deleteMutation = useMutation({
    mutationFn: () => apiData(api.DELETE('/api/sources/{source_id}', { params: { path: { source_id: source.id } } })),
    onSuccess: invalidateSources,
  })
  const setKeyMutation = useMutation({
    mutationFn: () => apiData(api.PUT('/api/sources/{source_id}/deploy-key', {
      params: { path: { source_id: source.id } }, body: { private_key: privateKey },
    })),
    onSuccess: async () => { setPrivateKey(''); await invalidateSources() },
  })
  const deleteKeyMutation = useMutation({
    mutationFn: () => apiData(api.DELETE('/api/sources/{source_id}/deploy-key', { params: { path: { source_id: source.id } } })),
    onSuccess: invalidateSources,
  })
  const syncMutation = useMutation({
    mutationFn: () => apiData(api.POST('/api/sources/{source_id}/sync-now', { params: { path: { source_id: source.id } } })),
    onSuccess: (result) => { completedRunRef.current = null; setSyncRunId(result.sync_run_id) },
  })
  const syncStatusQuery = useQuery({
    queryKey: ['runs', source.id, 'sync-status', syncRunId],
    queryFn: ({ signal }) => apiData(api.GET('/api/runs', { params: { query: { source_id: source.id } }, signal })),
    enabled: Boolean(syncRunId),
    refetchInterval: (query) => {
      const run = query.state.data?.find((item) => item.id === syncRunId)
      return run && run.status !== 'running' && run.status !== 'scheduled' ? false : 2_000
    },
  })
  const activeRun = syncStatusQuery.data?.find((run) => run.id === syncRunId)
  const syncing = syncMutation.isPending || Boolean(syncRunId && (!activeRun || activeRun.status === 'running' || activeRun.status === 'scheduled'))

  useEffect(() => {
    if (!syncRunId || !activeRun || activeRun.status === 'running' || activeRun.status === 'scheduled' || completedRunRef.current === syncRunId) return
    completedRunRef.current = syncRunId
    void queryClient.invalidateQueries({ queryKey: ['runs', source.id] })
    void queryClient.invalidateQueries({ queryKey: ['collections', collection.id, 'sources'] })
  }, [activeRun, collection.id, queryClient, source.id, syncRunId])

  const mutationError = updateMutation.error ?? setKeyMutation.error ?? deleteKeyMutation.error ?? syncMutation.error

  if (mode === 'edit') {
    return <div className="source-row editing"><div className="source-edit-inline">
      <div className="edit-row"><span className={`source-type-badge ${source.type}`}>{source.type}</span><span className="source-url-static">{formatSourceUrl(source.url)}</span></div>
      <div className="edit-row"><label className="checkbox-inline"><input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enabled</label><label className="select-inline">Interval: <select value={interval} onChange={(event) => setIntervalValue(Number(event.target.value))}>{[15, 30, 60, 120, 360, 720, 1440].map((value) => <option key={value} value={value}>{value === 60 ? 'Hourly' : value === 1440 ? 'Daily' : `${value} min`}</option>)}</select></label>{source.type === 'web' && <label className="input-inline">Max pages: <input type="number" min={1} max={1000} value={maxPages} onChange={(event) => setMaxPages(Number(event.target.value))} /></label>}</div>
      <div className="edit-actions"><button className="save-btn" disabled={updateMutation.isPending} onClick={() => updateMutation.mutate()}>{updateMutation.isPending ? 'Saving...' : 'Save'}</button><button className="cancel-btn" onClick={() => setMode('display')}>Cancel</button></div>
      {updateMutation.isError && <p className="inline-error">{apiErrorMessage(updateMutation.error, 'Failed to update source')}</p>}
    </div></div>
  }

  if (mode === 'key') {
    return <div className="source-row editing"><div className="source-key-inline">
      <div className="edit-row"><span className={`source-type-badge ${source.type}`}>{source.type}</span><span className="source-url-static">{formatSourceUrl(source.url)}</span><button onClick={() => setMode('display')} className="close-inline">×</button></div>
      {source.deploy_key_fingerprint ? <div className="key-info"><span className="key-status has">Key configured</span><code className="fingerprint">{source.deploy_key_fingerprint}</code><button onClick={() => deleteKeyMutation.mutate()} className="remove-key-btn" disabled={deleteKeyMutation.isPending}>{deleteKeyMutation.isPending ? 'Removing...' : 'Remove Key'}</button></div> : <div className="key-form"><p className="key-hint">Paste SSH private key for private repo access:</p><textarea placeholder="-----BEGIN OPENSSH PRIVATE KEY-----" value={privateKey} onChange={(event) => setPrivateKey(event.target.value)} rows={4} /><button onClick={() => setKeyMutation.mutate()} className="save-key-btn" disabled={setKeyMutation.isPending || !privateKey.trim()}>{setKeyMutation.isPending ? 'Saving...' : 'Save Key'}</button></div>}
      {(setKeyMutation.isError || deleteKeyMutation.isError) && <p className="inline-error">{apiErrorMessage(mutationError, 'Deploy key operation failed')}</p>}
    </div></div>
  }

  return <div className={`source-row ${source.enabled ? '' : 'disabled'}`}>
    <span className={`source-type-badge ${source.type}`}>{source.type}</span>
    <a href={source.url} target="_blank" rel="noopener noreferrer" className="source-url">{formatSourceUrl(source.url)}</a>
    <span className="source-docs">{source.document_count} docs</span>
    <span className="source-last-sync">{source.last_run_at ? `Synced ${new Date(source.last_run_at).toLocaleDateString()}` : 'Never synced'}</span>
    {collection.is_owner && <div className="source-actions">
      {source.type === 'github' && <button onClick={() => setMode('key')} className={`key-btn ${source.deploy_key_fingerprint ? 'has-key' : ''}`} title={source.deploy_key_fingerprint ? 'Manage deploy key' : 'Add deploy key'}>{source.deploy_key_fingerprint ? '🔑' : '🔐'}</button>}
      <button onClick={() => syncMutation.mutate()} className={`sync-btn ${syncing ? 'syncing' : ''}`} disabled={syncing}>{syncing ? 'Syncing...' : 'Sync'}</button>
      <button onClick={() => setMode('edit')} className="edit-btn">Edit</button>
      <button onClick={() => deleteMutation.mutate()} className="delete-btn" disabled={deleteMutation.isPending}>×</button>
    </div>}
    {syncMutation.isError && <p className="inline-error">{apiErrorMessage(syncMutation.error, 'Failed to start sync')}</p>}
  </div>
}

function CollectionRow({ collection }: Readonly<{ collection: Collection }>) {
  const queryClient = useQueryClient()
  const location = useLocation()
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState(false)
  const [editing, setEditing] = useState(false)
  const [shareOpen, setShareOpen] = useState(false)
  const [name, setName] = useState(collection.name)
  const [visibility, setVisibility] = useState<'global' | 'private'>(collection.visibility === 'global' ? 'global' : 'private')
  const [shareLogin, setShareLogin] = useState('')
  const [sourceType, setSourceType] = useState<'github' | 'web'>('github')
  const [sourceUrl, setSourceUrl] = useState('')
  const [sourceError, setSourceError] = useState<string | null>(null)

  const invalidateCollection = () => queryClient.invalidateQueries({ queryKey: ['collections'] })
  const invalidateSources = () => queryClient.invalidateQueries({ queryKey: ['collections', collection.id, 'sources'] })
  const sourcesQuery = useQuery({
    queryKey: ['collections', collection.id, 'sources'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections/{collection_id}/sources', { params: { path: { collection_id: collection.id } }, signal })),
    enabled: expanded,
  })
  const membersQuery = useQuery({
    queryKey: ['collections', collection.id, 'members'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections/{collection_id}/members', { params: { path: { collection_id: collection.id } }, signal })),
    enabled: shareOpen,
  })
  const invitesQuery = useQuery({
    queryKey: ['collections', collection.id, 'invites'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections/{collection_id}/invites', { params: { path: { collection_id: collection.id } }, signal })),
    enabled: shareOpen && collection.is_owner,
  })
  const editMutation = useMutation({
    mutationFn: () => apiData(api.PATCH('/api/collections/{collection_id}', { params: { path: { collection_id: collection.id } }, body: { name, visibility } })),
    onSuccess: async () => { await invalidateCollection(); setEditing(false) },
  })
  const deleteMutation = useMutation({
    mutationFn: () => apiData(api.DELETE('/api/collections/{collection_id}', { params: { path: { collection_id: collection.id } } })),
    onSuccess: invalidateCollection,
  })
  const shareMutation = useMutation({
    mutationFn: () => apiData(api.POST('/api/collections/{collection_id}/share', { params: { path: { collection_id: collection.id } }, body: { github_login: shareLogin } })),
    onSuccess: async () => { setShareLogin(''); await Promise.all([membersQuery.refetch(), invitesQuery.refetch(), invalidateCollection()]) },
  })
  const unshareMutation = useMutation({
    mutationFn: (identifier: string) => apiData(api.DELETE('/api/collections/{collection_id}/share/{identifier}', { params: { path: { collection_id: collection.id, identifier } } })),
    onSuccess: async () => { await Promise.all([membersQuery.refetch(), invitesQuery.refetch(), invalidateCollection()]) },
  })
  const createSourceMutation = useMutation({
    mutationFn: () => apiData(api.POST('/api/collections/{collection_id}/sources', {
      params: { path: { collection_id: collection.id } },
      body: { type: sourceType, url: sourceUrl, enabled: true, schedule_interval_minutes: 60 },
    })),
    onSuccess: async () => { setSourceUrl(''); setSourceError(null); await invalidateSources() },
    onError: (error) => setSourceError(apiErrorMessage(error, 'Failed to create source')),
  })
  const sources = sourcesQuery.data ?? []
  const sourceCount = sources.length
  const docCount = sources.reduce((sum, source) => sum + source.document_count, 0)

  const handleDelete = () => {
    if (confirm(`Are you sure you want to delete "${collection.name}"? This will delete all sources, documents, and chunks. This action cannot be undone.`)) deleteMutation.mutate()
  }

  return <div className={`collection-row ${expanded ? 'expanded' : ''}`}>
    <div className="collection-header-row" role="button" tabIndex={0} aria-label={`Toggle collection ${collection.name}`} onClick={(event) => { if (!(event.target as HTMLElement).closest('form, [role="toolbar"]')) setExpanded((value) => !value) }} onKeyDown={(event) => { if (event.key === 'Enter') setExpanded((value) => !value) }}>
      <button className="expand-toggle" aria-label={expanded ? 'Collapse' : 'Expand'}>{expanded ? '▼' : '▶'}</button>
      <div className="collection-info">{editing ? <form onSubmit={(event) => { event.preventDefault(); editMutation.mutate() }} className="edit-collection-form"><input type="text" value={name} onChange={(event) => setName(event.target.value)} className="edit-name-input" autoFocus /><select value={visibility} onChange={(event) => setVisibility(event.target.value as 'global' | 'private')} className="edit-visibility-select"><option value="private">Private</option><option value="global">Global</option></select><button type="submit" disabled={editMutation.isPending} className="save-btn">{editMutation.isPending ? '...' : 'Save'}</button><button type="button" onClick={() => setEditing(false)} className="cancel-btn">Cancel</button></form> : <><span className="collection-name">{collection.name}</span><span className={`visibility-badge ${collection.visibility}`}>{collection.visibility === 'private' ? '🔒' : '🌐'} {collection.visibility}</span></>}</div>
      <div className="collection-stats"><span className="stat">{sourceCount === 0 ? 'No sources' : `${sourceCount} source${sourceCount === 1 ? '' : 's'}`}</span><span className="stat">{docCount > 0 ? `${docCount} docs` : ''}</span></div>
      <div className="collection-actions-inline" role="toolbar" onClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
        <button className="action-btn cockpit-btn" onClick={() => navigate(routeLocation('cockpit', location.search, { collectionId: collection.id, view: DEFAULT_COCKPIT_VIEW, layer: DEFAULT_COCKPIT_LAYER }))}>Open in Cockpit</button>
        {collection.is_owner ? <><button className="action-btn edit-btn" onClick={() => setEditing(true)} title="Edit collection">⚙️</button><button className="action-btn share-btn" onClick={() => setShareOpen(true)} title="Share collection">🔗</button><button className="action-btn delete-btn" onClick={handleDelete} title="Delete collection">🗑️</button></> : <span className="owner-label">by @{collection.owner_github_login}</span>}
      </div>
    </div>

    {shareOpen && <dialog open className="share-popover" aria-label={`Share ${collection.name}`}><div className="popover-header"><h4>Share "{collection.name}"</h4><button className="close-btn" onClick={() => setShareOpen(false)}>×</button></div><div className="popover-content">
      {(membersQuery.data ?? []).length > 0 && <div className="members-mini"><span className="label">Members:</span>{membersQuery.data?.map((member) => <span key={member.user_id} className="member-chip">@{member.github_login}{member.is_owner && <span className="owner-tag">owner</span>}{!member.is_owner && <button className="remove-chip" onClick={() => unshareMutation.mutate(member.user_id)}>×</button>}</span>)}</div>}
      {(invitesQuery.data ?? []).length > 0 && <div className="invites-mini"><span className="label">Pending:</span>{invitesQuery.data?.map((invite) => <span key={invite.github_login} className="member-chip pending">@{invite.github_login}<button className="remove-chip" onClick={() => unshareMutation.mutate(invite.github_login)}>×</button></span>)}</div>}
      <form onSubmit={(event) => { event.preventDefault(); if (shareLogin.trim()) shareMutation.mutate() }} className="share-form-mini"><input type="text" placeholder="GitHub username" value={shareLogin} onChange={(event) => setShareLogin(event.target.value)} className="share-input-mini" /><button type="submit" className="add-btn">+ Add</button></form>
      {shareMutation.isError && <p className="share-error-mini">{apiErrorMessage(shareMutation.error, 'Failed to share collection')}</p>}
    </div></dialog>}

    {expanded && <div className="collection-sources-section">
      {sourcesQuery.isLoading ? <p className="loading-text">Loading sources...</p> : <>
        {collection.is_owner && <div className="add-source-inline"><form onSubmit={(event) => { event.preventDefault(); if (sourceUrl.trim()) createSourceMutation.mutate() }} className="source-form-inline"><select value={sourceType} onChange={(event) => setSourceType(event.target.value as 'github' | 'web')} className="source-type-select"><option value="github">GitHub</option><option value="web">Web</option></select><input type="text" placeholder={sourceType === 'github' ? 'https://github.com/owner/repo' : 'https://docs.example.com/'} value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} className="source-url-input" /><button type="submit" className="add-source-btn" disabled={!sourceUrl.trim() || createSourceMutation.isPending}>Add Source</button></form>{sourceError && <p className="source-error-inline">{sourceError}</p>}</div>}
        {sources.length === 0 ? <p className="no-sources-text">No sources yet. Add a GitHub repo or documentation URL above.</p> : <div className="sources-list">{sources.map((source) => <SourceRow key={source.id} source={source} collection={collection} />)}</div>}
      </>}
    </div>}
  </div>
}

export default function CollectionsPage() {
  const queryClient = useQueryClient()
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [slug, setSlug] = useState('')
  const [visibility, setVisibility] = useState<'global' | 'private'>('private')
  const collectionsQuery = useQuery({
    queryKey: ['collections'],
    queryFn: ({ signal }) => apiData(api.GET('/api/collections', { signal })),
  })
  const createMutation = useMutation({
    mutationFn: () => apiData(api.POST('/api/collections', { body: { name, slug, visibility } })),
    onSuccess: async () => { setName(''); setSlug(''); setVisibility('private'); setCreating(false); await queryClient.invalidateQueries({ queryKey: ['collections'] }) },
  })
  const collections = collectionsQuery.data ?? []

  return <section className="card collections-overview">
    <div className="collections-header"><h2>Collections</h2><button className="create-button-inline" onClick={() => setCreating(true)}>+ New Collection</button></div>
    {creating && <div className="create-collection-inline"><form onSubmit={(event) => { event.preventDefault(); if (name.trim() && slug.trim()) createMutation.mutate() }} className="collection-form-inline"><div className="form-row"><input type="text" placeholder="Collection name" value={name} onChange={(event) => setName(event.target.value)} className="collection-input" autoFocus /><input type="text" placeholder="slug" value={slug} onChange={(event) => setSlug(event.target.value.toLowerCase().replaceAll(/[^a-z0-9-]/g, '-'))} className="collection-input slug-input" /><select value={visibility} onChange={(event) => setVisibility(event.target.value as 'global' | 'private')} className="visibility-select"><option value="private">Private</option><option value="global">Global</option></select><button type="submit" className="create-button" disabled={createMutation.isPending}>Create</button><button type="button" className="cancel-button" onClick={() => setCreating(false)}>Cancel</button></div></form>{createMutation.isError && <p className="inline-error">{apiErrorMessage(createMutation.error, 'Failed to create collection')}</p>}</div>}
    {collectionsQuery.isLoading && <p className="loading-text">Loading collections...</p>}
    {!collectionsQuery.isLoading && collections.length === 0 && <div className="empty-state"><p>No collections yet</p><p className="note">Collections organize your documentation and code sources.</p><button className="create-button" onClick={() => setCreating(true)}>Create Your First Collection</button></div>}
    {collections.length > 0 && <div className="collections-list">{collections.map((collection) => <CollectionRow key={collection.id} collection={collection} />)}</div>}
  </section>
}
