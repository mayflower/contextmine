## Architecture Cockpit remediation plan

### Change A — graph paging
- Patch `packages/core/contextmine_core/twin/service.py` to replace blunt sorted-node paging with deterministic edge-aware windows and to expose slice metadata.
- Patch `apps/api/app/routes/twin.py` to expose the same slice metadata and warnings for topology, deep dive, and paged specialty graphs.

### Change B — architecture grouping
- Patch `packages/core/contextmine_core/twin/grouping.py` to make path fallback conservative and reject generic/support roots.
- Patch `packages/core/contextmine_core/twin/projections.py` to retain explicit-vs-heuristic provenance on projected architecture nodes.

### Change C — cockpit provenance and empty states
- Patch `apps/web/src/cockpit/types.ts` and `apps/web/src/cockpit/hooks/useCockpitData.ts` to carry slice/provenance/warning metadata end to end.
- Patch `apps/web/src/cockpit/views/TopologyView.tsx`, `DeepDiveView.tsx`, `TestMatrixView.tsx`, `OverviewView.tsx`, `EvolutionView.tsx`, and `C4DiffView.tsx` to render missing-data, heuristic, degraded, and truncation reasons more explicitly.
- Update backend/frontend tests for the changed payloads and warnings.
