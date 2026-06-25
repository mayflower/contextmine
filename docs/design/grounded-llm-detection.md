# Grounded LLM Detection

Replacing keyword/string-matching "analysis" with grounded, constrained, verified
LLM detection across the analyzer / knowledge / twin / architecture subsystems.

## Motivation

An audit of the analysis subsystem found a recurring "vapour" pattern: code whose
names and docstrings advertise semantic understanding (domain extraction, business
rules, architecture recovery, taint analysis, C4 decomposition) while the
implementation is hardcoded keyword lists, substring/regex matching, path-prefix
splitting, and hand-picked confidence literals. In several modules a genuine LLM
path exists in the same file but is off by default, opt-in, discarded, or
structurally broken — while the heuristic output is what ships.

Representative findings:

- `twin/grouping.py` derives "bounded contexts / containers / components" from
  `path.split("/")[1:3]`.
- `twin/ops.py` taint sources/sinks are substring matches on function names
  (`"request"` -> source, `"execute"` -> sink), the default engine, with no
  `approximation` caveat.
- `architecture/arc42.py` fabricates "Architecture Decisions / Solution Strategy /
  Security" sections from node counts and tag presence; this is the default
  production arc42 document.
- `knowledge/communities.py` titles communities by concatenating member names
  while a real LLM title is computed in `summaries.py` and then discarded.
- `architecture/recovery_llm.py` implements correct constrained adjudication but
  calls `.adjudicate()` / `__call__()` on the provider, which `LLMProvider` does
  not expose — so it can never run in production.
- Confidence values across `tests.py`, `surface.py`, arc42 are hardcoded literals
  (`0.71`, `0.96`) rounded to 4 decimals, implying calibration that does not exist.

## Non-goals / what stays deterministic

LLMs do **not** replace work that string/AST matching does correctly:

- Structured-format parsing: OpenAPI / GraphQL / proto / Alembic / SQL DDL.
- The AST symbol/edge graph (`knowledge/builder.py`, `treesitter/`).
- Leiden community detection (`knowledge/communities.py`).
- Git-history metrics, ownership, coupling, CRAP (`twin/evolution.py`).
- SCIP -> LSP -> Joern call resolution (`traceability.py`).

The LLM replaces only the **semantic judgments the keyword heuristics faked**:
deciding what a thing *means*, which bucket it belongs to, whether two things are
related, and what an architecture *is*.

## Core principle

> Deterministic code generates **candidates + evidence** -> the LLM
> **judges / labels / decomposes** over those candidates -> code **validates** the
> LLM output against the real graph -> confidence is **earned**, not literal.

Both keyword heuristics and naive "ask the LLM to imagine the architecture" are
guessing. The fix is grounded, constrained generation: the model only ever selects
or labels over real candidates and real evidence, and its output is validated back
against the graph before it is trusted.

### The five anti-guessing rules

Every detector must obey all five:

1. **Closed-world.** The LLM may only choose from candidate ids / files / symbols we
   supply. Any id it returns that is not in the candidate set is rejected, not
   stored. (This is the existing design in `architecture/recovery_llm.py`.)
2. **Evidence-required.** Every finding must cite at least one real evidence id
   (file:line, node id, edge id) that we re-verify exists. Uncited findings are
   dropped.
3. **Abstention is valid.** Detectors must be able to return "none / unknown". The
   keyword heuristics' core failure is that they never abstain — they always emit
   something. Abstention beats fabrication.
4. **Confidence is derived.** From self-consistency (N samples agree) or a verifier
   pass — never a hardcoded float.
5. **Idempotent and cheap.** Temperature 0, content-hash skip, triage-gated so the
   LLM runs only on candidates worth the call. Same input -> same output.

## Existing assets to reuse (do not reinvent)

| Asset | Role in this design |
| --- | --- |
| `research/llm/provider.py` `LLMProvider.generate_structured[T]` | The structured-output workhorse (Pydantic-validated, temp 0, retry). |
| `analyzer/extractors/triage.py` | The gold "propose -> validate against input set -> fall back" pattern. `GroundedJudge` generalizes it. |
| `architecture/recovery_llm.py` | Constrained-adjudication packets + validation (closed-world + evidence-id checks). Resurrected by an adapter. |
| `knowledge/extraction.py` | Grounded entity/relationship extraction + embedding resolution + suspicious-merge audit. Model for domain extraction. |
| `knowledge/summaries.py` | Idempotent structured LLM with content-hash skip. Model for caching. |
| `research/verification/` | Verification status + adversarial-verify scaffolding. Model for the verifier pass / derived confidence. |
| `architecture/agent_sdk.py` + MCP graph tools | Agentic traversal (graph_neighborhood, find_symbol, references) for deep recovery tasks. |

## Component: `GroundedJudge`

Module: `packages/core/contextmine_core/grounding/`

A reusable wrapper around `LLMProvider.generate_structured` that enforces the five
rules so individual detectors do not re-implement prompt plumbing or validation.

```python
class Candidate(BaseModel):
    id: str                      # stable id the LLM must reference
    label: str                   # human-readable
    payload: dict[str, Any]      # features shown to the model

class Evidence(BaseModel):
    id: str                      # "ev-N"
    ref: str                     # file:line / node id / edge id
    snippet: str

class Finding(BaseModel):        # base class detectors extend
    candidate_ids: list[str]     # validated subset of supplied candidates
    evidence_ids: list[str]      # validated subset of supplied evidence
    rationale: str

async def judge(
    *, provider, system, task, candidates, evidence, output_schema,
    allow_abstain=True, samples=1,
) -> JudgeResult[T]: ...
```

`judge`:

1. Renders candidates + evidence into the prompt (closed-world framing, explicit
   "return [] / abstain if unsupported").
2. Calls `generate_structured` (temp 0; `samples>1` runs at low temperature for
   self-consistency).
3. Drops any `candidate_ids` / `evidence_ids` not in the supplied sets.
4. Drops findings with zero surviving evidence ids.
5. Derives confidence: with `samples>1`, agreement fraction across runs; otherwise
   an optional adversarial verifier pass (a second judge prompted to refute).
6. Returns surviving findings + derived confidence + the raw model output for audit.

### Derived confidence

`grounding/confidence.py`:

- `self_consistency(judge, n)` — run n times, confidence = fraction of runs that
  produce the finding (keyed by candidate+evidence signature).
- `adversarial_verify(provider, finding, evidence)` — a refute-prompted judge;
  survives -> keep, refuted by majority -> drop or downgrade.

No detector writes a literal confidence again.

## Component: recovery adjudicator adapter

`architecture/recovery_llm.py` already builds constrained packets and validates
output. The only break is the interface: `recovery.py` calls
`adjudicator.adjudicate(packet)` (or `adjudicator(packet)`), but the real
`LLMProvider` exposes only `generate_text` / `generate_structured`.

Fix: an adapter implementing `RecoveryAdjudicator.adjudicate(packet)` by calling
`generate_structured` with an `AdjudicationOutput` schema, returning the dict the
existing `validate_adjudication` / `apply_adjudication` already expect. This
resurrects ~379 lines of correct anti-hallucination machinery and lets the
constrained-adjudication path become the primary architecture decomposition
mechanism, not just an edge-case tie-breaker.

## Cost model

Two-tier on every detector (generalizes `triage.py`):

1. Cheap deterministic candidate generation (AST / graph).
2. Triage: "is this candidate worth an LLM call?"
3. Judge only survivors.
4. Cache by content hash; only re-run on changed files / subgraphs (piggyback on
   the incremental graph builder).

Small model for labeling; agentic / large model only for recovery.

## Phased rollout

Ordered cheapest-high-impact -> deepest. Each detector ships behind a flag in
**shadow mode** (runs alongside the heuristic, diffs output) and is promoted to
default only after clearing an eval bar.

### Foundation
- `GroundedJudge` + derived confidence.
- Recovery adjudicator adapter; wire as default when an LLM provider is configured.

### Phase A — stop shipping the lies
- A1 community title: write `summary.title` into `KnowledgeCommunity.title`.
- A2 taint: tag substring-engine results `approximation: true`; stop silent default.
- A3 arc42: stop count-derived "decisions/strategy/security"; only evidence-backed
  sections, else "none found"; grounded generator default, heuristic fallback.
- A4 jobs/schema/mcp_server docstrings: wire the LLM path or correct the claim.

### Phase B — the domain/architecture core (agentic)
- B1 bounded-context decomposition: replace `twin/grouping.py` path-split with a
  grounded detector (signals: symbols, edges, co-change, naming, shared tables);
  path prefix is one weak feature; assignments cited + confidence-scored.
- B2 architecture recovery: promote constrained adjudication to primary clustering.
- B3 C4 layer inference: ground in node evidence, or rename the constant lookup.

### Phase C — relationship & confidence truthfulness
- C1 `CO_OCCURS`: distinct provenance, not `SEMANTIC_RELATIONSHIP`.
- C2 kill fake confidence literals; derive or drop.
- C3 UI endpoint/route/flow: AST candidate generation + grounded verifier.

### Phase D — hardening
- Eval harness (golden fixtures + shadow-mode diff + adversarial verifier),
  cost/latency dashboards, rollout flags, stale-doc fixes.

## Evaluation (what makes this "not guessing")

- Golden fixtures: a few repos with human-labeled expected domains / rules / taint.
- Shadow mode: new detector runs alongside the heuristic; precision/recall vs golden
  is logged; promotion to default requires clearing a bar.
- Grounding invariants asserted in tests: every emitted finding's evidence ids
  resolve to real graph evidence; no candidate id outside the supplied set.
- Abstention is tested: a detector given unsupported input returns empty, not
  fabricated output.

## Acceptance criteria per detector

1. No hardcoded confidence literals; confidence is derived or absent.
2. Every finding cites resolvable evidence.
3. The detector can and does abstain on unsupported input (covered by a test).
4. Output validated closed-world against the graph (covered by a test).
5. Idempotent: same input -> same output (content-hash cached).
6. Ships behind a flag in shadow mode until it clears the eval bar.
