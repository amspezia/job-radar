# Synthetic Evaluation Dataset Design

**Goal**: eliminate pool bias and enable multi-query, multi-profile evaluation
without relying on the retrieval system itself to define the label space.

Two complementary approaches are described. Phase A (synthetic personas + injected
jobs) gives ground-truth labels by construction. Phase B (InPars-style query
generation) gives unbiased (query, job) positives from the live corpus. Both are
fully local — Ollama generates all synthetic text.

---

## Why This Matters

Current qrels come from a single pool drawn by the system being evaluated. Any
job the system doesn't find never gets labeled, so Recall is artificially high and
nDCG comparisons between configs are unstable (changing pool depth changes labels).
BPref (now in `eval/metrics.py`) mitigates the score distortion, but the underlying
labeling gap remains. Synthetic ground truth fixes it at the source.

References:
- InPars Toolkit — arXiv:2307.04601
- PJB Benchmark — arXiv:2603.17386 (domain heterogeneity dominates algorithm gains)
- Buckley & Voorhees 2004 (BPref, pool bias)

---

## Phase A — Synthetic Personas + Injected Jobs

### A1. Persona Definitions

Five developer personas stored as JSON fixtures in `eval/personas/`. Each is a
pure data file — no PII, no real CV. Committed to git.

| slug | seniority | core stack | domain | location |
|---|---|---|---|---|
| `alice-rust-senior` | senior | Rust, C++, WASM | systems / infra | global remote |
| `bob-python-ml-mid` | mid | Python, PyTorch, FastAPI | ML / NLP | Brazil / LATAM |
| `carol-ts-fullstack-mid` | mid | TypeScript, React, Node | product / SaaS | EU remote |
| `dave-go-platform-staff` | staff | Go, Kubernetes, gRPC | platform / distributed | global remote |
| `eve-data-junior` | junior | Python, SQL, dbt | data engineering | remote-friendly |

Schema for each file (`eval/personas/{slug}.json`):

```json
{
  "slug": "alice-rust-senior",
  "seniority": "senior",
  "target_titles": ["Senior Software Engineer", "Staff Engineer", "Systems Engineer"],
  "tech_stack": ["Rust", "C++", "WASM", "Linux", "async"],
  "domains": ["systems programming", "embedded", "performance engineering"],
  "location_rules": { "allowed_keywords": ["remote", "worldwide", "global", "anywhere"] },
  "remote_required": true,
  "cv_text": null
}
```

`cv_text` starts null. The generation script fills it (see A2).

### A2. Persona CV Generation (`eval/gen_personas.py`)

One-shot script. For each persona without a `cv_text`:

1. Prompt Ollama (generation model) with persona fields → produce a 300–500 word
   synthetic CV in plain prose. No real names, no real companies.
2. Embed the CV text via `embed()` → store as `cv_embedding` in the JSON.
3. Write back to `eval/personas/{slug}.json`.

The resulting JSON has `cv_text` and `cv_embedding` fields alongside the schema
above. The embedding field means personas can be used as a drop-in `Profile`
substitute without hitting the DB.

### A3. Synthetic Job Generation (`eval/gen_synthetic_jobs.py`)

For each persona, generate 20 synthetic job postings — 5 per grade tier — via
Ollama. Grade is defined by construction (the generator knows the intent):

| Grade | Intent | Example for `alice-rust-senior` |
|---|---|---|
| 3 | Exact match | Senior Rust engineer, distributed systems, global remote |
| 2 | Relevant | Senior C++ engineer, game engine (adjacent stack), remote |
| 1 | Marginal | Mid Python backend engineer (wrong level + stack) |
| 0 | Irrelevant | Junior PHP developer, on-site São Paulo |

Prompt template:

```
Write a realistic job posting body (150–250 words) for the following role.
Write as the hiring company. No markdown, no bullet points.
Role intent: {grade_description}
Target persona: seniority={seniority}, stack={tech_stack}, domain={domain}
Return JSON: {"title": "...", "company": "...", "location": "...",
              "description": "...", "requirements": "...", "responsibilities": "..."}
```

Output: `eval/fixtures/jobs_{slug}.json` — a list of job objects with a `grade`
field. Committed to git (synthetic text, no PII).

### A4. DB Injection (`eval/inject_synthetic.py`)

Inserts persona profiles and synthetic jobs into the live DB for a full eval run:

1. For each persona, upsert a `Profile` row with `source="synthetic"` (or a
   dedicated flag if a migration is added — see schema note below).
2. For each synthetic job, upsert a `Job` row with `source="synthetic"`.
3. Compute missing embeddings (persona CV embedding, job description embedding).
4. Insert `EvalLabel` rows with the construction-defined grade.

**Idempotent**: re-running inject replaces rows keyed by `(source, slug)`.

**Schema note**: `Job.source` already exists. `Profile` needs either a
`is_synthetic: bool` column (requires Alembic migration) or use a naming
convention (`full_name = "synthetic:{slug}"`). Decision deferred to implementation;
the injection script should document the chosen convention.

### A5. Cleanup (`eval/inject_synthetic.py --teardown`)

Deletes all rows where `source="synthetic"` from `jobs`, and matching `profiles`
and `eval_labels`. Safe to run before re-generating fixtures.

### A6. Eval Integration

Add a `just eval-synthetic` recipe that:
1. Runs inject (idempotent)
2. Runs `job-radar-eval-run` filtered to synthetic profile IDs
3. Reports per-persona nDCG@10 + BPref

The PJB benchmark finding is relevant here: report metrics **stratified by persona
domain**, not just averaged. A system that scores well on `alice-rust-senior` but
poorly on `eve-data-junior` has a real failure mode worth surfacing.

---

## Phase B — InPars-Style Query Generation

### B1. Approach

For each job in the live DB (real corpus, not synthetic), prompt Ollama to generate
3 query variants that the job would satisfy:

```
Given this job posting, write 3 different search queries a job seeker might use
to find it. Return JSON: {"specific": "...", "medium": "...", "broad": "..."}
  - specific: stack name + title (e.g. "rust systems engineer remote")
  - medium:   domain + level (e.g. "senior backend engineer distributed systems")
  - broad:    domain only (e.g. "systems programming remote")

Job title: {title}
Requirements: {requirements[:400]}
```

### B2. Storage (`eval/synthetic_qrels.json`)

A flat JSON file mapping `query_text → {job_id: 3}` (the source job is grade 3
for that query by construction). Non-source jobs are not labeled — they are simply
absent from the qrels for that query, which is the correct TREC treatment.

Schema:
```json
[
  {
    "query": "rust systems engineer remote",
    "qrels": { "<job_uuid>": 3 },
    "source_job_id": "<job_uuid>",
    "variant": "specific"
  }
]
```

### B3. Multi-Query Eval Runner (`eval/run_synthetic_queries.py`)

For each query in `synthetic_qrels.json`:
1. Run BM25 + HyDE + CV retrieval (no profile filter, since these are corpus-only queries)
2. Compute nDCG@10 + BPref against the single-job qrel (grade 3)
3. Average across all queries in the file

This gives **per-query-intent** metric breakdown:
- `specific` queries should score highest (BM25 advantage)
- `broad` queries reveal semantic retrieval quality (vector arm advantage)
- The gap between specific and broad nDCG quantifies how much the system relies on exact keyword overlap

### B4. Scale

Generate queries for a random sample of 200–500 jobs from the corpus (not all —
Ollama generation is the bottleneck). The `eval/gen_queries.py` script should
accept `--limit N` and be resumable (skip jobs already in `synthetic_qrels.json`).

---

## Implementation Order

| Step | Script / File | Output | Prereqs |
|---|---|---|---|
| A1 | Write persona JSON fixtures manually | `eval/personas/*.json` | None |
| A2 | `eval/gen_personas.py` | CV text + embeddings in fixtures | Ollama running |
| A3 | `eval/gen_synthetic_jobs.py` | `eval/fixtures/jobs_*.json` | A2 done |
| A4 | `eval/inject_synthetic.py` | DB rows + EvalLabel rows | A3, DB running |
| A5 | `just eval-synthetic` recipe | Per-persona metrics | A4 done |
| B1 | `eval/gen_queries.py` | `eval/synthetic_qrels.json` | Ollama, real corpus |
| B2 | `eval/run_synthetic_queries.py` | Per-query-intent metrics | B1 done |

Phase A and Phase B are independent — either can be implemented first.

---

## Eval Quality Gates (Proposed)

Once synthetic data is in place, add to CI (or `just eval-synthetic`):

- Hybrid HYBRID persona nDCG@10 ≥ 0.60 for each persona (grade-3 jobs must rank in top 5)
- `specific` query nDCG@10 ≥ 0.50 averaged across 200 queries (BM25 sanity)
- `broad` query nDCG@10 ≥ 0.30 (semantic arm contributing)
- Grade-0 jobs for a persona must not appear in top-10 for that persona's query

These thresholds are starting points — calibrate after the first full run.

---

## What This Unlocks

| Capability | Pool-based labels | Phase A | Phase B |
|---|---|---|---|
| Zero pool bias | ✗ | ✓ | ✓ |
| Multi-profile eval | ✗ | ✓ | ✗ (corpus-only) |
| Multi-query eval | ✗ (1 query) | ✓ (5 personas × 4 query types) | ✓ (200+ queries) |
| Regression gate in CI | fragile | stable | stable |
| No human labeling needed | ✗ | ✓ | ✓ |
| Tests real corpus | ✗ | ✗ | ✓ |
