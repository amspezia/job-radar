# Fit Analysis Throughput — Implementation Plan

> Status: **plan, ready to build**. Targets the wall-clock cost of `job-radar-fit`,
> which currently spends ~13 minutes scoring a 100-job candidate set. Every decision
> below is grounded in measurements taken 2026-08-08 against the live corpus (9,902
> jobs) on Apple M5 Pro / 24 GB with `qwen2.5:7b` Q4_K_M. **No quality is traded for
> speed without an eval gate.**

---

## 0. The measurement

One instrumented `analyze_fit` call, via Ollama's own `prompt_eval_*` / `eval_*` counters:

| Phase | Work | Time | Share |
|---|---|---|---|
| Prompt eval (prefill) | 2,511 tok | **0.4 s** | 0.3 % |
| Decode (output) | 933 tok | **109.5 s** | **93 %** |
| Model load | — | 0.1 s | ~0 % |

Prefill processes the whole prompt in one batched pass. Decode runs **one forward pass
over 7.6 B parameters per output token**, strictly sequentially. The ~800 tokens each job
emits therefore cost ~200× more than the ~2,500 tokens it reads.

The run's cost reduces to one equation:

```
wall_time ≈ (jobs × output_tokens_per_job) ÷ aggregate_decode_throughput
          ≈ (100  ×      ~800           ) ÷      ~100 tok/s              ≈ 13 min
```

**Measurement caveat, recorded honestly:** the absolute per-stream rates above were
captured while a second fit pipeline was competing for the same 12 Ollama slots, so
they understate uncontended throughput. What is *not* affected by that confound, and
what this plan rests on: the **prefill/decode ratio**, the **output-token volume and its
composition**, and all **A/B comparisons** (each pair measured under identical load).

### Output composition (one 643-token response)

| Component | Share of output |
|---|---|
| `Requirement.text` | 20 % |
| `Evidence.quote` | 20 % |
| `summary` (289 chars) | ~11 % |
| JSON structural tokens + `kind`/`satisfaction` | remainder |

Evidence density measured **1.0 quote per requirement** — the model is not over-quoting.

---

## 1. Locked decisions

| # | Point | Decision |
|---|---|---|
| 1 | Analysis breadth | **All retrieved jobs stay analyzed.** Truncating to top-N by RRF rank was considered and **rejected**: it removes output rather than work per job, and substitutes a weaker signal (retrieval rank) for the stronger one it would skip (the fit judgment). |
| 2 | Primary lever | **Eliminate repeat work** — persist assessments so re-runs analyze only what changed (§3). |
| 3 | Secondary lever | **Cut output tokens** by removing write-only schema fields (§4). Decode is 93 % of runtime; tokens not emitted are time not spent. |
| 4 | Model tiering | `qwen2.5:3b` evaluated behind config + an eval gate (§5). **Adopt only on evidence.** A 3 B model is *faster*, not *better*. |
| 5 | Structured output | **Keep.** Measured 8.7 tok/s (JSON schema) vs 9.1 tok/s (free text) — grammar-constrained decoding is essentially free. The grounding thesis costs nothing. |
| 6 | Prompt / prefix caching | **Rejected.** Prefill is 0.3 % of runtime; the prompt already orders profile+CV first (optimal for per-slot prefix reuse). Optimizing it is worth ~0.4 s of 118 s. |
| 7 | Concurrency | **Leave at 12.** `_MAX_CONCURRENT_ANALYSES` already matches `OLLAMA_NUM_PARALLEL=12`. Sweep only if measuring *aggregate* throughput (§6). |
| 8 | HTTP client pooling | **Out of scope.** `generate()` builds a fresh `AsyncClient` per call; real, but <1 % at 100 s-scale latencies. Fix it for hygiene, not speed. |

---

## 2. Measured dead ends

Recorded so they are not re-litigated. Each is the intuitive fix, and each is worth
approximately nothing here:

- **Shrinking or caching the CV in the prompt** (4,454 chars). Targets the 0.4 s prefill.
- **Dropping the JSON schema for free-text + parsing.** Measured 4 % faster decode; costs
  the schema guarantee that makes the score unhallucinatable.
- **Raising `OLLAMA_NUM_PARALLEL`.** RAM scales with `NUM_PARALLEL × CONTEXT_LENGTH`;
  12 slots × 4096 ctx on 24 GB shared with the OS is already ambitious, and per-stream
  rate falls as slots rise.
- **Running two fit pipelines concurrently.** Roughly halves both.

---

## 3. Work item 1 — persistent fit assessments

**Problem.** There is no fit-results table. Every invocation re-analyzes all 100 jobs,
including the ~90 % that are byte-identical to the previous run.

### 3.1 Schema

New table `fit_assessments`, mirroring `FitAssessment` plus its cache key:

| Column | Type | Note |
|---|---|---|
| `id` | UUID PK | |
| `profile_id` | UUID FK → `profile.id`, indexed | |
| `job_id` | UUID FK → `jobs.id`, indexed | |
| `content_hash` | String(64) | copied from `Job.content_hash` at write time |
| `model` | String(255) | the generation model that produced it |
| `prompt_version` | Integer | bumped whenever `_PROMPT` or `FitJudgment` changes |
| `score` | Integer, nullable | `None` = pre-flight refusal |
| `verdict` | String(50) | |
| `gate_failed` | Boolean | |
| `summary` | Text | |
| `judgment` | JSON, nullable | the full grounded `FitJudgment` |
| `created_at` | timestamptz | |

Unique constraint on `(profile_id, job_id, content_hash, model, prompt_version)`.

**The cache key is the load-bearing part.** `content_hash` invalidates on an edited
posting; `model` and `prompt_version` prevent §4 and §5 from silently serving assessments
produced by a *different* schema or a *different* model. Omitting either turns this cache
into a correctness bug the moment the next two work items land.

Precedent for caching derived LLM output on a row: `Profile.dense_query_cache`.

### 3.2 Integration

In `run_fit_pipeline`, between retrieval and analysis:

1. One batch `SELECT` for cached rows matching the retrieved job ids and the current key.
2. Analyze only the misses (the existing semaphore-bounded `asyncio.gather`).
3. Bulk-insert the new assessments.
4. Merge cached + fresh, then sort as today.

Deterministic re-runs must return identical output whether served from cache or not —
`score_fit` is already pure, so a cached `judgment` re-scores identically.

### 3.3 CLI

`--refresh` to bypass and overwrite the cache for this run. No `--no-cache`: reading stale
results is the thing worth an explicit opt-out, and `--refresh` already covers it.

### 3.4 Verification gate

- Unit: cache hit returns a `FitAssessment` equal to the uncached path.
- Unit: each of `content_hash`, `model`, `prompt_version` changing forces a miss.
- Manual: run `job-radar-fit` twice; second run logs ~100 hits and finishes in seconds.

---

## 4. Work item 2 — trim output tokens

**Problem.** ~20 % of every response is `Requirement.text`, which **no code reads.**
Verified: `score_fit` consumes only `.kind` and `.satisfaction`; `fit/cli.py` prints only
score, verdict, title, and URL. It is write-only data generated at ~9 tok/s.

### 4.1 Changes

1. **Drop `Requirement.text`** from `fit/schema.py`. The `Evidence` entry with
   `source: "posting"` already identifies which requirement is being judged, verbatim —
   so auditability survives. This is the honest tradeoff to weigh: `text` is the model's
   *paraphrase* of the requirement, the posting quote is the *source*. Keeping the source
   and dropping the paraphrase is the right side of that trade.
1b. **Order `evidence` before `kind`/`satisfaction`** in the schema. Constrained decoding
   emits fields in declaration order, so field order *is* generation order, and `text`
   was silently acting as reasoning scaffolding — the model restated the requirement
   before labelling it. Removing it without replacement made the model markedly harsher:
   whole postings flipped to all-`unmet` (measured, §4.3). Putting `evidence` first
   restores the scaffolding at quote-cost instead of paraphrase-cost.
2. **Bound `summary`.** Instruct a one-sentence limit in `_PROMPT` and enforce with a
   `max_length` on the field so an over-long draft fails validation rather than costing
   ~70 tokens every call.
3. **Update `_PROMPT`** to stop asking for the removed field.
4. **Bump `prompt_version`** (§3.1) in the same commit — non-negotiable, or the cache
   serves old-schema rows.

Expected: ~25–30 % fewer output tokens, translating ~linearly into wall time.

### 4.2 Verification gate

- `just test` green; `FitJudgment` round-trips.
- Re-run the §0 measurement; record the new tokens/job in this file.
- Spot-check ~10 assessments: verdicts must not shift. A score change here means the
  removed field was influencing the model's reasoning, not just its output size — in
  which case revert and reconsider.

### 4.3 Measured result — **token gate passed, judgment gate NOT passed**

A/B over 5 postings, old (prompt+schema with `text`) vs new, alternating calls back to
back on `qwen2.5:7b` so both arms see identical load:

| Ordering | Mean output tok/job | Δ |
|---|---|---|
| Old (`text` first) | 1,041 | — |
| New, judgment-first (`kind`→`satisfaction`→`evidence`) | 847 | −19 % |
| **New, evidence-first** (shipped) | **604** | **−42 %** |

Evidence-first both saves more and behaves better — surfacing the quotes gives the model
something to condition on, so it stops defaulting to `unmet`.

**What has not been established: that the new judgments are as good.** They still diverge
from the old ones on the same postings — one of five flipped from mostly-`unmet` to
mostly-`met`, and enumerated requirement counts fell sharply (10→2, 9→3). Since
`_coverage` averages over enumerated requirements, fewer requirements makes each score
noisier, which moves the number independently of any change in judgment quality.

Five samples with no ground truth cannot say which version is *right*. Resolving it needs
the fit-quality comparison specified in §5.2 — the same harness, run against `EvalLabel`
grades. **Treat §4 as unverified until that runs.** `--refresh` plus `PROMPT_VERSION`
makes reverting cheap: bump the version and the old judgments are simply re-generated.

---

## 5. Work item 3 — evaluate `qwen2.5:3b`

**Hypothesis, not a decision.** Measured 87 tok/s uncontended vs the 7 B's 8.7 tok/s
contended — not a fair pairing. Expect roughly parameter-proportional gain (~2–2.5×),
and verify.

### 5.1 Changes

1. Add `fit_model: str | None = None` to `config.Settings`, defaulting to
   `generation_model` when unset. Exact precedent: `extraction_model`.
2. Thread it through `analyze_fit` → `generate(model=...)`.
3. Add to `.env.example` with the same commented guidance as `EXTRACTION_MODEL`.

### 5.2 The evaluation gap — read before running

The M6 harness measures **retrieval** (nDCG@10, Recall@pool). It does **not** measure fit
quality, and swapping the fit model does not move a single retrieval metric. Adopting 3 B
on the strength of the existing eval gate would be measuring the wrong thing.

A fit-quality comparison is therefore a prerequisite. Cheapest sufficient version:
freeze one candidate set, run it under both models, and report verdict-agreement plus
mean absolute score delta against the human `EvalLabel` grades already in the DB.

**Decision rule:** adopt 3 B only if verdict agreement with the 7 B is ≥ 90 % on
non-`none` verdicts and mean |Δscore| ≤ 5. Otherwise keep 7 B and take the win from
§3 + §4 alone.

---

## 6. Concurrency note

Do not change `_MAX_CONCURRENT_ANALYSES` or `OLLAMA_NUM_PARALLEL` speculatively. If swept:
measure **aggregate** throughput (jobs ÷ total wall time), never per-request tok/s, which
declines with concurrency by construction. Keep the two values equal — a semaphore larger
than the server's slot count only moves the queue into Ollama.

---

## 7. Build order

Sequential; each lands on its own branch, lint-clean and CI-green.

| Step | Item | Why this order |
|---|---|---|
| 1 | §4 output trim | Smallest diff, no schema migration, immediately measurable. |
| 2 | §3 assessment cache | Lands *after* the schema settles so `prompt_version` starts at a stable value. |
| 3 | §5 model evaluation | Needs the cache (§3) to make repeated A/B runs affordable. |

---

## 8. Projected outcome

Baseline 100 jobs ≈ 13 min. Projections, to be replaced with measurements as each lands:

| After | Cold run | Warm re-run |
|---|---|---|
| Today | ~13 min | ~13 min |
| §4 output trim | ~9–10 min | ~9–10 min |
| + §3 cache | ~9–10 min | **< 1 min** (only new postings) |
| + §5 3 B *(if the gate passes)* | ~4–5 min | < 1 min |

The warm-run number is the one that matters for daily use: with the 30-day window from
the `--max-age-days` work, a second run analyzes only newly ingested postings.
