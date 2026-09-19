# Eval Harness — Usage Guide

Measures whether the hybrid retrieval system surfaces the right jobs for your profile.
The methodology is TREC-style graded relevance: you provide ground-truth labels, the
system retrieves and ranks, metrics quantify agreement between the two.

---

## Quick start

```bash
just eval-ollama-start                        # dedicated Ollama on :11435
just eval-label --fully-auto --limit 80       # bootstrap labels via LLM
just eval-label --review                      # correct any LLM mistakes
just eval-run                                 # measure baseline
just eval-sweep                               # find best parameters
just eval-ollama-stop
```

---

## Labels

Labels are your ground truth — the only signal that gives metrics meaning.
Every label is a `(profile, job, grade)` triple stored in `EVAL_LABEL`.

| Grade | Meaning |
|---|---|
| 3 | **Strong** — exactly the role you want; right stack, level, domain |
| 2 | **Relevant** — you'd apply; maybe one thing off but belongs on the page |
| 1 | **Marginal** — adjacent role, off-by-one seniority, one key skill missing |
| 0 | **Not relevant** — wrong stack, domain, geo-blocked, or clearly off |

**Target:** 50–80 labeled pairs per profile, at least 15 at grade ≥ 2.
Below that, metrics are too noisy to trust.

### Label quality matters more than quantity

nDCG weights grade-3 jobs 7× more than grade-2. A mislabeled grade-3 (should
be grade-2) shifts nDCG more than 10 correctly labeled grade-0 jobs. The
grade-2/3 boundary is the one to get right. When in doubt, lean grade-2.

### Labeling commands

| Command | What it does |
|---|---|
| `just eval-label --fully-auto --limit 80` | LLM labels all jobs, no prompts — fast bootstrap |
| `just eval-label --auto-prelabel` | LLM suggests, you confirm each one |
| `just eval-label` | Pure human, no AI seed |
| `just eval-label --review` | Review and correct existing labels one by one |

Always run `--review` after `--fully-auto`. The LLM is wrong ~15–25% of the
time on marginal cases (grade-1 vs grade-2 boundary especially).

---

## Commands

### `just eval-run`

Evaluates three canonical retrieval configurations against your labeled qrels
and writes a timestamped JSON to `eval/results/`.

| Config | Arms active |
|---|---|
| `hybrid` | lexical (BM25) + HyDE vector + CV vector |
| `vector_only` | HyDE + CV — no keyword matching |
| `keyword_only` | BM25 only — no vectors |

**What to look for:**

- `hybrid` should beat both ablations on nDCG@10. If it doesn't, one arm is
  hurting more than it helps — check the sweep.
- Check `recall_at_50` first. If it's below ~0.6, the pool is missing relevant
  jobs before ranking even starts. Increase `pool` in the sweep.
- Large gap between `hybrid` and `vector_only` → BM25 is contributing. Large
  gap between `hybrid` and `keyword_only` → vectors are contributing. If gaps
  are small, arms are redundant at current weights.

### `just eval-sweep [--dimension k|weights|pool|boosts|all]`

One-at-a-time (OAT) parameter sweep. Holds all other parameters at default
while varying one dimension across a grid.

| Dimension | What it controls | Default | Grid |
|---|---|---|---|
| `k` | RRF rank-sensitivity constant | 60 | 20 30 40 60 90 120 |
| `weights` | Per-arm RRF multipliers | equal | 6 combinations |
| `pool` | Candidates fetched per arm | 50 | 20 50 100 |
| `boosts` | BM25 field weights (title/req/resp) | 5/2/1 | 6 combinations |

**What to look for:**

- Pick the winner per dimension by nDCG@10, not Recall@50 (Recall measures
  coverage, nDCG measures ranking quality — you want both, but nDCG is primary).
- If `k=20` and `k=120` produce the same nDCG, k is not a sensitive parameter
  for your data — keep the default.
- If the `weights` sweep shows one arm dominates (e.g., HyDE↑ always wins),
  lock that weight and re-sweep the others.
- `pool=20` vs `pool=100` gap on Recall@50 tells you how many relevant jobs
  exist outside the top-20. A large gap means you need a bigger pool.
- BM25 boosts matter when job descriptions use consistent field structure
  (requirements/responsibilities clearly separated). If extraction quality is
  low, boost differences will be noise.

After each dimension sweep, lock the winner before sweeping the next.

---

## Metrics

### nDCG@10 — primary

Normalized Discounted Cumulative Gain at 10. Measures ranking quality in the
top 10 results. Score in [0, 1]; higher is better.

**How it works:** each result at position `i` contributes `(2^grade − 1) / log2(i+1)`.
Normalized against the perfect ranking so scores are comparable across profiles.

| Grade | Points |
|---|---|
| 3 | 7 |
| 2 | 3 |
| 1 | 1 |
| 0 | 0 |

Position 1 contributes full points; position 10 contributes ~30% of full points.
A grade-3 job at rank 5 scores the same as ~2.3 grade-2 jobs at rank 1.

**What scores mean in practice:**

| nDCG@10 | What it suggests |
|---|---|
| ≥ 0.7 | Strong — grade-2/3 jobs are consistently near the top |
| 0.5–0.7 | Decent — relevant jobs appear in top 10 but not always first |
| 0.3–0.5 | Weak — relevant jobs present but buried |
| < 0.3 | Poor — retrieval is not finding your target roles |

### Recall@50

Fraction of all labeled grade ≥ 2 jobs that appear in the top 50 results.
Measures *coverage* — whether the retrieval is missing good jobs entirely.

Check this before nDCG. If Recall@50 is low, the ranking problem is secondary
to the pool problem — relevant jobs aren't being retrieved at all.

### MRR — Mean Reciprocal Rank

`1 / rank` of the first grade ≥ 1 result. Measures: "how quickly does
something useful appear?" Useful as a sanity check; a low MRR with a decent
nDCG means the very first result is often irrelevant.

### P@5, P@10 — Precision at k

Fraction of the top-5 / top-10 that are grade ≥ 2. Measures purity of the
top of the list. More interpretable than nDCG for a human reading the results
page — "how many of the first 5 jobs shown would I actually consider?"

---

## Reading a result file

`eval/results/eval-<timestamp>.json` contains one entry per configuration:

```json
{
  "config_name": "hybrid",
  "ndcg_at_10": 0.612,
  "recall_at_50": 0.74,
  "mrr": 0.833,
  "p_at_5": 0.6,
  "p_at_10": 0.5,
  "pool_size": 48,
  "num_labeled": 62
}
```

- `pool_size` — how many distinct jobs were returned across all arms after
  fusion. If this is much smaller than `limit`, some arms are returning
  overlapping results or the DB has few matching jobs.
- `num_labeled` — labeled pairs used. If this is much smaller than your total
  labels, some labeled jobs never appear in the retrieval pool — they are
  invisible to the system at current settings.

---

## The regression gate

Once you have a good baseline, commit the winning config and its nDCG@10 to
`eval/golden/`:

```
eval/golden/qrels.json   — your labeled pairs (profile_id, job_id, grade)
eval/golden/result.json  — winning config + ndcg_at_10
```

From that point, `just test` enforces: **current nDCG@10 ≥ golden − 0.02**.

The 0.02 band is the Craswell et al. noise floor — smaller deltas are
indistinguishable from label noise on a set of this size. A real regression
(broken arm, bad weight change, schema migration affecting BM25 indexing) will
exceed it and fail CI before merging.

---

## Things that can fool the metrics

**Too few labels at grade ≥ 2.** nDCG normalizes against the ideal ranking.
With only 3 grade-2 jobs labeled, moving one from rank 8 to rank 3 swings
nDCG by ~0.15 — larger than the regression gate. Label more.

**Pool bias.** Labels come from the retrieval pool itself (jobs the system
already returns). The system can't be penalized for jobs it never surfaces.
Recall@50 will look artificially high. Periodically add labels from outside
the pool (e.g., jobs you found manually) to get an unbiased recall estimate.

**LLM label noise.** `--fully-auto` labels are noisier near the grade-1/2
boundary. Always review after auto-labeling. A wrong grade-3 label has more
impact than five wrong grade-0 labels.

**Model changes.** If you switch the generation or embedding model, old labels
remain valid (they reflect job content, not model behavior), but the golden
result should be re-established since the embedding space shifts. To *choose*
between embedding models in the first place, don't use this harness — see the
next section.

---

## Comparing embedding models

Choosing an embedder is a component question — *which model ranks this pool best?* — and
the hybrid eval above can't answer it: BM25 and RRF dilute the embedder's effect, the
embedder is hard-wired into `embed()` / `search_vector`, labels were pooled from the
incumbent (a challenger that surfaces unlabeled jobs is scored as wrong), and the golden
result is saturated (MRR, P@5, P@10 are 1.0; nDCG@10 0.886).

A separate, **dense-only** harness (`eval/embedding/`) varies nothing but the embedder. The
job pool, the HyDE query texts and the production embed text (title + extracted
requirements/responsibilities) are frozen at import. It reads production **one way**,
through a read-only connection, into its own database (`job_radar_evals`, `embedding_*`
tables) and never writes prod. It reuses this document's grade definitions (0–3) and the
pure functions in `eval/metrics.py`; it is not part of the regression gate, and the
evaluation never runs in CI.

It does **not** tell you: whether the winner survives BM25 fusion (re-run `just eval-run`
after any cutover), how it does for other profiles or another query recipe, or how it does
on text the extraction step never saw. Design:
[plans/EMBEDDING_EVAL_DESIGN.md](plans/EMBEDDING_EVAL_DESIGN.md); build plan:
[plans/EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md](plans/EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md).

### Runbook

Prerequisites: Postgres up, `EVALS_DATABASE_URL` in `.env`, `ollama pull qwen3-embedding:0.6b
&& ollama pull bge-m3`. `embedding-import` and `embedding-verify` also load prod code, so
`.env` needs the four prod variables (`DATABASE_URL`, `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`,
`GENERATION_MODEL`); the HyDE texts must already be cached (`job-radar-fit` once).

```bash
docker compose -f infra/docker-compose.yml up -d
just evals-db-init                                        # create + migrate job_radar_evals
just embedding-import --tier A --name <topic>             # read-only from prod; freezes the topic
just eval-ollama-start                                    # optional dedicated Ollama
just embedding-run --topic <topic> --embedder all --per-text
just embedding-verify --topic <topic>                     # both parity checks must pass
just embedding-label --topic <topic> --n 50                # blind-label ~50 jobs yourself (resumable)
just embedding-judge --topic <topic> --model Gemma3:12b --limit 24    # timing spike
just embedding-judge-calibrate --topic <topic> --judge-run <id>
just embedding-judge --topic <topic> --model Gemma3:12b               # full pool, ~1 h
just embedding-judge-calibrate --topic <topic> --judge-run <id>
just embedding-evaluate --topic <topic> --view all --judge-run <id>
just embedding-evaluate --topic <topic> --view all --judge-run <id> --language en   # English only
just embedding-labels-export --topic <topic>              # backup of your blind labels
```

- **import** prints pool size, HyDE-text count and prod row counts and Alembic revision
  before and after. It imports **no labels**: the prod `eval_labels` were produced by the fit
  LLM, not a human. The imported `cv_text` is scrubbed of
  name, email, phone and URLs (best effort).
- **run** embeds the pool with every candidate in `eval/embedders.toml` and stores full
  rankings. Vectors are cached per embedder fingerprint, so a second run is a cache hit;
  `--per-text` adds one run per HyDE text for the query-sensitivity table.
- **verify** records `parity_ranking` (the harness reproduces prod's dense top-100) and
  `parity_reconstruction` (re-embedded incumbent vectors match the stored ones). If the
  second fails, stop: the corpus changed and the comparison is suspect.
- **label-blind** (`just embedding-label`) shows you one job at a time — the rubric, then the
  400-char label view, never an LLM grade, score, rank or run — and stores each grade the
  moment you give it. It draws ~60% from *pooled* jobs (in the top 30 of some but not all
  embedders) and ~40% *random* ones, in an order fixed by `--seed`, so rerunning continues
  where you stopped. `s` skips, `q` quits. Aim for ~50 (`--n 50`): these `human_blind`
  labels are the only gold. Only English postings are offered (`--language en`, the default;
  `any` offers all), so the postings you skip or never see cannot reappear.
- **judge** labels every job. Its 4 few-shot exemplars (1 per grade, `--fewshot-per-grade`)
  come from your blind labels, so label first. Iterate the prompt against them, freeze
  `PROMPT_VERSION`, then run the full pool (resumable with `--resume`).
- **judge-calibrate** scores the run against every non-exemplar blind label; `--held-out-half`
  opts into the held-out half only (too small to be decisive at ~50 labels).

### Reading the output

`--language en` scores only English documents (a stopword heuristic, `eval/embedding/language.py`,
that recognizes Spanish and Portuguese and calls the rest English): rankings, labels, R and
the Recall@100 ceiling are restricted to them, and the report adds the *off-language leakage*,
how many non-English documents each embedder still puts in its unrestricted top 100. Use it
for the English-only corpus; without it the report adds a per-language slice table instead.

`evaluate` prints one table per view and writes
`eval/results/embedding-<topic>-<view>-<ts>.json` (gitignored) with run metadata: model
digest, quantization, Ollama version, docs/s, truncated documents, git sha, and the checks.

**Views.** `silver`: the Gemma labels for every job — the headline view, needs `--judge-run`.
`human`: only your blind labels (`human_blind`) and authored `constructed` ones — partial, and refused with "no human labels yet" until you have made some. `effective`: human over judge by
precedence; it mixes labelers, and the judge over-rates against humans, so human-labeled
documents grade lower than equivalent silver-only ones. Read `silver` first.

**Headline metrics: nDCG@100, P@100, AP** (grade ≥ 2). 100 is what the pipeline hands to
fit analysis, so P@100 is "how much of that list is relevant". Secondary: nDCG@10,
Recall@50, Recall@100, bpref, P@10, judged@10/@100.

**Recall@100 always prints with its ceiling `min(1, 100/R)`**, where R is the number of
grade ≥ 2 documents. On the real pool R is estimated at ≈ 430–660, so Recall@100 cannot
exceed ≈ 0.15–0.23 for any embedder: it is a coverage bound, not a quality score.

**The partial `human` view reports only condensed metrics** (unjudged documents removed),
bpref and judged@k. With unjudged = 0, the metric rewards how many of the labeled
documents a ranking happens to surface: deliberately degraded nomic variants scored above
the incumbent (nDCG@100 0.261 vs 0.199) while only 26–34% of any top-100 was judged (measured
against the 212 LLM-made prod labels, not human).
Always read judged@k next to a `human` number.

**Negative controls.** A random ranking and a shuffled-tail version of the incumbent are
scored beside the candidates. If a control beats the incumbent, the table is flagged
unreadable.

**Intervals.** `--bootstrap N` (default 1,000) draws a seeded paired bootstrap over
documents and prints each candidate's difference vs the incumbent with a 95% interval; an
interval containing 0 is not a difference. On the ~212 prod-labeled documents (LLM-made, not human) the SD of an
AP difference is ≈ 0.035–0.04, so a gap below ≈ 0.07–0.08 AP is noise at that label count.
Complete silver labels shrink the document variance, but no interval here covers judge
error or profile variation.

**Query sensitivity.** A recipe table compares the mean of the HyDE texts against each text
alone and shows the embedder order per recipe. The real profile's three texts are near-identical
(cosine 0.96–0.99), so the query is effectively one vector; a different query *recipe* is not tested.

**`UNCALIBRATED` and `UNVERIFIED`.** `silver` and `effective` require a *passed*
`judge_calibration` for the pinned `--judge-run`: quadratic-weighted κ ≥ 0.6 against your
blind labels (a single-shot probe against the LLM-made prod labels scored 0.41, so expect several
prompt rounds).
Any comparison also needs both parity checks. `evaluate` refuses otherwise;
`--allow-unverified` prints the table anyway, stamped `UNCALIBRATED` / `UNVERIFIED`. The gate
is coarse: with ~50 labels (46 after the exemplars) κ_w has a standard error of ≈ 0.1, and the
~30 pooled labels come from the embedders' top-30s, so calibration is optimistic at the top of
the ranking (the ~20 random ones are the unbiased part).

**The label view.** `label-blind` shows `eval/label.py`'s view: title,
company, source, seniority, and requirements/responsibilities clipped to 400 characters
each (the description only when both are empty) — no location. The judge is shown exactly
that view so calibration compares like with like. Embedders and the fit stage read the
unclipped text (85% of pool documents are clipped in the label view).

**Truncation.** Ollama caps `nomic-embed-text` and `bge-m3` at 2,048 tokens whatever
`num_ctx` says; Qwen3 honors 8,192. Truncation is detected per document
(`truncate=false`, retry truncated) and the count is printed per run. Quantization also
differs (Qwen3 Q8_0, the others F16) and is recorded, not corrected for.

**Slices.** By extraction branch, descriptions over 4,000 characters, one-field-only
extractions, and source.

### What a result supports

A ranked shortlist with intervals — not yet a decision. A claim needs `silver`
(calibrated) and `effective` to agree, with condensed `human` as a sanity check. Later
stages, not built: more blind labeling of the disagreement region (beyond the first ~50), a
pre-registered decision rule and judge-error sensitivity (Stage 2); synthetic-persona
topics (Stage 3); cutover of the winner (Stage 4).
