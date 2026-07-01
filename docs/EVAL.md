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
result should be re-established since the embedding space shifts.
