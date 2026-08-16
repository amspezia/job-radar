# M6 — Evaluation Harness Implementation Plan

> Status: **plan, ready to build**. Realizes the M6 milestone and §12 of
> `PHASE_1_DESIGN.md`. This file locks the metric definitions, the file-by-file
> build, the labeling protocol, and the CI regression gate. It is grounded in the
> TREC evaluation paradigm; every primitive and metric below carries its citation.

---

## 0. Why this exists

The project's whole thesis is **reliability** — evaluation, observability, human-in-the-loop,
guardrails — not raw retrieval cleverness (CLAUDE.md). Up to now every retrieval change
(HyDE, structured-field extraction, BM25 boosts, RRF arms) has been justified *theoretically*.
None has been **measured**. The eval harness closes that gap: it converts "this should be
better" into "this is +0.04 nDCG@10, outside noise." Without it, parameter tuning is guessing,
and the §12 "hybrid vs vector-only vs keyword-only, reported honestly" deliverable cannot ship.

The harness answers four questions, in priority order:

1. **Recall ceiling** — what fraction of relevant jobs even *enter* the candidate pool?
   (If low, ranking tuning is pointless — fix recall first.)
2. **Ranking quality** — given the pool, are the best jobs at the top? (nDCG@10.)
3. **Configuration tuning** — which `k` / arm-weights / pool / field-boosts maximize the above?
4. **Regression safety** — does a new change silently degrade retrieval before merge?

---

## 1. Locked decisions

| # | Point | Decision |
|---|-------|----------|
| 1 | Evaluation paradigm | **TREC qrels + run + gain-based metrics** ([Voorhees & Harman]). The canonical IR methodology; everything below is a specialization of it. |
| 2 | Relevance scale | **4-grade graded relevance** (0/1/2/3), not binary. Enables nDCG's graded gain ([Järvelin & Kekäläinen]). Maps onto the existing `EVAL_LABEL` table. |
| 3 | Primary metric | **nDCG@10** — the web-IR standard ([Craswell et al.], TREC Deep Learning). |
| 4 | Recall metric | **Recall@pool (Recall@50)** — measured *first*; it is the ceiling on end-to-end quality ([Thakur et al.], BEIR). |
| 5 | Secondary metrics | **MRR** (first useful result early) and **P@5 / P@10** (spot-check precision). |
| 6 | Gain transform | **Exponential gain** `2^grade − 1` — the modern nDCG formulation ([Burges et al.], LambdaRank), emphasizes highly-relevant docs. |
| 7 | Metric engine | **`ranx`** ([Bassani]) for compute + run comparison + significance tests; we own a tiny pure-Python fallback for the core metrics so the math is auditable and unit-tested. |
| 8 | Query identity | **One `Profile` = one query.** Labels are query-relative; `EvalLabel.profile_id` (added in the Part-2 fix) is the topic id. |
| 9 | Label bootstrapping | **LLM-assisted pre-labeling**, human-confirmed. `analyze_fit` verdict seeds the grade; the human overrides ([Saad-Falcon et al.], ARES; [Faggioli et al.]). |
| 10 | Labeling target | **Retrieval relevance** ("should this appear?"), not application intent ("would I apply?"). These are different signals ([Craswell et al.]). |
| 11 | Tuning method | **One-at-a-time (OAT) sweep**, not full grid — sweep `k`, then weights, then pool, then field-boosts ([Voorhees & Harman]). |
| 12 | CI gate | **Golden-config regression test**: nDCG@10 must not drop below `golden − 0.02` ([Craswell et al.] noise threshold). |

---

## 2. The primitives (what we are actually building)

Three objects, lifted directly from TREC ([Voorhees & Harman], *TREC: Experiment and
Evaluation in Information Retrieval*). Understanding these three is understanding the harness.

### 2.1 The qrel (relevance judgment)

A triple `(query_id, doc_id, grade)`. In our system: `(profile_id, job_id, label)` — the
`EVAL_LABEL` row. The set of all qrels is the **ground truth**, fixed across experiments.

Grades are **ordinal and query-relative** ([Järvelin & Kekäläinen]): a Senior-Python posting
is grade 3 for this profile and grade 0 for a junior iOS profile. This is exactly why
`profile_id` was a blocking schema fix — a judgment without a query has no meaning.

Our 4-grade scale (calibrated to *retrieval relevance*, decision #10):

| Grade | Meaning | Seed from `analyze_fit` |
|-------|---------|-------------------------|
| 3 | Strong: squarely the role searched for | verdict `strong` |
| 2 | Relevant: would belong on the results page | verdict `moderate` |
| 1 | Marginal: adjacent domain / off-by-one level / missing one key skill | verdict `weak` |
| 0 | Not relevant: wrong stack/domain, or gated by region/seniority | verdict `none` |

### 2.2 The run

The ranked list the system returns for a query, best-first. In our system: the output of
`search()` expressed as `{job_id: rrf_score}`. TREC line format is
`query_id Q0 doc_id rank score run_name`; `ranx` consumes the dict form directly.

The **run is separable from the qrels** — that separation is what makes the harness
composable. The qrels never change; the run changes when we change `k`, `weights`, `pool`,
or field boosts. Every experiment = one fixed qrels × N runs.

### 2.3 The metric (gain-based scoring)

**DCG@k** ([Järvelin & Kekäläinen], with exponential gain from [Burges et al.]):

```
DCG@k = Σ_{i=1..k}  (2^grade_i − 1) / log2(i + 1)
```

The numerator (gain) rewards relevance; the denominator (discount) rewards putting it early.
Position 1 discount = 1.0; position 10 ≈ 0.29. A grade-3 result is worth 7.0 at the top,
2.03 at position 10 — ranking matters.

**nDCG@k = DCG@k / IDCG@k**, where IDCG is the DCG of the perfect ordering (all 3s, then 2s,
…). Normalizes to [0, 1] so queries with different numbers of relevant docs are comparable.

---

## 3. How real services do this (production grounding)

Per CLAUDE.md, every committed file is a hiring signal; the harness mirrors how production
search teams actually evaluate, not a toy.

- **Microsoft Bing / TREC Deep Learning Track** ([Craswell et al.]) — the reference design
  for hybrid-search eval at scale: pooled judgments, nDCG@10 as headline, MRR secondary.
  Our harness is a single-tenant version of the same loop.
- **Google** popularized **nDCG** as the production web-ranking metric; it is the direct
  descendant of [Järvelin & Kekäläinen]'s DCG.
- **LinkedIn (Galene / Talent Search)** ([Geyik et al.], *Fairness-Aware Ranking*) — evaluates
  candidate↔job ranking with nDCG over graded relevance, the exact person–job-fit shape we have.
- **Elastic** ships the **Ranking Evaluation API** (`_rank_eval`) computing P@k, MRR, nDCG over
  saved qrels — productized proof that "qrels + run + metric" is the industry primitive.
- **Pinecone / vector-DB practice** ([Pinecone, *Evaluating Retrieval*]) — "measure **Recall@k
  first**; ranking tuning is wasted if relevant docs aren't retrieved at all." This is why
  decision #4 puts Recall@50 ahead of nDCG.
- **BEIR** ([Thakur et al.]) — established that BM25 *beats* dense retrieval on
  domain-specific, specialized-vocabulary corpora (job titles, tech-stack tokens). Direct
  justification for keeping the lexical arm in the hybrid and for measuring all three
  configurations honestly (§12).
- **RAGAS / ARES** ([Es et al.]; [Saad-Falcon et al.]) — formalize **LLM-assisted labeling**:
  an LLM judge correlates ~0.85–0.92 with humans on *binary* relevance, lower (~0.72) on
  4-grade. Good enough to *seed* labels (decision #9), not to replace the human.

---

## 4. File-by-file build

Everything lands in `eval/` (per `PHASE_1_DESIGN.md` §18 layout). Build order is
dependency order: metrics → labeling → runner → sweep → CI gate.

```
eval/
  __init__.py
  metrics.py        # 4.1  pure functions, no I/O — the auditable math
  qrels.py          # 4.2  load EVAL_LABEL → qrels dict; build run dict from search()
  label.py          # 4.3  CLI: LLM-seeded, human-confirmed grading
  run.py            # 4.4  one config → all metrics → JSON result
  sweep.py          # 4.5  OAT parameter sweep + comparison report
  results/          #      versioned JSON run artifacts (gitignored except goldens)
  golden/           #      committed golden qrels + golden result for the CI gate
tests/
  test_eval_metrics.py   # 4.1  unit tests with hand-computed nDCG/MRR/recall
  test_eval_qrels.py     # 4.2  label→qrels and search→run mapping
  test_eval_gate.py      # 4.6  the regression gate itself
```

### 4.1 `eval/metrics.py` — the auditable core (build first)

Pure functions, **no DB, no `ranx`** — so the math is unit-testable against hand-computed
values and reviewers can verify it. `ranx` is used in `run.py` as the production engine; this
module is the ground-truth reference the tests pin both against.

Signatures (all take `ranking: list[UUID]` best-first and `labels: dict[UUID, int]` grade 0–3):

- `dcg(ranking, labels, k) -> float` — `Σ (2^g − 1)/log2(i+1)`.
- `ndcg(ranking, labels, k) -> float` — `dcg / ideal_dcg`; **0.0 when IDCG is 0** (no relevant
  docs — guard the divide).
- `mrr(ranking, labels, rel_threshold=1) -> float` — `1/position` of first grade ≥ threshold.
- `precision_at_k(ranking, labels, k, rel_threshold=2) -> float`.
- `recall_at_k(ranking, labels, k, rel_threshold=2) -> float` — denominator is **all** docs in
  `labels` with grade ≥ threshold (the recall ceiling), not just those retrieved.

**Verify:** `test_eval_metrics.py` — a perfect ranking → nDCG 1.0; a reversed ranking →
known-low value computed by hand; empty-relevant → nDCG 0.0 not a crash; MRR of
`[grade0, grade0, grade2]` → 1/3.

### 4.2 `eval/qrels.py` — adapters between our DB and the TREC primitives

- `load_qrels(session, profile_id) -> dict[UUID, int]` — read `EVAL_LABEL` rows for the
  profile, map the string `label` → integer grade.
- `build_run(session, profile, config) -> dict[UUID, float]` — call `search()` /
  `run_fit_pipeline`'s retrieval half with the given `config` (k, weights, pool, boosts),
  return `{job_id: fused_score}`. This is the seam where a config becomes a run.

**Verify:** `test_eval_qrels.py` — round-trip a couple of labels; assert `build_run` honors
the config (e.g. a pool override changes the candidate count).

### 4.3 `eval/label.py` — LLM-seeded, human-confirmed labeling CLI

The labor-intensive primitive. Per decision #9/#10 it labels **retrieval relevance**, seeded
by `analyze_fit` to make the human ~3–4× faster ([Saad-Falcon et al.]).

Flow:
1. Load the stored profile; run retrieval (default config) for a candidate pool.
2. For each job not already labeled for this profile (`--skip-labeled`):
   - show title, company, source, and the **extracted requirements/responsibilities**
     (the structured fields — concise, no boilerplate);
   - if `--auto-prelabel`, run `analyze_fit` and show the seed grade (verdict→grade map, #9);
   - prompt `[0/1/2/3 / s=skip / q=quit]`, default = seed; persist to `EVAL_LABEL` with
     `labeled_by` and optional `notes`.
3. Flags: `--limit N`, `--skip-labeled`, `--auto-prelabel`.

**Target set size:** **50–80 graded pairs per profile, ≥ ~15 at grade ≥ 2.** Below ~30
relevant judgments per topic, metric variance is too high to trust deltas ([Voorhees & Harman]
on judgment depth; [Saad-Falcon et al.] find ~50/topic sufficient for stable variance).

**PII boundary:** labeling runs entirely locally; CV/profile text stays on the local model
(CLAUDE.md, §13/§15). Committed golden qrels store **only** `(profile_id, job_id, grade)` — no
job text, no PII.

### 4.4 `eval/run.py` — one config → all metrics

- Input: a named `config` dict (`{k, pool, weights, field_boosts}`) and optional `profile_id`.
- Steps: `load_qrels` → `build_run` → compute nDCG@10, MRR, P@5, P@10, Recall@20, Recall@50
  (via `ranx`, cross-checked against `metrics.py` in tests).
- Output: one JSON object `{config, ndcg@10, mrr, p@5, p@10, recall@20, recall@50, pool,
  num_labeled, run_at, git_sha}` to `eval/results/`.

**Report all three §12 configurations** in every run: hybrid, vector-only, keyword-only —
"reported honestly even where hybrid does not win on this corpus" (`PHASE_1_DESIGN.md` §M6).

### 4.5 `eval/sweep.py` — OAT tuning + comparison

Not a full grid (combinatorial blow-up). One-at-a-time, in impact order ([Cormack et al.] for
k-sensitivity; [Craswell et al.] for weight-tuning gains):

1. **RRF `k`** ∈ {20, 30, 40, 60, 90, 120} — highest-sensitivity RRF parameter ([Cormack et al.]
   chose 60 for *web*; small homogeneous corpora often prefer 20–40, [Bruch et al.]).
2. **Arm weights** (lexical / HyDE / CV) ∈ {[1,1,1], [2,1,1], [1,2,1], [1,1,2], [2,2,1], …} —
   **fixes the accidental 2:1 vector:lexical bias** flagged in the Part-1 assessment (two
   vector arms each get weight 1.0 today). Tuning weights yields +3–8 nDCG@10 over equal
   weighting in TREC DL ([Craswell et al.]).
3. **Pool** ∈ {20, 50, 100} — plot Recall@pool to find the elbow ([Robertson & Zaragoza] on
   retrieval depth).
4. **BM25 field boosts** — `title^{3,5,7} requirements^{1,2,3} responsibilities^{1,2}`.

`ranx.compare()` produces the side-by-side table with **paired significance tests**
(Fisher/Student) so we only claim a win when the delta clears noise ([Bassani]; [Smucker et al.]
on significance testing in IR).

### 4.6 CI regression gate — `tests/test_eval_gate.py` + workflow

- Commit `eval/golden/qrels.json` (the labeled set, IDs+grades only) and
  `eval/golden/result.json` (the locked best-config metrics).
- The test runs the golden config against the golden qrels and asserts
  `ndcg@10 >= golden.ndcg@10 − 0.02` (the [Craswell et al.] small-set noise band).
- Wire into `.github/workflows/` alongside ruff/pytest/gitleaks (§M6, CLAUDE.md). A retrieval
  regression fails the build — the "regression fails the build" §12 deliverable. Elastic's
  `_rank_eval` is the productized analogue.

---

## 5. How this tests and affects *our* service

| Harness output | What it changes in the codebase |
|----------------|----------------------------------|
| Recall@50 low | Raise `pool` in `run_fit_pipeline`, or fix an arm that's missing relevant jobs, **before** any ranking work. |
| Vector:lexical 2:1 bias confirmed harmful | Pass explicit `weights` into `reciprocal_rank_fusion` from `search()` ([fusion.py]). |
| Best `k` ≠ 60 | Change the `k` default in `reciprocal_rank_fusion` ([fusion.py]). |
| Best field boosts ≠ `5/2/1` | Update `_boosted_query` ([bm25.py]). |
| Hybrid ≤ BM25-only | Investigate the dense arms (HyDE prompt, CV single-vector); honest §12 reporting either way. |
| Golden gate | New CI check; every future retrieval PR must hold nDCG@10 within noise. |

The harness is **measurement, not behavior** — it imports `search()` and scores it; it does
not change runtime retrieval. The only production code it touches is the *result* of tuning:
the constants in `fusion.py` / `bm25.py` and the explicit `weights` wired through `search()`.

---

## 6. Build sequence (checklist)

1. `eval/metrics.py` + `test_eval_metrics.py` — auditable math, green.
2. `eval/qrels.py` + `test_eval_qrels.py` — DB ↔ primitive adapters, green.
3. `eval/label.py` — labeling CLI; **then label 50–80 pairs** for the real profile.
4. `eval/run.py` — measure the **baseline**: Recall@50 first, then nDCG@10, all three configs.
5. `eval/sweep.py` — OAT sweep `k` → weights → pool → boosts; pick the winner.
6. Lock the winning constants into `fusion.py` / `bm25.py` / `search()`.
7. Commit `eval/golden/*`; add `test_eval_gate.py`; wire the CI workflow.

---

## 7. Out of scope (deferred)

- **LLM-as-judge for draft quality** — Phase 2 (§12; `PHASE_1_DESIGN.md` §6 note). M6 uses the
  human-labeled set directly; the judge is not validated here.
- **Multi-profile / fairness slicing** — the schema (`profile_id`) supports it, but Phase 1
  has one real profile. ([Geyik et al.] fairness ranking is a Phase-2 extension.)
- **Learned fusion / LTR** — RRF stays; learning-to-rank weights is a future lever once the
  label set is large enough to train on ([Bruch et al.]).

---

## References

**Foundational IR evaluation**
- Järvelin & Kekäläinen, *Cumulated Gain-Based Evaluation of IR Techniques*, ACM TOIS 2002 —
  [DOI](https://dl.acm.org/doi/10.1145/582415.582418) (defines DCG/nDCG).
- Voorhees & Harman, *TREC: Experiment and Evaluation in Information Retrieval*, MIT Press 2005
  (the qrels/run/judgment-depth methodology).
- Burges et al., *Learning to Rank using Gradient Descent* / LambdaRank line —
  [arXiv context](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
  (exponential-gain nDCG).
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, FnTIR 2009 —
  [DOI](https://dl.acm.org/doi/10.1561/1500000019) (BM25; retrieval depth).
- Smucker, Allan & Carterette, *A Comparison of Statistical Significance Tests for IR
  Evaluation*, CIKM 2007 — [DOI](https://dl.acm.org/doi/10.1145/1321440.1321528).

**Hybrid search, fusion, benchmarks**
- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion outperforms Condorcet…*, SIGIR 2009 —
  [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF; k=60; sensitivity).
- Craswell, Mitra, Yilmaz, Campos, *Overview of the TREC 2020 Deep Learning Track*, 2021 —
  [arXiv:2102.07662](https://arxiv.org/abs/2102.07662) (nDCG@10 standard; weight tuning gains).
- Thakur, Reimers et al., *BEIR: A Heterogeneous Benchmark for Zero-Shot IR*, NeurIPS 2021 —
  [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) (BM25 vs dense by domain; Recall focus).
- Gao, Ma, Lin, Callan, *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE),
  SIGIR 2023 — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).
- Bruch, Gai, Ingber et al., *An Analysis of Fusion Functions for Hybrid Retrieval*, ACM TOIS
  2023 — [arXiv:2210.11934](https://arxiv.org/abs/2210.11934) (fusion/normalization; small-k).

**LLM-assisted labeling**
- Saad-Falcon et al., *ARES: An Automated Evaluation Framework for RAG*, NAACL 2024 —
  [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) (LLM-judge correlation; ~50/topic).
- Es et al., *RAGAS: Automated Evaluation of RAG*, EACL 2024 —
  [arXiv:2309.15217](https://arxiv.org/abs/2309.15217).
- Faggioli et al., *Perspectives on LLMs for Relevance Judgment*, ICTIR 2023 —
  [arXiv:2304.09161](https://arxiv.org/abs/2304.09161).

**Person–job fit / fairness**
- Geyik, Ambler, Kenthapadi, *Fairness-Aware Ranking in Search & Recommendation* (LinkedIn
  Talent Search), KDD 2019 — [arXiv:1905.01989](https://arxiv.org/abs/1905.01989).

**Tooling & production references**
- Bassani, *ranx: A Blazing-Fast Python Library for Ranking Evaluation and Fusion*, ECIR 2022 —
  [GitHub](https://github.com/AmenRa/ranx).
- `pytrec_eval` (Van Gysel & de Rijke) — [GitHub](https://github.com/cvangysel/pytrec_eval).
- Elastic, *Ranking Evaluation API (`_rank_eval`)* —
  [docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-rank-eval.html).
- Pinecone, *Evaluating Retrieval / Recall@k first* —
  [guide](https://www.pinecone.io/learn/offline-evaluation/).
