# Embedding Eval — Independent Review Report

Reviewed: `EMBEDDING_EVAL_DESIGN.md` and `EMBEDDING_EVAL_IMPLEMENTATION_PLAN.md` (2026-09-18), against
the real code, the live (read-only) database and local Ollama 0.30.11. Every finding is labeled
**VERIFIED** (read or run — evidence given), **PLAUSIBLE** (reasoned, not run) or **UNVERIFIED**.
No number in this report is invented; experiment numbers are from the runs in §10. Literature
citations are from memory, not fetched.

How this report was used: its findings are folded into the revised design and plan (each change
cites its ID, e.g. *R-M2*).

---

## 1. Verdict

**GO-WITH-CHANGES.** Nothing would touch production and nothing is unbuildable; the factual base
is unusually sound (16 of 22 facts reproduced exactly). But three things must change before the
Stage-1 *output* can be trusted, and one spec is factually wrong:

1. **The comparison output can mislead.** The raw partial `human` view rewards *coverage by the
   labeled set*, not quality — deliberately degraded embedders outscored the incumbent on it
   (R-M1). Recall@100 is ceiling-bound at ≈0.15–0.23 because roughly 430–670 of the 1,395 pool
   documents are grade ≥ 2, not 100–200 (R-M2). The only complete view (`silver`) rests on a judge
   that scored QWK 0.41 in a single-shot probe, and `evaluate` does not require the calibration
   check (R-M3, R-M4).
2. **Ollama's context model is not what the plan assumes.** `nomic-embed-text` and `bge-m3` clamp
   input at 2,048 tokens whatever `num_ctx` says; the plan's truncation detector
   (`prompt_eval_count >= num_ctx`) can never fire (R-M7).
3. **The isolation and PII guarantees are overstated.** The AST test misses transitive imports
   (R-M5); `cv_text` contains the candidate's name, email, phone and a URL despite the "no name,
   email or links" claim (R-M6).

Also: the baseline suite is already red (1 failure) and local `pytest` writes to whatever
`DATABASE_URL` points at (R-M8). No blockers.

---

## 2. Findings

Severity: BLOCKER (would not work / touch prod / invalid conclusions) · MAJOR (likely rework, wrong
result under plausible conditions, materially misleading claim) · MINOR · NIT.
Categories A–H as in the review brief.

### MAJOR

**R-M1 — The raw partial `human` view is a coverage artifact.** (C, VERIFIED)
Evidence (Appendix E6, real vectors for all six candidates + degraded variants, 212 imported
labels, unlabeled = grade 0 as `eval/metrics.py` does):

| variant | nDCG@100 | Recall@100 | judged@100 |
|---|---|---|---|
| nomic-v1.5 (incumbent) | 0.199 | 0.194 | 0.26 |
| **DEG** nomic docs without prefix | **0.261** | **0.250** | 0.34 |
| **DEG** nomic, no prefix anywhere | 0.230 | 0.241 | 0.33 |
| bge-m3 | 0.235 | 0.204 | 0.28 |

The degraded variants "win" because they surface more of the 212 already-labeled documents (only
26–34% of any top-100 is judged). In the *condensed* view (212 docs only) MRL-128 nDCG@10 = 0.695
vs incumbent 0.530. Only "tail shuffled" and "random" separate clearly.
Impact: printing raw partial-view metrics invites a wrong conclusion.
Fix: partial views report **only** condensed metrics, bpref and judged@k; never raw nDCG/Recall.
(Plan WP10.)

**R-M2 — Recall@100 has a low ceiling; it is the wrong headline.** (C, VERIFIED-indicative)
A 55-job random sample of the pool judged by Gemma3:12b (few-shot) gave 47% grade ≥ 2 (95% CI ±13);
among the 212 human-labeled jobs the human rate is 51% (108/212). Estimated R ≈ 660 (475–844) from
the over-rating judge; correcting for judge precision 0.66 gives ≈ 430. Either way R ≫ 100, so
Recall@100 ≤ 100/R ≈ 0.15–0.23. (The eligible pool is *eligible*, not *relevant*: "Software Engineer /
Backend Developer" is broad.)
Fix: headline = **nDCG@100, P@100** ("what share of the 100 candidates handed to fit analysis is
relevant"), **AP**; Recall@100 secondary, always printed with its ceiling `min(1, 100/R)`; R shown per
view. (Plan WP10.)

**R-M3 — The judge is unproven and its job view is under-specified.** (C, VERIFIED-indicative)
Single-shot few-shot prompt, 400-char label view, 45 human-labeled non-exemplar jobs: QWK **0.413**,
exact 35.6%, within-1 80%, binary(≥2) precision 0.66 / recall 0.88; 8 of 9 human-grade-3 jobs were
graded 2 (top compressed); 6 of 13 human-0 jobs graded 2. Schema adherence 100/100. This is *below*
the plan's proposed κ_w ≥ 0.6 gate, as expected before iteration — the gate is right; the risk is
that nothing enforces it (R-M4).
Job view: `label.py::_show_job` shows title, company, **source**, seniority and requirements /
responsibilities **clipped to 400 chars each** (description only if both empty) — **not location**
(design §2.4 says location) and **not the full extracted text**: in the pool 1,176 of 1,378
extracted-branch documents (85.3%) have a field > 400 chars; the median labeler saw 59% of the
extracted text. The fit judge (`fit/analyze.py::_posting_body`) and the embedders see it
**unclipped**. WP9 says only "fixed truncation limits held as constants".
Fix: define the judge's job section as a byte-for-byte mirror of `_show_job` (400-char clips, no
location) so calibration compares like with like; state that all three "production views" differ in
clipping; a full-text view is the Stage-2 diagnostic. (Design §2.4, WP9.)

**R-M4 — `evaluate` does not require the judge check; `resolve_effective` is under-pinned.** (B/E, VERIFIED by reading spec + PG)
`REQUIRED_CHECKS = (parity_ranking, parity_reconstruction)` — `judge_calibration` is absent, so
silver/effective tables print without a calibrated judge. `resolve_effective(...)` has an *optional*
judge-run pin: after a spike run (`--limit 20`) then a full run, or a re-run with a new prompt,
"latest row wins" silently mixes prompt versions. `created_at` defaults to `now()`, which is
constant within a transaction (VERIFIED: `now()` equal across statements 0.2 s apart) → ties inside
a batch. Mixed-source `effective` also has a heteroscedastic artifact: the judge over-rates
(binary ≥2 rate 0.71 vs human 0.53 on the same 45 jobs), so human-labeled documents are graded
lower than equivalent silver-only ones.
Fix: judge-based views **require** a `judge_run_id` and a passed `judge_calibration` check for that
run (else stamped `UNCALIBRATED`); label rows get a `BIGINT IDENTITY` id and "latest" = max id;
document the `effective` artifact and make `silver` the headline view.

**R-M5 — The import-isolation test does not isolate.** (B, VERIFIED)
Prototype: a non-allow-listed `runner.py` importing an allow-listed `parity.py` passes the AST test
(`violations: []`), as does `importlib.import_module("job_radar…")`; both fail at runtime with
`ValidationError: 4 validation errors for Settings`. Importing `job_radar.retrieval.vector` pulls
`job_radar.config` (requires `DATABASE_URL`, `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`,
`GENERATION_MODEL`) and `job_radar.db.base`, which builds a read-write `AsyncEngine` at import (lazy
connect, but present). So `import-topic` and `verify` need the four prod variables, not just
`database_url`; the read-only story holds only because nothing uses `async_session_factory`.
Fix: keep the AST rule (cheap, catches direct imports) **and** add a subprocess check: import every
non-allow-listed module in one clean interpreter with the prod variables stripped and assert no
`job_radar*`/`langfuse` in `sys.modules`; allow-listed modules may import only
`job_radar.db.models`, `job_radar.retrieval.{vector,filters}`, `job_radar.ingest.embed_text`;
`cli.py` imports them lazily. (Plan WP11 → now WP2.)

**R-M6 — `cv_text` contains PII; the "no name, email or links" claim is false.** (D/G, VERIFIED, counts only)
`profile.cv_text` (4,454 chars): contains an email-like token (equal to `profile.email`), the full
name, a URL token and a phone-like number. The design (§2.3) and plan (WP6.4) say the snapshot keeps
`cv_text` with "no name, email or links"; the planned test only checks *keys*.
Fix: `assemble()` scrubs email, phone, URLs and the profile's name tokens from `cv_text` (name used
only to redact, never stored); the test asserts *values*. The EVALS DB stays local (gitignored);
say "best-effort scrubbed" not "no PII". (WP6.)

**R-M7 — Ollama's effective context is not `num_ctx`; the truncation detector is wrong.** (B/C, VERIFIED)
Ollama 0.30.11, `/api/embed`, `num_ctx` ∈ {2048, 8192, 32768}:
- `nomic-embed-text` (GGUF context 2048; `/api/ps` shows loaded ctx 2048): `prompt_eval_count`
  saturates at **2048** for every setting.
- `bge-m3` (GGUF context 8192, loaded ctx 8192): `prompt_eval_count` saturates at **2048** for every
  setting — an Ollama cap, not the model's.
- `qwen3-embedding:0.6b`: honors 8192 (2,661 tokens processed); with no `num_ctx` it stops at 4,095.
- `"truncate": false` → HTTP 400 `the input length exceeds the context length` for all three — the
  correct, model-independent detector.
So the design's "identical 8192 → truncation isn't a confound" is false, `prompt_eval_count ≥ 8192`
would report **0** truncated documents for nomic/bge-m3 (truth on the pool: 1 document each; 0 for
Qwen3), and production's own comment ("raise num_ctx or Ollama truncates at ~2048") is wrong for this
model — prod also truncates at 2048 (1 pool document; ≤ ~1% affected in any case: only ~16 pool
documents would exceed 2,048 tokens even as full description, estimated at 5.13 chars/token).
Also observed: switching `num_ctx` on a resident runner returned HTTP 400 `EOF` (2048→8192, →32768,
→16384) — not retried by the planned client (timeouts/5xx only).
Fix: embed with `truncate:false`; on the 400, retry with truncation and store `truncated = true` on
the cached vector; report truncated-document counts from that flag; describe effective ceilings
honestly. (Design §3.3, WP7.)

**R-M8 — The baseline is red, and local `pytest` writes to the dev database.** (G, VERIFIED)
Against a freshly migrated *empty* database (CI's situation) with the repo's fixture seeding:
`322 passed, 1 failed` — `tests/test_eval_gate.py::test_ndcg_meets_golden_threshold`
(nDCG@10 0.5596 < 0.8661). `docs/STATUS.md` records it as pre-existing. So "existing suite passes
unchanged" (WP1 gate) and "every commit CI-green" (§2) are unattainable as written (CI status on
`main` itself: UNVERIFIED — no `gh`). Separately, `tests/test_bm25.py` inserts/deletes `Job` rows and
the golden test seeds fixture rows through `db_session`, i.e. **into `DATABASE_URL`**; the plan's
"never write prod, in code or tests" does not hold for the existing suite when run locally.
Fix: gates read "no new failures vs baseline (322 passed, 1 known failure)"; new prod-facing tests
run against the throwaway database (same guard mechanism, no real-prod risk); note the local-pytest
hazard. (Plan §2, WP0.)

**R-M9 — No inference in Stage 1, though the minimum detectable difference is measurable.** (C, VERIFIED)
Paired bootstrap over the 212 labeled documents (condensed view, B = 2000): SD of the difference vs
incumbent — AP 0.034–0.040, nDCG@50 0.041–0.057, P@50 0.061–0.068 (95% CI half-width ≈ 2 SD:
**≈ 0.07–0.08 AP, ≈ 0.11 nDCG@50, ≈ 0.13 P@50**). Observed gaps are inside it (bge-m3 +0.053 AP,
CI [−0.011, +0.121]; Qwen3 +0.034). Only MRL-128 (−0.056, CI [−0.101, −0.013]) separates. A
document bootstrap over 1,395 silver-labeled documents will be ≈ 2.6× tighter but does not capture
judge error or profile variation.
Query sensitivity (free, from the stored HyDE texts): each of the 3 texts alone gives top-100
overlap 87–96 with the mean, and the embedder order (bge-m3 > qwen3-inst > nomic, condensed AP) is
identical for all four queries — the *3-text* variation is harmless, but it says nothing about a
different query **recipe** (F22 stands).
Fix: Stage 1 ships `evaluate --bootstrap N` (paired document bootstrap vs incumbent, seeded) and a
per-text query-sensitivity table; a second *recipe* (e.g. titles+stack) is a method-config change
over the same topic (labels are per topic/job), so the design's "shared labels across topics" schema
decision is unnecessary. (WP10.)

**R-M10 — The imported "human" labels are LLM-made.** (C, MAJOR, VERIFIED by user statement 2026-09-19 + the calibration run)
Evidence: the 221 prod `eval_labels` (212 in Tier A's pool) all read `labeled_by = 'human'`, but the
user states the fit LLM produced them; the column never distinguished the two (F11: `--labeled-by`
defaults to `human` in every mode). A calibration of the Gemma judge against them over 44 pairs gave
κ_w 0.481, exact 0.50, within-1 0.795, binary (≥ 2) precision 0.76 / recall 0.73: two LLMs agreeing
moderately, not a judge measured against a person.
Consequences: (1) the judge calibration was not against humans, so the κ_w gate meant nothing about
human agreement; (2) the `human` view was not human, and every "human-labeled" number derived from it
(R-M1, R-M9, F10, F24, F26) describes agreement with an LLM; (3) the labels are circular with the
qwen2.5 HyDE generator's family of judgments, biasing the comparison toward what that pipeline finds
relevant. Those measurements stay in the record with that caveat, not deleted.
Resolution: the labels are **dropped** (Tier A imports none; `imported_prod` is removed from
`labels.py`, the views, the tier builder and the tests) and a **`label-blind`** command is added: the
user grades ~50 jobs (~30 pooled + ~20 random) seeing only the 400-char label view, and those
`human_blind` labels are the only gold — for the `human` view, the judge's few-shot exemplars (4, one
per grade) and its calibration (n ≈ 46 non-exemplar labels, κ_w SE ≈ 0.1, so the gate is coarse).
`evaluate --view human` now refuses with "no human labels yet — run `label-blind`" until they exist.

### MINOR

- **R-m1 — Alembic template lacks the pgvector import.** (B, VERIFIED) Autogenerating
  `Vector()` (no dimension) works and `alembic check` is **clean** — but the generated migration
  uses `pgvector.sqlalchemy.vector.VECTOR()` without importing it: `NameError: name 'pgvector' is not
  defined` at `upgrade` (prod's `4a1001525d0a` has a hand-added import). Add
  `import pgvector.sqlalchemy` to `alembic_evals/script.py.mako`. Worker-thread `alembic upgrade`
  inside a running loop: works (0.06 s).
- **R-m2 — `warm()` and tag matching.** (B, VERIFIED) `/api/generate {"model": embedder}` (prod's
  `warm` mechanism) returns **400** for embedding models — warm an embedder with a real `/api/embed`
  call. `/api/tags` names carry `:latest` (`bge-m3:latest`, `nomic-embed-text:latest`) — `ready()`
  must normalize. `/api/tags` exposes `digest`, `details.quantization_level` and
  `details.context_length` (the latter is *not* the effective cap, R-M7).
- **R-m3 — `parity_ranking` tie groups are undefined.** (B, VERIFIED) Real run (prod's own
  `search_vector` over the frozen ids, read-only, vs numpy over stored vectors): top-100 **sets
  equal**, exact order differs at 4 positions (28, 29, 76, 78) — the byte-identical-vector ties;
  max |Δscore| 2.4e-7; 8 adjacent pairs closer than 1e-5, 4 of them byte-identical, none
  non-identical below 1e-6. F6 ties never reach the rank-100 boundary for this query (gap to rank 101
  is 1.0e-4). Spec `compare_rankings` as: position-wise sorted scores within `eps`, then id sets equal
  per *eps-chain* score group, except the group crossing rank `k`.
- **R-m4 — The read-only guard is accident-proof, not adversary-proof.** (B/D, VERIFIED)
  `-c default_transaction_read_only=on` blocked CREATE/INSERT/UPDATE/DELETE/TRUNCATE and an ORM
  flush (`ReadOnlySqlTransaction`), survived `pool_pre_ping` reconnect after the backend was
  terminated (fresh connection re-applied the option). But `SET default_transaction_read_only=off`
  **persisted across pool checkouts** (next `SHOW` = `off`) and `SET TRANSACTION READ WRITE` as the
  first statement succeeded (also inside `postgresql_readonly=True`/`BEGIN READ ONLY` — that is not
  stronger). The entry `SHOW` assertion catches a *stale* override on checkout. Use `NullPool` (no
  state leaks between `prod_session()`s), keep the `SHOW`, and reword "provably untouched" →
  "untouched (before/after counts)". A read-only role would be stronger but the design forbids
  creating roles.
- **R-m5 — Tooling/CI details.** (G, PLAUSIBLE) Adding `numpy` to the `dev` group changes the
  project's `uv.lock` entry; CI runs `uv sync --locked`, so `uv lock` must be committed in the same
  commit. The console-script line for `job-radar-eval-embedding` is listed under WP2 as "(WP10)" but
  the CLI module is created in WP7. The CI step `job-radar-evals-db init` is dead weight when tests
  use throwaway databases, and `EVALS_DATABASE_URL` is unnecessary in CI if the fixture derives the
  server from `DATABASE_URL` (superuser `job_radar`: `rolsuper`/`rolcreatedb` true here; same image
  and `POSTGRES_USER` in CI). Net: **no `ci.yml` change needed**.
- **R-m6 — Promised but unscheduled.** (E, VERIFIED by reading) Design §2.1 makes `labels export` /
  `import` "the backup path"; no work package builds it. The 212 human labels currently live only in
  prod `eval_labels` (there is a `TRUNCATE`-ing script, `scripts/reset_eval_labels.py`) — the import
  is their second copy, so an export path matters.
- **R-m7 — Schema nits.** (D, VERIFIED by reading) `text_sha` names two different things (`sha256` of
  `embed_text` on `embedding_job`; `sha256(prefix + text)` on `embedding_vector`) — rename the job's
  to `embed_text_sha`. `embedding_run_ranking` needs a unique `(run_id, job_id)`, not only
  `(run_id, rank)`. `embedding_vector` needs `truncated` (R-M7).
- **R-m8 — Over-engineering under CLAUDE.md.** (F) See §7.
- **R-m9 — Stale/incorrect facts.** (E, VERIFIED) F5 "exactly 1.0000": true for nomic (bit-identical
  on repeat), **false for Qwen3** (cos 0.99998939) and bge-m3 (0.99999915). F19 "only 89 overlap the
  human labels": 89 is *rows*; 62 distinct jobs. Design §0 "records 62 labels": `eval/golden/qrels.json`
  holds **122**; `result.json` records no label count. Design §2.4 lists *location* among the labeled
  view, contradicting F21/`label.py` (source, no location). Plan header says "F1…F19" but the design
  has F1…F22 and the plan cites F20/F21. Design §4 layout omits `judge/calibration.py`, `doc_text.py`.
- **R-m10 — Nomic Matryoshka.** (B, PLAUSIBLE→measured small) nomic's model card applies
  `layer_norm` before slicing; the plan's `apply_dim` slices raw output. On this data layernorm+slice
  vs slice-only at 256 dims differed by ≤0.02 on condensed P@50 and 0 elsewhere. No nomic `dim`
  variant is scheduled, so note-only.

- **R-m11 — Qwen3 on Ollama crashes on very long inputs (found while verifying agent C's client).**
  (B, VERIFIED) `/api/embed` with `qwen3-embedding:0.6b`, `num_ctx` 8192, `"word " * 3000` (and
  `* 6000`), with or without `truncate` → HTTP 400 `do embedding request: … /v1/embeddings: EOF`
  (the runner dies), deterministically; natural text of 2,661 tokens embedded fine earlier and every
  one of the 1,395 pool documents (max 2,024 Qwen3 tokens) embeds without error. Also: the first
  Qwen3 call after a model load returned a vector differing from later calls at the 4th digit
  (cos 0.99998939 — the F5 non-determinism). Impact: Stage 1's production-text runs are safe; a future
  full-text representation or a long fallback document (up to ~3k tokens) may hit it, and a
  deterministic EOF is not fixed by the client's 3 retries — a persistent failure must surface as a
  failed run, not a silent gap (it does: the error propagates).

### Resolved (the author's open risks)

- **Qwen3 end-of-text (author weak spot g): RESOLVED — Ollama matches the reference.** (VERIFIED)
  30 texts (27 extracted + 3 fallback), fp32 HF `Qwen/Qwen3-Embedding-0.6B` with `<|endoftext|>`
  appended (the tokenizer does so by default, id 151643): Ollama cosine min **0.99938**, mean
  0.99956; against a no-EOS reference mean 0.66. The diagnostic can shrink to a one-off note.
- **Unconstrained `vector` + Alembic (e): works** modulo R-m1. 3-d and 1024-d vectors round-trip
  exactly (float32, `np.array_equal`) in one column; ORM returns `ndarray[float32]`, raw
  `text()` returns `str` (use the ORM column or `register_vector`); cross-dimension `<=>` errors
  (`different vector dimensions`) — irrelevant because ranking is in numpy.
- **Prefix caching (judge): true in Ollama, sequentially.** (VERIFIED) 2.6k-token few-shot prefix:
  first call `prompt_eval_duration` 3.35 s, next calls 0.24–0.45 s; identical repeat 0.04 s.
  `prompt_eval_count` still reports the full count — use `prompt_eval_duration`. Concurrency-4
  steady state: **UNVERIFIED** (my 8-job probe, 0.26 jobs/s, was dominated by cold slots; the larger
  sweep was not run).

---

## 3. Fact-check table (F1–F22)

| # | Verdict | Evidence |
|---|---|---|
| F1 | Confirmed | Real filtered query and `enable_seqscan=off` both `Seq Scan` + `Sort` (16,411 est. rows) |
| F2 | Confirmed | All 1,395 pool documents re-embedded with the prefix: min cosine 1.000000 to stored |
| F3 | Confirmed | Whole pool, both branches, incl. the 8 longest fallbacks (12 k chars) and earliest-ingest rows: 0 misses < 0.9999 |
| F4 | Confirmed | Stored norms 0.999999–1.000001; fresh 1.0 |
| F5 | Partly contradicted | nomic bit-identical on repeat; Qwen3 0.99998939, bge-m3 0.99999915 (R-m9) |
| F6 | Confirmed | 788 clusters / 2,327 rows; pool: 14 / 29; no tie at rank 100 (gap 1.0e-4); 4 identical vectors inside top-100 |
| F7 | Confirmed | 1,145 clusters / 3,337 rows |
| F8 | Confirmed | 108.8 docs/s at concurrency 8 (bge-m3 40.8, Qwen3 30.1) |
| F9 | Confirmed | Sizes/quantization match; note Qwen3 is **Q8_0**, nomic and bge-m3 **F16** (a confound to report) |
| F10 | Confirmed exactly | pool 1,395; 212/221 in pool (0:56 1:48 2:72 3:36); 9 outside |
| F11 | Confirmed | 221 labels all `human`; `--labeled-by` default `"human"` (`label.py:411`) |
| F12 | Partly verified | `providers.py` hardcodes model/`num_ctx` (read); the `No active span` log line not reproduced |
| F13 | Confirmed | Importing `job_radar.db.base` builds the engine; `config` needs 4 required vars |
| F14 | Confirmed | `quality/cli.py:136`, `relevance.py` `DEFAULT_THRESHOLD = 0.5`, `profile/loader.py:62` |
| F15 | Confirmed | `bm25.py` has no `embed` reference |
| F16 | Confirmed, with limits | R-m4 |
| F17 | Confirmed locally | `job_radar` is superuser+createdb; `CREATE DATABASE`/`CREATE EXTENSION vector` worked; CI same image (UNVERIFIED there) |
| F18 | Partly verified | 100/100 schema-valid; **2.55 s/job median sequential with few-shot** (n=100); concurrency-4 0.55 jobs/s UNVERIFIED |
| F19 | Confirmed, corrected | 428 judgments; overlap is 89 *rows* = 62 distinct jobs |
| F20 | Confirmed exactly | 1,378 / 17; median 3,017 → 1,239; ratio 0.45; 522 (37.4%) > 4,000; 186 (13.3%) one field; corpus 57.6% |
| F21 | Confirmed, sharpened | `label.py` clips to 400 (85% of pool docs clipped); fit stage reads unclipped (R-M3) |
| F22 | Confirmed exactly | 157 shared chars; text sim 0.46/0.49/0.79; cos 0.9609–0.9863; to mean 0.9872–0.9957 |

---

## 4. Work-package verdicts

| WP | Verdict | Why |
|---|---|---|
| WP0 | GO-WITH-CHANGES | Add baseline note (R-M8); Qwen3 diagnostic already answered |
| WP1 | GO | `build_embed_text` reproduces prod exactly (1,395/1,395); note `filter(None,…)` also drops an empty title; gate = "no new failures" |
| WP2 | GO-WITH-CHANGES | mako import (R-m1), no CI change (R-m5), tests use throwaway DB; verified worker-thread Alembic |
| WP3 | GO-WITH-CHANGES | Test the guard against the throwaway DB, `NullPool`, wording (R-m4) |
| WP4 | GO-WITH-CHANGES | Cut `basis`/`selection`/`role`/`method_kind`; add `truncated`, identity label id, unique ranking key (R-m7, §7) |
| WP5 | GO-WITH-CHANGES | Mandatory judge-run pin; drop `basis` (R-M4) |
| WP6 | GO-WITH-CHANGES | PII scrub (R-M6); persist path fine (statement/param sizes OK) |
| WP7 | GO-WITH-CHANGES | `truncate:false` detector, warm/tags (R-M7, R-m2); cut `doc_text` registry |
| WP8 | GO-WITH-CHANGES | Tie-group spec (R-m3) |
| WP9 | GO-WITH-CHANGES | Mirror the label view, measure concurrency at the gate, calibration gate enforced (R-M3/M4) |
| WP10 | GO-WITH-CHANGES | Headline metrics, condensed-only partial views, bootstrap, query sensitivity (R-M1/M2/M9) |
| WP11 | GO-WITH-CHANGES | Isolation test upgraded and moved earlier (R-M5); export/import (R-m6) |

---

## 5. Methodology assessment

**Will Stage 1 support an embedder comparison?** Partly. With complete silver labels, a calibrated
judge and the additions above it produces a defensible *ranked shortlist with intervals* — not a
decision. What it cannot tell the developer: (i) whether the silver ordering survives judge error
(needs the Stage-2 human buckets); (ii) anything about other profiles or another query recipe;
(iii) whether the winner survives BM25 fusion (cutover check, correctly deferred); (iv) which
embedder is best on text the extractor never saw (needs a `full`-view label set).

**Biggest gap and the minimum fix.** The judge. Iterate the prompt on the dev half (my single-shot
0.41 says expect several rounds; ideas: keep the 3-vs-2 boundary explicit in the rubric, reason
after evidence, add 1–2 more grade-3 exemplars), enforce κ_w ≥ 0.6 as a *hard* prerequisite for
the silver view, and if it cannot be met with Gemma3:12b use the paid `Judge` behind the same
protocol for the ~1,395 calls. In parallel, the cheapest human effort is *pooled*: the union of the
six variants' top-30 is **73 documents** (only 4 in all six), and top-100 sets overlap only ~50%
between nomic and Qwen3/bge-m3 — labeling that disagreement region blind is ~1 hour and directly
decides the ranking.

**Other checks.** The imported labels are not visibly incumbent-biased for a *dense-only*
ranking (judged@100: nomic 0.26, Qwen3-instruct 0.32, bge 0.28) — they came from three hybrid
configs incl. BM25 — but their 73% Himalayas share and clipped view still make the calibration
optimistic at the top. With 212 labels, 8 exemplars and a dev/test split leave ≈ 100 per half:
κ_w has SE ≈ 0.08–0.1, so "≥ 0.6" is a coarse gate. QWK is a sound choice for ordinal grades;
report binary (≥ 2) precision/recall alongside because prevalence is near 50% (not skewed).

---

## 6. Best-practice gaps (citations from memory)

- **Pooling and reusability** — TREC pooling (Voorhees & Harman); Zobel (SIGIR 1998) on pool bias;
  Buckley & Voorhees (SIGIR 2004) bpref; Carterette et al. minimal test collections. The plan's
  complete-label design is *stronger* than pooling for this pool size. Departure justified.
- **LLM-as-judge** — Thomas et al. 2023 and Faggioli et al. 2023 (agreement is task/prompt
  dependent; over-rating); Clarke & Dietz (avoid judge/system circularity — the Gemma-vs-qwen2.5
  split is good). Angelopoulos et al. 2023 (prediction-powered inference) needs a *random* human
  sample and ~dozens of queries; the plan borrows only the idea — honest.
- **Statistical testing in IR** — Smucker/Allan/Carterette (CIKM 2007) and Sakai on paired
  bootstrap / randomization tests; report intervals, not only point estimates (R-M9). Multiple-
  comparison correction (Carterette 2012) once >2 variants are tested against one incumbent.
- **Benchmarks** — BEIR (Thakur et al. 2021) and MTEB (Muennighoff et al. 2022) treat *nDCG@10 on
  many queries* as headline; a single-profile, deep-cutoff setting justifies nDCG@100/P@100 but not a
  headline Recall@100 when R > 100.
- **Reproducibility/immutability** — content-addressed IDs and immutable runs (the `uuid5`/`text_sha`
  scheme) match `ir_datasets`/`ranx` run-file practice; append-only labels are right.
- **Schema** — `TEXT` + code validation for open vocabularies and a `CHECK` on `grade` is sound.
  Store `truncated`/`n_tokens` per vector; give append-only tables an ordered surrogate key.
- **Testing** — fakes + throwaway-DB integration + contract tests is good practice; add the
  subprocess import test (R-M5) and a value-level PII test (R-M6).

---

## 7. Cut / defer and missing lists

**Cut or defer (CLAUDE.md "no speculative abstraction"):**
`embedding_label.basis` and `.selection` (Stage 2 adds them by migration — a nullable column is one
line); `embedding_topic_job.role`; `embedding_run.method_kind` (derivable from `method_config`);
`embedding_topic.builder_version`; `tiers/registry.py` and `embedders/registry.py` dispatch tables
(one implementation each — lazy import in the CLI is the isolation mechanism); `doc_text.py` and its
registry (Stage 2, after a `full` label set exists); `basis` param of `resolve_effective`; the CI
`evals-db init` step and `EVALS_DATABASE_URL` in CI; R-precision (AP covers deep ranks).
**Keep despite CLAUDE.md** (the developer asked for extensibility): the `TopicBuilder`, `Embedder`,
`RankingMethod` and `Judge` protocols and the two contract tests — each is ≤ 10 lines and has a fake.

**Missing:** `labels export`/`import` (R-m6); a `judge_calibration` requirement (R-M4); the
`truncated` flag (R-M7); per-text query sensitivity and bootstrap (R-M9); subprocess isolation test
(R-M5); PII value test (R-M6); `uv lock` in the numpy commit (R-m5); baseline note (R-M8).

**Four extensions, mentally implemented:** (1) *Tier B closed corpora* — `persist_topic` + a builder +
`constructed` labels; leaks: `embedding_topic_job` already carries membership (fine), but the
`prod_embedding` column and parity checks are prod-only (nullable, ok). (2) *Hybrid/reranker* — a new
`RankingMethod`; leaks: `evaluate` groups by embedder name — must group by method fingerprint (WP10
already keys on it). (3) *Paid-API judge* — a `Judge` impl; leaks: the client is Ollama-typed
(`ChatClient` protocol in the plan fixes it) and cost/rate-limit handling. (4) *Second eval family* —
new `<family>_*` tables and migration; leaks: `alembic_evals/env.py` imports
`eval.embedding.models` explicitly — needs an aggregation import in `evals_db`. All four are additions
without touching the core, as claimed.

---

## 8. Open questions for the developer

1. Judge fallback: if Gemma3:12b cannot reach κ_w ≥ 0.6 after ~3 prompt rounds, is a paid model for
   the 1,395 calls acceptable (about the same tokens as the Stage-2 use)?
2. Should the judge see the 400-char *label view* (comparable to your 212 labels; recommended) or the
   full extracted text (closer to what embedders see; needs a Stage-2 diagnostic)?
3. Is the ~1 h blind pooled labeling of the 73-document top-30 disagreement region acceptable *inside*
   Stage 1 (it makes the comparison decision-grade sooner) or strictly Stage 2?
4. Keep the six-variant list, or drop `qwen3-0.6b-instruct-d768` (near-duplicate of `-instruct`, 89/100
   top-100 overlap)?

---

## 9. Do-before-starting checklist (all folded into the revised plan)

1. Baseline note and "no new failures" gates (R-M8). 2. mako import (R-m1). 3. `truncate:false`
detector and effective-ceiling wording (R-M7). 4. Headline metrics + condensed-only partial views
(R-M1/M2). 5. Judge-run pin, calibration requirement, label-view mirror (R-M3/M4). 6. PII scrub
+ value test (R-M6). 7. Subprocess isolation test (R-M5). 8. Bootstrap + query sensitivity (R-M9).
9. Cut list (§7). 10. `parity` tie-group spec (R-m3).

**Pre-mortem — if Stage 1 ran exactly as written and the output misled:** (1) coverage artifact in the
raw `human` view (R-M1) — symptom: degraded variants outrank the incumbent; guard: sanity-check with
the `DEG` variants shipped as a built-in *negative control* in `evaluate`; (2) uncalibrated silver
view (R-M3/M4) — symptom: κ_w < 0.6 or missing; guard: hard requirement; (3) Recall@100 read as
quality (R-M2) — symptom: all embedders ≈ 0.2; guard: ceiling printed; (4) judge prompt-version mix
(R-M4); (5) one query recipe (R-M9); (6) silent truncation counts of 0 (R-M7); (7) quantization
(Qwen3 Q8_0 vs F16) read as a model difference — guard: recorded per run; (8) repost clusters
inflating ties (F7) — guard: de-dup slice; (9) `effective` heteroscedasticity (R-M4); (10) PII copied
into an export (R-M6).

Opinions (not findings): keep the six-variant set small; the "negative control" idea (degraded
variants as sanity checks) is cheap and would have caught R-M1; consider dropping the `d768` variant.

---

## 10. Appendix — experiments run (all on 2026-09-18; scratch scripts outside the repo)

| # | What | Result |
|---|---|---|
| E1 | Throwaway DB `job_radar_evals_test_<hex>`: create as `job_radar` (AUTOCOMMIT), `CREATE EXTENSION vector`, unconstrained `vector`, Alembic env mirroring the plan, autogenerate, `alembic check`, upgrade from a worker thread inside a running loop, drop | `NameError` until mako imports `pgvector`; then upgrade OK (0.06 s), `alembic check` clean, 3-d + 1024-d round-trip exact, DB dropped (verified `pg_database`) |
| E2 | Read-only guard attacks (§R-m4) | writes blocked; reconnect re-applies; `SET` override persists in pool; `SET TRANSACTION READ WRITE` succeeds; prod `eval_labels` = 221, no `_ro_probe*` tables afterwards |
| E3 | Import graph / AST prototype (§R-M5) | 10 `job_radar` modules pulled by `retrieval.vector`+`filters`; RW engine at import; AST test passes on transitive+dynamic leaks; stripped-env import fails with `ValidationError` |
| E4 | Embedders via Ollama (§R-M7): dims 768/1024/1024, unit norm, prefix sensitivity (Qwen3 prefixed vs not cos 0.985), throughput 108.8 / 40.8 / 30.1 docs/s, `prompt_eval_count` ceilings, `truncate:false`; Qwen3 vs HF reference | see findings |
| E5 | Gemma3:12b sample: 100 jobs (45 labeled non-exemplar, 55 random unlabeled), few-shot 8 exemplars, 400-char view | 100/100 valid; 2.55 s median; QWK 0.413; est. R (§R-M2) |
| E6 | Metric discrimination (§R-M1, R-M9): 6 variants + 8 degraded/control variants on stored/real vectors, 212 labels, bootstrap B = 2000 | tables in findings |
| E7 | `resolve_effective` edge cases | by reading + `now()` probe (R-M4); not run as a script |
| E8 | Baseline `ruff check` (clean, 123 files formatted) and `pytest -q` on a throwaway migrated DB | 322 passed, 1 failed (golden gate) |

Not run: full-pool Gemma judging (one 8-job concurrency probe only), CI itself, Tier-B persona pools
(F10's 4,534–10,477 figures: UNVERIFIED), a full 24-job concurrency sweep (declined by the user).
