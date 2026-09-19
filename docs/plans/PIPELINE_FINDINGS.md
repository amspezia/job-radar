# Pipeline findings beyond the embedder choice

Status: 2026-09-19, from the embedding-eval work on one profile and its 1,395-job eligible pool.
Silver labels = Gemma3:12b judge (prompt v2, κ_w 0.735 vs the user's 62 blind labels, in-sample);
human labels = the user's 66 blind grades. The embedder comparison itself is in
`eval/results/` and [EMBEDDING_EVAL_REVIEW_REPORT.md](EMBEDDING_EVAL_REVIEW_REPORT.md).

## 1. Ingest: the extraction step (qwen2.5:3b)

- **It echoes its own prompt.** In 182 of 1,395 pool jobs (13%) the "requirements" or "responsibilities"
  field is the instruction text ("all skills, technologies, qualifications…", "what the person will
  actually do…"); 106 have both fields as pure echo. Ingest falls back to the full description only when
  both fields are *empty*, so these jobs are embedded as title + boilerplate. Their descriptions are short
  (median 1,080 chars vs 3,017). Effect on ranking is small (relevant echo jobs rank ~7 percentile points
  lower); the labeler and the judge also see the junk view.
- **It sees only the first 4,000 chars:** 37% of pool descriptions (58% of the corpus) are longer. The
  extracted text is ~45% of the description length (median 1,239 chars); 13% of jobs have only one field.
- **Not yet tested:** whether extraction helps or hurts vs embedding the full description (ablation pending;
  note the labels/judge saw only the clipped extracted view, which favors extraction).

## 2. Corpus: language

- **~23–25% of the pool is not English** (345 Spanish, 8 Portuguese; getonboard 186, himalayas 91,
  smartrecruiters 34) although the intended corpus is English-only. There is no language filter at ingest.
- English-only nomic ranks almost none of them in its top-100 (1); the multilingual models put 32–36 there.
  The eval now supports `evaluate --language en` and `label-blind --language en`.

## 3. Retrieval stage and hand-off

- **Only ~11% of the pool is relevant** (159 of 1,395 by the strict judge; grades 0:706, 1:530, 2:140, 3:19).
  Dense top-100 is ~4× better than random (P@100 0.38–0.53 vs 0.11).
- **False positives are adjacent roles, not off-topic ones:** in Qwen3's top-100, 43 relevant, 49 grade 1
  (adjacent), only 8 grade 0. Level/stack/role nuance is not something embeddings resolve; that is the
  fit-analysis stage's job.
- **The top-100 cut is the main bottleneck:** relevant jobs have median rank 224 (Qwen3) / 381 (nomic);
  Recall@100 is 0.21–0.28 against a ceiling of 0.63. (Dense-only; production also fuses BM25.)
- **Seniority:** the default filter admits one level above the profile. Senior titles are 40 of Qwen3's
  top-100 and relevant 38% of the time (47% for the rest); the user grades most of them ≤ 1.
  `seniority_rules.allowed_levels` could tighten this (impact not measured).

## 4. Query side (HyDE)

- **Production HyDE is generated at temperature 0:** the 3 samples are near-identical (embedding cosine
  0.96–0.99), so averaging does nothing.
- **A plain "target titles + tech stack" string beats it:** +0.085 AP for nomic and Qwen3 on silver labels
  (intervals exclude 0); on the user's 66 blind labels condensed AP is 0.41/0.45/0.52 (nomic/Qwen3/bge-m3)
  for HyDE vs 0.69/0.69/0.79 for the string. Caveat: the judge's rubric names the same profile fields, and
  the human set was pooled from HyDE-based runs.
- **It is not keyword search:** production BM25 with the same terms scores AP 0.317; Qwen3 dense with them
  0.430, and their top-100 lists share only 36 jobs. BM25 fusion does not help Qwen3 (0.414 vs 0.430 alone)
  but helps nomic (+0.069).
- **LLM-generated queries in the extraction structure** (same model qwen2.5:3b, whole profile; mean of 5
  samples at temp 0.7) gained ~+0.07–0.08 AP over HyDE, about tying the plain string. Guarded prompts
  (no meta-commentary, no copied projects) scored *lower* than the unguarded one, because copied stack
  terms drive the silver score. A pre-declared six-prompt test (winner chosen on a dev half) did not beat
  the plain string on the test half (all intervals include 0). Query *structure* is not the driver;
  content (titles, stack) is.

## 4b. Labels and judge

- **The 221 prod `eval_labels` marked `human` were made by an LLM** (per the user). They are not imported
  and must never be used as gold. Gold = `human_blind` labels only (66 so far: 0:20, 1:25, 2:15, 3:6).
- **Judge prompt v1** (rubric only) was lenient: κ_w 0.48, 16 of the user's 24 grade-1 jobs rated 2, it
  ignored seniority. **v2** adds level/role/stack rules: κ_w 0.735, exact 0.69, within-1 0.98, but strict
  (binary recall 0.47). Silver absolute numbers are comparisons only, not real precision/recall.
- Throughput: concurrency 8 ≈ 38 jobs/min (1,395 in ~37 min) vs 2.5 s/job sequential.

## 5. Measurement and infrastructure

- **`eval/metrics.py::bpref` was uncapped and could go negative** with complete labels; fixed (penalty
  capped at min(R, N)) with a regression test.
- **Ollama embedding limits (v0.30.11):** nomic and bge-m3 cap input at 2,048 tokens whatever `num_ctx`
  says (production's comment claims 8,192 for nomic; 1 pool job affected); Qwen3 crashes its runner
  (HTTP 400 `EOF`) on inputs of ~3,000+ tokens (pool max is 2,024).
- **The repo's existing tests write to `DATABASE_URL`** when run locally; baseline has 1 pre-existing
  failure (`test_ndcg_meets_golden_threshold`).
- `search_vector` has no secondary sort key (2,327 byte-identical vectors); the two HNSW indexes are
  unused (exact scan is used).
- The raw CV text contained the name, email, phone and a URL; the harness scrubs it before storing.

## Candidate production changes (none applied; impact unverified unless stated)

1. Treat extraction echo as a failure: fall back to the full description or re-extract (~13% of jobs).
2. Filter non-English postings at ingest (or per profile).
3. Replace or supplement HyDE with a titles+stack query, or sample HyDE at temperature > 0 (verified
   direction on silver and on 66 human labels; confirm with more human labels first).
4. Revisit the top-100 hand-off depth and the seniority range.
5. Fix the nomic `num_ctx` comment; add an id tie-break to `search_vector`; drop the unused HNSW indexes.

## Open

More blind labels covering disagreements between the current and new queries; the extraction-vs-full-text
ablation; the query-set option in the harness; confirmation in a BM25-fused production run.
