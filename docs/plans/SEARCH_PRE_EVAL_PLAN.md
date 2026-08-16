# Search — Pre-Eval Correctness Upgrades

> Status: **planned, not started.** This is the batch of *structural corrections* to the
> hybrid retriever that should land **before** the evaluation framework, because each one is
> a known-correct fix (a thing that should have been built this way from the start), not a
> parameter to be tuned.
>
> Relationship to other docs:
> - Supersedes **E3** of [SEARCH_ENHANCEMENT_DESIGN.md](SEARCH_ENHANCEMENT_DESIGN.md)
>   (the `'simple'`→`'english'` tsvector migration). A real **BM25** lexical arm subsumes that
>   migration and fixes more than stemming alone.
> - Promotes **E4** (nomic task prefixes) from "deferred / last" to a pre-eval change, with the
>   justification spelled out in §2 C2 (the original doc had only asserted it).
> - The eval framework + parameter tuning it references (`E7`) is the **next** doc; nothing
>   here picks fusion weights, `k`, `pool`, or `limit` — those are tuned with the eval set.
>   See the assessment prompt [HYBRID_SEARCH_ASSESSMENT_PROMPT.md](HYBRID_SEARCH_ASSESSMENT_PROMPT.md).

---

## 0. Why a pre-eval batch at all

The whole project thesis is *measure before you change*. So why apply anything before the
eval set exists? Because two classes of change exist:

1. **Corrections** — the implementation is *wrong* relative to how the component is meant to
   work, with the directional effect established in the literature, independent of our corpus.
   BM25 over `ts_rank`, nomic prefixes, stemming, per-arm queries. These do not need our
   labels to justify; they need our labels only to *quantify*.
2. **Tuning** — choices with no a-priori right answer that depend on our corpus: RRF `k`,
   `pool`, arm weights, fusion method, `limit`, cross-encoder rerank inclusion. These **must**
   wait for the eval set.

This doc covers class (1) only. Class (2) is explicitly deferred (§7).

---

## 1. Current lexical arm is the weakest link

Today the lexical arm ([retrieval/fts.py](../../src/job_radar/retrieval/fts.py)) is:

```python
tsquery = func.websearch_to_tsquery('simple', query_text)   # implicit AND, no stemming
rank    = func.ts_rank(Job.search_vector, tsquery)          # TF + field weight, NO IDF, NO length norm
```

over a generated `tsvector` column ([db/models.py](../../src/job_radar/db/models.py)):

```sql
setweight(to_tsvector('simple', coalesce(title,'')),       'A') ||
setweight(to_tsvector('simple', coalesce(description,'')), 'B')
```

Three independent defects compound here:

| Defect | Mechanism | Consequence on a job corpus |
|---|---|---|
| **`ts_rank` ≠ BM25** | No IDF; no document-length saturation (default `normalization=0`). | "engineer", "team", "remote" (in every posting) count as much as "Kafka". Long boilerplate JDs outrank tight relevant ones. |
| **Implicit AND** | `websearch_to_tsquery` ANDs all bare terms. | The profile query is a *bag* of titles+stack+domains; **no posting contains all of them**, so the arm frequently returns **zero rows** and the system silently collapses to dense-only. |
| **`'simple'` config** | No stemming, no stop-wording. | `developer`≠`developers`, `engineer`≠`engineering`. |

`ts_rank` was the wrong tool from the start; **BM25 is the standard lexical ranker** (Robertson
& Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in
IR, 2009) precisely because it adds the IDF and length-saturation that `ts_rank` lacks. Crucially,
**BM25 also fixes the implicit-AND problem for free**: it scores documents by weighted term
*overlap*, so a posting matching 3 of 5 query terms still ranks — no manual OR-tsquery
construction needed.

---

## 2. The changes (priority order)

### C1 — Replace the lexical arm with real BM25  ⭐ headline

**What.** Remove `ts_rank` + the `'simple'` generated `tsvector` **entirely** and put BM25 in its
place. Index `jobs(title, description)` with a BM25 index, score with BM25, with a tokenizer that
handles stemming + stop-words (see C1a for the tokenizer choice) and a title boost (the A/B
`setweight` analog). There is **no transitional `ts_rank` fallback** — it is deleted, not flagged.

**How (Postgres-native, no external vector DB).** Use **ParadeDB `pg_search`** — a Postgres
extension exposing a BM25 index via the `@@@` operator and `paradedb.score(id)`, backed by a
Tantivy (Lucene-class) index. It runs *inside* Postgres, honoring the "Postgres + pgvector, no
external search service" constraint.

- **Infra:** switch the dev/CI Postgres image to one that bundles `pgvector` **and**
  `pg_search` (the `paradedb/paradedb` image ships both), then `CREATE EXTENSION pg_search;`
  in a migration. This is the one infra change in this plan.
- **Schema / migration:** create the BM25 index over title + description with a title field boost.
  **Drop** the generated `search_vector` column and its GIN index (nothing else reads
  `search_vector` — geo filtering uses raw `~*` regex on `location`/`title`/`description`, not the
  tsvector, so this is safe).
- **Code:** delete [retrieval/fts.py](../../src/job_radar/retrieval/fts.py)'s `ts_rank` body and
  replace it with a `search_bm25` (new `retrieval/bm25.py`) querying via `@@@` and ordering by
  `paradedb.score()`, returning the same `list[tuple[UUID, float]]` contract so `search()` and RRF
  are untouched. The `extra_filter` (geo/remote/seniority/salary) still composes as ordinary `AND`
  predicates alongside `@@@`.

**Why / what it fixes.** All three §1 defects at once: IDF + length-norm (BM25 core), partial
matching (replaces implicit-AND, **supersedes E3's OR-semantics work**), and stemming/stop-words
(via the tokenizer, **supersedes E3's `'english'` migration**).

**Cost.** Infra image swap + one migration (create BM25 index, drop tsvector/GIN) + ~1 rewritten
module. **No re-embedding.** Re-indexing only (fast at 5–10k rows).

**Reference.** Robertson & Zaragoza 2009 (BM25); ParadeDB `pg_search` docs (BM25 in Postgres);
Weaviate and Elasticsearch both use BM25 as the lexical arm of hybrid search.

> **Alternative considered:** `vchord_bm25` (VectorChord/TensorChord) — lighter, pgvector-adjacent
> BM25 via a `bm25vector` type. Viable fallback if the ParadeDB image is undesirable in CI.
> **Rejected:** computing BM25 by hand from term stats (too slow, reinvents Tantivy); staying on
> `ts_rank` with only the `'english'` migration (E3) — leaves IDF and length-norm unfixed, i.e.
> still not BM25.

#### C1a — Tokenizer choice: does `simple`→`english` stemming hurt tech-stack keywords?

This question only concerns the **lexical/BM25 arm**. The dense arm embeds with nomic, which
handles tech terms semantically and is unaffected by tokenization. So this is purely "what
tokenizer should the BM25 index use."

**What `english` (Snowball) stemming actually does to common tokens:**

| Token | `english` stem | Effect |
|---|---|---|
| developer / developers | `develop` | ✅ unifies plural/variant — desired |
| engineer / engineering | `engin` | ✅ unifies |
| analytics / analyst | `analyt` / `analyst` | ⚠️ *do not* unify (different stems) — usually fine |
| kubernetes | `kubernet` | ➖ harmless: same stem on query and doc |
| postgres / postgresql | `postgr` / `postgresql` | ⚠️ do **not** unify — "postgres" and "postgresql" become different terms |
| redis | `redi` | ➖ harmless if consistent |
| rails | `rail` | ⚠️ collides with the ordinary word "rail" — rare in JDs |
| react | `react` | ✅ unchanged (already a word) |

**The key insight: stemming is applied to *both* the query and the document**, so a term like
`kubernetes → kubernet` still matches itself. Over-stemming a tech token to an ugly stem is
**harmless for recall** as long as it is consistent. The genuine hazards are narrower:

1. **Symbol/punctuation tokens** — `C++`, `C#`, `.NET`, `Node.js`, `F#`, `C++17`. These are a
   *tokenizer* problem, not a stemming one: a naive tokenizer splits on punctuation and collapses
   `C++`→`c`, `C#`→`c`, colliding with each other and with the letter "c". This is the **real**
   risk and it exists regardless of stemming.
2. **Stop-word removal** dropping a meaningful short token (rare for tech, but worth a curated
   stop-word list).
3. **Stem collisions** mapping two distinct concepts to one stem (e.g. `rails`→`rail`). Low
   frequency, and BM25's IDF dampens common-stem noise.

**Assessment / recommendation.** `english` stemming is **net-positive** for the natural-language
role vocabulary (developer/engineer/manage…) and **mostly neutral** for tech keywords because of
the consistency argument — the morphological-recall win clearly outweighs the rare collision risk.
The thing that actually needs attention is **symbol-laden tech tokens**, which is a tokenizer
concern. `pg_search` supports per-field tokenizers, so the recommended config is:

- **description** → `english` stemming tokenizer (prose benefits most from stemming).
- **title** → a tokenizer that preserves tech symbols better (e.g. a `whitespace`/`raw`-style or
  ngram tokenizer), since titles are where exact tech/role tokens live and where `C++` vs `C#`
  precision matters most.

Exact tokenizer settings are a small **verification gate in Phase B** (§4): confirm `C++`, `C#`,
`Node.js`, `.NET` are retrievable and don't collide, and that `developers`/`developer` unify.
This is config, not a separate migration.

### C2 — nomic task prefixes (the "is this required?" question, answered)

**What.** Prefix `search_document:` on every indexed text (jobs at ingest, the CV at load) and
`search_query:` on the query embedded at search time.

**Is it required? Is it used in production? — the honest answer.**

- **The model authors mandate it.** `nomic-embed-text` was contrastively trained *with*
  task-instruction prefixes (`search_query:`, `search_document:`, `clustering:`,
  `classification:`). The model card (`nomic-ai/nomic-embed-text-v1.5` on Hugging Face) states the
  prefixes are **required**, and the paper documents prefix-conditioned training: Nussbaum,
  Morris, Mulyar & Duderstadt, *Nomic Embed: Training a Reproducible Long Context Text Embedder*,
  2024 (arXiv:2402.01613). Every MTEB score Nomic reports is *with* prefixes — there is no
  published "no-prefix" number to rely on.
- **It is a standard production pattern, not a Nomic quirk.** The same query/document asymmetry is
  baked into the most widely deployed open retrieval encoders: E5 requires `query:` / `passage:`
  (Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training*, 2022,
  arXiv:2212.03533), and instruction-conditioned embeddings generalize the idea (Su et al.,
  *One Embedder, Any Task: Instruction-Finetuned Text Embeddings* — INSTRUCTOR, 2022,
  arXiv:2212.09741). Nomic Atlas and the common Ollama/LangChain/LlamaIndex nomic integrations all
  apply the prefixes.
- **Why it matters for us specifically:** ours is *asymmetric* retrieval — a short query against
  long documents — which is exactly the case the `search_query:`/`search_document:` split was
  trained for. The prefix nudges query and document into the aligned region of the space; without
  it they sit in a mismatched region and similarity is noisier.
- **When would we drop it?** Only if (a) we couldn't re-embed, or (b) all comparisons were
  document↔document symmetric (they aren't). The current code is at least *consistent* (no prefix
  on either side), so this is a quality opportunity rather than an outright bug — but the upside is
  free-after-one-backfill and author-recommended.

**Recommendation: adopt it.** The cost is a single one-shot re-embed; the change is endorsed by
the model authors and matches our exact (asymmetric) use case. Keep it in this batch.

**How.**
- [adapters/embeddings.py](../../src/job_radar/adapters/embeddings.py): change `embed(text)` →
  `embed(text, *, task: Literal["query", "document"])` that prepends the correct prefix. Make
  `task` required so no call site can forget it.
- Update call sites: [ingest/pipeline.py](../../src/job_radar/ingest/pipeline.py) and
  [profile/loader.py](../../src/job_radar/profile/loader.py) → `document`;
  [retrieval/search.py](../../src/job_radar/retrieval/search.py) query embed → `query`.
- **Backfill:** one-shot script to re-embed all `jobs.embedding` and the `profile.cv_embedding`
  with the `document` prefix. Query and document vectors **must** live in the same prefixed
  space — a half-applied prefix *degrades* results, so the backfill is mandatory, not optional.

**Cost.** Small code change + **full re-embed of the `jobs` table and the CV** (one-shot Ollama
backfill; ~minutes at 5–10k rows). Re-embedding, not re-indexing.

**Note on measuring it.** The re-embed overwrites the stored vectors (one-way door). If we want
the eval to A/B prefixed-vs-not, snapshot the corpus and tag a `corpus_version` before the
backfill; otherwise we rely on the literature above and quantify only the post-change state.

### C3 — Per-arm query construction

**What.** Stop feeding one concatenated bag to all arms. Build two query forms from the profile:
- `lexical_query` — discriminative keywords (titles + stack + domains) for the BM25 arm; with
  BM25 these now *broaden* recall (partial match) instead of AND-narrowing.
- `dense_query` — a short natural-language synthesis (e.g. *"Senior backend engineer using
  Python and Kafka in fintech, remote"*) for the query-vector arm, since dense encoders expect
  query-like text, not a token bag.

**How.** Split [fit/pipeline.py](../../src/job_radar/fit/pipeline.py) `build_query()` into
`build_lexical_query()` / `build_dense_query()`; thread both through
[retrieval/search.py](../../src/job_radar/retrieval/search.py) so the BM25 arm and the
query-vector arm receive their own text. The CV-vector arm is unchanged (uses `cv_embedding`).

**Why.** The current single bag is simultaneously bad for both arms — it triggers the lexical
zero-match (pre-BM25) and is out-of-distribution for the dense encoder. The two arms want
different inputs.

**Cost.** ~30 lines, no migration, no re-embed. **Risk:** low.

### C4 — Fusion plumbing: per-arm weights + configurable `k`/`pool`

**What.** Add the *capability* to weight arms and to set `k`/`pool`/`limit` — **without choosing
the values**. Defaults stay exactly at today's behavior (equal weights, `k=60`, `pool=50`,
`limit=20`), so this change is behavior-neutral until the eval tunes it.

**How.** [retrieval/fusion.py](../../src/job_radar/retrieval/fusion.py): accept a `weights`
sequence and compute `score += w_arm * 1/(k+rank)`; default `w=1.0` for every arm. Thread the
knobs through `search()`.

**Why.** Two of the three arms are dense (query-vector + CV-vector), so equal-weight RRF already
gives dense ⅔ of the vote and the CV arm is query-independent (a static prior) — these *need*
weighting, but the *right* weight is a tuned value. Building the knob now means the eval phase has
something to turn; picking the value now would be guessing.

**Cost.** ~5 lines + signature plumbing. No migration, no re-embed. **Risk:** none (behavior-
neutral at default weights).

### C5 — Embedding context / truncation safety

**What.** Ensure the Ollama embedding call uses a context window large enough for full job
descriptions (nomic supports long context via rotary scaling; Ollama may truncate at a smaller
default `num_ctx`). Verify long JDs aren't silently cut at ingest.

**How.** Confirm/raise `num_ctx` (or the equivalent option) in
[adapters/embeddings.py](../../src/job_radar/adapters/embeddings.py); if a JD exceeds the limit,
truncate deliberately (e.g. title + leading N tokens) rather than letting the back half vanish.

**Why.** A truncated document vector drops the tail of the description (often the tech-stack list)
from retrieval. Correctness, not tuning.

**Cost.** Small; bundle with C2 (same module; if `num_ctx` changes the vectors, fold into C2's
backfill). **Risk:** low.

---

## 3. Target retrieval shape (after C1–C4)

```
lexical_q = caller arg OR build_lexical_query(profile)     # titles + stack + domains (BM25 partial-match)
dense_q   = caller arg OR build_dense_query(profile)       # NL synthesis
cv_vec    = profile.cv_embedding                           # may be None

arms = [
    search_bm25(lexical_q, geo_filter),                                  # pg_search @@@ + paradedb.score
    search_vector(embed(dense_q, task="query"), geo_filter),            # prefixed query vector
    search_vector(cv_vec, geo_filter) if cv_vec is not None else None,  # candidate prior
]
fused = weighted_rrf(arms, weights=[1,1,1], k=60, pool=50, limit=20)    # weights/k/pool tuned later by eval
```

What changed vs today: `ts_rank`/`tsvector` → BM25; one bag → two per-arm queries; raw embeds →
prefixed embeds; equal-only RRF → weight-capable RRF (defaults unchanged).

---

## 4. Phased build (with verification gates)

Each phase is independently shippable and lint/CI-green.

- **Phase A — C3 + C4 (no migration, no re-embed).** Per-arm queries + fusion plumbing. Lowest
  risk, unblocks everything else. *Verify:* unit tests on the two query builders; RRF with
  non-uniform weights; default weights reproduce current ordering exactly.
- **Phase B — C1 (infra + migration).** ParadeDB image, `CREATE EXTENSION pg_search`, BM25 index,
  cut `search_bm25` over, delete `ts_rank`, drop `search_vector`/GIN. *Verify:* a *"Backend
  Developer"* posting surfaces for a "Backend Engineer" profile (partial match + stemming); a
  posting matching a rare term ("Kafka") outranks a boilerplate-heavy non-match (IDF working); the
  arm no longer returns `[]` on a multi-keyword bag; **tokenizer gate (C1a):** `C++`/`C#`/`Node.js`/
  `.NET` are retrievable and distinct, and `developers`/`developer` unify.
- **Phase C — C2 + C5 (re-embed backfill).** Prefix-aware `embed()`, update all call sites, run
  the one-shot backfill, confirm `num_ctx`. *Verify:* query/doc vectors are prefixed consistently;
  spot-check that semantic neighbors improve on a few known queries; tag `corpus_version`.

Order rationale: A is free and unblocking; B is the headline correction and only re-indexes; C is
last because the re-embed is the one-way door and benefits from B already being in place.

---

## 5. Locked vs open decisions

**Proposed-locked (pending your nod):**

| # | Decision |
|---|---|
| 1 | Lexical arm becomes **BM25 via ParadeDB `pg_search`** (Postgres-native), **fully replacing** `ts_rank` + the `'simple'` tsvector — no fallback kept. This **supersedes E3**. |
| 2 | BM25 partial-match + tokenizer stemming **replaces** the planned OR-semantics and `'english'` migration — no separate tsvector regeneration. Per-field tokenizers (prose stemmed, title symbol-preserving) per C1a. |
| 3 | nomic prefixes (**E4**) ship in this batch behind a full re-embed backfill — justified as author-mandated + standard for asymmetric retrieval (§2 C2), not deferred. |
| 4 | Query construction splits **per-arm** (lexical bag vs dense NL). |
| 5 | Fusion gains **weights + configurable `k`/`pool`/`limit` as plumbing only**; values stay at current defaults and are tuned by the eval, not here. |

**Explicitly deferred to the eval phase (NOT in this batch):** see §7.

---

## 6. Cost summary

| Change | Migration | Re-index | Re-embed | Infra | Risk |
|---|---|---|---|---|---|
| C1 BM25 (+ C1a tokenizer) | ✅ (create BM25, drop tsvector) | ✅ | — | ✅ (ParadeDB image) | medium (headline cutover) |
| C2 prefixes | — | — | ✅ full | — | low (one-way door) |
| C3 per-arm query | — | — | — | — | low |
| C4 fusion plumbing | — | — | — | — | none (neutral default) |
| C5 ctx/truncation | — | — | maybe (fold into C2) | — | low |

---

## 7. Deferred to the eval phase (tuning, not corrections)

Out of scope here — no corpus-independent "right" answer; must be measured on the labeled set:

- **RRF `k`, per-arm `pool`, final `limit`** — sweep.
- **Arm weights** — choose the values (the *knob* lands in C4; the *value* is tuned).
- **Fusion method** — RRF vs score-normalized fusion (min-max / DBSF).
- **BM25 `k1`/`b` parameters** — saturation/length-norm tuning.
- **Cross-encoder rerank stage** (local `bge-reranker`/`mxbai-rerank`) — the largest expected
  precision win, but additive and parameter-laden; validate before/after with the eval, do not
  bundle it with these corrections.
- **CV-arm replace-vs-keep and its weight** — the dual-arm exists (E1b); whether to down-weight or
  drop it is an eval decision.

---

## 8. Open questions

1. **Infra appetite for the ParadeDB image** in dev + CI, or prefer `vchord_bm25` as the BM25
   provider? (Both keep BM25 inside Postgres; ParadeDB is more mature.)
2. **Corpus size** — how many `jobs` rows today? Gates the re-embed backfill runtime for C2.
3. **Title boost factor** for the BM25 index — start with the existing A/B intuition (title ≫
   description) and leave the exact factor for the eval sweep, or pick a fixed boost now?
4. **`dense_query` synthesis** — template-based assembly from profile fields, or a one-time LLM
   pass to write a natural-language query? (Template is deterministic and free; LLM is richer but
   adds a dependency.)
5. **Snapshot before the C2 re-embed?** Worth a `corpus_version` snapshot to enable a
   prefixed-vs-not A/B in the eval, or accept the literature and skip it?
