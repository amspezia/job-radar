# M3 — Fit Analysis (and CV parsing) — Design & Concepts

> Status: **draft for discussion**. This document explains every open design point
> for M3 in depth — what it does, why it matters, the choices, and a recommendation.
> Nothing here is built yet. It also folds in the "parse the whole CV into structured
> data" idea, which reshapes how the profile is populated.

---

## 0. Where M3 sits

M0–M2 built the machine that **finds** jobs (ingest → store → hybrid search). M3 is the
first milestone that **judges** a job against a *person*: given someone's profile and a
single posting, produce a grounded verdict — what matches, what's missing, and how good
the fit is.

Two important firsts in M3:

1. **First use of a *generation* model.** Until now we only used an *embedding* model
   (text → vector). M3 needs a model that **writes** — produces words/structured answers.
   Both run locally via Ollama for now; the client lives in `adapters/generation.py`,
   right next to `adapters/embeddings.py`.
2. **First place fabrication becomes dangerous.** A search result that's slightly off is
   tolerable. A fit analysis that *invents* a qualification you don't have, or a
   requirement the posting never stated, is a correctness failure — and avoiding that is
   the entire thesis of this project (reliability, grounding, human-in-the-loop).

---

## 1. Concepts primer (read this first)

These terms recur throughout. Plain-language definitions:

### Embedding vs. generation
- **Embedding model** (what we already use): turns text into a list of 768 numbers (a
  *vector*) that captures meaning. Good for "find me things *similar* to this." It is a
  black box — you can't read a vector and explain *why* two things are similar.
- **Generation model** (new in M3): turns a prompt into *written output* — sentences, or
  structured JSON. Good for "explain", "extract", "judge". You *can* read its output.

For fit analysis we need **explanations you can read and verify**, so generation is the
tool. Embeddings still help elsewhere (e.g. pre-ranking jobs by similarity to a CV).

### Structured output / constrained decoding
By default a generation model emits free-form text. We instead want a **fixed shape**,
e.g. `{ "verdict": "...", "score": 72, "matches": [...] }`, so code can use it reliably.

- **JSON mode**: ask the model to "reply in JSON." You get valid JSON, but not
  necessarily *your* fields — it might rename or omit things.
- **Schema-constrained decoding**: you hand the model a *JSON Schema* (a formal
  description of the exact shape), and the runtime forces every token it generates to
  fit that schema. The output is **guaranteed** to parse into your structure. Ollama
  supports this via its `format` parameter.

### RAG (Retrieval-Augmented Generation)
A model only "knows" what it was trained on. **RAG** means: before asking it a question,
you *retrieve the relevant source material* and paste it into the prompt as context, then
instruct the model to answer **only from that context**. For us the "retrieved" material
is the profile + the posting. This is what keeps answers tied to *this* candidate and
*this* job, instead of the model's generic training memory.

### Grounding & citations
**Grounding** = every claim the model makes is backed by a specific piece of source text.
**Citations** = the model is required to quote (or point to) the exact source line for
each claim. This lets us *check* the model afterwards: if it claims "8 years of Python"
but that phrase appears nowhere in the CV, we can flag or reject it. Grounding is how we
turn "trust the LLM" into "verify the LLM."

---

## 2. The two LLM tasks in M3

Adding your CV-parsing idea, M3 actually has **two** generation tasks, both using the same
local client and the same structured-output mechanism:

| Task | When it runs | Input | Output |
|---|---|---|---|
| **A. CV parsing** | once, at *profile load* | CV **file (PDF or text)** | structured profile: tech stack, roles, years, seniority, domains |
| **B. Fit analysis** | per job, at *query time* | structured profile + posting | grounded matches, gaps, verdict, score |

Task A has a **pre-stage** the table glosses over: the CV usually arrives as a *PDF*, which
must first be turned into text before any LLM sees it (see Point 2, stage 2a).

Doing (A) once and storing the result means (B) compares against **clean, consistent,
inspectable** data every time — instead of re-reading a noisy CV on every single job.

---

## 3. Open points

Each point below: **what it does → why it matters (impact) → the choices → recommendation.**

---

### Point 1 — The generation client + how we get structured output

**What it does.** A thin async function in `adapters/generation.py` that sends a prompt to
the local Ollama model and returns structured data. It's the shared engine for *both* LLM
tasks (CV parsing and fit analysis).

**Why it matters.** Everything else depends on it. If structured output is unreliable,
every downstream feature has to defensively parse and retry. Getting this boundary right
once makes the rest clean.

**The choices (how to force structure):**

| Option | How it works | Pros | Cons |
|---|---|---|---|
| **A. JSON mode** | tell the model "reply in JSON" | simple, works everywhere | valid JSON but maybe wrong fields; must validate + sometimes retry |
| **B. Schema-constrained** | hand Ollama a JSON Schema via `format` | output *cannot* violate the shape | needs a recent Ollama; can slow/lower quality slightly |
| **C. Prompt + parse + retry** | free text, then parse it ourselves | model-agnostic | brittle, wasteful, lots of edge cases |

**Recommendation: B, with a Pydantic model as the single source of truth.** We define the
output shape once as a Pydantic class; Pydantic *generates* the JSON Schema we give Ollama,
and *validates* the response we get back. Conformance + a typed Python object, one
definition. (If a model ever misbehaves under B, A is the graceful fallback.)

**Also needed:** a `generation_model` setting in `config.py` (alongside the existing
`embedding_model`), so the model is swappable without code changes.

---

### Point 2 — CV ingestion: file → text → structured profile (your idea)

This is a **two-stage** pipeline, because the CV usually arrives as a **PDF**, not text:

```
CV file (PDF or .txt) ──[2a] extract text──▶ raw cv_text ──[2b] LLM structuring──▶ structured profile
```

#### Stage 2a — PDF → text

**What it does.** Turn the uploaded CV file into plain text (and pass `.txt`/`.md` through
untouched). This is *not* an external-I/O concern (no network — it reads a local file the
user gave us), so it lives in the profile feature package, not `adapters/`.

**Why it matters.** The LLM parser (2b) *and* the grounding quotes (Point 4) both operate
on text. If extraction comes back empty or garbled, everything downstream is poisoned — and
feeding empty text to an LLM is a prime fabrication trigger (it will happily invent a whole
profile out of nothing).

**The choices (which extractor library):**

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **pypdf** | pure-Python text extraction | lightweight, permissive (BSD) license, no system deps | weaker on multi-column / table layouts (can jumble word order) |
| **pdfplumber** | built on pdfminer.six | better column/table/layout fidelity | heavier, slower |
| **PyMuPDF (fitz)** | C-backed extractor | fastest, very robust | **AGPL license** (matters if ever distributed); native build step |

**Recommendation: pypdf.** CV layouts vary wildly, but the *next* stage is an LLM that
tolerates messy, slightly-out-of-order text — so we don't need pixel-perfect extraction. A
lightweight extractor feeding a robust LLM parser is the right balance. Upgrade to
pdfplumber only if real CVs come out too garbled to parse.

**Two guards (not optional):**
- **Accept `.txt`/`.md` directly** — skip extraction entirely. Handy for tests and for
  users who paste plain text.
- **Detect empty / near-empty extraction** (e.g. an image-only *scanned* PDF with no text
  layer) and **fail loudly** — never pass empty text to the LLM. OCR for scanned CVs is
  deliberately out of scope for v1 (it's a heavy dependency, Tesseract et al.); we
  detect-and-refuse instead, and the user converts to a text-layer PDF. This ties directly
  into the anti-fabrication guards in Point 5.

#### Stage 2b — text → structured profile (the original idea)

**What it does.** At profile-load time, feed the *entire* extracted CV text to the model and
have it extract structured facts: tech stacks/keywords, roles held, years of experience,
seniority level, domains. These populate the structured `Profile` columns we already have
(`work_history`, `domains_keywords`, `target_titles`, `seniority`).

**Why it matters (impact).**
- **Consistency:** fit analysis compares against clean fields, not raw CV noise.
- **Efficiency:** parse once, reuse for thousands of job-fit checks.
- **Transparency & human-in-the-loop:** you can *read* the extracted profile, correct it,
  and approve it before it's used — which is exactly this project's thesis. A raw embedding
  can't be reviewed; a parsed `{"stacks": ["Python","Postgres"], "seniority": "senior"}`
  can.

**Important clarification — parsing vs. embedding (they're not the same, and not rivals):**
- `cv_embedding` (a vector) is for *semantic similarity* — "find jobs broadly like this
  CV." Black box, not readable.
- *Parsed structure* (keywords/roles) is for *explicit, explainable matching* — "this job
  needs Kubernetes; the CV lists Kubernetes." Readable, checkable.
- We can keep **both**: embedding for retrieval-side pre-ranking, parsed structure for fit
  analysis. Your idea adds the second; it doesn't replace the first.

**The choices:**

| Option | Pros | Cons |
|---|---|---|
| **A. Parse once → structured profile (your idea)** | clean, cheap-per-fit, inspectable, editable | one extra LLM step at load; extraction can be wrong (mitigated by human review) |
| **B. No parsing — pass raw CV to fit every time** | nothing to build | re-reads noisy CV on every job; nothing to inspect/correct; more tokens per fit |
| **C. Hybrid** — parse structured fields *and* keep raw CV text for grounding quotes | precise matching *and* verifiable quotes | both code paths to maintain |

**Recommendation: C (which includes A).** Parse the CV into structured fields for precise,
explainable matching, **and** keep the raw `cv_text` so the fit step can quote exact lines
as grounding evidence. The structured fields answer "does it match?"; the raw text answers
"prove it." Both are already columns in `Profile`.

---

### Point 3 — What context the fit step actually receives (RAG granularity)

**What it does.** Decides *how much* source text we paste into the fit prompt.

**Why it matters.** Too little context → the model guesses (fabrication risk). Too much →
slower, costlier, and the model can lose the thread. With CV parsing (Point 2) in place,
the profile side is already compact and structured.

**The choices:**

| Option | Pros | Cons |
|---|---|---|
| **A. Whole structured profile + whole posting in context** | simplest; nothing relevant dropped; both fit easily in an 8B model's context | larger prompt (still fine at this size) |
| **B. Chunk + retrieve only "relevant" pieces via embeddings** | scales to very long docs | real complexity; risk of dropping the one line that mattered; overkill here |

**Recommendation: A.** A structured profile + one posting comfortably fit a local model's
context window. This is still "RAG" — the generation is augmented with retrieved
profile+posting — we just don't sub-chunk. Chunked retrieval is premature optimization for
single-profile fit.

---

### Point 4 — The fit output schema (matches / gaps / score + citations)

**What it does.** Defines the exact shape the fit model must return, and — critically — how
each claim **cites its source**.

**Why it matters.** This schema *is* the product of M3. Its citation design is what makes
fit assessments verifiable instead of "trust me." Its score design is what later lets us
measure the model against human labels (M6).

**Choice 4a — how claims cite sources:**

| Option | Pros | Cons |
|---|---|---|
| **A. Verbatim quote + source tag** (`{source:"posting", quote:"5+ years Go"}`) | *verifiable* — we can check the quote really exists in the source; enables an automated grounding check later | model may paraphrase; a bit verbose |
| **B. Reference IDs** (`posting:req_3`) | compact, precise | requires pre-splitting sources into numbered units (infra we don't have) |
| **C. Coarse tag only** (`"profile"`) | trivial | weak; can't verify; defeats the purpose |

**Choice 4b — how the score is expressed:**
- **Categorical verdict** (`strong / moderate / weak / none`) — aligns with the
  `EVAL_LABEL.label` text column, so M6 can measure agreement with human labels directly.
- **Numeric score** (`0–100`) — enables ranking and thresholds.
- These aren't exclusive — we can return **both**.

**Recommendation: verbatim-quote citations (4a → A) + both a verdict and a nullable
numeric score.** Verbatim quotes are the simplest grounding a future guardrail can
*verify* (reject any quote not found in the source). The categorical verdict aligns with
the eval set; the number enables ranking. Sketch:

```python
class Evidence(BaseModel):
    source: Literal["profile", "posting"]
    quote: str                      # verbatim — must appear in the source

class Point(BaseModel):
    claim: str                      # e.g. "Meets the 5+ years backend requirement"
    evidence: list[Evidence]

class FitAssessment(BaseModel):
    verdict: Literal["strong", "moderate", "weak", "none"]
    score: int | None               # 0–100; None = the model abstained (see Point 5)
    matches: list[Point]
    gaps: list[Point]
    summary: str
```

---

### Point 5 — Thin / empty / contradictory input (anti-fabrication)

**What it does.** Decides what happens when there isn't enough to judge — an empty profile,
a one-line posting, or contradictory data.

**Why it matters.** This is the explicit M3 acceptance criterion: *thin/empty input must
not fabricate a score.* A system that confidently scores garbage is worse than one that
says "I can't tell."

**The choices:**

| Option | Pros | Cons |
|---|---|---|
| **A. Deterministic pre-flight in code** — if profile/posting lack minimum content, return an "insufficient" result *without* calling the model | airtight, cheap, zero fabrication risk | needs a sensible threshold; misses nuance |
| **B. Prompt the model to abstain** ("return null score if unsure") | handles subtle/contradictory cases | models fabricate under pressure; costs a call; unreliable alone |
| **C. Both (defense in depth)** | covers blunt-empty *and* subtle-thin | slightly more code |

**Recommendation: C.** Code pre-flight catches clearly-empty cases (returns
`verdict="none", score=None`, no model call). A **nullable `score`** in the schema lets the
model *abstain* on thin input. The verbatim quotes from Point 4 enable the eventual
backstop: a grounding check that rejects any claim whose quote isn't in the source
(deferred to the guardrails milestone, but *enabled* now by the schema).

The same anti-fabrication discipline applies one stage earlier: an **empty PDF extraction**
(Stage 2a) must be refused before the CV-parsing LLM is ever called — same principle, same
reason.

---

### Point 6 — Populating a profile for dev, without putting PII in git

**What it does.** Gets a real profile into the database to develop/test against, while
honoring the hygiene rule: **no real CV, name, or email in the repo.**

**Why it matters.** M3 can't be tested without a profile, but a CV is exactly the kind of
PII that must never be committed. We need real data at runtime and safe data in tests.

**The choices:**

| Option | Pros | Cons |
|---|---|---|
| **A. Loader CLI reading a gitignored file** (e.g. `data/cv.pdf` → extract → parse → `Profile` row) | real data for manual dev, stays out of git; reuses the full Point-2 pipeline (2a extract + 2b parse) | a small loader to write |
| **B. Committed synthetic fixture** (a fictional person) | deterministic test data, no PII | not your real CV |
| **C. Defer to an M5 API/MCP tool** | nothing now | blocks M3 testing |

**Recommendation: A + B.** A committed **synthetic** profile fixture for automated tests
(no PII), and a `job-radar-profile load <path>` CLI that reads a **gitignored** CV file,
runs it through the Point-2 parser, and writes the `Profile` row for real manual runs.
Same fixture-vs-live split the adapters already use.

---

## 4. Proposed decisions (summary)

| # | Point | Recommendation |
|---|---|---|
| 1 | Generation client | Schema-constrained Ollama + Pydantic; client in `adapters/generation.py`; new `generation_model` config |
| 2a | PDF → text | `pypdf` extraction; accept `.txt`/`.md` directly; refuse empty/scanned PDFs (no OCR in v1) |
| 2b | CV parsing | Parse whole CV text → structured profile fields **and** keep raw `cv_text` (your idea, hybrid) |
| 3 | Fit context | Whole structured profile + whole posting (no chunking) |
| 4 | Fit output schema | Verbatim-quote citations + categorical verdict + nullable numeric score |
| 5 | Thin input | Code pre-flight + nullable score + prompt-to-abstain |
| 6 | Profile population | Synthetic fixture (tests) + gitignored loader CLI (dev) |

---

## 5. Suggested build order

Smallest, most self-contained piece first (same approach as M2):

1. `adapters/generation.py` — the local Ollama generate client with schema-constrained
   output + a `generation_model` setting. *Verify live with a trivial schema.*
2. **PDF → text** (Stage 2a) — `pypdf` extraction with the empty/passthrough guards. *Pure,
   offline-testable with a tiny fixture PDF + a `.txt` passthrough.*
3. **CV parsing** (Stage 2b) — Pydantic profile-extraction schema + the parse function.
   *Verify on a synthetic CV.*
4. **Profile loader CLI** (file → 2a → 2b → `Profile` row) + synthetic test fixture.
5. **Fit analysis** — the `FitAssessment` schema + the fit function (pre-flight guards,
   prompt, grounding-ready citations).
6. Tests: schema/validation logic (pure, offline) + a couple of live end-to-end checks.

> Open question for you: should CV parsing also fill **PII fields** (`full_name`, `email`)
> from the CV, or only the non-PII professional fields (stacks/roles/seniority)? PII in the
> *database* is allowed; it just never enters git. Your call on whether the parser touches
> those.
