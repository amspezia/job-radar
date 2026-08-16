# M3 — Implementation Plan (Fit Analysis + CV ingestion)

> Status: **plan, ready to build**. Companion to `M3_FIT_DESIGN.md` (which explores the
> options). This file records the *locked* decisions, defines the **fit-score metric**
> precisely, and lays out the file-by-file build with verification gates.

---

## 1. Locked decisions

| # | Point | Decision |
|---|---|---|
| 1 | Generation client | **Schema-constrained** Ollama output + Pydantic as the single schema source. Client in `adapters/generation.py`; new `generation_model` config. |
| 2a | PDF → text | **pdfplumber** (best layout fidelity among permissively-licensed extractors; one-time job, quality > speed). Accept `.txt`/`.md` directly; refuse empty/scanned PDFs (no OCR in v1). |
| 2b | CV parsing | **Hybrid**: parse CV → structured profile fields **and** keep raw `cv_text` for grounding quotes. |
| 3 | Fit context | **Whole** structured profile + whole posting (no chunking). |
| 4a | Citations | **Verbatim quote + source tag** on every judgment. |
| 4b | Score | **Numeric 0–100, computed deterministically by code** from the LLM's grounded per-requirement judgments — see §2. |
| 5 | Thin input | **Defense in depth**: code pre-flight refusal + nullable score + prompt-to-abstain. |
| 6 | Profile / PII | Test with a real CV via a **gitignored** loader. **PII (name/email) may be extracted and stored locally** and used by the *local* model; it must **never** be sent to an *online* model (redaction boundary — see §4). |

**Why pdfplumber over my earlier pypdf rec:** this is a one-time, per-profile job, so
extraction *quality* matters more than speed/weight. pdfplumber (built on pdfminer.six,
MIT-licensed) handles the multi-column/table layouts common in CVs far better than pypdf.
PyMuPDF extracts marginally better still but is **AGPL** — rejected on licensing.

---

## 2. The fit-score metric (the core of M3)

The headline reliability decision: **the LLM never outputs the score.** It outputs
*grounded per-requirement classifications*; pure Python computes the number. This makes the
score reproducible, auditable, tunable without re-prompting, and impossible to hallucinate.
The approach is the standard recruiting **weighted scoring model with knockout gates**
([si-labs], [foundire]), applied to **person–job fit** ([Qin et al., arXiv]).

### Stage 1 — LLM produces grounded judgments (no number)

For the posting, the model extracts each requirement and judges it against the profile:

- `kind`: `required` | `preferred`
- `is_gate`: `true` for **knockout** dealbreakers (work authorization, location/remote
  eligibility, a hard mandatory credential). Knockouts are non-compensatory.
- `satisfaction`: `met` (1.0) | `partial` (0.5) | `unmet` (0.0)
- `evidence`: verbatim quotes from profile and/or posting

Plus two whole-posting judgments: **seniority alignment** (`exact`/`adjacent`/`mismatch`)
and **domain relevance** (`strong`/`partial`/`weak`), each with evidence.

### Stage 2 — Code computes the score deterministically

**Knockout gate first (non-compensatory):** if *any* `is_gate` requirement is `unmet` →
`verdict = "none"`, `score` capped at a low fixed value (e.g. ≤ 20). A dealbreaker cannot
be offset by strengths elsewhere. This is handled *outside* the weighted sum, per the
research.

**Otherwise, a weighted compensatory sum across four dimensions** (weights are tunable
constants, initial values shown — refined later against the M6 eval set):

| Dimension | Weight | How its 0–1 sub-score is computed |
|---|---|---|
| Required-requirement coverage | 0.50 | Σ(satisfaction) / count of `required` items |
| Preferred-requirement coverage | 0.15 | Σ(satisfaction) / count of `preferred` items |
| Seniority alignment | 0.25 | `exact`→1.0, `adjacent`→0.6, `mismatch`→0.2 |
| Domain relevance | 0.10 | `strong`→1.0, `partial`→0.6, `weak`→0.2 |

```
score = round(100 * Σ_d (weight_d * subscore_d))      # 0–100
```

Empty-dimension rule: if a dimension has no items (e.g. a posting lists no preferred
requirements), drop it and **renormalize** the remaining weights so they still sum to 1.0
(don't penalize a posting for not having preferences).

**Verdict bands** (also tunable): `≥80 strong` · `60–79 moderate` · `40–59 weak` ·
`<40 none`. A failed gate forces `none` regardless of score.

### Why this satisfies the project thesis
- **Explainable:** "score 74 = required 5/6 met +1 partial (0.50×0.92), preferred 2/4
  (0.15×0.50), seniority exact (0.25×1.0), domain partial (0.10×0.6)." Every term traces to
  grounded evidence.
- **Reproducible & tunable:** same judgments → same score, always; weights live in code/
  config and can be calibrated against human labels in M6 without touching the model.
- **Non-fabricable:** the model can't emit a flattering number; it can only classify
  requirements, each backed by a quote a later guardrail can verify.

---

## 3. Data shapes (Pydantic)

### CV parsing output → populates the ORM `Profile`
```python
# profile/schema.py
class StructuredProfile(BaseModel):
    full_name: str | None              # PII — local only (see §4)
    email: str | None                  # PII — local only
    seniority: str                     # e.g. "senior"
    target_titles: list[str]
    tech_stack: list[str]              # keywords/skills
    domains: list[str]
    work_history: list[WorkItem]       # role, company, years
    years_experience: float | None
```

### Fit judgments (LLM output) and assessment (code output)
```python
# fit/schema.py
class Evidence(BaseModel):
    source: Literal["profile", "posting"]
    quote: str                         # verbatim; a guardrail can later verify presence

class Requirement(BaseModel):
    text: str
    kind: Literal["required", "preferred"]
    is_gate: bool
    satisfaction: Literal["met", "partial", "unmet"]
    evidence: list[Evidence]

class SeniorityJudgment(BaseModel):
    posting_level: str
    candidate_level: str
    alignment: Literal["exact", "adjacent", "mismatch"]
    evidence: list[Evidence]

class DomainJudgment(BaseModel):
    relevance: Literal["strong", "partial", "weak"]
    evidence: list[Evidence]

class FitJudgment(BaseModel):            # <-- the LLM returns THIS (no score)
    requirements: list[Requirement]
    seniority: SeniorityJudgment
    domain: DomainJudgment
    summary: str

class FitAssessment(BaseModel):          # <-- code returns THIS to callers
    score: int | None                    # None = pre-flight refused (insufficient input)
    verdict: Literal["strong", "moderate", "weak", "none"]
    gate_failed: bool
    judgment: FitJudgment | None         # the grounded evidence behind the score
    summary: str
```
`matches`/`gaps` are *derived views* over `judgment.requirements` (met vs unmet/partial),
not separately stored — single source of truth.

---

## 4. PII / redaction policy

- **Local is fine.** The CV parser may extract `full_name`/`email` and store them in the
  `Profile` row. The *local* Ollama model may receive them. PII in the database is allowed;
  it just never enters git (CV files live in gitignored `data/`).
- **Online is not.** When the paid/online quality-pass model lands (later milestone), PII
  must be **redacted at that boundary** before any prompt leaves the machine. M3 is
  **local-only**, so no redaction is needed yet — but the policy is fixed now, and the
  redaction seam belongs in the `guardrails/` layer at the online-client boundary, not
  scattered through callers.

---

## 5. New dependency & config

- **Dependency:** `pdfplumber` (added on this feature's branch, per the per-feature rule).
- **Config:** add `generation_model: str` to `config.py` / `.env.example` (mirrors the
  existing `embedding_model`). Local generation uses the existing `ollama_base_url`.

---

## 6. File-by-file plan

```
src/job_radar/
  adapters/
    generation.py        NEW  local Ollama generate(prompt, schema) -> dict (schema-constrained)
  profile/               NEW feature package (CV → Profile)
    extract.py                pdfplumber: file path -> raw text; .txt/.md passthrough; empty-refusal
    schema.py                 StructuredProfile (+ WorkItem) pydantic
    parse.py                  raw text -> StructuredProfile via adapters.generation
    loader.py                 orchestrate: file -> extract -> parse -> upsert Profile row
    cli.py                    `job-radar-profile load <path>`
  fit/                   NEW feature package (Profile + posting -> assessment)
    schema.py                 Evidence / Requirement / FitJudgment / FitAssessment
    score.py                  PURE: FitJudgment -> (score, verdict, gate_failed)  [the metric]
    analyze.py                build prompt (profile + posting) -> generation -> FitJudgment;
                              pre-flight guards; assemble FitAssessment via score.py
config.py                ADD  generation_model
pyproject.toml           ADD  pdfplumber dep + `job-radar-profile` script
```

Layering stays clean: `profile/` and `fit/` are **feature** packages depending downward on
`adapters/` (generation) and `db/`; `adapters/generation.py` depends only on `config`.
`profile/extract.py` is local file processing (no network) → it stays in the feature
package, not `adapters/`.

---

## 7. Testing strategy

- **Pure / offline (real unit tests):**
  - `fit/score.py` — the metric. The most important surface: knockout gate forces `none`;
    weighted partial-credit arithmetic; empty-dimension renormalization; verdict bands.
    Hand-built `FitJudgment` inputs with computed expected scores (like the RRF tests).
  - `profile/extract.py` — tiny fixture PDF, `.txt` passthrough, and empty/scanned PDF
    refusal.
- **Live (LLM/DB, verified by hand like adapter `fetch`):**
  - `profile/parse.py` on a synthetic CV; `fit/analyze.py` end-to-end on a real
    profile + a real posting.
- **Fixtures:** a committed **synthetic** profile (fictional person, no PII) for tests; the
  real CV stays gitignored.

---

## 8. Build order (each step verified before the next)

1. **`adapters/generation.py`** + `generation_model` config. *Verify live with a trivial
   schema (e.g. extract `{sentiment}` from a sentence).*
2. **`profile/extract.py`** (pdfplumber + guards). *Unit-test offline: fixture PDF, txt
   passthrough, empty refusal.*
3. **`profile/schema.py` + `profile/parse.py`**. *Verify live: parse a synthetic CV into a
   `StructuredProfile`.*
4. **`profile/loader.py` + `profile/cli.py`** + synthetic fixture. *Load your real CV
   (gitignored) → inspect the stored `Profile`.*
5. **`fit/schema.py` + `fit/score.py`** (pure metric) + **thorough unit tests**. *No LLM
   needed — this is where the score correctness is nailed down.*
6. **`fit/analyze.py`** (prompt + generation → `FitJudgment`, pre-flight guards). *Verify
   live: real profile + a real posting → grounded `FitAssessment`.*
7. **End-to-end glue:** load profile → `retrieval.search` (M2) for jobs → `fit.analyze` the
   top results. *Confirms M2 + M3 compose.*

---

## 9. Deferred (explicitly out of M3 scope)

- OCR for scanned/image-only PDFs (detect-and-refuse instead).
- The online/paid quality-pass model and its **PII-redaction** guardrail (lands with the
  guardrails milestone; policy fixed here in §4).
- The automated **grounding check** (verify each evidence quote actually appears in its
  source) — enabled by the schema now, implemented in the guardrails milestone.
- Weight/band **calibration** against human labels — happens in M6 (eval harness).

---

## References

- Qin et al., *Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural
  Network Approach* — [arXiv:1812.08947](https://arxiv.org/pdf/1812.08947)
- *Scoring Model: Guide, Practical Example & Method Comparison* —
  [si-labs](https://www.si-labs.com/en/articles/scoring-model/)
- *How to Design Perfect Knockout Criteria* —
  [foundire](https://foundire.com/blog/how-to-design-perfect-knockout-criteria/)
- *Weighted Scoring Framework for Shortlisting* —
  [evalufy](https://www.evalufy.com/blog/candidate-assessment-selection/weighted-scoring-framework-for-shortlisting/)
- *What is Job Fit Scoring?* —
  [mokaHR](https://www.mokahr.io/myblog/job-fit-scoring/)
