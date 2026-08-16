# Job Radar — Project Status & Agent Architecture Assessment

> Snapshot as of **2026-08-15**. This document exists because the project's own history shows
> what happens without one: `docs/EVAL.md` (the newest tracked doc as of this writing) already
> describes a search-config table one commit out of date with the code (§2.2 below). Treat this
> as a living document, not another dated plan — update §1/§3 as code lands, and prefer editing
> this file over writing a new one when the picture below changes.

---

## 1. What's built today

### 1.1 Subsystems

| Module | What it does | Status |
|---|---|---|
| `adapters/sources/` | Remotive, Arbeitnow, Himalayas, Greenhouse, Lever, GetOnBoard fetch+map adapters; `discovery.py` for ingest-by-URL | Built |
| `adapters/embeddings.py` | Ollama `nomic-embed-text` client, task-prefixed (`search_query:`/`search_document:`), `num_ctx` sized for long JDs | Built |
| `adapters/generation.py` | Hand-rolled Ollama chat client, Pydantic-schema-constrained JSON output, truncation detection | Built |
| `ingest/` | Fetch → dedupe (content-hash **and** URL, so an edited posting doesn't collide on re-run) → extract requirements/responsibilities → embed → insert, `asyncio.Semaphore`-bounded at 20 concurrent | Built |
| `retrieval/` | Hybrid search: BM25 (ParadeDB `pg_search`, per-field boosts) + HyDE vector arm, fused via RRF; deterministic seniority ladder + geo + salary-floor + max-age filters | Built |
| `fit/` | CV→profile parsing (`profile/`), HyDE-based candidate retrieval, per-job grounded fit judgment (evidence-quote-cited requirements + domain relevance), **deterministic** scoring (knockout gate + weighted sum) — LLM never outputs the number | Built |
| `fit/cache.py` | Persistent fit-judgment cache keyed on `(profile_id, job_id, content_hash, model, prompt_version)` | **Uncommitted**, working |
| `quality/` | Per-source data-quality metrics (salary/location/HTML-leakage/short-description rates) for ingest health checks | Built |
| `eval/` | TREC-style offline retrieval eval: nDCG@10 (primary), Recall@50, MRR, P@5/P@10, BPref; LLM-assisted labeling with human review; OAT parameter sweep; CI golden-config regression gate | Built |
| `agents/` (LangGraph graph) | — | **Not started** |
| `application/` (Requirements/Drafting/Critic/Submission) | — | **Not started** |
| `guardrails/` | — | **Not started** |
| `app/` (FastAPI/MCP/web UI) | — | **Not started, deliberately deferred** (see §4 — not needed while driving everything via CLI) |
| Observability (Langfuse/OTel) | Declared in `pyproject.toml`; **zero call sites** in the codebase | **Not started** |

### 1.2 CLI surface (`pyproject.toml [project.scripts]`)

`job-radar-ingest`, `job-radar-scheduler`, `job-radar-assess` (quality), `job-radar-profile`,
`job-radar-fit`, `job-radar-eval-label`, `job-radar-eval-run`, `job-radar-eval-sweep`. Everything
today is driven through these — no server process required.

### 1.3 Uncommitted work in the tree right now

`fit/cache.py`, an Alembic migration adding `fit_judgments_cache`, and touch-ups to
`generation.py`, `config.py`, `db/models.py`, `fit/{analyze,cli,pipeline,schema}.py`,
`retrieval/filters.py`, and their tests — this is the persistent-cache half of
`docs/plans/FIT_THROUGHPUT_PLAN.md` (§3 of that doc), already working per the diff. Worth
committing before starting new work so it isn't sitting in a half-finished state indefinitely.

---

## 2. How the plan evolved, and where the docs went stale

Twelve design/plan documents exist under `docs/` today; only four are actually committed to git
(`DESIGN.md`, `docs/EVAL.md`, `docs/plans/PHASE_1_DESIGN.md`, `docs/plans/IMPLEMENTATION_PLAN.md`)
— the other eight (`M3_FIT_DESIGN`, `M3_IMPLEMENTATION_PLAN`, `FIT_THROUGHPUT_PLAN`,
`SEARCH_PRE_EVAL_PLAN`, `SEARCH_ENHANCEMENT_DESIGN`, `HYBRID_SEARCH_ASSESSMENT_PROMPT`,
`M6_EVAL_IMPLEMENTATION_PLAN`, `SYNTHETIC_EVAL_DESIGN`) are untracked, written between commits as
the design evolved.

### 2.1 Timeline (reconstructed from commit history + cross-references inside the docs)

```
64bedbf Initial commit                          → DESIGN.md, (pinned, never revised since)
40410ad persistence layer                       → PHASE_1_DESIGN.md, IMPLEMENTATION_PLAN.md
df0b938 retrieval: fusion, fts, vector
d600640 refactor: adapters layer
37ae1fb cv → profile scanning + fit judgement    → M3_FIT_DESIGN.md → M3_IMPLEMENTATION_PLAN.md
082b7eb profile → posting analyzer                 (design doc explored options; impl doc locked
67f30bc profile+jobs fit pipeline                   them, reversing the PDF-parser choice en route)
2c38d57 geo filtering, fit scoring                → SEARCH_ENHANCEMENT_DESIGN.md (E1–E6, "partially
                                                     implemented" at the time)
a720e95 structured seniority filtering,             ↳ HYBRID_SEARCH_ASSESSMENT_PROMPT.md — a frozen
        three-arm hybrid, deterministic gates        code snapshot fed to an external model, written
                                                       to solicit the plan below
d8ebf84 BM25 keyword search                       → SEARCH_PRE_EVAL_PLAN.md (explicitly "supersedes
                                                     E3, promotes E4" from the enhancement doc)
e4df807 eval testing suite                        → M6_EVAL_IMPLEMENTATION_PLAN.md ("the next doc"
                                                     per SEARCH_PRE_EVAL_PLAN's own text)
4a79765 enhance evals, remove CV-arm search         ↳ SYNTHETIC_EVAL_DESIGN.md (post-dates this — its
                                                       own text references BPref, added after M6's
                                                       original metric set)
b29e254 tune hybrid search                        → docs/EVAL.md (newest tracked doc)
(working tree) fit cache, schema trim             → FIT_THROUGHPUT_PLAN.md (dated 2026-08-08
                                                     measurements; §3/§4 now implemented, §5
                                                     model-tiering not yet evaluated)
```

### 2.2 Known doc/code divergences

1. **`docs/EVAL.md`'s search-config table is stale by one commit.** It documents a `vector_only`
   config as "HyDE + CV" and a `hybrid` config including a CV arm. The live `eval/qrels.py`
   removed the CV arm entirely (commit `4a79765`, *"CV arm removed after ablation showed it
   consistently degrades all metrics — backward-looking resume embedding conflicts with
   forward-looking HyDE"*) — one commit before `docs/EVAL.md` itself was added. Live configs are
   `hybrid`, `hybrid_prod`, `hyde_only`, `keyword_only` — no `vector_only`, no CV arm anywhere.
2. **BM25 field-boost default mismatch.** `docs/EVAL.md` states 5/2/1 (title/requirements/
   responsibilities); the live default in `retrieval/bm25.py` is 5/**3**/1.
3. **The dense retrieval arm is HyDE, undocumented in that depth anywhere.** No design doc
   specifies "generate 3 synthetic employer-voice postings, embed each, average" — that's what
   `fit/pipeline.py::build_hyde_embedding` actually does. Code is ahead of the docs here, not
   behind.
4. **`M3_IMPLEMENTATION_PLAN.md`'s fit-score design no longer matches `fit/score.py`.** The
   locked plan specified an LLM-judged `is_gate: bool` per requirement and an LLM-judged
   seniority alignment. Neither exists — seniority and region gating are fully deterministic
   against structured `Job.seniority` (set at ingest) and `profile.location_rules`; the LLM
   never sees or judges either. This is a real, load-bearing architecture change (a knockout gate
   moved from LLM judgment to code) that happened without the plan doc being updated.
5. **`FIT_THROUGHPUT_PLAN.md`'s proposed cache schema differs from what's implemented**, and the
   implementation is the better call: the plan proposed caching the full `FitAssessment`
   (including score/verdict); the live `FitJudgmentCache` caches only the LLM's `FitJudgment`,
   because score/verdict depend on runtime `--level` overrides and scoring constants that can
   change independently of the model's output — caching the derived number would silently serve
   stale scores across a constants change.
6. **`SYNTHETIC_EVAL_DESIGN.md` (pool-bias mitigation via synthetic personas/queries) is a
   proposal, not yet built** — no `eval/personas/`, `gen_personas.py`, `gen_synthetic_jobs.py`,
   `inject_synthetic.py` exist.
7. **`DESIGN.md` itself describes Phase 1 retrieval as "hybrid search (vector + FTS + RRF)"** —
   true at the level DESIGN.md operates at, but it doesn't name BM25-via-ParadeDB, HyDE, nomic
   task prefixes, structured seniority gating, or the fit-judgment cache, all of which are real
   and shipped. Expected for a north-star doc pinned at commit 1 (CLAUDE.md says not to
   relitigate it), but worth knowing it describes the shape, not the current mechanism.

### 2.3 Recommendation

None of this needs fixing retroactively — the historical docs are a genuinely useful design
record (the M3/search docs in particular show real engineering reasoning: rejected alternatives,
measured tradeoffs, citations). But `docs/EVAL.md` §"Commands" table should get a quick correction
pass since it's user-facing and actively wrong, and it's worth deciding whether the eight
untracked docs get committed as a historical record (e.g. under `docs/plans/archive/`) or stay
working-tree-only. Not blocking — flagging so it doesn't silently rot further.

---

## 3. Roadmap position

Against `DESIGN.md` §17's three phases and `PHASE_1_DESIGN.md`'s nine Phase-1 milestones:

| Milestone | Scope | Status |
|---|---|---|
| M0 | Persistence foundation | Done |
| M1 | Ingestion + dedup + embedding cache | Done |
| M2 | Hybrid retrieval, RRF, p95 < 500ms verify | Done (evolved past the original design — BM25, not FTS) |
| M3 | Fit analysis (RAG, grounded, cited) | Done (evolved — deterministic gates replaced LLM-judged ones) |
| M4 | LangGraph search/fit graph | **Not started — this is the current focus** |
| M5 | Surfaces (MCP / FastAPI / web UI) | **Deliberately deferred** — not needed while CLI-driven |
| M6 | Eval harness + CI gate | Done, and extended beyond the original plan (BPref added; synthetic pool-bias mitigation designed but not built) |
| M7 | Observability + cost (OTel → Langfuse, PII redaction) | **Not started** — dependencies declared, zero instrumentation |
| M8 | Deployment + README | Not started |

**Phase 2** (Requirements/Drafting/Critic agents, HITL, Submission Handler, guardrails) and
**Phase 3** (tracking board) — not started; see §5 for the architecture assessment ahead of
building Phase 2's real agents.

---

## 4. Next steps

**Immediate:**
1. Commit the in-flight fit-cache work (§1.3) rather than let it sit uncommitted.
2. Correct `docs/EVAL.md`'s config table (§2.2 item 1–2).

**Current focus — close out Phase 1 (M4 + M7), no `app/` needed:**
3. Design + build a small LangGraph graph over the existing search+fit pipeline. Decided in this
   conversation: linear `interpret → retrieve → fit → END`, no separate Supervisor node (nothing
   branches yet), a new `interpret` node (NL ask → structured `FitCriteria`, genuinely new
   capability) sitting in front of `retrieve`/`fit` nodes that are thin extractions of what
   `fit/pipeline.py::run_fit_pipeline` already does.
4. Wire Langfuse/OTel tracing through the two existing LLM call sites (HyDE synthesis,
   `analyze_fit`) before adding the interpret node's new call, to get a trace baseline on known
   behavior first.
5. New `job-radar-agent` CLI entrypoint running the graph.
6. M8 (deployment/README) — low priority until there's a working agent graph worth demoing.

**Later — Phase 2, once M4/M7 are solid:** design and build the real agents one at a time,
following this project's own established pattern (a `*_DESIGN.md` exploring options, then a
locked `*_IMPLEMENTATION_PLAN.md`, per how M3 and the search work were actually built) — starting
with **Fit** (already exists, just needs graph-wiring + reuse in the application flow), then
**Requirements** (needs its scope question resolved first, §5.4), then **Drafting**+**Critic**
together (they're one evaluator-optimizer loop, not two independent builds, §5.3).

---

## 5. Agent architecture assessment

### 5.1 Vocabulary: workflow vs. agent

Per Anthropic's own engineering guidance on this exact question
([Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)):

> **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
> **Agents** are systems where LLMs dynamically direct their own processes and tool usage,
> maintaining control over how they accomplish tasks.

And on when each is warranted:

> Use workflows for predictability and consistency on well-defined tasks; use agents when
> flexibility and model-driven decision-making are needed at scale... Agentic systems often trade
> latency and cost for better task performance — only adopt them when simpler approaches fall
> short.

This matters for Job Radar specifically: **`DESIGN.md` §8 calls the whole application-assist
flow a "multi-agent architecture,"** and LangGraph's own vocabulary calls every graph node an
"agent" loosely — but under Anthropic's stricter definition, a component only earns that name if
it dynamically directs its own process (re-plans, loops, calls tools of its own choosing, runs an
unpredictable number of steps). None of Requirements, Fit, or Drafting do that at the scope
DESIGN.md specifies — each is a single structured-output LLM call with a fixed input shape and a
fixed output schema. That's not a criticism of the design; it's the more defensible design.
Anthropic's guidance is explicit that most production systems shouldn't reach for open-ended
agents at all, and this system's own §8 already half-agrees: *"a node that is only a different
prompt over the same context is implemented as a function, not promoted to an agent."* The
correction is just to apply that rule all the way through, and to use "node" for the bounded
LLM steps and reserve "agent" for anything that actually loops.

### 5.2 The decomposition test, applied to every node in both flows

| Node | Flow | Category | Why |
|---|---|---|---|
| Retrieval | search | **Tool** | Deterministic hybrid search, zero LLM judgment. Already an instance of Anthropic's **Parallelization (sectioning)** pattern — BM25 and HyDE run concurrently, fused by RRF — this was true before the pattern was named. |
| Interpreter | search | **Workflow node — Routing pattern** | Single-shot NL-ask → structured-criteria classification. Fixed input/output shape, no looping. |
| **Fit** | both | **Workflow node — Augmented LLM (chained)** | Structured, evidence-grounded classification call; deterministic scoring downstream. The most substantial node in the system and the only one reused across both flows — real design weight, but still a single bounded call, not a looping agent. Built. |
| Requirements | application | **Workflow node — Routing, narrower than drafted** | See §5.4 — part of its stated job is already solved (ATS type is ingest-time structured data), part needs a tool that doesn't exist yet. |
| **Drafting + Critic** | application | **Workflow — Evaluator-Optimizer pattern** | This is the one place a genuine iterative loop with measurable value is justified: Drafting generates, Critic evaluates against explicit criteria (grounding/voice/relevance), and revision is worth another pass. Anthropic names this exact shape as its own pattern — treat it as one designed unit, not two independent agents. See §5.4 for splitting *what* Critic checks. |
| Submission Handler | application | **Tool** | "Approved package in, mapped fields out" — deterministic per-ATS field mapping, no fresh LLM call needed. |
| Supervisor | both | **Control flow — Prompt Chaining with gates** | Routing + HITL approval gates are exactly Anthropic's Prompt Chaining pattern's *"programmatic checkpoints between steps."* No LLM call of its own unless a routing decision itself needs judgment (none currently do). |
| HITL node | application | **Not a node** | A LangGraph `interrupt()`/resume call inside Supervisor's control flow, not a routed step. |
| Guardrails (injection screen, grounding check, redaction) | both | **Wrappers, not nodes** | They sit around agent boundaries (screen input into Requirements/Drafting, check Drafting's output, redact at the trace-export boundary), not in the routing diagram. |

**Net result:** zero components in this system currently warrant Anthropic's strict "agent"
label. That's a feature at this scope — matches both Anthropic's "start simple" guidance and this
project's own CLAUDE.md ethos (no premature abstraction, no boilerplate). If a genuinely
open-ended, tool-looping agent ever becomes justified, the strongest candidate is Drafting given a
tool to pull additional profile/CV context mid-draft and multiple revision passes against Critic
feedback — but that's a real complexity/cost increase to adopt only on evidence, the same
discipline `FIT_THROUGHPUT_PLAN.md` already applies to model tiering ("adopt only on evidence,"
its own words).

### 5.3 Full-system pattern mapping

Read end to end, DESIGN.md's two flows compose entirely from five named, well-understood patterns
— nothing here needs inventing:

- **Search flow** = Routing (Interpreter) → Parallelization/sectioning (Retrieval's BM25+HyDE
  arms) → Augmented LLM chained per candidate (Fit).
- **Application flow** = Routing (Requirements) → Augmented LLM (Fit, reused) → Evaluator-Optimizer
  loop (Drafting↔Critic) → Prompt Chaining checkpoint (HITL gate) → Tool call (Submission Handler).

### 5.4 Two open scope gaps worth resolving before building

**Requirements agent is narrower than DESIGN.md §8's table states.** Its stated job is "parse a
posting into required fields; detect custom questions; identify ATS type." ATS type is *already*
structured data — `Job.source_type`/`Job.source`, set at ingest by the adapter that pulled the
posting (`adapters/sources/{greenhouse,lever,getonboard}.py` already encode per-ATS shape). What's
genuinely unsolved is extracting the posting's *custom application questions* — and that's not a
prompting problem, it's a missing **tool**: nothing in the codebase today fetches or parses the
live application *form* (as opposed to the posting description the ingestion adapters already
capture). Resolve this before designing the Requirements node's I/O: does it need a new
`fetch_application_form` adapter first, or does its scope shrink to "parse posting text only" for
a first cut?

**Critic's grounding check should probably not be a second LLM call grading the first one.** The
Fit schema already made this exact call for a structurally identical problem — its own docstring:
evidence quotes are verbatim *"so a later guardrail can verify the quote actually appears in its
source"* rather than trusting a model to grade its own honesty. Drafting's grounding dimension
(does every claim trace to real profile/CV data) is the same kind of code-checkable
quote-matching problem. Only the fuzzy dimensions — voice adherence, relevance — need an actual
LLM judge. Design Critic as **grounding = code, voice/relevance = LLM-as-judge**, not one
monolithic "Critic agent," and that judge should follow Anthropic's own evaluator lessons (§5.5).

### 5.5 Production lessons worth carrying forward

From Anthropic's [multi-agent research system writeup](https://www.anthropic.com/engineering/built-multi-agent-research-system)
— written about a much larger swarm-of-subagents system than Job Radar needs, but the operational
lessons transfer directly to a single orchestrated workflow:

- **Token/cost cost of decomposition is real and multiplicative** — their multi-agent system used
  ~15× the tokens of a single chat call; tool calls alone run ~4×. This is a direct argument for
  keeping Job Radar's node count small and justified (§5.2's result) rather than decomposing for
  its own sake, especially since this system pays real Ollama/paid-API cost per run
  (`DESIGN.md` §15's `< $0.05/query` target already assumes a lean call graph).
- **Statefulness makes partial failure catastrophic unless designed for.** *"Agents can run for
  long periods of time... minor system failures can be catastrophic"* — build to resume from
  where a run failed, not restart from scratch. This directly validates `DESIGN.md` §10's
  interrupt/resume design (LangGraph checkpointing across HITL pauses) — that's not
  gold-plating, it's the documented failure mode of skipping it.
- **Tool-interface quality is as load-bearing as the prompt.** *"Agent-tool interfaces are as
  critical as human-computer interfaces... bad descriptions send agents down completely wrong
  paths."* Relevant directly to Submission Handler's future per-ATS field-mapping tool and any
  `fetch_application_form` tool from §5.4 — their schemas need the same care `fit/schema.py`
  already put into evidence-quote ordering (field order = generation order under constrained
  decoding, already a documented lesson in this codebase).
- **Evaluate early on a small set, not after a large dataset exists.** *"Testing... often allowed
  us to clearly see the impact of changes"* with as few as 20 examples. This is already exactly
  M6's approach (labeled sets of 50–80 pairs) — a validation of the eval-first discipline already
  built, worth keeping for Phase 2's draft-quality eval rather than waiting for a bigger set.
- **Full production tracing is what makes multi-step failures diagnosable at all** — directly
  validates prioritizing M7 (Langfuse/OTel) before Phase 2's agents multiply the number of steps
  a failure could hide in.

### 5.6 Recommended build order

1. **M4 + M7 first** (§4) — the graph + tracing spine, on the two nodes that already exist
   (Fit, Retrieval) plus the one genuinely new node (Interpreter). Low node count, gets LangGraph
   + Langfuse fluency on working logic before Phase 2 multiplies the surface area.
2. **Fit's graph integration** carries forward unchanged into the application flow — it's already
   built, this is just reuse.
3. **Requirements**, once §5.4's tool-scope question is resolved — smallest genuinely-new node.
4. **Drafting + Critic as one evaluator-optimizer design**, not two separate agent builds — the
   grounding/voice split from §5.4 first, since it changes Critic's shape materially.
5. **Submission Handler** last — pure tool, no LLM design questions, lowest risk, matches
   `DESIGN.md`'s own protected-core-first sequencing (§17: "breadth is cut before the core path").

Each of 3–5 gets its own `*_DESIGN.md` → `*_IMPLEMENTATION_PLAN.md` pair when we get there,
matching how M3 and the search work were actually built — not designed all at once here.

---

## Sources

- [Building Effective AI Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [How we built our multi-agent research system — Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system)
