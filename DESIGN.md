# Job Radar — Design Document

> An observable, evaluated agentic system that finds remote developer roles, drafts tailored applications under human approval, and tracks them to outcome — exposed over the Model Context Protocol.

| | |
|---|---|
| **Status** | Draft for review (pre-implementation) |
| **Version** | 0.1.0 |
| **Owner** | Afonso Martini Spezia |
| **Document type** | Architecture & design (C4 + dynamic views) |

*Diagrams are authored as Mermaid (diagrams-as-code) so they render in GitHub and version alongside the code.*

---

## 1. Overview

Job Radar is a single-user system that turns a fragmented, repetitive job-application process into one grounded workflow. It collects remote developer postings, ranks them against a user-defined profile, and — for the roles the user chooses to pursue — gathers the position's requirements, detects its custom application questions, drafts answers in the user's own voice, and produces a complete application package that the user reviews and submits. Every application is then tracked through to outcome.

The system is built around two principles. First, the user's identity (name, contact, work history, CV, and standard answers) is entered **once** and reused everywhere, instead of being re-typed into every new application form. Second, the agent **drafts and prepares but never submits**: all irreversible actions are gated behind explicit human approval. The engineering emphasis is on making the agentic system measurable, traceable, and safe to operate.

---

## 2. Motivation & problem statement

### 2.1 The pain

Applying to remote roles today means repeating the same low-value work across many incompatible systems. Each posting links out to a different applicant tracking system — Greenhouse, Lever, Ashby, Workday, or a bespoke form — and each one asks for the same things again: name, email, links, work history, an uploaded CV. On top of that, most roles attach a few open questions that demand a tailored answer: *"What brings you to us?"*, *"What differentiates you?"*, *"Why this role?"*. Answering these well requires real thought; answering them ten times a week, from scratch, across ten different forms, does not scale, and the quality of the answers degrades exactly when volume matters most.

The cost is **mechanical** (re-entering identity that never changes) and **cognitive** (rewriting bespoke answers under fatigue). Generic job boards address neither: they match keywords, so they cannot tell whether a role fits, and they stop at the listing, so the application itself remains manual.

### 2.2 The thesis

Job Radar addresses both costs. The mechanical cost is removed by a **reusable profile**: identity is structured once and supplied to every application. The cognitive cost is reduced by **grounded drafting under approval**: the system reads what a position needs, asks the user only for the personal raw material it cannot infer, and proposes complete, on-voice answers the user edits and approves before anything is sent.

The application workflow involves multiple agents, branching, and irreversible actions, which makes evaluation, observability, human-in-the-loop control, and guardrails core concerns of the design rather than additions.

### 2.3 Goals

- Surface remote developer roles that match a user-defined profile, ranked by grounded fit.
- Eliminate repeated identity entry through a single reusable profile.
- Draft tailored answers to position-specific questions in the user's voice, grounded only in real profile data.
- Keep every irreversible action (submission, sending data to third parties) under explicit human approval.
- Track each application from draft through to outcome with editable state.
- Make the agent's behavior measurable, traceable, and safe — evaluation, observability, and guardrails as core deliverables.
- Ingest from a configurable, compliance-first set of job sources, and let the user add a specific role by URL when the most relevant openings are not reachable by any compliant API.

### 2.4 Non-goals

- **Autonomous form submission.** The system prepares approved packages; the human submits. (See §10, §17.)
- **Multi-user / multi-tenant.** One profile, one user by design.
- **Broad-industry coverage.** Scope is remote developer roles, bounded by what the configured sources carry.
- **Real-time ingestion.** Daily batch is sufficient and compliant.
- **Scraping sources that forbid it.** No scraping of LinkedIn, Indeed, Wellfound, or per-tenant Workday/Oracle endpoints. Ingestion uses compliant APIs and feeds only; roles behind hostile terms are reached, when needed, by the user pasting a URL. (See Appendix A.)

### 2.5 Persona

A single primary user: an experienced developer running an active remote job search, technically capable, applying across many platforms, who values control over what is sent on their behalf and wants to stop re-entering the same information.

---

## 3. Solution overview

Job Radar is delivered as three cooperating capabilities over one reliability spine.

1. **Discovery** — ingest, index, and hybrid-search postings; rank against the profile with grounded fit analysis.
2. **Assisted application** — gather a position's requirements, detect custom questions, collect personal raw material via a human-in-the-loop prompt, draft and critique answers, assemble an approved package, and hand off for human submission.
3. **Tracking** — persist each application with its details, dates, and an editable lifecycle state.

The spine — evaluation, observability, human-in-the-loop control, and guardrails — wraps all three.

The governing stance is **assisted, not autonomous**: the agent performs preparation and reasoning and stops at the boundary of irreversible action. Approval is per-action and explicit.

```mermaid
flowchart LR
    U([User]) -->|defines profile once| P[Reusable Profile]
    U -->|asks in natural language| D[Discovery]
    D -->|ranked, grounded matches| U
    U -->|chooses a role to pursue| A[Assisted Application]
    A -->|asks only for personal raw material| U
    U -->|edits + approves draft| A
    A -->|prepared package| U
    U -->|reviews + submits| EXT[(External ATS form)]
    A -->|records application| T[Tracking]
    U -->|updates outcome state| T
    P -.supplies identity.-> A
```

---

## 4. System context (C4 L1)

Job Radar is one system interacting with the user and a small set of external systems. It never lets postings reach the user as actions, never submits to external forms autonomously, and isolates third-party form interaction behind human approval.

```mermaid
flowchart TB
    user([User<br/>job seeker])

    subgraph radar [Job Radar]
        sys["Agentic job-search & application system"]
    end

    boards[(Job boards<br/>Remotive, RemoteOK, Himalayas, Arbeitnow, ...)]
    atsapi[(ATS job-board APIs<br/>Greenhouse, Lever, Ashby, SmartRecruiters)]
    community[(Aggregators &amp; community<br/>HN 'Who is hiring', The Muse, ...)]
    llm[(LLM providers<br/>local Ollama / paid API)]
    ats[(External ATS pages<br/>Greenhouse, Lever, Ashby, ...)]
    mcpclient[MCP clients<br/>Claude Desktop, IDE agents]
    langfuse[(Langfuse<br/>tracing backend)]

    user -->|profile, questions, approvals| sys
    sys -->|matches, drafts, status| user
    mcpclient -->|call tools| sys

    sys -->|daily batch pull| boards
    sys -->|daily batch pull| atsapi
    sys -->|daily batch pull| community
    user -->|paste a job URL to ingest| sys
    sys -->|grounded generation| llm
    sys -->|exports traces| langfuse
    sys -. prepared package, human-submitted .-> ats
```

**External dependencies and trust:** posting sources and LLM providers are untrusted inputs (postings may contain injected instructions — see §11). External ATS pages are reached only with a prepared package and a human in control of submission.

---

## 5. Container view (C4 L2)

The system decomposes into independently deployable and testable containers.

```mermaid
flowchart TB
    user([User])
    mcpclient[MCP client]

    subgraph radar [Job Radar]
        web[Web UI<br/>profile, review, tracking board]
        api[FastAPI application<br/>HTTP API + orchestration entry]
        mcp[MCP Server<br/>FastMCP: tool surface]
        agent[Agent Service<br/>LangGraph supervisor + workers]
        ingest[Ingestion Worker<br/>source adapters · daily batch · ingest-by-URL]
        db[(PostgreSQL + pgvector<br/>postings, profile, applications, eval, FTS)]
        obs[Observability<br/>OpenTelemetry exporter]
    end

    sources[(Job sources<br/>boards · ATS APIs · aggregators · community · manual URL)]
    llm[(LLM providers)]
    langfuse[(Langfuse)]

    user --> web --> api
    mcpclient --> mcp --> agent
    api --> agent
    agent --> db
    agent --> llm
    ingest --> sources
    ingest --> db
    agent --> obs --> langfuse
    api --> obs
```

| Container | Responsibility | Technology |
|---|---|---|
| Web UI | Profile editing, draft review/approval, tracking board | Thin web front-end served by FastAPI |
| FastAPI app | HTTP API, session, orchestration entry point | Python 3.12, FastAPI |
| MCP server | Exposes capabilities as discoverable tools | FastMCP |
| Agent service | Supervisor + worker agents, the workflows | LangGraph |
| Ingestion worker | Per-source adapters, daily collection, normalization, cross-source dedup, embedding; plus on-demand ingest-by-URL | APScheduler + per-source adapters |
| Database | Postings, vectors, FTS, profile, applications, eval set, traces metadata | PostgreSQL, pgvector |
| Observability | Trace/metric export | OpenTelemetry → Langfuse |

---

## 6. Component view (C4 L3)

The agent service and retrieval internals. The multi-agent topology is detailed in §8; this view places the components.

```mermaid
flowchart TB
    subgraph agentsvc [Agent Service]
        supervisor[Supervisor<br/>state machine + routing]
        interp[Interpreter<br/>NL → structured criteria]
        retr[Retrieval<br/>hybrid search + RRF]
        fit[Fit Analyzer<br/>RAG vs profile]
        reqs[Requirements Agent<br/>parse posting + detect questions]
        draft[Drafting Agent<br/>tailored answers, user voice]
        critic[Critic Agent<br/>grounding + voice + relevance]
        hitl[HITL Gateway<br/>interrupt / resume / approve]
        sub[Submission Handler<br/>assemble package, no submit]
    end

    subgraph retrieval [Retrieval substrate]
        vec[(pgvector index)]
        fts[(Postgres full-text)]
    end

    profile[(Profile store)]

    supervisor --> interp --> retr
    retr --> vec
    retr --> fts
    supervisor --> fit --> profile
    supervisor --> reqs
    supervisor --> draft --> profile
    draft --> critic
    supervisor --> hitl
    supervisor --> sub
```

---

## 7. Workflows (dynamic views)

### 7.1 Search & fit

```mermaid
sequenceDiagram
    actor U as User
    participant S as Supervisor
    participant I as Interpreter
    participant R as Retrieval
    participant F as Fit Analyzer
    participant DB as Postgres

    U->>S: "What's worth applying to this week?"
    S->>I: interpret request + profile
    I-->>S: structured criteria (terms + filters)
    S->>R: hybrid search (vector + FTS + RRF)
    R->>DB: query
    DB-->>R: candidate postings
    R-->>S: ranked candidates
    loop per candidate
        S->>F: analyze_fit(posting, profile)
        F-->>S: matches, gaps, score (grounded)
    end
    S-->>U: ranked matches with reasoning + sources
    Note over S,U: If results are thin, report the thin result.<br/>Filters are not relaxed to produce additional matches.
```

### 7.2 Assisted application (with human-in-the-loop)

```mermaid
sequenceDiagram
    actor U as User
    participant S as Supervisor
    participant RQ as Requirements Agent
    participant D as Drafting Agent
    participant C as Critic Agent
    participant H as HITL Gateway
    participant SUB as Submission Handler

    U->>S: "Apply to job #1234"
    S->>RQ: parse requirements + detect custom questions
    RQ-->>S: required fields + question list + ATS type
    alt questions need personal input
        S->>H: interrupt — request raw material
        H->>U: "This role asks: 'What brings you to us?' — your take?"
        U-->>H: personal raw input
        H-->>S: resume with input
    end
    S->>D: draft answers (grounded in profile + user input)
    D->>C: review each draft
    C-->>D: grounding/voice/relevance verdict
    D-->>S: candidate package
    S->>H: present full package for approval
    H->>U: review + edit + approve
    U-->>H: approved
    H-->>S: approval recorded
    S->>SUB: assemble final package (identity + answers + CV)
    SUB-->>U: prepared package + pre-filled form opened
    Note over U,SUB: User performs the actual submission.<br/>System never submits autonomously.
    U->>S: confirm submitted
    S->>S: record application (status: applied)
```

### 7.3 Application tracking lifecycle

```mermaid
stateDiagram-v2
    [*] --> Drafted
    Drafted --> Applied: user submits
    Drafted --> Abandoned: user discards
    Applied --> InProgress: response received
    Applied --> Refused: rejected
    InProgress --> Interviewing
    Interviewing --> Offer
    Interviewing --> Refused
    Offer --> Accepted
    Offer --> Declined
    Refused --> [*]
    Accepted --> [*]
    Declined --> [*]
    Abandoned --> [*]
```

State is user-editable at any point; the system records timestamps on each transition.

---

## 8. Multi-agent architecture

The application workflow is heterogeneous: each agent has a distinct job, distinct grounding, and distinct tools, which is the criterion for decomposing it into separate agents. A single coordinating **supervisor** owns the state machine and routes between specialized **workers**.

```mermaid
flowchart TB
    SUP{Supervisor<br/>state + routing}
    SUP --> REQ[Requirements Agent]
    SUP --> FIT[Fit Agent]
    SUP --> DRAFT[Drafting Agent]
    DRAFT --> CRIT[Critic Agent]
    SUP --> HITL[[HITL node<br/>interrupt / resume]]
    SUP --> SUB[Submission Handler]
```

| Agent | Job | Grounded in | Tools |
|---|---|---|---|
| Supervisor | Owns the application state machine; routes; enforces approval gates | Graph state | — |
| Requirements | Parse a posting into required fields; detect custom questions; identify ATS type | The posting (untrusted) | — |
| Fit | Match a posting against the profile; surface gaps; score | Posting + profile | `analyze_fit` |
| Drafting | Generate a tailored answer per question in the user's voice | Profile + CV + user's raw input + role context | — |
| Critic | Verify each draft is grounded, on-voice, and answers the question | Draft + profile | — |
| Submission handler | Assemble the approved package; map fields; pre-fill | Approved package | `prepare_submission` |

**Topology decision:** a single supervisor over a typed tool surface fits single-user scope; a federated topology would only be warranted at larger scale. A node that is only a different prompt over the same context is implemented as a function, not promoted to an agent.

---

## 9. Data model & persistence

The **profile** stores identity once and supplies it to every application, removing the re-entry cost described in §2.

```mermaid
erDiagram
    PROFILE ||--o{ APPLICATION : "submits"
    JOB ||--o{ APPLICATION : "targets"
    JOB ||--o{ EVAL_LABEL : "labeled in"
    APPLICATION ||--o{ ANSWER : "contains"

    PROFILE {
        string full_name
        string email
        json links
        json work_history
        text cv_text
        vector cv_embedding
        json target_titles
        string seniority
        json domains_keywords
        int salary_floor
        string currency
        json location_rules
        bool remote_required
    }
    JOB {
        uuid id
        string source
        string source_type
        string ingested_via
        string source_id
        string url
        string title
        string company
        text description
        int salary_min
        int salary_max
        string currency
        string location
        bool remote
        string job_type
        timestamp published_at
        timestamp collected_at
        vector embedding
        string content_hash
    }
    APPLICATION {
        uuid id
        uuid job_id
        string status
        timestamp drafted_at
        timestamp applied_at
        json status_history
        text notes
    }
    ANSWER {
        uuid id
        uuid application_id
        text question
        text user_raw_input
        text draft
        bool approved
    }
    EVAL_LABEL {
        uuid job_id
        string label
        string labeled_by
        text notes
    }
```

`content_hash` drives both deduplication and the embedding cache (an unchanged posting is never re-embedded), and it makes deduplication work *across* sources: the same role pulled from a job board and from an ATS API collapses to one record. `source` names the origin, `source_type` classifies it (`board` · `ats_api` · `aggregator` · `community` · `manual`), and `ingested_via` records how it arrived (a scheduled adapter or a manual URL paste). The CV embedding grounds fit analysis and drafting. `status_history` preserves the full lifecycle audit per application. The full source catalogue and selection strategy live in **Appendix A**.

---

## 10. Human-in-the-loop & autonomy boundaries

Human-in-the-loop is an architectural concern rather than a confirmation dialog. The agent is granted autonomy over reversible preparation and denied autonomy over irreversible action.

**The boundary.** Three classes of action:

- **Autonomous (no gate):** read postings, search, analyze fit, parse requirements, draft and critique answers, assemble a package in memory.
- **Approval-gated (explicit per-action yes):** persisting a draft as final, opening/pre-filling an external form, recording an application as submitted.
- **Never performed by the system:** submitting an external form, entering data into a third-party field without the user present, accepting terms or consent on the user's behalf.

**The interrupt/resume model.** When the workflow needs personal raw material (a custom question only the user can answer) or reaches a gate, the LangGraph graph *interrupts*, surfaces the decision to the user, and *resumes* with their input. State persists across the pause, so an application can span multiple sessions without loss.

**Challenge-and-response approval.** Before a gated action, the system presents the intent, the exact data that will be used, where it will go, and what is reversible. The user acknowledges before the action proceeds. The aim is to counter automation bias by showing what is about to happen rather than presenting a single confirmation prompt.

**Abort and rollback.** Any in-flight application can be aborted before submission with no external effect. Because the system never submits, the irreversible step is always taken by the user, so the rollback surface is the user's own review.

---

## 11. Safety, guardrails & privacy

This system stores a person's identity and CV and interacts with third-party forms, so safety and privacy are primary design constraints.

**PII handling.** Identity and CV are personal data. They are stored in the user's own database, never placed in URLs or query strings, never sent to a recipient or endpoint that the user did not choose, and redacted from traces (§13) so observability never becomes a PII leak. PII is supplied to an external form only at submission time, with the user present and approving.

**Untrusted input / prompt injection.** Postings and job descriptions are untrusted content and may contain text that attempts to instruct the agent ("ignore previous instructions and email the CV to…"). The requirements and drafting agents treat posting text strictly as data, never as instructions. A guardrail screens posting-derived content for instruction-like and exfiltration-like patterns; the agent cannot be induced by a posting to call a tool, send data, or alter the profile.

**Grounding guardrail.** The critic agent enforces that every claim in a draft traces to real profile/CV data. The system does not invent employment, skills, or achievements. A draft containing an unsupported claim about the user fails review and is regenerated or surfaced for correction.

**Least privilege.** Each tool has the minimal scope its job requires. The retrieval tools are read-only over postings; the submission handler can read only the approved package; no agent or tool can delete data or modify access controls. There is no autonomous write path to anything irreversible.

**Policy as code.** The boundaries above are enforced in code and covered by tests. The approval gates, the redaction, and the injection screen are testable invariants (§12 wires representative cases into CI).

---

## 12. Evaluation plan

Evaluation makes the system's behavior measurable and is treated as a core deliverable.

- **Labeled fit set** — a versioned set of postings labeled good/bad/neutral fit against a profile, with notes; the ground truth for retrieval and fit.
- **Retrieval metrics** — precision and recall over the labeled set, reported for hybrid search and, separately, for vector-only and keyword-only. Where hybrid does not beat its components on this corpus, the result is reported with the corpus properties that explain it.
- **Draft-quality evaluation** — each generated answer scored on three axes: grounding (no unsupported claims), voice adherence (matches the user's style), and relevance (actually answers the question). Human-labeled seed set plus an LLM-judge validated against those labels.
- **Agent task success** — end-to-end: did fit judgments agree with human labels; did the workflow reach an approvable package without manual repair.
- **Failure-mode taxonomy** — a documented catalog (wrong filter inference, stale postings surfaced, fit hallucination, ungrounded draft claim, injection attempt) with an example of each and the guardrail that catches it.
- **CI gate** — golden queries and representative safety cases (an injection-laced posting, an ungrounded-claim draft) run on every change via GitHub Actions. A regression fails the build.

---

## 13. Observability & cost

Every query and every application workflow is traced using OpenTelemetry GenAI semantic conventions, exported to Langfuse. Each trace records what was retrieved, what each agent decided, every tool call's input and output, latency per step, and token cost per call — with PII redacted at the export boundary. When an answer or a draft looks wrong, the trace attributes the cause to retrieval, routing, drafting, or the critic.

Cost is reported per query and per application, decomposed by step. The system runs on local Ollama during development and swaps to a paid API for a final quality pass; the local-versus-API quality and cost comparison is a reported deliverable. The content-hash embedding cache is the first measured optimization, with cost-per-query reported before and after.

---

## 14. Tech stack & rationale

| Layer | Technology | Rationale |
|---|---|---|
| Language / API | Python 3.12, FastAPI | Ecosystem fit for the agent/ML libraries used |
| Store & search | PostgreSQL, pgvector, Postgres FTS, RRF | Hybrid search in one system, no second datastore |
| Embeddings | `nomic-embed-text` via Ollama, content-hash cached | Local, zero-cost dev; cache as first optimization |
| Generation | Local (Qwen/Llama) for dev; paid API for final pass | Cost/quality benchmark |
| MCP server | FastMCP | Typed, discoverable tool surface |
| Orchestration | LangGraph | State machine + HITL interrupt/resume |
| Observability | Langfuse + OpenTelemetry | Agent-level tracing |
| Evaluation | Custom harness + RAGAS-style metrics, gated in CI | Measurable behavior; regression gate |
| Ingestion | APScheduler, daily batch | Compliant, simple |
| Deployment | Docker, Fly.io | Reproducible, runnable demo |
| CI | GitHub Actions | Eval + safety regression gate |

---

## 15. Non-functional requirements

Stated as measurable quality scenarios.

| Attribute | Scenario | Target |
|---|---|---|
| Retrieval latency | Hybrid search over the corpus, retrieval only | p95 < 500 ms |
| Query latency | End-to-end NL query incl. fit analysis (API model) | p95 < 8 s |
| Ingestion | Daily batch across all configured sources, idempotent | Completes < 15 min; zero duplicates on re-run (incl. cross-source) |
| Cost | Fit query on cached embeddings, paid API model | Reported; target < $0.05/query |
| Safety invariant | Autonomous external form submissions | **Exactly zero** — enforced in code and CI |
| Privacy invariant | PII in URLs, query strings, or unredacted traces | **Zero occurrences** |
| Grounding | Approved drafts containing unsupported claims about the user | Zero in the eval set; injection cases caught in CI |
| Availability | Single-user demo deployment | Best-effort; runnable on demand |

---

## 16. Deployment & operations

```mermaid
flowchart TB
    subgraph host [Deployment: Fly.io / Docker Compose]
        appc[App container<br/>FastAPI + Web UI + MCP server]
        agentc[Agent container<br/>LangGraph service]
        ingestc[Ingestion container<br/>APScheduler]
        pg[(Postgres + pgvector volume)]
    end
    ollama[(Ollama / paid API)]
    lf[(Langfuse)]
    src[(Remotive / RemoteOK)]

    appc --> agentc --> pg
    ingestc --> src
    ingestc --> pg
    agentc --> ollama
    agentc --> lf
    appc --> lf
```

Ingestion is idempotent and retried on transient failure. Containers expose health checks. Configuration (model selection, source list, API keys) is environment-based; secrets are never committed. The deployed demo exposes the web UI and the MCP server endpoint.

---

## 17. Roadmap / phased delivery

The scope is larger than one month and is delivered in independently shippable phases.

**Phase 1 — Discovery + reliability spine (v1, ~1 month).**
Ingestion, indexing, hybrid search, the MCP tool surface, the LangGraph search/fit graph, the evaluation harness, observability, cost panel, deployment, README. This phase is a complete, standalone deliverable.

**Phase 2 — Assisted application + multi-agent (v2, ~2–4 additional weeks).**
Requirements/drafting/critic agents, the HITL interrupt/resume approval flow, the submission handler with assisted (human-gated) submission, the guardrail and privacy enforcement, and draft-quality evaluation.

**Phase 3 — Tracking + polish.**
The application tracking board, lifecycle state management, and end-to-end packaging.

**Deferred beyond v2:** autonomous form submission (out of scope by design), additional sources, multi-user, cross-session memory beyond the application workflow.

Within each phase, the protected core is the working path plus its evaluation and safety gates; breadth (extra optimizations, larger eval sets) is cut before the core path.

---

## 18. Repository layout

```
app/            FastAPI app, web UI, MCP server entrypoint
src/
  agents/       supervisor + worker agents (LangGraph)
  retrieval/    hybrid search, RRF, embeddings
  fit/          RAG fit analysis
  application/  requirements, drafting, critic, submission handler
  guardrails/   injection screen, grounding check, PII redaction
eval/           labeled sets, metrics, golden queries, CI eval runner
infra/          Docker, deployment, scheduler
docs/
  diagrams/     Mermaid sources
```

Version control from the first commit.

---

## 19. Glossary & references

**Glossary.** *MCP* — Model Context Protocol, a standard for exposing tools to agents. *RAG* — retrieval-augmented generation. *RRF* — Reciprocal Rank Fusion, the method for fusing vector and keyword rankings. *HITL* — human-in-the-loop. *ATS* — applicant tracking system. *pgvector* — Postgres extension for vector similarity. *OTel* — OpenTelemetry. *NFR* — non-functional requirement.

**References.** C4 model (Simon Brown) for the context/container/component views; OpenTelemetry GenAI semantic conventions for trace structure; LangGraph documentation for interrupt/resume and persistence; Model Context Protocol specification for the tool surface.

---

## Appendix A — Job data sources & ingestion strategy

The system does not hard-code a source list. Ingestion is built around a **source-adapter interface**: each source is a small adapter that fetches, maps its payload to the common `JOB` schema (§9), and reports what fields it can and cannot supply. Adding, removing, or reordering sources is configuration plus one adapter, never a change to the core. This appendix catalogues the candidate sources surveyed and the strategy for choosing among them; it deliberately leaves the *enabled set* open, because the right mix depends on which roles the user is actually targeting and on terms that must be re-verified at integration time.

### A.1 The adapter contract

Every adapter implements the same shape:

- `fetch()` — pull current postings (paged), respecting the source's rate limits and cadence.
- `map()` — normalise to the common schema; populate `source`, `source_type`, `ingested_via`, and whatever of `salary`, `remote`, `published_at`, `description` the source provides.
- `capabilities()` — declare which schema fields are reliable, so retrieval and fit analysis know when a field is missing versus genuinely empty.
- `terms()` — record the storage/attribution/rate constraints the adapter must honour (link-back, no-resubmission, trial limits), enforced in code.

A posting that arrives without salary or a remote flag is expected; the system treats those fields as unknown and does not fabricate them.

### A.2 Source classes (selection guidance, not a fixed list)

The surveyed sources fall into classes with very different value-to-effort and compliance profiles. The intended posture for each:

- **Per-ATS public boards (Greenhouse, Lever, Ashby, SmartRecruiters, Workable, Recruitee, Personio).** High quality-to-effort and the recommended backbone: public, no-auth, structured, and rich (Lever and Ashby carry salary and a remote/workplace flag; Ashby is especially strong for AI-native startups). They require a curated list of target companies and are the same systems the Phase-2 application workflow interacts with. Coverage is only as broad as the company list.
- **Remote-first boards & aggregators (Himalayas, Arbeitnow, Remotive, RemoteOK, The Muse).** Good breadth for remote roles with low integration effort; Himalayas adds seniority and salary filters, Arbeitnow adds remote and visa-sponsorship flags useful for the EU target. Each carries attribution and no-resubmission obligations the adapter must honour. Used for volume and as a baseline, with quality skewing more mid-level than the ATS boards.
- **Community (HN "Who is hiring" via the Algolia API).** High signal for senior backend/AI startup roles with compensation often stated, at the cost of LLM parsing of free-form comments — an unstructured-extraction task the system's stack already supports.
- **Aggregator/meta APIs (Adzuna, JSearch, commercial ATS-aggregators).** Broad reach including roles sourced from otherwise-inaccessible boards, but with sharper trade-offs: truncated descriptions, predicted rather than stated salary, trial/commercial-use limits, or scraped upstream provenance. Optional and evaluated individually against terms before enabling; not part of the compliant backbone by default.
- **Government (USAJobs).** Narrow (US federal) and low fit for this target; available if relevant.
- **Hostile / scraping-only (LinkedIn, Indeed, Wellfound, per-tenant Workday/Oracle/iCIMS).** Out of scope by policy (§2.4). These hold many high-value senior roles that no compliant API reaches cleanly; those roles are covered by **ingest-by-URL** (A.3) rather than a terms violation.
- **Dead/deprecated (GitHub Jobs, Stack Overflow Jobs, the old Indeed Publisher and AngelList APIs).** Not used; listed so they are not re-investigated.

### A.3 Ingest-by-URL

Because some relevant senior roles live behind sources the system will not scrape, the user can paste a single job URL to ingest that one posting. The fetcher retrieves the page, an extraction step maps it to the common schema (with the same "unknown, never fabricated" rule), and the posting enters the corpus tagged `ingested_via = manual`. This covers roles outside the compliant APIs without changing the compliance posture, and lets the corpus reflect the user's actual search rather than only what open APIs carry.

### A.4 What is intentionally left open

Three decisions are deferred to implementation, because they depend on the user's live targeting and on terms verified at build time:

1. **The enabled source set and ordering** — which adapters ship first. The recommended starting backbone is a handful of per-ATS boards for target companies plus one or two remote aggregators, but the set is configuration.
2. **The company seed list for ATS boards** — derived from the user's target companies; expected to grow over the search.
3. **Whether any aggregator is enabled** — decided per source after reading its current terms; off by default.

All terms, rate limits, and field availability in the survey that informed this appendix must be re-verified against each provider's official documentation before an adapter is shipped; this space changes frequently.
