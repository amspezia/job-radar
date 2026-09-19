# Rating guide (0–3) for the blind labeling

You grade each job for **you, as in the current profile**: would it belong on your results page, and
would you apply? You see only the 400-character view of the posting; grade what is visible.

Profile in one line: Python / AI-agent / backend engineer (targets: Software Engineer, AI Engineer,
Backend Developer); fintech and telecom; remote, Brazil or worldwide; English postings only.
Stack: Python, Kotlin, SQL, JavaScript/React, MCP, RAG, vector search, multi-agent orchestration,
LLM evaluation, Langfuse, OpenTelemetry, FastAPI, Spring Boot, Kafka, PostgreSQL, Docker, Kubernetes.

## What to check, in this order

1. **Role:** is the core job one of your target titles?
2. **Stack:** does it overlap your own stack? Generic skills (APIs, SQL, cloud, microservices) do not count.
3. **Level:** compare with your own seniority; decide once how you treat "one level above" and stay consistent.
4. **Domain:** fintech and telecom are a plus; an unfamiliar domain alone is only a small deduction.
5. **Would you apply?** If yes, it is at least a 2.

## The grades

| Grade | Meaning | Typical for you |
|---|---|---|
| **3** | Exactly what you look for; you would apply without hesitation | Backend or AI Engineer on Python with agents/MCP/RAG/evaluation, or Python/FastAPI/Kafka distributed services, at your level |
| **2** | You would apply; one thing is off | Right role and stack but one level off, or missing 1–2 stack items; Kotlin/Spring Boot backend; ML or full-stack role with a strong Python back end |
| **1** | Adjacent; you would think twice | Data engineering, DevOps/SRE/platform only, solutions architect, front-end-heavy full-stack, ML research, QA automation; a good stack match at a clearly wrong level; a role built on a stack you do not use (Go-only, C#, Node-only, Java-only) |
| **0** | Not for you | Sales, management (Manager, Director, CTO), non-engineering, front-end or mobile only, .NET/PHP/Ruby stacks |

## Rules of thumb

- **2 vs 1** is the boundary that matters most (grade 2 and above counts as "relevant"). Ask: would I apply?
- **3 vs 2:** when unsure, lean 2. Grade 3 should be rare.
- **Skip (`s`)** postings that are not in English, and ones with no usable content (for example the fields
  only repeat "all skills, technologies, qualifications…" or "what the person will actually do…").
- **Ignore what you cannot see:** location is already filtered, and salary and company prestige are not part of the grade.
- **Same job listed twice → same grade.**
- **Add a short note** when a grade was a close call; it is stored with the label.

## How the grades are used

Grade ≥ 2 = relevant for precision, recall and AP. nDCG weights grades as 3 → 7, 2 → 3, 1 → 1, 0 → 0, so
a 3 near the top counts most.
