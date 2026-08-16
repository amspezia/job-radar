# CV-Parsing Eval Failure Analysis

Root-cause analysis of the 11 `[FAIL]` checks from the most recent
`job-radar-eval-profile-parsing` run (22/33 checks passed across 5 personas).
Each entry below groups failures that share one root cause, classified as:

- **(a)** eval expectation itself is wrong (ground truth or matching logic)
- **(b)** genuine LLM parsing failure (prompt is fine, model still erred)
- **(c)** missing or ambiguous prompt instruction in `parse.py`'s `_PROMPT`

---

## 1. `check_work_history_years` matches by company name only, so same-employer roles collide

- **Classification:** (a) eval expectation/script is wrong — bug in
  `eval_profile_parsing.py`, not in `parse_cv`'s output.
- **Affected checks:**
  - `alice-rust-senior: work_history.years[Meridian Pay]` — expected 2.5, got 4.0
  - `bob-python-ml-mid: work_history.years[Vireo Health Analytics]` — expected 1.58, got 2.5
  - `carol-ts-fullstack-mid: work_history.years[Bazaario]` — expected 1.67, got 3.0
  - `eve-data-junior: work_history.years[Cartway Retail]` — expected 0.58, got 1.375

- **Evidence:**

  `check_work_history_years` in [eval_profile_parsing.py:138-145](eval/eval_profile_parsing.py):
  ```python
  match = next(
      (
          p
          for p in parsed_entries
          if _normalize(p.get("company") or "") == _normalize(gt["company"])
      ),
      None,
  )
  ```
  This searches the **full, unfiltered** `parsed_entries` list by company name
  only — no role, no start date, and no removal of an entry once it has been
  used to satisfy one ground-truth check. All 4 affected personas are exactly
  the ones the fixture design deliberately gives **two work_history entries at
  the same company** (README: *"Two personas (dave, alice) share an employer
  name (Cartway Retail / used again for eve) at non-overlapping times —
  deliberate, not an error"* — and Bazaario/Vireo are each single-employer,
  two-role CVs by construction). In every affected case, the ground-truth
  entry being checked is the **closed** role, but `next()` keeps returning the
  same first match in `parsed_entries` — which is the **ongoing** role's
  entry — because nothing consumes it after use.

  The smoking gun is **`dave-go-platform-staff`**, which has two *closed*
  Cartway Retail roles (Senior Infrastructure Engineer, expected 3.17; Site
  Reliability Engineer, expected 3.33) and reports the *identical* `got 3.5`
  for both:
  ```
  [PASS] work_history.years[Cartway Retail]: expected 3.17, got 3.5 (±0.5)
  [PASS] work_history.years[Cartway Retail]: expected 3.33, got 3.5 (±0.5)
  ```
  Two different ground-truth date ranges cannot both genuinely equal the same
  parsed duration — this only happens if `next()` matched the same single
  parsed record twice. Dave's case happens to pass both times purely because
  3.5 falls within ±0.5 of both 3.17 and 3.33; the other four personas aren't
  so lucky.

  This is corroborated quantitatively: computing each affected persona's
  *ongoing*-role duration (start → 2026-08-16, the eval's `today()`) lands far
  closer to the reported "got" value than the closed role's ground truth does:
  | persona | ongoing-role duration (computed) | reported "got" | closed-role gt |
  |---|---|---|---|
  | alice (Meridian Pay) | 3.58 | 4.0 | 2.5 |
  | bob (Vireo) | 2.42 | 2.5 | 1.58 |
  | carol (Bazaario) | 3.17 | 3.0 | 1.67 |
  | eve (Cartway) | 1.58 | 1.375 | 0.58 |

  In every row, "got" sits close to the ongoing role's duration, not the
  closed role being checked — the eval is comparing the wrong pair.

- **Suggested fix:** In `eval/eval_profile_parsing.py::check_work_history_years`,
  match parsed entries to ground-truth entries using `(company, start)` (or at
  minimum `(company, role)`), not company alone, and remove/mark a parsed
  entry as consumed once it's matched so a second same-company ground-truth
  entry can't re-match it. E.g. track a `used: set[int]` of matched parsed
  indices, and when multiple parsed entries share a company, prefer the one
  whose `start` is closest to `gt["start"]`.

---

## 2. `target_titles` overshoots into unsupported management/lead titles

- **Classification:** (c) missing/ambiguous prompt instruction
- **Affected checks:**
  - `alice-rust-senior: target_titles` — forbidden `Principal Backend Engineer`, `Engineering Manager` present
  - `carol-ts-fullstack-mid: target_titles` — forbidden `Technical Lead`, `Engineering Manager` present

- **Evidence:** Both ground-truth files explicitly call out that these
  personas have **no management-track evidence**. Alice's notes: *"'Engineering
  Manager' has zero support in the CV (no stated interest or people-management
  experience) and 'Principal' is two levels above her actual seniority —
  either should count as an error."* Carol's notes: *"No leadership or
  management signal anywhere in the CV, so any lead/manager/staff-track title
  is unsupported."*

  Notably, in both cases the model got `seniority` **correct** (`senior` for
  Alice, `mid` for Carol — both PASS), so this isn't a seniority-inference
  failure; it's the `target_titles` field independently drifting past that
  established level. The prompt's only guardrail is about the *low* end:
  > "Do NOT simply copy titles that appear in `work_history` — in particular,
  > exclude early-career titles (e.g. 'Intern', 'Junior ...') that no longer
  > match the candidate's current seniority"

  There is no equivalent instruction capping the *high* end — nothing says a
  suggested title must stay within one level of the established `seniority`,
  and nothing defines what evidence (e.g. explicit people-management
  responsibility, not just "own[ing] a service" or "mentor[ing] two
  engineers") is required to justify a management-track title at all. Alice's
  CV bullet *"Mentor two mid-level engineers; run the team's incident review
  process"* is exactly the kind of IC-leadership-adjacent language that, with
  no explicit rule to lean on, plausibly gets over-read by the model as
  license to suggest "Engineering Manager" and jump to "Principal."

- **Suggested fix:** In `src/job_radar/profile/parse.py`'s `_PROMPT`, extend
  the `target_titles` paragraph with an explicit upper bound and an
  evidence bar, e.g.: *"Do not suggest a title more than one level above the
  `seniority` you determined, and do not suggest people-management titles
  (e.g. 'Engineering Manager', 'Lead') unless the CV states actual
  people-management responsibility — mentoring or owning a service alone is
  not sufficient."*

---

## 3. `tech_stack` misses terms stated only in a bullet, not the SKILLS line

- **Classification:** (c) missing/ambiguous prompt instruction
- **Affected checks:**
  - `alice-rust-senior: tech_stack` — missing `c++`
  - `dave-go-platform-staff: tech_stack` — missing `python`

- **Evidence:** Both misses are the exact deliberately-planted test case
  described in the README: *"Two personas (dave, alice) deliberately place
  one stack term only in a bullet, never in the SKILLS line, to test whether
  extraction reads prose or just harvests the skills list."* Alice's
  ground-truth notes: *"C++ appears only in a bullet ('Maintained a C++
  telemetry ingestion service'), not the SKILLS line."* Dave's: *"'Python'
  appears only in a bullet ('CI/CD for ~30 services (Go, some Python)'), not
  the SKILLS line — a second deliberate bullet-only test case."* The model
  failed this exact test in both of the two personas it was posed to (2/2),
  and in both cases `precision=1.0` (nothing invented) — so the model isn't
  guessing wildly, it's specifically narrowing its search to one location.

  This is best explained by a genuine gap in `_PROMPT`: seniority,
  target_titles, domains, and work_history each get a dedicated instruction
  paragraph, but **`tech_stack` gets none at all** — it's not mentioned
  anywhere in `_PROMPT` outside the schema itself. With an explicit
  `SKILLS` header present in every persona's CV and no countervailing
  instruction, the model has an obvious structural cue to anchor on and no
  instruction telling it to also mine prose bullets.

- **Suggested fix:** Add a dedicated `tech_stack` paragraph to `_PROMPT` in
  `src/job_radar/profile/parse.py`, e.g.: *"For `tech_stack`, collect every
  technology explicitly named anywhere in the CV — not only from a SKILLS
  section, but also from experience bullets and project descriptions."*

---

## 4. No instruction for computing `years_experience`

- **Classification:** (c) missing/ambiguous prompt instruction
- **Affected checks:**
  - `alice-rust-senior: years_experience` — expected ~8.0 (from Aug 2018), got 6.5

- **Evidence:** `_PROMPT` gives a detailed rule for computing each
  work_history entry's `years` for an ongoing role (*"For an entry still in
  progress, compute `years` as the time from `start` to today's date above"*),
  but **never mentions `years_experience` at all** — no instruction for what
  it means (total career span? sum of role durations?) or which date it
  should be measured from (`career_start`, i.e. the earliest role's start).

  This isn't visible only in Alice's outright failure — it's a systemic
  pattern across all 5 personas, all landing at-or-below the expected value:

  | persona | expected | got | diff |
  |---|---|---|---|
  | alice | 8.0 | 6.5 | **1.5 (FAIL)** |
  | bob | 5.25 | 5.0 | 0.25 (pass) |
  | carol | 4.92 | 4.5 | 0.42 (pass) |
  | dave | 11.0 | 11.0 | 0.0 (pass) |
  | eve | 2.25 | 2.0 | 0.25 (pass) |

  Four of five are absorbed by the eval's ±0.5 tolerance, but the consistent
  undercount direction indicates the model is guessing at a computation the
  prompt never specifies, rather than reliably applying `career_start → today`
  the way the ground truth (and `personas_lib.years_between`) does. Alice is
  the persona with the most work_history entries (3, vs. 2 for everyone else)
  and the only one where the gap breaks tolerance — consistent with error
  compounding across a computation the model has no defined procedure for.

- **Suggested fix:** Add an explicit `years_experience` rule to `_PROMPT` in
  `src/job_radar/profile/parse.py`, e.g.: *"For `years_experience`, compute
  the time from the **start date of the earliest work_history entry** to
  today's date above, including all roles — do not exclude early-career
  roles from this total."*

---

## 5. `domains` conflates business vertical with technique/function

- **Classification:** (c) missing/ambiguous prompt instruction
- **Affected checks:**
  - `bob-python-ml-mid: domains` — forbidden `risk prediction`, `data analysis` present

- **Evidence:** Bob's ground-truth notes are explicit that this is a prompt
  ambiguity, not just a model slip: *"'Risk prediction' and 'data analysis'
  describe technique/function, not a business domain — including them
  conflates the two categories **the parse_cv prompt's own example list
  (fintech, proptech, telecom, ...) is ambiguous about**."*

  The prompt's `domains` paragraph:
  > "For `domains`, name the business/problem areas the candidate has worked
  > in (e.g. 'fintech', 'proptech', 'telecom', 'LLM/agent infrastructure')."

  Three of the four examples (fintech, proptech, telecom) are business
  verticals; the fourth, "LLM/agent infrastructure," is a technical/functional
  area, not an industry. That inconsistency in the prompt's own example set
  is exactly the ambiguity the ground-truth author flagged — it hands the
  model implicit license to output functional descriptions like "risk
  prediction" and "data analysis" for Bob's CV as if they belonged in the
  same category as "healthcare."

- **Suggested fix:** In `src/job_radar/profile/parse.py`'s `_PROMPT`, replace
  "LLM/agent infrastructure" with a genuine business-vertical example (or add
  a clarifying clause), e.g.: *"...name the business/problem areas the
  candidate has worked in (e.g. 'fintech', 'proptech', 'telecom',
  'healthcare') — the industry or problem space the employer operates in, not
  the technique or function the candidate performed (e.g. not 'risk
  prediction' or 'data analysis')."*

---

## 6. `seniority` returned `null` despite an explicit, unambiguous title match

- **Classification:** (b) genuine LLM parsing failure
- **Affected checks:**
  - `eve-data-junior: seniority` — expected `junior`, got `None`

- **Evidence:** Eve's current title is verbatim "Junior Data Engineer" (CV:
  *"Junior Data Engineer — Cartway Retail | Jan 2025 – Present"*), with a
  single prior internship and ~2.25 years of total experience — the least
  ambiguous seniority call of the five personas. The prompt's seniority rule
  (*"return exactly one of: intern / junior / mid / senior / staff /
  principal — inferred from years of experience and the most recent job
  title. Return null if you cannot confidently determine the level"*) is
  clear and gives the model everything it needs here; there's no missing
  category, no conflicting instruction, and no edge case like Alice's or
  Dave's more senior, judgment-call personas. Eve's own ground-truth notes
  flag this as a previously-observed, not newly-hypothesized, issue: *"This is
  the easiest of the five seniority calls and the one most worth checking
  directly — the live parser returned null here on the first real run,
  despite the word 'Junior' being in her current title."* With no plausible
  prompt gap to point to on the clearest case in the set, this is a plain
  model miss.

- **Suggested fix:** Not a prompt or eval fix — treat as a known model
  failure mode to monitor. If it recurs across more real CVs (not just this
  one persona), consider adding one-shot guidance or lowering
  temperature/adding a self-check step in `generate()`, but a single miss on
  one persona doesn't yet justify a prompt change.

---

## Summary

| # | Root cause | Classification | Failures explained |
|---|---|---|---|
| 1 | `check_work_history_years` matches same-company roles by name only, colliding closed/ongoing entries | (a) eval script bug | 4 — alice/Meridian Pay, bob/Vireo, carol/Bazaario, eve/Cartway |
| 2 | `target_titles` overshoots into unsupported management/lead titles | (c) missing prompt instruction | 2 — alice/target_titles, carol/target_titles |
| 3 | `tech_stack` misses terms stated only in a bullet, not the SKILLS line | (c) missing prompt instruction | 2 — alice/tech_stack, dave/tech_stack |
| 4 | No instruction for computing `years_experience` | (c) missing prompt instruction | 1 — alice/years_experience |
| 5 | `domains` prompt example list conflates business vertical with technique | (c) missing prompt instruction | 1 — bob/domains |
| 6 | `seniority` null despite unambiguous "Junior" title | (b) genuine LLM failure | 1 — eve/seniority |

**Total: 11/11 failures accounted for.**
