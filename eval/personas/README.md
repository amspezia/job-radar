# Synthetic personas

Five hand-authored, non-PII developer personas used as ground truth for the CV-parsing
eval (B3) and as fixture profiles for the synthetic multi-persona retrieval eval (B1) —
see `docs/plans/FIXES_IMPLEMENTATION_PLAN.md`. Spread deliberately across the full
seniority ladder (junior → staff) and across stacks/domains with different lexical
properties (a rare, discriminative stack like Rust vs. an everywhere-term like Python),
not clustered at one level.

Each persona has two files:

- **`<id>.cv.txt`** — a realistic CV, written first, as a real person's CV would read
  (narrative, dates, achievements), not a restatement of the ground truth fields. Every
  persona shows genuine career progression from an early-career title to their current
  level, and every current role is open-ended ("Present"), specifically to exercise
  `parse_cv`'s handling of (a) excluding stale early-career titles from `target_titles`
  and (b) computing duration for an ongoing role against today's date rather than the
  model's own sense of "now."
- **`<id>.ground_truth.json`** — the answer key, derived by close-reading the CV
  *after* it was written, not the other way around. Writing the CV first and deriving
  ground truth second avoids the ground truth being a trivial restatement the CV is
  suspiciously aligned to.

## Ground truth schema conventions

- **`years` is only given for closed (dated) roles.** Those durations are fixed and
  time-invariant — computed once, correct forever. The current (Present) role's `years`
  is `null` with a `years_note` saying to compute it dynamically from `start` against
  the eval run date. `years_experience` is handled the same way via `career_start` +
  `years_experience_note`, instead of a hardcoded total. A hardcoded number for an
  ongoing role goes stale the day after it's written; computing it at eval time doesn't.
- **`target_titles` and `domains` are acceptable-sets, not exact strings** —
  `expected_any_of` (should appear, loosely/synonym-matched) and `must_not_include`
  (a real error if present, not just an imprecision), each with `notes` explaining
  *why* a given title/domain is or isn't supported by that specific CV. Every
  `must_not_include` entry should be traceable to something concrete: a stale
  early-career title, an unsupported career-track jump (e.g. an individual-contributor
  CV with zero management signal producing "Engineering Manager"), or a domain that
  conflates business vertical with technical function.
- **`tech_stack` is exact-set, not loose** — every item is literally stated somewhere
  in the CV (skills line or a bullet); nothing is inferred, matching `parse_cv`'s own
  "do NOT invent" instruction. Two personas (`dave`, `alice`) deliberately place one
  stack term only in a bullet, never in the SKILLS line, to test whether extraction
  reads prose or just harvests the skills list.
- **`search_preferences`** (location, remote, salary, currency) are authored directly,
  not derived from the CV — a CV doesn't state these, and `profile/loader.py` doesn't
  extract them from one either (they're defaulted on first profile creation and
  preserved on reload). They exist here only for B1's synthetic-profile construction.

## Known deliberate cross-persona design choices

- `dave-go-platform-staff` is the one persona with **no strong single domain** —
  platform engineering is domain-agnostic, and his two employers (streaming media,
  retail) are different verticals. This is intentional: it tests whether `parse_cv`
  recognizes thin/absent domain signal instead of listing every employer's industry as
  if it were the candidate's specialization.
- `eve-data-junior` is the one persona where **keeping "Junior" in a target title is
  correct**, not stale — she genuinely is junior. The other four personas test
  "exclude a title inconsistent with current seniority"; Eve tests that the same rule
  doesn't over-fire and strip an accurate current-level title.
- Two personas (`dave`, `alice`) share an employer name (`Cartway Retail` / used again
  for `eve`) at non-overlapping times — deliberate, not an error; real corpora have
  multiple people from the same company.
