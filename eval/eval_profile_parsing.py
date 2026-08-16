"""CV-parsing quality eval (B3) — parse_cv() against the 5 personas' ground truth.

    uv run job-radar-eval-profile-parsing

The most upstream, least-checked artifact in the system (see docs/ASSESSMENT.md's
"Profile / CV analyzer" section: everything downstream — the lexical query, the
HyDE prompt, the fit prompt, the domain-relevance scoring dimension — trusts
`parse_cv`'s output unconditionally, and until now nothing validated it at all).

Reuses eval/personas/*.cv.txt + *.ground_truth.json — the same fixtures B1's
synthetic retrieval eval uses — rather than a second fixture set; see
eval/personas/README.md for why the ground truth was derived from the CV, not
the other way around, and why `years` fields are only fixed for closed roles.

Checks, per persona:
  - seniority: exact match against the ladder
  - target_titles / domains: acceptable-set match (expected_any_of / must_not_include)
  - tech_stack: exact-set precision/recall
  - work_history years (closed roles) and years_experience (dynamic): numeric
    tolerance against ground truth, computed fresh each run so nothing goes stale
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field

from eval import personas_lib
from job_radar.profile.parse import parse_cv
from job_radar.profile.schema import StructuredProfile

logger = logging.getLogger(__name__)

_YEARS_TOLERANCE = 0.5  # years; a rounding/approximation slop this project treats as acceptable
_ONGOING_END_MARKERS = {"", "present", "current", "now"}
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})(?:-\d{2})?$")


def _is_ongoing(end: str | None) -> bool:
    return end is None or end.strip().lower() in _ONGOING_END_MARKERS


def _parse_any_month_year(text: str | None) -> tuple[int, int] | None:
    """Best-effort (year, month) extraction. The ground truth always uses the
    fixed 'Mon YYYY' format (see personas_lib.parse_mon_year), but parse_cv's
    own model output has been observed to ignore the schema's 'as written on
    the CV' instruction and emit ISO-ish strings instead (e.g. '2023-01-01',
    '2019-01') — comparing raw normalized strings between the two silently
    never matches. Tolerating both formats here, rather than only the
    ground truth's, is what makes (company, start) matching actually work.
    Returns None if `text` matches neither shape.
    """
    if not text:
        return None
    text = text.strip()
    try:
        d = personas_lib.parse_mon_year(text)
        return (d.year, d.month)
    except (ValueError, KeyError):
        pass
    m = _ISO_DATE_RE.match(text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _matches_expected(parsed_item: str, expected_item: str) -> bool:
    """Generous: either direction may contain the other (handles abbreviation-
    style variants like 'ML Engineer' vs a differently-phrased expected item),
    good enough for this fixture's small, hand-curated lists without pulling in
    a fuzzy-matching dependency for five personas' worth of checks.
    """
    p, e = _normalize(parsed_item), _normalize(expected_item)
    if p == e:
        return True
    return len(e) >= 4 and (e in p or p in e)


def _matches_forbidden(parsed_item: str, forbidden_item: str) -> bool:
    """Asymmetric, deliberately NOT the same containment test as _matches_expected:
    only flags when the forbidden phrase is actually present in (or equal to)
    what was parsed. A parsed item that merely happens to be a short substring
    of a longer forbidden phrase must not be flagged — e.g. 'Backend Engineer'
    is fine even though 'Principal Backend Engineer' is forbidden; the reverse
    (parsed='Principal Backend Engineer', forbidden='Backend Engineer') is a
    real violation and must still be caught, hence forbidden-in-parsed only.

    Mirrors _matches_expected's len>=4 guard: a forbidden term shorter than
    that is only matched on exact equality, so a short forbidden word (e.g.
    'Lead') can't false-positive against an unrelated parsed string that
    merely contains it as a substring.
    """
    p, f = _normalize(parsed_item), _normalize(forbidden_item)
    if p == f:
        return True
    return len(f) >= 4 and f in p


@dataclass
class FieldCheck:
    field: str
    passed: bool
    detail: str


@dataclass
class PersonaReport:
    persona_id: str
    checks: list[FieldCheck] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)


def check_seniority(parsed: str | None, ground_truth: str) -> FieldCheck:
    ok = parsed == ground_truth
    return FieldCheck("seniority", ok, f"expected {ground_truth!r}, got {parsed!r}")


def check_acceptable_set(field_name: str, parsed: list[str], rule: dict) -> FieldCheck:
    """Shared logic for target_titles and domains: expected_any_of / must_not_include /
    optional max_items (e.g. dave's domains rule, where listing every employer's
    industry — instead of recognizing thin/absent domain signal — is itself the
    failure mode being tested, not just an imprecision expected_any_of/
    must_not_include can catch)."""
    expected = rule.get("expected_any_of", [])
    forbidden = rule.get("must_not_include", [])
    max_items = rule.get("max_items")

    violations = [p for p in parsed if any(_matches_forbidden(p, f) for f in forbidden)]
    has_expected = not expected or any(_matches_expected(p, e) for p in parsed for e in expected)
    over_inclusive = max_items is not None and len(parsed) > max_items

    ok = has_expected and not violations and not over_inclusive
    detail = f"parsed={parsed!r}"
    if not has_expected:
        detail += f" — none of expected_any_of={expected!r} present"
    if violations:
        detail += f" — forbidden term(s) present: {violations!r}"
    if over_inclusive:
        detail += f" — over-inclusive: {len(parsed)} items > max_items={max_items}"
    return FieldCheck(field_name, ok, detail)


def check_tech_stack(parsed: list[str], ground_truth: list[str]) -> FieldCheck:
    parsed_norm = {_normalize(p) for p in parsed}
    gt_norm = {_normalize(g) for g in ground_truth}
    missing = gt_norm - parsed_norm
    extra = parsed_norm - gt_norm
    recall = 1.0 if not gt_norm else round(len(gt_norm & parsed_norm) / len(gt_norm), 2)
    precision = 1.0 if not parsed_norm else round(len(gt_norm & parsed_norm) / len(parsed_norm), 2)
    ok = not missing and not extra
    detail = f"recall={recall} precision={precision}"
    if missing:
        detail += f" missing={sorted(missing)!r}"
    if extra:
        detail += f" extra(possibly invented)={sorted(extra)!r}"
    return FieldCheck("tech_stack", ok, detail)


def check_work_history_years(
    parsed_entries: list[dict], gt_entries: list[dict]
) -> list[FieldCheck]:
    """Only checks CLOSED (dated, non-Present) ground-truth roles — those have a
    fixed, time-invariant expected `years`. Matches by (company, start); if a
    ground-truth company has no matching parsed entry, that's a check failure
    in its own right (a dropped/misparsed role), not a skip.

    Two personas (alice, dave) deliberately have multiple work_history entries
    at the same company (see eval/personas/README.md) — matching by company
    name alone would let a single parsed entry (typically the still-ongoing
    role) satisfy every ground-truth check for that company, silently
    comparing the wrong pair of dates. Two things are needed to actually fix
    that, not just one: (1) a parsed entry that's still ongoing (`end` is
    null/"Present"/etc.) is never a valid candidate for a CLOSED ground-truth
    role, since this function only ever checks closed roles — without this,
    a same-company fallback can still grab the ongoing entry ahead of the
    correct closed one; and (2) start-date comparison must tolerate the
    model's actual output formats (observed to ignore the schema's "as
    written on the CV" instruction and emit ISO-ish strings), not just the
    ground truth's fixed 'Mon YYYY' format, via `_parse_any_month_year`.
    """
    checks = []
    used_indices: set[int] = set()
    for gt in gt_entries:
        if (
            gt["years"] is None
        ):  # ongoing role — not this function's job, see check_years_experience
            continue
        gt_company = _normalize(gt["company"])
        gt_start = _parse_any_month_year(gt.get("start"))

        candidates = [
            (i, p)
            for i, p in enumerate(parsed_entries)
            if i not in used_indices
            and _normalize(p.get("company") or "") == gt_company
            and not _is_ongoing(p.get("end"))
        ]
        match_idx = next(
            (
                i
                for i, p in candidates
                if gt_start and _parse_any_month_year(p.get("start")) == gt_start
            ),
            None,
        )
        if match_idx is None and candidates:
            match_idx = candidates[0][0]  # same-company fallback, first unused closed entry

        label = f"work_history.years[{gt['company']}]"
        if match_idx is None:
            checks.append(
                FieldCheck(label, False, f"no parsed entry for company {gt['company']!r}")
            )
            continue
        used_indices.add(match_idx)
        parsed_years = parsed_entries[match_idx].get("years")
        if parsed_years is None:
            checks.append(FieldCheck(label, False, "parsed years is null"))
            continue
        ok = abs(float(parsed_years) - gt["years"]) <= _YEARS_TOLERANCE
        checks.append(
            FieldCheck(
                label, ok, f"expected {gt['years']}, got {parsed_years} (±{_YEARS_TOLERANCE})"
            )
        )
    return checks


def check_years_experience(parsed_value: float | None, career_start: str, today) -> FieldCheck:
    expected = personas_lib.years_between(personas_lib.parse_mon_year(career_start), today)
    if parsed_value is None:
        return FieldCheck("years_experience", False, f"expected ~{expected}, got null")
    ok = abs(parsed_value - expected) <= _YEARS_TOLERANCE
    return FieldCheck(
        "years_experience", ok, f"expected ~{expected} (from {career_start}), got {parsed_value}"
    )


def check_highlights_verbatim(cv_text: str, parsed_entries: list[dict]) -> FieldCheck:
    """The prompt requires `highlights` to be verbatim CV quotes, not paraphrases —
    the one check in this eval that directly tests for fabrication rather than
    imprecision, which matters given nothing downstream re-validates work_history
    against the source CV (see docs/ASSESSMENT.md's note that parse_cv's output is
    trusted unconditionally)."""
    cv_norm = _normalize(cv_text)
    fabricated = [
        h
        for entry in parsed_entries
        for h in entry.get("highlights", [])
        if _normalize(h) not in cv_norm
    ]
    ok = not fabricated
    detail = "all highlights verbatim" if ok else f"not found in CV text: {fabricated!r}"
    return FieldCheck("highlights_verbatim", ok, detail)


def build_report(
    persona_id: str, parsed: StructuredProfile, ground_truth: dict, cv_text: str
) -> PersonaReport:
    report = PersonaReport(persona_id)
    today = personas_lib.today()

    report.checks.append(check_seniority(parsed.seniority, ground_truth["seniority"]))
    report.checks.append(
        check_acceptable_set("target_titles", parsed.target_titles, ground_truth["target_titles"])
    )
    report.checks.append(check_acceptable_set("domains", parsed.domains, ground_truth["domains"]))
    report.checks.append(check_tech_stack(parsed.tech_stack, ground_truth["tech_stack"]))

    parsed_work_history = [item.model_dump() for item in parsed.work_history]
    report.checks.append(check_highlights_verbatim(cv_text, parsed_work_history))
    report.checks.extend(
        check_work_history_years(parsed_work_history, ground_truth["work_history"])
    )
    report.checks.append(
        check_years_experience(parsed.years_experience, ground_truth["career_start"], today)
    )
    return report


def _print_report(report: PersonaReport) -> None:
    print(f"\n=== {report.persona_id}: {report.passed}/{report.total} checks passed ===")
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] {c.field}: {c.detail}")


async def _run() -> None:
    reports = []
    for persona_id in personas_lib.persona_ids():
        persona = personas_lib.load_persona(persona_id)
        parsed = await parse_cv(persona.cv_text)
        report = build_report(persona_id, parsed, persona.ground_truth, persona.cv_text)
        reports.append(report)
        _print_report(report)

    total_passed = sum(r.passed for r in reports)
    total_checks = sum(r.total for r in reports)
    n = len(reports)
    print(f"\n=== Overall: {total_passed}/{total_checks} checks passed across {n} personas ===")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_run())


if __name__ == "__main__":
    main()
