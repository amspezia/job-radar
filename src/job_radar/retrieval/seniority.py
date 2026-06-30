import re

from job_radar.db.models import Profile

# Canonical IC seniority ladder, low to high. Management/ambiguous titles are
# deliberately not on it — they normalize to None ("unknown") rather than being
# force-fit, so they are never gated on level.
LADDER = ["intern", "junior", "mid", "senior", "staff", "principal"]
_RANK = {level: i for i, level in enumerate(LADDER)}

# Synonyms -> canonical level, matched as whole words against free text/titles.
_SYNONYMS: dict[str, str] = {
    "intern": "intern",
    "trainee": "intern",
    "junior": "junior",
    "jr": "junior",
    "entry": "junior",
    "associate": "junior",
    "mid": "mid",
    "pleno": "mid",
    "intermediate": "mid",
    "senior": "senior",
    "sr": "senior",
    "staff": "staff",
    "lead": "staff",
    "principal": "principal",
    "distinguished": "principal",
}


def rank(level: str | None) -> int | None:
    return _RANK.get(level) if level else None


def normalize_level(text: str | None) -> str | None:
    """Map free text (e.g. "Senior", "Sr.") to a canonical ladder level, or None.

    Scans high-to-low so a compound title like "Senior Staff Engineer" resolves
    to the higher level it actually denotes.
    """
    if not text:
        return None
    for level in reversed(LADDER):
        keywords = [kw for kw, lvl in _SYNONYMS.items() if lvl == level]
        if any(re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE) for kw in keywords):
            return level
    return None


def default_allowed(candidate_level: str | None) -> list[str]:
    """Levels accepted when the profile sets no explicit rule.

    At or below the candidate's level, plus one above (a Senior can plausibly
    reach a Staff role). Unknown candidate level -> accept everything.
    """
    c = rank(candidate_level)
    if c is None:
        return list(LADDER)
    return [level for level in LADDER if _RANK[level] <= c + 1]


def allowed_levels(profile: Profile) -> list[str]:
    """The posting levels this profile accepts — explicit rule, else the default."""
    configured = (profile.seniority_rules or {}).get("allowed_levels")
    if configured:
        return configured
    return default_allowed(normalize_level(profile.seniority))
