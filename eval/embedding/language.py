"""Which language a posting is written in, by stopword ratios; pure, no dependencies.

The corpus is meant to be English-only, but the pool also holds Spanish and a few Portuguese
postings, and the eval scores only the language the user rates (see `evaluate --language`). A
document is Spanish or Portuguese when that language's stopwords make up clearly more of its
alphabetic tokens than English's do, and English otherwise. This is a heuristic for a
job-posting pool, not a language identifier: it knows only these three languages, files any other
language under English unless it has no alphabetic tokens at all ("other"), and is unreliable on
very short texts or postings that are mostly code or tool names.
"""

import re

LANGUAGES = ("en", "es", "pt", "other")
DESCRIPTION_CHARS = 1500

_TOKEN = re.compile(r"[^\W\d_]+")

_EN = frozenset(
    {"the", "and", "with", "for", "of", "to", "in", "a", "an", "experience", "you", "will"}
    | {"our", "we", "is", "are", "team", "years", "work", "skills"}
)
_ES = frozenset(
    {"de", "la", "el", "y", "en", "con", "para", "los", "las", "del", "que", "una", "un", "por"}
    | {"se", "experiencia", "desarrollo", "conocimientos", "equipo", "tu", "tus", "años", "como"}
    | {"más", "al", "es", "ser", "trabajo"}
)
_PT = frozenset(
    {"não", "uma", "para", "com", "você", "experiência", "desenvolvimento", "equipe", "anos"}
    | {"são"}
)

_ES_MIN, _PT_MIN = 0.08, 0.06


def detect_language(text: str) -> str:
    """`"en"`, `"es"`, `"pt"`, or `"other"` for text with no alphabetic tokens."""
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return "other"

    def ratio(stopwords: frozenset[str]) -> float:
        return sum(1 for token in tokens if token in stopwords) / len(tokens)

    en, es, pt = ratio(_EN), ratio(_ES), ratio(_PT)
    if es > en and es > pt and es > _ES_MIN:
        return "es"
    if pt > en and pt > _PT_MIN:
        return "pt"
    return "en"


def doc_language(
    title: str,
    description: str,
    requirements: str | None,
    responsibilities: str | None,
) -> str:
    """The language of a posting, from its title and the start of its full description.

    `requirements` and `responsibilities` are accepted so a caller can pass a document's fields
    as they are, but never read: the extraction prompt can leave English boilerplate in them for
    a Spanish posting, so only the description tells the truth.
    """
    return detect_language(f"{title} {description[:DESCRIPTION_CHARS]}")
