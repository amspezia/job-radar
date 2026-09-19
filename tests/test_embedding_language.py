"""The stopword language detector: clear samples, the description-only rule and empty text."""

import pytest

from eval.embedding.language import LANGUAGES, detect_language, doc_language

ENGLISH = (
    "We are looking for a senior backend engineer with experience in Python and the team "
    "you will work with is fully remote. You will build our services and own the roadmap."
)
SPANISH = (
    "Buscamos un desarrollador backend con experiencia en Python para el equipo de la empresa. "
    "Tendrás conocimientos de bases de datos y trabajo con los servicios de la plataforma."
)
PORTUGUESE = (
    "Procuramos uma pessoa desenvolvedora com experiência em Python. Você não precisa de anos "
    "de experiência, mas as tarefas são de desenvolvimento e a equipe é remota."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (ENGLISH, "en"),
        (SPANISH, "es"),
        (PORTUGUESE, "pt"),
        ("Senior Kubernetes SRE", "en"),  # no stopwords at all: English by default
        ("", "other"),
        ("   \n ", "other"),
        ("1234 5678 --- !!!", "other"),
    ],
)
def test_detect_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected
    assert expected in LANGUAGES


def test_case_and_accents_do_not_matter() -> None:
    assert detect_language(SPANISH.upper()) == "es"
    assert detect_language("AÑOS DE EXPERIENCIA EN EL DESARROLLO DE LA PLATAFORMA") == "es"


def test_a_few_spanish_words_in_an_english_posting_stay_english() -> None:
    text = ENGLISH + " Spanish is a plus: experiencia en el desarrollo."
    assert detect_language(text) == "en"


def test_doc_language_reads_the_description_not_the_extracted_fields() -> None:
    boilerplate = "all skills, technologies, qualifications and the experience you will need"
    assert (
        doc_language("Desarrollador backend", SPANISH, boilerplate, "the team will work with you")
        == "es"
    )
    assert doc_language("Backend developer", ENGLISH, SPANISH, SPANISH) == "en"
    assert doc_language("Desenvolvedor", PORTUGUESE, None, None) == "pt"


def test_doc_language_looks_at_the_start_of_the_description_only() -> None:
    assert doc_language("Backend", ENGLISH + " " * 2000 + SPANISH * 30, None, None) == "en"


def test_doc_language_of_an_empty_document_is_other() -> None:
    assert doc_language("", "", None, None) == "other"
