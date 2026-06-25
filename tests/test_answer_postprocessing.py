from hsh_scraper.web_app_runtime import clean_answer_text


def test_clean_answer_text_trims_repeated_source_sections() -> None:
    answer = """Die Anmeldung erfolgt ueber das iCMS.

**Quellen:**
- Quelle 1 -- Fakultaet I, Stand: 2026-05-06
- Quelle 2 -- Allgemeine Anmeldung, Stand: 2026-05-06

Hinweis:
- Die Anmeldung erfolgt ueber das iCMS.

**Quellen:**
- Quelle 1 -- Fakultaet I, Stand: 2026-05-06
"""

    assert clean_answer_text(answer) == """Die Anmeldung erfolgt ueber das iCMS.

**Quellen:**
- Quelle 1 -- Fakultaet I, Stand: 2026-05-06
- Quelle 2 -- Allgemeine Anmeldung, Stand: 2026-05-06"""


def test_clean_answer_text_keeps_hint_before_sources() -> None:
    answer = """Hinweis:
Bitte pruefen Sie aktuelle Fristen direkt im iCMS.

**Quellen:**
- Quelle 1 -- Fakultaet I"""

    assert clean_answer_text(answer) == answer


def test_clean_answer_text_collapses_adjacent_duplicate_paragraphs() -> None:
    answer = """Die Anmeldung erfolgt ueber das iCMS.

Die Anmeldung erfolgt ueber das iCMS.

**Quellen:**
- Quelle 1 -- Fakultaet I"""

    assert clean_answer_text(answer) == """Die Anmeldung erfolgt ueber das iCMS.

**Quellen:**
- Quelle 1 -- Fakultaet I"""
