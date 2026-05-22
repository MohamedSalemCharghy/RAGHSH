from hsh_scraper.query_assist import assess_query, build_retrieval_queries
from hsh_scraper.retrieval_planner import should_use_retrieval_planner


def test_faculty_shorthand_is_not_misread_as_definition_code() -> None:
    question = "wie viele Studiengänge gibt es in Fak1?"

    assessment = assess_query(question)
    retrieval_queries = build_retrieval_queries(assessment)

    assert assessment.detected_faculties == ["Fakultät I"]
    assert assessment.detected_codes == []
    assert "definition" not in assessment.intents
    assert not any("fak1 bedeutung" in query.casefold() for query in retrieval_queries)
    assert any(
        "fakultät i" in query.casefold() and "studieng" in query.casefold()
        for query in retrieval_queries
    )


def test_person_lookup_keeps_name_and_skips_llm_planner() -> None:
    question = "wer ist herr homman?"

    assessment = assess_query(question)
    retrieval_queries = build_retrieval_queries(assessment)

    assert "person_lookup" in assessment.intents
    assert any("homman" in query.casefold() for query in retrieval_queries)
    assert any("personenfinder" in query.casefold() for query in retrieval_queries)
    assert not should_use_retrieval_planner(assessment, question)


def test_study_program_fact_query_adds_admission_and_internship_variants() -> None:
    question = "Ist der Bachelorstudiengang Mechatronik zulassungsfrei und gibt es ein Vorpraktikum?"

    assessment = assess_query(question)
    retrieval_queries = build_retrieval_queries(assessment)

    assert "study_program_facts" in assessment.intents
    assert should_use_retrieval_planner(assessment, question)
    assert any(
        "mechatronik" in query.casefold() and "zulass" in query.casefold()
        for query in retrieval_queries
    )
    assert any(
        "mechatronik" in query.casefold() and "vorpraktikum" in query.casefold()
        for query in retrieval_queries
    )


def test_study_program_fact_query_adds_degree_duration_and_start_variants() -> None:
    question = "Welchen Abschluss hat Mechatronik, wie lange dauert das Studium und wann kann man beginnen?"

    assessment = assess_query(question)
    retrieval_queries = build_retrieval_queries(assessment)

    assert "study_program_facts" in assessment.intents
    assert any(
        "mechatronik" in query.casefold() and "abschluss" in query.casefold()
        for query in retrieval_queries
    )
    assert any(
        "mechatronik" in query.casefold() and "regelstudienzeit" in query.casefold()
        for query in retrieval_queries
    )
    assert any(
        "mechatronik" in query.casefold() and "studienbeginn" in query.casefold()
        for query in retrieval_queries
    )


def test_online_application_clarification_keeps_focus_on_bewerbungsportal() -> None:
    question = "Wie stelle ich einen Online-Antrag für eine Bewerbung oder das Hochladen von Unterlagen?"

    assessment = assess_query(question)
    retrieval_queries = build_retrieval_queries(assessment)

    assert "workflow" in assessment.intents
    assert any("bewerbungsportal" in query.casefold() for query in retrieval_queries)
