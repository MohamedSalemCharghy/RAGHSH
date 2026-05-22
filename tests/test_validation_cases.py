import json
from pathlib import Path

from hsh_scraper.evals import validation_system


def _minimal_result(case: validation_system.EvalCase) -> dict:
    return {
        "timestamp": "2026-05-06T00:00:00",
        "case": {
            "id": case.id,
            "question": case.question,
            "reference_answer": case.reference_answer,
            "case_type": case.case_type,
            "expected_behavior": case.expected_behavior,
            "source_url": case.source_url,
            "source_policy": case.source_policy,
            "accepted_source_urls": list(case.accepted_source_urls),
            "clarification": case.clarification,
            "clarification_expected": case.clarification_expected,
            "selected_option": case.selected_option,
            "clarification_text": case.clarification_text,
            "negative_case": validation_system.is_negative_case(case),
            "required_facts": list(case.required_facts),
            "optional_facts": list(case.optional_facts),
            "forbidden_claims": list(case.forbidden_claims),
            "answer_variants": list(case.answer_variants),
            "case_difficulty": case.case_difficulty,
            "fairness_risk": case.fairness_risk,
            "evaluation_notes": case.evaluation_notes,
            "tags": list(case.tags),
        },
        "chatbot_model": "chatbot",
        "evaluator_model": "judge",
        "transcript": [],
        "answer": "Testantwort",
        "process_trace": {},
        "results": [],
        "timings": {},
        "evaluation": {
            "scores": {field: 5 for field in validation_system.RUBRIC_FIELDS},
            "overall_score": 100.0,
            "summary": "ok",
            "strengths": [],
            "issues": [],
            "raw_judgement": "{}",
        },
        "evidence_evaluation": {
            "scores": {field: 5 for field in validation_system.EVIDENCE_RUBRIC_FIELDS},
            "overall_score": 100.0,
            "verdict": "fully_grounded",
            "source_check": {
                "expected_source_urls": [case.source_url] if case.source_url else [],
                "retrieved_source_urls": [case.source_url] if case.source_url else [],
                "gold_source_found": bool(case.source_url),
                "gold_source_rank": 1 if case.source_url else None,
                "gold_source_url": case.source_url,
                "source_policy": case.source_policy,
                "source_policy_satisfied": bool(case.source_url),
                "source_policy_match_rank": 1 if case.source_url else None,
                "source_policy_match_url": case.source_url,
                "matched_source_variant": "primary" if case.source_url else "",
                "source_retrieval_score": 5,
            },
            "summary": "grounded",
            "supported_claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
            "issues": [],
            "raw_judgement": "{}",
        },
        "diagnostics": {
            "failure_types": [],
            "stage_gap": 0.0,
            "human_review_recommended": False,
            "notes": [],
        },
    }


def test_default_validation_cases_loads_verified_twenty_case_dataset() -> None:
    cases = validation_system.load_eval_cases()

    assert len(cases) == 20
    assert [case.id for case in cases] == [f"q{index:02d}" for index in range(1, 21)]
    assert validation_system.CASES_FILE.name == "validation_cases.json"

    q04 = next(case for case in cases if case.id == "q04")
    assert q04.case_type == "clarification"
    assert q04.expected_behavior == "clarify_then_answer"
    assert q04.clarification_expected is True
    assert q04.clarification["selected_option"]
    assert q04.clarification["clarification_text"]
    assert q04.fairness_risk == "hoch"

    q10 = next(case for case in cases if case.id == "q10")
    assert q10.source_url == "https://f4.hs-hannover.de/service/formulare-und-informationen/sonstiges/mensa"
    assert q10.source_policy == "exact_gold_source"
    assert q10.expected_behavior == "missing_detail_answer"
    assert q10.required_facts
    assert q10.forbidden_claims
    assert validation_system.is_negative_case(q10)
    assert q10.tags == ("negativfall",)
    assert "keine konkreten Öffnungszeiten" in q10.reference_answer

    q06 = next(case for case in cases if case.id == "q06")
    assert any("Stephan Rittmüller" in fact for fact in q06.required_facts)
    assert any("Sprechzeiten" in fact for fact in q06.optional_facts)

    q05 = next(case for case in cases if case.id == "q05")
    assert len(q05.answer_variants) == 2
    assert {variant["id"] for variant in q05.answer_variants} == {
        "f1_pruefungsverwaltung_fundort",
        "icms_operatives_verfahren",
    }
    assert any("iCMS" in fact for variant in q05.answer_variants for fact in variant["required_facts"])

    q18 = next(case for case in cases if case.id == "q18")
    assert q18.source_policy == "accepted_sources"
    assert len(q18.answer_variants) == 2
    assert {variant["id"] for variant in q18.answer_variants} == {
        "profil_rund_9000",
        "ueber_uns_mehr_als_10000",
    }

    q20 = next(case for case in cases if case.id == "q20")
    assert q20.expected_behavior == "direct_answer"
    assert q20.fairness_risk == "niedrig"
    assert not validation_system.is_negative_case(q20)
    assert "ab 13. Februar 2019" in q20.question


def test_run_remaining_validation_cases_produces_result_for_every_loaded_case(
    monkeypatch, tmp_path: Path
) -> None:
    cases = validation_system.load_eval_cases()
    monkeypatch.setattr(validation_system, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        validation_system,
        "run_strict_dialog_case",
        lambda *, case, **_: _minimal_result(case),
    )

    state = validation_system.start_validation_run(
        chatbot_model="chatbot",
        evaluator_model="judge",
        cases=cases,
        store_in_session=False,
    )
    results = validation_system.run_remaining_validation_cases(
        state=state,
        cases=cases,
        qdrant=None,
        dense_embedder=None,
        sparse_embedder=None,
        reranker=None,
        openai_client=None,
        top_k=6,
        inter_case_delay_seconds=0,
    )

    expected_ids = [f"q{index:02d}" for index in range(1, 21)]
    assert [result["case"]["id"] for result in results] == expected_ids
    assert [result["case"]["id"] for result in state["results"]] == expected_ids
    assert state["case_index"] == len(cases)
    assert state["completed"] is True

    run_dir = Path(state["run_dir"])
    assert sorted(path.stem for path in run_dir.glob("q*.json")) == expected_ids
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_cases"] == 20
    assert summary["evidence_completed_cases"] == 20
    assert summary["evidence_overall_score"] == 100.0
    assert summary["failure_type_counts"] == {}


def test_run_remaining_validation_cases_does_not_skip_missing_cases_before_saved_index(
    monkeypatch, tmp_path: Path
) -> None:
    cases = validation_system.load_eval_cases()
    expected_ids = [f"q{index:02d}" for index in range(1, 21)]
    monkeypatch.setattr(validation_system, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        validation_system,
        "run_strict_dialog_case",
        lambda *, case, **_: _minimal_result(case),
    )

    state = validation_system.start_validation_run(
        chatbot_model="chatbot",
        evaluator_model="judge",
        cases=cases,
        store_in_session=False,
    )
    state["results"] = [_minimal_result(cases[0])]
    state["case_index"] = 5

    new_results = validation_system.run_remaining_validation_cases(
        state=state,
        cases=cases,
        qdrant=None,
        dense_embedder=None,
        sparse_embedder=None,
        reranker=None,
        openai_client=None,
        top_k=6,
        inter_case_delay_seconds=0,
    )

    assert [result["case"]["id"] for result in new_results] == expected_ids[1:]
    assert [result["case"]["id"] for result in state["results"]] == expected_ids
    assert state["completed"] is True
    summary = json.loads((Path(state["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_cases"] == 20
    assert summary["evidence_completed_cases"] == 20


def test_rate_limit_retry_waits_and_continues(monkeypatch) -> None:
    sleeps = []
    messages = []
    attempts = []

    class FakeRateLimitError(Exception):
        status_code = 429

    def operation():
        attempts.append("called")
        if len(attempts) == 1:
            raise FakeRateLimitError("rate limit")
        return "ok"

    monkeypatch.setattr(validation_system, "VALIDATION_RATE_LIMIT_RETRY_SECONDS", 2)
    monkeypatch.setattr(validation_system.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = validation_system._run_with_rate_limit_retry(
        operation,
        description="Testschritt",
        notify=messages.append,
    )

    assert result == "ok"
    assert attempts == ["called", "called"]
    assert sleeps == [2]
    assert "API-Limit erreicht" in messages[0]


def test_negative_cases_with_no_result_are_evaluated_as_unavailable_answers(monkeypatch) -> None:
    case = next(case for case in validation_system.load_eval_cases() if case.id == "q10")

    monkeypatch.setattr(
        validation_system,
        "prepare_chat_turn",
        lambda **_: {
            "timings": {},
            "process_trace": {},
            "results": [],
            "no_results": True,
            "guard_answer": None,
        },
    )

    captured = {}

    def fake_score_with_evaluator(*, case, answer, **_):
        captured["case_id"] = case.id
        captured["answer"] = answer
        return {
            "scores": {field: 4 for field in validation_system.RUBRIC_FIELDS},
            "overall_score": 80.0,
            "summary": "Grounded unavailable answer is acceptable for this negative case.",
            "strengths": ["Keine externe Mensa-Zeit erfunden."],
            "issues": [],
            "diagnostics": {
                "matched_required_facts": [],
                "missing_required_facts": [],
                "matched_optional_facts": [],
                "missing_optional_facts": [],
                "forbidden_claims_found": [],
            },
            "failure_types": [],
            "raw_judgement": "{}",
        }

    monkeypatch.setattr(validation_system, "_score_with_evaluator", fake_score_with_evaluator)

    result = validation_system.run_strict_dialog_case(
        case=case,
        qdrant=None,
        dense_embedder=None,
        sparse_embedder=None,
        reranker=None,
        openai_client=None,
        chatbot_model="chatbot",
        evaluator_model="judge",
        top_k=6,
    )

    assert captured == {
        "case_id": "q10",
        "answer": validation_system.DEFAULT_NO_INFO_ANSWER,
    }
    assert result["answer"] == validation_system.DEFAULT_NO_INFO_ANSWER
    assert result["case"]["negative_case"] is True
    assert result["evaluation"]["overall_score"] == 80.0
    assert result["evidence_evaluation"]["verdict"] == "retrieval_failed"
    assert result["evidence_evaluation"]["source_check"]["gold_source_found"] is False
    assert "retrieval_failed" in result["diagnostics"]["failure_types"]
    assert "Die Assistant-Antwort war leer" not in result["evaluation"]["summary"]
