import json
from types import SimpleNamespace

from hsh_scraper.evals.evidence_validation import build_source_check, score_evidence_grounding


def _case(**overrides):
    values = {
        "id": "q-test",
        "question": "Testfrage?",
        "reference_answer": "Referenz",
        "case_type": "direct",
        "expected_behavior": "direct_answer",
        "source_url": "https://example.edu/page",
        "source_policy": "exact_gold_source",
        "accepted_source_urls": (),
        "clarification": {"expected": False, "selected_option": "", "clarification_text": ""},
        "required_facts": (),
        "optional_facts": (),
        "forbidden_claims": (),
        "answer_variants": (),
        "evaluation_notes": "",
        "tags": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(url: str, text: str = "Evidence text") -> dict:
    return {
        "id": url,
        "score": 1.0,
        "payload": {
            "title": "Title",
            "source_url": url,
            "section_heading": "Section",
            "crawl_date": "2026-05-06",
            "text": text,
        },
    }


def test_source_check_finds_gold_source_with_normalized_url() -> None:
    source_check = build_source_check(
        _case(source_url="https://example.edu/page/"),
        [_result("https://example.edu/other"), _result("https://example.edu/page")],
    )

    assert source_check["gold_source_found"] is True
    assert source_check["gold_source_rank"] == 2
    assert source_check["source_policy_satisfied"] is True
    assert source_check["source_policy_match_rank"] == 2
    assert source_check["source_retrieval_score"] == 4


def test_source_check_can_accept_alternate_sources() -> None:
    source_check = build_source_check(
        _case(
            source_policy="accepted_sources",
            accepted_source_urls=("https://example.edu/alternate",),
        ),
        [_result("https://example.edu/alternate")],
    )

    assert source_check["gold_source_found"] is False
    assert source_check["source_policy_satisfied"] is True
    assert source_check["source_policy_match_url"] == "https://example.edu/alternate"
    assert source_check["source_retrieval_score"] == 5


def test_source_check_can_accept_answer_variant_source() -> None:
    source_check = build_source_check(
        _case(
            answer_variants=(
                {
                    "id": "icms_variant",
                    "source_policy": "exact_gold_source",
                    "source_url": "https://example.edu/icms.pdf",
                    "accepted_source_urls": [],
                },
            )
        ),
        [_result("https://example.edu/icms.pdf")],
    )

    assert source_check["gold_source_found"] is False
    assert source_check["source_policy_satisfied"] is True
    assert source_check["matched_source_variant"] == "icms_variant"
    assert source_check["source_retrieval_score"] == 5


def test_source_check_can_accept_any_official_hsh_source() -> None:
    source_check = build_source_check(
        _case(source_policy="official_hsh_any"),
        [_result("https://f2.hs-hannover.de/studium")],
    )

    assert source_check["source_policy_satisfied"] is True
    assert source_check["source_policy_match_rank"] == 1
    assert source_check["source_retrieval_score"] == 5


def test_no_results_refusal_is_scored_without_model_call() -> None:
    evaluation = score_evidence_grounding(
        openai_client=None,
        evaluator_model="judge",
        case=_case(),
        answer="Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor.",
        results=[],
        default_no_info_answer="Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor.",
    )

    assert evaluation["verdict"] == "retrieval_failed"
    assert evaluation["scores"]["source_retrieval"] == 0
    assert evaluation["scores"]["refusal_behavior"] == 5
    assert evaluation["unsupported_claims"] == []


def test_model_evidence_scores_keep_deterministic_source_retrieval_score() -> None:
    raw = {
        "scores": {
            "source_retrieval": 0,
            "evidence_sufficiency": 5,
            "answer_grounding": 5,
            "citation_support": 4,
            "refusal_behavior": 5,
        },
        "verdict": "fully_grounded",
        "summary": "ok",
        "supported_claims": ["claim"],
        "unsupported_claims": [],
        "contradicted_claims": [],
        "issues": [],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw)))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )

    evaluation = score_evidence_grounding(
        openai_client=client,
        evaluator_model="judge",
        case=_case(),
        answer="Antwort",
        results=[_result("https://example.edu/page")],
    )

    assert evaluation["verdict"] == "fully_grounded"
    assert evaluation["scores"]["source_retrieval"] == 5
    assert evaluation["scores"]["answer_grounding"] == 5
    assert evaluation["overall_score"] == 96.0


def test_wrong_source_policy_caps_evidence_scores() -> None:
    raw = {
        "scores": {
            "source_retrieval": 5,
            "evidence_sufficiency": 5,
            "answer_grounding": 5,
            "citation_support": 5,
            "refusal_behavior": 5,
        },
        "verdict": "fully_grounded",
        "summary": "ok",
        "supported_claims": ["claim"],
        "unsupported_claims": [],
        "contradicted_claims": [],
        "issues": [],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw)))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )

    evaluation = score_evidence_grounding(
        openai_client=client,
        evaluator_model="judge",
        case=_case(source_policy="exact_gold_source"),
        answer="Antwort",
        results=[_result("https://example.edu/other")],
    )

    assert evaluation["verdict"] == "wrong_source"
    assert evaluation["scores"]["source_retrieval"] == 0
    assert evaluation["scores"]["evidence_sufficiency"] == 2
    assert evaluation["scores"]["citation_support"] == 2
