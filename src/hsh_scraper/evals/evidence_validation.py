"""Evidence-grounding checks for strict-dialog validation runs."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse


EVIDENCE_RUBRIC_FIELDS = (
    "source_retrieval",
    "evidence_sufficiency",
    "answer_grounding",
    "citation_support",
    "refusal_behavior",
)

EVIDENCE_VERDICTS = {
    "fully_grounded",
    "mostly_grounded",
    "partially_grounded",
    "unsupported",
    "contradicted",
    "retrieval_failed",
    "wrong_source",
    "correct_refusal",
    "wrong_refusal",
}

SOURCE_POLICIES = {
    "exact_gold_source",
    "accepted_sources",
    "official_hsh_any",
}

EVIDENCE_EVALUATOR_PROMPT = """\
Du bist ein Evidence-Auditor fuer ein RAG-System der Hochschule Hannover (HsH).

Ziel:
- Pruefe, ob die finale Assistant-Antwort durch die tatsaechlich aus Qdrant
  abgerufenen Chunks gedeckt ist.
- Bewerte nicht gegen Weltwissen und nicht gegen die Referenzantwort als Beweis.
  Die Referenz dient nur als Orientierung fuer die erwartete Antwort.
- Trenne Retrieval-Probleme von Antwort-/Grounding-Problemen.
- Beachte source_policy:
  exact_gold_source = die Goldquelle muss gefunden werden.
  accepted_sources = eine der akzeptierten Quellen muss gefunden werden.
  official_hsh_any = eine offizielle HsH-Quelle reicht als Quelle aus.
- Wenn answer_variants vorhanden sind, reicht fuer die Quellenpruefung ein erfuellter
  gueltiger Antwortpfad.

Wichtige Regeln:
- Nutze ausschliesslich retrieved_chunks als Evidenz.
- Wenn source_policy nicht erfuellt wurde, ist das ein Quellenproblem, auch wenn andere Chunks einzelne Claims stuetzen.
- Wenn die Chunks genug Evidenz enthalten, die Antwort diese Evidenz aber falsch
  nutzt oder Dinge erfindet, ist das ein Grounding-Problem.
- Bei negativfall oder fehlendes-detail ist eine klare Abgrenzung gut:
  Wenn ein Detail in den Chunks nicht belegt ist, darf/soll die Antwort das sagen.
- Externe oder nicht belegte Details sind streng zu bestrafen.
- Gib fuer jedes Kriterium eine ganze Zahl zwischen 0 und 5.
- Antworte NUR mit einem JSON-Objekt, ohne Markdown.

Kriterien:
- source_retrieval: Wurde die source_policy durch die Qdrant-Chunks erfuellt?
- evidence_sufficiency: Reichen die Chunks aus, um die Frage belastbar zu beantworten?
- answer_grounding: Sind die sachlichen Claims der Antwort in den Chunks belegt?
- citation_support: Passen Quellen/URLs/Abschnitte der Antwort zu den Chunks?
- refusal_behavior: Wurde bei fehlender Evidenz korrekt abgegrenzt oder verweigert?

Erwartetes JSON-Schema:
{
  "scores": {
    "source_retrieval": 0,
    "evidence_sufficiency": 0,
    "answer_grounding": 0,
    "citation_support": 0,
    "refusal_behavior": 0
  },
  "verdict": "fully_grounded",
  "summary": "",
  "supported_claims": [],
  "unsupported_claims": [],
  "contradicted_claims": [],
  "issues": []
}
"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)

    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _sanitize_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, score))


def _sanitize_string(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _sanitize_string_list(values: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = _sanitize_string(value)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) >= limit:
            break
    return cleaned


def _normalize_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    try:
        parsed = urlparse(value)
    except ValueError:
        return value.rstrip("/").casefold()
    path = (parsed.path or "/").rstrip("/") or "/"
    return parsed._replace(fragment="", params="", path=path).geturl().casefold()


def _case_value(case: Any, name: str, default: Any = "") -> Any:
    if isinstance(case, dict):
        return case.get(name, default)
    return getattr(case, name, default)


def _case_tags(case: Any) -> list[str]:
    tags = _case_value(case, "tags", [])
    if isinstance(tags, (list, tuple, set)):
        return [str(tag) for tag in tags]
    return []


def _case_clarification(case: Any) -> dict[str, Any]:
    clarification = _case_value(case, "clarification", {})
    if isinstance(clarification, dict):
        return clarification
    return {
        "expected": bool(_case_value(case, "clarification_expected", False)),
        "selected_option": _case_value(case, "selected_option", ""),
        "clarification_text": _case_value(case, "clarification_text", ""),
    }


def _case_list(case: Any, name: str) -> list[str]:
    values = _case_value(case, name, [])
    if isinstance(values, (list, tuple, set)):
        return [str(value).strip() for value in values if str(value or "").strip()]
    return []


def _case_answer_variants(case: Any) -> list[dict[str, Any]]:
    variants = _case_value(case, "answer_variants", [])
    if not isinstance(variants, (list, tuple)):
        return []
    return [variant for variant in variants if isinstance(variant, dict)]


def _case_source_policy(case: Any) -> str:
    policy = str(_case_value(case, "source_policy", "") or "").strip().casefold()
    return policy if policy in SOURCE_POLICIES else "exact_gold_source"


def _is_no_info_answer(answer: str, default_no_info_answer: str) -> bool:
    return _sanitize_string(answer).casefold() == _sanitize_string(default_no_info_answer).casefold()


def _expected_source_urls(case: Any) -> list[str]:
    source_url = str(_case_value(case, "source_url", "") or "").strip()
    urls = [source_url]
    urls.extend(_case_list(case, "accepted_source_urls"))
    urls.extend(_case_list(case, "additional_source_urls"))
    cleaned: list[str] = []
    seen: set[str] = set()
    for url in urls:
        key = _normalize_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(url)
    return cleaned


def _source_policy_candidates(case: Any) -> list[dict[str, Any]]:
    candidates = [
        {
            "id": "primary",
            "title": "Primärer Antwortpfad",
            "source_policy": _case_source_policy(case),
            "expected_source_urls": _expected_source_urls(case),
        }
    ]
    for index, variant in enumerate(_case_answer_variants(case), start=1):
        variant_id = str(variant.get("id") or f"variant_{index}").strip()
        policy = str(variant.get("source_policy") or "").strip().casefold()
        if policy not in SOURCE_POLICIES:
            policy = _case_source_policy(case)
        candidates.append(
            {
                "id": variant_id,
                "title": str(variant.get("title") or variant_id).strip(),
                "source_policy": policy,
                "expected_source_urls": _expected_source_urls(variant),
            }
        )
    return candidates


def _is_official_hsh_source(url: str) -> bool:
    try:
        host = urlparse(url).netloc.casefold()
    except ValueError:
        return False
    return host == "hs-hannover.de" or host.endswith(".hs-hannover.de")


def _source_retrieval_score(rank: int | None, has_policy_target: bool) -> int:
    if not has_policy_target:
        return 5
    if rank is None:
        return 0
    if rank <= 1:
        return 5
    if rank <= 3:
        return 4
    if rank <= 6:
        return 3
    return 2


def build_source_check(case: Any, results: list[dict]) -> dict[str, Any]:
    source_policy = _case_source_policy(case)
    gold_source_url = str(_case_value(case, "source_url", "") or "").strip()
    candidates = _source_policy_candidates(case)
    expected_urls = []
    for candidate in candidates:
        expected_urls.extend(candidate["expected_source_urls"])
    expected_urls = list(dict.fromkeys(expected_urls))
    normalized_expected = {_normalize_url(url) for url in expected_urls}
    normalized_gold = _normalize_url(gold_source_url)

    retrieved_sources = []
    gold_found_rank: int | None = None
    gold_found_url = ""
    policy_match_rank: int | None = None
    policy_match_url = ""
    matched_source_variant = ""
    for rank, result in enumerate(results, start=1):
        payload = result.get("payload") or {}
        url = str(payload.get("source_url", "") or "").strip()
        retrieved_sources.append(url)
        normalized_url = _normalize_url(url)
        if gold_found_rank is None and normalized_gold and normalized_url == normalized_gold:
            gold_found_rank = rank
            gold_found_url = url
        if policy_match_rank is not None:
            continue
        for candidate in candidates:
            candidate_policy = candidate["source_policy"]
            candidate_expected = {
                _normalize_url(candidate_url)
                for candidate_url in candidate["expected_source_urls"]
            }
            if candidate_policy == "official_hsh_any" and _is_official_hsh_source(url):
                policy_match_rank = rank
                policy_match_url = url
                matched_source_variant = candidate["id"]
                break
            if candidate_policy in {"exact_gold_source", "accepted_sources"} and normalized_url in candidate_expected:
                policy_match_rank = rank
                policy_match_url = url
                matched_source_variant = candidate["id"]
                break

    has_policy_target = any(
        candidate["expected_source_urls"] or candidate["source_policy"] == "official_hsh_any"
        for candidate in candidates
    )

    return {
        "source_policy": source_policy,
        "source_policy_candidates": candidates,
        "expected_source_urls": expected_urls,
        "retrieved_source_urls": retrieved_sources,
        "gold_source_found": gold_found_rank is not None,
        "gold_source_rank": gold_found_rank,
        "gold_source_url": gold_found_url,
        "source_policy_satisfied": policy_match_rank is not None or not has_policy_target,
        "source_policy_match_rank": policy_match_rank,
        "source_policy_match_url": policy_match_url,
        "matched_source_variant": matched_source_variant,
        "source_retrieval_score": _source_retrieval_score(policy_match_rank, has_policy_target),
    }


def _format_retrieved_chunks(
    results: list[dict],
    *,
    limit: int = 6,
    text_limit: int = 1800,
) -> list[dict[str, Any]]:
    chunks = []
    for rank, result in enumerate(results[:limit], start=1):
        payload = result.get("payload") or {}
        chunks.append(
            {
                "rank": rank,
                "score": result.get("score", 0),
                "title": payload.get("title", ""),
                "url": payload.get("source_url", ""),
                "section": payload.get("section_heading", ""),
                "date": payload.get("crawl_date", ""),
                "text": str(payload.get("text", "") or "")[:text_limit],
            }
        )
    return chunks


def _overall_score(scores: dict[str, int]) -> float:
    total = sum(scores.get(field, 0) for field in EVIDENCE_RUBRIC_FIELDS)
    return round(total / (len(EVIDENCE_RUBRIC_FIELDS) * 5) * 100, 1)


def _finalize_model_evaluation(
    parsed: dict[str, Any],
    *,
    raw: str,
    source_check: dict[str, Any],
) -> dict[str, Any]:
    parsed_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    scores = {
        field: _sanitize_score(parsed_scores.get(field))
        for field in EVIDENCE_RUBRIC_FIELDS
    }
    scores["source_retrieval"] = int(source_check["source_retrieval_score"])
    if not source_check.get("source_policy_satisfied", False):
        scores["evidence_sufficiency"] = min(scores["evidence_sufficiency"], 2)
        scores["citation_support"] = min(scores["citation_support"], 2)

    verdict = _sanitize_string(parsed.get("verdict")).casefold()
    if verdict not in EVIDENCE_VERDICTS:
        verdict = "retrieval_failed" if not source_check["source_policy_satisfied"] else "partially_grounded"
    if not source_check.get("source_policy_satisfied", False) and verdict in {
        "fully_grounded",
        "mostly_grounded",
    }:
        verdict = "wrong_source"

    return {
        "scores": scores,
        "overall_score": _overall_score(scores),
        "verdict": verdict,
        "source_check": source_check,
        "summary": _sanitize_string(parsed.get("summary")),
        "supported_claims": _sanitize_string_list(parsed.get("supported_claims")),
        "unsupported_claims": _sanitize_string_list(parsed.get("unsupported_claims")),
        "contradicted_claims": _sanitize_string_list(parsed.get("contradicted_claims")),
        "issues": _sanitize_string_list(parsed.get("issues")),
        "raw_judgement": raw.strip(),
    }


def _empty_answer_evaluation(source_check: dict[str, Any]) -> dict[str, Any]:
    scores = {field: 0 for field in EVIDENCE_RUBRIC_FIELDS}
    scores["source_retrieval"] = int(source_check["source_retrieval_score"])
    return {
        "scores": scores,
        "overall_score": _overall_score(scores),
        "verdict": "unsupported",
        "source_check": source_check,
        "summary": "Die Assistant-Antwort war leer; Grounding konnte nicht sinnvoll geprueft werden.",
        "supported_claims": [],
        "unsupported_claims": [],
        "contradicted_claims": [],
        "issues": ["Leere Assistant-Antwort."],
        "raw_judgement": "",
    }


def _no_results_evaluation(
    *,
    answer: str,
    source_check: dict[str, Any],
    default_no_info_answer: str,
) -> dict[str, Any]:
    no_info = _is_no_info_answer(answer, default_no_info_answer)
    scores = {field: 0 for field in EVIDENCE_RUBRIC_FIELDS}
    scores["source_retrieval"] = int(source_check["source_retrieval_score"])
    if no_info:
        scores["answer_grounding"] = 5
        scores["citation_support"] = 5
        scores["refusal_behavior"] = 5
    return {
        "scores": scores,
        "overall_score": _overall_score(scores),
        "verdict": "retrieval_failed",
        "source_check": source_check,
        "summary": "Keine Qdrant-Chunks wurden abgerufen; die Antwort kann daher nicht durch Evidenz belegt werden.",
        "supported_claims": [],
        "unsupported_claims": [] if no_info else [_sanitize_string(answer)[:220]],
        "contradicted_claims": [],
        "issues": ["Die erwartete Quelle wurde nicht in den Retrieval-Ergebnissen gefunden."],
        "raw_judgement": "",
    }


def score_evidence_grounding(
    *,
    openai_client,
    evaluator_model: str,
    case: Any,
    answer: str,
    results: list[dict],
    process_trace: dict[str, Any] | None = None,
    default_no_info_answer: str = "",
) -> dict[str, Any]:
    source_check = build_source_check(case, results)
    answer = _sanitize_string(answer)
    if not answer:
        return _empty_answer_evaluation(source_check)
    if not results:
        return _no_results_evaluation(
            answer=answer,
            source_check=source_check,
            default_no_info_answer=default_no_info_answer,
        )

    payload = {
        "case": {
            "id": _case_value(case, "id", ""),
            "question": _case_value(case, "question", ""),
            "reference_answer": _case_value(case, "reference_answer", ""),
            "case_type": _case_value(case, "case_type", ""),
            "expected_behavior": _case_value(case, "expected_behavior", ""),
            "source_url": _case_value(case, "source_url", ""),
            "source_policy": _case_source_policy(case),
            "accepted_source_urls": _case_list(case, "accepted_source_urls"),
            "clarification": _case_clarification(case),
            "negative_case": any(
                tag.casefold() in {"negativfall", "fehlendes-detail"}
                for tag in _case_tags(case)
            ),
            "required_facts": _case_list(case, "required_facts"),
            "optional_facts": _case_list(case, "optional_facts"),
            "forbidden_claims": _case_list(case, "forbidden_claims"),
            "answer_variants": _case_answer_variants(case),
            "evaluation_notes": _case_value(case, "evaluation_notes", ""),
            "tags": _case_tags(case),
        },
        "assistant_final_answer": answer,
        "source_check": source_check,
        "process_trace": {
            "retrieval_queries": (process_trace or {}).get("retrieval_queries", []),
            "assessment_intents": (process_trace or {}).get("assessment_intents", []),
            "selected_query": (process_trace or {}).get("selected_query", ""),
            "clarification": (process_trace or {}).get("clarification", ""),
        },
        "retrieved_chunks": _format_retrieved_chunks(results),
    }

    response = openai_client.chat.completions.create(
        model=evaluator_model,
        messages=[
            {"role": "system", "content": EVIDENCE_EVALUATOR_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    return _finalize_model_evaluation(
        _extract_json_object(raw) or {},
        raw=raw,
        source_check=source_check,
    )


__all__ = [
    "EVIDENCE_RUBRIC_FIELDS",
    "EVIDENCE_VERDICTS",
    "build_source_check",
    "score_evidence_grounding",
]
