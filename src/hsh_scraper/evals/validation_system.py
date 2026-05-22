"""Strict-dialog validation workflow for the Streamlit HsH chatbot."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover - CLI can run without Streamlit installed
    st = None  # type: ignore[assignment]
from openai import OpenAI

try:
    from ..paths import ARTIFACTS_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from paths import ARTIFACTS_DIR

try:
    from .evidence_validation import EVIDENCE_RUBRIC_FIELDS, score_evidence_grounding
except ImportError:  # pragma: no cover - script execution fallback
    from evidence_validation import EVIDENCE_RUBRIC_FIELDS, score_evidence_grounding

try:
    from ..web_app_runtime import (
        DEFAULT_FACULTY,
        DEFAULT_ROLE,
        RagPipelineConfig,
        normalize_pipeline_config,
        prepare_chat_turn,
        request_answer_sync,
    )
except ImportError:  # pragma: no cover - lets metadata tests run without Qdrant deps
    try:
        from web_app_runtime import (
            DEFAULT_FACULTY,
            DEFAULT_ROLE,
            RagPipelineConfig,
            normalize_pipeline_config,
            prepare_chat_turn,
            request_answer_sync,
        )
    except ImportError:
        DEFAULT_FACULTY = "Alle Fakultäten"
        DEFAULT_ROLE = "Studierender"
        RagPipelineConfig = None  # type: ignore[assignment]
        normalize_pipeline_config = None  # type: ignore[assignment]
        prepare_chat_turn = None  # type: ignore[assignment]
        request_answer_sync = None  # type: ignore[assignment]


DEFAULT_CASES_FILE = Path(__file__).parent / "validation_cases.json"
CASES_FILE = Path(os.getenv("RAG_VALIDATION_CASES_FILE", str(DEFAULT_CASES_FILE))).expanduser()
RESULTS_DIR = ARTIFACTS_DIR / "evals" / "results"

VALIDATION_STATE_KEY = "validation_state"
MAX_EVAL_API_REQUESTS_PER_MINUTE = int(os.getenv("RAG_VALIDATION_MAX_REQUESTS_PER_MINUTE", "10"))
VALIDATION_REQUESTS_PER_CASE_ESTIMATE = int(os.getenv("RAG_VALIDATION_REQUESTS_PER_CASE_ESTIMATE", "4"))
VALIDATION_INTER_CASE_DELAY_SECONDS = float(
    os.getenv(
        "RAG_VALIDATION_INTER_CASE_DELAY_SECONDS",
        str(round(60 * VALIDATION_REQUESTS_PER_CASE_ESTIMATE / MAX_EVAL_API_REQUESTS_PER_MINUTE)),
    )
)
VALIDATION_RATE_LIMIT_RETRY_SECONDS = float(
    os.getenv("RAG_VALIDATION_RATE_LIMIT_RETRY_SECONDS", "75")
)
VALIDATION_RATE_LIMIT_MAX_RETRIES = int(os.getenv("RAG_VALIDATION_RATE_LIMIT_MAX_RETRIES", "2"))
VALIDATION_TIMEOUT_RETRY_SECONDS = float(os.getenv("RAG_VALIDATION_TIMEOUT_RETRY_SECONDS", "30"))
VALIDATION_API_TIMEOUT_SECONDS = float(os.getenv("RAG_VALIDATION_API_TIMEOUT_SECONDS", "90"))
DEFAULT_NO_INFO_ANSWER = "Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor."
NEGATIVE_CASE_TAGS = frozenset({"negativfall", "fehlendes-detail"})
SOURCE_POLICIES = frozenset({"exact_gold_source", "accepted_sources", "official_hsh_any"})
EXPECTED_BEHAVIORS = frozenset(
    {
        "direct_answer",
        "clarify_then_answer",
        "missing_detail_answer",
        "refusal",
    }
)
FAILURE_TYPES = frozenset(
    {
        "retrieval_failed",
        "wrong_source_retrieved",
        "answer_ignored_evidence",
        "missing_required_fact",
        "missing_optional_fact",
        "hallucination",
        "ambiguous_question",
        "reference_too_strict",
        "source_conflict",
        "should_have_clarified",
        "wrong_refusal",
        "citation_problem",
        "dialog_problem",
        "stage_disagreement",
    }
)
RUBRIC_FIELDS = (
    "correctness",
    "completeness",
    "faithfulness",
    "source_use",
    "dialog_behavior",
)


def _normalize_eval_pipeline_config(config: Any = None, *, top_k: int | None = None):
    if normalize_pipeline_config is None:
        config_dict = config if isinstance(config, dict) else {}
        raw_top_k = config_dict.get("top_k", top_k or 6)
        try:
            parsed_top_k = int(raw_top_k)
        except (TypeError, ValueError):
            parsed_top_k = int(top_k or 6)
        return {
            "top_k": max(1, parsed_top_k),
            "query_assist_enabled": bool(config_dict.get("query_assist_enabled", True)),
            "retrieval_planner_enabled": bool(config_dict.get("retrieval_planner_enabled", True)),
            "rag_followup_enabled": bool(config_dict.get("rag_followup_enabled", True)),
        }
    return normalize_pipeline_config(config, top_k=top_k)


def _pipeline_config_dict(config: Any = None, *, top_k: int | None = None) -> dict[str, Any]:
    normalized = _normalize_eval_pipeline_config(config, top_k=top_k)
    if hasattr(normalized, "to_dict"):
        return normalized.to_dict()
    if isinstance(normalized, dict):
        return dict(normalized)
    return {
        "top_k": int(top_k or 6),
        "query_assist_enabled": True,
        "retrieval_planner_enabled": True,
        "rag_followup_enabled": True,
    }

EVALUATOR_PROMPT = """\
Du bist ein strenger Gutachter fuer ein RAG-System der Hochschule Hannover (HsH).

Ziel:
- Vergleiche die finale Chatbot-Antwort mit der Referenzantwort.
- Beruecksichtige auch den Dialogverlauf, das erwartete Klaerungsverhalten und die verwendeten Quellen.
- Bewerte zuerst den Fallvertrag: required_facts, optional_facts, forbidden_claims,
  expected_behavior, source_policy und answer_variants.
- Bevorzuge korrekte, hilfreiche und quellengetreue Antworten.
- Bestrafe erfundene Details, unnoetige Sicherheit und Antworten, die an der Referenz vorbeigehen.

Wichtige Regeln:
- Bewerte nur auf Basis der gegebenen Informationen.
- required_facts sind Pflicht. Fehlende required_facts sollen correctness und completeness deutlich senken.
- optional_facts sind Zusatzkontext. Fehlende optional_facts duerfen die Antwort nur leicht senken,
  wenn die Nutzerfrage trotzdem beantwortet wurde.
- forbidden_claims sind rote Linien. Wenn die Antwort solche Claims macht, bestrafe faithfulness/source_use streng.
- Wenn answer_variants vorhanden sind, kann die Antwort auch dann voll korrekt sein, wenn sie genau einen
  gueltigen Antwortpfad vollstaendig erfuellt. Bestrafe dann nicht, dass Pflichtfakten eines anderen
  Antwortpfads fehlen.
- Wenn mehrere Antwortpfade gleichzeitig hilfreich waeren, belohne eine kurze Einordnung der Mehrdeutigkeit.
- Wenn die Referenzantwort ausdruecklich sagt, dass keine Information aus offiziellen HsH-Dokumenten vorliegt,
  ist eine vorsichtige Ablehnung die beste Antwort.
- Wenn tags negativfall oder fehlendes-detail enthalten, ist eine klare Abgrenzung Teil der erwarteten Antwort:
  Die Assistant-Antwort darf nicht schlechter bewertet werden, nur weil sie sagt, dass eine verlangte Detailinformation
  in der HsH-Quelle nicht enthalten ist. Belohne stattdessen grounded refusal plus belegte HsH-Fakten.
- Bestrafe in negativfall/fehlendes-detail Faellen externe oder erfundene Details besonders deutlich, z.B.
  Studentenwerk-Oeffnungszeiten, wenn nur HsH-Quellen erlaubt sind und die HsH-Seite diese Zeiten nicht nennt.
- Bei case_type clarification soll das System die fest definierte Klaerung aus dem Fall respektieren, bevor es final antwortet.
- Wenn source_policy exact_gold_source oder accepted_sources ist, bewerte eine Antwort mit anderer Quelle kritischer,
  selbst wenn sie allgemein plausibel wirkt.
- Nutze failure_types nur aus dieser Liste, wenn sie wirklich passen:
  retrieval_failed, wrong_source_retrieved, answer_ignored_evidence, missing_required_fact,
  missing_optional_fact, hallucination, ambiguous_question, reference_too_strict,
  source_conflict, should_have_clarified, wrong_refusal, citation_problem, dialog_problem,
  stage_disagreement.
- Gib fuer jedes Kriterium eine ganze Zahl zwischen 0 und 5.
- Antworte NUR mit einem JSON-Objekt, ohne Markdown.

Erwartetes JSON-Schema:
{
  "scores": {
    "correctness": 0,
    "completeness": 0,
    "faithfulness": 0,
    "source_use": 0,
    "dialog_behavior": 0
  },
  "summary": "",
  "strengths": [],
  "issues": [],
  "diagnostics": {
    "matched_answer_variant": "",
    "matched_required_facts": [],
    "missing_required_facts": [],
    "matched_optional_facts": [],
    "missing_optional_facts": [],
    "forbidden_claims_found": []
  },
  "failure_types": []
}
"""


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    reference_answer: str
    case_type: str
    expected_behavior: str
    source_url: str
    source_policy: str
    accepted_source_urls: tuple[str, ...]
    required_facts: tuple[str, ...]
    optional_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    answer_variants: tuple[dict[str, Any], ...]
    case_difficulty: str
    fairness_risk: str
    selected_option: str
    clarification_text: str
    clarification_expected: bool
    evaluation_notes: str
    tags: tuple[str, ...]

    @property
    def clarification(self) -> dict[str, Any]:
        return {
            "expected": self.clarification_expected,
            "selected_option": self.selected_option,
            "clarification_text": self.clarification_text,
        }

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "expected_behavior": self.expected_behavior,
            "required_facts": list(self.required_facts),
            "optional_facts": list(self.optional_facts),
            "forbidden_claims": list(self.forbidden_claims),
            "answer_variants": list(self.answer_variants),
            "source_policy": self.source_policy,
            "source_url": self.source_url,
            "accepted_source_urls": list(self.accepted_source_urls),
            "case_difficulty": self.case_difficulty,
            "fairness_risk": self.fairness_risk,
        }


def _slugify(value: str) -> str:
    normalized = value.casefold()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or "run"


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


def _sanitize_string_list(values: Any, *, limit: int = 4) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.split()).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) >= limit:
            break
    return cleaned


def _clean_string_tuple(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = " ".join(value.split()).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return tuple(cleaned)


def _clean_answer_variants(values: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, list):
        return ()
    variants: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(values, start=1):
        if not isinstance(value, dict):
            continue
        variant_id = str(value.get("id") or f"variant_{index}").strip()
        if not variant_id:
            variant_id = f"variant_{index}"
        key = variant_id.casefold()
        if key in seen_ids:
            continue
        seen_ids.add(key)
        variants.append(
            {
                "id": variant_id,
                "title": str(value.get("title") or variant_id).strip(),
                "source_url": str(value.get("source_url") or "").strip(),
                "source_policy": _normalize_source_policy(value.get("source_policy")),
                "accepted_source_urls": list(_clean_string_tuple(value.get("accepted_source_urls", []))),
                "required_facts": list(_clean_string_tuple(value.get("required_facts", []))),
                "optional_facts": list(_clean_string_tuple(value.get("optional_facts", []))),
                "forbidden_claims": list(_clean_string_tuple(value.get("forbidden_claims", []))),
            }
        )
    return tuple(variants)


def _normalize_expected_behavior(value: Any, *, clarification_expected: bool, tags: tuple[str, ...]) -> str:
    behavior = str(value or "").strip().casefold()
    if behavior in EXPECTED_BEHAVIORS:
        return behavior
    if clarification_expected:
        return "clarify_then_answer"
    if any(tag.casefold() in NEGATIVE_CASE_TAGS for tag in tags):
        return "missing_detail_answer"
    return "direct_answer"


def _normalize_source_policy(value: Any) -> str:
    policy = str(value or "").strip().casefold()
    return policy if policy in SOURCE_POLICIES else "exact_gold_source"


def _sanitize_failure_types(values: Any, *, limit: int = 8) -> list[str]:
    cleaned = []
    for value in _sanitize_string_list(values, limit=limit * 2):
        key = value.casefold()
        if key in FAILURE_TYPES and key not in cleaned:
            cleaned.append(key)
        if len(cleaned) >= limit:
            break
    return cleaned


def _format_sources_for_evaluator(results: list[dict]) -> list[dict[str, str]]:
    formatted = []
    for result in results:
        payload = result.get("payload") or {}
        formatted.append(
            {
                "title": payload.get("title", ""),
                "url": payload.get("source_url", ""),
                "section": payload.get("section_heading", ""),
                "date": payload.get("crawl_date", ""),
            }
        )
    return formatted


def _case_has_tag(case: EvalCase, expected_tags: frozenset[str]) -> bool:
    return any(tag.casefold() in expected_tags for tag in case.tags)


def is_negative_case(case: EvalCase) -> bool:
    return _case_has_tag(case, NEGATIVE_CASE_TAGS)


def _case_payload(case: EvalCase) -> dict[str, Any]:
    return {
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
        "negative_case": is_negative_case(case),
        "required_facts": list(case.required_facts),
        "optional_facts": list(case.optional_facts),
        "forbidden_claims": list(case.forbidden_claims),
        "answer_variants": list(case.answer_variants),
        "case_difficulty": case.case_difficulty,
        "fairness_risk": case.fairness_risk,
        "evaluation_notes": case.evaluation_notes,
        "tags": list(case.tags),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    text = str(exc).casefold()
    return "rate limit" in text or "too many requests" in text or "429" in text


def _is_timeout_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "APITimeoutError":
        return True
    text = str(exc).casefold()
    return "timed out" in text or "timeout" in text


def _run_with_rate_limit_retry(
    operation,
    *,
    description: str,
    notify,
):
    for attempt in range(VALIDATION_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:
            if _is_rate_limit_error(exc):
                retry_reason = "API-Limit erreicht"
                retry_seconds = VALIDATION_RATE_LIMIT_RETRY_SECONDS
            elif _is_timeout_error(exc):
                retry_reason = "Request-Timeout"
                retry_seconds = VALIDATION_TIMEOUT_RETRY_SECONDS
            else:
                raise
            if attempt >= VALIDATION_RATE_LIMIT_MAX_RETRIES:
                raise
            notify(
                f"{description}: {retry_reason}. Warte "
                f"{retry_seconds:.0f}s und versuche es erneut."
            )
            time.sleep(retry_seconds)
    raise RuntimeError(f"{description}: API-Fehler konnte nicht abgefangen werden.")


def _sleep_between_cases(
    *,
    case_index: int,
    case_count: int,
    delay_seconds: float,
    status_writer=None,
) -> None:
    if delay_seconds <= 0 or case_index >= case_count:
        return
    if status_writer is not None:
        status_writer.write(
            f"Kurze Pause gegen API-Limits: {delay_seconds:.0f}s vor dem nächsten Fall."
        )
    time.sleep(delay_seconds)


def load_raw_eval_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    cases_path = Path(path).expanduser() if path is not None else CASES_FILE
    raw = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Validierungsdatei muss eine JSON-Liste enthalten: {cases_path}")
    return raw


def load_eval_cases(path: str | Path | None = None) -> list[EvalCase]:
    raw = load_raw_eval_cases(path)
    cases: list[EvalCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Validierungsfall an Position {index} muss ein JSON-Objekt sein.")
        clarification = item.get("clarification") or {}
        if not isinstance(clarification, dict):
            clarification = {}
        case_type = str(item.get("case_type") or "direct").strip() or "direct"
        clarification_expected = (
            bool(clarification.get("expected")) or case_type.casefold() == "clarification"
        )
        tags = _clean_string_tuple(item.get("tags", []))
        cases.append(
            EvalCase(
                id=str(item.get("id") or "").strip(),
                question=str(item.get("question") or "").strip(),
                reference_answer=str(item.get("reference_answer") or "").strip(),
                case_type=case_type,
                expected_behavior=_normalize_expected_behavior(
                    item.get("expected_behavior"),
                    clarification_expected=clarification_expected,
                    tags=tags,
                ),
                source_url=str(item.get("source_url") or "").strip(),
                source_policy=_normalize_source_policy(item.get("source_policy")),
                accepted_source_urls=_clean_string_tuple(item.get("accepted_source_urls", [])),
                required_facts=_clean_string_tuple(item.get("required_facts", [])),
                optional_facts=_clean_string_tuple(item.get("optional_facts", [])),
                forbidden_claims=_clean_string_tuple(item.get("forbidden_claims", [])),
                answer_variants=_clean_answer_variants(item.get("answer_variants", [])),
                case_difficulty=str(item.get("case_difficulty") or "mittel").strip() or "mittel",
                fairness_risk=str(item.get("fairness_risk") or "normal").strip() or "normal",
                selected_option=str(clarification.get("selected_option") or "").strip(),
                clarification_text=str(clarification.get("clarification_text") or "").strip(),
                clarification_expected=clarification_expected,
                evaluation_notes=str(item.get("evaluation_notes") or "").strip(),
                tags=tags,
            )
        )
    return cases


def _empty_validation_state() -> dict[str, Any]:
    return {
        "started": False,
        "run_id": "",
        "run_dir": "",
        "case_index": 0,
        "results": [],
        "active_case_id": "",
        "completed": False,
        "chatbot_model": "",
        "evaluator_model": "",
        "pipeline_config": _pipeline_config_dict(top_k=6),
    }


def _require_streamlit():
    if st is None:
        raise RuntimeError("Streamlit ist nicht installiert; die Web-Ansicht ist nicht verfügbar.")
    return st


def get_validation_state() -> dict[str, Any]:
    streamlit = _require_streamlit()
    if VALIDATION_STATE_KEY not in streamlit.session_state:
        streamlit.session_state[VALIDATION_STATE_KEY] = _empty_validation_state()
    return streamlit.session_state[VALIDATION_STATE_KEY]


def reset_validation_state() -> None:
    _require_streamlit().session_state[VALIDATION_STATE_KEY] = _empty_validation_state()


def _build_run_summary(state: dict[str, Any]) -> dict[str, Any]:
    results = state.get("results", [])
    averages = {
        field: round(mean(item["evaluation"]["scores"].get(field, 0) for item in results), 2)
        for field in RUBRIC_FIELDS
    } if results else {}
    overall = round(mean(item["evaluation"]["overall_score"] for item in results), 2) if results else 0.0
    summary = {
        "run_id": state.get("run_id", ""),
        "chatbot_model": state.get("chatbot_model", ""),
        "evaluator_model": state.get("evaluator_model", ""),
        "pipeline_config": _pipeline_config_dict(state.get("pipeline_config")),
        "completed_cases": len(results),
        "overall_score": overall,
        "averages": averages,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    evidence_results = [
        item["evidence_evaluation"]
        for item in results
        if isinstance(item.get("evidence_evaluation"), dict)
    ]
    if evidence_results:
        evidence_averages = {
            field: round(mean(item["scores"].get(field, 0) for item in evidence_results), 2)
            for field in EVIDENCE_RUBRIC_FIELDS
        }
        evidence_overall = round(
            mean(item.get("overall_score", 0.0) for item in evidence_results),
            2,
        )
        verdict_counts: dict[str, int] = {}
        for item in evidence_results:
            verdict = str(item.get("verdict") or "unknown")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        failure_type_counts: dict[str, int] = {}
        human_review_count = 0
        for result in results:
            diagnostics = result.get("diagnostics") or {}
            if diagnostics.get("human_review_recommended"):
                human_review_count += 1
            for failure_type in diagnostics.get("failure_types") or []:
                key = str(failure_type or "unknown")
                failure_type_counts[key] = failure_type_counts.get(key, 0) + 1

        summary.update(
            {
                "evidence_completed_cases": len(evidence_results),
                "evidence_overall_score": evidence_overall,
                "evidence_averages": evidence_averages,
                "evidence_verdict_counts": verdict_counts,
                "failure_type_counts": failure_type_counts,
                "human_review_recommended_cases": human_review_count,
                "two_stage_overall_score": round((overall + evidence_overall) / 2, 2),
            }
        )
    return summary


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _write_summary(state: dict[str, Any]) -> None:
    run_dir = Path(state["run_dir"])
    summary = _build_run_summary(state)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def start_validation_run(
    *,
    chatbot_model: str,
    evaluator_model: str,
    cases: list[EvalCase],
    pipeline_config: Any = None,
    store_in_session: bool = True,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = (
        f"{timestamp}__chatbot-{_slugify(chatbot_model)}__judge-{_slugify(evaluator_model)}"
    )
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config_data = _pipeline_config_dict(pipeline_config)

    manifest = {
        "run_id": run_id,
        "chatbot_model": chatbot_model,
        "evaluator_model": evaluator_model,
        "mode": "diagnostic_two_stage_validation",
        "cases_file": str(CASES_FILE),
        "case_count": len(cases),
        "pipeline_config": pipeline_config_data,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state = {
        "started": True,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "case_index": 0,
        "results": [],
        "active_case_id": "",
        "completed": False,
        "chatbot_model": chatbot_model,
        "evaluator_model": evaluator_model,
        "pipeline_config": pipeline_config_data,
    }
    if store_in_session:
        _require_streamlit().session_state[VALIDATION_STATE_KEY] = state
    _write_summary(state)
    return state


def _persist_case_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    run_dir = Path(state["run_dir"])
    path = run_dir / f"{result['case']['id']}.json"
    path.write_text(
        json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(state)


def _empty_answer_evaluation() -> dict[str, Any]:
    scores = {field: 0 for field in RUBRIC_FIELDS}
    return {
        "scores": scores,
        "overall_score": 0.0,
        "summary": "Die Assistant-Antwort war leer und wurde deshalb deterministisch als Fehlschlag gewertet.",
        "strengths": [],
        "issues": [
            "Die finale Antwort des Chatbot-Modells war leer.",
            "Der Fall wurde nicht an das Evaluator-Modell weitergegeben, um Fehlbewertungen leerer Antworten zu vermeiden.",
        ],
        "diagnostics": {
            "matched_answer_variant": "",
            "matched_required_facts": [],
            "missing_required_facts": [],
            "matched_optional_facts": [],
            "missing_optional_facts": [],
            "forbidden_claims_found": [],
        },
        "failure_types": ["wrong_refusal"],
        "raw_judgement": "",
    }


def _score_with_evaluator(
    *,
    openai_client: OpenAI,
    evaluator_model: str,
    case: EvalCase,
    answer: str,
    transcript: list[dict[str, str]],
    process_trace: dict,
    results: list[dict],
) -> dict[str, Any]:
    evaluator_payload = {
        "case": _case_payload(case),
        "assistant_final_answer": answer,
        "transcript": transcript,
        "retrieved_sources": _format_sources_for_evaluator(results),
        "process_trace": {
            "selected_query": process_trace.get("selected_query", ""),
            "clarification": process_trace.get("clarification", ""),
            "assessment_intents": process_trace.get("assessment_intents", []),
            "clarification_needed": process_trace.get("clarification_needed", False),
        },
    }

    response = openai_client.chat.completions.create(
        model=evaluator_model,
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {
                "role": "user",
                "content": json.dumps(evaluator_payload, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_object(raw) or {}

    scores = {
        field: _sanitize_score((parsed.get("scores") or {}).get(field))
        for field in RUBRIC_FIELDS
    }
    overall_score = round(sum(scores.values()) / (len(RUBRIC_FIELDS) * 5) * 100, 1)
    parsed_diagnostics = parsed.get("diagnostics") if isinstance(parsed.get("diagnostics"), dict) else {}
    return {
        "scores": scores,
        "overall_score": overall_score,
        "summary": " ".join(str(parsed.get("summary") or "").split()).strip(),
        "strengths": _sanitize_string_list(parsed.get("strengths"), limit=5),
        "issues": _sanitize_string_list(parsed.get("issues"), limit=5),
        "diagnostics": {
            "matched_answer_variant": " ".join(
                str(parsed_diagnostics.get("matched_answer_variant") or "").split()
            ).strip(),
            "matched_required_facts": _sanitize_string_list(
                parsed_diagnostics.get("matched_required_facts"),
                limit=8,
            ),
            "missing_required_facts": _sanitize_string_list(
                parsed_diagnostics.get("missing_required_facts"),
                limit=8,
            ),
            "matched_optional_facts": _sanitize_string_list(
                parsed_diagnostics.get("matched_optional_facts"),
                limit=8,
            ),
            "missing_optional_facts": _sanitize_string_list(
                parsed_diagnostics.get("missing_optional_facts"),
                limit=8,
            ),
            "forbidden_claims_found": _sanitize_string_list(
                parsed_diagnostics.get("forbidden_claims_found"),
                limit=8,
            ),
        },
        "failure_types": _sanitize_failure_types(parsed.get("failure_types")),
        "raw_judgement": raw.strip(),
    }


def _answer_looks_like_refusal(answer: str) -> bool:
    normalized = answer.casefold()
    refusal_markers = (
        "keine information",
        "keine informationen",
        "nicht gefunden",
        "liegen mir keine",
        "kann ich nicht",
        "nicht ersichtlich",
        "nicht beschrieben",
    )
    return any(marker in normalized for marker in refusal_markers)


def _build_case_diagnostics(
    *,
    case: EvalCase,
    answer: str,
    evaluation: dict[str, Any],
    evidence_evaluation: dict[str, Any],
) -> dict[str, Any]:
    failure_types = set(_sanitize_failure_types(evaluation.get("failure_types"), limit=12))
    stage_1_score = float(evaluation.get("overall_score", 0.0) or 0.0)
    stage_2_score = float(evidence_evaluation.get("overall_score", 0.0) or 0.0)
    stage_gap = round(stage_2_score - stage_1_score, 1)

    eval_diagnostics = evaluation.get("diagnostics") or {}
    if eval_diagnostics.get("missing_required_facts"):
        failure_types.add("missing_required_fact")
    if eval_diagnostics.get("missing_optional_facts") and stage_1_score < 90:
        failure_types.add("missing_optional_fact")
    if eval_diagnostics.get("forbidden_claims_found"):
        failure_types.add("hallucination")

    source_check = evidence_evaluation.get("source_check") or {}
    source_policy_satisfied = bool(source_check.get("source_policy_satisfied", False))
    if not source_policy_satisfied:
        if source_check.get("retrieved_source_urls"):
            failure_types.add("wrong_source_retrieved")
        else:
            failure_types.add("retrieval_failed")

    verdict = str(evidence_evaluation.get("verdict") or "").casefold()
    evidence_scores = evidence_evaluation.get("scores") or {}
    if verdict == "retrieval_failed":
        failure_types.add("retrieval_failed")
    if verdict == "wrong_source":
        failure_types.add("wrong_source_retrieved")
    if evidence_evaluation.get("unsupported_claims") or evidence_evaluation.get("contradicted_claims"):
        failure_types.add("hallucination")
    if (
        source_policy_satisfied
        and int(evidence_scores.get("evidence_sufficiency", 0) or 0) >= 3
        and int(evidence_scores.get("answer_grounding", 0) or 0) <= 2
    ):
        failure_types.add("answer_ignored_evidence")
    if int(evidence_scores.get("citation_support", 0) or 0) <= 2 and source_check.get("retrieved_source_urls"):
        failure_types.add("citation_problem")
    if case.clarification_expected and int(evaluation["scores"].get("dialog_behavior", 0)) <= 2:
        failure_types.add("should_have_clarified")
    if _answer_looks_like_refusal(answer) and case.expected_behavior == "direct_answer":
        failure_types.add("wrong_refusal")
    if abs(stage_gap) >= 30:
        failure_types.add("stage_disagreement")
        if stage_gap >= 30:
            failure_types.add("reference_too_strict")

    human_review_recommended = (
        bool(failure_types.intersection({"stage_disagreement", "reference_too_strict", "source_conflict"}))
        or case.fairness_risk.casefold() in {"hoch", "high"}
        or (case.fairness_risk.casefold() == "mittel" and bool(failure_types))
        or abs(stage_gap) >= 30
    )
    notes = []
    if stage_gap >= 30:
        notes.append(
            "Stufe 2 bewertet deutlich besser als Stufe 1; Referenz, Frage oder Quellenpolitik sollten geprüft werden."
        )
    elif stage_gap <= -30:
        notes.append(
            "Stufe 1 bewertet deutlich besser als Stufe 2; die Antwort passt zur Referenz, ist aber schwach belegt."
        )
    if not source_policy_satisfied:
        notes.append("Die konfigurierte Quellenpolitik wurde durch die Retrieval-Ergebnisse nicht erfüllt.")

    return {
        "failure_types": sorted(failure_types),
        "stage_gap": stage_gap,
        "human_review_recommended": human_review_recommended,
        "notes": notes,
    }


def run_strict_dialog_case(
    *,
    case: EvalCase,
    qdrant,
    dense_embedder,
    sparse_embedder,
    reranker,
    openai_client: OpenAI,
    chatbot_model: str,
    evaluator_model: str,
    top_k: int,
    pipeline_config: Any = None,
    status_writer=None,
) -> dict[str, Any]:
    if prepare_chat_turn is None:
        raise RuntimeError(
            "RAG runtime helpers are unavailable. Install runtime dependencies "
            "such as qdrant-client before running validation cases."
        )

    transcript: list[dict[str, str]] = [{"role": "user", "content": case.question}]

    def notify(message: str) -> None:
        if status_writer is not None:
            status_writer.write(message)

    selected_option = case.selected_option if case.clarification_expected else None
    clarification_text = case.clarification_text if case.clarification_expected else ""
    if case.clarification_expected:
        transcript.append(
            {
                "role": "assistant",
                "content": "Strict dialog: Query-Assist-Klärung wurde durch den festen Eval-Fall ausgelöst.",
            }
        )
        if selected_option:
            transcript.append({"role": "user", "content": selected_option})
        if clarification_text:
            transcript.append({"role": "user", "content": clarification_text})

    runtime_pipeline_config = _normalize_eval_pipeline_config(pipeline_config, top_k=top_k)
    prepared = prepare_chat_turn(
        qdrant=qdrant,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        reranker=reranker,
        openai_client=openai_client,
        model=chatbot_model,
        rolle=DEFAULT_ROLE,
        fakultaet=DEFAULT_FACULTY,
        question=case.question,
        top_k=top_k,
        selected_query=selected_option,
        clarification=clarification_text,
        conversation_memory=[],
        status_callback=notify,
        pipeline_config=runtime_pipeline_config,
    )

    timings = dict(prepared["timings"])
    process_trace = prepared["process_trace"]
    results = prepared["results"]

    if prepared["no_results"]:
        answer = ""
        notify("Keine Treffer in der Wissensdatenbank gefunden.")
        timings["Antwort-LLM"] = 0.0
    elif prepared["guard_answer"] is not None:
        answer = prepared["guard_answer"]
        notify("Zeitkritische Frage erkannt; Guard-Antwort verwendet.")
        timings["Antwort-LLM"] = 0.0
    else:
        if request_answer_sync is None:
            raise RuntimeError(
                "Answer-generation helper is unavailable. Install runtime dependencies "
                "such as qdrant-client before running validation cases."
            )
        notify("Finale Antwort wird mit dem Chatbot-Modell erzeugt.")
        llm_t0 = time.perf_counter()
        answer = _run_with_rate_limit_retry(
            lambda: request_answer_sync(
                openai_client,
                chatbot_model,
                prepared["llm_messages"],
            ),
            description="Antwort-LLM",
            notify=notify,
        )
        if not (answer or "").strip():
            notify("Leere Modellantwort erkannt; einmaliger Retry wird ausgeführt.")
            answer = _run_with_rate_limit_retry(
                lambda: request_answer_sync(
                    openai_client,
                    chatbot_model,
                    prepared["llm_messages"],
                ),
                description="Antwort-LLM Retry",
                notify=notify,
            )
        timings["Antwort-LLM"] = time.perf_counter() - llm_t0

    answer = answer.strip() or ""
    if not answer and (case.reference_answer == DEFAULT_NO_INFO_ANSWER or is_negative_case(case)):
        answer = DEFAULT_NO_INFO_ANSWER
    transcript.append({"role": "assistant", "content": answer})

    if not answer:
        notify("Leere Assistant-Antwort bleibt bestehen; der Fall wird deterministisch als Fehlschlag markiert.")
        evaluation = _empty_answer_evaluation()
    else:
        notify("Stufe 1: Antwort wird mit dem Evaluator-Modell bewertet.")
        stage_t0 = time.perf_counter()
        evaluation = _run_with_rate_limit_retry(
            lambda: _score_with_evaluator(
                openai_client=openai_client,
                evaluator_model=evaluator_model,
                case=case,
                answer=answer,
                transcript=transcript,
                process_trace=process_trace,
                results=results,
            ),
            description="Stufe 1",
            notify=notify,
        )
        timings["Stufe 1 Bewertung"] = time.perf_counter() - stage_t0

    notify("Stufe 2: Antwort wird gegen die abgerufenen Qdrant-Chunks geprüft.")
    evidence_t0 = time.perf_counter()
    evidence_evaluation = _run_with_rate_limit_retry(
        lambda: score_evidence_grounding(
            openai_client=openai_client,
            evaluator_model=evaluator_model,
            case=case,
            answer=answer,
            results=results,
            process_trace=process_trace,
            default_no_info_answer=DEFAULT_NO_INFO_ANSWER,
        ),
        description="Stufe 2",
        notify=notify,
    )
    timings["Stufe 2 Evidenzprüfung"] = time.perf_counter() - evidence_t0
    timings["Gesamt"] = round(sum(float(value) for value in timings.values()), 2)
    diagnostics = _build_case_diagnostics(
        case=case,
        answer=answer,
        evaluation=evaluation,
        evidence_evaluation=evidence_evaluation,
    )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case": _case_payload(case),
        "chatbot_model": chatbot_model,
        "evaluator_model": evaluator_model,
        "pipeline_config": _pipeline_config_dict(runtime_pipeline_config),
        "transcript": transcript,
        "answer": answer,
        "process_trace": process_trace,
        "results": results,
        "timings": timings,
        "evaluation": evaluation,
        "evidence_evaluation": evidence_evaluation,
        "diagnostics": diagnostics,
    }


def _render_evaluation(result: dict[str, Any]) -> None:
    evaluation = result["evaluation"]
    st.markdown("**Stufe 1: Antwortbewertung**")
    cols = st.columns(len(RUBRIC_FIELDS) + 1)
    for index, field in enumerate(RUBRIC_FIELDS):
        cols[index].metric(field.replace("_", " ").title(), f"{evaluation['scores'][field]}/5")
    cols[-1].metric("Gesamt", f"{evaluation['overall_score']:.1f}/100")

    if evaluation.get("summary"):
        st.markdown(f"**Bewertung:** {evaluation['summary']}")
    if evaluation.get("strengths"):
        st.markdown("**Stärken**")
        for item in evaluation["strengths"]:
            st.markdown(f"- {item}")
    if evaluation.get("issues"):
        st.markdown("**Verbesserungspunkte**")
        for item in evaluation["issues"]:
            st.markdown(f"- {item}")
    diagnostics = evaluation.get("diagnostics") or {}
    if (
        diagnostics.get("matched_answer_variant")
        or diagnostics.get("missing_required_facts")
        or diagnostics.get("forbidden_claims_found")
    ):
        with st.expander("Faktenvertrag Stufe 1", expanded=False):
            if diagnostics.get("matched_answer_variant"):
                st.markdown(f"**Erkannter Antwortpfad:** `{diagnostics['matched_answer_variant']}`")
            if diagnostics.get("missing_required_facts"):
                st.markdown("**Fehlende Pflichtfakten**")
                for item in diagnostics["missing_required_facts"]:
                    st.markdown(f"- {item}")
            if diagnostics.get("forbidden_claims_found"):
                st.markdown("**Gefundene verbotene Claims**")
                for item in diagnostics["forbidden_claims_found"]:
                    st.markdown(f"- {item}")


def _render_evidence_evaluation(result: dict[str, Any]) -> None:
    evidence = result.get("evidence_evaluation")
    if not isinstance(evidence, dict):
        return

    st.markdown("**Stufe 2: Evidenzprüfung**")
    cols = st.columns(len(EVIDENCE_RUBRIC_FIELDS) + 1)
    scores = evidence.get("scores") or {}
    for index, field in enumerate(EVIDENCE_RUBRIC_FIELDS):
        cols[index].metric(field.replace("_", " ").title(), f"{scores.get(field, 0)}/5")
    cols[-1].metric("Evidenz", f"{evidence.get('overall_score', 0):.1f}/100")

    verdict = evidence.get("verdict", "")
    source_check = evidence.get("source_check") or {}
    policy = source_check.get("source_policy", "exact_gold_source")
    policy_ok = source_check.get("source_policy_satisfied", source_check.get("gold_source_found"))
    source_label = "erfüllt" if policy_ok else "nicht erfüllt"
    source_rank = source_check.get("source_policy_match_rank", source_check.get("gold_source_rank"))
    if source_rank is not None:
        source_label += f" (Rang {source_rank})"
    st.caption(f"Verdikt: `{verdict}` · Quellenpolitik `{policy}`: {source_label}")

    if evidence.get("summary"):
        st.markdown(f"**Evidenzbewertung:** {evidence['summary']}")
    if evidence.get("unsupported_claims"):
        st.markdown("**Nicht belegte Claims**")
        for item in evidence["unsupported_claims"]:
            st.markdown(f"- {item}")
    if evidence.get("contradicted_claims"):
        st.markdown("**Widersprochene Claims**")
        for item in evidence["contradicted_claims"]:
            st.markdown(f"- {item}")
    if evidence.get("issues"):
        with st.expander("Evidenzhinweise", expanded=False):
            for item in evidence["issues"]:
                st.markdown(f"- {item}")


def _render_diagnostics(result: dict[str, Any]) -> None:
    diagnostics = result.get("diagnostics") or {}
    if not diagnostics:
        return
    failure_types = diagnostics.get("failure_types") or []
    if not failure_types and not diagnostics.get("notes"):
        return
    st.markdown("**Diagnose**")
    if failure_types:
        st.caption(" · ".join(f"`{item}`" for item in failure_types))
    if diagnostics.get("human_review_recommended"):
        st.warning("Dieser Fall sollte fachlich geprüft werden, bevor der Score als endgültig interpretiert wird.")
    for note in diagnostics.get("notes") or []:
        st.markdown(f"- {note}")


def _render_transcript(result: dict[str, Any]) -> None:
    with st.expander("Dialogverlauf", expanded=True):
        for turn in result["transcript"]:
            label = "Nutzer" if turn["role"] == "user" else "Assistent"
            st.markdown(f"**{label}:** {turn['content']}")


def _render_sources(result: dict[str, Any]) -> None:
    results = result.get("results") or []
    if not results:
        return
    with st.expander("Verwendete Quellen", expanded=False):
        for idx, item in enumerate(results, start=1):
            payload = item.get("payload") or {}
            title = payload.get("title", "(kein Titel)")
            url = payload.get("source_url", "")
            section = payload.get("section_heading", "")
            st.markdown(f"**{idx}. {title}**")
            if url:
                st.markdown(f"[{url}]({url})")
            meta = []
            if section:
                meta.append(f"Abschnitt: {section}")
            if payload.get("crawl_date"):
                meta.append(f"Stand: {payload['crawl_date']}")
            meta.append(f"Score: {item.get('score', 0):.4f}")
            st.caption(" · ".join(meta))
            if idx < len(results):
                st.divider()


def _render_process_trace(result: dict[str, Any]) -> None:
    trace = result.get("process_trace") or {}
    with st.expander("Systemprozess", expanded=False):
        st.markdown(f"**Originalfrage:** {trace.get('original_query', '')}")
        if trace.get("selected_query") and trace["selected_query"] != trace.get("original_query", ""):
            st.markdown(f"**Verwendete Formulierung:** {trace['selected_query']}")
        if trace.get("clarification"):
            st.markdown(f"**Zusatz:** {trace['clarification']}")
        if trace.get("assessment_intents"):
            st.markdown(f"**Erkannte Suchtypen:** {', '.join(trace['assessment_intents'])}")
        if trace.get("retrieval_queries"):
            st.markdown("**Suchpfade:**")
            for query in trace["retrieval_queries"]:
                st.markdown(f"- `{query}`")


def _render_case_result(result: dict[str, Any]) -> None:
    st.subheader(f"{result['case']['id']} abgeschlossen")
    st.markdown(f"**Frage:** {result['case']['question']}")
    st.markdown("**Referenzantwort**")
    st.markdown(result["case"]["reference_answer"])
    st.markdown("**Chatbot-Antwort**")
    st.markdown(result["answer"] or "_Keine Antwort erzeugt._")
    _render_evaluation(result)
    _render_evidence_evaluation(result)
    _render_diagnostics(result)
    _render_transcript(result)
    _render_sources(result)
    _render_process_trace(result)


def _render_previous_results(state: dict[str, Any]) -> None:
    results = state.get("results", [])
    if not results:
        return
    with st.expander("Bereits ausgewertete Fälle", expanded=False):
        for result in results:
            score = result["evaluation"]["overall_score"]
            evidence = result.get("evidence_evaluation") or {}
            evidence_score = evidence.get("overall_score")
            suffix = f" · Evidenz `{evidence_score:.1f}/100`" if isinstance(evidence_score, (int, float)) else ""
            st.markdown(
                f"- **{result['case']['id']}**: {result['case']['question']} "
                f"(`{score:.1f}/100`{suffix})"
            )


def record_validation_result(
    state: dict[str, Any],
    result: dict[str, Any],
    *,
    next_case_index: int | None = None,
) -> None:
    state["results"].append(result)
    state["active_case_id"] = result["case"]["id"]
    if next_case_index is not None:
        state["case_index"] = next_case_index
    _persist_case_result(state, result)


def complete_validation_run(state: dict[str, Any]) -> None:
    state["completed"] = True
    _write_summary(state)


def _run_current_case(
    *,
    state: dict[str, Any],
    case: EvalCase,
    qdrant,
    dense_embedder,
    sparse_embedder,
    reranker,
    openai_client: OpenAI,
    top_k: int,
) -> None:
    pipeline_config = _normalize_eval_pipeline_config(state.get("pipeline_config"), top_k=top_k)
    with st.status("Strict dialog läuft …", expanded=True) as status:
        result = run_strict_dialog_case(
            case=case,
            qdrant=qdrant,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            openai_client=openai_client,
            chatbot_model=state["chatbot_model"],
            evaluator_model=state["evaluator_model"],
            top_k=top_k,
            pipeline_config=pipeline_config,
            status_writer=status,
        )
        status.update(label="Fall abgeschlossen", state="complete")

    record_validation_result(state, result)


def run_remaining_validation_cases(
    *,
    state: dict[str, Any],
    cases: list[EvalCase],
    qdrant,
    dense_embedder,
    sparse_embedder,
    reranker,
    openai_client: OpenAI,
    top_k: int,
    pipeline_config: Any = None,
    status_writer=None,
    progress_callback=None,
    inter_case_delay_seconds: float | None = None,
) -> list[dict[str, Any]]:
    completed_ids = {
        item["case"]["id"]
        for item in state.get("results", [])
        if item.get("case", {}).get("id")
    }
    state["case_index"] = max(0, int(state.get("case_index", 0)))
    new_results: list[dict[str, Any]] = []
    delay_seconds = (
        VALIDATION_INTER_CASE_DELAY_SECONDS
        if inter_case_delay_seconds is None
        else inter_case_delay_seconds
    )
    runtime_pipeline_config = _normalize_eval_pipeline_config(
        pipeline_config if pipeline_config is not None else state.get("pipeline_config"),
        top_k=top_k,
    )
    state["pipeline_config"] = _pipeline_config_dict(runtime_pipeline_config)

    for index, case in enumerate(cases):
        if case.id in completed_ids:
            state["case_index"] = max(int(state.get("case_index", 0)), index + 1)
            continue

        state["active_case_id"] = case.id
        if status_writer is not None:
            status_writer.write(f"{case.id} wird ausgeführt ({index + 1}/{len(cases)}).")

        result = run_strict_dialog_case(
            case=case,
            qdrant=qdrant,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            openai_client=openai_client,
            chatbot_model=state["chatbot_model"],
            evaluator_model=state["evaluator_model"],
            top_k=top_k,
            pipeline_config=runtime_pipeline_config,
            status_writer=status_writer,
        )
        record_validation_result(
            state,
            result,
            next_case_index=max(int(state.get("case_index", 0)), index + 1),
        )
        completed_ids.add(case.id)
        new_results.append(result)

        if progress_callback is not None:
            progress_callback(index + 1, len(cases), result)
        _sleep_between_cases(
            case_index=index + 1,
            case_count=len(cases),
            delay_seconds=delay_seconds,
            status_writer=status_writer,
        )

    if len(completed_ids) >= len(cases):
        state["case_index"] = len(cases)
        complete_validation_run(state)
    else:
        _write_summary(state)

    return new_results


def _run_all_cases_from_ui(
    *,
    state: dict[str, Any],
    cases: list[EvalCase],
    qdrant,
    dense_embedder,
    sparse_embedder,
    reranker,
    openai_client: OpenAI,
    top_k: int,
) -> None:
    pipeline_config = _normalize_eval_pipeline_config(state.get("pipeline_config"), top_k=top_k)
    progress = st.progress(
        min(int(state.get("case_index", 0)), len(cases)) / max(1, len(cases)),
        text="Validierung läuft sequenziell.",
    )

    def update_progress(done: int, total: int, result: dict[str, Any]) -> None:
        score = result["evaluation"]["overall_score"]
        evidence_score = (result.get("evidence_evaluation") or {}).get("overall_score")
        if isinstance(evidence_score, (int, float)):
            text = f"{result['case']['id']} abgeschlossen (Antwort {score:.1f}/100, Evidenz {evidence_score:.1f}/100)."
        else:
            text = f"{result['case']['id']} abgeschlossen ({score:.1f}/100)."
        progress.progress(
            done / max(1, total),
            text=text,
        )

    with st.status("Alle Fälle laufen sequenziell …", expanded=True) as status:
        status.write(
            f"Rate-Limit aktiv: maximal {MAX_EVAL_API_REQUESTS_PER_MINUTE} GWDG-API-Aufrufe pro Minute."
        )
        if VALIDATION_INTER_CASE_DELAY_SECONDS > 0:
            status.write(
                f"Zusätzliche Pause: ca. {VALIDATION_INTER_CASE_DELAY_SECONDS:.0f}s zwischen Fällen."
            )
        status.write(
            f"Bei API-Limit-Fehlern wartet der Lauf {VALIDATION_RATE_LIMIT_RETRY_SECONDS:.0f}s "
            f"und bei Timeouts {VALIDATION_TIMEOUT_RETRY_SECONDS:.0f}s; er versucht denselben Schritt erneut "
            f"(max. {VALIDATION_RATE_LIMIT_MAX_RETRIES} Retry/Retrys)."
        )
        run_remaining_validation_cases(
            state=state,
            cases=cases,
            qdrant=qdrant,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            openai_client=openai_client,
            top_k=top_k,
            pipeline_config=pipeline_config,
            status_writer=status,
            progress_callback=update_progress,
        )
        status.update(label="Alle ausgewählten Fälle abgeschlossen", state="complete")


def render_validation_page(
    *,
    qdrant,
    dense_embedder,
    sparse_embedder,
    reranker,
    openai_client: OpenAI,
    bulk_openai_client: OpenAI | None = None,
    modelle: list[str],
    chatbot_model: str,
    evaluator_model: str,
    pipeline_config: Any = None,
) -> None:
    st.title("Validierungssystem")
    cases = load_eval_cases()
    state = get_validation_state()
    selected_pipeline_config = _normalize_eval_pipeline_config(pipeline_config)
    active_pipeline_config = _normalize_eval_pipeline_config(
        state.get("pipeline_config") if state["started"] else selected_pipeline_config,
        top_k=selected_pipeline_config.top_k if hasattr(selected_pipeline_config, "top_k") else None,
    )
    active_top_k = int(_pipeline_config_dict(active_pipeline_config).get("top_k", 6))
    active_chatbot_model = state["chatbot_model"] if state["started"] else chatbot_model
    active_evaluator_model = state["evaluator_model"] if state["started"] else evaluator_model

    st.caption(
        f"Feste Fallbasis aus `{CASES_FILE.name}` · "
        f"Chatbot: **{active_chatbot_model}** · Evaluator: **{active_evaluator_model}**"
    )

    st.markdown("**Zweistufige diagnostische Validierung**")
    st.markdown(
        "Die Fälle enthalten Pflichtfakten, optionale Fakten, verbotene Claims und Quellenpolitik. "
        "Stufe 1 bewertet die Antwortqualität; Stufe 2 prüft Retrieval, Quellenpolitik und Grounding."
    )
    st.caption(
        f"Fälle: {len(cases)} · Ergebnisse werden unter `{RESULTS_DIR}` gespeichert. "
        f"Alle-Fälle-Lauf: maximal {MAX_EVAL_API_REQUESTS_PER_MINUTE} GWDG-API-Aufrufe pro Minute, "
        f"plus ca. {VALIDATION_INTER_CASE_DELAY_SECONDS:.0f}s Pause zwischen Fällen. "
        f"Bei API-Limit-Fehlern oder Timeouts wartet der Lauf automatisch und versucht denselben Schritt erneut."
    )
    config_data = _pipeline_config_dict(active_pipeline_config)
    st.caption(
        "Pipeline-Konfiguration: "
        f"Top-K `{config_data['top_k']}` · "
        f"Query Assist `{'aktiv' if config_data['query_assist_enabled'] else 'inaktiv'}` · "
        f"Retrieval Planner `{'aktiv' if config_data['retrieval_planner_enabled'] else 'inaktiv'}` · "
        f"RAG Follow-up `{'aktiv' if config_data['rag_followup_enabled'] else 'inaktiv'}`"
    )
    if not cases:
        st.warning("Keine Validierungsfälle gefunden.")
        return

    if not state["started"]:
        st.markdown("**Aktive Fälle**")
        for case in cases:
            badge = "Klarstellung" if case.clarification_expected else "Direkt"
            st.markdown(f"- **{case.id}** [{badge}]: {case.question}")
        cols = st.columns(2)
        if cols[0].button("Aktuellen Fall ausführen", use_container_width=True):
            state = start_validation_run(
                chatbot_model=chatbot_model,
                evaluator_model=evaluator_model,
                cases=cases,
                pipeline_config=selected_pipeline_config,
            )
            _run_current_case(
                state=state,
                case=cases[0],
                qdrant=qdrant,
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
                reranker=reranker,
                openai_client=openai_client,
                top_k=active_top_k,
            )
            st.rerun()
        if cols[1].button("Alle Fälle ausführen", use_container_width=True):
            state = start_validation_run(
                chatbot_model=chatbot_model,
                evaluator_model=evaluator_model,
                cases=cases,
                pipeline_config=selected_pipeline_config,
            )
            _run_all_cases_from_ui(
                state=state,
                cases=cases,
                qdrant=qdrant,
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
                reranker=reranker,
                openai_client=bulk_openai_client or openai_client,
                top_k=active_top_k,
            )
            st.rerun()
        return

    st.caption(
        f"Run-ID: `{state['run_id']}` · Fortschritt: {min(state['case_index'] + 1, len(cases))}/{len(cases)}"
    )
    _render_previous_results(state)

    if state["completed"] or state["case_index"] >= len(cases):
        state["completed"] = True
        summary = _build_run_summary(state)
        st.success("Alle Fälle wurden ausgewertet.")
        cols = st.columns(2)
        cols[0].metric("Gesamtscore", f"{summary['overall_score']:.1f}/100")
        evidence_overall = summary.get("evidence_overall_score")
        if isinstance(evidence_overall, (int, float)):
            cols[1].metric("Evidenzscore", f"{evidence_overall:.1f}/100")
            st.caption(f"Fälle: {summary['completed_cases']}")
        else:
            cols[1].metric("Fälle", str(summary["completed_cases"]))
        if summary["averages"]:
            st.markdown("**Durchschnittswerte Stufe 1**")
            for field, value in summary["averages"].items():
                st.markdown(f"- {field.replace('_', ' ').title()}: {value}/5")
        if summary.get("evidence_averages"):
            st.markdown("**Durchschnittswerte Stufe 2**")
            for field, value in summary["evidence_averages"].items():
                st.markdown(f"- {field.replace('_', ' ').title()}: {value}/5")
        if summary.get("evidence_verdict_counts"):
            st.markdown("**Evidenz-Verdikte**")
            for verdict, count in sorted(summary["evidence_verdict_counts"].items()):
                st.markdown(f"- {verdict}: {count}")
        if summary.get("failure_type_counts"):
            st.markdown("**Diagnose-Typen**")
            for failure_type, count in sorted(summary["failure_type_counts"].items()):
                st.markdown(f"- {failure_type}: {count}")
        if summary.get("human_review_recommended_cases"):
            st.info(
                f"{summary['human_review_recommended_cases']} Fall/Fälle sind für eine fachliche Nachprüfung markiert."
            )
        if st.button("Neue Validierung starten", use_container_width=True):
            reset_validation_state()
            st.rerun()
        return

    case = cases[state["case_index"]]
    st.subheader(f"{case.id}: Fall {state['case_index'] + 1} von {len(cases)}")
    st.markdown(f"**Frage:** {case.question}")
    st.caption(
        f"Typ: {'Klarstellung' if case.clarification_expected else 'Direkt'}"
        + (f" · Quelle: [Link]({case.source_url})" if case.source_url else "")
    )
    if case.clarification_expected:
        st.markdown("**Fest definierte Klärung**")
        if case.selected_option:
            st.markdown(f"- Auswahl: `{case.selected_option}`")
        if case.clarification_text:
            st.markdown(f"- Zusatz: `{case.clarification_text}`")
    if case.evaluation_notes:
        st.info(case.evaluation_notes)

    current_result = next(
        (item for item in state["results"] if item["case"]["id"] == case.id),
        None,
    )
    if current_result is None:
        cols = st.columns(2)
        if cols[0].button("Aktuellen Fall ausführen", use_container_width=True):
            _run_current_case(
                state=state,
                case=case,
                qdrant=qdrant,
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
                reranker=reranker,
                openai_client=openai_client,
                top_k=active_top_k,
            )
            st.rerun()
        if cols[1].button("Alle Fälle ausführen", use_container_width=True):
            _run_all_cases_from_ui(
                state=state,
                cases=cases,
                qdrant=qdrant,
                dense_embedder=dense_embedder,
                sparse_embedder=sparse_embedder,
                reranker=reranker,
                openai_client=bulk_openai_client or openai_client,
                top_k=active_top_k,
            )
            st.rerun()
        return

    _render_case_result(current_result)
    cols = st.columns(2)
    if cols[0].button("Nächsten Fall laden", use_container_width=True):
        state["case_index"] += 1
        state["active_case_id"] = ""
        if state["case_index"] >= len(cases):
            state["completed"] = True
        st.rerun()
    if cols[1].button("Alle Fälle ausführen", use_container_width=True):
        _run_all_cases_from_ui(
            state=state,
            cases=cases,
            qdrant=qdrant,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            openai_client=bulk_openai_client or openai_client,
            top_k=active_top_k,
        )
        st.rerun()


__all__ = [
    "CASES_FILE",
    "DEFAULT_CASES_FILE",
    "EVIDENCE_RUBRIC_FIELDS",
    "MAX_EVAL_API_REQUESTS_PER_MINUTE",
    "RESULTS_DIR",
    "VALIDATION_API_TIMEOUT_SECONDS",
    "complete_validation_run",
    "is_negative_case",
    "load_eval_cases",
    "load_raw_eval_cases",
    "record_validation_result",
    "render_validation_page",
    "reset_validation_state",
    "run_remaining_validation_cases",
    "run_strict_dialog_case",
    "start_validation_run",
]
