"""
Gezielte zweite Retrieval-Runde fuer unvollstaendige Antworten.
"""

from __future__ import annotations

import json
import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

try:
    from .hybrid_search import build_rag_context, perform_hybrid_search
except ImportError:  # pragma: no cover - script execution fallback
    from hybrid_search import build_rag_context, perform_hybrid_search

FOLLOWUP_PLANNER_PROMPT = """\
Du bist ein Retrieval-Planer fuer ein RAG-System der Hochschule Hannover.

Aufgabe:
- Entscheide, ob der vorliegende Kontext fuer eine belastbare Antwort ausreicht.
- Wenn der Kontext ausreicht, gib "final_answer" zurueck.
- Wenn der Kontext unvollstaendig ist, gib "need_more_context" mit einer gezielten Nachlade-Aktion zurueck.

Wichtige Regeln:
- Antworte NUR mit einem JSON-Objekt, ohne Markdown und ohne Erklaerungen.
- Verwende "need_more_context" nur, wenn zusaetzlicher Kontext aus derselben Wissensdatenbank realistisch helfen kann.
- Fuer Echtzeitfragen wie "heute", "aktuell", "jetzt" sollst du KEINE weitere Retrieval-Runde anfordern, wenn der Korpus wahrscheinlich nicht aktuell genug ist.
- Fuer Definitionsluecken oder Legenden: bevorzuge "neighbor_chunks" oder "full_section".
- Fuer Vergleichsfragen: bevorzuge "same_group_documents" oder "new_search".

JSON-Schema:
{
  "mode": "final_answer" | "need_more_context",
  "reason": "definition_missing" | "comparison_incomplete" | "procedure_incomplete" | "scope_incomplete" | "date_currentness_unclear" | "",
  "requested_action": "neighbor_chunks" | "full_section" | "same_group_documents" | "new_search" | "",
  "target_source_url": "",
  "target_chunk_index": null,
  "target_section_heading": "",
  "query_hint": ""
}
"""
FACULTY_ALIASES = {
    "fakultaet i": "Fakultät I",
    "fakultaet ii": "Fakultät II",
    "fakultaet iii": "Fakultät III",
    "fakultaet iv": "Fakultät IV",
    "fakultaet v": "Fakultät V",
    "fakultät i": "Fakultät I",
    "fakultät ii": "Fakultät II",
    "fakultät iii": "Fakultät III",
    "fakultät iv": "Fakultät IV",
    "fakultät v": "Fakultät V",
}
FACULTY_SOURCE_HINTS = (
    ("f1.hs-hannover.de", "Fakultät I"),
    ("f2.hs-hannover.de", "Fakultät II"),
    ("f3.hs-hannover.de", "Fakultät III"),
    ("f4.hs-hannover.de", "Fakultät IV"),
    ("f5.hs-hannover.de", "Fakultät V"),
    ("fakultaet_i", "Fakultät I"),
    ("fakultaet_ii", "Fakultät II"),
    ("fakultaet_iii", "Fakultät III"),
    ("fakultaet_iv", "Fakultät IV"),
    ("fakultaet_v", "Fakultät V"),
    ("faculty_1", "Fakultät I"),
    ("faculty_2", "Fakultät II"),
    ("faculty_3", "Fakultät III"),
    ("faculty_4", "Fakultät IV"),
    ("faculty_5", "Fakultät V"),
)


def build_planner_context(results: list[dict]) -> str:
    blocks = []
    for i, result in enumerate(results, 1):
        payload = result.get("payload") or {}
        lines = [
            f"[Kontext {i}] {payload.get('title', '(kein Titel)')}",
            f"URL: {payload.get('source_url', '')}",
            f"Chunk-Index: {payload.get('chunk_index', '')}",
            f"Abschnitt: {payload.get('section_heading', '')}",
            "",
            payload.get("text", "")[:1600],
        ]
        blocks.append("\n".join(lines).strip())
    return "\n\n---\n\n".join(blocks)


def build_followup_planner_messages(question: str, planner_context: str) -> list[dict]:
    return [
        {"role": "system", "content": FOLLOWUP_PLANNER_PROMPT},
        {
            "role": "user",
            "content": (
                f"Frage: {question}\n\n"
                f"Kontext:\n\n{planner_context}"
            ),
        },
    ]


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


def _fallback_plan_from_text(text: str) -> dict[str, Any] | None:
    lower = (text or "").strip().lower()
    if not lower:
        return None
    if "need_more_context" in lower:
        action = "new_search"
        if "neighbor_chunks" in lower:
            action = "neighbor_chunks"
        elif "full_section" in lower:
            action = "full_section"
        elif "same_group_documents" in lower:
            action = "same_group_documents"
        return {
            "mode": "need_more_context",
            "reason": "scope_incomplete",
            "requested_action": action,
            "target_source_url": "",
            "target_chunk_index": None,
            "target_section_heading": "",
            "query_hint": "",
        }
    if "final_answer" in lower:
        return {
            "mode": "final_answer",
            "reason": "",
            "requested_action": "",
            "target_source_url": "",
            "target_chunk_index": None,
            "target_section_heading": "",
            "query_hint": "",
        }
    return None


def _faculty_mentions(question: str) -> list[str]:
    lower = question.lower()
    found = []
    for alias, canonical in sorted(
        FACULTY_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, lower) and canonical not in found:
            found.append(canonical)
    return found


def _payload_faculties(payload: dict[str, Any]) -> set[str]:
    faculties: set[str] = set()
    explicit = payload.get("faculty", "")
    if explicit:
        faculties.add(explicit)

    searchable = " ".join(
        part for part in (
            payload.get("source_url", ""),
            payload.get("source_family", ""),
            payload.get("document_group", ""),
        )
        if part
    ).casefold()
    for hint, faculty in FACULTY_SOURCE_HINTS:
        if hint in searchable:
            faculties.add(faculty)
    return faculties


def _infer_backend_plan(question: str, results: list[dict]) -> dict[str, Any] | None:
    if not results:
        return None

    lower_question = question.lower()
    first_payload = results[0].get("payload") or {}

    if re.search(r"\b[A-Z]{1,4}\d{1,3}\b", question):
        return {
            "mode": "need_more_context",
            "reason": "definition_missing",
            "requested_action": "full_section",
            "target_source_url": first_payload.get("source_url", ""),
            "target_chunk_index": first_payload.get("chunk_index"),
            "target_section_heading": first_payload.get("section_heading", ""),
            "query_hint": "",
        }

    if any(marker in lower_question for marker in ("heute", "aktuell", "jetzt")):
        return {
            "mode": "final_answer",
            "reason": "date_currentness_unclear",
            "requested_action": "",
            "target_source_url": "",
            "target_chunk_index": None,
            "target_section_heading": "",
            "query_hint": "",
        }

    mentioned_faculties = _faculty_mentions(question)
    present_faculties = {
        faculty
        for result in results
        for faculty in _payload_faculties(result.get("payload") or {})
    }
    if (
        len(mentioned_faculties) == 1
        and any(marker in lower_question for marker in ("pruefungsanmeldung", "prüfung anmelden", "campusmanagement"))
        and mentioned_faculties[0] not in present_faculties
    ):
        faculty = mentioned_faculties[0]
        return {
            "mode": "need_more_context",
            "reason": "procedure_incomplete",
            "requested_action": "new_search",
            "target_source_url": "",
            "target_chunk_index": None,
            "target_section_heading": "",
            "query_hint": f"{faculty} Prüfungsanmeldung iCMS Campusmanagement Hochschule Hannover",
        }

    if len(mentioned_faculties) >= 2 or "unterschied" in lower_question or "vergleich" in lower_question:
        missing = [faculty for faculty in mentioned_faculties if faculty not in present_faculties]
        if missing:
            return {
                "mode": "need_more_context",
                "reason": "comparison_incomplete",
                "requested_action": "new_search",
                "target_source_url": "",
                "target_chunk_index": None,
                "target_section_heading": "",
                "query_hint": f"{question} {' '.join(missing)}",
            }

    return None


def request_followup_plan(
    openai_client,
    model: str,
    question: str,
    results: list[dict],
    *,
    temperature: float = 0.0,
) -> dict[str, Any] | None:
    planner_context = build_planner_context(results)
    messages = build_followup_planner_messages(question, planner_context)
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    msg = response.choices[0].message
    content = msg.content or ""
    parsed = _extract_json_object(content)
    if parsed is not None:
        return parsed
    return _fallback_plan_from_text(content)


def _scroll_source_chunks(
    client: QdrantClient,
    source_url: str,
    *,
    section_heading: str | None = None,
    limit: int = 32,
) -> list[dict]:
    must = [
        qmodels.FieldCondition(
            key="source_url",
            match=qmodels.MatchValue(value=source_url),
        ),
    ]
    if section_heading:
        must.append(
            qmodels.FieldCondition(
                key="section_heading",
                match=qmodels.MatchValue(value=section_heading),
            )
        )

    hits, _ = client.scroll(
        collection_name="hsh_knowledge",
        scroll_filter=qmodels.Filter(must=must),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points = [hit.payload for hit in hits if hit.payload]
    return sorted(points, key=lambda payload: payload.get("chunk_index", 0))


def _inject_chunks_into_result(result: dict, chunks: list[dict]) -> dict:
    if not chunks:
        return result
    payload = dict(result.get("payload") or {})
    text_parts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
    if text_parts:
        payload["text"] = "\n\n".join(text_parts)
        result = {**result, "payload": payload}
    return result


def _merge_unique_results(primary: list[dict], additional: list[dict], *, top_k: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, Any]] = set()

    for result in primary + additional:
        payload = result.get("payload") or {}
        key = (
            payload.get("source_url", ""),
            payload.get("chunk_index", result.get("id")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= top_k:
            break

    return merged


def expand_results_for_followup_plan(
    qdrant: QdrantClient,
    dense_embedder,
    sparse_embedder,
    question: str,
    results: list[dict],
    plan: dict[str, Any] | None,
    *,
    reranker=None,
    top_k: int,
) -> list[dict]:
    if not plan or plan.get("mode") != "need_more_context":
        return results

    action = (plan.get("requested_action") or "").strip()
    source_url = (plan.get("target_source_url") or "").strip()
    section_heading = (plan.get("target_section_heading") or "").strip()
    query_hint = (plan.get("query_hint") or "").strip() or question
    chunk_index = plan.get("target_chunk_index")

    expanded = list(results)

    if action in {"neighbor_chunks", "full_section"} and source_url:
        target_pos = None
        for idx, result in enumerate(expanded):
            payload = result.get("payload") or {}
            if payload.get("source_url") == source_url:
                target_pos = idx
                if not section_heading:
                    section_heading = payload.get("section_heading", "") or section_heading
                if chunk_index is None:
                    chunk_index = payload.get("chunk_index")
                break

        source_chunks = _scroll_source_chunks(
            qdrant,
            source_url,
            section_heading=section_heading if action == "full_section" else None,
        )
        if source_chunks and target_pos is not None:
            if action == "neighbor_chunks" and chunk_index is not None:
                source_chunks = [
                    chunk
                    for chunk in source_chunks
                    if abs(chunk.get("chunk_index", -999) - int(chunk_index)) <= 2
                ]
            expanded[target_pos] = _inject_chunks_into_result(expanded[target_pos], source_chunks)

    elif action in {"same_group_documents", "new_search"}:
        query_filter = None
        if action == "same_group_documents" and source_url:
            target_payload = next(
                (
                    result.get("payload") or {}
                    for result in expanded
                    if (result.get("payload") or {}).get("source_url") == source_url
                ),
                {},
            )
            faculty = target_payload.get("faculty", "")
            if faculty:
                query_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="faculty",
                            match=qmodels.MatchValue(value=faculty),
                        )
                    ]
                )

        followup_results = perform_hybrid_search(
            qdrant,
            dense_embedder,
            sparse_embedder,
            query_hint,
            top_k=max(2, min(4, top_k)),
            query_filter=query_filter,
            reranker=reranker,
        )
        expanded = _merge_unique_results(expanded, followup_results, top_k=top_k + 2)

    return expanded


def maybe_expand_results_with_followup(
    openai_client,
    model: str,
    qdrant: QdrantClient,
    dense_embedder,
    sparse_embedder,
    question: str,
    results: list[dict],
    *,
    reranker=None,
    top_k: int,
    use_model_planner: bool = False,
) -> tuple[list[dict], dict[str, Any] | None]:
    if not results:
        return results, None

    model_plan = None
    if use_model_planner:
        model_plan = request_followup_plan(openai_client, model, question, results)
    heuristic_plan = _infer_backend_plan(question, results)
    plan = model_plan
    if heuristic_plan and (
        plan is None or plan.get("mode") != "need_more_context"
    ):
        plan = heuristic_plan
    expanded = expand_results_for_followup_plan(
        qdrant,
        dense_embedder,
        sparse_embedder,
        question,
        results,
        plan,
        reranker=reranker,
        top_k=top_k,
    )
    return expanded, plan


__all__ = [
    "build_followup_planner_messages",
    "build_planner_context",
    "build_rag_context",
    "expand_results_for_followup_plan",
    "maybe_expand_results_with_followup",
    "request_followup_plan",
]
