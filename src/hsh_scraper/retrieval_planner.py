"""LLM-gestuetzte Planung fuer retrieval-taugliche HsH-Suchanfragen."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from .query_assist import QueryAssessment
except ImportError:  # pragma: no cover - script execution fallback
    from query_assist import QueryAssessment


RETRIEVAL_PLANNER_PROMPT = """\
Du bist ein Retrieval-Planer fuer ein RAG-System der Hochschule Hannover (HsH).

Ziel:
- Formuliere eine Nutzerfrage so um, dass die Suche in den offiziellen HsH-Dokumenten bessere Treffer findet.
- Die Bedeutung der Frage darf NIEMALS veraendert oder verengt werden.
- Antworte NICHT auf die Frage selbst.

Arbeitsregeln:
- Bewahre die urspruengliche Bedeutung.
- Wenn die Frage zu breit ist und mehrere HsH-Prozesse meinen kann, setze needs_clarification=true.
- Nutze bevorzugt offizielle HsH-Begriffe, wenn sie naheliegen, z.B.:
  Bewerbung, Bewerbungsportal, Online-Antraege und -Bewerbungen, Beurlaubung,
  Pruefungsanmeldung, Pruefungsverwaltung, iCMS, Campusmanagement, Rueckmeldung,
  Semesterbeitrag, Studienberatung, Service Center, Namensaenderung,
  Aenderung der Anschrift, Unfallmeldung, Erstsemesterinformationen,
  Langzeitstudiengebuehren, Pruefungsform, Besondere Pruefungsordnung.
- Fuer Definitionen/Kuerzel darfst du Suchvarianten wie "Bedeutung", "steht fuer", "Legende" bilden.
- Fuer Kontaktfragen darfst du Typen wie Kontakt, Ansprechperson, Studienberatung,
  Service Center oder Pruefungsverwaltung bevorzugen, aber nur wenn sie semantisch zur Frage passen.
- Fuer Workflow-Fragen darfst du sinnvolle source_type_hints setzen, z.B. portal, faq, form, guide, regulation, contact.

Antwortformat:
- Antworte NUR mit einem JSON-Objekt, ohne Markdown.
- Schema:
{
  "mode": "use_original" | "guided_rewrite" | "clarify_first",
  "reason": "",
  "normalized_question": "",
  "needs_clarification": false,
  "clarification_prompt": "",
  "clarification_options": [],
  "canonical_hsh_terms": [],
  "source_type_hints": [],
  "query_variants": [],
  "must_not_assume": []
}

Zusatzregeln:
- Wenn mode="clarify_first" ist, muessen clarification_prompt und clarification_options sinnvoll gefuellt sein.
- query_variants muessen eng an der urspruenglichen Frage bleiben.
- Gib hoechstens 4 query_variants zurueck.
- Gib hoechstens 4 clarification_options zurueck.
- Wenn nichts verbessert werden kann, setze mode="use_original".
"""

SOURCE_TYPE_HINTS = {
    "portal",
    "faq",
    "form",
    "guide",
    "regulation",
    "contact",
}


def should_use_retrieval_planner(assessment: QueryAssessment, question: str) -> bool:
    """Aktiviert den LLM-Planer nur bei erkennbar schwierigen Retrieval-Faellen."""
    lower = (question or "").casefold()
    if "person_lookup" in assessment.intents:
        return False
    if assessment.clarification_needed:
        return True
    if assessment.intents:
        return True
    if assessment.reasons:
        return True
    if len(question.split()) <= 5:
        return True
    return any(marker in lower for marker in ("wie", "wo", "welche", "welcher", "welches", "bedeutet"))


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


def build_retrieval_planner_messages(
    question: str,
    assessment: QueryAssessment,
    *,
    conversation_memory_prompt: str = "",
) -> list[dict[str, str]]:
    hints = []
    if assessment.intents:
        hints.append(f"Erkannte Suchtypen: {', '.join(assessment.intents)}")
    if assessment.detected_abbreviations:
        hints.append(f"Erkannte Kuerzel: {', '.join(assessment.detected_abbreviations)}")
    if assessment.detected_faculties:
        hints.append(f"Erkannte Fakultät: {', '.join(assessment.detected_faculties)}")
    if assessment.reasons:
        hints.append("Analysehinweise: " + " | ".join(assessment.reasons))

    hint_block = "\n".join(f"- {hint}" for hint in hints)
    if hint_block:
        hint_block = f"Voranalyse:\n{hint_block}\n\n"

    return [
        {"role": "system", "content": RETRIEVAL_PLANNER_PROMPT},
        {
            "role": "user",
            "content": (
                f"{conversation_memory_prompt}"
                f"{hint_block}"
                f"Nutzerfrage: {question}"
            ),
        },
    ]


def _sanitize_string_list(values: Any, *, limit: int = 4) -> list[str]:
    if not isinstance(values, list):
        return []
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
        if len(cleaned) >= limit:
            break
    return cleaned


def normalize_planner_output(
    raw_plan: dict[str, Any] | None,
    *,
    question: str,
) -> dict[str, Any] | None:
    if not raw_plan:
        return None

    mode = str(raw_plan.get("mode") or "").strip()
    if mode not in {"use_original", "guided_rewrite", "clarify_first"}:
        return None

    normalized_question = " ".join(str(raw_plan.get("normalized_question") or question).split()).strip() or question
    clarification_options = _sanitize_string_list(raw_plan.get("clarification_options"), limit=4)
    query_variants = _sanitize_string_list(raw_plan.get("query_variants"), limit=4)
    canonical_hsh_terms = _sanitize_string_list(raw_plan.get("canonical_hsh_terms"), limit=4)
    must_not_assume = _sanitize_string_list(raw_plan.get("must_not_assume"), limit=4)
    source_type_hints = [
        item
        for item in _sanitize_string_list(raw_plan.get("source_type_hints"), limit=4)
        if item in SOURCE_TYPE_HINTS
    ]

    return {
        "mode": mode,
        "reason": " ".join(str(raw_plan.get("reason") or "").split()).strip(),
        "normalized_question": normalized_question,
        "needs_clarification": bool(raw_plan.get("needs_clarification")) or mode == "clarify_first",
        "clarification_prompt": " ".join(str(raw_plan.get("clarification_prompt") or "").split()).strip(),
        "clarification_options": clarification_options,
        "canonical_hsh_terms": canonical_hsh_terms,
        "source_type_hints": source_type_hints,
        "query_variants": query_variants,
        "must_not_assume": must_not_assume,
    }


def request_retrieval_plan(
    openai_client,
    model: str,
    question: str,
    assessment: QueryAssessment,
    *,
    conversation_memory_prompt: str = "",
    temperature: float = 0.0,
) -> dict[str, Any] | None:
    messages = build_retrieval_planner_messages(
        question,
        assessment,
        conversation_memory_prompt=conversation_memory_prompt,
    )
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = (response.choices[0].message.content or "").strip()
    return normalize_planner_output(
        _extract_json_object(content),
        question=question,
    )


__all__ = [
    "build_retrieval_planner_messages",
    "normalize_planner_output",
    "request_retrieval_plan",
    "should_use_retrieval_planner",
]
