"""Kompaktes Kurzgedaechtnis fuer die letzten Assistant-Turns."""

from __future__ import annotations

from typing import Any


MAX_MEMORY_TURNS = 5
QUESTION_LIMIT = 110
ANSWER_LIMIT = 220
FOCUS_LIMIT = 100


def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def _clip(text: str, limit: int) -> str:
    text = _normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _answer_summary(answer: str) -> str:
    cleaned = (answer or "").split("Quellen:", 1)[0]
    return _clip(cleaned, ANSWER_LIMIT)


def summarize_turn(
    question: str,
    answer: str,
    *,
    process_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Erzeugt eine kurze, prompt-taugliche Zusammenfassung eines Turns."""
    process_trace = process_trace or {}
    selected_query = _normalize_text(process_trace.get("selected_query", ""))
    original_query = _normalize_text(question)
    faculties = list(process_trace.get("detected_faculties", []))
    intents = list(process_trace.get("assessment_intents", []))

    summary = {
        "question": _clip(original_query, QUESTION_LIMIT),
        "answer": _answer_summary(answer),
        "faculties": faculties[:2],
        "intents": intents[:3],
    }
    if selected_query and selected_query.casefold() != original_query.casefold():
        summary["focus"] = _clip(selected_query, FOCUS_LIMIT)
    return summary


def append_summary(
    memory: list[dict[str, Any]] | None,
    summary: dict[str, Any],
    *,
    limit: int = MAX_MEMORY_TURNS,
) -> list[dict[str, Any]]:
    items = list(memory or [])
    items.append(summary)
    return items[-limit:]


def format_memory_entry(entry: dict[str, Any]) -> str:
    parts = [f"Frage: {entry.get('question', '')}"]
    focus = entry.get("focus", "")
    if focus:
        parts.append(f"Fokus: {focus}")
    faculties = entry.get("faculties") or []
    if faculties:
        parts.append(f"Fakultät: {', '.join(faculties)}")
    parts.append(f"Antwort: {entry.get('answer', '')}")
    return " | ".join(part for part in parts if part)


def build_memory_prompt(
    memory: list[dict[str, Any]] | None,
    *,
    limit: int = MAX_MEMORY_TURNS,
) -> str:
    entries = list(memory or [])[-limit:]
    if not entries:
        return ""

    lines = [
        "Kurzgedächtnis aus den letzten beantworteten Fragen (max. 5, komprimiert):",
        "Nutze diese Punkte nur als Gesprächskontext. Verbindlich bleiben ausschließlich die aktuellen offiziellen Kontextquellen dieser Runde.",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(f"{index}. {format_memory_entry(entry)}")
    return "\n".join(lines) + "\n\n"

