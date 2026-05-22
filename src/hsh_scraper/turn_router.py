"""Routing fuer Anschlussfragen, die keine neue Retrieval-Runde brauchen."""

from __future__ import annotations

import re
from typing import Any


URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

LINK_MARKERS = (
    "link",
    "links",
    "url",
    "oeffnet nicht",
    "öffnet nicht",
    "geht nicht",
    "funktioniert nicht",
    "nochmal",
    "erneut",
)

SOURCE_MARKERS = (
    "quelle",
    "quellen",
    "welche meinst du",
    "welchen meinst du",
    "welches meinst du",
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for url in urls:
        key = url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    return cleaned


def build_turn_state(
    *,
    question: str,
    answer: str,
    results: list[dict] | None = None,
    process_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Speichert die wichtigsten Referenzen des letzten Assistant-Turns."""
    results = list(results or [])
    process_trace = process_trace or {}

    answer_urls = _dedupe_urls(URL_PATTERN.findall(answer or ""))
    source_urls = _dedupe_urls(
        [
            (result.get("payload") or {}).get("source_url", "")
            for result in results
            if (result.get("payload") or {}).get("source_url")
        ]
    )

    return {
        "question": _normalize_text(question),
        "selected_query": _normalize_text(process_trace.get("selected_query", "")),
        "answer_urls": answer_urls,
        "source_urls": source_urls,
        "source_titles": [
            (result.get("payload") or {}).get("title", "")
            for result in results[:4]
        ],
    }


def _build_link_problem_answer(turn_state: dict[str, Any]) -> str:
    answer_urls = turn_state.get("answer_urls") or []
    source_urls = turn_state.get("source_urls") or []
    preferred_urls = answer_urls or source_urls

    if not preferred_urls:
        return (
            "Ich habe im letzten Schritt keinen klaren Direktlink gespeichert. "
            "Wenn du willst, suche ich dir den offiziellen Einstiegslink noch einmal gezielt heraus."
        )

    if len(preferred_urls) == 1:
        lines = [
            "Ich sende dir den Link noch einmal direkt:",
            preferred_urls[0],
        ]
        fallback = next((url for url in source_urls if url != preferred_urls[0]), "")
        if fallback:
            lines.extend([
                "",
                "Falls der Direktlink nicht öffnet, nutze bitte diese offizielle Quellseite als Einstieg:",
                fallback,
            ])
        lines.append("")
        lines.append("Wenn du willst, suche ich dir auch einen stabileren Einstiegslink oder den Navigationsweg dorthin.")
        return "\n".join(lines)

    lines = ["Ich habe im letzten Schritt mehrere Links verwendet. Meinst du einen dieser Links?"]
    lines.extend(f"- {url}" for url in preferred_urls[:3])
    lines.append("")
    lines.append("Wenn du magst, schreibe einfach „den ersten Link“ oder „den Bewerbungsportal-Link“.")
    return "\n".join(lines)


def _build_source_reference_answer(turn_state: dict[str, Any]) -> str:
    source_urls = turn_state.get("source_urls") or []
    source_titles = turn_state.get("source_titles") or []
    if not source_urls:
        return "Ich habe aus dem letzten Schritt gerade keine Quellenreferenz gespeichert."

    lines = ["Ich meinte diese Quelle(n) aus dem letzten Schritt:"]
    for title, url in zip(source_titles, source_urls):
        if title:
            lines.append(f"- {title} | {url}")
        else:
            lines.append(f"- {url}")
    return "\n".join(lines)


def route_turn(
    question: str,
    last_turn_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Leitet reine Anschlussfragen auf lokale Hilfsantworten um."""
    if not last_turn_state:
        return None

    lower = (question or "").casefold()

    if any(marker in lower for marker in LINK_MARKERS):
        return {
            "mode": "followup_link",
            "reason": "link_reference",
            "response": _build_link_problem_answer(last_turn_state),
        }

    if any(marker in lower for marker in SOURCE_MARKERS):
        return {
            "mode": "followup_source",
            "reason": "source_reference",
            "response": _build_source_reference_answer(last_turn_state),
        }

    return None


__all__ = [
    "build_turn_state",
    "route_turn",
]

