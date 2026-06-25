"""Shared runtime helpers for the Streamlit chatbot and validation flow."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Callable

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

try:
    from .conversation_memory import MAX_MEMORY_TURNS, build_memory_prompt
    from .retrieval_planner import request_retrieval_plan, should_use_retrieval_planner
    from .hybrid_search import (
        DENSE_MODEL,
        SPARSE_MODEL,
        build_rag_context,
        create_reranker,
        perform_guided_hybrid_search,
    )
    from .query_assist import apply_planner_guidance, assess_query, build_plain_query_assessment
    from .rag_followup import maybe_expand_results_with_followup
except ImportError:  # pragma: no cover - script execution fallback
    from conversation_memory import MAX_MEMORY_TURNS, build_memory_prompt
    from retrieval_planner import request_retrieval_plan, should_use_retrieval_planner
    from hybrid_search import (
        DENSE_MODEL,
        SPARSE_MODEL,
        build_rag_context,
        create_reranker,
        perform_guided_hybrid_search,
    )
    from query_assist import apply_planner_guidance, assess_query, build_plain_query_assessment
    from rag_followup import maybe_expand_results_with_followup


TEMPERATURE = 0.0
DEFAULT_ROLE = "Studierender"
DEFAULT_FACULTY = "Alle Fakultäten"

ROLLEN = ["Studierender", "Mitarbeitender", "Lehrender", "Besucher"]
FAKULTAETEN = [
    "Alle Fakultäten",
    "Fakultät I",
    "Fakultät II",
    "Fakultät III",
    "Fakultät IV",
    "Fakultät V",
]

ROLLEN_HINWEIS = {
    "Studierender": "Du sprichst mit einem Studierenden. Verwende klare, zugängliche Sprache und erkläre Fachbegriffe.",
    "Mitarbeitender": "Du sprichst mit einem Mitarbeitenden. Antworte fachlich präzise.",
    "Lehrender": "Du sprichst mit einer lehrenden Person. Antworte auf akademischem Niveau.",
    "Besucher": "Du sprichst mit einer interessierten Person. Erkläre allgemeinverständlich.",
}
DEFAULT_NO_INFO_ANSWER = "Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor."
LOW_SUPPORT_STOPWORDS = {
    "an",
    "anmelden",
    "fuer",
    "für",
    "ich",
    "kann",
    "kannich",
    "kannman",
    "man",
    "mich",
    "stellen",
    "wie",
}
LOW_SUPPORT_ACTION_MARKERS = (
    "anmelden",
    "anmeldung",
    "antrag",
    "beantragen",
    "hochladen",
    "einreichen",
)
LOW_SUPPORT_TITLE_MARKERS = (
    "anmeldung",
    "antrag",
    "bewerbung",
    "faq",
    "formular",
    "portal",
)
SOURCE_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?quellen(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*$"
)
POST_SOURCE_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?"
    r"(?:hinweis|fakultäts-hinweis|fakultaets-hinweis|aktuelle informationen|weitere informationen)"
    r"(?:\*\*)?[ \t]*:(?:\*\*)?[ \t]*$"
)


@dataclass(frozen=True)
class RagPipelineConfig:
    """Schalter und Parameter fuer Chatbot- und Validierungslaeufe."""

    top_k: int = 6
    query_assist_enabled: bool = True
    retrieval_planner_enabled: bool = True
    rag_followup_enabled: bool = True

    def normalized(self, *, fallback_top_k: int | None = None) -> "RagPipelineConfig":
        top_k = self.top_k if self.top_k is not None else fallback_top_k
        try:
            top_k_int = int(top_k if top_k is not None else 6)
        except (TypeError, ValueError):
            top_k_int = 6
        return RagPipelineConfig(
            top_k=max(1, top_k_int),
            query_assist_enabled=bool(self.query_assist_enabled),
            retrieval_planner_enabled=bool(self.retrieval_planner_enabled),
            rag_followup_enabled=bool(self.rag_followup_enabled),
        )

    def to_dict(self) -> dict[str, bool | int]:
        return asdict(self.normalized())


def normalize_pipeline_config(
    config: RagPipelineConfig | dict | None = None,
    *,
    top_k: int | None = None,
) -> RagPipelineConfig:
    if isinstance(config, RagPipelineConfig):
        base = config
    elif isinstance(config, dict):
        raw_top_k = config.get("top_k", top_k if top_k is not None else 6)
        try:
            parsed_top_k = int(raw_top_k)
        except (TypeError, ValueError):
            parsed_top_k = top_k if top_k is not None else 6
        base = RagPipelineConfig(
            top_k=parsed_top_k,
            query_assist_enabled=bool(config.get("query_assist_enabled", True)),
            retrieval_planner_enabled=bool(config.get("retrieval_planner_enabled", True)),
            rag_followup_enabled=bool(config.get("rag_followup_enabled", True)),
        )
    else:
        base = RagPipelineConfig(top_k=top_k if top_k is not None else 6)
    if top_k is not None and not isinstance(config, dict):
        base = RagPipelineConfig(
            top_k=top_k,
            query_assist_enabled=base.query_assist_enabled,
            retrieval_planner_enabled=base.retrieval_planner_enabled,
            rag_followup_enabled=base.rag_followup_enabled,
        )
    return base.normalized(fallback_top_k=top_k)


def _collapse_consecutive_duplicate_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n{2,}", text.strip())
    collapsed: list[str] = []
    previous_normalized = ""
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip().casefold()
        if normalized and normalized == previous_normalized:
            continue
        collapsed.append(paragraph.strip())
        previous_normalized = normalized
    return "\n\n".join(collapsed)


def clean_answer_text(answer: str) -> str:
    """Trim model run-ons that repeat terminal source or hint sections."""
    text = (answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    source_matches = list(SOURCE_HEADING_RE.finditer(text))
    if source_matches:
        first_source = source_matches[0]
        trim_at: int | None = None
        repeated_source = source_matches[1].start() if len(source_matches) > 1 else None
        post_source_section = POST_SOURCE_SECTION_RE.search(text, first_source.end())

        # Some models continue after the first final source block; keep the first complete answer.
        if post_source_section and repeated_source is not None:
            trim_at = min(post_source_section.start(), repeated_source)
        elif post_source_section:
            trim_at = post_source_section.start()
        elif repeated_source is not None:
            trim_at = repeated_source

        if trim_at is not None:
            text = text[:trim_at].rstrip()

    return _collapse_consecutive_duplicate_paragraphs(text).strip()


def build_system_prompt(rolle: str, fakultaet: str) -> str:
    rollen_hinweis = ROLLEN_HINWEIS.get(rolle, "")
    if fakultaet != "Alle Fakultäten":
        fak_hinweis = (
            f"Der Nutzer gehört der {fakultaet} an. "
            "Nimm ausschließlich Informationen dieser Fakultät und sortiere alles andere aus. "
            "Nenne immer den Kontext der Information."
        )
    else:
        fak_hinweis = (
            "Der Nutzer hat keine Fakultät gewählt. "
            "Weise immer auf unterschiedliche Regelungen je Fakultät hin, wenn relevant."
        )
    return f"""\
Du bist der offizielle Assistent der Hochschule Hannover (HsH).

Nutzer-Profil:
- Rolle: {rolle} — {rollen_hinweis}
- {fak_hinweis}

Regeln:
1. Antworte AUSSCHLIESSLICH auf Basis des bereitgestellten Kontextes.
2. Erfinde keine Informationen und spekuliere nicht.
3. Wenn der Kontext klar eine naheliegende Schreibvariante, Groß-/Kleinschreibungsvariante
   oder einen offensichtlichen Buchstabendreher eines Nutzerbegriffs zeigt, nenne die im
   Kontext belegte Schreibweise ausdrücklich und beantworte die Frage auf dieser Basis.
   Formuliere dann transparent, z.B.:
   "In den Dokumenten wird der Begriff <X> verwendet; dazu steht ..."
4. Für zeitkritische Fragen wie "heute", "aktuell" oder "jetzt" darfst du keine
   älteren allgemeinen Angaben als Ersatz für eine konkrete aktuelle Antwort verwenden.
   Wenn der Kontext die aktuelle Information nicht belastbar belegt, antworte nur mit
   dem vorgeschriebenen Satz.
5. Bei Ablauf-, Portal- oder Antragsfragen nenne nur Schritte, Systeme, Formulare und
   Menüpunkte, die im Kontext ausdrücklich belegt sind. Erfinde keine Login-Schritte,
   keine Navigationspfade und keine Zwischenstationen.
6. Wenn der Kontext ein konkretes System oder Portal nennt (z.B. Bewerbungsportal,
   Campusmanagement, iCMS), verwende genau diese Bezeichnung und ersetze sie nicht
   durch ein anderes System.
7. Wenn der Kontext nur Beispiele, Einzelprojekte, Kursseiten, Erfahrungsberichte oder
   thematisch ähnliche Treffer enthält, aber keine direkte allgemeine Aussage zur Frage,
   antworte nur mit dem vorgeschriebenen Satz.
8. Falls die Antwort im Kontext nicht enthalten ist, antworte wörtlich:
   "{DEFAULT_NO_INFO_ANSWER}"
9. Wenn sich Quellenangaben widersprechen, weise explizit darauf hin.
10. Wenn eine Quelle ein älteres Datum trägt (erkennbar am "Stand:"-Feld im Kontext),
   empfehle dem Nutzer, die Angabe direkt auf der offiziellen Website zu verifizieren.
11. Schreibe klar, präzise und auf Deutsch.
12. Nenne am Ende jeder Antwort nur die Quellen, die die konkrete Antwort tatsächlich tragen,
   im Format:
   **Quellen:**
   - <Titel> — <URL> — Abschnitt: <Abschnitt> (Stand: <Datum>)
"""


def _resolve_effective_faculty(
    fakultaet: str,
    query: str,
    *,
    query_assist_enabled: bool,
) -> str | None:
    if fakultaet != DEFAULT_FACULTY:
        return fakultaet
    if not query_assist_enabled:
        return None

    assessment = assess_query(query)
    if len(assessment.detected_faculties) == 1:
        return assessment.detected_faculties[0]
    return None


def _normalize_lookup_text(text: str) -> str:
    return (
        " ".join((text or "").split())
        .casefold()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _extract_anchor_terms(question: str) -> list[str]:
    anchors: list[str] = []
    for token in re.findall(r"[A-Za-zÄÖÜäöüß][\wÄÖÜäöüß-]*", question or ""):
        normalized = _normalize_lookup_text(token)
        if len(normalized) < 4:
            continue
        if normalized in LOW_SUPPORT_STOPWORDS:
            continue
        if normalized in anchors:
            continue
        anchors.append(normalized)
    return anchors


def _has_direct_title_support(results: list[dict], anchor_term: str) -> bool:
    for result in results[:3]:
        payload = result.get("payload") or {}
        title_level = _normalize_lookup_text(
            " ".join(
                part
                for part in (
                    payload.get("title", ""),
                    payload.get("section_heading", ""),
                )
                if part
            )
        )
        if anchor_term in title_level and any(marker in title_level for marker in LOW_SUPPORT_TITLE_MARKERS):
            return True
    return False


def _should_guard_for_weak_action_support(
    question: str,
    results: list[dict],
    process_trace: dict | None,
) -> bool:
    lower_question = _normalize_lookup_text(question)
    if not any(marker in lower_question for marker in LOW_SUPPORT_ACTION_MARKERS):
        return False

    process_trace = process_trace or {}
    if process_trace.get("assessment_intents"):
        return False
    if process_trace.get("detected_faculties"):
        return False
    if process_trace.get("detected_codes"):
        return False
    if process_trace.get("detected_abbreviations"):
        return False

    anchor_terms = _extract_anchor_terms(question)
    if len(anchor_terms) != 1:
        return False

    return not _has_direct_title_support(results, anchor_terms[0])


def search(
    qdrant: QdrantClient,
    dense_embedder,
    sparse_embedder,
    query: str,
    fakultaet: str,
    *,
    top_k: int,
    reranker=None,
    selected_query: str | None = None,
    clarification: str = "",
    planner_guidance: dict | None = None,
    query_assist_enabled: bool = True,
) -> tuple[list[dict], dict]:
    """Hybrid search with an optional faculty filter for the Streamlit app."""
    faculty_filter = None
    effective_faculty = _resolve_effective_faculty(
        fakultaet,
        query,
        query_assist_enabled=query_assist_enabled,
    )
    if effective_faculty is not None:
        faculty_filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="faculty",
                    match=qmodels.MatchValue(value=effective_faculty),
                ),
                qmodels.FieldCondition(
                    key="faculty",
                    match=qmodels.MatchValue(value=""),
                ),
            ]
        )

    return perform_guided_hybrid_search(
        qdrant,
        dense_embedder,
        sparse_embedder,
        query,
        top_k=top_k,
        query_filter=faculty_filter,
        reranker=reranker,
        selected_query=selected_query,
        clarification=clarification,
        planner_guidance=planner_guidance,
        query_assist_enabled=query_assist_enabled,
    )


def _is_time_sensitive_question(question: str) -> bool:
    lower = question.casefold()
    return any(marker in lower for marker in ("heute", "aktuell", "jetzt"))


def _build_guard_answer(
    question: str,
    followup_plan: dict | None,
    *,
    results: list[dict] | None = None,
    process_trace: dict | None = None,
) -> str | None:
    if not followup_plan:
        if _should_guard_for_weak_action_support(question, list(results or []), process_trace):
            return DEFAULT_NO_INFO_ANSWER
        return None
    if followup_plan.get("reason") == "date_currentness_unclear" and _is_time_sensitive_question(question):
        return DEFAULT_NO_INFO_ANSWER
    if _should_guard_for_weak_action_support(question, list(results or []), process_trace):
        return DEFAULT_NO_INFO_ANSWER
    return None


def build_streamlit_llm_messages(
    *,
    rolle: str,
    fakultaet: str,
    question: str,
    context: str,
    results: list[dict],
    process_trace: dict | None = None,
    followup_plan: dict | None = None,
    conversation_memory: list[dict] | None = None,
) -> list[dict]:
    system_prompt = build_system_prompt(rolle, fakultaet)
    dates = [r["payload"].get("crawl_date", "") for r in results if r.get("payload")]
    valid_dates = [d for d in dates if d]
    if valid_dates:
        oldest = min(valid_dates)
        try:
            if (date.today() - date.fromisoformat(oldest)) > timedelta(days=180):
                system_prompt += (
                    "\nHinweis: Mindestens eine der verwendeten Quellen ist älter als 6 Monate. "
                    "Weise den Nutzer darauf hin, zeitkritische Informationen direkt auf "
                    "der offiziellen HsH-Website zu verifizieren."
                )
        except ValueError:
            pass

    process_trace = process_trace or {}
    term_corrections = process_trace.get("term_corrections", [])
    correction_hint = ""
    if term_corrections:
        lines = [
            f"- Nutzerbegriff `{item.get('asked', '')}` passt wahrscheinlich zu `{item.get('matched', '')}` aus dem Kontext."
            for item in term_corrections
            if item.get("asked") and item.get("matched")
        ]
        if lines:
            correction_hint = (
                "Terminologie-Hinweis aus der Retrieval-Phase:\n"
                + "\n".join(lines)
                + "\n\n"
            )

    faculty_hint = ""
    detected_faculties = process_trace.get("detected_faculties", [])
    if detected_faculties:
        faculty_hint = (
            "Fakultäts-Hinweis aus der Retrieval-Phase:\n"
            f"- Die Frage bezieht sich auf {', '.join(detected_faculties)}. "
            "Bevorzuge Quellen dieser Fakultät und nutze allgemeine Quellen nur ergänzend.\n\n"
        )

    currentness_hint = ""
    if (followup_plan or {}).get("reason") == "date_currentness_unclear":
        currentness_hint = (
            "Zeit-Hinweis aus der Retrieval-Phase:\n"
            "- Die Frage ist zeitkritisch oder aktuellkeitsabhängig. "
            "Nutze keine älteren allgemeinen Angaben als Ersatz für eine konkrete aktuelle Antwort. "
            "Wenn der Kontext die aktuelle Antwort nicht belegt, antworte nur mit dem vorgeschriebenen Satz.\n\n"
        )

    workflow_hint = ""
    if "workflow" in (process_trace or {}).get("assessment_intents", []):
        workflow_hint = (
            "Workflow-Hinweis aus der Retrieval-Phase:\n"
            "- Beschreibe nur die Schritte und Systeme, die im Kontext ausdruecklich genannt sind.\n"
            "- Wenn der Kontext ein konkretes Portal nennt, verwende genau dieses und ersetze es nicht durch ein anderes.\n\n"
        )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{build_memory_prompt(conversation_memory)}"
                f"{correction_hint}"
                f"{faculty_hint}"
                f"{currentness_hint}"
                f"{workflow_hint}"
                f"Kontext aus den offiziellen HsH-Dokumenten:\n\n"
                f"{context}\n\n---\n\nFrage: {question}"
            ),
        },
    ]


def request_answer_sync(
    openai_client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    temperature: float = TEMPERATURE,
) -> str:
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    msg = response.choices[0].message
    return clean_answer_text(msg.content or "")


def prepare_chat_turn(
    *,
    qdrant: QdrantClient,
    dense_embedder,
    sparse_embedder,
    reranker,
    openai_client: OpenAI,
    model: str,
    rolle: str,
    fakultaet: str,
    question: str,
    top_k: int | None = None,
    selected_query: str | None = None,
    clarification: str = "",
    conversation_memory: list[dict] | None = None,
    status_callback: Callable[[str], None] | None = None,
    pipeline_config: RagPipelineConfig | dict | None = None,
) -> dict:
    """Prepare one Streamlit chatbot turn up to the answer-generation step."""
    timings: dict[str, float] = {}
    conversation_memory = list(conversation_memory or [])
    runtime_config = normalize_pipeline_config(pipeline_config, top_k=top_k)

    def notify(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    if runtime_config.query_assist_enabled:
        notify("Frage analysiert und Suchpfade vorbereitet.")
    else:
        notify("Query Assist deaktiviert; Originalfrage wird direkt gesucht.")
    notify(
        f"Konversationsgedächtnis aktiv: {len(conversation_memory)}/{MAX_MEMORY_TURNS} Kurzfassungen werden mitgeführt."
    )

    planner_guidance = None
    assessment = (
        assess_query(question)
        if runtime_config.query_assist_enabled
        else build_plain_query_assessment(question)
    )
    if (
        runtime_config.retrieval_planner_enabled
        and not (selected_query or clarification)
        and should_use_retrieval_planner(assessment, question)
    ):
        planner_t0 = time.perf_counter()
        try:
            planner_guidance = request_retrieval_plan(
                openai_client,
                model,
                question,
                assessment,
                conversation_memory_prompt=build_memory_prompt(conversation_memory),
            )
            assessment = apply_planner_guidance(assessment, planner_guidance)
            notify("LLM-Planer hat zusätzliche HsH-Suchvarianten vorbereitet.")
        except Exception as exc:  # pragma: no cover - network/model dependent
            notify(f"LLM-Planer übersprungen: {exc}")
        timings["Planer-LLM"] = time.perf_counter() - planner_t0
    elif not runtime_config.retrieval_planner_enabled:
        notify("Retrieval-Planer deaktiviert.")

    retrieval_t0 = time.perf_counter()
    results, process_trace = search(
        qdrant,
        dense_embedder,
        sparse_embedder,
        question,
        fakultaet,
        top_k=runtime_config.top_k,
        reranker=reranker,
        selected_query=selected_query,
        clarification=clarification,
        planner_guidance=planner_guidance,
        query_assist_enabled=runtime_config.query_assist_enabled,
    )
    process_trace["pipeline_config"] = runtime_config.to_dict()
    process_trace["retrieval_planner_enabled"] = runtime_config.retrieval_planner_enabled
    process_trace["rag_followup_enabled"] = runtime_config.rag_followup_enabled
    timings["Retrieval"] = time.perf_counter() - retrieval_t0
    notify("Hybrid-Suche in Qdrant ausgeführt.")
    if process_trace.get("glossary_hits"):
        notify("Glossar- und Definitionsspuren aus dem Korpus wurden geprüft.")

    if not results:
        return {
            "answer": "",
            "guard_answer": None,
            "followup_plan": None,
            "llm_messages": [],
            "process_trace": process_trace,
            "results": [],
            "timings": timings,
            "assessment": assessment,
            "planner_guidance": planner_guidance,
            "pipeline_config": runtime_config.to_dict(),
            "no_results": True,
        }

    if runtime_config.rag_followup_enabled:
        followup_t0 = time.perf_counter()
        results, followup_plan = maybe_expand_results_with_followup(
            openai_client,
            model,
            qdrant,
            dense_embedder,
            sparse_embedder,
            question,
            results,
            reranker=reranker,
            top_k=runtime_config.top_k,
            use_model_planner=False,
        )
        timings["Follow-up-Heuristik"] = time.perf_counter() - followup_t0
        notify("Kontext auf Vollständigkeit geprüft.")
    else:
        followup_plan = None
        timings["Follow-up-Heuristik"] = 0.0
        notify("Follow-up-Retrieval deaktiviert.")

    context_t0 = time.perf_counter()
    context = build_rag_context(results)
    llm_messages = build_streamlit_llm_messages(
        rolle=rolle,
        fakultaet=fakultaet,
        question=question,
        context=context,
        results=results,
        process_trace=process_trace,
        followup_plan=followup_plan,
        conversation_memory=conversation_memory,
    )
    guard_answer = _build_guard_answer(
        question,
        followup_plan,
        results=results,
        process_trace=process_trace,
    )
    timings["Kontext+Prompt"] = time.perf_counter() - context_t0

    return {
        "answer": "",
        "assessment": assessment,
        "context": context,
        "followup_plan": followup_plan,
        "guard_answer": guard_answer,
        "llm_messages": llm_messages,
        "no_results": False,
        "planner_guidance": planner_guidance,
        "pipeline_config": runtime_config.to_dict(),
        "process_trace": process_trace,
        "results": results,
        "timings": timings,
    }


__all__ = [
    "DENSE_MODEL",
    "DEFAULT_FACULTY",
    "DEFAULT_ROLE",
    "FAKULTAETEN",
    "MAX_MEMORY_TURNS",
    "ROLLEN",
    "SPARSE_MODEL",
    "TEMPERATURE",
    "RagPipelineConfig",
    "build_streamlit_llm_messages",
    "build_system_prompt",
    "clean_answer_text",
    "create_reranker",
    "normalize_pipeline_config",
    "prepare_chat_turn",
    "request_answer_sync",
    "search",
]
