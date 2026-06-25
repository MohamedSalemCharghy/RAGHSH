"""
HsH-Chatbot — KI-gestützter Auskunfts-Assistent der Hochschule Hannover.

Kurzbeschreibung
----------------
Interaktiver Chatbot, der Fragen zur Hochschule Hannover ausschließlich auf
Basis offizieller Dokumente beantwortet. Kombiniert die lokale Hybrid-Suche
(hybrid_search.py) mit einem großen Sprachmodell der GWDG ChatAI API zu einer
vollständigen RAG-Pipeline (Retrieval-Augmented Generation).

Ausführliche Beschreibung
--------------------------
Herkömmliche Chatbots halluzinieren — sie erfinden Antworten, die plausibel
klingen, aber falsch sind. Das ist bei offiziellen Informationen (Prüfungs-
ordnungen, Bewerbungsfristen, Modulhandbücher) inakzeptabel. Dieser Chatbot
löst das Problem durch die RAG-Architektur: Das LLM erhält nicht nur die Frage,
sondern auch relevante Textstellen aus den offiziellen Dokumenten als Kontext.

Verarbeitungsablauf pro Nutzeranfrage:

  Nutzerfrage
      │
      ▼
  1. Hybrid-Suche in Qdrant
     perform_hybrid_search() aus hybrid_search.py sucht die 4 relevantesten
     Textabschnitte (RAG_TOP_K=4) aus der Wissensdatenbank. Die Suche kombiniert
     semantische Vektorsuche mit Volltextsuche (RRF-Fusion, URL-Deduplizierung).
      │
      ▼
  2. Kontext aufbauen
     build_rag_context() formatiert die Treffer als lesbaren Text mit Quellenangaben
     (Titel, URL, Abschnittsüberschrift, Fakultät).
      │
      ▼
  3. Prompt zusammenstellen
     build_rag_messages() erstellt die Messages-Liste für die Chat-API:
     - system: Persona und strikte Regeln (nur aus Kontext antworten, Quellen nennen)
     - user: Kontext-Block + eigentliche Frage
      │
      ▼
  4. LLM-Anfrage an GWDG ChatAI
     ask_llm() sendet den Prompt an das gewählte Modell (OpenAI-kompatibler Client,
     Base-URL: https://chat-ai.academiccloud.de/v1). Temperature=0.0 eliminiert
     kreative Freiheiten für maximale Faktenreue.
      │
      ▼
  5. Ausgabe
     - [Thinking]-Block (nur bei Reasoning-Modellen wie DeepSeek R1)
     - Antworttext des Modells
     - Liste der verwendeten Quellen mit RRF-Score

Besonderheiten:

  Debug-Modus (DEBUG_PROMPT = True):
    Zeigt den vollständigen an das LLM gesendeten Prompt vor jeder Antwort an.
    Unentbehrlich beim Testen und Optimieren des System-Prompts. Mit
    DEBUG_PROMPT = False deaktivieren für den Produktionsbetrieb.

  Dynamische Modellauswahl:
    Beim Start werden alle verfügbaren Modelle von der GWDG-API abgefragt und
    nummeriert angezeigt. Der Nutzer wählt per Eingabe das gewünschte Modell.
    So können verschiedene Modelle (GPT-4, Llama, DeepSeek R1 etc.) ohne
    Code-Änderung getestet werden.

  Reasoning-Unterstützung:
    Manche Modelle (z.B. DeepSeek R1) liefern neben der Antwort auch einen
    internen Denkprozess (reasoning_content). Dieser wird separat in einem
    [Thinking]-Block ausgegeben.

Konfiguration:
   GWDG_API_KEY    — API-Schlüssel (aus .env-Datei, nicht im Code!)
   GWDG_API_BASE   — Basis-URL der GWDG ChatAI API
   RAG_TOP_K       — Anzahl der Kontext-Dokumente pro Anfrage (Standard: 4)
   TEMPERATURE     — Kreativität des LLM (0.0 = deterministisch/faktengetreu)
   DEBUG_PROMPT    — Vollständigen Prompt ausgeben (True/False)

Einrichtung:
   cp .env.example .env
   # GWDG_API_KEY in .env eintragen
   python -m hsh_scraper.hsh_chatbot

Abhängigkeiten: openai, python-dotenv, fastembed, qdrant-client
   sowie hybrid_search.py (lokales Modul)
"""

import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from openai import OpenAI, APIConnectionError, AuthenticationError
from qdrant_client import QdrantClient

try:
    from .paths import CONFIG_DIR
    from .conversation_memory import (
        MAX_MEMORY_TURNS,
        append_summary,
        build_memory_prompt,
        format_memory_entry,
        summarize_turn,
    )
    from .retrieval_planner import (
        request_retrieval_plan,
        should_use_retrieval_planner,
    )
    from .hybrid_search import (
        build_rag_context,
        create_reranker,
        perform_guided_hybrid_search,
        print_process_trace,
    )
    from .query_assist import (
        QueryAssessment,
        apply_planner_guidance,
        assess_query,
        build_plain_query_assessment,
    )
    from .rag_followup import maybe_expand_results_with_followup
    from .turn_router import build_turn_state, route_turn
    from .web_app_runtime import RagPipelineConfig, clean_answer_text
except ImportError:
    from paths import CONFIG_DIR
    from conversation_memory import (
        MAX_MEMORY_TURNS,
        append_summary,
        build_memory_prompt,
        format_memory_entry,
        summarize_turn,
    )
    from retrieval_planner import (
        request_retrieval_plan,
        should_use_retrieval_planner,
    )
    from hybrid_search import (
        build_rag_context,
        create_reranker,
        perform_guided_hybrid_search,
        print_process_trace,
    )
    from query_assist import (
        QueryAssessment,
        apply_planner_guidance,
        assess_query,
        build_plain_query_assessment,
    )
    from rag_followup import maybe_expand_results_with_followup
    from turn_router import build_turn_state, route_turn
    from web_app_runtime import RagPipelineConfig, clean_answer_text

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# .env aus dem Konfigurationsverzeichnis laden (überschreibt keine gesetzten Env-Vars)
load_dotenv(CONFIG_DIR / ".env")

GWDG_API_KEY  = os.getenv("GWDG_API_KEY", "")
GWDG_API_BASE = os.getenv("GWDG_API_BASE", "https://chat-ai.academiccloud.de/v1")

QDRANT_URL      = "http://localhost:6333"
EMBED_MODEL     = "jinaai/jina-embeddings-v3"
SPARSE_MODEL    = "Qdrant/bm25"
RAG_TOP_K       = 4      # Treffer für den Kontext
TEMPERATURE     = 0.0    # Maximale Faktenreue
DEBUG_PROMPT    = True   # Zeigt den vollständigen LLM-Prompt vor jeder Anfrage
MODEL_CACHE_DIR = Path(
    os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
).expanduser()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_CACHE_DIR))

SYSTEM_PROMPT = """\
Du bist der offizielle Assistent der Hochschule Hannover (HsH).

Regeln:
- Antworte AUSSCHLIESSLICH auf Basis des bereitgestellten Kontextes.
- Erfinde keine Informationen und spekuliere nicht.
- Wenn der Kontext klar eine naheliegende Schreibvariante, Groß-/Kleinschreibungsvariante
  oder einen offensichtlichen Buchstabendreher eines Nutzerbegriffs zeigt, nenne die im
  Kontext belegte Schreibweise ausdrücklich und beantworte die Frage auf dieser Basis.
  Formuliere dann transparent, z.B.:
  "In den Dokumenten wird der Begriff <X> verwendet; dazu steht ..."
- Für zeitkritische Fragen wie "heute", "aktuell" oder "jetzt" darfst du keine
  älteren allgemeinen Angaben als Ersatz für eine konkrete aktuelle Antwort verwenden.
  Wenn der Kontext die aktuelle Information nicht belastbar belegt, antworte nur mit
  dem vorgeschriebenen Satz.
- Falls die Antwort im Kontext nicht enthalten ist, antworte wörtlich:
  "Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor."
- Schreibe klar, präzise und auf Deutsch. Die verschiedenen Fakultäten der 
Hochschule Hannover haben unterschiedliche Regelungen, daher ist es wichtig, 
die Antwort so spezifisch wie möglich zu formulieren. Wenn die Fakultät nicht eindeutig 
aus dem Kontext hervorgeht und dies wichtig wäre, gib dies in der Antwort an.

- Nenne am Ende jeder Antwort die verwendeten Quellen im Format:
    Quellen:
    - <Titel> | <URL> | <Abschnitt>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interaktiver HsH-RAG-Chatbot mit konfigurierbarer Pipeline."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAG_TOP_K,
        help=f"Anzahl finaler Qdrant-Treffer pro Frage (Standard: {RAG_TOP_K}).",
    )
    parser.add_argument(
        "--no-query-assist",
        action="store_true",
        help="Query Assist deaktivieren: keine lokale Frageanalyse, keine Assist-Suchvarianten.",
    )
    parser.add_argument(
        "--no-retrieval-planner",
        action="store_true",
        help="LLM-Retrieval-Planer deaktivieren.",
    )
    parser.add_argument(
        "--no-rag-followup",
        action="store_true",
        help="Follow-up-Retrieval nach der ersten Suche deaktivieren.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k muss mindestens 1 sein.")
    return args


# ---------------------------------------------------------------------------
# GWDG / OpenAI API
# ---------------------------------------------------------------------------


def build_client() -> OpenAI:
    """Erstellt und validiert den OpenAI-kompatiblen GWDG-Client."""
    if not GWDG_API_KEY:
        print("Fehler: GWDG_API_KEY fehlt.")
        print("  → Kopiere .env.example nach .env und trage deinen Key ein.")
        sys.exit(1)
    return OpenAI(api_key=GWDG_API_KEY, base_url=GWDG_API_BASE)


def get_available_models(client: OpenAI) -> list[str]:
    """Fragt die Modellliste vom GWDG-Server ab."""
    try:
        return [m.id for m in client.models.list().data]
    except AuthenticationError:
        print("Fehler: API-Key ungültig oder abgelaufen.")
        sys.exit(1)
    except APIConnectionError as exc:
        print(f"Fehler: GWDG-API nicht erreichbar — {exc}")
        sys.exit(1)


def select_model(client: OpenAI) -> str:
    """Zeigt verfügbare Modelle und lässt den Nutzer auswählen."""
    print("Rufe Modellliste von der GWDG-API ab…\n")
    models = get_available_models(client)

    if not models:
        print("Keine Modelle gefunden.")
        sys.exit(1)

    for i, model_id in enumerate(models, 1):
        print(f"  [{i:2}] {model_id}")

    print()
    try:
        choice = int(input(f"Modell wählen (1–{len(models)}): "))
        selected = models[choice - 1]
    except (ValueError, IndexError):
        print("Ungültige Eingabe — erstes Modell wird verwendet.")
        selected = models[0]

    print(f"\n  → Aktiv: {selected}\n")
    return selected


def ask_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
) -> tuple[str, str]:
    """Sendet eine Anfrage an das GWDG-Modell.

    Gibt (antwort, denkprozess) zurück.
    Bei Modellen ohne reasoning_content ist denkprozess ein leerer String.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
    )
    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", "") or ""
    answer    = clean_answer_text(msg.content or "")
    return answer, reasoning


# ---------------------------------------------------------------------------
# RAG-Pipeline
# ---------------------------------------------------------------------------


def _is_time_sensitive_question(question: str) -> bool:
    lower = question.casefold()
    return any(marker in lower for marker in ("heute", "aktuell", "jetzt"))


def _build_guard_answer(question: str, followup_plan: dict | None) -> str | None:
    if not followup_plan:
        return None
    if followup_plan.get("reason") != "date_currentness_unclear":
        return None
    if not _is_time_sensitive_question(question):
        return None
    return "Dazu liegen mir keine Informationen aus den offiziellen Dokumenten der HsH vor."


def build_rag_messages(
    question: str,
    context: str,
    process_trace: dict | None = None,
    followup_plan: dict | None = None,
    conversation_memory: list[dict] | None = None,
) -> list[dict]:
    """Erstellt die Messages-Liste für die Chat-Completion-API."""
    correction_hint = ""
    corrections = (process_trace or {}).get("term_corrections", [])
    if corrections:
        lines = [
            f"- Nutzerbegriff `{item.get('asked', '')}` passt wahrscheinlich zu `{item.get('matched', '')}` aus dem Kontext."
            for item in corrections
            if item.get("asked") and item.get("matched")
        ]
        if lines:
            correction_hint = (
                "Terminologie-Hinweis aus der Retrieval-Phase:\n"
                + "\n".join(lines)
                + "\n\n"
            )
    faculty_hint = ""
    detected_faculties = (process_trace or {}).get("detected_faculties", [])
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
    memory_block = build_memory_prompt(conversation_memory)
    user_content = (
        f"{memory_block}"
        f"{correction_hint}"
        f"{faculty_hint}"
        f"{currentness_hint}"
        f"Kontext aus den offiziellen HsH-Dokumenten:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Frage: {question}"
    )
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


def format_sources(results: list[dict]) -> str:
    """Kompakte Quellenzeilen aus den Qdrant-Treffern."""
    lines = []
    for r in results:
        p       = r["payload"]
        url     = p.get("source_url", "—")
        title   = p.get("title", "—")
        heading = p.get("section_heading", "")
        line    = f"  • {title} | {url}"
        if heading:
            line += f" | {heading}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ausgabe-Helfer
# ---------------------------------------------------------------------------

SEP = "─" * 70


def print_thinking(reasoning: str) -> None:
    if not reasoning:
        return
    print(f"\n[Thinking]\n{SEP}")
    print(reasoning.strip())
    print(SEP)


def print_answer(answer: str, model: str) -> None:
    print(f"\n[{model}]\n{SEP}")
    print(answer.strip())
    print(SEP)


def print_followup_plan(plan: dict | None) -> None:
    if not plan or plan.get("mode") != "need_more_context":
        return
    action = plan.get("requested_action", "")
    reason = plan.get("reason", "")
    target = plan.get("target_source_url", "")
    query_hint = plan.get("query_hint", "")
    print(f"\n[Follow-up Retrieval]\n{SEP}")
    print(f"Aktion : {action}")
    if reason:
        print(f"Grund  : {reason}")
    if target:
        print(f"Quelle : {target}")
    if query_hint:
        print(f"Suche  : {query_hint}")
    print(SEP)


def print_llm_input(messages: list[dict]) -> None:
    """Gibt den vollständigen Prompt übersichtlich auf der Konsole aus."""
    if not DEBUG_PROMPT:
        return
    DBG = "░" * 70
    print(f"\n{DBG}")
    print("  DEBUG — An das LLM gesendete Messages")
    print(DBG)
    for msg in messages:
        role    = msg["role"].upper()
        content = msg["content"]
        print(f"\n  ┌─ [{role}]")
        for line in content.splitlines():
            print(f"  │  {line}")
        print(f"  └{'─' * 68}")
    print(f"{DBG}\n")


def print_retrieved_sources(results: list[dict]) -> None:
    print(f"\n[Verwendete Quellen  (RRF-Score)]")
    for r in results:
        p    = r["payload"]
        url  = p.get("source_url", "—")
        head = p.get("section_heading", "")
        print(f"  {r['score']:.5f}  {url}" + (f"  [{head}]" if head else ""))


def print_conversation_memory(memory: list[dict]) -> None:
    print(f"\n[Konversationsgedächtnis — letzte {MAX_MEMORY_TURNS} Kurzfassungen]")
    if not memory:
        print("  Noch keine gespeicherten Kurzfassungen.")
        return
    for idx, entry in enumerate(memory, start=1):
        print(f"  {idx}. {format_memory_entry(entry)}")


def print_timing_trace(timings: dict[str, float]) -> None:
    print("\n[Leistungsdaten]")
    for label, seconds in timings.items():
        print(f"  {label:<18} {seconds:.2f}s")


def resolve_query_with_user(assessment: QueryAssessment) -> tuple[str | None, str]:
    """Bietet bei unklaren Fragen lokale Umformulierungen und Ergänzungen an."""
    if not assessment.needs_user_choice:
        return None, ""

    print("\n[Query Assist]")
    if assessment.clarification_prompt:
        print(f"  {assessment.clarification_prompt}")
    for reason in assessment.reasons:
        print(f"  - {reason}")

    options = assessment.clarification_options or assessment.suggestions
    if options:
        print("  Präzisierungen:")
        for idx, suggestion in enumerate(options, start=1):
            print(f"    [{idx}] {suggestion}")

    print("  Hinweis: Im Hintergrund werden trotzdem zusätzliche Suchvarianten erzeugt.")
    print("  Enter = Originalfrage verwenden")
    choice = input("  Auswahl> ").strip()
    selected_query = None
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(options):
            selected_query = options[index]

    clarification = input(
        f"  Ergänzung (optional) — {assessment.clarification_hint}\n"
        "  Zusatz> "
    ).strip()
    return selected_query, clarification


# ---------------------------------------------------------------------------
# Chat-Schleife
# ---------------------------------------------------------------------------


def chat_loop(
    gwdg_client: OpenAI,
    model: str,
    qdrant_client: QdrantClient,
    embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    reranker=None,
    pipeline_config: RagPipelineConfig | None = None,
) -> None:
    """Interaktive RAG-Chat-Schleife."""
    pipeline_config = pipeline_config or RagPipelineConfig(top_k=RAG_TOP_K)
    print(f"HsH-Chatbot bereit.  Strg+C oder 'exit' zum Beenden.")
    print(
        f"Konversationsgedächtnis aktiv: Die letzten {MAX_MEMORY_TURNS} beantworteten Fragen "
        "werden als Kurzfassungen an das Modell weitergegeben.\n"
    )
    print(
        "Pipeline: "
        f"Top-K={pipeline_config.top_k}, "
        f"Query Assist={'aktiv' if pipeline_config.query_assist_enabled else 'inaktiv'}, "
        f"Retrieval Planner={'aktiv' if pipeline_config.retrieval_planner_enabled else 'inaktiv'}, "
        f"RAG Follow-up={'aktiv' if pipeline_config.rag_followup_enabled else 'inaktiv'}\n"
    )

    conversation_memory: list[dict] = []
    last_turn_state: dict | None = None

    while True:
        # ── Eingabe ───────────────────────────────────────────────────────
        try:
            question = input("Frage> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAuf Wiedersehen.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Auf Wiedersehen.")
            break

        routed_turn = route_turn(question, last_turn_state)
        if routed_turn is not None:
            print("\n[Follow-up Router]")
            print(f"  Modus : {routed_turn.get('mode', '')}")
            print(f"  Grund : {routed_turn.get('reason', '')}")
            answer = routed_turn.get("response", "")
            print_answer(answer, model)
            conversation_memory = append_summary(
                conversation_memory,
                summarize_turn(question, answer, process_trace={}),
            )
            last_turn_state = build_turn_state(
                question=question,
                answer=answer,
                results=[],
                process_trace={},
            )
            print(
                f"\n[Konversationsgedächtnis] Kurzfassung gespeichert "
                f"({len(conversation_memory)}/{MAX_MEMORY_TURNS}).\n"
            )
            continue

        # ── RAG: Suche + Kontext ──────────────────────────────────────────
        print_conversation_memory(conversation_memory)
        timings: dict[str, float] = {}
        assessment = (
            assess_query(question)
            if pipeline_config.query_assist_enabled
            else build_plain_query_assessment(question)
        )
        planner_guidance = None
        if (
            pipeline_config.retrieval_planner_enabled
            and should_use_retrieval_planner(assessment, question)
        ):
            planner_t0 = time.perf_counter()
            try:
                planner_guidance = request_retrieval_plan(
                    gwdg_client,
                    model,
                    question,
                    assessment,
                    conversation_memory_prompt=build_memory_prompt(conversation_memory),
                )
            except Exception as exc:
                planner_guidance = None
                print(f"\n[Retrieval-Planer]\n  Hinweis: Planner nicht verfügbar — {exc}")
            timings["planer_llm"] = time.perf_counter() - planner_t0
            assessment = apply_planner_guidance(assessment, planner_guidance)
        selected_query, clarification = (
            resolve_query_with_user(assessment)
            if pipeline_config.query_assist_enabled
            else (None, "")
        )

        retrieval_t0 = time.perf_counter()
        results, process_trace = perform_guided_hybrid_search(
            qdrant_client,
            embedder,
            sparse_embedder,
            question,
            top_k=pipeline_config.top_k,
            reranker=reranker,
            selected_query=selected_query,
            clarification=clarification,
            planner_guidance=planner_guidance,
            query_assist_enabled=pipeline_config.query_assist_enabled,
        )
        process_trace["pipeline_config"] = pipeline_config.to_dict()
        process_trace["retrieval_planner_enabled"] = pipeline_config.retrieval_planner_enabled
        process_trace["rag_followup_enabled"] = pipeline_config.rag_followup_enabled
        timings["retrieval"] = time.perf_counter() - retrieval_t0
        print_process_trace(process_trace)

        if not results:
            print("\nKeine passenden Dokumente in der Wissensdatenbank gefunden.\n")
            continue

        if pipeline_config.rag_followup_enabled:
            followup_t0 = time.perf_counter()
            results, followup_plan = maybe_expand_results_with_followup(
                gwdg_client,
                model,
                qdrant_client,
                embedder,
                sparse_embedder,
                question,
                results,
                reranker=reranker,
                top_k=pipeline_config.top_k,
                use_model_planner=False,
            )
            timings["followup"] = time.perf_counter() - followup_t0
        else:
            followup_plan = None
            timings["followup"] = 0.0
        print_followup_plan(followup_plan)

        context_t0 = time.perf_counter()
        context  = build_rag_context(results)
        guard_answer = _build_guard_answer(question, followup_plan)
        messages = build_rag_messages(
            question,
            context,
            process_trace=process_trace,
            followup_plan=followup_plan,
            conversation_memory=conversation_memory,
        )
        timings["context_prompt"] = time.perf_counter() - context_t0

        # ── Debug-Ausgabe des Prompts ─────────────────────────────────────
        print_llm_input(messages)

        # ── LLM-Anfrage ───────────────────────────────────────────────────
        if guard_answer is not None:
            answer, reasoning = guard_answer, ""
            timings["answer_llm"] = 0.0
        else:
            try:
                llm_t0 = time.perf_counter()
                answer, reasoning = ask_llm(gwdg_client, model, messages)
                timings["answer_llm"] = time.perf_counter() - llm_t0
            except Exception as exc:
                print(f"\nFehler bei der API-Anfrage: {exc}\n")
                continue

        # ── Ausgabe ───────────────────────────────────────────────────────
        print_thinking(reasoning)
        print_answer(answer, model)
        print_retrieved_sources(results)
        conversation_memory = append_summary(
            conversation_memory,
            summarize_turn(
                question,
                answer,
                process_trace=process_trace,
            ),
        )
        last_turn_state = build_turn_state(
            question=question,
            answer=answer,
            results=results,
            process_trace=process_trace,
        )
        print(
            f"\n[Konversationsgedächtnis] Kurzfassung gespeichert "
            f"({len(conversation_memory)}/{MAX_MEMORY_TURNS})."
        )
        timings["gesamt"] = sum(timings.values())
        print_timing_trace(timings)
        print()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pipeline_config = RagPipelineConfig(
        top_k=args.top_k,
        query_assist_enabled=not args.no_query_assist,
        retrieval_planner_enabled=not args.no_retrieval_planner,
        rag_followup_enabled=not args.no_rag_followup,
    )

    print("=" * 70)
    print("  HsH-Chatbot  —  RAG + GWDG ChatAI")
    print("=" * 70 + "\n")

    # ── GWDG-Client + Modellauswahl ───────────────────────────────────────
    gwdg_client = build_client()
    model       = select_model(gwdg_client)

    # ── Qdrant verbinden ──────────────────────────────────────────────────
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
        qdrant_client.get_collections()
    except Exception as exc:
        print(f"Fehler: Qdrant nicht erreichbar — {exc}")
        print("  → docker compose up -d")
        sys.exit(1)

    # ── Embedding-Modell laden ────────────────────────────────────────────
    print(f"Lade Embedding-Modell '{EMBED_MODEL}'…")
    embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(MODEL_CACHE_DIR))

    print(f"Lade Sparse-Modell '{SPARSE_MODEL}'…")
    sparse_embedder = SparseTextEmbedding(
        model_name=SPARSE_MODEL,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    reranker = None
    try:
        print("Lade Reranker…")
        reranker = create_reranker(cache_dir=str(MODEL_CACHE_DIR))
    except Exception as exc:
        print(f"Reranker nicht verfügbar — weiter ohne Reranking: {exc}")

    print("Modelle geladen.\n")

    # ── Chat starten ──────────────────────────────────────────────────────
    chat_loop(
        gwdg_client,
        model,
        qdrant_client,
        embedder,
        sparse_embedder,
        reranker=reranker,
        pipeline_config=pipeline_config,
    )


if __name__ == "__main__":
    main()
