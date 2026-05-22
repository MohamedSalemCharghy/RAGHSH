"""
HsH-Web-App — Streamlit-Oberfläche für den RAG-Chatbot der Hochschule Hannover.

Aufruf:
    streamlit run src/hsh_scraper/hsh_web_app.py

Abhängigkeiten:
    pip install streamlit openai python-dotenv fastembed qdrant-client
"""

import os
import time
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from fastembed import SparseTextEmbedding, TextEmbedding
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
    from .api_rate_limiter import SlidingWindowRateLimiter, build_rate_limited_http_client
    from .retrieval_planner import request_retrieval_plan, should_use_retrieval_planner
    from .query_assist import apply_planner_guidance, assess_query
    from .turn_router import build_turn_state, route_turn
    from .web_app_runtime import (
        DENSE_MODEL,
        FAKULTAETEN,
        ROLLEN,
        RagPipelineConfig,
        SPARSE_MODEL,
        create_reranker,
        prepare_chat_turn,
        request_answer_sync,
    )
    from .evals.validation_system import (
        MAX_EVAL_API_REQUESTS_PER_MINUTE,
        VALIDATION_API_TIMEOUT_SECONDS,
        render_validation_page,
        reset_validation_state,
    )
except ImportError:
    from paths import CONFIG_DIR
    from conversation_memory import (
        MAX_MEMORY_TURNS,
        append_summary,
        build_memory_prompt,
        format_memory_entry,
        summarize_turn,
    )
    from api_rate_limiter import SlidingWindowRateLimiter, build_rate_limited_http_client
    from retrieval_planner import request_retrieval_plan, should_use_retrieval_planner
    from query_assist import apply_planner_guidance, assess_query
    from turn_router import build_turn_state, route_turn
    from web_app_runtime import (
        DENSE_MODEL,
        FAKULTAETEN,
        ROLLEN,
        RagPipelineConfig,
        SPARSE_MODEL,
        create_reranker,
        prepare_chat_turn,
        request_answer_sync,
    )
    from evals.validation_system import (
        MAX_EVAL_API_REQUESTS_PER_MINUTE,
        VALIDATION_API_TIMEOUT_SECONDS,
        render_validation_page,
        reset_validation_state,
    )

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

load_dotenv(CONFIG_DIR / ".env")

GWDG_API_KEY  = os.getenv("GWDG_API_KEY", "")
GWDG_API_BASE = os.getenv("GWDG_API_BASE", "https://chat-ai.academiccloud.de/v1")
QDRANT_URL    = "http://localhost:6333"
COLLECTION    = "hsh_knowledge"
RAG_TOP_K     = 6
TEMPERATURE   = 0.0
MODEL_CACHE_DIR = Path(
    os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
).expanduser()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_CACHE_DIR))

# ---------------------------------------------------------------------------
# Seitenconfig — muss vor allen anderen st.*-Aufrufen stehen
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HsH-Assistent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Gecachte Ressourcen — analog zu hsh_chatbot.py, nur einmal geladen
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Verbinde mit Qdrant …")
def get_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    client.get_collections()
    return client


@st.cache_resource(show_spinner="Lade Dense-Embedding-Modell …")
def get_dense_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=DENSE_MODEL, cache_dir=str(MODEL_CACHE_DIR))


@st.cache_resource(show_spinner="Lade Sparse-Embedding-Modell (BM25) …")
def get_sparse_embedder() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=str(MODEL_CACHE_DIR))


@st.cache_resource(show_spinner="Lade Reranker-Modell …")
def get_reranker():
    try:
        return create_reranker(cache_dir=str(MODEL_CACHE_DIR))
    except Exception:
        return None


@st.cache_resource(show_spinner="Verbinde mit GWDG-API …")
def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=GWDG_API_KEY,
        base_url=GWDG_API_BASE,
        timeout=30,
    )


@st.cache_resource(show_spinner="Verbinde mit GWDG-API für Alle-Fälle-Validierung …")
def get_rate_limited_openai_client(max_requests_per_minute: int) -> OpenAI:
    return OpenAI(
        api_key=GWDG_API_KEY,
        base_url=GWDG_API_BASE,
        timeout=VALIDATION_API_TIMEOUT_SECONDS,
        http_client=build_rate_limited_http_client(
            timeout=VALIDATION_API_TIMEOUT_SECONDS,
            rate_limiter=SlidingWindowRateLimiter(max_requests=max_requests_per_minute),
        ),
    )


@st.cache_resource(show_spinner="Lade Modellliste …")
def get_model_list() -> list[str]:
    """Ruft die Modellliste ab — identisch mit hsh_chatbot.get_available_models()."""
    client = get_openai_client()
    try:
        return sorted(m.id for m in client.models.list().data)
    except Exception:
        return ["meta-llama-3.1-8b-instruct", "gpt-4o-mini", "gpt-4o"]


# ---------------------------------------------------------------------------
# Streaming — analog zu ask_llm() in hsh_chatbot.py, aber mit write_stream
# ---------------------------------------------------------------------------


def stream_response(
    openai_client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    answer_placeholder=None,
) -> str:
    """Streamt ausschließlich die sichtbare Antwort in die UI.

    Reasoning-Inhalte werden weiterhin aus dem Stream herausgefiltert, aber
    nicht mehr als separates Panel angezeigt. So bleibt die Oberfläche auf die
    Antwort und die strukturierten Zusatzbereiche fokussiert.
    """
    if answer_placeholder is None:
        answer_placeholder = st.empty()

    reasoning_parts: list[str] = []
    answer_parts:    list[str] = []

    # Zustandsautomat für <think>-Tag-Parsing
    in_think = False   # befinden wir uns gerade innerhalb von <think>…</think>?
    buffer   = ""      # sammelt angebrochene Tag-Grenzen über Chunk-Grenzen

    stream = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue

        # ── Variante 1: separates reasoning_content-Feld ─────────────────
        r = getattr(delta, "reasoning_content", None)
        if r:
            reasoning_parts.append(r)
            continue

        # ── Variante 2: <think>…</think> inline im content ────────────────
        token = delta.content or ""
        if not token:
            continue

        buffer += token

        # Verarbeite den Buffer zeichenweise soweit möglich
        while buffer:
            if in_think:
                end = buffer.find("</think>")
                if end == -1:
                    # Kompletter Rest ist noch Reasoning; letzten 8 Zeichen puffern
                    # (könnten Anfang von </think> sein)
                    safe = max(0, len(buffer) - 8)
                    reasoning_parts.append(buffer[:safe])
                    buffer = buffer[safe:]
                    break
                else:
                    reasoning_parts.append(buffer[:end])
                    buffer  = buffer[end + len("</think>"):]
                    in_think = False
            else:
                start = buffer.find("<think>")
                if start == -1:
                    # Kein Tag im Buffer — alles ist Antworttext; letzten 7 Zeichen puffern
                    safe = max(0, len(buffer) - 7)
                    answer_parts.append(buffer[:safe])
                    buffer = buffer[safe:]
                    break
                else:
                    # Text vor dem Tag
                    if start > 0:
                        answer_parts.append(buffer[:start])
                    buffer  = buffer[start + len("<think>"):]
                    in_think = True

        # UI aktualisieren
        if answer_parts:
            answer_placeholder.markdown("".join(answer_parts))

    # Restlichen Buffer leeren
    if buffer:
        if in_think:
            reasoning_parts.append(buffer)
        else:
            answer_parts.append(buffer)

    # Letztes UI-Update
    if answer_parts:
        answer_placeholder.markdown("".join(answer_parts))

    return "".join(answer_parts)


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --raghsh-surface: rgba(17, 24, 39, 0.78);
            --raghsh-surface-strong: rgba(15, 23, 42, 0.9);
            --raghsh-surface-border: rgba(56, 189, 248, 0.16);
            --raghsh-surface-border-strong: rgba(45, 212, 191, 0.28);
            --raghsh-surface-text: rgba(241, 245, 249, 0.96);
            --raghsh-surface-muted: rgba(148, 163, 184, 0.92);
            --raghsh-accent: #14b8a6;
            --raghsh-accent-strong: #0f766e;
        }
        .raghsh-step-shell {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            margin: 0.35rem 0 1rem 0;
        }
        .raghsh-step-label {
            color: var(--raghsh-accent);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .raghsh-step-item {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.72rem 0.85rem;
            border: 1px solid var(--raghsh-surface-border);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(17, 24, 39, 0.82));
            color: var(--raghsh-surface-text);
        }
        .raghsh-step-item.is-active {
            border-color: var(--raghsh-surface-border-strong);
            box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.12);
        }
        .raghsh-step-number {
            flex: none;
            width: 1.65rem;
            height: 1.65rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--raghsh-accent-strong);
            color: white;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1;
        }
        div[data-testid="stExpander"] details {
            border: 1px solid var(--raghsh-surface-border);
            border-radius: 16px;
            background: var(--raghsh-surface);
            overflow: hidden;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }
        div[data-testid="stExpander"] summary {
            background: rgba(15, 23, 42, 0.72);
        }
        div[data-testid="stExpander"] summary:hover {
            background: rgba(15, 23, 42, 0.82);
        }
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary svg,
        div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            color: var(--raghsh-surface-text);
        }
        div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            background: rgba(17, 24, 39, 0.68);
            border-top: 1px solid rgba(148, 163, 184, 0.12);
        }
        div[data-testid="stExpander"] summary p {
            font-weight: 600;
        }
        div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] > div {
            background: transparent;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.52);
            padding: 0.25rem 0.35rem;
        }
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricValue"] p,
        div[data-testid="stMetricDeltaDescription"] p {
            color: var(--raghsh-surface-text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_step_list(
    steps: list[str],
    *,
    placeholder=None,
    highlight_last: bool = False,
) -> None:
    if not steps:
        return

    items: list[str] = []
    for index, step in enumerate(steps, start=1):
        css_class = "raghsh-step-item is-active" if highlight_last and index == len(steps) else "raghsh-step-item"
        items.append(
            f'<div class="{css_class}">'
            f'<span class="raghsh-step-number">{index}</span>'
            f"<div>{escape(step)}</div>"
            "</div>"
        )

    markup = (
        '<div class="raghsh-step-shell">'
        '<div class="raghsh-step-label">Schritt für Schritt</div>'
        + "".join(items)
        + "</div>"
    )
    if placeholder is None:
        st.markdown(markup, unsafe_allow_html=True)
    else:
        placeholder.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Quellen-Anzeige
# ---------------------------------------------------------------------------


def render_sources(results: list[dict]) -> None:
    if not results:
        return
    with st.expander("Quellen", expanded=False):
        for i, r in enumerate(results, 1):
            p = r["payload"]
            title   = p.get("title", "(kein Titel)")
            url     = p.get("source_url", "")
            heading = p.get("section_heading", "")
            faculty = p.get("faculty", "")

            st.markdown(f"**{i}. {title}**")
            if url:
                st.markdown(f"[{url}]({url})")
            meta_parts = []
            if heading:
                meta_parts.append(f"Abschnitt: {heading}")
            if faculty:
                meta_parts.append(f"Fakultät: {faculty}")
            meta_parts.append(f"RRF-Score: {r['score']:.4f}")
            st.caption("  ·  ".join(meta_parts))

            if i < len(results):
                st.divider()

def render_process_trace(process_trace: dict | None, followup_plan: dict | None = None) -> None:
    if not process_trace and not followup_plan:
        return
    with st.expander("Systemprozess", expanded=False):
        if followup_plan and followup_plan.get("mode") == "need_more_context":
            action = followup_plan.get("requested_action", "")
            reason = followup_plan.get("reason", "")
            source = followup_plan.get("target_source_url", "")
            parts = []
            if action:
                parts.append(f"Kontext-Erweiterung: {action}")
            if reason:
                parts.append(f"Grund: {reason}")
            if source:
                parts.append(f"Quelle: {source}")
            if parts:
                st.markdown("**Kontext-Erweiterung:**")
                for part in parts:
                    st.markdown(f"- {part}")
                if process_trace:
                    st.divider()
        if not process_trace:
            return
        st.markdown(f"**Originalfrage:** {process_trace.get('original_query', '')}")
        selected = process_trace.get("selected_query", "")
        if selected and selected != process_trace.get("original_query", ""):
            st.markdown(f"**Gewählte Formulierung:** {selected}")
        if process_trace.get("clarification"):
            st.markdown(f"**Zusatz:** {process_trace['clarification']}")
        specificity = process_trace.get("specificity")
        if specificity:
            st.markdown(f"**Frageschärfe:** {specificity}")
        intents = process_trace.get("assessment_intents", [])
        if intents:
            st.markdown(f"**Erkannte Suchtypen:** {', '.join(intents)}")
        abbreviations = process_trace.get("detected_abbreviations", [])
        if abbreviations:
            st.markdown(f"**Erkannte Kürzel:** {', '.join(abbreviations)}")
        faculties = process_trace.get("detected_faculties", [])
        if faculties:
            st.markdown(f"**Erkannte Fakultätsform:** {', '.join(faculties)}")
        glossary_hits = process_trace.get("glossary_hits", [])
        if glossary_hits:
            st.markdown("**Glossar- und Definitionsspuren:**")
            for hit in glossary_hits:
                st.markdown(f"- {hit}")
        term_corrections = process_trace.get("term_corrections", [])
        if term_corrections:
            st.markdown("**Naheliegende Schreibvarianten im Kontext:**")
            for item in term_corrections:
                st.markdown(f"- `{item.get('asked', '')}` -> `{item.get('matched', '')}`")
        reasons = process_trace.get("assessment_reasons", [])
        if reasons:
            st.markdown("**Analyse:**")
            for reason in reasons:
                st.markdown(f"- {reason}")
        clarification_prompt = process_trace.get("clarification_prompt", "")
        if clarification_prompt:
            st.markdown(f"**Präzisierungshinweis:** {clarification_prompt}")
        clarification_options = process_trace.get("clarification_options", [])
        if clarification_options:
            st.markdown("**Vorgeschlagene Richtungen:**")
            for option in clarification_options:
                st.markdown(f"- {option}")
        planner_guidance = process_trace.get("planner_guidance", {})
        if planner_guidance:
            st.markdown("**LLM-Planer:**")
            if planner_guidance.get("normalized_question"):
                st.markdown(f"- Normalisiert: `{planner_guidance['normalized_question']}`")
            for term in planner_guidance.get("canonical_hsh_terms", []) or []:
                st.markdown(f"- HsH-Begriff: `{term}`")
            for hint in planner_guidance.get("source_type_hints", []) or []:
                st.markdown(f"- Quelltyp: `{hint}`")
        retrieval_queries = process_trace.get("retrieval_queries", [])
        if retrieval_queries:
            st.markdown("**Suchpfade:**")
            for retrieval_query in retrieval_queries:
                st.markdown(f"- `{retrieval_query}`")


def render_conversation_memory(memory: list[dict]) -> None:
    with st.expander(f"Konversationsgedächtnis (max. {MAX_MEMORY_TURNS})", expanded=False):
        st.caption(
            "Die letzten beantworteten Fragen werden stark komprimiert und nur als Gesprächskontext "
            "an das Modell weitergegeben. Verbindlich bleiben die aktuellen offiziellen Quellen."
        )
        if not memory:
            st.markdown("Noch keine gespeicherten Kurzfassungen.")
            return
        for idx, entry in enumerate(memory, start=1):
            st.markdown(f"**{idx}.** {format_memory_entry(entry)}")


def render_timings(timings: dict[str, float] | None) -> None:
    if not timings:
        return
    with st.expander("Leistungsdaten", expanded=False):
        for label, seconds in timings.items():
            st.markdown(f"- **{label}**: {seconds:.2f}s")


def run_routed_turn(question: str, route: dict[str, str]) -> None:
    """Antwortet auf Anschlussfragen direkt aus dem letzten Turn-Zustand."""
    answer = route.get("response", "")
    steps = [
        "Anschlussfrage erkannt.",
        "Der letzte Gesprächsschritt wurde als Kontext wiederverwendet.",
        "Es wird keine neue Retrieval-Runde gestartet.",
        "Antwort fertig.",
    ]
    with st.chat_message("assistant"):
        render_step_list(steps)
        with st.expander("Antwort", expanded=True):
            st.markdown(answer)
        render_timings({"Router-Antwort": 0.0})

    st.session_state.conversation_memory = append_summary(
        st.session_state.get("conversation_memory", []),
        summarize_turn(question, answer, process_trace={}),
    )
    st.session_state.last_turn_state = build_turn_state(
        question=question,
        answer=answer,
        results=[],
        process_trace={},
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "steps": steps,
        "timings": {"Router-Antwort": 0.0},
        "turn_state": st.session_state.last_turn_state,
    })


def run_assistant_turn(
    *,
    qdrant: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    reranker,
    openai_client: OpenAI,
    modell: str,
    rolle: str,
    fakultaet: str,
    question: str,
    selected_query: str | None = None,
    clarification: str = "",
    conversation_memory: list[dict] | None = None,
    pipeline_config: RagPipelineConfig | None = None,
) -> None:
    """Führt einen kompletten Assistant-Turn mit sichtbarem Prozess aus."""
    steps: list[str] = []

    def log_step(message: str) -> None:
        cleaned = message.strip()
        if not cleaned:
            return
        if steps and steps[-1] == cleaned:
            return
        steps.append(cleaned)
        render_step_list(steps, placeholder=steps_placeholder, highlight_last=True)

    with st.chat_message("assistant"):
        steps_placeholder = st.empty()
        with st.expander("Antwort", expanded=True):
            answer_placeholder = st.empty()
            answer_placeholder.caption("Antwort wird vorbereitet …")

        prepared = prepare_chat_turn(
            qdrant=qdrant,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            openai_client=openai_client,
            model=modell,
            rolle=rolle,
            fakultaet=fakultaet,
            question=question,
            top_k=(pipeline_config.top_k if pipeline_config is not None else RAG_TOP_K),
            selected_query=selected_query,
            clarification=clarification,
            conversation_memory=conversation_memory,
            status_callback=log_step,
            pipeline_config=pipeline_config,
        )

        timings = dict(prepared["timings"])
        results = prepared["results"]
        process_trace = prepared["process_trace"]
        followup_plan = prepared["followup_plan"]

        if prepared["no_results"]:
            answer = "Keine passenden Dokumente in der Wissensdatenbank gefunden."
            log_step("Keine passenden Treffer gefunden.")
            answer_placeholder.warning(answer)
            timings["Gesamt"] = sum(timings.values())
            render_step_list(steps, placeholder=steps_placeholder, highlight_last=False)
            render_process_trace(process_trace, followup_plan=followup_plan)
            render_timings(timings)
        elif prepared["guard_answer"] is not None:
            log_step("Zeitkritische Frage erkannt; Antwort wird vorsichtig direkt ausgegeben.")
            answer = prepared["guard_answer"]
            answer_placeholder.markdown(answer)
            timings["Antwort-LLM"] = 0.0
            log_step("Antwort fertig.")
            timings["Gesamt"] = sum(timings.values())
            render_step_list(steps, placeholder=steps_placeholder, highlight_last=False)
            render_process_trace(process_trace, followup_plan=followup_plan)
            render_sources(results)
            render_timings(timings)
        else:
            log_step("Antwort wird mit dem gewählten Modell formuliert.")
            try:
                llm_t0 = time.perf_counter()
                answer = stream_response(
                    openai_client,
                    modell,
                    prepared["llm_messages"],
                    answer_placeholder=answer_placeholder,
                )
                if not answer.strip():
                    log_step("Streaming lieferte keinen sichtbaren Antworttext; Fallback wird ausgeführt.")
                    answer = request_answer_sync(
                        openai_client,
                        modell,
                        prepared["llm_messages"],
                    )
                    answer = answer.strip()
                    if answer:
                        answer_placeholder.markdown(answer)
                    else:
                        raise RuntimeError("Leere Modellantwort erhalten.")
                timings["Antwort-LLM"] = time.perf_counter() - llm_t0
            except Exception as exc:
                log_step("Fehler bei der API-Anfrage.")
                render_step_list(steps, placeholder=steps_placeholder, highlight_last=False)
                answer_placeholder.error(f"Fehler bei der API-Anfrage: {exc}")
                return

            log_step("Antwort fertig.")
            timings["Gesamt"] = sum(timings.values())
            render_step_list(steps, placeholder=steps_placeholder, highlight_last=False)
            render_process_trace(process_trace, followup_plan=followup_plan)
            render_sources(results)
            render_timings(timings)

    st.session_state.conversation_memory = append_summary(
        st.session_state.get("conversation_memory", []),
        summarize_turn(
            question,
            answer,
            process_trace=process_trace,
        ),
    )
    st.session_state.last_turn_state = build_turn_state(
        question=question,
        answer=answer,
        results=results,
        process_trace=process_trace,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "followup_plan": followup_plan,
        "steps": steps,
        "sources": results,
        "process_trace": process_trace,
        "timings": timings,
        "turn_state": st.session_state.last_turn_state,
    })


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(modelle: list[str]) -> dict[str, str]:
    with st.sidebar:
        st.markdown("## ⚙️ Einstellungen")
        app_mode = st.radio(
            "Modus",
            ["Chatbot", "Validierungssystem"],
            index=0,
        )

        if app_mode == "Chatbot":
            st.markdown("### Mein Profil")
            rolle = st.radio("Ich bin …", ROLLEN, index=0)
            fakultaet = st.selectbox("Fakultät", FAKULTAETEN, index=0)

            st.markdown("### Modell")
            modell = st.selectbox("GWDG-Modell", modelle, index=0)
            evaluator_model = modell
        else:
            rolle = ROLLEN[0]
            fakultaet = FAKULTAETEN[0]
            st.markdown("### Validierungssystem")
            modell = st.selectbox("Chatbot-Modell", modelle, index=0, key="validation_chatbot_model")
            default_eval_index = 1 if len(modelle) > 1 else 0
            evaluator_model = st.selectbox(
                "Evaluator-Modell",
                modelle,
                index=default_eval_index,
                key="validation_evaluator_model",
            )

        st.markdown("### RAG-Pipeline")
        top_k = int(
            st.number_input(
                "Qdrant Top-K",
                min_value=1,
                max_value=20,
                value=RAG_TOP_K,
                step=1,
                help="Anzahl finaler Qdrant-Treffer/Chunks für Chatbot und Validierung.",
            )
        )
        query_assist_enabled = st.checkbox(
            "Query Assist aktiv",
            value=True,
            help="Analysiert Fragen, erkennt Intents/Fakultäten/Kürzel und erzeugt Suchvarianten.",
        )
        retrieval_planner_enabled = st.checkbox(
            "Retrieval Planner aktiv",
            value=True,
            help="Nutzt optional ein LLM, um schwierige Suchfragen besser für Retrieval vorzubereiten.",
        )
        rag_followup_enabled = st.checkbox(
            "RAG Follow-up aktiv",
            value=True,
            help="Prüft den ersten Kontext und lädt bei Bedarf zusätzliche Evidenz nach.",
        )
        pipeline_config = RagPipelineConfig(
            top_k=top_k,
            query_assist_enabled=query_assist_enabled,
            retrieval_planner_enabled=retrieval_planner_enabled,
            rag_followup_enabled=rag_followup_enabled,
        )

        st.divider()
        if app_mode == "Chatbot":
            st.info(
                "**HsH-KI-Assistent**\n\n"
                "Beantwortet Fragen ausschließlich auf Basis offizieller "
                "HsH-Dokumente. Keine Terminvereinbarung, keine Formulare.\n\n"
                f"Betrieben mit GWDG ChatAI & lokalem Qdrant-Index.\n\n"
                f"Konversationsgedächtnis: max. {MAX_MEMORY_TURNS} komprimierte beantwortete Fragen."
            )
            render_conversation_memory(st.session_state.get("conversation_memory", []))

            if st.button("Chat zurücksetzen", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_memory = []
                st.session_state.pending_query_assist = None
                st.rerun()
        else:
            st.info(
                "**Strict dialog v1**\n\n"
                "Verwendet feste Fälle aus dem Codeordner, führt sie einzeln gegen den Chatbot aus "
                "und bewertet jede Antwort direkt mit einem zweiten Modell."
            )
            if st.button("Validierung zurücksetzen", use_container_width=True):
                reset_validation_state()
                st.rerun()

    return {
        "app_mode": app_mode,
        "rolle": rolle,
        "fakultaet": fakultaet,
        "modell": modell,
        "evaluator_model": evaluator_model,
        "pipeline_config": pipeline_config,
    }


# ---------------------------------------------------------------------------
# Haupt-App
# ---------------------------------------------------------------------------


def main() -> None:
    # ── Pflichtprüfungen ──────────────────────────────────────────────────
    if not GWDG_API_KEY:
        st.error("GWDG_API_KEY fehlt. Bitte `.env`-Datei prüfen.")
        st.stop()

    try:
        qdrant = get_qdrant_client()
    except Exception as exc:
        st.error(f"Qdrant nicht erreichbar: {exc}\n\n`docker compose up -d` ausführen.")
        st.stop()

    dense_embedder  = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()
    reranker        = get_reranker()
    openai_client   = get_openai_client()
    eval_all_client = get_rate_limited_openai_client(MAX_EVAL_API_REQUESTS_PER_MINUTE)
    modelle         = get_model_list()

    # ── Sidebar ───────────────────────────────────────────────────────────
    sidebar_state = render_sidebar(modelle)
    rolle = sidebar_state["rolle"]
    fakultaet = sidebar_state["fakultaet"]
    modell = sidebar_state["modell"]
    evaluator_model = sidebar_state["evaluator_model"]
    app_mode = sidebar_state["app_mode"]
    pipeline_config = sidebar_state["pipeline_config"]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "last_turn_state" not in st.session_state:
        st.session_state.last_turn_state = None

    inject_custom_styles()

    if app_mode == "Validierungssystem":
        render_validation_page(
            qdrant=qdrant,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            openai_client=openai_client,
            bulk_openai_client=eval_all_client,
            modelle=modelle,
            chatbot_model=modell,
            evaluator_model=evaluator_model,
            pipeline_config=pipeline_config,
        )
        return

    # ── Titel ─────────────────────────────────────────────────────────────
    st.title("🎓 HsH-Assistent")
    st.caption(f"Modell: **{modell}** · Rolle: **{rolle}** · Fakultät: **{fakultaet}**")

    # ── Chat-Verlauf ──────────────────────────────────────────────────────

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] != "assistant":
                st.markdown(msg["content"])
                continue

            if msg.get("steps"):
                render_step_list(msg["steps"])
            with st.expander("Antwort", expanded=True):
                st.markdown(msg["content"])
            render_process_trace(msg.get("process_trace"), followup_plan=msg.get("followup_plan"))
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("timings"):
                render_timings(msg["timings"])

    pending = st.session_state.get("pending_query_assist")
    if pending and not pipeline_config.query_assist_enabled:
        st.session_state.pending_query_assist = None
        pending = None
    if pending:
        with st.chat_message("assistant"):
            st.markdown("Ich brauche hier eine präzisere Richtung, damit ich nicht den falschen HsH-Prozess beantworte.")
            if pending.get("clarification_prompt"):
                st.markdown(pending["clarification_prompt"])
            for reason in pending.get("reasons", []):
                st.markdown(f"- {reason}")
            st.caption("Im Hintergrund werden zusätzlich passende Suchvarianten vorbereitet.")
            with st.form("query_assist_form"):
                options = [pending["question"], *(pending.get("clarification_options") or pending.get("suggestions", []))]
                selected = st.radio("Formulierung wählen", options, index=0)
                clarification = st.text_input(
                    "Optional ergänzen",
                    help=pending.get("clarification_hint", ""),
                )
                submit = st.form_submit_button("Suche starten", use_container_width=True)

            if submit:
                st.session_state.pending_query_assist = None
                run_assistant_turn(
                    qdrant=qdrant,
                    dense_embedder=dense_embedder,
                    sparse_embedder=sparse_embedder,
                    reranker=reranker,
                    openai_client=openai_client,
                    modell=modell,
                    rolle=rolle,
                    fakultaet=fakultaet,
                    question=pending["question"],
                    selected_query=None if selected == pending["question"] else selected,
                    clarification=clarification,
                    conversation_memory=st.session_state.get("conversation_memory", []),
                    pipeline_config=pipeline_config,
                )
                st.rerun()
        return

    # ── Nutzereingabe ─────────────────────────────────────────────────────
    question = st.chat_input("Stellen Sie Ihre Frage zur Hochschule Hannover …")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    routed_turn = route_turn(question, st.session_state.get("last_turn_state"))
    if routed_turn is not None:
        run_routed_turn(question, routed_turn)
        return
    if pipeline_config.query_assist_enabled:
        assessment = assess_query(question)
        if pipeline_config.retrieval_planner_enabled and should_use_retrieval_planner(assessment, question):
            try:
                planner_guidance = request_retrieval_plan(
                    openai_client,
                    modell,
                    question,
                    assessment,
                    conversation_memory_prompt=build_memory_prompt(
                        st.session_state.get("conversation_memory", [])
                    ),
                )
                assessment = apply_planner_guidance(assessment, planner_guidance)
            except Exception:
                pass
        if assessment.needs_user_choice:
            st.session_state.pending_query_assist = {
                "question": question,
                "reasons": assessment.reasons,
                "suggestions": assessment.suggestions,
                "clarification_prompt": assessment.clarification_prompt,
                "clarification_options": assessment.clarification_options,
                "clarification_hint": assessment.clarification_hint,
            }
            st.rerun()

    run_assistant_turn(
        qdrant=qdrant,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        reranker=reranker,
        openai_client=openai_client,
        modell=modell,
        rolle=rolle,
        fakultaet=fakultaet,
        question=question,
        conversation_memory=st.session_state.get("conversation_memory", []),
        pipeline_config=pipeline_config,
    )


if __name__ == "__main__":
    main()
