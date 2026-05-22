"""CLI runner for the strict-dialog validation system."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_MAX_REQUESTS_PER_MINUTE = 10
DEFAULT_QDRANT_URL = "http://localhost:6333"
FALLBACK_MODELS = ["meta-llama-3.1-8b-instruct", "gpt-4o-mini", "gpt-4o"]

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = Path(os.getenv("RAG_CONFIG_DIR", str(REPO_ROOT / "config"))).expanduser()
DEFAULT_CASES_FILE = Path(__file__).parent / "validation_cases.json"
CASES_FILE = Path(os.getenv("RAG_VALIDATION_CASES_FILE", str(DEFAULT_CASES_FILE))).expanduser()


class ConsoleStatus:
    def write(self, message: str) -> None:
        print(f"  {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Führt die feste Strict-Dialog-Fallbasis gegen Chatbot- und Evaluator-Modell aus. "
            "Ohne Fallauswahl werden alle Fälle sequenziell gestartet."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Alle Fälle ausführen (Standard).")
    selection.add_argument("--case", help="Nur den Fall mit dieser ID ausführen, z.B. q03.")
    selection.add_argument("--case-index", type=int, help="Nur den 1-basierten Fallindex ausführen.")
    parser.add_argument("--list-cases", action="store_true", help="Fallliste ausgeben und beenden.")
    parser.add_argument("--chatbot-model", help="Chatbot-Modell. Standard: erstes verfügbares Modell.")
    parser.add_argument(
        "--evaluator-model",
        help="Evaluator-Modell. Standard: zweites verfügbares Modell, sonst Chatbot-Modell.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help=f"Qdrant-URL (Standard: {DEFAULT_QDRANT_URL}).",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Anzahl Retrieval-Treffer pro Fall.")
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
    parser.add_argument(
        "--max-requests-per-minute",
        type=int,
        default=DEFAULT_MAX_REQUESTS_PER_MINUTE,
        help="Maximale GWDG-API-Aufrufe pro Minute.",
    )
    return parser.parse_args(argv)


def read_raw_cases() -> list[dict[str, Any]]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def list_cases() -> None:
    for index, case in enumerate(read_raw_cases(), start=1):
        clarification = case.get("clarification") or {}
        behavior = case.get("expected_behavior") or (
            "clarify_then_answer" if clarification.get("expected") else "direct_answer"
        )
        policy = case.get("source_policy") or "exact_gold_source"
        print(
            f"{index:>2}. {case.get('id', '')} [{behavior}, {policy}] "
            f"{case.get('question', '')}"
        )


def load_validation_runtime():
    try:
        from .validation_system import (
            VALIDATION_INTER_CASE_DELAY_SECONDS,
            VALIDATION_RATE_LIMIT_MAX_RETRIES,
            VALIDATION_RATE_LIMIT_RETRY_SECONDS,
            load_eval_cases,
            run_remaining_validation_cases,
            start_validation_run,
        )
        from ..web_app_runtime import RagPipelineConfig
    except ImportError:  # pragma: no cover - script execution fallback
        from evals.validation_system import (
            VALIDATION_INTER_CASE_DELAY_SECONDS,
            VALIDATION_RATE_LIMIT_MAX_RETRIES,
            VALIDATION_RATE_LIMIT_RETRY_SECONDS,
            load_eval_cases,
            run_remaining_validation_cases,
            start_validation_run,
        )
        from web_app_runtime import RagPipelineConfig

    return (
        load_eval_cases,
        run_remaining_validation_cases,
        start_validation_run,
        RagPipelineConfig,
        VALIDATION_INTER_CASE_DELAY_SECONDS,
        VALIDATION_RATE_LIMIT_RETRY_SECONDS,
        VALIDATION_RATE_LIMIT_MAX_RETRIES,
    )


def select_cases(args: argparse.Namespace, cases: list[Any]) -> list[Any]:
    if args.case:
        selected = [case for case in cases if case.id == args.case]
        if not selected:
            raise SystemExit(f"Unbekannte Fall-ID: {args.case}")
        return selected

    if args.case_index is not None:
        if args.case_index < 1 or args.case_index > len(cases):
            raise SystemExit(f"Fallindex muss zwischen 1 und {len(cases)} liegen.")
        return [cases[args.case_index - 1]]

    return cases


def build_openai_client(*, max_requests_per_minute: int):
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(f"Fehlende Python-Abhängigkeit für den Eval-Lauf: {exc}") from exc

    try:
        from ..api_rate_limiter import SlidingWindowRateLimiter, build_rate_limited_http_client
    except ImportError:  # pragma: no cover - script execution fallback
        from api_rate_limiter import SlidingWindowRateLimiter, build_rate_limited_http_client

    load_dotenv(CONFIG_DIR / ".env")
    api_key = os.getenv("GWDG_API_KEY", "")
    api_base = os.getenv("GWDG_API_BASE", "https://chat-ai.academiccloud.de/v1")
    if not api_key:
        raise SystemExit("GWDG_API_KEY fehlt. Bitte config/.env prüfen.")

    def on_wait(seconds: float) -> None:
        print(f"  Rate-Limit: warte {seconds:.1f}s für maximal {max_requests_per_minute}/Minute.", flush=True)

    return OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=30,
        http_client=build_rate_limited_http_client(
            timeout=30,
            rate_limiter=SlidingWindowRateLimiter(
                max_requests=max_requests_per_minute,
                on_wait=on_wait,
            ),
        ),
    )


def resolve_models(
    client,
    *,
    chatbot_model: str | None,
    evaluator_model: str | None,
) -> tuple[str, str]:
    available = []
    try:
        available = sorted(model.id for model in client.models.list().data)
    except Exception as exc:
        print(f"Modellliste konnte nicht geladen werden, nutze Fallbacks: {exc}", file=sys.stderr)
        available = FALLBACK_MODELS

    chatbot = chatbot_model or available[0]
    evaluator = evaluator_model or (available[1] if len(available) > 1 else chatbot)
    return chatbot, evaluator


def load_runtime(qdrant_url: str):
    try:
        from dotenv import load_dotenv
        from fastembed import SparseTextEmbedding, TextEmbedding
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit(f"Fehlende Python-Abhängigkeit für den Eval-Lauf: {exc}") from exc

    try:
        from ..hybrid_search import DENSE_MODEL, SPARSE_MODEL
        from ..web_app_runtime import create_reranker
    except ImportError:  # pragma: no cover - script execution fallback
        from hybrid_search import DENSE_MODEL, SPARSE_MODEL
        from web_app_runtime import create_reranker

    load_dotenv(CONFIG_DIR / ".env")
    model_cache_dir = Path(
        os.getenv("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))
    ).expanduser()
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(model_cache_dir))

    print(f"Verbinde mit Qdrant: {qdrant_url}")
    qdrant = QdrantClient(url=qdrant_url, timeout=10)
    qdrant.get_collections()

    print(f"Lade Dense-Modell: {DENSE_MODEL}")
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL, cache_dir=str(model_cache_dir))
    print(f"Lade Sparse-Modell: {SPARSE_MODEL}")
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=str(model_cache_dir))

    try:
        print("Lade Reranker")
        reranker = create_reranker(cache_dir=str(model_cache_dir))
    except Exception as exc:
        print(f"Reranker nicht verfügbar, weiter ohne Reranking: {exc}")
        reranker = None

    return qdrant, dense_embedder, sparse_embedder, reranker


def print_summary(run_dir: str) -> None:
    summary_path = Path(run_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\nValidierung abgeschlossen")
    print(f"Run-ID: {summary['run_id']}")
    print(f"Ergebnisse: {run_dir}")
    print(f"Fälle: {summary['completed_cases']}")
    print(f"Gesamtscore: {summary['overall_score']:.1f}/100")
    if "evidence_overall_score" in summary:
        print(f"Evidenzscore: {summary['evidence_overall_score']:.1f}/100")
    if "two_stage_overall_score" in summary:
        print(f"Zweistufiger Score: {summary['two_stage_overall_score']:.1f}/100")
    if summary.get("failure_type_counts"):
        print("Diagnose-Typen:")
        for failure_type, count in sorted(summary["failure_type_counts"].items()):
            print(f"  - {failure_type}: {count}")
    if summary.get("human_review_recommended_cases"):
        print(f"Fachliche Nachprüfung empfohlen: {summary['human_review_recommended_cases']} Fall/Fälle")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_cases:
        list_cases()
        return
    if args.max_requests_per_minute < 1:
        raise SystemExit("--max-requests-per-minute muss mindestens 1 sein.")
    if args.top_k < 1:
        raise SystemExit("--top-k muss mindestens 1 sein.")

    (
        load_eval_cases,
        run_remaining_validation_cases,
        start_validation_run,
        RagPipelineConfig,
        inter_case_delay_seconds,
        rate_limit_retry_seconds,
        rate_limit_max_retries,
    ) = load_validation_runtime()
    cases = select_cases(args, load_eval_cases())
    client = build_openai_client(max_requests_per_minute=args.max_requests_per_minute)
    chatbot_model, evaluator_model = resolve_models(
        client,
        chatbot_model=args.chatbot_model,
        evaluator_model=args.evaluator_model,
    )
    qdrant, dense_embedder, sparse_embedder, reranker = load_runtime(args.qdrant_url)
    pipeline_config = RagPipelineConfig(
        top_k=args.top_k,
        query_assist_enabled=not args.no_query_assist,
        retrieval_planner_enabled=not args.no_retrieval_planner,
        rag_followup_enabled=not args.no_rag_followup,
    )

    print(
        "\nStrict-dialog Validierung startet "
        f"({len(cases)} Fall/Fälle, max. {args.max_requests_per_minute} GWDG-API-Aufrufe/Minute)."
    )
    print(f"Chatbot-Modell: {chatbot_model}")
    print(f"Evaluator-Modell: {evaluator_model}")
    print(
        "Pipeline: "
        f"Top-K={pipeline_config.top_k}, "
        f"Query Assist={'aktiv' if pipeline_config.query_assist_enabled else 'inaktiv'}, "
        f"Retrieval Planner={'aktiv' if pipeline_config.retrieval_planner_enabled else 'inaktiv'}, "
        f"RAG Follow-up={'aktiv' if pipeline_config.rag_followup_enabled else 'inaktiv'}"
    )
    print(f"Pause zwischen Fällen: ca. {inter_case_delay_seconds:.0f}s")
    print(
        "Bei API-Limit-Fehlern: "
        f"{rate_limit_retry_seconds:.0f}s warten und denselben Schritt erneut versuchen "
        f"(max. {rate_limit_max_retries} Retry/Retrys).\n"
    )

    state = start_validation_run(
        chatbot_model=chatbot_model,
        evaluator_model=evaluator_model,
        cases=cases,
        pipeline_config=pipeline_config,
        store_in_session=False,
    )
    run_remaining_validation_cases(
        state=state,
        cases=cases,
        qdrant=qdrant,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        reranker=reranker,
        openai_client=client,
        top_k=args.top_k,
        pipeline_config=pipeline_config,
        status_writer=ConsoleStatus(),
    )
    print_summary(state["run_dir"])


if __name__ == "__main__":
    main()
