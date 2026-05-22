"""
Hybrid-Suche — Semantische und schlüsselwortbasierte Suche in der Wissensdatenbank.

Kurzbeschreibung
----------------
Durchsucht die Qdrant-Collection 'hsh_knowledge' mit einer kombinierten
Hybrid-Suche: Dense Vektorsuche (Jina) und BM25 Sparse Vektorsuche werden
per Qdrant-nativer Prefetch+RRF-Fusion zusammengeführt. Kann als
eigenständige CLI oder als importiertes Modul vom Chatbot verwendet werden.

Ausführliche Beschreibung
--------------------------
Eine einfache Vektorsuche findet semantisch ähnliche Texte, übersieht aber
exakte Schlüsselwörter (z.B. Modulnummern, Namen). Eine reine Keyword-Suche
findet exakte Begriffe, versteht aber keine Bedeutungen. Die Hybrid-Suche
kombiniert beide Verfahren und erreicht so höhere Treffergenauigkeit.

Sucharchitektur:

1. Semantische Suche (Dense Search, Arm 1)
   Die Nutzeranfrage wird mit jinaai/jina-embeddings-v3 vektorisiert
   (task="retrieval.query"). Qdrant vergleicht den Vektor gegen den
   benannten Vektor "dense" und liefert die CANDIDATE_LIMIT ähnlichsten Treffer.

2. BM25 Sparse Search (Arm 2)
   Die Anfrage wird mit SparseTextEmbedding("Qdrant/bm25") in einen
   Sparse-Vektor umgewandelt. Qdrant vergleicht diesen gegen den
   benannten Vektor "sparse" — echte Relevanz-Scores, nicht Storage-Reihenfolge.

3. Qdrant-native RRF-Fusion (Prefetch + FusionQuery)
   Beide Arme laufen als Prefetch in einem einzigen query_points()-Aufruf.
   Qdrant führt die Reciprocal Rank Fusion intern durch — kein Python-RRF-Code.
   FusionQuery(RRF) belohnt Treffer, die in beiden Armen gut ranken.

4. Deduplizierung nach Quell-URL (max_per_url=2)
   Nach der Fusion: maximal MAX_PER_URL Chunks pro source_url werden behalten.
   Dies verhindert, dass viele Chunks desselben Dokuments alle Top-Plätze
   belegen, erlaubt aber einen zweiten relevanten Absatz pro Quelle.

5. Semantisches Reranking (Cross-Encoder)
   Nach URL-Dedup werden die Treffer mit jinaai/jina-reranker-v2-base-multilingual
   nach echter semantischer Relevanz neu bewertet und sortiert. Der Cross-Encoder
   bewertet (Query, Passage)-Paare — deutlich präziser als reine Vektornähe.
   Aktivierbar via USE_RERANKER, deaktivierbar falls Modell nicht verfügbar.

6. Context Augmentation (Nachbar-Chunks, Top-N)
   Für die AUGMENT_TOP_N am höchsten gerankten Treffer werden die unmittelbar
   benachbarten Chunks (chunk_index - 1 und chunk_index + 1) derselben source_url
   nachgeladen. Dies verhindert, dass Informationen an Chunk-Grenzen zerrissen werden.
   Nachbar-Chunks werden dem Kontext vorangestellt/angehängt, aber nicht
   als eigenständige Top-Treffer gezählt.

Konfiguration:
   TOP_K           — Anzahl der finalen Ergebnisse (Standard: 8)
   CANDIDATE_LIMIT — Kandidaten pro Sucharm (Standard: 100)
   DEDUP_BUFFER    — Faktor für Überabtastung vor URL-Dedup (Standard: 4)
   MAX_PER_URL     — Max. Chunks pro source_url nach Dedup (Standard: 2)
   USE_RERANKER    — Semantisches Reranking aktivieren (Standard: True)
   AUGMENT_TOP_N   — Für wie viele Top-Treffer Nachbar-Chunks geladen werden (Standard: 3)

Voraussetzungen:
   - Qdrant läuft lokal, Collection 'hsh_knowledge' befüllt mit Dense+Sparse
     (Ausgabe von hpc_vectorizer.py + local_importer.py)

Abhängigkeiten: qdrant-client, fastembed
"""

import re
import sys

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

try:
    from .glossary_index import find_similar_corpus_terms, lookup_glossary_results
    from .query_assist import (
        QueryAssessment,
        assess_query,
        build_plain_query_assessment,
        build_retrieval_queries,
    )
except ImportError:
    from glossary_index import find_similar_corpus_terms, lookup_glossary_results
    from query_assist import (
        QueryAssessment,
        assess_query,
        build_plain_query_assessment,
        build_retrieval_queries,
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "hsh_knowledge"
DENSE_MODEL     = "jinaai/jina-embeddings-v3"
SPARSE_MODEL    = "Qdrant/bm25"
RERANKER_MODEL  = "jinaai/jina-reranker-v2-base-multilingual"

# URLs die in Suchergebnissen nicht angezeigt werden sollen
BLOCKED_URL_PREFIXES = (
    "https://serwiss.bib.",
    "http://serwiss.bib.",
)

TOP_K           = 8    # finale Ergebnisse nach URL-Dedup
CANDIDATE_LIMIT = 100  # Kandidaten pro Sucharm (Prefetch limit)
DEDUP_BUFFER    = 4    # TOP_K * DEDUP_BUFFER = Ergebnisse von Qdrant vor Dedup
MAX_PER_URL     = 2    # max. Chunks pro source_url nach Dedup
USE_RERANKER    = True # Cross-Encoder Reranking nach URL-Dedup
AUGMENT_TOP_N   = 3    # Nachbar-Chunks nur für die Top-N Treffer laden

DEFINITION_BOOST_MARKERS = (
    "bedeutet",
    "steht fuer",
    "legende",
    "abkuerzung",
    "pruefungsform",
    "pruefungsformen",
    "hinweise",
    "klausur",
    "semesterwochenstunden",
    "pruefungsordnung",
)
WORKFLOW_BOOST_MARKERS = (
    "bewerbungsportal",
    "campusmanagement",
    "pruefungsanmeldung",
    "online-antrag",
    "online-antraege",
    "formular",
    "antragsformular",
    "hochladen",
    "digital",
    "einreichen",
    "portal",
)
EXAM_REGISTRATION_MARKERS = (
    "campusmanagement",
    "pruefungsanmeldung",
    "pruefungen und studium",
    "icms",
)
CONTACT_BOOST_MARKERS = (
    "akademische angelegenheiten",
    "kontakt",
    "ansprech",
    "studierendenservice",
    "studieninteressierte",
    "bewerber",
    "studienberatung",
    "service center",
    "servicecenter",
    "email",
    "e-mail",
    "telefon",
    "sprechstunden",
    "zustaendig",
)
TERM_ALIAS_HINTS = {
    "BPO": (
        "besondere pruefungsordnung",
        "besonderer teil der pruefungsordnung",
    ),
    "ECTS": (
        "european credit transfer system",
        "leistungspunkte",
        "leistungspunkt",
        "workload",
    ),
    "K60": (
        "klausur 60 minuten",
        "60 minutige klausur",
        "60 minuetige klausur",
    ),
    "K90": (
        "klausur 90 minuten",
        "90 minutige klausur",
        "90 minuetige klausur",
    ),
    "PO": (
        "pruefungsordnung",
    ),
    "SWS": (
        "semesterwochenstunden",
    ),
}
PROTECTED_QUERY_TERMS = frozenset(TERM_ALIAS_HINTS)

TERM_LIKE_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{2,15}\b")
IGNORED_TERM_TOKENS = {
    "was",
    "wie",
    "wer",
    "wen",
    "wo",
    "wann",
    "warum",
    "bedeutet",
    "ist",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "und",
    "fuer",
    "für",
}
FACULTY_URL_HINTS = (
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

# ---------------------------------------------------------------------------
# Embedding-Hilfsfunktionen
# ---------------------------------------------------------------------------


def embed_query_dense(embedder: TextEmbedding, query: str) -> list[float]:
    """Dense-Embedding für eine Suchanfrage (task=retrieval.query)."""
    return list(embedder.embed([query], task="retrieval.query"))[0].tolist()


def embed_query_sparse(embedder: SparseTextEmbedding, query: str) -> models.SparseVector:
    """BM25-Sparse-Embedding für eine Suchanfrage."""
    result = list(embedder.embed([query]))[0]
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def _embed_queries_dense(embedder: TextEmbedding, queries: list[str]) -> list[list[float]]:
    """Dense-Embeddings für mehrere Suchanfragen in einem Batch."""
    return [vector.tolist() for vector in embedder.embed(queries, task="retrieval.query")]


def _embed_queries_sparse(embedder: SparseTextEmbedding, queries: list[str]) -> list[models.SparseVector]:
    """BM25-Sparse-Embeddings für mehrere Suchanfragen in einem Batch."""
    vectors = []
    for result in embedder.embed(queries):
        vectors.append(
            models.SparseVector(
                indices=result.indices.tolist(),
                values=result.values.tolist(),
            )
        )
    return vectors


def _normalize_for_match(text: str) -> str:
    return (
        text.casefold()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _is_term_like(token: str) -> bool:
    if len(token) < 3:
        return False
    normalized = _normalize_for_match(token)
    if normalized in IGNORED_TERM_TOKENS:
        return False
    if token.upper() in PROTECTED_QUERY_TERMS:
        return True
    if any(ch.isdigit() for ch in token):
        return True

    upper_count = sum(ch.isupper() for ch in token)
    lower_count = sum(ch.islower() for ch in token)
    return upper_count >= 2 or (upper_count >= 1 and lower_count >= 1 and not token[0].isupper())


def _should_attempt_term_correction(token: str) -> bool:
    if not _is_term_like(token):
        return False
    if token.upper() in PROTECTED_QUERY_TERMS:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    return True


def _extract_query_term_candidates(query: str) -> list[str]:
    candidates = []
    for token in TERM_LIKE_PATTERN.findall(query):
        if _is_term_like(token):
            candidates.append(token)
    return list(dict.fromkeys(candidates))


def _extract_context_terms(results: list[dict], *, limit: int = 5) -> dict[str, int]:
    scores: dict[str, int] = {}
    for result in results[:limit]:
        payload = result.get("payload") or {}
        fields = [
            payload.get("glossary_term", ""),
            payload.get("title", ""),
            payload.get("section_heading", ""),
            payload.get("text", "")[:500],
        ]
        for field_index, text in enumerate(fields):
            if not text:
                continue
            for token in TERM_LIKE_PATTERN.findall(text):
                if not _is_term_like(token):
                    continue
                weight = 4 if field_index == 0 else 2 if field_index < 3 else 1
                scores[token] = scores.get(token, 0) + weight
    return scores


def _damerau_levenshtein_distance(left: str, right: str, *, max_distance: int = 2) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        row_min = max_distance + 1
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
            row_min = min(row_min, dp[i][j])
        if row_min > max_distance:
            return max_distance + 1

    return dp[-1][-1]


def _detect_term_corrections(query: str, results: list[dict]) -> list[dict[str, str]]:
    query_terms = [
        token for token in _extract_query_term_candidates(query)
        if _should_attempt_term_correction(token)
    ]
    if not query_terms:
        return []

    context_terms = _extract_context_terms(results)
    if not context_terms:
        return []

    corrections: list[dict[str, str]] = []
    for query_term in query_terms:
        normalized_query = _normalize_for_match(query_term)
        best_term = ""
        best_score = None
        best_distance = None

        for context_term, frequency in context_terms.items():
            normalized_context = _normalize_for_match(context_term)
            if normalized_context == normalized_query:
                continue
            distance = _damerau_levenshtein_distance(normalized_query, normalized_context, max_distance=1)
            if distance > 1:
                continue

            score = (-distance, frequency, -abs(len(query_term) - len(context_term)))
            if best_score is None or score > best_score:
                best_score = score
                best_distance = distance
                best_term = context_term

        if best_term and best_distance is not None:
            corrections.append({
                "asked": query_term,
                "matched": best_term,
            })

    return corrections


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------


def _fetch_neighbor_chunk(
    client: QdrantClient,
    source_url: str,
    chunk_index: int,
) -> dict | None:
    """Lädt einen einzelnen Nachbar-Chunk (chunk_index) einer source_url aus Qdrant.

    Gibt das Payload-Dict zurück oder None wenn kein Treffer.
    """
    hits, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_url",
                    match=models.MatchValue(value=source_url),
                ),
                models.FieldCondition(
                    key="chunk_index",
                    match=models.MatchValue(value=chunk_index),
                ),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return hits[0].payload if hits else None


def _scroll_source_chunks(
    client: QdrantClient,
    source_url: str,
    *,
    max_points: int = 256,
) -> list:
    """Lädt Chunks einer Quelle paginiert für lokale Zusatz-Analyse."""
    points = []
    offset = None

    while len(points) < max_points:
        batch, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_url",
                        match=models.MatchValue(value=source_url),
                    ),
                ]
            ),
            limit=min(64, max_points - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None or not batch:
            break

    return points


def _payload_search_text(payload: dict | None) -> str:
    payload = payload or {}
    return _normalize_for_match(
        " ".join(
            part
            for part in (
                payload.get("title", ""),
                payload.get("section_heading", ""),
                payload.get("text", ""),
                payload.get("source_url", ""),
                payload.get("document_kind", ""),
                payload.get("source_family", ""),
                payload.get("document_group", ""),
                payload.get("faculty", ""),
            )
            if part
        )
    )


def _infer_payload_faculties(payload: dict | None) -> list[str]:
    payload = payload or {}
    faculties: list[str] = []

    explicit = payload.get("faculty", "")
    if explicit:
        faculties.append(explicit)

    searchable_parts = [
        payload.get("source_url", ""),
        payload.get("source_family", ""),
        payload.get("document_group", ""),
    ]
    searchable_text = _normalize_for_match(" ".join(part for part in searchable_parts if part))
    for hint, faculty in FACULTY_URL_HINTS:
        if hint in searchable_text and faculty not in faculties:
            faculties.append(faculty)

    return faculties


def _definition_aliases(assessment: QueryAssessment) -> list[str]:
    aliases = []
    for term in assessment.retrieval_terms:
        key = term.upper()
        aliases.extend(TERM_ALIAS_HINTS.get(key, ()))
        normalized = _normalize_for_match(term)
        if "pruefungsform" in normalized:
            aliases.extend((
                "legende der pruefungsformen",
                "arten der pruefungsleistung",
                "pruefungsformen",
            ))
    return list(dict.fromkeys(aliases))


def _lookup_glossary_results_for_assessment(
    assessment: QueryAssessment,
    *,
    limit: int,
) -> list[dict]:
    if "definition" not in assessment.intents:
        return []

    candidate_phrases = [*assessment.retrieval_terms, *_definition_aliases(assessment)]
    return lookup_glossary_results(
        terms=assessment.retrieval_terms,
        candidate_phrases=candidate_phrases,
        limit=limit,
    )


def _payload_boost(assessment: QueryAssessment, payload: dict | None) -> float:
    """Vergibt kleine Zusatz-Boni für Definitionen, Workflows und Kontakte."""
    text = _payload_search_text(payload)
    if not text:
        return 0.0

    boost = 0.0
    term_hits = 0
    for term in assessment.retrieval_terms:
        normalized = _normalize_for_match(term)
        if normalized and normalized in text:
            term_hits += 1

    if "definition" in assessment.intents:
        boost += 0.6 * term_hits
        boost += 0.5 * sum(marker in text for marker in DEFINITION_BOOST_MARKERS)
        boost += 0.8 * sum(alias in text for alias in _definition_aliases(assessment))

    if "workflow" in assessment.intents:
        boost += 0.35 * sum(marker in text for marker in WORKFLOW_BOOST_MARKERS)
    if "exam_registration" in assessment.intents:
        boost += 0.7 * sum(marker in text for marker in EXAM_REGISTRATION_MARKERS)
        if "online-antraege und -bewerbungen" in text or "bewerbungsportal" in text:
            boost -= 1.2
        if "anleitung" in text and (
            "pruefungsanmeldung" in text or "an-und abmeldung zu pruefungen" in text
        ):
            boost += 1.1
        if "studierende" in text and ("campusmanagement" in text or "icms" in text):
            boost += 0.7

    if "contact" in assessment.intents:
        boost += 0.45 * sum(marker in text for marker in CONTACT_BOOST_MARKERS)
        if "@" in text or "sprechstunden" in text:
            boost += 0.5
        normalized_query = _normalize_for_match(assessment.original_query)
        if "bewerb" in normalized_query:
            if (
                "bewerb" in text
                or "studieninteressierte" in text
                or "studierendenservice" in text
                or "akademische angelegenheiten" in text
            ):
                boost += 1.3
            else:
                boost -= 0.8
        if any(
            marker in text
            for marker in (
                "karriere.hs-hannover.de",
                "professur",
                "ausschreibung",
                "kennziffer",
                "berufung",
                "anstellungsart",
            )
        ):
            boost -= 4.0

    payload_faculties = _infer_payload_faculties(payload)
    if assessment.detected_faculties and payload_faculties:
        if any(faculty in assessment.detected_faculties for faculty in payload_faculties):
            boost += 1.1
            if "exam_registration" in assessment.intents:
                boost += 1.4
        else:
            boost -= 0.6
            if "exam_registration" in assessment.intents:
                boost -= 1.1

    return boost


def _boost_results_for_assessment(
    results: list[dict],
    assessment: QueryAssessment,
) -> list[dict]:
    """Hebt Treffer mit passenden Marker-Texten leicht an."""
    boosted = []
    for result in results:
        bonus = _payload_boost(assessment, result.get("payload"))
        boosted.append({**result, "score": result["score"] + bonus})
    return sorted(boosted, key=lambda item: item["score"], reverse=True)


def _merge_extra_results(
    results: list[dict],
    extras: list[dict],
    *,
    limit: int,
) -> list[dict]:
    if not extras:
        return results

    merged = list(results)
    seen_ids = {str(result["id"]) for result in results}
    seen_payloads = {
        (
            (result.get("payload") or {}).get("source_url", ""),
            (result.get("payload") or {}).get("section_heading", ""),
            (result.get("payload") or {}).get("text", "")[:220],
        )
        for result in results
    }

    for extra in extras:
        payload = extra.get("payload") or {}
        payload_key = (
            payload.get("source_url", ""),
            payload.get("section_heading", ""),
            payload.get("text", "")[:220],
        )
        if str(extra["id"]) in seen_ids or payload_key in seen_payloads:
            continue
        seen_ids.add(str(extra["id"]))
        seen_payloads.add(payload_key)
        merged.append(extra)

    merged.sort(key=lambda item: item["score"], reverse=True)
    return merged[:limit]


def _build_similar_term_results(hints: list[dict[str, str]]) -> list[dict]:
    results = []
    for hint in hints:
        matched = hint.get("matched", "")
        snippet = hint.get("snippet", "")
        if not matched or not snippet:
            continue
        results.append({
            "id": f"termhint::{hint.get('asked', '')}->{matched}::{hint.get('source_url', '')}",
            "score": 7.4,
            "payload": {
                "title": hint.get("title", ""),
                "source_url": hint.get("source_url", ""),
                "section_heading": hint.get("section_heading", "") or "Terminologie",
                "faculty": hint.get("faculty", ""),
                "crawl_date": hint.get("crawl_date", ""),
                "document_kind": "term_hint",
                "text": f"{matched}\n\n{snippet}",
                "glossary_term": matched,
                "glossary_meaning": "",
            },
        })
    return results


def _definition_rescue_score(
    assessment: QueryAssessment,
    payload: dict | None,
) -> float:
    text = _payload_search_text(payload)
    if not text:
        return 0.0

    term_hits = 0
    for term in assessment.retrieval_terms:
        normalized = _normalize_for_match(term)
        if normalized and normalized in text:
            term_hits += 1

    if not term_hits:
        return 0.0

    marker_hits = sum(marker in text for marker in DEFINITION_BOOST_MARKERS)
    alias_hits = sum(alias in text for alias in _definition_aliases(assessment))
    return (1.4 * term_hits) + (0.7 * marker_hits) + (1.0 * alias_hits)


def _rescue_definition_chunks(
    client: QdrantClient,
    results: list[dict],
    assessment: QueryAssessment,
    *,
    limit: int,
) -> list[dict]:
    """Sucht definierende Chunks innerhalb bereits relevanter Dokumente."""
    if "definition" not in assessment.intents or not assessment.retrieval_terms:
        return results

    seen_ids = {str(result["id"]) for result in results}
    candidate_urls = []
    for result in results:
        source_url = (result.get("payload") or {}).get("source_url", "")
        if source_url and source_url not in candidate_urls:
            candidate_urls.append(source_url)
        if len(candidate_urls) == 4:
            break

    rescued = []
    for source_url in candidate_urls:
        best_point = None
        best_score = 0.0
        for point in _scroll_source_chunks(client, source_url):
            if str(point.id) in seen_ids:
                continue
            score = _definition_rescue_score(assessment, point.payload)
            if score > best_score:
                best_score = score
                best_point = point

        if best_point and best_score >= 2.8:
            rescued.append({
                "id": best_point.id,
                "score": best_score,
                "payload": best_point.payload,
            })

    if not rescued:
        return results

    merged = results + rescued
    merged = _boost_results_for_assessment(merged, assessment)
    return merged[:limit]


def create_reranker(*, cache_dir: str | None = None):
    """Erzeugt den Cross-Encoder-Reranker mit Import-Fallback."""
    if not USE_RERANKER:
        return None
    try:
        try:
            from fastembed import TextCrossEncoder
        except ImportError:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        kwargs = {"model_name": RERANKER_MODEL}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        return TextCrossEncoder(**kwargs)
    except Exception:
        raise


def _rerank_results(query: str, results: list[dict], reranker=None) -> list[dict]:
    """Sortiert Treffer optional mit dem Cross-Encoder neu."""
    if reranker is None or not results:
        return results
    passages = [r["payload"].get("text", "") for r in results]
    scores = list(reranker.rerank(query, passages))
    return [r for _, r in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]


def _augment_results(client: QdrantClient, results: list[dict]) -> list[dict]:
    """Lädt Nachbar-Chunks für die besten Treffer nach."""
    for i, result in enumerate(results):
        if i >= AUGMENT_TOP_N:
            break

        payload = result["payload"] or {}
        source_url = payload.get("source_url", "")
        chunk_index = payload.get("chunk_index")
        total = payload.get("total_chunks", 0)

        if source_url and chunk_index is not None:
            prev_text = ""
            next_text = ""

            if chunk_index > 0:
                prev = _fetch_neighbor_chunk(client, source_url, chunk_index - 1)
                if prev:
                    prev_text = prev.get("text", "")

            if chunk_index < total - 1:
                nxt = _fetch_neighbor_chunk(client, source_url, chunk_index + 1)
                if nxt:
                    next_text = nxt.get("text", "")

            core_text = payload.get("text", "")
            parts = []
            if prev_text:
                parts.append(prev_text)
            parts.append(core_text)
            if next_text:
                parts.append(next_text)
            result["payload"] = {**payload, "text": "\n\n".join(parts)}

    return results


def _query_points_with_vectors(
    client: QdrantClient,
    dense_vec: list[float],
    sparse_vec: models.SparseVector,
    *,
    top_k: int,
    query_filter: models.Filter | None = None,
) -> list[dict]:
    """Führt einen einzigen Qdrant-Hybrid-Query mit vorberechneten Vektoren aus."""
    raw_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=CANDIDATE_LIMIT,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=CANDIDATE_LIMIT,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k * DEDUP_BUFFER,
        with_payload=True,
    ).points

    raw_results = [
        hit for hit in raw_results
        if not (hit.payload or {}).get("source_url", "").startswith(BLOCKED_URL_PREFIXES)
    ]

    url_counts: dict[str, int] = {}
    results: list[dict] = []

    for hit in raw_results:
        source_url = (hit.payload or {}).get("source_url", str(hit.id))
        count = url_counts.get(source_url, 0)
        if count < MAX_PER_URL:
            url_counts[source_url] = count + 1
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            })
        if len(results) == top_k:
            break

    return results


def perform_hybrid_search(
    client: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    query: str,
    top_k: int = TOP_K,
    query_filter: models.Filter | None = None,
    reranker=None,
    augment_results: bool = True,
) -> list[dict]:
    """Hybrid-Suche via Qdrant-nativem Prefetch + RRF-Fusion.

    Beide Sucharme (Dense + Sparse) laufen in einem einzigen Qdrant-Aufruf.
    Die RRF-Fusion erfolgt serverseitig — kein Python-RRF-Code nötig.
    Nach der Fusion:
      1. URL-Dedup: max. MAX_PER_URL Chunks pro source_url.
      2. Semantisches Reranking via Cross-Encoder (wenn reranker übergeben).
      3. Context Augmentation: für die AUGMENT_TOP_N besten Treffer werden
         Nachbar-Chunks (chunk_index ± 1) nachgeladen und dem Payload angehängt.
    """
    dense_vec  = embed_query_dense(dense_embedder, query)
    sparse_vec = embed_query_sparse(sparse_embedder, query)
    results = _query_points_with_vectors(
        client,
        dense_vec,
        sparse_vec,
        top_k=top_k,
        query_filter=query_filter,
    )

    results = _rerank_results(query, results, reranker=reranker)
    if augment_results:
        results = _augment_results(client, results)
    return results


def _fuse_variant_results(query_runs: list[tuple[str, list[dict]]], *, limit: int) -> list[dict]:
    """Führt mehrere Resultatlisten per lokalem Reciprocal Rank Fusion zusammen."""
    fused: dict[str, dict] = {}
    for variant, results in query_runs:
        for rank, result in enumerate(results, start=1):
            result_id = str(result["id"])
            bonus = 1.0 / (60 + rank)
            existing = fused.get(result_id)
            if existing is None:
                fused[result_id] = {
                    **result,
                    "score": bonus,
                    "matched_queries": [variant],
                }
            else:
                existing["score"] += bonus
                if variant not in existing["matched_queries"]:
                    existing["matched_queries"].append(variant)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def perform_guided_hybrid_search(
    client: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    query: str,
    top_k: int = TOP_K,
    query_filter: models.Filter | None = None,
    reranker=None,
    selected_query: str | None = None,
    clarification: str = "",
    planner_guidance: dict | None = None,
    query_assist_enabled: bool = True,
) -> tuple[list[dict], dict]:
    """Führt Hybrid-Suche mit Query-Bewertung, Varianten und Prozess-Trace aus."""
    assessment = assess_query(query) if query_assist_enabled else build_plain_query_assessment(query)
    if query_assist_enabled:
        pre_retrieval_term_hints = find_similar_corpus_terms(
            _extract_query_term_candidates(query),
            limit=3,
        )
        similar_term_results = _build_similar_term_results(pre_retrieval_term_hints)
        glossary_results = _lookup_glossary_results_for_assessment(
            assessment,
            limit=max(3, top_k),
        )
        retrieval_queries = build_retrieval_queries(
            assessment,
            selected_query=selected_query,
            clarification=clarification,
        )
    else:
        pre_retrieval_term_hints = []
        similar_term_results = []
        glossary_results = []
        primary = f"{(selected_query or query).strip()} {clarification.strip()}".strip()
        retrieval_queries = [primary or query]
    if planner_guidance:
        normalized_question = " ".join(
            str(planner_guidance.get("normalized_question") or "").split()
        ).strip()
        if normalized_question and normalized_question.casefold() != query.casefold():
            retrieval_queries.insert(0, normalized_question)
        retrieval_queries.extend(
            variant
            for variant in planner_guidance.get("query_variants", [])
            if isinstance(variant, str) and variant.strip()
        )
        retrieval_queries.extend(
            f"{term} Hochschule Hannover"
            for term in planner_guidance.get("canonical_hsh_terms", [])
            if isinstance(term, str) and term.strip()
        )
    for hint in pre_retrieval_term_hints:
        matched = hint.get("matched", "")
        if matched:
            retrieval_queries.extend([
                f"{matched} Hochschule Hannover",
                f"{matched} Bedeutung Hochschule Hannover",
                (selected_query or query).replace(hint.get("asked", ""), matched) if hint.get("asked") else matched,
            ])
    retrieval_queries = list(dict.fromkeys(q for q in retrieval_queries if q))[:10]
    dense_vectors = _embed_queries_dense(dense_embedder, retrieval_queries)
    sparse_vectors = _embed_queries_sparse(sparse_embedder, retrieval_queries)

    if len(retrieval_queries) == 1:
        results = _query_points_with_vectors(
            client,
            dense_vectors[0],
            sparse_vectors[0],
            top_k=top_k,
            query_filter=query_filter,
        )
        results = _boost_results_for_assessment(results, assessment)
        results = _rescue_definition_chunks(
            client,
            results,
            assessment,
            limit=top_k * DEDUP_BUFFER,
        )
        results = _merge_extra_results(
            results,
            glossary_results,
            limit=top_k * DEDUP_BUFFER,
        )
        results = _merge_extra_results(
            results,
            similar_term_results,
            limit=top_k * DEDUP_BUFFER,
        )
        results = _rerank_results(
            selected_query or query,
            results,
            reranker=reranker,
        )
        results = results[:top_k]
        results = _augment_results(client, results)
    else:
        query_runs: list[tuple[str, list[dict]]] = []
        for retrieval_query, dense_vec, sparse_vec in zip(
            retrieval_queries,
            dense_vectors,
            sparse_vectors,
        ):
            partial = _query_points_with_vectors(
                client,
                dense_vec,
                sparse_vec,
                top_k=top_k,
                query_filter=query_filter,
            )
            query_runs.append((retrieval_query, partial))

        results = _fuse_variant_results(query_runs, limit=top_k * DEDUP_BUFFER)
        results = _boost_results_for_assessment(results, assessment)
        results = _rescue_definition_chunks(
            client,
            results,
            assessment,
            limit=top_k * DEDUP_BUFFER,
        )
        results = _merge_extra_results(
            results,
            glossary_results,
            limit=top_k * DEDUP_BUFFER,
        )
        results = _merge_extra_results(
            results,
            similar_term_results,
            limit=top_k * DEDUP_BUFFER,
        )
        results = _rerank_results(selected_query or query, results, reranker=reranker)
        results = results[:top_k]
        results = _augment_results(client, results)

    glossary_hits = []
    for entry in glossary_results[:3]:
        payload = entry.get("payload") or {}
        label = payload.get("glossary_term") or payload.get("section_heading") or payload.get("title")
        source = payload.get("title") or payload.get("source_url", "")
        if label and source:
            glossary_hits.append(f"{label} -> {source}")
        elif source:
            glossary_hits.append(source)

    term_corrections = _detect_term_corrections(query, results)
    if not term_corrections:
        term_corrections = [
            {"asked": item.get("asked", ""), "matched": item.get("matched", "")}
            for item in pre_retrieval_term_hints
            if item.get("asked") and item.get("matched")
        ]

    process_trace = {
        "original_query": query,
        "selected_query": selected_query or query,
        "clarification": clarification.strip(),
        "assessment_reasons": assessment.reasons,
        "assessment_intents": assessment.intents,
        "specificity": assessment.specificity,
        "clarification_prompt": assessment.clarification_prompt,
        "clarification_options": assessment.clarification_options,
        "planner_guidance": planner_guidance or {},
        "suggestions": assessment.suggestions,
        "retrieval_queries": retrieval_queries,
        "detected_codes": assessment.detected_codes,
        "detected_abbreviations": assessment.detected_abbreviations,
        "detected_faculties": assessment.detected_faculties,
        "clarification_needed": assessment.clarification_needed,
        "glossary_hits": glossary_hits,
        "term_corrections": term_corrections,
        "query_assist_enabled": query_assist_enabled,
    }
    return results, process_trace


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------


def build_rag_context(results: list[dict]) -> str:
    """Wandelt Suchergebnisse in einen formatierten Kontext-String für das LLM um."""
    blocks = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        lines = [
            f"[Quelle {i}] {p.get('title', '(kein Titel)')}",
            f"URL: {p.get('source_url', '—')}",
            f"Stand: {p.get('crawl_date', 'unbekannt')}",
        ]
        faculties = _infer_payload_faculties(p)
        if faculties:
            lines.append(f"Fakultät: {faculties[0]}")
        if p.get("section_heading"):
            lines.append(f"Abschnitt: {p['section_heading']}")
        lines.append("")
        lines.append(p.get("text", ""))
        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(results: list[dict]) -> None:
    """Pretty-print ranked results to the terminal."""
    if not results:
        print("  Keine Treffer gefunden.")
        return

    for i, r in enumerate(results, 1):
        p       = r["payload"]
        score   = r["score"]
        url     = p.get("source_url", "—")
        faculty = p.get("faculty", "")
        heading = p.get("section_heading", "")
        date    = p.get("crawl_date", "")
        preview = p.get("text", "")[:200].replace("\n", " ")

        print(f"\n  ┌─ Treffer {i}  (RRF-Score: {score:.5f})")
        print(f"  │  URL      : {url}")
        if faculty:
            print(f"  │  Fakultät : {faculty}")
        if heading:
            print(f"  │  Abschnitt: {heading}")
        if date:
            print(f"  │  Stand    : {date}")
        print(f"  │  Vorschau : {preview}…")
        print(f"  └{'─' * 62}")


def print_process_trace(process_trace: dict) -> None:
    """Zeigt, wie die Suchanfrage intern vorbereitet wurde."""
    print("\n[Systemprozess]")
    original = process_trace.get("original_query", "")
    selected = process_trace.get("selected_query", "")
    clarification = process_trace.get("clarification", "")
    reasons = process_trace.get("assessment_reasons", [])
    intents = process_trace.get("assessment_intents", [])
    specificity = process_trace.get("specificity", "")
    clarification_prompt = process_trace.get("clarification_prompt", "")
    clarification_options = process_trace.get("clarification_options", [])
    planner_guidance = process_trace.get("planner_guidance", {})
    retrieval_queries = process_trace.get("retrieval_queries", [])
    detected_codes = process_trace.get("detected_codes", [])
    detected_abbreviations = process_trace.get("detected_abbreviations", [])
    detected_faculties = process_trace.get("detected_faculties", [])
    glossary_hits = process_trace.get("glossary_hits", [])
    term_corrections = process_trace.get("term_corrections", [])

    print(f"  Original   : {original}")
    if selected and selected != original:
        print(f"  Auswahl    : {selected}")
    if clarification:
        print(f"  Zusatz     : {clarification}")
    if specificity:
        print(f"  Schärfe    : {specificity}")
    if intents:
        print(f"  Typen      : {', '.join(intents)}")
    if detected_codes:
        print(f"  Codes      : {', '.join(detected_codes)}")
    if detected_abbreviations:
        print(f"  Kürzel     : {', '.join(detected_abbreviations)}")
    if detected_faculties:
        print(f"  Fakultät   : {', '.join(detected_faculties)}")
    if glossary_hits:
        print("  Glossar    :")
        for hit in glossary_hits:
            print(f"    - {hit}")
    if term_corrections:
        print("  Schreibweise:")
        for correction in term_corrections:
            print(f"    - {correction.get('asked', '')} -> {correction.get('matched', '')}")
    if reasons:
        print("  Analyse    :")
        for reason in reasons:
            print(f"    - {reason}")
    if clarification_prompt:
        print(f"  Präzisieren: {clarification_prompt}")
    if clarification_options:
        print("  Richtungen :")
        for option in clarification_options:
            print(f"    - {option}")
    if planner_guidance:
        print("  LLM-Planer :")
        normalized = planner_guidance.get("normalized_question", "")
        if normalized:
            print(f"    - Normalisiert: {normalized}")
        for term in planner_guidance.get("canonical_hsh_terms", []) or []:
            print(f"    - HsH-Begriff: {term}")
        for hint in planner_guidance.get("source_type_hints", []) or []:
            print(f"    - Quelltyp: {hint}")
    print("  Suchpfade  :")
    for idx, retrieval_query in enumerate(retrieval_queries, start=1):
        print(f"    {idx}. {retrieval_query}")


# ---------------------------------------------------------------------------
# Main / interactive CLI
# ---------------------------------------------------------------------------


def main() -> None:

    # ── Connect ───────────────────────────────────────────────────────────
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=10)
        client.get_collections()
    except Exception as exc:
        print(f"Fehler: Qdrant nicht erreichbar — {exc}")
        print("  → docker compose up -d")
        sys.exit(1)

    # ── Embedding-Modelle laden ───────────────────────────────────────────
    print(f"\nLade Dense-Modell '{DENSE_MODEL}'…")
    dense_embedder = TextEmbedding(model_name=DENSE_MODEL)

    print(f"Lade Sparse-Modell '{SPARSE_MODEL}'…")
    sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL)

    reranker = None
    if USE_RERANKER:
        try:
            print(f"Lade Reranker '{RERANKER_MODEL}'…")
            reranker = create_reranker()
        except Exception as exc:
            print(f"Reranker nicht verfügbar — weiter ohne Reranking: {exc}")

    print("Bereit.  Tippe eine Frage, Enter zum Suchen, Strg+C zum Beenden.\n")

    # ── Interactive loop ──────────────────────────────────────────────────
    while True:
        try:
            query = input("Frage> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAuf Wiedersehen.")
            break

        if not query:
            continue

        results, process_trace = perform_guided_hybrid_search(
            client,
            dense_embedder,
            sparse_embedder,
            query,
            reranker=reranker,
        )
        print_process_trace(process_trace)
        print_results(results)


if __name__ == "__main__":
    main()
