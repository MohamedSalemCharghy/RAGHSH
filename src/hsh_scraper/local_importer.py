"""
Lokaler Qdrant-Importer — Parquet-Datei in Qdrant-Datenbank laden.

Kurzbeschreibung
----------------
Liest die von hpc_vectorizer.py erzeugte Parquet-Datei (Dense + BM25 Sparse
Vektoren) und lädt die Vektoren samt Metadaten per Upsert in die lokale
Qdrant-Instanz.

Workflow (aktuell — HPC berechnet beide Vektortypen):
    1. HPC:   python -m hsh_scraper.hpc_vectorizer   →  artifacts/hsh_vectors.parquet
    2. Lokal: python -m hsh_scraper.local_importer   →  Qdrant befüllen

Fallback (ältere Parquet ohne Sparse-Spalten):
    1. Lokal: python enrich_sparse.py    →  artifacts/hsh_vectors_enriched.parquet
    2. Lokal: PARQUET_FILE anpassen, dann python -m hsh_scraper.local_importer

Collection-Schema:
    Benannte Vektoren (erforderlich für Prefetch/Hybrid-Search):
      "dense"  — 1024-dim Jina-Embeddings (Cosine)
      "sparse" — BM25 Sparse Vectors (Qdrant/bm25)

Verhalten bei erneutem Aufruf:
    Punkt-IDs sind deterministisch (UUID5 aus source_url + chunk_index).
    Ein erneuter Lauf überschreibt vorhandene Einträge (Upsert), erzeugt
    keine Duplikate.

Konfiguration:
    PARQUET_FILE    — Pfad zur angereicherten Eingabedatei
    QDRANT_URL      — Adresse der lokalen Qdrant-Instanz
    COLLECTION_NAME — Name der Collection
    UPLOAD_BATCH    — Punkte pro upsert-Aufruf
    RESUME_FROM_ROW — Neustart ab Zeile N (0 = von Anfang an)

Abhängigkeiten (lokales VirtualEnv):
    pip install qdrant-client pyarrow
"""

import logging
import sys

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

try:
    from .paths import ARTIFACTS_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from paths import ARTIFACTS_DIR

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_ENRICHED = ARTIFACTS_DIR / "hsh_vectors_enriched.parquet"
_PLAIN    = ARTIFACTS_DIR / "hsh_vectors.parquet"
PARQUET_FILE = _ENRICHED if _ENRICHED.exists() else _PLAIN
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "hsh_knowledge"
VECTOR_SIZE     = 1024

UPLOAD_BATCH    = 256  # Punkte pro upsert-Aufruf

# Neustart nach Unterbrechung: 0-basierter Zeilenindex.
# 0 = von Anfang an. Bei Abbruch bei Batch N: RESUME_FROM_ROW = N * UPLOAD_BATCH
RESUME_FROM_ROW = 0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_collection(client: QdrantClient) -> None:
    """Erstellt die Collection und Payload-Indizes falls noch nicht vorhanden.

    Schema: benannte Dense- und Sparse-Vektoren für Prefetch-basierte Hybrid-Suche.
    Payload-Index auf 'source_url' und 'chunk_index' für Context Augmentation.
    Payload-Index auf 'faculty' für Fakultätsfilter (ohne Full-Scan).
    Volltext-Index auf 'text' mit MULTILINGUAL-Tokenizer (deutsches Stemming,
    Kompositaaufspaltung).
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )
        logger.info("Collection '%s' angelegt (dense=%ddim, sparse=BM25)",
                    COLLECTION_NAME, VECTOR_SIZE)

        # Volltext-Index auf 'text' — MULTILINGUAL für deutsches Stemming
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="text",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=50,
                lowercase=True,
            ),
        )
        logger.info("Volltext-Index auf 'text' angelegt (MULTILINGUAL-Tokenizer)")

        # Keyword-Index auf 'faculty' für schnelle Filterung
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="faculty",
            field_schema=models.KeywordIndexParams(type="keyword"),
        )

        # Integer-Index auf 'chunk_index' für Context Augmentation
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="chunk_index",
            field_schema=models.IntegerIndexParams(type="integer", lookup=True, range=False),
        )

        # Keyword-Index auf 'source_url' für Context Augmentation
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source_url",
            field_schema=models.KeywordIndexParams(type="keyword"),
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="document_kind",
            field_schema=models.KeywordIndexParams(type="keyword"),
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="document_group",
            field_schema=models.KeywordIndexParams(type="keyword"),
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="language",
            field_schema=models.KeywordIndexParams(type="keyword"),
        )
        logger.info(
            "Payload-Indizes angelegt: faculty, chunk_index, source_url, document_kind, document_group, language"
        )
    else:
        logger.info("Collection '%s' existiert bereits — Upsert-Modus", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── 1. Parquet-Datei prüfen ───────────────────────────────────────────
    if not PARQUET_FILE.exists():
        logger.error("Datei nicht gefunden: %s", PARQUET_FILE)
        logger.error("Bitte zuerst hpc_vectorizer.py auf dem HPC ausführen und die Datei übertragen.")
        logger.error("Für ältere Parquet-Dateien ohne Sparse-Spalten: enrich_sparse.py ausführen")
        logger.error("  und PARQUET_FILE auf artifacts/hsh_vectors_enriched.parquet setzen.")
        sys.exit(1)

    pf         = pq.ParquetFile(PARQUET_FILE)
    total_rows = pf.metadata.num_rows
    size_mb    = PARQUET_FILE.stat().st_size / 1_048_576
    logger.info("Parquet-Datei: %s  (%.1f MB, %d Zeilen)", PARQUET_FILE, size_mb, total_rows)

    # Schema-Prüfung: sparse-Spalten vorhanden?
    schema_names = set(pf.schema_arrow.names)
    required = {"id", "vector", "text", "sparse_indices", "sparse_values"}
    missing = required - schema_names
    if missing:
        logger.error("Fehlende Spalten in Parquet: %s", missing)
        logger.error("Bitte enrich_sparse.py ausführen, um sparse_indices/sparse_values hinzuzufügen.")
        sys.exit(1)

    # ── 2. Qdrant verbinden ───────────────────────────────────────────────
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=30)
        client.get_collections()
        logger.info("Verbunden mit Qdrant: %s", QDRANT_URL)
    except Exception as exc:
        logger.error("Verbindung zu Qdrant fehlgeschlagen: %s — %s", QDRANT_URL, exc)
        logger.error("Docker-Container starten:  docker compose up -d")
        sys.exit(1)

    ensure_collection(client)

    # ── 3. Row-Group-weise einlesen und hochladen ─────────────────────────
    num_row_groups = pf.metadata.num_row_groups
    logger.info("Parquet enthält %d Row-Group(s)", num_row_groups)

    if RESUME_FROM_ROW > 0:
        logger.info("Überspringe die ersten %d Zeilen (RESUME_FROM_ROW=%d)",
                    RESUME_FROM_ROW, RESUME_FROM_ROW)

    uploaded     = 0
    skipped_rows = 0
    row_cursor   = 0

    for rg_idx in range(num_row_groups):
        # Row-Group lesen — bei Beschädigung überspringen statt abbrechen
        try:
            table = pf.read_row_group(rg_idx)
        except Exception as exc:
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            logger.warning("Row-Group %d/%d beschädigt — übersprungen (%d Zeilen): %s",
                           rg_idx + 1, num_row_groups, rg_rows, exc)
            skipped_rows += rg_rows
            row_cursor   += rg_rows
            continue

        # RESUME: Zeilen vor RESUME_FROM_ROW überspringen
        rg_start   = row_cursor
        rg_end     = row_cursor + len(table)
        row_cursor = rg_end

        if rg_end <= RESUME_FROM_ROW:
            continue
        if rg_start < RESUME_FROM_ROW:
            table = table.slice(RESUME_FROM_ROW - rg_start)

        # Row-Daten als Python-Listen extrahieren
        batch_dict = table.to_pydict()

        ids              = batch_dict["id"]
        dense_vectors    = batch_dict["vector"]
        sparse_indices   = batch_dict["sparse_indices"]
        sparse_values    = batch_dict["sparse_values"]
        source_urls      = batch_dict["source_url"]
        titles           = batch_dict["title"]
        crawl_dates      = batch_dict["crawl_date"]
        content_types    = batch_dict["content_type"]
        faculties        = batch_dict["faculty"]
        languages        = batch_dict.get("language", [""] * len(ids))
        quality_scores   = batch_dict.get("quality_score", [""] * len(ids))
        document_kinds   = batch_dict.get("document_kind", [""] * len(ids))
        source_families  = batch_dict.get("source_family", [""] * len(ids))
        document_groups  = batch_dict.get("document_group", [""] * len(ids))
        topic_tags       = batch_dict.get("topic_tags", [""] * len(ids))
        section_headings = batch_dict["section_heading"]
        texts            = batch_dict["text"]
        chunk_indices    = batch_dict["chunk_index"]
        total_chunks_col = batch_dict["total_chunks"]

        # Row-Group in UPLOAD_BATCH-große Häppchen aufteilen
        for offset in range(0, len(ids), UPLOAD_BATCH):
            end = offset + UPLOAD_BATCH

            points = [
                models.PointStruct(
                    id=ids[i],
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": models.SparseVector(
                            indices=sparse_indices[i],
                            values=sparse_values[i],
                        ),
                    },
                    payload={
                        "source_url":      source_urls[i],
                        "title":           titles[i],
                        "crawl_date":      crawl_dates[i],
                        "content_type":    content_types[i],
                        "faculty":         faculties[i],
                        "language":        languages[i],
                        "quality_score":   quality_scores[i],
                        "document_kind":   document_kinds[i],
                        "source_family":   source_families[i],
                        "document_group":  document_groups[i],
                        "topic_tags":      topic_tags[i],
                        "section_heading": section_headings[i],
                        "text":            texts[i],
                        "chunk_index":     chunk_indices[i],
                        "total_chunks":    total_chunks_col[i],
                    },
                )
                for i in range(offset, min(end, len(ids)))
            ]

            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
            except Exception as exc:
                abs_row = row_cursor - len(ids) + offset
                logger.error("Upsert fehlgeschlagen bei Zeile %d: %s", abs_row, exc)
                logger.error("Zum Fortsetzen: RESUME_FROM_ROW = %d", abs_row)
                sys.exit(1)

            uploaded += len(points)
            del points

        del table, batch_dict

        pct = 100.0 * (uploaded + skipped_rows) / max(total_rows, 1)
        logger.info("  Row-Group %d/%d  [%5.1f%%]  %d Punkte hochgeladen (gesamt)",
                    rg_idx + 1, num_row_groups, pct, uploaded)

    logger.info("─" * 60)
    logger.info("Fertig. %d Punkte hochgeladen, %d Zeilen übersprungen (beschädigt).",
                uploaded, skipped_rows)
    logger.info("Collection: %s", COLLECTION_NAME)
    logger.info("─" * 60)


if __name__ == "__main__":
    main()
