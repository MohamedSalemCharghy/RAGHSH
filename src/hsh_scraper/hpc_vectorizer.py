"""
HPC-Vektorisierung — Markdown-Dateien vektorisieren und als Parquet exportieren.

Kurzbeschreibung
----------------
Dieses Skript ist für den Einsatz auf einem HPC-Cluster (z.B. GWDG/KISSKI mit
NVIDIA A100/H100) vorgesehen. Es hat keine Qdrant-Abhängigkeit. Stattdessen
werden alle erzeugten Vektoren und Metadaten in eine komprimierte Parquet-Datei
geschrieben, die anschließend auf den lokalen Rechner übertragen und dort mit
local_importer.py in Qdrant geladen werden kann.

Workflow:
    1. HPC: python -m hsh_scraper.hpc_vectorizer   →  artifacts/hsh_vectors.parquet
    2. Transfer: scp artifacts/hsh_vectors.parquet user@localhost:~/RAGHSH/artifacts/
    3. Lokal:  python -m hsh_scraper.local_importer

Speicherverwaltung:
    Die PyArrow-ParquetWriter-API wird genutzt, um dateiweise Row-Groups zu
    schreiben. Es wird nie mehr als eine Datei (ihre Chunks + Vektoren) gleichzeitig
    im RAM gehalten. Peak-Verbrauch ≈ Dense-Modell (~500 MB) + Sparse-Modell (~50 MB)
    + EMBED_BATCH_SIZE × VECTOR_SIZE × 4 Byte pro Dense-Mini-Batch.

Parquet-Schema:
    id              string          — deterministischer UUID5 (source_url + chunk_index)
    vector          list<float32>   — 1024-dimensionaler Dense-Einbettungsvektor (Jina)
    source_url      string
    title           string
    crawl_date      string
    content_type    string          — 'html' oder 'pdf'
    faculty         string
    section_heading string
    text            string          — Volltext des Chunks
    chunk_index     int32
    total_chunks    int32
    sparse_indices  list<int32>     — Token-IDs der BM25-Terme
    sparse_values   list<float32>   — BM25-Gewichte der Terme

Konfiguration:
    RESUME_FROM_FILE    — Neustart ab Dateinummer (1 = von Anfang an)
    DENSE_BATCH_SIZE    — Chunks pro Dense-Embedding-Aufruf (A100/H100: 64 empfohlen)
    SPARSE_BATCH_SIZE   — Chunks pro BM25-Aufruf (CPU-only, größere Batches OK)
    DENSE_BACKEND       — auto | transformers | fastembed
    HF_LOCAL_FILES_ONLY — 1/true/yes fuer rein lokalen Hugging-Face-Cache
    OUTPUT_FILE         — Ausgabedatei (Parquet)

Abhängigkeiten (HPC-VirtualEnv):
    pip install fastembed langchain-text-splitters pyarrow "numpy<2" "transformers==4.56.2" einops torch
"""

import gc
import logging
import os
import sys
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fastembed import SparseTextEmbedding, TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from .paths import ARTIFACTS_DIR, DATA_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from paths import ARTIFACTS_DIR, DATA_DIR

try:
    import torch
except ImportError:  # pragma: no cover - optional GPU backend
    torch = None

try:
    from transformers import AutoModel
except ImportError:  # pragma: no cover - optional GPU backend
    AutoModel = None

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

RAW_INGESTED_DIR = DATA_DIR / "ingested"
CURATED_DIR = DATA_DIR / "curated"
INGESTED_DIR = Path(
    os.getenv(
        "RAG_SOURCE_DIR",
        CURATED_DIR if CURATED_DIR.exists() and any(CURATED_DIR.glob("*.md")) else RAW_INGESTED_DIR,
    )
)
OUTPUT_FILE     = ARTIFACTS_DIR / "hsh_vectors.parquet"

DENSE_MODEL      = "jinaai/jina-embeddings-v3"
SPARSE_MODEL     = "Qdrant/bm25"
VECTOR_SIZE      = 1024          # Ausgabedimension von jina-embeddings-v3
DENSE_BACKEND    = os.getenv("DENSE_BACKEND", "auto").lower()
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "").lower() in {
    "1",
    "true",
    "yes",
}
MODEL_CACHE_DIR = Path(
    os.getenv("HF_HOME")
    or os.getenv("FASTEMBED_CACHE_PATH")
    or str(Path.home() / ".cache" / "fastembed")
).expanduser()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR = str(MODEL_CACHE_DIR)
os.environ.setdefault("FASTEMBED_CACHE_PATH", HF_CACHE_DIR)

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
DENSE_BATCH_SIZE  = 64           # H100 verarbeitet größere Batches effizient
SPARSE_BATCH_SIZE = 256          # BM25 ist CPU-only, größere Batches sind kein Problem

RESUME_FROM_FILE = 1             # 1 = von Anfang an; N = ab Datei N weitermachen

HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3")]

FACULTY_MAP = {
    "f1": "Fakultät I",
    "f2": "Fakultät II",
    "f3": "Fakultät III",
    "f4": "Fakultät IV",
    "f5": "Fakultät V",
}

# ---------------------------------------------------------------------------
# Parquet-Schema
# ---------------------------------------------------------------------------

PARQUET_SCHEMA = pa.schema([
    pa.field("id",              pa.string()),
    pa.field("vector",          pa.list_(pa.float32())),
    pa.field("source_url",      pa.string()),
    pa.field("title",           pa.string()),
    pa.field("crawl_date",      pa.string()),
    pa.field("content_type",    pa.string()),
    pa.field("faculty",         pa.string()),
    pa.field("language",        pa.string()),
    pa.field("quality_score",   pa.string()),
    pa.field("document_kind",   pa.string()),
    pa.field("source_family",   pa.string()),
    pa.field("document_group",  pa.string()),
    pa.field("topic_tags",      pa.string()),
    pa.field("section_heading", pa.string()),
    pa.field("text",            pa.string()),
    pa.field("chunk_index",     pa.int32()),
    pa.field("total_chunks",    pa.int32()),
    pa.field("sparse_indices",  pa.list_(pa.int32())),
    pa.field("sparse_values",   pa.list_(pa.float32())),
])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class DenseEmbeddingBackend:
    """Abstraktion fuer Dense-Embeddings mit optionalem GPU-Backend."""

    def embed(self, texts: list[str]) -> list:
        raise NotImplementedError


class FastEmbedDenseBackend(DenseEmbeddingBackend):
    """Fallback-Backend ueber fastembed (ONNX, haeufig CPU-lastig)."""

    def __init__(self, model_name: str, *, cache_dir: str | None) -> None:
        self._embedder = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed(self, texts: list[str]) -> list:
        return list(self._embedder.embed(texts, task="retrieval.passage"))


class TransformersDenseBackend(DenseEmbeddingBackend):
    """CUDA-faehiges Dense-Backend ueber Hugging Face / PyTorch."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None,
        local_files_only: bool,
    ) -> None:
        if AutoModel is None or torch is None:
            raise RuntimeError("transformers/torch sind nicht installiert")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA ist fuer das Transformers-Backend nicht verfuegbar")

        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        self._device = "cuda"
        self._model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self._model.to(self._device)
        self._model.eval()

    def embed(self, texts: list[str]) -> list:
        with torch.inference_mode():
            embeddings = self._model.encode(
                texts,
                task="retrieval.passage",
                show_progress_bar=False,
                device=self._device,
            )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return embeddings


def build_dense_backend(model_name: str) -> tuple[DenseEmbeddingBackend, str]:
    """Waehlt ein Dense-Embedding-Backend. Bevorzugt CUDA ueber transformers."""
    if DENSE_BACKEND in {"auto", "transformers"}:
        try:
            backend = TransformersDenseBackend(
                model_name,
                cache_dir=HF_CACHE_DIR,
                local_files_only=HF_LOCAL_FILES_ONLY,
            )
            device_label = "cuda" if torch and torch.cuda.is_available() else "cpu"
            return backend, f"transformers/{device_label}"
        except Exception as exc:
            if DENSE_BACKEND == "transformers":
                raise
            logger.warning(
                "Dense-Backend 'transformers' nicht verfuegbar (%s) — fallback auf fastembed",
                exc,
            )

    backend = FastEmbedDenseBackend(model_name, cache_dir=HF_CACHE_DIR)
    return backend, "fastembed"

# ---------------------------------------------------------------------------
# Hilfsfunktionen fuer Markdown-Parsing, Chunking und Parquet-Export
# ---------------------------------------------------------------------------


def parse_markdown_file(path: Path) -> tuple[dict, str] | None:
    """Liest YAML-Frontmatter und Body aus einer Markdown-Datei."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        logger.warning("Kein YAML-Header in %s — übersprungen", path.name)
        return None
    yaml_block = parts[1]
    body = parts[2].strip()
    if not body:
        logger.warning("Leerer Body in %s — übersprungen", path.name)
        return None
    meta: dict[str, str] = {}
    for line in yaml_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta, body


def sort_markdown_files(md_files: list[Path]) -> list[Path]:
    """Sort curated files by semantic grouping before vectorization."""
    sortable: list[tuple[tuple[str, str, str, str], Path]] = []
    for path in md_files:
        result = parse_markdown_file(path)
        if result is None:
            key = ("zz_unknown", "zz_unknown", "", path.name)
        else:
            meta, _ = result
            key = (
                meta.get("source_family", "zz_unknown"),
                meta.get("document_group", "zz_unknown"),
                meta.get("source_url", ""),
                path.name,
            )
        sortable.append((key, path))
    return [path for _, path in sorted(sortable, key=lambda item: item[0])]


def extract_faculty(url: str) -> str:
    """Ermittelt die Fakultätszugehörigkeit aus der URL.

    Prüft zuerst URL-Pfad-Segmente (/f1/ bis /f5/), dann Subdomains.
    Gibt "" zurück für zentrale Einrichtungen ohne Fakultätszugehörigkeit.
    """
    lower = url.lower()

    # Pfad-basierte Erkennung: /f1/ bis /f5/
    for code, name in FACULTY_MAP.items():
        if f"/{code}/" in lower:
            return name

    # Subdomain-basierte Erkennung
    from urllib.parse import urlparse
    netloc = urlparse(lower).netloc
    if netloc.startswith("karriere."):
        return "Karriere"
    if netloc.startswith("bibliothek."):
        return "Bibliothek"
    if netloc.startswith("international."):
        return "International"

    return ""


def chunk_document(meta: dict, body: str,
                   rec_splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Zweistufiges strukturelles Chunking (Überschriften + Größenbegrenzung)."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_sections = header_splitter.split_text(body)

    base_payload = {
        "source_url":   meta.get("source_url", ""),
        "title":        meta.get("title", ""),
        "crawl_date":   meta.get("crawl_date", ""),
        "content_type": meta.get("content_type", "html"),
        "faculty":      extract_faculty(meta.get("source_url", "")),
        "language":     meta.get("language", ""),
        "quality_score": meta.get("quality_score", ""),
        "document_kind": meta.get("document_kind", ""),
        "source_family": meta.get("source_family", ""),
        "document_group": meta.get("document_group", ""),
        "topic_tags":    meta.get("topic_tags", ""),
    }

    final_chunks: list[dict] = []
    chunk_idx = 0

    for section in header_sections:
        heading_parts = [
            section.metadata[k]
            for k in ("h1", "h2", "h3")
            if section.metadata.get(k)
        ]
        section_heading = " > ".join(heading_parts)
        text = section.page_content.strip()
        if not text:
            continue
        sub_texts = rec_splitter.split_text(text) if len(text) > CHUNK_SIZE else [text]
        for sub in sub_texts:
            final_chunks.append({
                **base_payload,
                "text":            sub,
                "section_heading": section_heading,
                "chunk_index":     chunk_idx,
            })
            chunk_idx += 1

    total = len(final_chunks)
    for c in final_chunks:
        c["total_chunks"] = total
    return final_chunks


def make_point_id(source_url: str, chunk_index: int) -> str:
    """Deterministischer UUID5 fuer stabile Upserts in Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url}#{chunk_index}"))


def chunks_to_record_batch(chunks: list[dict],
                           dense_vectors: list,
                           sparse_indices: list[list[int]],
                           sparse_values: list[list[float]]) -> pa.RecordBatch:
    """Erstellt einen PyArrow-RecordBatch aus Chunks, Dense- und Sparse-Vektoren."""
    return pa.record_batch(
        {
            "id":              [make_point_id(c["source_url"], c["chunk_index"]) for c in chunks],
            "vector":          [v.tolist() if hasattr(v, "tolist") else v for v in dense_vectors],
            "source_url":      [c["source_url"]      for c in chunks],
            "title":           [c["title"]           for c in chunks],
            "crawl_date":      [c["crawl_date"]      for c in chunks],
            "content_type":    [c["content_type"]    for c in chunks],
            "faculty":         [c["faculty"]         for c in chunks],
            "language":        [c["language"]        for c in chunks],
            "quality_score":   [c["quality_score"]   for c in chunks],
            "document_kind":   [c["document_kind"]   for c in chunks],
            "source_family":   [c["source_family"]   for c in chunks],
            "document_group":  [c["document_group"]  for c in chunks],
            "topic_tags":      [c["topic_tags"]      for c in chunks],
            "section_heading": [c["section_heading"] for c in chunks],
            "text":            [c["text"]            for c in chunks],
            "chunk_index":     [c["chunk_index"]     for c in chunks],
            "total_chunks":    [c["total_chunks"]    for c in chunks],
            "sparse_indices":  sparse_indices,
            "sparse_values":   sparse_values,
        },
        schema=PARQUET_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ── 1. Dateien einlesen ───────────────────────────────────────────────
    md_files = sort_markdown_files(list(INGESTED_DIR.glob("*.md")))
    if not md_files:
        logger.error("Keine .md-Dateien gefunden in %s", INGESTED_DIR)
        sys.exit(1)
    logger.info("Gefunden: %d .md-Datei(en) in %s", len(md_files), INGESTED_DIR)

    if RESUME_FROM_FILE > 1:
        skip = RESUME_FROM_FILE - 1
        logger.info("Überspringe die ersten %d Datei(en) (RESUME_FROM_FILE=%d)",
                    skip, RESUME_FROM_FILE)
        md_files = md_files[skip:]

    # ── 2. Splitter vorbereiten ───────────────────────────────────────────
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # ── 3. Embedding-Modelle laden ────────────────────────────────────────
    logger.info("Lade Dense-Modell '%s' …", DENSE_MODEL)
    dense_embedder, dense_backend_name = build_dense_backend(DENSE_MODEL)
    logger.info("Dense-Backend aktiv: %s", dense_backend_name)

    logger.info("Lade Sparse-Modell '%s' (BM25, CPU-only) …", SPARSE_MODEL)
    sparse_embedder = SparseTextEmbedding(
        model_name=SPARSE_MODEL,
        cache_dir=HF_CACHE_DIR,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    logger.info("Beide Modelle bereit.")

    # ── 4. Parquet-Writer öffnen ──────────────────────────────────────────
    # Schreibt dateiweise Row-Groups → konstanter RAM-Bedarf
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        OUTPUT_FILE,
        schema=PARQUET_SCHEMA,
        compression="zstd",
        compression_level=3,
    )
    logger.info("Schreibe nach: %s", OUTPUT_FILE)

    total_points = 0
    skipped      = 0
    total_all    = len(md_files) + (RESUME_FROM_FILE - 1)

    try:
        for files_done, path in enumerate(md_files, start=1):
            result = parse_markdown_file(path)
            if result is None:
                skipped += 1
                continue

            meta, body = result
            chunks = chunk_document(meta, body, rec_splitter)
            if not chunks:
                skipped += 1
                continue

            file_batches: list[pa.RecordBatch] = []
            texts = [c["text"] for c in chunks]

            # ── Dense-Batches (GPU) ───────────────────────────────────────
            dense_vecs_all: list = []
            for d_start in range(0, len(texts), DENSE_BATCH_SIZE):
                batch_texts = texts[d_start : d_start + DENSE_BATCH_SIZE]
                dense_vecs_all.extend(
                    dense_embedder.embed(batch_texts)
                )

            # ── Sparse-Batches (CPU/BM25) ─────────────────────────────────
            sparse_indices_all: list[list[int]]   = []
            sparse_values_all:  list[list[float]] = []
            for s_start in range(0, len(texts), SPARSE_BATCH_SIZE):
                batch_texts = texts[s_start : s_start + SPARSE_BATCH_SIZE]
                for emb in sparse_embedder.embed(batch_texts):
                    sparse_indices_all.append(emb.indices.tolist())
                    sparse_values_all.append(emb.values.tolist())

            # ── RecordBatches zusammenstellen ─────────────────────────────
            for sub_start in range(0, len(chunks), DENSE_BATCH_SIZE):
                sub_end    = sub_start + DENSE_BATCH_SIZE
                sub_chunks = chunks[sub_start:sub_end]
                batch = chunks_to_record_batch(
                    sub_chunks,
                    dense_vecs_all[sub_start:sub_end],
                    sparse_indices_all[sub_start:sub_end],
                    sparse_values_all[sub_start:sub_end],
                )
                file_batches.append(batch)

            # Alle Batches dieser Datei als eine Row-Group schreiben
            if file_batches:
                table = pa.Table.from_batches(file_batches, schema=PARQUET_SCHEMA)
                writer.write_table(table)
                file_points   = len(table)
                total_points += file_points

                abs_done = files_done + (RESUME_FROM_FILE - 1)
                pct      = 100.0 * abs_done / total_all
                logger.info(
                    "  [%d/%d  %5.1f%%]  %-55s → %3d Chunk(s)  (gesamt: %d)",
                    abs_done, total_all, pct, path.name, file_points, total_points,
                )
                del table, file_batches

            del chunks, texts, dense_vecs_all, sparse_indices_all, sparse_values_all
            gc.collect()

    finally:
        writer.close()

    if total_points == 0:
        logger.error("Kein verwendbarer Inhalt — abgebrochen.")
        sys.exit(1)

    size_mb = OUTPUT_FILE.stat().st_size / 1_048_576
    logger.info("─" * 60)
    logger.info("Ausgabe      : %s  (%.1f MB)", OUTPUT_FILE, size_mb)
    logger.info("Dateien      : %d  (übersprungen: %d)",
                len(md_files) - skipped, skipped)
    logger.info("Chunks gesamt: %d  (Dense + Sparse)", total_points)
    logger.info("─" * 60)
    logger.info("Nächster Schritt: Datei auf lokalen Rechner übertragen")
    logger.info("  scp %s user@localhost:~/RAGHSH/artifacts/", OUTPUT_FILE)
    logger.info("Dann direkt: python -m hsh_scraper.local_importer  (enrich_sparse.py nicht nötig)")


if __name__ == "__main__":
    main()
