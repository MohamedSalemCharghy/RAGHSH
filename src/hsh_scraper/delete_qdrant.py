"""
Qdrant-Collection zurücksetzen — alle Vektoren löschen und Collection neu anlegen.

WARNUNG: Dieses Skript löscht unwiderruflich alle Vektoren und Metadaten
aus der Collection. Danach muss local_importer.py erneut vollständig
durchgeführt werden, um die Datenbank wieder zu befüllen.

Anwendungsfall:
    - Nach einem vollständigen Neucrawl, wenn alle Inhalte aktualisiert wurden
    - Wenn die Collection durch fehlerhafte Daten korrumpiert wurde
    - Zum Zurücksetzen auf einen leeren Zustand für Testzwecke
    - Nach Änderungen am Vektorschema (z.B. neue Sparse Vectors)

Aufruf:
    python -m hsh_scraper.delete_qdrant
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Name der Collection — muss mit COLLECTION_NAME in local_importer.py übereinstimmen
COLLECTION_NAME = "hsh_knowledge"
VECTOR_SIZE     = 1024

# Verbindung zum lokalen Qdrant-Docker-Container
client = QdrantClient(host="localhost", port=6333)

# ── Schritt 1: Bestehende Collection vollständig löschen ──────────────────────
# Entfernt alle Vektoren, Payloads und Indizes dieser Collection.
client.delete_collection(collection_name=COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' wurde gelöscht.")

# ── Schritt 2: Leere Collection neu erstellen ─────────────────────────────────
# Legt eine neue, leere Collection mit benannten Dense- und Sparse-Vektoren an:
#   dense  — 1024-dim Jina-Embeddings (Cosine-Ähnlichkeit)
#   sparse — BM25 Sparse Vectors (Qdrant/bm25 via FastEmbed)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "sparse": qmodels.SparseVectorParams(),
    },
)
print(f"Collection '{COLLECTION_NAME}' wurde leer neu angelegt.")
print("  Dense-Vektoren : 'dense'  (1024 dim, Cosine)")
print("  Sparse-Vektoren: 'sparse' (BM25)")

# ── Schritt 3: Payload-Indizes anlegen ────────────────────────────────────────
# WICHTIG: Indizes müssen VOR dem Import angelegt werden. local_importer.py
# überspringt ensure_collection() wenn die Collection bereits existiert.

# Volltext-Index auf 'text' — MULTILINGUAL für deutsches Stemming + Komposita
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="text",
    field_schema=qmodels.TextIndexParams(
        type="text",
        tokenizer=qmodels.TokenizerType.MULTILINGUAL,
        min_token_len=2,
        max_token_len=50,
        lowercase=True,
    ),
)
print("  Volltext-Index 'text' (MULTILINGUAL) angelegt.")

# Keyword-Index auf 'faculty' für schnelle Filterung
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="faculty",
    field_schema=qmodels.KeywordIndexParams(type="keyword"),
)
print("  Keyword-Index 'faculty' angelegt.")

# Integer-Index auf 'chunk_index' für Context Augmentation
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="chunk_index",
    field_schema=qmodels.IntegerIndexParams(type="integer", lookup=True, range=False),
)
print("  Integer-Index 'chunk_index' angelegt.")

# Keyword-Index auf 'source_url' für Context Augmentation
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="source_url",
    field_schema=qmodels.KeywordIndexParams(type="keyword"),
)
print("  Keyword-Index 'source_url' angelegt.")

# Keyword-Index auf 'document_kind' für zielgerichtete Filter
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="document_kind",
    field_schema=qmodels.KeywordIndexParams(type="keyword"),
)
print("  Keyword-Index 'document_kind' angelegt.")

# Keyword-Index auf 'document_group' für gruppierte Nachbarsuche
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="document_group",
    field_schema=qmodels.KeywordIndexParams(type="keyword"),
)
print("  Keyword-Index 'document_group' angelegt.")

# Keyword-Index auf 'language' für Sprachfilter/Diagnose
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="language",
    field_schema=qmodels.KeywordIndexParams(type="keyword"),
)
print("  Keyword-Index 'language' angelegt.")

print("Nächster Schritt: python -m hsh_scraper.local_importer ausführen, um die Datenbank zu befüllen.")
