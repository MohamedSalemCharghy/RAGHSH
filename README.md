# RAGHSH — RAG-System für die Hochschule Hannover

**RAGHSH** ist ein vollständiges **Retrieval-Augmented Generation (RAG)**-System, das Fragen zur Hochschule Hannover ausschließlich auf Basis offizieller Dokumente beantwortet. Es kombiniert einen automatischen Web-Spider mit einem kuratierten Markdown-Korpus, einer hybriden Qdrant-Vektordatenbank, Glossar-/Definitionshilfen, einem gesteuerten Retrieval-Planer und einem großen Sprachmodell (LLM) der GWDG ChatAI API.

---

## Inhaltsverzeichnis

1. [Konzept: Was ist RAG und warum?](#konzept-was-ist-rag-und-warum)
2. [Systemübersicht](#systemübersicht)
3. [Verzeichnisstruktur](#verzeichnisstruktur)
4. [Abhängigkeiten und Bibliotheken](#abhängigkeiten-und-bibliotheken)
5. [Einrichtung](#einrichtung)
6. [Nutzung — Schritt für Schritt](#nutzung--schritt-für-schritt)
7. [Programmübersicht](#programmübersicht)
8. [Technische Architektur](#technische-architektur)
9. [Leistung und Laufzeiten](#leistung-und-laufzeiten)
10. [Konfigurationsparameter](#konfigurationsparameter)
11. [Ideen zur Weiterentwicklung](#ideen-zur-weiterentwicklung)

---

## Konzept: Was ist RAG und warum?

### Das Problem mit reinen LLMs

Große Sprachmodelle wie GPT-4 oder Llama sind auf riesigen Textmengen trainiert und können fließend antworten — aber sie haben fundamentale Schwächen für institutionelle Wissenssysteme:

- **Halluzinierung**: LLMs erfinden plausibel klingende, aber falsche Fakten
- **Wissensstichtag**: Das Trainingskorpus endet zu einem bestimmten Datum; aktuelle Änderungen (Prüfungsordnungen, Fristen, Ansprechpartner) sind unbekannt
- **Fehlende Spezifität**: Allgemeine Informationen zur Hochschule Hannover sind im Training kaum vorhanden
- **Keine Quellenangaben**: Woher kommt die Information? Lässt sich die Aussage nachvollziehen?

### Die RAG-Lösung

**Retrieval-Augmented Generation** löst diese Probleme durch eine zweistufige Architektur:

```
Nutzerfrage
    │
    ▼
[Stufe 1: Retrieval]
Suche in der Vektordatenbank nach den
relevantesten Textstellen aus offiziellen
HsH-Dokumenten (Hybrid-Suche: Dense + BM25)
    │
    ▼
[Stufe 2: Generation]
LLM erhält Frage + Kontext und darf
NUR auf Basis dieser Textstellen antworten
    │
    ▼
Faktentreue Antwort mit Quellenangaben
```

Das LLM fungiert dabei als **intelligenter Leser und Formulierer**, nicht als Wissensquelle. Die Wissensbasis bleibt jederzeit aktualisierbar und nachvollziehbar.

### Hybrid-Suche: Dense + Sparse Vectors

Dieses System verwendet zwei komplementäre Suchmethoden, die per **Reciprocal Rank Fusion (RRF)** kombiniert werden:

| Methode | Stärke | Schwäche |
|---------|--------|----------|
| **Dense Search** (Jina Embeddings) | Semantisches Verstehen, Synonyme, Paraphrasen | Exakte Bezeichnungen können verloren gehen |
| **BM25 Sparse Search** | Exakte Schlüsselwörter, Modulnummern, Namen | Kein semantisches Verständnis |

**Beispiel:** Die Frage „Wie unterbreche ich mein Studium?" findet über Dense Search semantisch verwandte Texte über „Beurlaubung", die das Wort „Unterbrechung" nicht enthalten. Über BM25 findet die Suche Texte mit dem exakten Begriff „Beurlaubungsantrag".

### Reranking: die dritte Qualitätsstufe

Nach der RRF-Fusion werden die Ergebnisse einem **Cross-Encoder-Reranker** übergeben (`jinaai/jina-reranker-v2-base-multilingual`). Anders als Embedding-Modelle, die Dokument und Anfrage separat kodieren, bewertet ein Cross-Encoder jedes (Frage, Passage)-Paar gemeinsam — diese direkte Interaktion ermöglicht präzisere Relevanzurteile auf Kosten von mehr Rechenzeit.

### Context Augmentation

Texte werden in Chunks aufgeteilt, die an Grenzen „abgeschnitten" werden können. RAGHSH lädt für die Top-3-Treffer automatisch die benachbarten Chunks (vorheriger und nachfolgender) nach und hängt sie an den Kerntext an — damit gehen keine Informationen an Chunk-Grenzen verloren.

Zusätzlich gibt es eine **begrenzte zweite Retrieval-Runde**, wenn der erste Kontext erkennbar unvollständig ist. Dann fordert der Chatbot gezielt mehr Kontext an, z.B. Nachbar-Chunks, einen ganzen Abschnitt oder eine zweite Suche für eine fehlende Vergleichsseite. So bleibt der Standard-Kontext klein, ohne bei Regelwerken, Abkürzungen oder Vergleichsfragen vorschnell zu halluzinieren.

### Guided Retrieval vor der Antwort

Die aktuelle Pipeline schickt Nutzerfragen nicht mehr blind direkt in die Vektorsuche. Davor laufen mehrere kleine, kontrollierte Schritte:

- **Query-Bewertung:** Ist die Frage eine Definition, ein Workflow, eine Kontaktfrage oder zu allgemein?
- **Präzisierung:** Zu breite Fragen wie „Wie stelle ich einen Online-Antrag?" werden erst konkretisiert.
- **Glossar-/Legendenhilfe:** Häufige Kürzel und Prüfungsform-Codes werden gezielt gesucht.
- **Optionaler Retrieval-Planer:** Ein LLM darf Suchpfade strukturieren, aber nicht selbst antworten.

So bleibt die spätere Antwort näher an den tatsächlichen HsH-Dokumenten und wird robuster gegen Alltagssprache, Schreibvarianten und sehr kurze Fragen.

---

## Systemübersicht

```
Phase 1: Datensammlung
  main.py / resume_crawler.py  →  data/ingested/*.md
  (Web-Spider: HTML + PDF → Markdown mit YAML-Header)

Phase 1b: Qualitätsprüfung
  check_quality.py  →  Audit des Rohkorpus
  (bewertet Dateien, zeigt Sprache, Score und Problemgruende)

Phase 1c: Kuratierung und Organisation
  clean_corpus.py  →  data/curated/*.md + data/curated_report.json
  (entfernt schlechte Links/Boilerplate, markiert Sprache, Gruppen, Topics)

Phase 2a: Vektorisierung auf dem HPC-Cluster  [empfohlen für große Datenmengen]
  hpc_vectorizer.py  →  artifacts/hsh_vectors.parquet
  (Dense + BM25 Sparse Vectors; bevorzugt data/curated, faellt sonst auf data/ingested zurueck)

Phase 2b: Legacy-Kompatibilität für ältere Artefakte
  local_importer.py  →  nutzt bei Bedarf vorhandene artifacts/hsh_vectors_enriched.parquet-Dateien
  (lokale Sparse-Anreicherung wird nicht mehr als eigenes Skript mitgeliefert)

Phase 3: Datenbank befüllen
  local_importer.py  →  Qdrant-Collection 'hsh_knowledge'
  (erkennt automatisch: enriched > plain Parquet)

Phase 4: Suche testen (optional, CLI)
  hybrid_search.py  →  Interaktive Guided-Hybrid-Suche ohne Antwort-LLM
  (Query-Analyse, Glossar-Hits, mehrstufige Suchpfade, Prozess-Trace)

Phase 5a: Web-App
  hsh_web_app.py (Streamlit)  →  Chatbot mit Rollen- und Fakultätsfilter
  (Präzisierungsdialog, 5-Turn-Kurzgedächtnis, Prozess-/Timing-Ansicht)

Phase 5b: CLI-Chatbot
  hsh_chatbot.py  →  Interaktiver Terminal-Chatbot
  (Retrieval-Planer, lokale Follow-up-Router, Leistungsdaten im Terminal)

Hilfsprogramme:
  query_assist.py   →  lokale Query-Bewertung, Präzisierungslogik, Retrieval-Varianten
  glossary_index.py →  Glossar-/Legenden-/Abkürzungsindex aus data/curated
  retrieval_planner.py → optionaler LLM-Planer für schwierige Retrieval-Fälle
  conversation_memory.py → kompaktes 5-Turn-Kurzgedächtnis
  turn_router.py    → Router für Anschlussfragen wie "der Link öffnet nicht"
  corpus_quality.py  →  gemeinsame Bewertungs-/Kuratierungslogik
  check_quality.py   →  read-only Qualitätsprüfung der Markdown-Dateien
  delete_qdrant.py   →  Collection zurücksetzen (nach Schema-Änderungen)
```

---

## Verzeichnisstruktur

```
RAGHSH/
├── docker-compose.yml               # Qdrant-Vektordatenbank als Docker-Container
├── pyproject.toml                   # Paket-/Test-Konfiguration
├── requirements.txt                 # Python-Abhängigkeiten für Crawl, App, Import und Tests
├── config/
│   └── .env.example                 # Vorlage für lokale API-Konfiguration
├── tests/                           # Automatisierte Regressionstests
├── src/
│   └── hsh_scraper/                 # Importierbares Python-Paket und ausführbare Module
│       ├── main.py                  # Phase 1: Web-Spider (BFS-Crawler)
│       ├── resume_crawler.py        # Phase 1b: Spider fortsetzen / Lücken schließen
│       ├── clean_corpus.py          # Phase 1c: Rohkorpus bereinigen und organisieren
│       ├── hpc_vectorizer.py        # Phase 2a: HPC-Vektorisierung (Dense + BM25 → Parquet)
│       ├── local_importer.py        # Phase 3: Parquet → Qdrant
│       ├── hybrid_search.py         # Phase 4: Guided-Hybrid-Suche
│       ├── hsh_web_app.py           # Phase 5a: Streamlit Web-App
│       ├── hsh_chatbot.py           # Phase 5b: CLI-Chatbot
│       └── evals/                   # Validierungssystem und Fallbasis
├── data/                            # Ignorierte Crawl-/Kurationsdaten, README versioniert
├── artifacts/                       # Ignorierte Vektoren und Validierungsläufe
└── qdrant_data/                     # Ignorierter persistenter Qdrant-Speicher
```

---

## Abhängigkeiten und Bibliotheken

### Infrastruktur

| Komponente | Beschreibung |
|---|---|
| **Docker** | Laufzeitumgebung für Qdrant |
| **Qdrant** (`qdrant/qdrant`) | Vektordatenbank, Port 6333 (HTTP/REST) und 6334 (gRPC) |
| **Python 3.11+** | Laufzeitumgebung für alle Skripte |
| **GWDG ChatAI API** | OpenAI-kompatibler LLM-Dienst für Hochschulen |

### Python-Bibliotheken

| Bibliothek | Verwendung |
|---|---|
| **crawl4ai** | Asynchroner Web-Crawler mit JavaScript-Unterstützung (Playwright-Backend) |
| **pymupdf4llm** | PDF → LLM-optimiertes Markdown |
| **python-slugify** | URL-sichere Dateinamen aus URL-Pfaden |
| **openpyxl** | Excel-Fehlerbericht nach dem Crawl |
| **qdrant-client** | Python-Client für Qdrant (Upsert, Hybrid-Suche, Payload-Indizes) |
| **fastembed** | Lokale Embedding-Modelle ohne externe API — Sparse (`Qdrant/bm25`) und Reranker (`jinaai/jina-reranker-v2-base-multilingual`); Dense bleibt als Fallback nutzbar |
| **transformers** + **torch** + **einops** | GPU-beschleunigte Dense-Embeddings auf dem HPC (`jinaai/jina-embeddings-v3` via `trust_remote_code`) |
| **pyarrow** | Parquet-Lesen/-Schreiben mit Row-Group-Streaming |
| **langchain-text-splitters** | Zweistufiges Chunking: `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` |
| **streamlit** | Web-App-Framework für `hsh_web_app.py` |
| **openai** | OpenAI-kompatibler HTTP-Client für GWDG ChatAI |
| **python-dotenv** | Lädt den API-Schlüssel aus `.env` |
| **httpx** | Async-HTTP-Client für PDF-Downloads |
| **pytest** | Lokale Regressionstests |

Hinweis: `requirements.txt` deckt den lokalen Standardbetrieb ab. Für den optionalen GPU-HPC-Pfad mit `transformers/cuda` kommen zusätzliche Pakete dazu; die konkreten Installationsbefehle stehen im Abschnitt [Einrichtung](#einrichtung).

---

## Einrichtung

### 1. Voraussetzungen

- Docker und Docker Compose installiert
- Python 3.11 oder neuer installiert
- GWDG ChatAI API-Schlüssel

### 2. Qdrant-Datenbank starten

```bash
cd RAGHSH
docker compose up -d
```

Prüfen ob Qdrant läuft:
```bash
curl http://localhost:6333/healthz
# {"title":"qdrant - vector search engine","version":"..."}
```

### 3. Python-Umgebung einrichten

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS

python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Playwright-Browser für crawl4ai herunterladen:
playwright install chromium

# Nur für den GPU-HPC-Pfad mit transformers/cuda:
pip install "numpy<2" "transformers==4.56.2" einops torch
```

Lokaler Modellcache:

- `hsh_web_app.py` und `hsh_chatbot.py` verwenden standardmäßig einen persistenten `fastembed`-Cache unter `~/.cache/fastembed`.
- Dadurch werden Dense-Modell, BM25 und Reranker nach dem ersten Download bei späteren Starts wiederverwendet, statt unvorhersehbar aus dem System-Temp-Verzeichnis neu geladen zu werden.
- Optional kann der Pfad überschrieben werden:

```bash
export FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed"
```

### 4. API-Schlüssel konfigurieren

```bash
cp config/.env.example config/.env
# config/.env öffnen und GWDG_API_KEY eintragen
```

Inhalt der `config/.env`:
```ini
GWDG_API_KEY=dein-schluessel-hier
GWDG_API_BASE=https://chat-ai.academiccloud.de/v1
```

---

## Nutzung — Schritt für Schritt

### Phase 1: Website crawlen

```bash
python -m hsh_scraper.main
```

Crawlt `www.hs-hannover.de` per Breadth-First-Search (BFS). Jede Seite wird als Markdown-Datei mit YAML-Header gespeichert. Bereits frisch gecachte Seiten (< `MAX_AGE_DAYS` Tage) werden übersprungen.

Zusätzlich bewertet ein gemeinsamer RAG-Filter jede neu entdeckte URL, bevor sie in die Queue gelangt. Geblockt werden aktuell u.a.:

- der englische Bereich unter `/en` auf allen HsH-Subdomains
- Medien-Dateien wie Bilder, Audio und Video
- technische Assets wie CSS/JS/Archive
- nicht-oeffentliche App-/Login-Bereiche wie `moodle.hs-hannover.de` und `intranet.hs-hannover.de` sowie Auth-Pfade (`/login`, `/logout`, `/shibboleth`, `/saml`, `/oauth`)
- `fileadmin/_processed_`-Assets
- Office-Dokumente ohne Ingest-Support (`.docx`, `.pptx`, `.xlsx`, ...)
- offensichtlich kaputte URLs mit rohen Markdown-/Junk-Zeichen wie `*` oder `|`
- bekannte Backend-/Interndomains wie `serwiss.bib.hs-hannover.de` und `typo3backend-live.hs-hannover.de`

Jede Entscheidung wird in `data/url_decisions.db` gespeichert.

```
2026-03-14 [INFO] Crawling: https://www.hs-hannover.de/
2026-03-14 [INFO] Saved data/ingested/2026-03-14_index.md
...
2026-03-14 [INFO] Done. 847 succeeded, 3 failed, 0 skipped out of 850 URLs visited.
```

### Phase 1b: Crawler fortsetzen

```bash
python -m hsh_scraper.resume_crawler            # Analysiert und crawlt fehlende/veraltete URLs
python -m hsh_scraper.resume_crawler --dry-run  # nur Analyse, kein Crawlen
```

`resume_crawler.py` liest alle vorhandenen Markdown-Dateien, extrahiert darin enthaltene Links und crawlt nur Seiten, die fehlen oder veraltet sind. Ideal für inkrementelle Aktualisierungen.

Der gleiche RAG-Filter wird auch im Resume-Pfad verwendet. `--dry-run` zeigt dadurch nicht nur die Crawl-Kategorien (frisch / veraltet / fehlend), sondern auch eine Zusammenfassung der Filterentscheidungen inklusive Gruenden und Beispiel-URLs.

Die SQLite-Datei `data/url_decisions.db` wird bei Bedarf automatisch angelegt und fortlaufend aktualisiert.

### (Optional) Qualität prüfen

```bash
python -m hsh_scraper.check_quality
```

Gibt eine Tabelle aller Roh-Markdown-Dateien aus und nutzt dieselben Regeln wie die spätere Kuratierung:

```
Filename                          Words  Type    Lang   Score  Quality
───────────────────────────────────────────────────────────────────────
2026-03-14_index.md               1.234  html    de        96  OK
2026-03-14_preview-page.md           52  html    mixed     58  PRUEFEN
2026-03-14_exchange-info.md         410  html    en        25  PRUEFEN
```

Es werden u.a. bewertet:

- Mindesttextmenge für HTML/PDF
- Fehlerseitenmuster
- sprachlich gemischte oder englische Seiten
- Preview-/Backend-Links im Body
- ältere Duplikate desselben URL-Slugs

### Phase 1c: Korpus bereinigen und organisieren

```bash
python -m hsh_scraper.clean_corpus
```

`clean_corpus.py` liest `data/ingested/`, schreibt ein bereinigtes Korpus nach `data/curated/` und erzeugt einen JSON-Report in `data/curated_report.json`.

Bereinigungen und Anreicherungen:

- entfernt geblockte Preview-/Backend-Links direkt aus dem Markdown-Body
- normalisiert relative Links zu offiziellen absoluten URLs
- entfernt typische Boilerplate-Zeilen wie Teilen-/Scroll-Hinweise
- verwirft englische Seiten standardmäßig
- ergänzt Metadaten wie `language`, `quality_score`, `document_kind`, `source_family`, `document_group`, `topic_tags`

Optional mit eigenen Pfaden:

```bash
python -m hsh_scraper.clean_corpus \
  --input-dir data/ingested \
  --output-dir data/curated \
  --report-file data/curated_report.json
```

### Phase 2a: Vektorisierung auf dem HPC-Cluster (empfohlen)

```bash
# Auf dem Login-Node:
module load miniforge3
source activate /pfad/zur/.venv-hpc
srun --partition=kisski --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash -l

# Im GPU-Job:
cd /pfad/zu/RAGHSH
export HF_HOME="$PWD/.hf_cache"
export RAG_SOURCE_DIR="$PWD/data/curated"
export DENSE_BACKEND=transformers
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m hsh_scraper.hpc_vectorizer
```

`hpc_vectorizer.py` erzeugt **beide** Vektortypen in einem Durchlauf und verwendet automatisch `data/curated/`, wenn dort bereits bereinigte Markdown-Dateien vorliegen. Falls nicht, fällt das Skript auf `data/ingested/` zurück. Das Dense-Embedding nutzt bevorzugt das neue `transformers/cuda`-Backend; `fastembed` bleibt als Fallback erhalten.

Für den Modellcache gilt:

- Wenn `HF_HOME` gesetzt ist, verwendet `hpc_vectorizer.py` diesen Pfad für `transformers`, Sparse-BM25 und den `fastembed`-Fallback.
- Wenn `HF_HOME` nicht gesetzt ist, fällt das Skript auf `FASTEMBED_CACHE_PATH` zurück.
- Wenn auch `FASTEMBED_CACHE_PATH` nicht gesetzt ist, wird standardmäßig `~/.cache/fastembed` verwendet statt des temporären Systemverzeichnisses.

Es erzeugt:
- Dense Embeddings (GPU-beschleunigt, Batch-Größe 64)
- BM25 Sparse Embeddings (CPU-only, Batch-Größe 256)
- Chunk-Metadaten für spätere Filterung und Gruppierung im Retrieval

Output: `artifacts/hsh_vectors.parquet` (komprimiert mit ZSTD, enthält Dense-, Sparse- und Qualitätsmetadaten)

Wichtige Praxisnotizen für KISSKI/HLRN:
- Die Compute-Nodes sind häufig offline. Modelle und Remote-Code müssen deshalb vorab in `HF_HOME` gecacht werden.
- Für `jinaai/jina-embeddings-v3` hat sich `transformers==4.56.2`, `numpy<2` und `einops` als stabile Kombination bewährt.
- Der fertige Export kann direkt mit `local_importer.py` importiert werden; `enrich_sparse.py` ist für diesen Workflow nicht nötig.

Datei auf den lokalen Rechner übertragen:

```bash
scp -i ~/.ssh/id_ed25519 \
user@cluster.example:/pfad/zu/RAGHSH/artifacts/hsh_vectors.parquet \
/lokaler/pfad/zu/RAGHSH/artifacts/
```

### Phase 2b: Legacy-Fallback für ältere Parquet-Artefakte

Diese Version liefert kein `enrich_sparse.py` mehr mit. Der reguläre Workflow ist deshalb:

1. `clean_corpus.py`
2. `hpc_vectorizer.py`
3. `local_importer.py`

`local_importer.py` bleibt jedoch kompatibel zu bereits vorhandenen `artifacts/hsh_vectors_enriched.parquet`-Dateien aus älteren Läufen.

> **Hinweis:** Wenn sowohl `artifacts/hsh_vectors_enriched.parquet` als auch `artifacts/hsh_vectors.parquet` vorhanden sind, bevorzugt `local_importer.py` weiterhin die angereicherte Datei.

### Phase 3: Datenbank befüllen

```bash
python -m hsh_scraper.delete_qdrant    # Alte Collection löschen + neu mit Indizes anlegen
python -m hsh_scraper.local_importer   # Parquet-Datei in Qdrant hochladen
```

`delete_qdrant.py` ist nötig nach:
- Schema-Änderungen (z.B. neue Vektortypen)
- Komplettem Neucrawl
- Fehlerhafte Daten in der Datenbank

`local_importer.py` lädt die Daten in Batches von 256 Punkten hoch. Bei Unterbrechung kann `RESUME_FROM_ROW` gesetzt werden, um den Import fortzusetzen.

```
2026-03-14 [INFO] Parquet-Datei: hsh_vectors.parquet (245.3 MB, 28.451 Zeilen)
2026-03-14 [INFO] Collection 'hsh_knowledge' angelegt (dense=1024dim, sparse=BM25)
2026-03-14 [INFO]   Row-Group 1/847  [  0.1%]  256 Punkte hochgeladen (gesamt)
...
2026-03-14 [INFO] Fertig. 28.451 Punkte hochgeladen.
```

### Phase 4: Suche testen (optional)

```bash
python -m hsh_scraper.hybrid_search
```

Interaktive Guided-Hybrid-Suche ohne Antwort-LLM. Nützlich zur Diagnose der Trefferqualität, der Query-Aufbereitung und der Retrieval-Pfade:

```
Frage> Bewerbungsfristen Bachelor Informatik

  ┌─ Treffer 1  (RRF-Score: 0.03226)
  │  URL      : https://www.hs-hannover.de/studium/bewerbung/...
  │  Fakultät : Fakultät IV
  │  Abschnitt: Bewerbung > Fristen
  │  Stand    : 2026-03-10
  │  Vorschau : Die Bewerbungsfrist für den Bachelorstudiengang...
  └──────────────────────────────────────────────────────────────
```

Zusätzlich zeigt das Tool einen **Systemprozess-Trace** mit:

- lokaler Query-Bewertung (`definition`, `workflow`, `contact`, `exam_registration`)
- erkannten Kürzeln und Fakultäten
- Glossar-/Legenden-Hits
- eventuellen Schreibvarianten / Near-Match-Korrekturen
- allen tatsächlich verwendeten Suchpfaden

### Phase 5a: Web-App starten

```bash
streamlit run src/hsh_scraper/hsh_web_app.py
# → http://localhost:8501
```

Beim ersten Start werden Dense-Modell, BM25 und Reranker ggf. einmal heruntergeladen und danach standardmäßig aus `~/.cache/fastembed` wiederverwendet. Mit `FASTEMBED_CACHE_PATH` kann ein anderer persistenter Cache-Pfad gesetzt werden.

Die Web-App bietet:
- **Rollenauswahl**: Studierender / Mitarbeitender / Lehrender / Besucher (passt den Ton des LLM an)
- **Fakultätsfilter**: Ergebnisse werden auf die gewählte Fakultät + fakultätsübergreifende Seiten (ohne Fakultätszuordnung) eingeschränkt
- **Präzisierungsdialoge**: zu allgemeine Fragen wie „Wie stelle ich einen Online-Antrag?" werden vor der Antwort konkretisiert
- **Streaming-Antworten** mit Quellenangaben und optionalem Denkprozess-Expander
- **Veralterungswarnung**: Falls eine Quelle älter als 6 Monate ist, empfiehlt das System Nachprüfung
- **Follow-up Router**: Anschlussfragen wie „der Link öffnet nicht" oder „welche Quelle meinst du" werden lokal aus dem letzten Turn beantwortet statt blind neu zu suchen
- **5-Turn-Kurzgedächtnis**: die letzten fünf beantworteten Fragen werden komprimiert mitgeführt, aber nicht als offizielle Quelle behandelt
- **Systemprozess + Leistungsdaten**: Expander für Retrieval-Schritte, Planner-Hinweise, Glossar-Hits und Zeitmessungen pro Phase

### Phase 5b: CLI-Chatbot

```bash
python -m hsh_scraper.hsh_chatbot
```

Auch der CLI-Chatbot verwendet standardmäßig `~/.cache/fastembed` als persistenten Modellcache und lädt dieselben `fastembed`-Modelle deshalb nach dem ersten erfolgreichen Start nicht jedes Mal neu herunter.

Startet eine interaktive Terminal-Session mit dynamischer Modellauswahl:

```
Modell wählen:
  [1] meta-llama-3.1-70b-instruct
  [2] deepseek-r1
  [3] gpt-4o
  ...
Auswahl (Enter = 1): 2

HsH-Chatbot bereit.  Strg+C oder 'exit' zum Beenden.

Frage> Wie beantrage ich eine Beurlaubung?
```

Der CLI-Chatbot zeigt zusätzlich:

- **Kurzgedächtnis**: kompakte Vorschau der letzten maximal fünf beantworteten Fragen
- **Präzisierungsdialog** für allgemeine oder mehrdeutige Fragen
- **Systemprozess**: Query-Analyse, Planner-Hinweise, Glossar-Hits und Suchpfade
- **Leistungsdaten**: Zeiten für Planner-LLM, Retrieval, Follow-up-Heuristik, Prompt-Aufbau und Antwort-LLM

### Phase 5c: Validierung per CLI

```bash
python -m hsh_scraper.evals.validation_cli --all
# oder nach Installation als Paket:
raghsh-eval --all
```

Der CLI-Lauf verwendet dieselbe feste Strict-Dialog-Fallbasis wie die Web-App
und führt die Fälle nacheinander aus. Jeder Fall wird zweistufig bewertet:
zuerst Antwort gegen Referenz, danach Antwort gegen die tatsächlich aus Qdrant
abgerufenen Chunks. GWDG-API-Aufrufe werden standardmäßig auf maximal 10 pro
Minute begrenzt. Einzelne Fälle können gezielt gestartet werden:

```bash
python -m hsh_scraper.evals.validation_cli --case q03
python -m hsh_scraper.evals.validation_cli --list-cases
```

Ergebnisse landen wie in der Web-App unter `artifacts/evals/results/RUN_ID/`.
Jede Fall-Datei enthält `evaluation` für Stufe 1 und `evidence_evaluation`
für die Qdrant-Grounding-Prüfung.

---

## Programmübersicht

### `main.py` — Web-Spider

Crawlt die gesamte HsH-Website per **Breadth-First-Search (BFS)**.

| Parameter | Standard | Beschreibung |
|---|---|---|
| `SEED_URLS` | `["https://www.hs-hannover.de/"]` | Startseiten |
| `MAX_PAGES` | `10.000` | Maximale Seitenanzahl |
| `ALLOWED_DOMAIN` | `hs-hannover.de` | Nur diese Domain und ihre Subdomains |
| `BLOCKED_DOMAINS` | `{"serwiss.bib.hs-hannover.de", "typo3backend-live.hs-hannover.de"}` | Geblockte Subdomains |
| `MAX_AGE_DAYS` | `7` | Cache-Alter in Tagen |
| `RATE_LIMIT_SECONDS` | `0.5` | Pause zwischen Requests |

- **HTML**: Crawl4AI (Playwright) mit CSS-Selektor `main, .content-main, #content, .frame-default`; boilerplate (Navigation, Header, Footer, Cookie-Banner) wird ausgeblendet
- **PDF**: httpx-Download + pymupdf4llm-Konvertierung
- **Dateiformat**: `YYYY-MM-DD_url-slug.md` mit YAML-Frontmatter (`source_url`, `title`, `crawl_date`, `content_type`)
- **RAG-Filter**: Neue Links werden vor dem Queueing durch `url_filter.py` bewertet und in `data/url_decisions.db` protokolliert
- **Sitemap-Seeding**: erkannte `sitemap.xml`-Dateien werden zusätzlich als Seed-Quelle genutzt
- **Soft-Priorisierung**: URLs werden als `allow_high_value`, `allow_low_value` oder `block` klassifiziert; High-Value-Links werden bevorzugt gecrawlt
- **Post-Crawl-Quality-Gate**: sehr kurze, nav-lastige oder offensichtliche Junk-Seiten werden nach der Extraktion noch verworfen

---

### `resume_crawler.py` — Crawler fortsetzen

Analysiert den Bestand der Markdown-Dateien und crawlt ergänzend:

| Parameter | Standard | Beschreibung |
|---|---|---|
| `MAX_PAGES` | `40.000` | Erhöhtes Limit für Resume |
| `BLOCKED_DOMAINS` | identisch mit `main.py` | Geblockte Subdomains |

Kategorisiert URLs in: frisch gecacht / veraltet / nur als Link bekannt, nicht gecrawlt.

Zusätzlich:

- bewertet `resume_crawler.py` alle gespeicherten `source_url`-Einträge und alle im Markdown gefundenen Links mit demselben RAG-Filter
- speichert die Entscheidungen in `data/url_decisions.db`
- zeigt bei `--dry-run` eine Filter-Zusammenfassung nach Gruenden

---

### `url_filter.py` — Gemeinsame URL-Policy

Zentrale Bewertungslogik fuer `main.py` und `resume_crawler.py`.

- normalisiert URLs strenger und entfernt Tracking-/Print-/Fragment-Varianten
- blockiert klar unnuetze RAG-Ziele wie Preview-/Backend-/Dev-Hosts, `/en`-Bereiche, Moodle-/Intranet-/Auth-Pfade, Medien-Dateien, technische Assets, `_processed_`-Dateien und bekannte Backend-Domains
- blockiert Low-Value-Pfade wie News-Archive, Galerie-/Tag-/Promo-Seiten deutlich aggressiver
- bewertet erlaubte Ziele als `allow_high_value` oder `allow_low_value`, damit Studium-/Bewerbungs-/Pruefungsseiten frueher gecrawlt werden
- verwendet zusaetzliche PDF-Heuristiken, um z.B. Ordnungen/Formulare/Faqs zu bevorzugen
- speichert jede Entscheidung in einer kleinen SQLite-Datenbank (`data/url_decisions.db`) inklusive Grund

---

### `hpc_vectorizer.py` — HPC-Vektorisierung

Erzeugt Dense + BM25 Sparse Vectors aus Markdown-Dateien und speichert sie als Parquet. Wenn `data/curated/` vorhanden und nicht leer ist, wird dieses bereinigte Korpus bevorzugt verarbeitet; andernfalls dient `data/ingested/` als Fallback. Alternativ kann per `RAG_SOURCE_DIR=/pfad/...` ein eigenes Quellverzeichnis gesetzt werden.

| Parameter | Standard | Beschreibung |
|---|---|---|
| `DENSE_MODEL` | `jinaai/jina-embeddings-v3` | 1024-dim multilinguales Embedding-Modell |
| `SPARSE_MODEL` | `Qdrant/bm25` | Statistisches BM25-Modell (CPU-only) |
| `DENSE_BACKEND` | `auto` | Bevorzugt `transformers/cuda`, erlaubt aber Fallback auf `fastembed` |
| `HF_LOCAL_FILES_ONLY` | `False` | Erzwingt rein lokale Hugging-Face-Dateien (`1/true/yes`) |
| `CHUNK_SIZE` | `1.000` | Max. Zeichen pro Chunk |
| `CHUNK_OVERLAP` | `200` | Überlappung zwischen Chunks (20%) |
| `DENSE_BATCH_SIZE` | `64` | Chunks pro GPU-Embedding-Aufruf |
| `SPARSE_BATCH_SIZE` | `256` | Chunks pro BM25-Aufruf |
| `RESUME_FROM_FILE` | `1` | Dateinummer für Neustart nach Abbruch |

Empfohlene Offline-Umgebung auf dem HPC:

```bash
export HF_HOME="$PWD/.hf_cache"
export RAG_SOURCE_DIR="$PWD/data/curated"
export DENSE_BACKEND=transformers
export HF_LOCAL_FILES_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**Zweistufige Chunking-Strategie:**
1. `MarkdownHeaderTextSplitter` — teilt an `#`, `##`, `###`; Überschriften-Hierarchie wird als `section_heading`-Metadatum (`H1 > H2 > H3`) bewahrt
2. `RecursiveCharacterTextSplitter` — teilt zu große Abschnitte weiter

**Parquet-Schema (19 Spalten):**

| Spalte | Typ | Beschreibung |
|---|---|---|
| `id` | string | UUID5 aus `source_url + chunk_index` (deterministisch) |
| `vector` | list\<float32\> | 1024-dim Dense-Einbettungsvektor |
| `source_url` | string | Ursprungs-URL |
| `title` | string | Seitentitel |
| `crawl_date` | string | ISO-8601-Datum |
| `content_type` | string | `html` oder `pdf` |
| `faculty` | string | Aus URL extrahiert: `/f1/`–`/f5/`, Subdomains |
| `language` | string | Sprachklassifikation aus der Kuratierung (`de`, `en`, `mixed`, `unknown`) |
| `quality_score` | string | Qualitätswert aus `clean_corpus.py` |
| `document_kind` | string | Z.B. `regulation`, `module_handbook`, `contact_service` |
| `source_family` | string | Grobe Zugehörigkeit, z.B. `faculty_4`, `internationales` |
| `document_group` | string | Feiner Gruppenschlüssel für zusammengehörige Dokumente |
| `topic_tags` | string | Mit `|` getrennte Themen-Tags, z.B. `bewerbung|rueckmeldung` |
| `section_heading` | string | Überschriften-Breadcrumb |
| `text` | string | Volltext des Chunks |
| `chunk_index` | int32 | Position im Dokument |
| `total_chunks` | int32 | Gesamtanzahl Chunks des Dokuments |
| `sparse_indices` | list\<int32\> | BM25 Token-IDs |
| `sparse_values` | list\<float32\> | BM25 Gewichte |

**Fakultätserkennung** aus der URL:
- Pfadbasiert: `/f1/` → `Fakultät I`, ..., `/f5/` → `Fakultät V`
- Subdomain-basiert: `karriere.*` → `Karriere`, `bibliothek.*` → `Bibliothek`, `international.*` → `International`

---

### Legacy-Hinweis zu `hsh_vectors_enriched.parquet`

`local_importer.py` prüft weiterhin zuerst, ob eine vorhandene `artifacts/hsh_vectors_enriched.parquet` aus einem älteren Lauf vorliegt. Ein Skript zur lokalen Sparse-Anreicherung gehört jedoch nicht mehr zum Repo; der Standardweg ist `hpc_vectorizer.py` → `artifacts/hsh_vectors.parquet` → `local_importer.py`.

---

### `local_importer.py` — Qdrant-Import

Lädt die Parquet-Datei in Qdrant. Erkennt automatisch:
1. vorhandene `artifacts/hsh_vectors_enriched.parquet`-Artefakte aus älteren Läufen
2. `artifacts/hsh_vectors.parquet` aus dem aktuellen HPC-Workflow

| Parameter | Standard | Beschreibung |
|---|---|---|
| `UPLOAD_BATCH` | `256` | Punkte pro Upsert-Aufruf |
| `RESUME_FROM_ROW` | `0` | Neustart-Zeilenindex bei Abbruch |

**Collection-Schema:**
- Benannte Dense-Vektoren: `"dense"` (1024-dim, Cosine)
- Sparse-Vektoren: `"sparse"` (BM25)

**Payload-Indizes** (für effiziente Suche und Augmentation):
- `text` — Volltext-Index (MULTILINGUAL-Tokenizer: deutsches Stemming, Kompositaaufspaltung)
- `faculty` — Keyword-Index (Fakultätsfilter)
- `chunk_index` — Integer-Index (Context Augmentation)
- `source_url` — Keyword-Index (Context Augmentation + Dedup)
- `document_kind` — Keyword-Index (Regelwerk/Formular/FAQ gezielt filterbar)
- `document_group` — Keyword-Index (verwandte Dokumente gruppiert nutzbar)
- `language` — Keyword-Index (deutsch/englisch trennbar)

---

### `query_assist.py` — Lokale Query-Bewertung und Präzisierung

Analysiert eine Nutzerfrage **vor** der eigentlichen Suche und erkennt:

- Definitionen / Abkürzungen (`K90`, `BPO`, `ECTS`, `SWS`, ...)
- Workflow-Fragen (`hochladen`, `online antrag`, `prüfungsanmeldung`, ...)
- Kontaktfragen (`wen kontaktiere`, `ansprechperson`, `service center`, ...)
- Fakultätsangaben in Ziffern- oder römischer Schreibweise (`fak 1`, `Fakultät IV`, ...)
- zu allgemeine Fragen, die mehrere HsH-Prozesse meinen könnten

Wichtige Aufgaben:

- erzeugt **Retrieval-Varianten**, ohne die ursprüngliche Bedeutung zu ändern
- baut **Präzisierungsoptionen** für breite Fragen auf
- normiert Fakultäten und bekannte Kürzel
- markiert Frage-Typen (`definition`, `workflow`, `contact`, `exam_registration`)

Beispiel:
- `Wie stelle ich einen Online-Antrag?`
  → wird als **allgemein** markiert und bietet konkrete Richtungen wie Bewerbung, Beurlaubung, iCMS/Prüfungsanmeldung oder Datenänderung an

---

### `glossary_index.py` — Glossar-, Legenden- und Abkürzungsindex

Extrahiert definierende Snippets aus `data/curated/`, z.B.:

- `K90` → `Klausur 90 Minuten`
- `SWS` → `Semesterwochenstunden`
- `BPO` → `Besonderer Teil der Prüfungsordnung`

Der Index dient zwei Zwecken:

- **Glossar-Treffer** für Definitionsfragen direkt in die Ergebnisliste einmischen
- **Similar-Term-Hinweise** für offensichtliche Schreibvarianten oder nahe Begriffe aus dem Korpus

Dadurch verbessert sich die Suche bei Fragen, die sonst nur viele Erwähnungen, aber nicht die eigentliche Definition finden würden.

---

### `retrieval_planner.py` — Optionaler LLM-Planer

Ein schmaler, kontrollierter LLM-Schritt **vor dem Retrieval**, der nur bei schwierigeren Fragen aktiviert wird.

Der Planer:

- beantwortet die Frage **nicht**
- darf die Bedeutung **nicht verändern oder verengen**
- liefert nur ein JSON mit:
  - `normalized_question`
  - `needs_clarification`
  - `clarification_options`
  - `canonical_hsh_terms`
  - `source_type_hints`
  - `query_variants`
  - `must_not_assume`

Typische Einsätze:

- mehrdeutige Workflow-Fragen
- kurze Definitionen / Kürzel
- Suche nach offiziellen HsH-Begriffen statt Alltagssprache

---

### `hybrid_search.py` — Guided-Hybrid-Suchpipeline

Guided-Hybrid-Suchpipeline mit Qdrant-nativer Fusion, Query-Varianten, Glossar-Hits, Reranking und Nachbar-Augmentation.

| Parameter | Standard | Beschreibung |
|---|---|---|
| `CANDIDATE_LIMIT` | `100` | Kandidaten pro Sucharm (Dense + Sparse je 100) |
| `TOP_K` | `8` | Finale Ergebnisse nach URL-Dedup |
| `DEDUP_BUFFER` | `4` | Faktor vor URL-Dedup: 8 × 4 = 32 Rohergebnisse |
| `MAX_PER_URL` | `2` | Max. Chunks pro `source_url` nach Dedup |
| `USE_RERANKER` | `True` | Cross-Encoder-Reranking aktivieren |
| `AUGMENT_TOP_N` | `3` | Nachbar-Chunks für die Top-N Treffer laden |
| `BLOCKED_URL_PREFIXES` | `("https://serwiss.bib.", ...)` | Gefilterte URL-Präfixe |

**Pipeline-Schritte:**

```
1. `query_assist.assess_query()` analysiert Typ, Kürzel, Fakultäten und Spezifität
2. Optionaler Retrieval-Planer ergänzt normierte HsH-Begriffe und Suchvarianten
3. `build_retrieval_queries()` erzeugt mehrere enge Suchpfade aus Originalfrage + Varianten
4. Dense- und Sparse-Embeddings werden gebatcht für alle Suchpfade berechnet
5. Qdrant-Prefetch + RRF-Fusion pro Suchpfad
6. Mehrere Suchpfade werden zusammengeführt (`matched_queries`)
7. Intent-spezifische Boosts für Definition / Workflow / Kontakt / Prüfungsanmeldung
8. Definitions-Rettung innerhalb derselben Quelle (`same-document rescue`)
9. Glossar-/Legenden-Hits und ähnliche Korpusterme werden beigemischt
10. URL-Dedup und Cross-Encoder-Reranking
11. Context Augmentation: für Top-3 Nachbar-Chunks nachladen
```

**Prozess-Trace**

`perform_guided_hybrid_search()` liefert nicht nur Treffer, sondern auch einen strukturierten `process_trace`, u.a. mit:

- Originalfrage und ggf. ausgewählter Präzisierung
- erkannte Typen / Kürzel / Fakultäten
- Planner-Hinweise
- Glossar-Treffer
- Schreibvarianten / Near-Matches
- allen tatsächlich verwendeten Suchpfaden

**`build_rag_context()`** formatiert die Ergebnisse für das Antwort-LLM:
```
[Quelle 1] Titel
URL: https://...
Stand: 2026-03-14
Fakultät: Fakultät IV
Abschnitt: Prüfungen > Prüfungsformen

<Chunk-Text ggf. mit Nachbar-Chunks>
```

---

### `rag_followup.py` — Begrenzte zweite Retrieval-Runde

Prüft nach der ersten Suche heuristisch, ob zusätzlicher Kontext nötig ist, z.B. bei:

- Definitionslücken
- unvollständigen Verfahrensbeschreibungen
- Vergleichen zwischen Fakultäten

Wichtige Eigenschaft:

- die zweite Runde ist **begrenzt**
- Standardpfad ist aktuell **heuristisch**, nicht nochmal ein eigener LLM-Planer
- für zeitkritische Fragen wie `heute`, `aktuell`, `jetzt` wird bewusst **nicht** endlos nachgeladen

Nachlade-Aktionen:

- `neighbor_chunks`
- `full_section`
- `same_group_documents`
- `new_search`

---

### `conversation_memory.py` — Kompaktes Kurzgedächtnis

Speichert die letzten **maximal fünf beantworteten Turns** als kurze Zusammenfassungen.

Wichtig:

- das Kurzgedächtnis ist **nur Gesprächskontext**
- die **einzigen verbindlichen Quellen** bleiben die aktuellen HsH-Kontextblöcke der jeweiligen Runde

Gespeichert werden u.a.:

- gekürzte Frage
- gekürzte Antwort
- erkannte Fakultäten / Intents
- ggf. ein kompakter Retrieval-Fokus

---

### `turn_router.py` — Router für Anschlussfragen

Erkennt Anschlussfragen, die **keine neue RAG-Suche** brauchen, z.B.:

- `der Link öffnet nicht`
- `schick den Link nochmal`
- `welche Quelle meinst du`

Dafür wird pro Assistant-Turn ein kleiner Zustand mitgeführt:

- zuletzt genannte URLs
- Quell-URLs
- Quelltitel
- ausgewählte Query

So werden Link-/Quellen-Follow-ups lokal beantwortet, statt mit einer neuen semantischen Suche zu halluzinieren.

---

### `hsh_web_app.py` — Streamlit Web-App

| Parameter | Standard | Beschreibung |
|---|---|---|
| `RAG_TOP_K` | `6` | Kontext-Chunks pro Anfrage |
| `TEMPERATURE` | `0.0` | Deterministisch, keine Kreativitäts-Halluzinierung |

**Rollenanpassung:** Das LLM ändert seinen Sprachstil je nach Nutzerrolle (Studierender, Mitarbeitender, Lehrender, Besucher).

**Fakultätsfilter (OR-Logik):**
```
faculty == gewählte Fakultät  ODER  faculty == ""
```
Zentrale Einrichtungen (ohne Fakultätszuordnung) erscheinen immer — unabhängig vom Filter.

**Zusätzliche Web-App-Mechanik:**

- optionaler Retrieval-Planer vor der Suche
- Präzisierungsformular für breite Fragen
- 5-Turn-Kurzgedächtnis im Sidebar-/Status-Kontext
- Prozess-Expander mit Query-Analyse, Planner-Hinweisen, Glossar-Hits und Suchpfaden
- Timing-Expander mit Zeiten für Planer, Retrieval, Follow-up-Heuristik, Prompt-Bau und Antwort-LLM
- lokaler Router für Link-/Quellen-Anschlussfragen

**Veralterungswarnung:** Wenn die älteste gefundene Quelle > 180 Tage alt ist, enthält der System-Prompt automatisch eine Empfehlung zur Nachprüfung.

**System-Prompt-Regeln:**
1. Antworten ausschließlich auf Basis des Kontexts
2. Keine Spekulation oder Erfindung
3. Offensichtliche Schreibvarianten nur transparent benennen, nicht stillschweigend umdeuten
4. Bei fehlendem Kontext: definierte Standardantwort
5. Widersprüche zwischen Quellen explizit benennen
6. Alte Quellen (erkennbar am `Stand:`-Feld) kenntlich machen
7. Quellenangabe am Ende jeder Antwort (Titel, URL, Abschnitt, Datum)

---

### `hsh_chatbot.py` — CLI-Chatbot

| Parameter | Standard | Beschreibung |
|---|---|---|
| `EMBED_MODEL` | `jinaai/jina-embeddings-v3` | Dense-Embedding |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse-Embedding |
| `RAG_TOP_K` | `4` | Kontext-Chunks pro Anfrage |
| `DEBUG_PROMPT` | `True` | Vollständigen LLM-Prompt ausgeben |

Dynamische Modellauswahl: Beim Start werden alle verfügbaren Modelle von der GWDG-API abgefragt. Reasoning-Modelle (DeepSeek R1 o.ä.) zeigen ihren Denkprozess in einem separaten `[Thinking]`-Block.

Zusätzlich nutzt der Chatbot:

- ein kompaktes 5-Turn-Kurzgedächtnis
- lokale Präzisierungsdialoge für allgemeine Fragen
- optional den Retrieval-Planer vor der Suche
- eine begrenzte heuristische Follow-up-Retrieval-Runde
- einen lokalen Turn-Router für Link-/Quellen-Follow-ups
- einen ausführlichen Terminal-Trace:
  - `[Systemprozess]`
  - `[Leistungsdaten]`
  - ggf. Vorschau des Kurzgedächtnisses

---

### `delete_qdrant.py` — Collection zurücksetzen

Löscht die bestehende Collection vollständig und legt sie mit allen Payload-Indizes neu an. **Muss vor `local_importer.py` ausgeführt werden**, wenn sich das Schema geändert hat oder ein Neuimport gewünscht ist.

> ⚠️ Unwiderruflich — alle Vektoren und Metadaten gehen verloren.

---

### `check_quality.py` — Qualitätsprüfung

Analysiert alle Markdown-Dateien auf Qualitätsprobleme:

| Kriterium | Schwellwert |
|---|---|
| Mindestwörter (HTML) | 30 |
| Mindestwörter (PDF) | 50 |
| Erkannte Fehlermuster | „404", „Seite nicht gefunden", „Zugriff verweigert", … |
| Duplikate | Ältere Version desselben URL-Slugs |
| Sprachheuristik | `de`, `en`, `mixed`, `unknown` |
| Preview-/Backend-Links im Body | werden als Qualitätsproblem markiert |

### `clean_corpus.py` — Kuratierung des Rohkorpus

Transformiert `data/ingested/` in ein bereinigtes `data/curated/`.

- entfernt geblockte Links und offensichtliche Boilerplate
- ergänzt Qualitäts- und Gruppen-Metadaten
- verwirft problematische Dateien vor der Vektorisierung
- schreibt mit `data/curated_report.json` einen maschinenlesbaren Prüfbericht pro Datei

---

## Technische Architektur

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Nutzerfrage                                   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  turn_router.py                                                    │
│  • Link-/Quellen-Follow-ups lokal beantworten                      │
│  • z.B. "der Link öffnet nicht", "welche Quelle meinst du"         │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │ falls KEIN lokaler Follow-up-Turn
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  query_assist.py + retrieval_planner.py                              │
│  • Frage klassifizieren (definition / workflow / contact / ...)      │
│  • Kürzel und Fakultäten erkennen                                    │
│  • Präzisierung nötig?                                               │
│  • optionaler LLM-Planer erzeugt normierte Suchpfade                 │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  hybrid_search.py — Guided-Hybrid-Retrieval                          │
│                                                                      │
│  Anfrage → mehrere enge Suchpfade                                    │
│  Anfrage → Dense-Embeddings (Jina v3, 1024-dim)                      │
│  Anfrage → BM25-Sparse-Embeddings (Qdrant/bm25)                      │
│                                                                      │
│  ┌──────────────────────────┐  ┌───────────────────────────────┐     │
│  │  Dense Prefetch          │  │  Sparse Prefetch (BM25)       │     │
│  │  using="dense", limit=100│  │  using="sparse", limit=100    │     │
│  └────────────┬─────────────┘  └──────────────┬────────────────┘     │
│               └────────────────┬──────────────┘                      │
│                                ▼                                     │
│              Qdrant-native RRF-Fusion pro Suchpfad                   │
│                                │                                     │
│                                ▼                                     │
│              Mehrpfad-Fusion + URL-/Host-Filter                      │
│                                │                                     │
│                                ▼                                     │
│              Definition-/Workflow-/Kontakt-Boosts                    │
│                                │                                     │
│                                ▼                                     │
│              Glossar-Hits + Same-Document-Definition-Rescue          │
│                                │                                     │
│                                ▼                                     │
│              URL-Dedup + Cross-Encoder-Reranking                     │
│                                │                                     │
│                                ▼                                     │
│              Context Augmentation (Nachbar-Chunks Top-3)             │
│                                │                                     │
│              Top-Kontext + Prozess-Trace                             │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  rag_followup.py — begrenzte zweite Retrieval-Runde                  │
│  • neighbour_chunks / full_section / same_group_documents / new_search│
│  • heuristisch, maximal eine zusätzliche Runde                       │
│  • keine Endlosschleifen für "heute"/"aktuell"/"jetzt"              │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  hsh_web_app.py / hsh_chatbot.py — Prompt-Aufbau                     │
│                                                                      │
│  [System]  Rolle + Fakultätskontext + Regelwerk                      │
│            + ggf. Veralterungswarnung (>180 Tage)                    │
│            + Kurzgedächtnis (max. 5 komprimierte Turns)              │
│                                                                      │
│  [User]    Kontext aus offiziellen HsH-Dokumenten:                   │
│            [Quelle 1] Titel | URL | Stand | Abschnitt | Text         │
│            ---                                                       │
│            [Quelle 2] ...                                            │
│            ---                                                       │
│            Frage: {nutzerfrage}                                      │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GWDG ChatAI API (OpenAI-kompatibel)                                 │
│  Modelle: GPT-4o, Llama, DeepSeek, openai-gpt-oss, …                 │
│  Temperature: 0.0 (deterministisch)                                  │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Ausgabe                                                             │
│  • Präzisierung oder Router-Antwort                                   │
│  • [Denkprozess] (nur Reasoning-Modelle, in Expander)               │
│  • Antworttext                                                       │
│  • Quellenangaben: Titel — URL — Abschnitt (Stand: Datum)           │
│  • Prozess-Trace + Leistungsdaten (Web/CLI)                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Leistung und Laufzeiten

Die aktuelle Pipeline ist **nicht nur API-limitiert**. In der Praxis kommt ein großer Teil der Wartezeit von lokalen Retrieval-Schritten auf dem eigenen Rechner.

### Beobachtete Engpässe

- **Qdrant selbst** ist vergleichsweise schnell.
- Die Hauptkosten entstehen lokal durch:
  - Dense-Query-Embeddings (`jinaai/jina-embeddings-v3`)
  - ONNX-Reranking (`jina-reranker-v2-base-multilingual`)
- Der **erste Query nach Start** ist deutlich langsamer als Folgeanfragen (Warm-up-Effekt).

### Gemessene End-to-End-Zeiten

| Frage | Planer-LLM | Retrieval | Antwort-LLM | Gesamt |
|---|---:|---:|---:|---:|
| `Wo finde ich das Bewerbungsportal ...` | 4.64 s | 43.68 s | 4.11 s | 52.42 s |
| `Was bedeutet BPO?` | 3.68 s | 13.23 s | 5.35 s | 22.26 s |
| `Wie kann ich meine Anschrift ... ändern?` | 6.82 s | 11.30 s | 12.50 s | 30.62 s |

### Warm-Cache-Effekt

Gleiche Retrieval-Frage, dreimal direkt hintereinander:

| Lauf | Zeit |
|---|---:|
| 1 | 43.78 s |
| 2 | 16.09 s |
| 3 | 11.84 s |

### Profiling-Ergebnis

Bei einer bereits angewärmten Guided-Hybrid-Suche lagen die Hauptkosten ungefähr hier:

- Dense-Embedding (ONNX / FastEmbed): ca. **6.95 s**
- Reranker (ONNX): ca. **6.24 s**
- Qdrant HTTP + Query: nur ca. **0.25 s**

**Praxis-Fazit:** Wenn das System langsam wirkt, liegt das oft eher an lokaler Embedding-/Reranker-Inferenz als an der GWDG-API.

Empfohlene Optimierungen:

- Dense-Embedder und Reranker beim Start „anwärmen"
- Reranking bei einfachen Hochkonfidenz-Fragen optional überspringen
- Query-Embeddings cachen
- unnötig viele Retrieval-Varianten vermeiden

---

## Konfigurationsparameter

### Schnellreferenz: alle konfigurierbaren Werte

| Skript | Parameter | Standard | Bedeutung |
|--------|-----------|----------|-----------|
| `main.py` | `MAX_PAGES` | 10.000 | Max. gecrawlte Seiten |
| `main.py` | `MAX_AGE_DAYS` | 7 | Cache-TTL in Tagen |
| `main.py` | `RATE_LIMIT_SECONDS` | 0.5 | Pause zwischen Requests |
| `main.py` | `BLOCKED_DOMAINS` | `{serwiss.bib..., typo3backend-live...}` | Geblockte Subdomains |
| `resume_crawler.py` | `MAX_PAGES` | 40.000 | Erhöhtes Limit |
| `url_filter.py` | `BLOCKED_DOMAINS` | `{serwiss.bib..., typo3backend-live...}` | Geblockte/technische Subdomains |
| `url_filter.py` | `BLOCKED_APP_HOSTS` | `{moodle.hs-hannover.de, intranet.hs-hannover.de}` | Nicht-oeffentliche App-Hosts |
| `url_filter.py` | `BLOCKED_AUTH_PATH_MARKERS` | `("/login", "/logout", ...)` | Auth-/Login-Pfade |
| `url_filter.py` | `BROKEN_URL_MARKERS` | `("*", "|", ...)` | Kaputte/artefaktbehaftete URLs blockieren |
| `url_filter.py` | `DECISION_DB_PATH` | `data/url_decisions.db` | SQLite-Datei fuer URL-Entscheidungen |
| `clean_corpus.py` | `--keep-english` | `False` | Englischsprachige Dateien standardmaessig verwerfen |
| `hpc_vectorizer.py` | `CHUNK_SIZE` | 1.000 | Max. Chunk-Zeichen |
| `hpc_vectorizer.py` | `CHUNK_OVERLAP` | 200 | Überlappung |
| `hpc_vectorizer.py` | `DENSE_BATCH_SIZE` | 64 | GPU-Batch |
| `hpc_vectorizer.py` | `SPARSE_BATCH_SIZE` | 256 | CPU-Batch |
| `hpc_vectorizer.py` | `RESUME_FROM_FILE` | 1 | Neustart-Dateinummer |
| `hpc_vectorizer.py` | `DENSE_BACKEND` | `auto` | `transformers/cuda` bevorzugen, sonst Fallback |
| `hpc_vectorizer.py` | `HF_HOME` | unset | Bevorzugter persistenter Modellcache für `transformers` und `fastembed` |
| `hpc_vectorizer.py` | `FASTEMBED_CACHE_PATH` | `HF_HOME` oder `~/.cache/fastembed` | Fallback-Cache für `fastembed`, falls `HF_HOME` nicht gesetzt ist |
| `hpc_vectorizer.py` | `HF_LOCAL_FILES_ONLY` | `False` | Nur lokale Hugging-Face-Artefakte verwenden |
| `hpc_vectorizer.py` | `RAG_SOURCE_DIR` | automatisch | Bevorzugt `data/curated/`, sonst `data/ingested/` |
| `local_importer.py` | `UPLOAD_BATCH` | 256 | Punkte pro Upsert |
| `local_importer.py` | `RESUME_FROM_ROW` | 0 | Neustart-Zeilenindex |
| `hybrid_search.py` | `CANDIDATE_LIMIT` | 100 | Kandidaten pro Arm |
| `hybrid_search.py` | `TOP_K` | 8 | Finale Ergebnisse |
| `hybrid_search.py` | `DEDUP_BUFFER` | 4 | Überabtastung vor URL-Dedup |
| `hybrid_search.py` | `MAX_PER_URL` | 2 | Max. Chunks/URL |
| `hybrid_search.py` | `USE_RERANKER` | `True` | Reranking an/aus |
| `hybrid_search.py` | `AUGMENT_TOP_N` | 3 | Nachbar-Augmentation |
| `hybrid_search.py` | `BLOCKED_URL_PREFIXES` | `("https://serwiss.bib.", ...)` | URLs ausblenden, die im Retrieval nicht erscheinen sollen |
| `conversation_memory.py` | `MAX_MEMORY_TURNS` | 5 | Anzahl komprimierter Antwort-Turns im Kurzgedächtnis |
| `hsh_web_app.py` | `RAG_TOP_K` | 6 | Kontext-Chunks |
| `hsh_web_app.py` | `TEMPERATURE` | 0.0 | Deterministische Antwortgenerierung |
| `hsh_web_app.py` | `FASTEMBED_CACHE_PATH` | `~/.cache/fastembed` | Persistenter Modellcache für Dense, BM25 und Reranker |
| `hsh_chatbot.py` | `RAG_TOP_K` | 4 | Kontext-Chunks (CLI) |
| `hsh_chatbot.py` | `TEMPERATURE` | 0.0 | Deterministische Antwortgenerierung |
| `hsh_chatbot.py` | `DEBUG_PROMPT` | `True` | Prompt ausgeben |
| `hsh_chatbot.py` | `FASTEMBED_CACHE_PATH` | `~/.cache/fastembed` | Persistenter Modellcache für Dense, BM25 und Reranker |

---

## Ideen zur Weiterentwicklung

Die folgenden Punkte sind offen oder ausbaubar. Bereits umgesetzt sind unter anderem: URL-Filter für englische `/en`-Bereiche, Moodle/Intranet/Auth-Pfade, Backend-/Preview-Hosts, Medien- und Office-Dateien; Korpus-Kuratierung mit `--keep-english` als bewusstem Override; Guided Retrieval mit Query-Analyse, Glossar-Hits, optionalem Retrieval-Planer; begrenztes Follow-up-Nachladen; sowie Web-/CLI-Validierung mit zweistufiger Evidenzprüfung.

### 1. Qualitätssicherung ausbauen

**Validierungsbasis erweitern:**
Das Repository enthält bereits eine feste Fallbasis und einen zweistufigen Judge-Lauf (`raghsh-eval --all`). Sinnvolle nächste Schritte sind mehr kuratierte Fragen, mehr Fakultäts-/Rollenfälle, Grenzfälle für Klarstellungsdialoge und akzeptierte Alternativquellen pro Fall.

**CI-Integration:**
Die bestehenden Unit-Tests laufen mit `pytest`. Für Releases könnte zusätzlich ein kleiner, schneller Validierungs-Subset in CI laufen; vollständige LLM-/Qdrant-Evals bleiben wegen Laufzeit, Kosten und API-Limits eher ein manueller oder geplanter Lauf.

**Retrieval-Monitoring:**
Logging der RRF-Scores, Reranker-Scores, Trefferquellen und Failure-Typen in eine Zeitreihe (z.B. SQLite oder Prometheus) würde Trendbrüche sichtbar machen, etwa wenn ein Neucrawl schlechte Daten einspielt oder ein Modell-Update das Retrieval verändert.

---

### 2. Datenqualität durch LLMs verbessern

**LLM-gestützte Chunk-Bereinigung:**
Die aktuelle Qualitätsprüfung nutzt heuristische Regeln für Sprache, Länge, Fehlerseiten, Backend-Links und Dokumenttypen. Ergänzend könnte ein schnelles Sprachmodell (z.B. `Llama-3.1-8B`) jede Markdown-Datei bewerten:
- Enthält diese Seite tatsächlich informative Inhalte?
- Ist der Text kohärent oder besteht er aus Navigationselementen?
- Ist die Seite auf Deutsch verfasst?

Prompt-Vorlage:
```
Bewerte diesen Text auf einer Skala von 1 (wertlos) bis 5 (sehr informativ).
Antworte nur mit der Zahl und einer einzeiligen Begründung.

Text: {chunk_text[:500]}
```

Seiten unter einem Schwellwert (z.B. < 3) werden vor der Vektorisierung gefiltert.

**Automatische Metadaten-Extraktion:**
Das System erzeugt bereits heuristische Metadaten wie `language`, `document_kind`, `source_family`, `document_group` und `topic_tags`. Ein LLM könnte daraus präzisere oder vollständigere strukturierte Metadaten ableiten:
- Studiengang-Tags (`bachelor`, `master`, `weiterbildung`)
- Themen-Tags (`prüfungen`, `bewerbung`, `stundenplan`, `finanzen`)
- Zielgruppen-Tags

Diese Tags erweitern die Payload und ermöglichen präzisere Filter in der Suche.

**Volltext-Normalisierung:**
Deutsche Komposita werden von Suchmaschinen oft nicht aufgespalten. Ein LLM kann Komposita in Chunks vorverarbeiten: `„Modulhandbuchseite"` → `„Modulhandbuch Seite"`, was BM25-Treffer verbessert.

---

### 3. Retrieval weiter verfeinern

**Domain- und Dokumentfilter empirisch nachschärfen:**
Der URL-Filter schließt bereits viele technische oder wenig nützliche Quellen aus. Offen bleibt eine empirische Prüfung weiterer Domain-/Pfadgruppen, z.B. ob reine Publikationslisten, Paper-Inhalte oder sehr alte News-Seiten für die Zielantworten relevant genug sind.

**Query-Expansion:**
Es gibt bereits lokale Retrieval-Varianten und Planner-Hinweise. Ausbaubar wäre eine systematischere Erweiterung der Suchanfrage um Synonyme und verwandte Begriffe vor der Vektorisierung, z.B. `„Urlaubssemester"` → `„Beurlaubung Exmatrikulation Studienunterbrechung"`. Das erhöht den Recall besonders für BM25.

**Chunk-Größenoptimierung:**
Aktuell: 1.000 Zeichen mit 200 Zeichen Überlappung. Experimentell wären 1.500 Zeichen oder adaptive Chunk-Größen basierend auf dem Dokumenttyp, weil PDFs oft längere zusammenhängende Abschnitte enthalten als HTML-Seiten.

**Semantisches Chunking:**
Statt fester Zeichengrenzen könnten Chunks an semantischen Grenzen geteilt werden, z.B. mit einem Sentence-Transformer, der Ähnlichkeiten aufeinanderfolgender Sätze misst.

**Frischegewichtung:**
`crawl_date` als Ranking-Signal nutzen: neuere Seiten erhalten einen leichten Relevanzbonus. Wichtig für Seiten mit häufig aktualisierten Inhalten wie Fristen, Ansprechpartnern und Prüfungsinformationen.

---

### 4. Kontextsteuerung und Feedback

**Vollständige Dokumente bei Bedarf:**
Das System lädt bereits Nachbar-Chunks, ganze Abschnitte und zusätzliche Suchergebnisse nach. Für bestimmte Regelwerke oder Legenden könnte ein Modus ergänzt werden, der gezielt alle Chunks derselben `source_url` lädt, wenn die Antwort sonst ohne Dokumentkontext unvollständig bleibt.

**Feedback-Loop:**
Antworten könnten künftig positiv oder negativ bewertet werden. Negativ bewertete Anfragen könnten anschließend in einen Goldstandard-Datensatz einfließen und helfen, Schwächen systematisch zu identifizieren.

**Antwortanalyse im Betrieb:**
Häufige Rückfragen, Link-Follow-ups und Quellenprobleme könnten aggregiert werden, um gezielt schlechte Dokumentgruppen, fehlende Synonyme oder unklare UI-Flows zu finden.
