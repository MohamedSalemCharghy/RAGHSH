# RAGHSH Validierungssystem

Das **RAGHSH Validierungssystem** prüft den HsH-RAG-Chatbot mit einer festen Fallbasis aus `validation_cases.json`. Jeder Fall wird gegen dieselbe RAG-Pipeline ausgeführt, die auch Web-App und CLI verwenden. Danach bewertet ein Evaluator-Modell die Antwort in zwei Stufen: zuerst gegen Referenz und Fallvertrag, danach gegen die tatsächlich aus Qdrant abgerufenen Chunks.

## Inhaltsverzeichnis

1. [Konzept: Warum Validierung?](#konzept-warum-validierung)
2. [Systemübersicht](#systemübersicht)
3. [Verzeichnisstruktur](#verzeichnisstruktur)
4. [Validierungsfälle](#validierungsfälle)
5. [Stufe 1: Antwortbewertung](#stufe-1-antwortbewertung)
6. [Stufe 2: Evidenzprüfung](#stufe-2-evidenzprüfung)
7. [Pipeline-Konfiguration](#pipeline-konfiguration)
8. [Nutzung in Streamlit](#nutzung-in-streamlit)
9. [Nutzung über die CLI](#nutzung-über-die-cli)
10. [Ergebnisdateien](#ergebnisdateien)
11. [Diagnose und Interpretation](#diagnose-und-interpretation)
12. [Ideen zur Weiterentwicklung](#ideen-zur-weiterentwicklung)

---

## Konzept: Warum Validierung?

Ein RAG-System kann auf verschiedene Arten falsch liegen:

- **Retrieval-Fehler:** Die richtige Quelle oder der richtige Chunk wird nicht gefunden.
- **Grounding-Fehler:** Die richtige Quelle ist da, aber das Modell nutzt sie falsch.
- **Referenz-Fehler:** Die Validierungsfrage ist zu eng modelliert, obwohl mehrere offizielle HsH-Antwortpfade gültig sind.
- **Dialog-Fehler:** Das System hätte nachfragen oder ein fehlendes Detail klar abgrenzen müssen.
- **Quellen-Fehler:** Die Antwort nennt Quellen, die den konkreten Claim nicht tragen.

Darum bewertet das System nicht nur einen Gesamtscore. Es speichert Antwort, Quellen, Chunks, Prozess-Trace, Timings, Stufe-1-Bewertung, Stufe-2-Bewertung und Diagnose-Typen pro Fall.

---

## Systemübersicht

```
Validierungsfall
    │
    ▼
RAG-Pipeline
    ├─ Query Assist optional
    ├─ Retrieval Planner optional
    ├─ Qdrant Hybrid Search
    └─ RAG Follow-up optional
    │
    ▼
Chatbot-Antwort
    │
    ├─ Stufe 1: Antwort gegen Referenz und Fallvertrag
    │
    └─ Stufe 2: Antwort gegen abgerufene Qdrant-Chunks
    │
    ▼
JSON-Ergebnisse + Summary + Diagnose
```

Die Validierung nutzt den gemeinsamen Runtime-Einstieg `prepare_chat_turn(...)` aus `web_app_runtime.py`. Dadurch wird kein künstlicher Testpfad geprüft, sondern der echte RAG-Ablauf.

---

## Verzeichnisstruktur

```
src/hsh_scraper/evals/
├── README.md                  # Diese Dokumentation
├── validation_cases.json      # Feste Fallbasis
├── validation_system.py       # Streamlit-Validierung und Kernlogik
├── validation_cli.py          # Terminal-Runner für Validierungsläufe
├── evidence_validation.py     # Stufe 2: Grounding- und Evidenzprüfung
└── __init__.py

artifacts/evals/results/
└── <run_id>/
    ├── manifest.json          # Run-Metadaten und Pipeline-Konfiguration
    ├── summary.json           # Aggregierte Scores und Diagnose-Zählungen
    └── qXX.json               # Einzelfall mit Antwort, Chunks, Stufe 1, Stufe 2
```

---

## Validierungsfälle

Die Datei `validation_cases.json` enthält feste Fälle. Jeder Fall kann folgende Felder enthalten:

| Feld | Bedeutung |
|---|---|
| `id` | Stabile Fall-ID, z.B. `q05` |
| `question` | Nutzerfrage |
| `reference_answer` | Erwartete fachliche Antwort |
| `required_facts` | Pflichtfakten, die eine gute Antwort enthalten muss |
| `optional_facts` | Zusatzfakten, die hilfreich, aber nicht immer zwingend sind |
| `forbidden_claims` | Aussagen, die nicht vorkommen dürfen |
| `source_url` | Primäre Goldquelle |
| `source_policy` | Quellenregel: `exact_gold_source`, `accepted_sources` oder `official_hsh_any` |
| `accepted_source_urls` | Weitere erlaubte Quellen |
| `answer_variants` | Alternative gültige Antwortpfade |
| `clarification` | Fest definierte Klärung für Mehrdeutigkeitsfälle |
| `tags` | Markierungen wie `negativfall` oder `fehlendes-detail` |
| `evaluation_notes` | Hinweise für den Judge und spätere Auswertung |

Wichtig: Wenn eine Frage mehrere offizielle HsH-Antworten zulässt, sollte der Fall mit `answer_variants`, `accepted_source_urls` und passenden `required_facts` modelliert werden. Sonst bewertet der Judge eine belegte Alternativantwort eventuell unfair.

---

## Stufe 1: Antwortbewertung

Stufe 1 bewertet die finale Chatbot-Antwort gegen den Fallvertrag und die Referenzantwort.

Der Evaluator sieht:

- den kompletten Fall aus `validation_cases.json`
- die finale Chatbot-Antwort
- den Dialogverlauf
- ausgewählte Quellenmetadaten
- Prozessinformationen wie gewählte Suchformulierung und erkannte Intents

Bewertungskriterien:

| Kriterium | Frage |
|---|---|
| `correctness` | Ist die Antwort fachlich richtig? |
| `completeness` | Sind die Pflichtfakten vollständig genug enthalten? |
| `faithfulness` | Erfindet die Antwort keine Details? |
| `source_use` | Passt die Antwort zur erwarteten Quellenpolitik? |
| `dialog_behavior` | Klärt oder verweigert die Antwort passend zum Fall? |

Jedes Kriterium erhält 0 bis 5 Punkte. Der Stufe-1-Score wird auf 100 skaliert.

---

## Stufe 2: Evidenzprüfung

Stufe 2 prüft nicht primär die Referenzantwort, sondern die tatsächliche Evidenz aus Qdrant.

Der Evaluator sieht:

- die finale Chatbot-Antwort
- die tatsächlich abgerufenen Qdrant-Chunks aus `results`
- Titel, URL, Abschnitt, Datum, Rang und Score der Chunks
- die Quellenpolitik des Falls
- die Retrieval Queries aus dem Prozess-Trace

Bewertungskriterien:

| Kriterium | Frage |
|---|---|
| `source_retrieval` | Wurde die erwartete oder akzeptierte Quelle gefunden? |
| `evidence_sufficiency` | Reichen die Chunks für eine belastbare Antwort? |
| `answer_grounding` | Sind die Claims der Antwort in den Chunks belegt? |
| `citation_support` | Passen Quellenangaben und Claims zusammen? |
| `refusal_behavior` | Wird bei fehlender Evidenz korrekt abgegrenzt? |

Mögliche Verdikte sind zum Beispiel:

- `fully_grounded`
- `mostly_grounded`
- `partially_grounded`
- `unsupported`
- `wrong_source`
- `retrieval_failed`
- `correct_refusal`
- `wrong_refusal`

Stufe 2 hilft dabei, Retrieval-Probleme von Antwort-Problemen zu trennen.

---

## Pipeline-Konfiguration

Vor einem Run kann die RAG-Pipeline konfiguriert werden. Diese Konfiguration wird im Run gespeichert.

| Parameter | Standard | Bedeutung |
|---|---:|---|
| `top_k` | Streamlit/Validierung: 6, Chatbot-CLI: 4 | Anzahl finaler Qdrant-Treffer/Chunks |
| `query_assist_enabled` | `true` | Lokale Frageanalyse, Intents, Fakultäten, Kürzel und Suchvarianten |
| `retrieval_planner_enabled` | `true` | Optionaler LLM-Planer für schwierige Suchfragen |
| `rag_followup_enabled` | `true` | Zweite/ergänzende Retrieval-Runde bei unvollständiger Evidenz |

`turn_router.py` ist bewusst nicht Teil der Validierungs-Konfiguration. Der Router ist für Anschlussfragen im Chat gedacht, zum Beispiel „welche Quelle meinst du?“ oder „der Link geht nicht“. Die festen Strict-Dialog-Fälle prüfen dagegen den RAG-Kern.

Beispielkonfiguration:

```json
{
  "top_k": 6,
  "query_assist_enabled": true,
  "retrieval_planner_enabled": true,
  "rag_followup_enabled": true
}
```

Für Ablation-Tests kann man gezielt Komponenten ausschalten:

- Alles aktiv: normales System
- Ohne Query Assist: reine Suche mit Originalfrage
- Ohne Retrieval Planner: keine LLM-Suchplanung
- Ohne RAG Follow-up: keine Kontext-Erweiterung nach der ersten Suche
- `top_k=6` gegen `top_k=10`: Einfluss der Kontextgröße prüfen

---

## Nutzung in Streamlit

Start:

```bash
python3 -m streamlit run src/hsh_scraper/hsh_web_app.py
```

In der Sidebar:

1. Modus `Validierungssystem` wählen.
2. Chatbot-Modell auswählen.
3. Evaluator-Modell auswählen.
4. Unter `RAG-Pipeline` die Parameter setzen:
   - `Qdrant Top-K`
   - `Query Assist aktiv`
   - `Retrieval Planner aktiv`
   - `RAG Follow-up aktiv`
5. `Aktuellen Fall ausführen` oder `Alle Fälle ausführen` starten.

Sobald ein Run gestartet wurde, ist die Pipeline-Konfiguration für diesen Run fest. Für eine neue Konfiguration muss eine neue Validierung gestartet werden.

---

## Nutzung über die CLI

Alle Fälle mit Standardkonfiguration:

```bash
python -m hsh_scraper.evals.validation_cli --all
```

Ein einzelner Fall:

```bash
python -m hsh_scraper.evals.validation_cli --case q05
```

Andere Kontextgröße:

```bash
python -m hsh_scraper.evals.validation_cli --all --top-k 10
```

Ablation ohne Query Assist:

```bash
python -m hsh_scraper.evals.validation_cli --all --no-query-assist
```

Ablation ohne Retrieval Planner und ohne Follow-up:

```bash
python -m hsh_scraper.evals.validation_cli --all --no-retrieval-planner --no-rag-followup
```

Wichtige CLI-Optionen:

| Option | Bedeutung |
|---|---|
| `--all` | Alle Fälle ausführen |
| `--case qXX` | Nur einen Fall ausführen |
| `--case-index N` | Fall nach Position ausführen |
| `--list-cases` | Fallliste anzeigen |
| `--chatbot-model` | Chatbot-Modell festlegen |
| `--evaluator-model` | Evaluator-Modell festlegen |
| `--qdrant-url` | Qdrant-Adresse |
| `--top-k` | Anzahl finaler Retrieval-Treffer |
| `--no-query-assist` | Query Assist deaktivieren |
| `--no-retrieval-planner` | Retrieval Planner deaktivieren |
| `--no-rag-followup` | RAG Follow-up deaktivieren |
| `--max-requests-per-minute` | Rate-Limit für API-Aufrufe |

---

## Ergebnisdateien

Jeder Run wird unter `artifacts/evals/results/<run_id>/` gespeichert.

### `manifest.json`

Enthält:

- Run-ID
- Chatbot-Modell
- Evaluator-Modell
- Modus
- Pfad zur Fallbasis
- Fallanzahl
- Pipeline-Konfiguration
- Erstellungszeitpunkt

### `summary.json`

Enthält:

- Anzahl abgeschlossener Fälle
- Stufe-1-Gesamtscore
- Stufe-1-Durchschnittswerte
- Stufe-2-Evidenzscore
- Stufe-2-Durchschnittswerte
- zweistufigen Gesamtscore
- Verdikt-Zählungen
- Diagnose-Zählungen
- Anzahl der Fälle mit empfohlener fachlicher Nachprüfung
- Pipeline-Konfiguration

### `qXX.json`

Enthält pro Fall:

- Fallvertrag
- Dialogverlauf
- Chatbot-Antwort
- `process_trace`
- `results` mit den abgerufenen Qdrant-Chunks
- Timings
- Stufe-1-Bewertung
- Stufe-2-Bewertung
- Diagnose
- Pipeline-Konfiguration

---

## Diagnose und Interpretation

Wichtige Diagnose-Typen:

| Diagnose | Interpretation |
|---|---|
| `retrieval_failed` | Die erwartete Evidenz wurde nicht gefunden. |
| `wrong_source_retrieved` | Es wurden Quellen gefunden, aber nicht die richtige oder akzeptierte Quelle. |
| `answer_ignored_evidence` | Die Evidenz war vorhanden, wurde aber falsch oder gar nicht genutzt. |
| `missing_required_fact` | Ein Pflichtfakt fehlt. |
| `missing_optional_fact` | Ein Zusatzfakt fehlt; meist weniger kritisch. |
| `hallucination` | Die Antwort enthält erfundene oder verbotene Claims. |
| `citation_problem` | Quellenangaben tragen die Claims nicht sauber. |
| `stage_disagreement` | Stufe 1 und Stufe 2 widersprechen sich stark. |
| `reference_too_strict` | Die Referenz oder der Fallvertrag könnte zu eng sein. |

Interpretation der Stufen:

| Muster | Bedeutung |
|---|---|
| Stufe 1 hoch, Stufe 2 hoch | Antwort und Evidenz passen gut zusammen. |
| Stufe 1 niedrig, Stufe 2 hoch | Die Antwort ist belegt, aber die Referenz könnte zu streng sein. |
| Stufe 1 hoch, Stufe 2 niedrig | Die Antwort passt zur Referenz, ist aber durch die gefundenen Chunks schwach belegt. |
| Beide niedrig | Retrieval, Antwort oder Fallbasis müssen geprüft werden. |

---

## Ideen zur Weiterentwicklung

- Fallbasis weiter verbessern, besonders Fragen mit mehreren offiziellen Antwortpfaden.
- Mehr `answer_variants` und `accepted_source_urls` ergänzen.
- Automatische Vergleichsberichte für Ablation-Runs erzeugen.
- Separate Testgruppe für Folgefragen entwickeln, falls `turn_router.py` später evaluiert werden soll.
- Zusätzliche Metriken für Retrieval-Rang, Goldquellen-Fundrate und Chunk-Abdeckung ausgeben.
- Optionale Wiederholung desselben Runs zur Messung von Judge-Stabilität.
