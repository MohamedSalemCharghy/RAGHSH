"""
Resume-Crawler — Setzt einen unterbrochenen Crawl auf Basis vorhandener Markdown-Dateien fort.

Kurzbeschreibung
----------------
Analysiert die bereits gescrapten Markdown-Dateien in data/ingested/, extrahiert alle
darin enthaltenen Links und ermittelt, welche URLs noch nicht gescrapet wurden oder
veraltet sind. Startet dann den Crawl-Prozess nur für diese fehlenden URLs.

Ablauf
------
1. Alle Markdown-Dateien in data/ingested/ einlesen.
2. Aus dem YAML-Header: source_url + crawl_date → Menge der bekannten URLs aufbauen.
3. Aus dem Markdown-Body: alle internen Links extrahieren (Ziel-URLs entdecken).
4. Drei Listen erstellen und ausgeben:
     a) Bereits gescrapt & aktuell  (jünger als MAX_AGE_DAYS)
     b) Bereits gescrapt & veraltet (älter als MAX_AGE_DAYS → wird neu gecrawlt)
     c) Noch nicht gescrapt         (in Links gefunden, aber keine Datei vorhanden)
5. Nur Listen b) und c) werden gecrawlt (kein Doppel-Crawl aktueller Dateien).

Aufruf:
    python -m hsh_scraper.resume_crawler            # Analyse + Crawl
    python -m hsh_scraper.resume_crawler --dry-run  # Nur Analyse, kein Crawl

Abhängigkeiten: crawl4ai, httpx, pymupdf4llm, python-slugify, openpyxl
"""

import argparse
import asyncio
import logging
import os
import re
import sys
import tempfile
from collections import Counter, deque
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
import openpyxl
import pymupdf4llm
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from slugify import slugify

try:
    from .crawl_helpers import (
        assess_markdown_quality,
        build_crawler_config,
        discover_sitemap_links,
        format_host_summary,
    )
    from .paths import DATA_DIR
    from .url_filter import (
        ALLOWED_DOMAIN,
        BLOCKED_DOMAINS,
        DecisionStats,
        UrlDecisionStore,
        evaluate_rag_url,
        format_decision_summary,
        normalize_url,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from crawl_helpers import (
        assess_markdown_quality,
        build_crawler_config,
        discover_sitemap_links,
        format_host_summary,
    )
    from paths import DATA_DIR
    from url_filter import (
        ALLOWED_DOMAIN,
        BLOCKED_DOMAINS,
        DecisionStats,
        UrlDecisionStore,
        evaluate_rag_url,
        format_decision_summary,
        normalize_url,
    )

# ---------------------------------------------------------------------------
# Konfiguration — identisch mit main.py
# ---------------------------------------------------------------------------

SEED_URLS      = ["https://www.hs-hannover.de/"]
MAX_PAGES      = 40000
MAX_AGE_DAYS   = 7

OUTPUT_DIR         = DATA_DIR / "ingested"
RATE_LIMIT_SECONDS = 0.5
CRAWL_LOW_VALUE_URLS = True
USE_SITEMAP_SEEDS = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL-Hilfsfunktionen (aus main.py übernommen)
# ---------------------------------------------------------------------------


def make_filename(url: str, today: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    slug = slugify(path) if path else "index"
    return f"{today}_{slug}.md"


def build_yaml_header(source_url, title, crawl_date, content_type="html") -> str:
    return (
        "---\n"
        f'source_url: "{source_url}"\n'
        f'title: "{title}"\n'
        f'crawl_date: "{crawl_date}"\n'
        f'content_type: "{content_type}"\n'
        "---\n"
    )


def save_markdown(filepath: Path, header: str, body: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(header + "\n" + body, encoding="utf-8")
    logger.info("Gespeichert: %s", filepath)


# ---------------------------------------------------------------------------
# Analyse der vorhandenen Markdown-Dateien
# ---------------------------------------------------------------------------

# Regex für Markdown-Links: [Text](https://...) und rohe https://-URLs
_MD_LINK_RE  = re.compile(r'\[.*?\]\((https?://[^)\s]+)\)')
_RAW_LINK_RE = re.compile(r'(?<![(\["])https?://[^\s)"\'<>]+')


def parse_frontmatter(text: str) -> dict:
    """Extrahiert Schlüssel-Wert-Paare aus einem YAML-Frontmatter-Block."""
    meta = {}
    if not text.startswith("---"):
        return meta
    parts = text.split("---", 2)
    if len(parts) < 3:
        return meta
    for line in parts[1].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta


def extract_links_from_body(body: str) -> set[str]:
    """Findet alle http(s)-URLs im Markdown-Body."""
    found: set[str] = set()
    for m in _MD_LINK_RE.finditer(body):
        found.add(m.group(1))
    for m in _RAW_LINK_RE.finditer(body):
        url = m.group(0).rstrip(".,;:!?)")
        # Eckige Klammern in der URL deuten auf IPv6-ähnliche Syntax hin → überspringen
        if "[" not in url and "]" not in url:
            found.add(url)
    return found


def analyse_ingested_dir(
    decision_store: UrlDecisionStore,
) -> tuple[dict, dict, set[str], DecisionStats]:
    """Analysiert data/ingested/ und gibt drei Strukturen zurück:

    fresh    : {norm_url: filepath}   — aktuell gescrapte URLs
    stale    : {norm_url: filepath}   — veraltete gescrapte URLs
    discovered: set[norm_url]         — in Links gefundene, noch nicht gescrapte URLs
    """
    cutoff = date.today() - timedelta(days=MAX_AGE_DAYS)

    fresh:      dict[str, Path] = {}
    stale:      dict[str, Path] = {}
    all_links:  set[str]        = set()
    filter_stats = DecisionStats()

    md_files = sorted(OUTPUT_DIR.glob("*.md"))
    logger.info("Analysiere %d Markdown-Dateien …", len(md_files))

    for filepath in md_files:
        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception:
            continue

        meta = parse_frontmatter(text)
        source_url = meta.get("source_url", "")
        crawl_date_str = meta.get("crawl_date", "")

        if source_url:
            source_decision = evaluate_rag_url(source_url)
            filter_stats.add(source_decision)
            decision_store.record(source_decision, source="resume")

            if source_decision.is_allowed:
                norm = source_decision.normalized_url

                # Frische-Prüfung anhand crawl_date im Header
                try:
                    file_date = date.fromisoformat(crawl_date_str)
                    is_fresh  = file_date >= cutoff
                except ValueError:
                    is_fresh = False

                if is_fresh:
                    fresh[norm] = filepath
                else:
                    stale[norm] = filepath

        # Links aus dem Body extrahieren
        if "---" in text:
            parts = text.split("---", 2)
            body = parts[2] if len(parts) == 3 else ""
        else:
            body = text

        for link in extract_links_from_body(body):
            link_decision = evaluate_rag_url(link)
            filter_stats.add(link_decision)
            decision_store.record(link_decision, source="resume")
            if link_decision.is_allowed:
                all_links.add(link_decision.normalized_url)

    # URLs die in Links auftauchen, aber nicht als fresh bekannt sind
    all_scraped = set(fresh) | set(stale)
    discovered = all_links - all_scraped

    return fresh, stale, discovered, filter_stats


def print_report(
    fresh: dict,
    stale: dict,
    discovered: set,
    filter_stats: DecisionStats,
) -> None:
    """Gibt eine übersichtliche Analyse auf der Konsole aus."""
    total = len(fresh) + len(stale) + len(discovered)
    print()
    print("=" * 70)
    print("  Crawl-Analyse")
    print("=" * 70)
    print(f"  Bereits gescrapt & aktuell   : {len(fresh):6d} URLs")
    print(f"  Bereits gescrapt & veraltet  : {len(stale):6d} URLs  → werden neu gecrawlt")
    print(f"  In Links entdeckt, fehlt noch: {len(discovered):6d} URLs  → werden gecrawlt")
    print(f"  ──────────────────────────────────────")
    print(f"  Gesamt bekannte URLs          : {total:6d}")
    print("=" * 70)

    if stale:
        print(f"\nVeraltete Dateien (erste 10 von {len(stale)}):")
        for url in list(stale)[:10]:
            print(f"  {url}")
        if len(stale) > 10:
            print(f"  … und {len(stale) - 10} weitere")

    if discovered:
        print(f"\nFehlende URLs (erste 10 von {len(discovered)}):")
        for url in list(discovered)[:10]:
            print(f"  {url}")
        if len(discovered) > 10:
            print(f"  … und {len(discovered) - 10} weitere")

    summary_lines = format_decision_summary(filter_stats, max_reasons=8, max_samples=3)
    if summary_lines:
        print("\nRAG-Filter-Zusammenfassung:")
        for line in summary_lines:
            print(f"  {line}")
    print()


# ---------------------------------------------------------------------------
# Crawl-Logik (aus main.py übernommen)
# ---------------------------------------------------------------------------


async def is_pdf_url(url: str) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.head(url, timeout=10)
            return "application/pdf" in resp.headers.get("content-type", "").lower()
    except httpx.HTTPError:
        return False


async def process_pdf(url: str, today: str) -> None:
    logger.info("PDF herunterladen: %s", url)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    title = Path(urlparse(url).path).stem or "PDF Document"
    header = build_yaml_header(url, title, today, content_type="pdf")
    save_markdown(OUTPUT_DIR / make_filename(url, today), header, md_text)


async def process_html(
    crawler: AsyncWebCrawler,
    url: str,
    today: str,
    config: CrawlerRunConfig,
    *,
    url_decision=None,
    save: bool = True,
):
    logger.info("Crawling: %s", url)
    result = await crawler.arun(url=url, config=config)

    if not result.success:
        raise RuntimeError(result.error_message or "Crawl fehlgeschlagen")

    md_content = ""
    if hasattr(result, "markdown_v2") and result.markdown_v2:
        md_content = getattr(result.markdown_v2, "raw_markdown", "") or ""
    if not md_content:
        md_content = result.markdown or ""

    title = ""
    if hasattr(result, "metadata") and result.metadata:
        title = result.metadata.get("title", "") or ""
    if not title:
        for line in md_content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
    if not title:
        title = url

    saved = False
    quality_reasons: tuple[str, ...] = ()
    if save:
        quality = assess_markdown_quality(
            url,
            title,
            md_content,
            is_high_value=bool(getattr(url_decision, "is_high_value", False)),
        )
        if quality.keep:
            header = build_yaml_header(url, title, today, content_type="html")
            save_markdown(OUTPUT_DIR / make_filename(url, today), header, md_content)
            saved = True
        else:
            quality_reasons = quality.reasons
            logger.info(
                "Verwerfe Seite mit geringem Inhaltswert: %s (%s)",
                url,
                ", ".join(quality.reasons),
            )

    return result, saved, quality_reasons


def write_error_report(errors: list[dict], today: str) -> None:
    if not errors:
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Fehler"
    ws.append(["URL", "Gefunden auf (Referrer)", "Fehlermeldung"])
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    for entry in errors:
        ws.append([entry["url"], entry["referrer"], entry["error"]])
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 80)
    filepath = OUTPUT_DIR / f"{today}_fehler.xlsx"
    wb.save(filepath)
    logger.info("Fehlerbericht gespeichert: %s (%d Einträge)", filepath, len(errors))


# ---------------------------------------------------------------------------
# Haupt-Crawl-Schleife
# ---------------------------------------------------------------------------


async def run_crawl(to_crawl: set[str], decision_store: UrlDecisionStore) -> None:
    """Crawlt alle URLs in to_crawl (stale + discovered)."""
    today   = date.today().isoformat()
    errors: list[dict] = []
    filter_stats = DecisionStats()
    config = build_crawler_config()

    # Bekannte, aktuell gescrapte URLs ermitteln (Duplikat-Schutz während des Laufs)
    cutoff = date.today() - timedelta(days=MAX_AGE_DAYS)
    visited: set[str] = set()
    for filepath in OUTPUT_DIR.glob("*.md"):
        try:
            text = filepath.read_text(encoding="utf-8")
            meta = parse_frontmatter(text)
            src  = meta.get("source_url", "")
            crawl_date_str = meta.get("crawl_date", "")
            if src:
                norm = normalize_url(src)
                try:
                    if date.fromisoformat(crawl_date_str) >= cutoff:
                        visited.add(norm)   # frisch → nicht nochmal crawlen
                except ValueError:
                    pass
        except Exception:
            continue

    # Queue mit den zu crawlenden URLs befüllen
    high_queue: deque[str] = deque()
    low_queue: deque[str] = deque()
    referrers: dict[str, str] = {}
    decision_by_url: dict[str, object] = {}
    queued_counts: Counter[str] = Counter()
    crawled_counts: Counter[str] = Counter()
    quality_reasons: Counter[str] = Counter()
    crawled_hosts: Counter[str] = Counter()
    filtered = 0
    sitemap_seed_count = 0
    low_value_skipped = 0

    def enqueue_decision(decision, *, referrer: str, source: str) -> None:
        nonlocal low_value_skipped
        filter_stats.add(decision)
        decision_store.record(decision, source=source)
        if not decision.is_allowed:
            return
        if decision.normalized_url in visited:
            return
        if decision.is_low_value and not CRAWL_LOW_VALUE_URLS:
            low_value_skipped += 1
            return
        visited.add(decision.normalized_url)
        decision_by_url[decision.normalized_url] = decision
        referrers[decision.normalized_url] = referrer
        queued_counts[decision.decision] += 1
        if decision.is_high_value:
            high_queue.append(decision.normalized_url)
        else:
            low_queue.append(decision.normalized_url)

    for url in to_crawl:
        enqueue_decision(evaluate_rag_url(url), referrer="(resume)", source="resume_seed")

    if USE_SITEMAP_SEEDS:
        sitemap_links = await discover_sitemap_links(SEED_URLS)
        for sitemap_link in sitemap_links:
            before = len(visited)
            enqueue_decision(
                evaluate_rag_url(sitemap_link),
                referrer="(sitemap)",
                source="resume_sitemap",
            )
            if len(visited) > before:
                sitemap_seed_count += 1
        logger.info(
            "Sitemap-Seeding: %d Kandidaten gefunden, %d zusätzlich eingereiht.",
            len(sitemap_links),
            sitemap_seed_count,
        )

    total     = len(high_queue) + len(low_queue)
    done      = 0
    success   = 0
    fail      = 0

    logger.info("Starte Resume-Crawl: %d URLs in der Warteschlange", total)

    async with AsyncWebCrawler() as crawler:
        while (high_queue or low_queue) and done < MAX_PAGES:
            url  = high_queue.popleft() if high_queue else low_queue.popleft()
            url_decision = decision_by_url.get(url)
            done += 1
            if url_decision is not None:
                crawled_counts[url_decision.decision] += 1
            crawled_hosts[urlparse(url).netloc.lower()] += 1

            pct = 100.0 * done / max(total, 1)
            logger.info("[%d/%d  %5.1f%%] %s", done, total, pct, url)

            try:
                if await is_pdf_url(url):
                    await process_pdf(url, today)
                    success += 1
                else:
                    result, saved, skip_reasons = await process_html(
                        crawler,
                        url,
                        today,
                        config,
                        url_decision=url_decision,
                        save=True,
                    )
                    if saved:
                        success += 1
                    else:
                        filtered += 1
                        quality_reasons.update(skip_reasons)
                    # Neu entdeckte interne Links in die Queue aufnehmen
                    links_data = getattr(result, "links", {}) or {}
                    for link_dict in links_data.get("internal", []):
                        href = link_dict.get("href", "")
                        if not href or not href.startswith("http"):
                            continue
                        decision = evaluate_rag_url(href)
                        before = len(visited)
                        enqueue_decision(decision, referrer=url, source="resume")
                        if len(visited) > before:
                            total += 1

                visited.add(normalize_url(url))

            except Exception as exc:
                logger.error("Fehler bei %s: %s", url, exc)
                fail += 1
                errors.append({
                    "url":      url,
                    "referrer": referrers.get(url, "(unbekannt)"),
                    "error":    str(exc),
                })

            if (high_queue or low_queue) and done < MAX_PAGES:
                await asyncio.sleep(RATE_LIMIT_SECONDS)

    write_error_report(errors, today)
    for line in format_decision_summary(filter_stats):
        logger.info(line)
    if quality_reasons:
        logger.info("Nach dem Crawl verworfene Seiten: %d", sum(quality_reasons.values()))
        for reason, count in quality_reasons.most_common(8):
            logger.info("  %s: %d", reason, count)
    logger.info(
        "Queue-Mix: high=%d, low=%d, low_skipped_by_policy=%d, sitemap_seeded=%d",
        queued_counts.get("allow_high_value", 0),
        queued_counts.get("allow_low_value", 0),
        low_value_skipped,
        sitemap_seed_count,
    )
    logger.info(
        "Crawl-Mix: high=%d, low=%d, filtered_after_crawl=%d",
        crawled_counts.get("allow_high_value", 0),
        crawled_counts.get("allow_low_value", 0),
        filtered,
    )
    for line in format_host_summary(crawled_hosts):
        logger.info(line)
    logger.info(
        "Fertig. %d erfolgreich gespeichert, %d fehlgeschlagen, %d nach dem Crawl verworfen, %d URLs verarbeitet.",
        success,
        fail,
        filtered,
        done,
    )


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-Crawler: analysiert vorhandene MD-Dateien und crawlt fehlende URLs."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur Analyse ausgeben, keinen Crawl starten."
    )
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        logger.error("Ausgabeverzeichnis nicht gefunden: %s", OUTPUT_DIR)
        sys.exit(1)

    with UrlDecisionStore() as decision_store:
        # ── Analyse ───────────────────────────────────────────────────────
        fresh, stale, discovered, filter_stats = analyse_ingested_dir(decision_store)
        print_report(fresh, stale, discovered, filter_stats)

        to_crawl = set(stale) | discovered

        if not to_crawl:
            logger.info("Alle bekannten URLs sind aktuell. Kein Crawl notwendig.")
            return

        if args.dry_run:
            logger.info("--dry-run: Crawl wird nicht gestartet.")
            return

        # ── Crawl ─────────────────────────────────────────────────────────
        await run_crawl(to_crawl, decision_store)


if __name__ == "__main__":
    asyncio.run(main())
