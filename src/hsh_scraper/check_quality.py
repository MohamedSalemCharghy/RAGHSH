"""
Qualitätsprüfung — Analyse der gescrapten Markdown-Dateien.

Liest das Rohkorpus aus data/ingested/ und bewertet jede Datei mit denselben
Regeln, die später auch die Kuratierung in clean_corpus.py nutzt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .corpus_quality import (
        CleanedBody,
        assess_markdown_file,
        clean_markdown_body,
        duplicate_delete_set,
        final_quality_decision,
        split_front_matter,
    )
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from corpus_quality import (
        CleanedBody,
        assess_markdown_file,
        clean_markdown_body,
        duplicate_delete_set,
        final_quality_decision,
        split_front_matter,
    )
    from paths import DATA_DIR

DEFAULT_INGESTED_DIR = DATA_DIR / "ingested"
DEFAULT_KEEP_ENGLISH = False


def classify_file(filepath: Path, *, duplicate: bool) -> tuple[dict, str, CleanedBody, object]:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    meta, body = split_front_matter(text)
    if not meta:
        assessment = assess_markdown_file({}, "", duplicate=duplicate)
        return meta, body, CleanedBody("", 0, 0, 0), assessment

    cleaned = clean_markdown_body(body, source_url=meta.get("source_url", ""))
    if "faculty" not in meta:
        # Wird nur für die spätere Organisation mitgeführt.
        meta["faculty"] = ""
    assessment = assess_markdown_file(meta, cleaned.text, duplicate=duplicate)
    return meta, cleaned.text, cleaned, assessment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueft das Rohkorpus auf Qualitaetsprobleme.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INGESTED_DIR,
        help="Verzeichnis mit Markdown-Dateien",
    )
    parser.add_argument(
        "--keep-english",
        action="store_true",
        default=DEFAULT_KEEP_ENGLISH,
        help="Englischsprachige Dateien wie clean_corpus nicht automatisch verwerfen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingested_dir = args.input_dir
    keep_english = args.keep_english

    if not ingested_dir.exists():
        print(f"Directory not found: {ingested_dir}")
        sys.exit(1)

    md_files = sorted(ingested_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {ingested_dir}")
        sys.exit(0)

    duplicate_delete = duplicate_delete_set(md_files)
    results = []
    for filepath in md_files:
        meta, body, cleaned, assessment = classify_file(
            filepath,
            duplicate=filepath in duplicate_delete,
        )
        results.append((filepath, meta, body, cleaned, assessment))

    col_file = max(max(len(path.name) for path, *_ in results), len("Filename"))
    header = (
        f"{'Filename':<{col_file}}  {'Words':>7}  {'Type':<6}  "
        f"{'Lang':<6}  {'Score':>5}  {'Quality'}"
    )
    separator = "-" * len(header)

    print()
    print(header)
    print(separator)

    delete_candidates = []
    total_words = 0
    total_removed_links = 0

    for filepath, meta, body, cleaned, assessment in results:
        total_words += assessment.words
        total_removed_links += cleaned.blocked_links_removed
        content_type = meta.get("content_type", "?") if meta else "?"
        final_keep, final_reasons = final_quality_decision(
            assessment,
            keep_english=keep_english,
            has_metadata=bool(meta),
        )
        quality = "OK" if final_keep else "PRUEFEN"
        print(
            f"{filepath.name:<{col_file}}  {assessment.words:>7,}  {content_type:<6}  "
            f"{assessment.language:<6}  {assessment.score:>5}  {quality}"
        )
        if not final_keep:
            delete_candidates.append((filepath, final_reasons))

    print(separator)
    print(
        f"{'TOTAL':<{col_file}}  {total_words:>7,}          "
        f"        blocked-links-entfernt-potentiell: {total_removed_links}"
    )
    print()

    if not delete_candidates:
        print("Keine kritischen Qualitätsfunde — das Korpus sieht insgesamt stabil aus.")
        return

    print("=" * 78)
    print(f"  QUALITÄTSFUNDE  ({len(delete_candidates)} von {len(md_files)} Dateien)")
    print("=" * 78)
    for filepath, reasons in delete_candidates:
        print(f"\n  {filepath.name}")
        for reason in reasons:
            print(f"    • {reason}")
    print()


if __name__ == "__main__":
    main()
