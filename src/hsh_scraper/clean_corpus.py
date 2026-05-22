"""
Kuratierung des Markdown-Korpus.

Liest Rohdateien aus data/ingested/, entfernt problematische Links und
Boilerplate, ergänzt Qualitäts-/Organisations-Metadaten und schreibt das
bereinigte Korpus nach data/curated/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .corpus_quality import (
        assess_markdown_file,
        build_front_matter,
        clean_markdown_body,
        duplicate_delete_set,
        enrich_metadata,
        final_quality_decision,
        split_front_matter,
    )
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from corpus_quality import (
        assess_markdown_file,
        build_front_matter,
        clean_markdown_body,
        duplicate_delete_set,
        enrich_metadata,
        final_quality_decision,
        split_front_matter,
    )
    from paths import DATA_DIR

DEFAULT_RAW_DIR = DATA_DIR / "ingested"
DEFAULT_CURATED_DIR = DATA_DIR / "curated"
DEFAULT_REPORT_FILE = DATA_DIR / "curated_report.json"
DEFAULT_KEEP_ENGLISH = False


def write_curated_file(path: Path, meta: dict[str, str], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_front_matter(meta) + body.strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bereinigt und organisiert das Markdown-Korpus.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Verzeichnis mit den Roh-Markdown-Dateien",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CURATED_DIR,
        help="Zielverzeichnis fuer das bereinigte Korpus",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help="Pfad fuer den JSON-Report",
    )
    parser.add_argument(
        "--keep-english",
        action="store_true",
        default=DEFAULT_KEEP_ENGLISH,
        help="Englischsprachige Dateien nicht verwerfen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.input_dir
    curated_dir = args.output_dir
    report_file = args.report_file
    keep_english = args.keep_english

    if not raw_dir.exists():
        print(f"Directory not found: {raw_dir}")
        sys.exit(1)

    md_files = sorted(raw_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {raw_dir}")
        sys.exit(0)

    curated_dir.mkdir(parents=True, exist_ok=True)
    for old in curated_dir.glob("*.md"):
        old.unlink()

    duplicate_delete = duplicate_delete_set(md_files)
    kept = 0
    rejected = 0
    report: list[dict] = []

    for filepath in md_files:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        meta, body = split_front_matter(text)

        if not meta:
            rejected += 1
            report.append(
                {
                    "file": filepath.name,
                    "status": "reject",
                    "reasons": ["kein_yaml_header"],
                    "output_file": "",
                }
            )
            continue

        cleaned = clean_markdown_body(body, source_url=meta.get("source_url", ""))
        if "faculty" not in meta:
            meta["faculty"] = ""

        assessment = assess_markdown_file(
            meta,
            cleaned.text,
            duplicate=filepath in duplicate_delete,
        )

        keep, final_reasons = final_quality_decision(
            assessment,
            keep_english=keep_english,
        )

        enriched_meta = enrich_metadata(meta, assessment, cleaned)
        output_file = curated_dir / filepath.name

        if keep:
            write_curated_file(output_file, enriched_meta, cleaned.text)
            kept += 1
            status = "keep"
        else:
            rejected += 1
            status = "reject"

        report.append(
            {
                "file": filepath.name,
                "status": status,
                "output_file": output_file.name if keep else "",
                "score": assessment.score,
                "language": assessment.language,
                "document_kind": assessment.document_kind,
                "source_family": assessment.source_family,
                "document_group": assessment.document_group,
                "topic_tags": list(assessment.topic_tags),
                "blocked_links_removed": cleaned.blocked_links_removed,
                "normalized_links_rewritten": cleaned.normalized_links_rewritten,
                "removed_noise_lines": cleaned.removed_noise_lines,
                "reasons": list(final_reasons),
            }
        )

    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rohdateien geprüft : {len(md_files)}")
    print(f"Behalten           : {kept}")
    print(f"Verworfen          : {rejected}")
    print(f"Curated corpus     : {curated_dir}")
    print(f"Report             : {report_file}")


if __name__ == "__main__":
    main()
