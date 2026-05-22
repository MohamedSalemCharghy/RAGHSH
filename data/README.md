Generated data lives in this directory.

- `ingested/` contains the raw Markdown corpus produced by the crawler.
- `curated/` contains the cleaned/organized Markdown corpus produced by `clean_corpus.py`.
- `curated_report.json` is the JSON report describing keep/reject decisions from the cleaning step.
- `url_decisions.db` stores URL filter decisions made during crawling.

These outputs are normally generated locally or on a server/HPC workflow and are usually not committed to git.
