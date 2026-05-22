"""Extrahiert Glossar-, Legenden- und Abkürzungs-Hinweise aus dem kuratierten Korpus."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import re

try:
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from paths import DATA_DIR


CURATED_DIR = DATA_DIR / "curated"

FRONTMATTER_PATTERN = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
ABBREVIATION_TERM_PATTERN = re.compile(r"^[A-ZÄÖÜ][A-Z0-9.\[\]-]{0,12}$")
BOLDED_PAIR_PATTERN = re.compile(r"\*\*([A-ZÄÖÜ][A-Z0-9.\[\]-]{0,12})\*\*\s*\(([^)\n]{2,140})\)")
PLAIN_PAIR_PATTERN = re.compile(r"(?<!\*)\b([A-ZÄÖÜ][A-Z0-9.\[\]-]{1,12})\b\s*\(([^)\n]{2,140})\)")
COMPACT_PAIR_PATTERN = re.compile(
    r"(?<!\S)([A-ZÄÖÜ][A-Z0-9.\[\]-]{0,12})\s+([A-ZÄÖÜa-zäöüß][^()\n]{3,80}?)(?=(?:\s+[A-ZÄÖÜ][A-Z0-9.\[\]-]{0,12}\s+)|$)"
)
TERM_LIKE_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{2,15}\b")
PROTECTED_QUERY_TERMS = frozenset({"BPO", "ECTS", "K60", "K90", "PO", "SWS"})

DEFINITION_MARKERS = (
    "abkuerzung",
    "abkürzung",
    "bedeutet",
    "credit entspricht",
    "credits dienen",
    "ects-leistungspunkt",
    "ein credit entspricht",
    "ein ects-leistungspunkt entspricht",
    "european credit transfer system",
    "legende",
    "pruefungsleistung",
    "pruefungsleistungen",
    "pruefungsform",
    "pruefungsformen",
    "semesterwochenstunden",
    "steht fuer",
    "steht für",
    "workload",
)


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    meaning: str
    snippet: str
    source_url: str
    title: str
    section_heading: str
    faculty: str
    crawl_date: str
    document_kind: str
    search_text: str


@dataclass(frozen=True)
class GlossaryIndex:
    entries_by_term: dict[str, tuple[GlossaryEntry, ...]]
    snippets: tuple[GlossaryEntry, ...]
    term_mentions_by_term: dict[str, tuple[GlossaryEntry, ...]]


def _normalize(text: str) -> str:
    return (
        text.casefold()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def _clean_inline_markup(text: str) -> str:
    cleaned = (
        text.replace("**", " ")
        .replace("`", " ")
        .replace("<br>", " ")
        .replace("<br/>", " ")
        .replace("<br />", " ")
    )
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" :-")


def _looks_like_term(token: str) -> bool:
    normalized = _normalize(token)
    if len(token) < 3:
        return False
    if token.upper() in PROTECTED_QUERY_TERMS:
        return True
    if any(ch.isdigit() for ch in token):
        return True

    upper_count = sum(ch.isupper() for ch in token)
    lower_count = sum(ch.islower() for ch in token)
    return upper_count >= 2 or (upper_count >= 1 and lower_count >= 1 and not token[0].isupper())


def _should_attempt_similar_term_lookup(token: str) -> bool:
    if not _looks_like_term(token):
        return False
    if token.upper() in PROTECTED_QUERY_TERMS:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    return True


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, text[match.end():]


def _build_snippet(lines: list[str], index: int) -> str:
    snippet_lines: list[str] = []
    for offset in range(0, 4):
        pos = index + offset
        if pos >= len(lines):
            break
        candidate = _clean_inline_markup(lines[pos])
        if not candidate:
            if snippet_lines:
                break
            continue
        snippet_lines.append(candidate)
    snippet = " ".join(snippet_lines)
    return snippet[:700].strip()


def _add_entry(
    entries_by_term: dict[str, list[GlossaryEntry]],
    snippets: list[GlossaryEntry],
    seen_entries: set[tuple[str, str, str]],
    seen_snippets: set[tuple[str, str, str]],
    *,
    metadata: dict[str, str],
    heading: str,
    term: str,
    meaning: str,
    snippet: str,
) -> None:
    term = term.strip()
    meaning = _clean_inline_markup(meaning)
    snippet = _clean_inline_markup(snippet)
    if not term or not meaning or not snippet:
        return
    if not ABBREVIATION_TERM_PATTERN.fullmatch(term):
        return

    normalized_term = _normalize(term)
    source_url = metadata.get("source_url", "")
    title = metadata.get("title", "")
    key = (normalized_term, meaning, source_url)
    if key in seen_entries:
        return
    seen_entries.add(key)

    entry = GlossaryEntry(
        term=term,
        meaning=meaning,
        snippet=snippet,
        source_url=source_url,
        title=title,
        section_heading=heading,
        faculty=metadata.get("faculty", ""),
        crawl_date=metadata.get("crawl_date", ""),
        document_kind=metadata.get("document_kind", ""),
        search_text=_normalize(" ".join((term, meaning, heading, snippet))),
    )
    entries_by_term.setdefault(normalized_term, []).append(entry)

    snippet_key = (source_url, heading, snippet)
    if snippet_key not in seen_snippets:
        seen_snippets.add(snippet_key)
        snippets.append(entry)


def _add_snippet_only(
    snippets: list[GlossaryEntry],
    seen_snippets: set[tuple[str, str, str]],
    *,
    metadata: dict[str, str],
    heading: str,
    snippet: str,
) -> None:
    snippet = _clean_inline_markup(snippet)
    if not snippet:
        return

    source_url = metadata.get("source_url", "")
    key = (source_url, heading, snippet)
    if key in seen_snippets:
        return
    seen_snippets.add(key)

    snippets.append(
        GlossaryEntry(
            term="",
            meaning="",
            snippet=snippet,
            source_url=source_url,
            title=metadata.get("title", ""),
            section_heading=heading,
            faculty=metadata.get("faculty", ""),
            crawl_date=metadata.get("crawl_date", ""),
            document_kind=metadata.get("document_kind", ""),
            search_text=_normalize(" ".join((heading, snippet))),
        )
    )


def _add_term_mentions(
    term_mentions_by_term: dict[str, list[GlossaryEntry]],
    seen_terms: set[tuple[str, str, str]],
    *,
    metadata: dict[str, str],
    heading: str,
    snippet: str,
) -> None:
    for token in TERM_LIKE_PATTERN.findall(snippet):
        if not _looks_like_term(token):
            continue

        normalized_term = _normalize(token)
        source_url = metadata.get("source_url", "")
        key = (normalized_term, source_url, heading)
        if key in seen_terms:
            continue
        seen_terms.add(key)

        entry = GlossaryEntry(
            term=token,
            meaning="",
            snippet=snippet,
            source_url=source_url,
            title=metadata.get("title", ""),
            section_heading=heading,
            faculty=metadata.get("faculty", ""),
            crawl_date=metadata.get("crawl_date", ""),
            document_kind=metadata.get("document_kind", ""),
            search_text=_normalize(" ".join((token, heading, snippet))),
        )
        term_mentions_by_term.setdefault(normalized_term, []).append(entry)


def _extract_pairs(line: str) -> list[tuple[str, str]]:
    cleaned_line = _clean_inline_markup(line)
    pairs: list[tuple[str, str]] = []

    for pattern in (BOLDED_PAIR_PATTERN, PLAIN_PAIR_PATTERN):
        for match in pattern.finditer(line):
            pairs.append((match.group(1), match.group(2)))

    normalized = _normalize(cleaned_line)
    if "abkuerzung" in normalized or "pruefungsleistung" in normalized:
        for match in COMPACT_PAIR_PATTERN.finditer(cleaned_line):
            term = match.group(1)
            meaning = match.group(2).strip(" ,;")
            if len(meaning.split()) > 10:
                continue
            pairs.append((term, meaning))

    if "ects" in normalized and "workload" in normalized:
        pairs.append(("ECTS", cleaned_line))
    elif "credit" in normalized and "workload" in normalized:
        pairs.append(("ECTS", cleaned_line))

    if "semesterwochenstunden" in normalized:
        pairs.append(("SWS", cleaned_line))

    if "besonderer teil der pruefungsordnung" in normalized:
        pairs.append(("BPO", cleaned_line))

    if "pruefungsform" in normalized or "pruefungsleistungen" in normalized:
        pairs.append(("Prüfungsform", cleaned_line))

    unique: dict[tuple[str, str], None] = {}
    for pair in pairs:
        unique[(pair[0].strip(), _clean_inline_markup(pair[1]))] = None
    return list(unique)


def _iter_glossary_entries(
    path: Path,
) -> tuple[dict[str, list[GlossaryEntry]], list[GlossaryEntry], dict[str, list[GlossaryEntry]]]:
    text = path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(text)
    lines = body.splitlines()

    entries_by_term: dict[str, list[GlossaryEntry]] = {}
    snippets: list[GlossaryEntry] = []
    term_mentions_by_term: dict[str, list[GlossaryEntry]] = {}
    seen_entries: set[tuple[str, str, str]] = set()
    seen_snippets: set[tuple[str, str, str]] = set()
    seen_terms: set[tuple[str, str, str]] = set()

    heading = metadata.get("title", "")
    abbreviation_block = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            heading = _clean_inline_markup(heading_match.group(1))
            normalized_heading = _normalize(heading)
            abbreviation_block = (
                "abkuerzung" in normalized_heading
                or "legende" in normalized_heading
                or "pruefungsleistungen" in normalized_heading
                or "pruefungsformen" in normalized_heading
            )
            continue

        snippet = _build_snippet(lines, index)
        normalized_line = _normalize(_clean_inline_markup(line))
        pairs = _extract_pairs(line)

        if snippet:
            _add_term_mentions(
                term_mentions_by_term,
                seen_terms,
                metadata=metadata,
                heading=heading,
                snippet=snippet,
            )

        for term, meaning in pairs:
            _add_entry(
                entries_by_term,
                snippets,
                seen_entries,
                seen_snippets,
                metadata=metadata,
                heading=heading,
                term=term,
                meaning=meaning,
                snippet=snippet,
            )

        if abbreviation_block or any(marker in normalized_line for marker in DEFINITION_MARKERS):
            _add_snippet_only(
                snippets,
                seen_snippets,
                metadata=metadata,
                heading=heading,
                snippet=snippet,
            )

    return entries_by_term, snippets, term_mentions_by_term


@lru_cache(maxsize=1)
def build_glossary_index() -> GlossaryIndex:
    entries_by_term: dict[str, list[GlossaryEntry]] = {}
    snippets: list[GlossaryEntry] = []
    term_mentions_by_term: dict[str, list[GlossaryEntry]] = {}

    if not CURATED_DIR.exists():
        return GlossaryIndex(entries_by_term={}, snippets=(), term_mentions_by_term={})

    for path in sorted(CURATED_DIR.glob("*.md")):
        file_entries, file_snippets, file_term_mentions = _iter_glossary_entries(path)
        for term, values in file_entries.items():
            entries_by_term.setdefault(term, []).extend(values)
        snippets.extend(file_snippets)
        for term, values in file_term_mentions.items():
            term_mentions_by_term.setdefault(term, []).extend(values)

    return GlossaryIndex(
        entries_by_term={term: tuple(values) for term, values in entries_by_term.items()},
        snippets=tuple(snippets),
        term_mentions_by_term={term: tuple(values) for term, values in term_mentions_by_term.items()},
    )


def _score_entry_text(text: str, normalized_terms: list[str], normalized_phrases: list[str]) -> float:
    if not text:
        return 0.0

    score = 0.0
    for term in normalized_terms:
        if term and term in text:
            score += 2.8

    for phrase in normalized_phrases:
        if not phrase:
            continue
        if phrase in text:
            score += 2.2
            continue

        tokens = [token for token in re.split(r"[^a-z0-9]+", phrase) if len(token) >= 2]
        if not tokens:
            continue
        overlap = sum(token in text for token in tokens)
        if overlap >= 2:
            score += 0.45 * overlap

    score += 0.3 * sum(marker in text for marker in DEFINITION_MARKERS)
    return score


def _result_id(entry: GlossaryEntry) -> str:
    digest = hashlib.sha1(
        "|".join((entry.term, entry.source_url, entry.section_heading, entry.snippet)).encode("utf-8")
    ).hexdigest()
    return f"glossary::{digest}"


def lookup_glossary_results(
    *,
    terms: list[str],
    candidate_phrases: list[str],
    limit: int = 4,
) -> list[dict]:
    """Liefert definierende Snippets aus dem kuratierten Korpus im Ergebnisformat der Suche."""
    normalized_terms = [_normalize(term) for term in terms if term]
    normalized_phrases = [_normalize(phrase) for phrase in candidate_phrases if phrase]
    if not normalized_terms and not normalized_phrases:
        return []

    index = build_glossary_index()
    results: list[dict] = []
    seen_ids: set[str] = set()

    for term in normalized_terms:
        for entry in index.entries_by_term.get(term, ()):
            score = 6.0 + _score_entry_text(entry.search_text, [term], normalized_phrases)
            result_id = _result_id(entry)
            if result_id in seen_ids:
                continue
            seen_ids.add(result_id)
            results.append({
                "id": result_id,
                "score": score,
                "payload": {
                "title": entry.title,
                "source_url": entry.source_url,
                "section_heading": entry.section_heading or "Glossar/Legende",
                "faculty": entry.faculty,
                "crawl_date": entry.crawl_date,
                "document_kind": entry.document_kind or "glossary",
                "text": f"{entry.term}: {entry.meaning}\n\n{entry.snippet}",
                "glossary_term": entry.term or term.upper(),
                "glossary_meaning": entry.meaning,
            },
        })

    for entry in index.snippets:
        score = _score_entry_text(entry.search_text, normalized_terms, normalized_phrases)
        if score < 2.6:
            continue

        result_id = _result_id(entry)
        if result_id in seen_ids:
            continue
        seen_ids.add(result_id)
        results.append({
            "id": result_id,
            "score": score,
            "payload": {
                "title": entry.title,
                "source_url": entry.source_url,
                "section_heading": entry.section_heading or "Glossar/Legende",
                "faculty": entry.faculty,
                "crawl_date": entry.crawl_date,
                "document_kind": entry.document_kind or "glossary",
                "text": entry.snippet,
                "glossary_term": entry.term,
                "glossary_meaning": entry.meaning,
            },
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


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


def find_similar_corpus_terms(query_terms: list[str], *, limit: int = 4) -> list[dict[str, str]]:
    """Findet sehr ähnliche Fachbegriffe oder Systemnamen aus dem kuratierten Korpus."""
    index = build_glossary_index()
    suggestions: list[dict[str, str]] = []

    for query_term in query_terms:
        if not _should_attempt_similar_term_lookup(query_term):
            continue
        normalized_query = _normalize(query_term)
        best_match = None
        best_score = None

        for normalized_term, mentions in index.term_mentions_by_term.items():
            if normalized_term == normalized_query:
                continue
            distance = _damerau_levenshtein_distance(normalized_query, normalized_term, max_distance=1)
            if distance > 1:
                continue

            mention = mentions[0]
            score = (-distance, len(mentions), len(normalized_term))
            if best_score is None or score > best_score:
                best_score = score
                best_match = {
                    "asked": query_term,
                    "matched": mention.term,
                    "title": mention.title,
                    "section_heading": mention.section_heading,
                    "source_url": mention.source_url,
                    "faculty": mention.faculty,
                    "crawl_date": mention.crawl_date,
                    "snippet": mention.snippet,
                }

        if best_match:
            suggestions.append(best_match)
        if len(suggestions) >= limit:
            break

    return suggestions
