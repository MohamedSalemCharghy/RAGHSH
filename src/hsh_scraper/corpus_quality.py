"""
Gemeinsame Qualitäts- und Kuratierungslogik für Markdown-Korpora.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    from .url_filter import evaluate_rag_url
except ImportError:  # pragma: no cover - script execution fallback
    from url_filter import evaluate_rag_url

MIN_WORDS_HTML = 30
MIN_WORDS_PDF = 50

ERROR_PAGE_PATTERNS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "404_not_found",
        re.compile(
            r"(?im)^\s*404\s*$"
            r"|\b404\s*[-:]?\s*(?:not\s+found|seite\s+nicht\s+gefunden|fehler|error)\b"
            r"|\b(?:seite|page)\s+(?:nicht\s+gefunden|not\s+found)\b"
        ),
        False,
    ),
    (
        "access_denied",
        re.compile(r"\b(?:zugriff\s+verweigert|access\s+denied|403\s+forbidden)\b", re.I),
        False,
    ),
    (
        "load_error",
        re.compile(r"\b(?:fehler\s+beim\s+laden|error\s+loading)\b", re.I),
        False,
    ),
    (
        "not_available",
        re.compile(
            r"\b(?:seite|page|inhalt|content|angebot)\s+(?:ist\s+)?"
            r"(?:nicht\s+verf(?:ü|ue)gbar|not\s+available)\b"
            r"|\b(?:nicht\s+verf(?:ü|ue)gbar|not\s+available)\b",
            re.I,
        ),
        True,
    ),
)

GERMAN_MARKERS = {
    " und ",
    " der ",
    " die ",
    " das ",
    " für ",
    " studium ",
    " bewerbung ",
    " ordnung ",
}
ENGLISH_MARKERS = {
    " and ",
    " contact ",
    " courses offered ",
    " deadline ",
    " english language ",
    " the ",
    " course ",
    " courses ",
    " exchange students ",
    " information for ",
    " international office ",
    " offered in english ",
    " application ",
}
TOPIC_KEYWORDS = {
    "bewerbung": "bewerbung",
    "beurlaub": "beurlaubung",
    "campuscard": "campuscard",
    "faq": "faq",
    "immatrik": "immatrikulation",
    "international": "internationales",
    "kontakt": "kontakt",
    "modulhandbuch": "modulhandbuch",
    "ordnung": "ordnungen",
    "praxis": "praxis",
    "pruef": "pruefungen",
    "rueckmeldung": "rueckmeldung",
    "semester": "semester",
    "studiengang": "studiengaenge",
}
LINE_NOISE_PATTERNS = (
    re.compile(r"^\s*\*\*==> picture .* omitted <==\*\*\s*$"),
    re.compile(r"^\s*scrollen für mehr infos!?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[\s*teilen\s*\]\(.*\)\s*$", re.IGNORECASE),
    re.compile(r"^\s*[_*]\s*[_*]\s*$"),
)
IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$")
MARKDOWN_LINK_RE = re.compile(
    r"(!?)\[([^\]]*)\]\(\s*(<[^>\n]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)"
)
AUTOLINK_RE = re.compile(r"<(https?://[^<>\s]+)>")
RAW_URL_RE = re.compile(r"(?<!\()(?<!\[)(?<!<)(https?://[^\s<>\"]+)")
TABLE_SEPARATOR_RE = re.compile(r"\|[\s\-:]+\|")
LIBRARY_CATALOG_HOSTS = {
    "katalog.bib.hs-hannover.de",
}
TECHNICAL_API_PATH_MARKERS = (
    "/apidoc",
    "/api-doc",
    "/api_docs",
    "/javadoc",
    "/swagger",
    "/openapi",
)
TECHNICAL_API_TITLE_MARKERS = (
    "apidoc",
    "api doc",
    "api-doc",
    "javadoc",
    "swagger",
    "openapi",
)
TRAILING_URL_PUNCTUATION = ".,;:!?"
TRAILING_URL_CLOSERS = {
    ")": "(",
    "]": "[",
    "}": "{",
}


@dataclass(frozen=True)
class CleanedBody:
    text: str
    blocked_links_removed: int
    normalized_links_rewritten: int
    removed_noise_lines: int


@dataclass(frozen=True)
class QualityAssessment:
    keep: bool
    score: int
    reasons: tuple[str, ...]
    words: int
    tables: int
    language: str
    document_kind: str
    source_family: str
    document_group: str
    topic_tags: tuple[str, ...]


def final_quality_decision(
    assessment: QualityAssessment,
    *,
    keep_english: bool = False,
    has_metadata: bool = True,
) -> tuple[bool, tuple[str, ...]]:
    if not has_metadata:
        return False, ("kein_yaml_header",)

    keep = assessment.keep
    reasons = list(assessment.reasons)

    if assessment.language == "en" and not keep_english:
        keep = False
        if "englischsprachiger_inhalt" not in reasons:
            reasons.append("englischsprachiger_inhalt")

    return keep, tuple(reasons)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if not text.startswith("---"):
        return meta, text.strip()

    parts = text.split("---", 2)
    if len(parts) < 3:
        return meta, text.strip()

    block = parts[1].strip()
    body = parts[2].strip()
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')

    return meta, body


def build_front_matter(meta: dict[str, str]) -> str:
    lines = ["---"]
    for key, value in meta.items():
        safe = str(value).replace('"', "'")
        lines.append(f'{key}: "{safe}"')
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def count_words(text: str) -> int:
    return len(text.split())


def count_tables(text: str) -> int:
    return sum(1 for line in text.splitlines() if TABLE_SEPARATOR_RE.search(line))


def detect_language(text: str, *, hints: str = "") -> str:
    sample = f" {text.lower()} {hints.lower()} "
    german_hits = sum(sample.count(marker) for marker in GERMAN_MARKERS)
    english_hits = sum(sample.count(marker) for marker in ENGLISH_MARKERS)

    if german_hits == 0 and english_hits == 0:
        return "unknown"
    if german_hits >= english_hits * 2 and german_hits >= 2:
        return "de"
    if english_hits >= german_hits * 2 and english_hits >= 2:
        return "en"
    return "mixed"


def _resolve_link(base_url: str, href: str) -> str:
    if href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return urljoin(base_url, href)


def _unwrap_markdown_href(href: str) -> tuple[str, bool]:
    href = href.strip()
    if href.startswith("<") and href.endswith(">"):
        return href[1:-1].strip(), True
    return href, False


def _format_markdown_href(href: str, *, prefer_angles: bool) -> str:
    if prefer_angles or any(char.isspace() for char in href) or any(
        char in href for char in "()"
    ):
        return f"<{href}>"
    return href


def _split_trailing_url_punctuation(href: str) -> tuple[str, str]:
    suffix = ""
    while href:
        last = href[-1]
        if last in TRAILING_URL_PUNCTUATION:
            suffix = last + suffix
            href = href[:-1]
            continue
        opener = TRAILING_URL_CLOSERS.get(last)
        if opener and href.count(last) > href.count(opener):
            suffix = last + suffix
            href = href[:-1]
            continue
        break
    return href, suffix


def _normalize_or_drop_link(base_url: str, href: str) -> tuple[str | None, bool]:
    href, _ = _unwrap_markdown_href(href)
    absolute = _resolve_link(base_url, href)
    if not absolute.startswith(("http://", "https://")):
        return href, False
    decision = evaluate_rag_url(absolute)
    if not decision.is_allowed:
        return None, True
    normalized = decision.normalized_url or absolute
    if normalized != absolute:
        return normalized, False
    return absolute, False


def clean_markdown_body(body: str, *, source_url: str) -> CleanedBody:
    blocked_links_removed = 0
    normalized_links_rewritten = 0
    removed_noise_lines = 0
    cleaned_lines: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line
        if any(pattern.match(line) for pattern in LINE_NOISE_PATTERNS):
            removed_noise_lines += 1
            continue

        image_match = IMAGE_LINE_RE.match(line)
        if image_match:
            target, dropped = _normalize_or_drop_link(source_url, image_match.group(1))
            if dropped or (target and "/fileadmin/_processed_/" in target):
                blocked_links_removed += 1
                removed_noise_lines += 1
                continue

        def replace_markdown_link(match: re.Match[str]) -> str:
            nonlocal blocked_links_removed, normalized_links_rewritten
            is_image, label, href = match.groups()
            href_value, href_was_angled = _unwrap_markdown_href(href)
            target, dropped = _normalize_or_drop_link(source_url, href_value)
            if dropped:
                blocked_links_removed += 1
                return "" if is_image else label.strip()
            if target != href_value:
                normalized_links_rewritten += 1
            formatted_target = _format_markdown_href(target, prefer_angles=href_was_angled)
            return f"{'!' if is_image else ''}[{label}]({formatted_target})"

        line = MARKDOWN_LINK_RE.sub(replace_markdown_link, line)

        def replace_autolink(match: re.Match[str]) -> str:
            nonlocal blocked_links_removed, normalized_links_rewritten
            href = match.group(1)
            target, dropped = _normalize_or_drop_link(source_url, href)
            if dropped:
                blocked_links_removed += 1
                return ""
            if target != href:
                normalized_links_rewritten += 1
            return f"<{target}>"

        line = AUTOLINK_RE.sub(replace_autolink, line)

        def replace_raw_url(match: re.Match[str]) -> str:
            nonlocal blocked_links_removed, normalized_links_rewritten
            href, suffix = _split_trailing_url_punctuation(match.group(1))
            if not href:
                return suffix
            target, dropped = _normalize_or_drop_link(source_url, href)
            if dropped:
                blocked_links_removed += 1
                return suffix
            if target != href:
                normalized_links_rewritten += 1
            return f"{target}{suffix}"

        line = RAW_URL_RE.sub(replace_raw_url, line)
        line = re.sub(r"\s{2,}", " ", line).rstrip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return CleanedBody(
        text=text,
        blocked_links_removed=blocked_links_removed,
        normalized_links_rewritten=normalized_links_rewritten,
        removed_noise_lines=removed_noise_lines,
    )


def infer_document_kind(meta: dict[str, str], body: str) -> str:
    text = f"{meta.get('source_url', '')} {meta.get('title', '')} {body[:2000]}".lower()
    if "modulhandbuch" in text or "modulbeschreibung" in text:
        return "module_handbook"
    if any(keyword in text for keyword in ("satzung", "ordnung", "verkuendungsblatt", "verordnung")):
        return "regulation"
    if any(keyword in text for keyword in ("formulare", "formular", "antrag")):
        return "form"
    if "faq" in text or "häufig gestellte fragen" in text:
        return "faq"
    if any(keyword in text for keyword in ("sprechzeiten", "ansprechperson", "beratung", "kontakt")):
        return "contact_service"
    if any(keyword in text for keyword in ("broschüre", "broschuere", "flyer")):
        return "brochure"
    return "service_page"


def infer_faculty(meta: dict[str, str]) -> str:
    source_url = meta.get("source_url", "").lower()
    for code, label in (
        ("f1", "Fakultät I"),
        ("f2", "Fakultät II"),
        ("f3", "Fakultät III"),
        ("f4", "Fakultät IV"),
        ("f5", "Fakultät V"),
    ):
        if source_url.startswith(f"https://{code}.") or f"/{code}/" in source_url:
            return label
    return meta.get("faculty", "")


def infer_topic_tags(meta: dict[str, str], body: str) -> tuple[str, ...]:
    text = f"{meta.get('source_url', '')} {meta.get('title', '')} {body[:4000]}".lower()
    tags = []
    for keyword, tag in TOPIC_KEYWORDS.items():
        if keyword in text and tag not in tags:
            tags.append(tag)
    return tuple(tags[:6])


def infer_source_family(meta: dict[str, str], body: str) -> str:
    if is_technical_api_source(meta):
        return "technical_api_docs"

    faculty = meta.get("faculty", "").strip()
    if faculty:
        slug = (
            faculty.lower()
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace(" ", "_")
        )
        return slug

    source_url = meta.get("source_url", "")
    parsed = urlparse(source_url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    for faculty_code in ("f1", "f2", "f3", "f4", "f5"):
        if host.startswith(faculty_code + ".") or f"/{faculty_code}/" in path:
            return f"faculty_{faculty_code[-1]}"

    if "akademische-angelegenheiten" in path:
        return "akademische_angelegenheiten"
    if "international" in path or host.startswith("international."):
        return "internationales"
    if "bibliothek" in path or host.startswith("bibliothek."):
        return "bibliothek"
    if "servicezentrum-beratung" in path:
        return "servicezentrum_beratung"
    if host:
        return host.replace(".", "_")
    return "zentral"


def infer_document_group(
    source_family: str,
    document_kind: str,
    topic_tags: tuple[str, ...],
) -> str:
    topic = topic_tags[0] if topic_tags else "allgemein"
    return f"{source_family}_{document_kind}_{topic}"


def is_library_catalog_source(meta: dict[str, str]) -> bool:
    source_url = meta.get("source_url", "").lower()
    if not source_url:
        return False
    parsed = urlparse(source_url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return host in LIBRARY_CATALOG_HOSTS or "vufind-record" in path


def is_technical_api_source(meta: dict[str, str]) -> bool:
    source_url = meta.get("source_url", "").lower()
    title = meta.get("title", "").lower()
    if any(marker in title for marker in TECHNICAL_API_TITLE_MARKERS):
        return True
    if not source_url:
        return False
    parsed = urlparse(source_url)
    path = parsed.path.lower()
    return any(marker in path for marker in TECHNICAL_API_PATH_MARKERS)


def detect_error_page_pattern(meta: dict[str, str], body: str, *, words: int) -> str | None:
    sample = f"{meta.get('title', '')}\n{body[:2500]}"
    for reason, pattern, require_sparse_content in ERROR_PAGE_PATTERNS:
        if not pattern.search(sample):
            continue
        if require_sparse_content and words > 160:
            continue
        return reason
    return None


def assess_markdown_file(
    meta: dict[str, str],
    body: str,
    *,
    duplicate: bool = False,
) -> QualityAssessment:
    words = count_words(body)
    tables = count_tables(body)
    language = detect_language(
        body,
        hints=f"{meta.get('title', '')} {meta.get('source_url', '')}",
    )
    document_kind = infer_document_kind(meta, body)
    topic_tags = infer_topic_tags(meta, body)
    source_family = infer_source_family(meta, body)
    document_group = infer_document_group(source_family, document_kind, topic_tags)

    reasons: list[str] = []
    score = 100
    content_type = meta.get("content_type", "html")
    threshold = MIN_WORDS_PDF if content_type == "pdf" else MIN_WORDS_HTML

    if not meta.get("source_url"):
        reasons.append("fehlende_quell_url")
        score -= 50
    if not body.strip():
        reasons.append("leerer_body")
        score -= 80
    if is_library_catalog_source(meta):
        reasons.append("bibliothekskatalog_ohne_rag_mehrwert")
        score -= 90
    if is_technical_api_source(meta):
        reasons.append("technische_api_dokumentation_ohne_rag_mehrwert")
        score -= 90

    if words < threshold:
        reasons.append(f"zu_wenig_text:{words}")
        score -= 40

    body_lower = body.lower()
    error_page_reason = detect_error_page_pattern(meta, body, words=words)
    if error_page_reason:
        reasons.append(f"fehlerseitenmuster:{error_page_reason}")
        score -= 60

    placeholder_count = body.count("intentionally omitted")
    if placeholder_count >= 8:
        reasons.append(f"viele_bildplatzhalter:{placeholder_count}")
        score -= 10

    if "typo3backend-live" in body_lower or "f5-preview" in body_lower:
        reasons.append("enthaelt_preview_backend_links")
        score -= 20

    if "scrollen für mehr infos" in body_lower or "scrollen fuer mehr infos" in body_lower:
        reasons.append("enthaelt_landingpage_boilerplate")
        score -= 10

    if language == "en":
        reasons.append("englischsprachiger_inhalt")
        score -= 35
    elif language == "mixed":
        reasons.append("sprachlich_gemischt")
        score -= 10

    if duplicate:
        reasons.append("duplikat_aeltere_version")
        score -= 25

    hard_reject_prefixes = (
        "fehlerseitenmuster",
        "bibliothekskatalog_ohne_rag_mehrwert",
        "technische_api_dokumentation_ohne_rag_mehrwert",
    )
    keep = score >= 45 and not any(
        reason.startswith(prefix) for reason in reasons for prefix in hard_reject_prefixes
    )
    return QualityAssessment(
        keep=keep,
        score=max(score, 0),
        reasons=tuple(reasons),
        words=words,
        tables=tables,
        language=language,
        document_kind=document_kind,
        source_family=source_family,
        document_group=document_group,
        topic_tags=topic_tags,
    )


def find_duplicates(md_files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for filepath in md_files:
        stem = filepath.stem
        parts = stem.split("_", 1)
        slug = parts[1] if len(parts) == 2 else stem
        groups[slug].append(filepath)
    return {slug: paths for slug, paths in groups.items() if len(paths) > 1}


def duplicate_delete_set(md_files: list[Path]) -> set[Path]:
    duplicate_delete: set[Path] = set()
    for paths in find_duplicates(md_files).values():
        for old_path in sorted(paths, reverse=True)[1:]:
            duplicate_delete.add(old_path)
    return duplicate_delete


def enrich_metadata(
    meta: dict[str, str],
    assessment: QualityAssessment,
    cleaned: CleanedBody,
) -> dict[str, str]:
    enriched = dict(meta)
    enriched["faculty"] = infer_faculty(meta)
    enriched["language"] = assessment.language
    enriched["quality_score"] = str(assessment.score)
    enriched["quality_flags"] = "|".join(assessment.reasons)
    enriched["document_kind"] = assessment.document_kind
    enriched["source_family"] = assessment.source_family
    enriched["document_group"] = assessment.document_group
    enriched["topic_tags"] = "|".join(assessment.topic_tags)
    enriched["blocked_links_removed"] = str(cleaned.blocked_links_removed)
    enriched["normalized_links_rewritten"] = str(cleaned.normalized_links_rewritten)
    enriched["removed_noise_lines"] = str(cleaned.removed_noise_lines)
    return enriched
