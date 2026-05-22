"""
Gemeinsame URL-Filterung und kleine SQLite-Ablage fuer den HsH-Crawler.

Das Ziel ist nicht, moeglichst viele Links zu crawlen, sondern moeglichst
nuetzliche Quellen fuer das RAG-System zu behalten. Deshalb bewertet dieses
Modul jede URL nach ihrem voraussichtlichen Wissenswert fuer das Korpus.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

try:
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover - script execution fallback
    from paths import DATA_DIR

ALLOWED_DOMAIN = "hs-hannover.de"
BLOCKED_DOMAINS = {
    "katalog.bib.hs-hannover.de",
    "serwiss.bib.hs-hannover.de",
    "typo3backend-live.hs-hannover.de",
}
CANONICAL_HOST_MAP = {
    "hs-hannover.de": "www.hs-hannover.de",
}
BLOCKED_HOST_SEGMENTS = {
    "backend",
    "preview",
    "staging",
    "test",
    "wp",
}
BLOCKED_HOST_FRAGMENT_MARKERS = (
    "backend",
    "-preview",
    ".preview",
    "-staging",
    ".staging",
    "-test",
    ".test",
    "-dev",
    ".dev",
)

DECISION_DB_PATH = DATA_DIR / "url_decisions.db"

ENGLISH_SECTION_PREFIX = "/en"
PROCESSED_ASSET_MARKER = "/fileadmin/_processed_/"
BLOCKED_APP_HOSTS = {
    "intranet.hs-hannover.de",
    "moodle.hs-hannover.de",
}
BLOCKED_AUTH_PATH_MARKERS = (
    "/login",
    "/logout",
    "/shibboleth",
    "/saml",
    "/oauth",
)
BROKEN_URL_MARKERS = ("*", "|", "<", ">", "{", "}", '"')
INDEX_FILENAMES = {
    "index.html",
    "index.htm",
    "index.php",
    "default.aspx",
    "default.asp",
}
NOISY_QUERY_PARAMS = {
    "cHash",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "pk_campaign",
    "pk_kwd",
    "pk_medium",
    "pk_source",
    "print",
    "ref",
    "refsrc",
    "share",
    "source",
    "tracking",
    "utm_campaign",
    "utm_content",
    "utm_id",
    "utm_medium",
    "utm_source",
    "utm_term",
}
NOISY_QUERY_PARAMS_LOWER = {param.lower() for param in NOISY_QUERY_PARAMS}
BLOCKED_PATH_MARKERS = (
    "/aktuelles/",
    "/news/",
    "/news-",
    "/news_",
    "/neuigkeiten/",
    "/veranstaltungen/",
    "/veranstaltung/",
    "/events/",
    "/event/",
    "/galerie/",
    "/gallery/",
    "/blog/",
    "/blogs/",
    "/tag/",
    "/tags/",
    "/category/",
    "/kategorie/",
    "/author/",
    "/apidoc",
    "/api-doc",
    "/api_docs",
    "/javadoc",
    "/swagger",
    "/openapi",
    "/feed",
    "/rss",
)
LOW_VALUE_PATH_MARKERS = (
    "/standort-oeffnungszeiten",
    "/oeffnungszeiten",
    "/anfahrt",
    "/lageplan",
    "/impressum",
    "/datenschutz",
)
HIGH_VALUE_KEYWORDS = {
    "antrag": 5,
    "bewerbung": 6,
    "campuscard": 4,
    "faq": 3,
    "immatrik": 6,
    "international": 4,
    "kontakt": 3,
    "modulhandbuch": 6,
    "ordnung": 7,
    "pruef": 6,
    "rueckmeldung": 6,
    "satzung": 7,
    "semestertermine": 6,
    "studium": 4,
    "studiengang": 4,
    "verkuendungsblatt": 7,
    "zulassung": 6,
}
LOW_VALUE_KEYWORDS = {
    "broschuere": -4,
    "broschueren": -4,
    "flyer": -4,
    "galerie": -5,
    "gallery": -5,
    "news": -4,
    "oeffnungszeiten": -3,
    "poster": -5,
    "plakat": -5,
    "stundenplan": -3,
    "veranstaltung": -5,
    "veranstaltungen": -5,
}
PREFERRED_PDF_KEYWORDS = {
    "antrag",
    "faq",
    "modulhandbuch",
    "ordnung",
    "satzung",
    "verkuendungsblatt",
}
DEPRIORITIZED_PDF_KEYWORDS = {
    "broschuere",
    "broschueren",
    "flyer",
    "poster",
    "stundenplan",
}

MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".tiff",
    ".ico",
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
    ".aac",
    ".flac",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
}

TECHNICAL_EXTENSIONS = {
    ".css",
    ".js",
    ".json",
    ".xml",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".xz",
}

OFFICE_EXTENSIONS = {
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".odt",
    ".ods",
    ".odp",
}

HTML_LIKE_EXTENSIONS = {
    "",
    ".html",
    ".htm",
    ".php",
    ".asp",
    ".aspx",
    ".jsp",
    ".jspx",
}


@dataclass(frozen=True)
class UrlDecision:
    url: str
    normalized_url: str
    decision: str
    reason: str
    priority_score: int = 0

    @property
    def is_allowed(self) -> bool:
        return self.decision.startswith("allow")

    @property
    def is_high_value(self) -> bool:
        return self.decision == "allow_high_value"

    @property
    def is_low_value(self) -> bool:
        return self.decision == "allow_low_value"


@dataclass
class DecisionStats:
    decision_counts: Counter[str] = field(default_factory=Counter)
    reason_counts: Counter[str] = field(default_factory=Counter)
    blocked_samples: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, decision: UrlDecision, *, sample_limit: int = 5) -> None:
        self.decision_counts[decision.decision] += 1
        self.reason_counts[decision.reason] += 1
        if decision.decision == "block":
            samples = self.blocked_samples[decision.reason]
            if decision.normalized_url not in samples and len(samples) < sample_limit:
                samples.append(decision.normalized_url)


def normalize_url(url: str) -> str:
    """Normalisiert Host/Pfad und entfernt Query/Fragment-Rauschen."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url

    scheme = parsed.scheme.lower()
    try:
        port = parsed.port
    except ValueError:
        return url
    hostname = (parsed.hostname or "").lower()
    netloc = hostname
    if netloc in CANONICAL_HOST_MAP:
        netloc = CANONICAL_HOST_MAP[netloc]
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{netloc}:{port}"

    path = parsed.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if path.endswith("/"):
        last_segment = path.rstrip("/").rsplit("/", 1)[-1].lower()
        if last_segment in INDEX_FILENAMES:
            path = path[: -len(last_segment) - 1] or "/"
    else:
        filename = path.rsplit("/", 1)[-1].lower()
        if filename in INDEX_FILENAMES:
            path = path[: -len(filename)] or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in NOISY_QUERY_PARAMS_LOWER:
            continue
        query_pairs.append((key, value))

    return parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path,
        query=urlencode(query_pairs, doseq=True),
        fragment="",
        params="",
    ).geturl()


def _host_has_blocked_marker(host: str) -> bool:
    segments = host.split(".")
    if any(segment in BLOCKED_HOST_SEGMENTS for segment in segments):
        return True
    return any(marker in host for marker in BLOCKED_HOST_FRAGMENT_MARKERS)


def _contains_path_marker(path: str, markers: tuple[str, ...]) -> bool:
    return any(marker in path for marker in markers)


def _score_text(text: str, weights: dict[str, int]) -> int:
    score = 0
    for keyword, weight in weights.items():
        if keyword in text:
            score += weight
    return score


def _compute_priority_score(host: str, path_lower: str, suffix: str) -> int:
    text = f"{host}{path_lower}"
    score = _score_text(text, HIGH_VALUE_KEYWORDS)
    score += _score_text(text, LOW_VALUE_KEYWORDS)

    if suffix == ".pdf":
        if any(keyword in text for keyword in PREFERRED_PDF_KEYWORDS):
            score += 3
        if any(keyword in text for keyword in DEPRIORITIZED_PDF_KEYWORDS):
            score -= 3

    return score


def is_same_domain(url: str) -> bool:
    """True nur fuer oeffentliche HsH-URLs ausserhalb geblockter Subdomains."""
    netloc = urlparse(normalize_url(url)).netloc.lower()
    if not netloc or netloc in BLOCKED_DOMAINS:
        return False
    return netloc == ALLOWED_DOMAIN or netloc.endswith("." + ALLOWED_DOMAIN)


def evaluate_rag_url(url: str) -> UrlDecision:
    """Bewertet, ob eine URL wahrscheinlich nuetzlich fuer das RAG-Korpus ist."""
    normalized = normalize_url(url)

    try:
        parsed = urlparse(normalized)
    except ValueError:
        return UrlDecision(url, normalized, "block", "blockiert_ungueltige_url")

    if parsed.scheme not in {"http", "https"}:
        return UrlDecision(url, normalized, "block", "blockiert_nicht_http")

    host = parsed.netloc.lower()
    path = parsed.path or "/"
    path_lower = path.lower()

    if host == "katalog.bib.hs-hannover.de":
        return UrlDecision(url, normalized, "block", "blockiert_bibliothekskatalog")

    if host in BLOCKED_DOMAINS:
        return UrlDecision(url, normalized, "block", "blockiert_backend_domain")

    if _host_has_blocked_marker(host):
        return UrlDecision(
            url,
            normalized,
            "block",
            "blockiert_preview_dev_backend_host",
        )

    if not is_same_domain(normalized):
        return UrlDecision(url, normalized, "block", "blockiert_externe_domain")

    if host in BLOCKED_APP_HOSTS:
        return UrlDecision(url, normalized, "block", "blockiert_nicht_oeffentliche_app")

    if any(marker in normalized for marker in BROKEN_URL_MARKERS):
        return UrlDecision(url, normalized, "block", "blockiert_vermutlich_kaputte_url")

    if path_lower == ENGLISH_SECTION_PREFIX or path_lower.startswith(
        ENGLISH_SECTION_PREFIX + "/"
    ):
        return UrlDecision(
            url,
            normalized,
            "block",
            "blockiert_englischen_en_bereich",
        )

    if any(marker in path_lower for marker in BLOCKED_AUTH_PATH_MARKERS):
        return UrlDecision(url, normalized, "block", "blockiert_auth_pfad")

    if PROCESSED_ASSET_MARKER in path_lower:
        return UrlDecision(url, normalized, "block", "blockiert_processed_asset")

    if _contains_path_marker(path_lower, BLOCKED_PATH_MARKERS):
        return UrlDecision(url, normalized, "block", "blockiert_low_value_pfad")

    suffix = Path(path_lower).suffix
    priority_score = _compute_priority_score(host, path_lower, suffix)

    if suffix in MEDIA_EXTENSIONS:
        return UrlDecision(url, normalized, "block", "blockiert_medien_datei")

    if suffix in TECHNICAL_EXTENSIONS:
        return UrlDecision(url, normalized, "block", "blockiert_technisches_asset")

    if suffix == ".pdf":
        if priority_score >= 4:
            return UrlDecision(
                url,
                normalized,
                "allow_high_value",
                "erlaubt_high_value_pdf",
                priority_score=priority_score,
            )
        return UrlDecision(
            url,
            normalized,
            "allow_low_value",
            "erlaubt_low_value_pdf",
            priority_score=priority_score,
        )

    if suffix in OFFICE_EXTENSIONS:
        return UrlDecision(
            url,
            normalized,
            "block",
            "blockiert_office_dokument_ohne_ingest_support",
        )

    if suffix in HTML_LIKE_EXTENSIONS:
        if _contains_path_marker(path_lower, LOW_VALUE_PATH_MARKERS):
            return UrlDecision(
                url,
                normalized,
                "allow_low_value",
                "erlaubt_low_value_html_pfad",
                priority_score=priority_score,
            )
        if priority_score >= 4:
            return UrlDecision(
                url,
                normalized,
                "allow_high_value",
                "erlaubt_high_value_html_seite",
                priority_score=priority_score,
            )
        return UrlDecision(
            url,
            normalized,
            "allow_low_value",
            "erlaubt_low_value_html_seite",
            priority_score=priority_score,
        )

    if priority_score >= 4:
        return UrlDecision(
            url,
            normalized,
            "allow_high_value",
            "erlaubt_high_value_datei",
            priority_score=priority_score,
        )

    return UrlDecision(
        url,
        normalized,
        "allow_low_value",
        "erlaubt_low_value_datei",
        priority_score=priority_score,
    )


def format_decision_summary(
    stats: DecisionStats,
    *,
    max_reasons: int = 10,
    max_samples: int = 3,
) -> list[str]:
    """Erzeugt kurze Zusammenfassungszeilen fuer Logs oder Reports."""
    total = sum(stats.decision_counts.values())
    if total == 0:
        return []

    lines = [
        (
            "RAG-Filter: "
            f"{total} Entscheidungen "
            f"(high={stats.decision_counts.get('allow_high_value', 0)}, "
            f"low={stats.decision_counts.get('allow_low_value', 0)}, "
            f"blockiert={stats.decision_counts.get('block', 0)})"
        )
    ]

    for reason, count in stats.reason_counts.most_common(max_reasons):
        lines.append(f"  {reason}: {count}")

    sample_reasons = sorted(
        stats.blocked_samples,
        key=lambda reason: (-stats.reason_counts[reason], reason),
    )[:max_samples]
    for reason in sample_reasons:
        lines.append(f"  Beispiel {reason}: {stats.blocked_samples[reason][0]}")

    return lines


class UrlDecisionStore:
    """Kleine SQLite-Ablage fuer URL-Entscheidungen."""

    def __init__(
        self,
        db_path: Path = DECISION_DB_PATH,
        *,
        commit_interval: int = 100,
    ) -> None:
        self.db_path = Path(db_path)
        self.commit_interval = max(1, commit_interval)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._pending_writes = 0

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS url_decisions (
                normalized_url TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                first_source TEXT NOT NULL,
                last_source TEXT NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_url_decisions_reason
            ON url_decisions(reason)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_url_decisions_last_seen
            ON url_decisions(last_seen)
            """
        )
        self._conn.commit()

    def record(self, decision: UrlDecision, *, source: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._conn.execute(
            """
            INSERT INTO url_decisions (
                normalized_url,
                decision,
                reason,
                first_seen,
                last_seen,
                first_source,
                last_source,
                seen_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(normalized_url) DO UPDATE SET
                decision = excluded.decision,
                reason = excluded.reason,
                last_seen = excluded.last_seen,
                last_source = excluded.last_source,
                seen_count = url_decisions.seen_count + 1
            """,
            (
                decision.normalized_url,
                decision.decision,
                decision.reason,
                now,
                now,
                source,
                source,
            ),
        )
        self._pending_writes += 1
        if self._pending_writes >= self.commit_interval:
            self.commit()

    def lookup(self, url: str) -> dict[str, str | int] | None:
        normalized = normalize_url(url)
        row = self._conn.execute(
            """
            SELECT normalized_url, decision, reason, first_seen, last_seen,
                   first_source, last_source, seen_count
            FROM url_decisions
            WHERE normalized_url = ?
            """,
            (normalized,),
        ).fetchone()
        if row is None:
            return None
        return {
            "normalized_url": row[0],
            "decision": row[1],
            "reason": row[2],
            "first_seen": row[3],
            "last_seen": row[4],
            "first_source": row[5],
            "last_source": row[6],
            "seen_count": row[7],
        }

    def commit(self) -> None:
        self._conn.commit()
        self._pending_writes = 0

    def close(self) -> None:
        if self._conn is not None:
            if self._pending_writes:
                self.commit()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "UrlDecisionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
