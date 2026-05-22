"""
Gemeinsame Hilfsfunktionen fuer Crawler und Resume-Crawler.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import httpx
from crawl4ai import CrawlerRunConfig

DEFAULT_SITEMAP_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
)
LOW_VALUE_TITLE_MARKERS = (
    "anfahrt",
    "campusleben",
    "galerie",
    "lageplan",
    "oeffnungszeiten",
    "rueckblick",
    "standort",
)
ACTIONABLE_MARKERS = (
    "antrag",
    "bewerbung",
    "faq",
    "formular",
    "kontakt",
    "ordnung",
    "pruef",
    "rueckmeldung",
    "sprech",
)
WORD_RE = re.compile(r"\b[\w-]+\b", re.UNICODE)
LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)]+|/[^)]+)\)|https?://\S+")


@dataclass(frozen=True)
class MarkdownQuality:
    keep: bool
    reasons: tuple[str, ...]
    word_count: int
    link_count: int
    duplicate_short_lines: int


def build_crawler_config() -> CrawlerRunConfig:
    return CrawlerRunConfig(
        css_selector="main, .content-main, #content, .frame-default",
        excluded_tags=[
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "iframe",
            "script",
            "style",
            "noscript",
        ],
        excluded_selector=(
            "nav.main-menu, .main-menu, .main-menu__mainmenu, "
            "ul#tabmenu1, ul[role='menubar'], "
            "header, .quicklinks, "
            ".breadcrumb, .breadcrumbs, [aria-label='breadcrumb'], "
            "footer, .footer, .site-footer, "
            "#CybotCookiebotDialog, #CybotCookiebotDialogBody, "
            "[id*='Cookiebot'], [class*='cookiebot'], "
            "aside, .sidebar, .widget, "
            ".search-form, form[role='search'], "
            ".social-media, .share-buttons"
        ),
    )


def assess_markdown_quality(
    url: str,
    title: str,
    body: str,
    *,
    is_high_value: bool,
) -> MarkdownQuality:
    words = WORD_RE.findall(body)
    word_count = len(words)
    link_count = len(LINK_RE.findall(body))

    short_lines = []
    for line in body.splitlines():
        cleaned = re.sub(r"^[#*\-\d.\s]+", "", line).strip().lower()
        if 3 <= len(cleaned) <= 60:
            short_lines.append(cleaned)
    duplicate_short_lines = sum(
        count for _, count in Counter(short_lines).items() if count >= 3
    )

    title_text = f"{url} {title}".lower()
    reasons: list[str] = []
    min_words = 18 if is_high_value else 35

    if word_count < min_words:
        reasons.append(f"zu_wenig_text:{word_count}")

    if link_count >= 12 and word_count / max(link_count, 1) < 7:
        reasons.append("zu_viele_links_fuer_zu_wenig_prosa")

    if duplicate_short_lines >= 6 and word_count < 220:
        reasons.append("vermutlich_navigation_footer_fragmente")

    if any(marker in title_text for marker in LOW_VALUE_TITLE_MARKERS):
        has_actionable_signal = any(marker in title_text for marker in ACTIONABLE_MARKERS)
        if not has_actionable_signal and word_count < 220:
            reasons.append("generische_oeffnungszeiten_promo_uebersicht")

    return MarkdownQuality(
        keep=not reasons,
        reasons=tuple(reasons),
        word_count=word_count,
        link_count=link_count,
        duplicate_short_lines=duplicate_short_lines,
    )


def format_host_summary(host_counts: Counter[str], *, max_hosts: int = 8) -> list[str]:
    if not host_counts:
        return []
    lines = ["Crawl-Hosts (Top):"]
    for host, count in host_counts.most_common(max_hosts):
        lines.append(f"  {host}: {count}")
    return lines


def _extract_sitemap_locs(xml_text: str) -> list[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    locs: list[str] = []
    for elem in root.iter():
        if elem.tag.endswith("loc") and elem.text:
            loc = elem.text.strip()
            if loc:
                locs.append(loc)
    return locs


async def discover_sitemap_links(seed_urls: Iterable[str], *, timeout: int = 15) -> set[str]:
    base_urls: set[str] = set()
    sitemap_urls: set[str] = set()

    for seed in seed_urls:
        parsed = urlparse(seed)
        if not parsed.scheme or not parsed.netloc:
            continue
        base = f"{parsed.scheme}://{parsed.netloc}"
        base_urls.add(base)
        for path in DEFAULT_SITEMAP_PATHS:
            sitemap_urls.add(base + path)

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        for base in base_urls:
            try:
                resp = await client.get(base + "/robots.txt")
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.partition(":")[2].strip()
                    if sitemap_url:
                        sitemap_urls.add(sitemap_url)

        pending = list(sitemap_urls)
        seen_sitemaps: set[str] = set()
        discovered_links: set[str] = set()

        while pending and len(seen_sitemaps) < 25:
            sitemap_url = pending.pop()
            if sitemap_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sitemap_url)

            try:
                resp = await client.get(sitemap_url)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue

            for loc in _extract_sitemap_locs(resp.text):
                lower = loc.lower()
                if lower.endswith(".xml") and "sitemap" in lower:
                    if loc not in seen_sitemaps:
                        pending.append(loc)
                else:
                    discovered_links.add(loc)

    return discovered_links
