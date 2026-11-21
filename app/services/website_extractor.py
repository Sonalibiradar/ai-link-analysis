"""Fetch a website safely and reduce it to a compact, LLM-friendly text digest.

Safety:  every URL (and every redirect hop) goes through validate_public_url (SSRF guard),
         responses are streamed with a hard byte cap, only HTML is accepted.
Cost:    scripts/styles/nav/footer boilerplate is stripped, duplicate lines are removed and
         the final digest is truncated to MAX_CONTENT_CHARS before it ever reaches the LLM.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from app.config import Settings
from app.errors import BlockedURLError, FetchError, InvalidURLError
from app.utils.validators import validate_public_url

# Pages most likely to describe what the business actually offers.
PRIORITY_PATH_HINTS = [
    "services", "service", "solutions", "products", "product", "what-we-do", "offerings",
    "about", "about-us", "company", "pricing", "industries",
]

_PARSER_CANDIDATES = ("lxml", "html.parser")


@dataclass
class ExtractedPage:
    url: str
    title: str = ""
    meta_description: str = ""
    site_name: str = ""
    json_ld: list[str] = field(default_factory=list)
    nav_items: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)  # (anchor text, absolute href)


@dataclass
class FetchedPage:
    final_url: str
    html: str


def _make_soup(html: str) -> BeautifulSoup:
    for parser in _PARSER_CANDIDATES:
        try:
            return BeautifulSoup(html, parser)
        except Exception:  # parser missing / failed
            continue
    raise FetchError("Could not parse the page HTML.")


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ----------------------------------------------------------------------------- fetching


def fetch_html(url: str, settings: Settings, session: requests.Session | None = None) -> FetchedPage:
    """GET a page following redirects manually so each hop is re-validated against SSRF rules."""
    session = session or requests.Session()
    current = validate_public_url(url)
    headers = {
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        "Accept-Language": "en;q=0.9",
    }
    for _ in range(settings.fetch_max_redirects + 1):
        try:
            resp = session.get(
                current,
                headers=headers,
                timeout=(settings.fetch_connect_timeout_s, settings.fetch_read_timeout_s),
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.Timeout:
            raise FetchError("The website took too long to respond.")
        except requests.exceptions.SSLError:
            raise FetchError("The website has an invalid or untrusted SSL certificate.")
        except requests.exceptions.ConnectionError:
            raise FetchError("Could not connect to the website. It may be down or blocking requests.")
        except requests.exceptions.RequestException as e:
            raise FetchError(f"Could not fetch the website ({type(e).__name__}).")

        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise FetchError("The website returned a redirect without a destination.")
                try:
                    current = validate_public_url(urljoin(current, location))
                except (InvalidURLError, BlockedURLError) as e:
                    raise FetchError(f"The website redirected to a disallowed address: {e.message}")
                continue

            if resp.status_code in (401, 403, 429, 451, 503):
                raise FetchError(
                    f"The website refused automated access (HTTP {resp.status_code}). "
                    "It may use bot protection or require a login."
                )
            if resp.status_code >= 400:
                raise FetchError(f"The website returned HTTP {resp.status_code}.")

            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype and "html" not in ctype and "xml" not in ctype:
                raise FetchError(f"The URL does not point to an HTML page (Content-Type: {ctype.split(';')[0]}).")

            chunks, size = [], 0
            for chunk in resp.iter_content(chunk_size=16_384):
                chunks.append(chunk)
                size += len(chunk)
                if size >= settings.fetch_max_bytes:
                    break  # truncate oversized pages instead of failing
            raw = b"".join(chunks)[: settings.fetch_max_bytes]
            encoding = resp.encoding if resp.encoding and "charset" in ctype else None
            html = raw.decode(encoding or "utf-8", errors="replace")
            return FetchedPage(final_url=current, html=html)
        finally:
            resp.close()

    raise FetchError("The website redirected too many times.")


# ----------------------------------------------------------------------------- parsing


def extract_content(html: str, page_url: str) -> ExtractedPage:
    soup = _make_soup(html)
    page = ExtractedPage(url=page_url)

    if soup.title and soup.title.string:
        page.title = _clean(soup.title.string)

    def meta(**attrs) -> str:
        tag = soup.find("meta", attrs=attrs)
        return _clean(tag.get("content", "")) if tag else ""

    page.meta_description = meta(name="description") or meta(property="og:description")
    page.site_name = meta(property="og:site_name") or meta(name="application-name")

    for tag in soup.find_all("script", type="application/ld+json")[:5]:
        raw = (tag.string or tag.get_text() or "").strip()
        if raw:
            page.json_ld.append(_clean(raw)[:1200])

    # Links (before removing nav/footer so we can discover service/about pages).
    seen_links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(page_url, href).split("#")[0]
        if absolute in seen_links:
            continue
        seen_links.add(absolute)
        page.links.append((_clean(a.get_text(" "))[:80], absolute))

    for nav in soup.find_all("nav")[:2]:
        for a in nav.find_all("a"):
            label = _clean(a.get_text(" "))
            if 1 < len(label) <= 40 and label not in page.nav_items:
                page.nav_items.append(label)
    page.nav_items = page.nav_items[:25]

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "nav", "footer", "aside", "form"]):
        tag.decompose()

    for h in soup.find_all(["h1", "h2", "h3"]):
        text = _clean(h.get_text(" "))
        if 2 < len(text) <= 160 and text not in page.headings:
            page.headings.append(text)
    page.headings = page.headings[:40]

    lines, seen = [], set()
    for raw_line in soup.get_text("\n").splitlines():
        line = _clean(raw_line)
        if len(line) < 3:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    page.text = "\n".join(lines)
    return page


def _same_site(a: str, b: str) -> bool:
    host = lambda u: (urlsplit(u).hostname or "").lower().removeprefix("www.")
    return host(a) == host(b)


def pick_extra_pages(home: ExtractedPage, limit: int) -> list[str]:
    """Choose a few same-site pages (services/about/...) that best describe the business."""
    if limit <= 0:
        return []
    scored: list[tuple[int, str]] = []
    home_path = urlsplit(home.url).path.rstrip("/")
    for label, href in home.links:
        if not _same_site(home.url, href):
            continue
        parts = urlsplit(href)
        path = parts.path.rstrip("/")
        if not path or path == home_path or parts.query:
            continue
        if re.search(r"\.(pdf|jpe?g|png|gif|svg|zip|docx?|xlsx?|mp4)$", path, re.I):
            continue
        slug = path.strip("/").lower().split("/")
        hay = " ".join(slug + [label.lower().replace(" ", "-")])
        for rank, hint in enumerate(PRIORITY_PATH_HINTS):
            if hint in hay and len(slug) <= 2:
                scored.append((rank, href))
                break
    scored.sort(key=lambda t: t[0])
    picked: list[str] = []
    for _, href in scored:
        if href not in picked:
            picked.append(href)
        if len(picked) >= limit:
            break
    return picked


# ----------------------------------------------------------------------------- orchestration


@dataclass
class WebsiteDigest:
    final_url: str
    pages: list[ExtractedPage]
    context: str  # compact text sent to the LLM

    @property
    def total_text_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def build_context(pages: list[ExtractedPage], max_chars: int) -> str:
    """Compact digest: metadata first (high signal), then body text under a per-page budget."""
    parts: list[str] = []
    home = pages[0]
    parts.append(f"WEBSITE: {home.url}")
    if home.site_name:
        parts.append(f"SITE NAME: {home.site_name}")
    if home.title:
        parts.append(f"TITLE: {home.title}")
    if home.meta_description:
        parts.append(f"META DESCRIPTION: {home.meta_description}")
    if home.nav_items:
        parts.append("NAVIGATION: " + " | ".join(home.nav_items))
    if home.json_ld:
        parts.append("STRUCTURED DATA: " + " ".join(home.json_ld)[:1500])

    header = "\n".join(parts)
    remaining = max(max_chars - len(header), 1000)
    per_page = remaining // len(pages)
    # Give the homepage a larger share; unused budget from short pages is not redistributed (keeps it simple).
    budgets = [int(per_page * 1.4)] + [int(per_page * 0.8)] * (len(pages) - 1) if len(pages) > 1 else [remaining]

    for page, budget in zip(pages, budgets):
        block = [f"\n--- PAGE: {page.url} ---"]
        if page is not home and page.title:
            block.append(f"TITLE: {page.title}")
        if page.headings:
            block.append("HEADINGS: " + " | ".join(page.headings))
        block.append("TEXT:\n" + page.text)
        text = "\n".join(block)
        parts.append(text[:budget])
    return ("\n".join(parts))[:max_chars]


def extract_website(url: str, settings: Settings, session: requests.Session | None = None) -> WebsiteDigest:
    session = session or requests.Session()
    first = fetch_html(url, settings, session)
    home = extract_content(first.html, first.final_url)
    pages = [home]

    extra_urls = pick_extra_pages(home, settings.extra_pages)
    if extra_urls:
        def _fetch(u: str) -> ExtractedPage | None:
            try:
                fetched = fetch_html(u, settings, requests.Session())
                return extract_content(fetched.html, fetched.final_url)
            except Exception:
                return None  # extra pages are best-effort

        with ThreadPoolExecutor(max_workers=len(extra_urls)) as pool:
            for page in pool.map(_fetch, extra_urls):
                if page and page.text:
                    pages.append(page)

    context = build_context(pages, settings.max_content_chars)
    return WebsiteDigest(final_url=first.final_url, pages=pages, context=context)
