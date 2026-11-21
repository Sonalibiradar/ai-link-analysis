"""Offline tests: no network and no real LLM calls (DNS, HTTP and the LLM are faked).

Endpoint tests need fastapi + httpx and are skipped automatically when those are missing.
"""

import json
import socket
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.errors import (  # noqa: E402
    BlockedURLError, FetchError, InsufficientContentError, InvalidURLError, LLMConfigError, LLMOutputError,
)
from app.services import competitor_service  # noqa: E402
from app.services.analysis_service import AnalysisService  # noqa: E402
from app.services.llm_service import LLMClient, LLMResult, estimate_cost_usd, extract_json  # noqa: E402
from app.services.website_extractor import (  # noqa: E402
    extract_content, extract_website, fetch_html, pick_extra_pages,
)
from app.utils.cache import TTLCache  # noqa: E402
from app.utils.validators import validate_public_url  # noqa: E402

PUBLIC_IP = "93.184.216.34"

SAMPLE_HTML = """
<html><head><title>Acme Web Studio - Websites for Small Business</title>
<meta name="description" content="We design affordable websites for small businesses.">
<meta property="og:site_name" content="Acme Web Studio">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Web Studio"}</script>
<style>body{color:red}</style></head>
<body><nav><a href="/services">Services</a><a href="/about">About</a><a href="/blog?page=2">Blog</a></nav>
<h1>Websites that win customers</h1>
<p>Acme Web Studio builds fast, affordable websites and online stores for small businesses in Pune.</p>
<p>Acme Web Studio builds fast, affordable websites and online stores for small businesses in Pune.</p>
<h2>Web Design</h2><p>Custom responsive design for startups and local shops.</p>
<h2>E-commerce</h2><p>Online stores with payments and inventory.</p>
<script>alert('x')</script><footer>Copyright Acme</footer></body></html>
"""

LLM_JSON = {
    "brand": {"name": "Acme Web Studio", "category": "Web Development", "summary": "Builds websites.",
              "main_products_services": ["Web design", "E-commerce"]},
    "analysis": {
        "competitor_analysis": {
            "brand_overview": "Small-business web studio.",
            "brand_profile": {"main_service": "Web design", "target_market": "Small businesses",
                              "key_strength": "Affordable", "opportunity": "Improve visibility"},
            "competitors": [
                {"name": "Acme Web Studio", "website": "https://acme.test", "main_service": "x"},  # the brand itself
                {"name": "Wix", "website": "https://www.wix.com", "main_service": "Site builder",
                 "target_market": "SMBs", "strengths": ["Brand"], "weaknesses_or_gaps": ["Generic"]},
                {"name": "wix", "website": "wix.com"},  # duplicate
                {"name": "Squarespace", "website": "squarespace.com"},
                {"name": "Webflow", "website": "webflow.com"},
            ],
            "strengths": ["Affordable"], "opportunities": ["Add case studies"],
        },
        "target_search_queries": [
            {"query": f"q{i}", "platforms": ["twitter", "Google Search", "Bogus"], "intent": "Service Discovery"}
            for i in range(7)
        ],
        "seo_improvements": {"primary_keyword": "web design pune", "secondary_keywords": "single string",
                             "recommended_title": "Web Design Pune | Acme", "meta_description": "Affordable sites.",
                             "supporting_content": [{"type": "FAQ", "title": "How much is a site?", "rationale": "r"}]},
        "top_services": [
            {"name": f"S{i}", "description": "d", "source": "inferred" if i == 0 else "website"} for i in range(6)
        ],
        "target_audience": {"primary_audience": ["Small businesses"], "geographic_focus": None},
    },
}


# ----------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    """Tests must not depend on the developer's local .env."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("FALLBACK_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "claude-sonnet-5-5")


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve hostnames without the network: private-looking names map to private IPs."""
    real = socket.getaddrinfo

    def fake(host, port, *a, **k):
        table = {
            "internal.example.com": "10.0.0.5",
            "meta.example.com": "169.254.169.254",
            "loop.example.com": "127.0.0.1",
            "v6.example.com": "::1",
            "mapped.example.com": "::ffff:127.0.0.1",
        }
        if host in table or host.endswith(".example.com") or host.endswith(".test") or host.endswith(".com"):
            ip = table.get(host, PUBLIC_IP)
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            return [(fam, socket.SOCK_STREAM, 6, "", (ip, port))]
        return real(host, port, *a, **k)

    monkeypatch.setattr(socket, "getaddrinfo", fake)


class FakeResp:
    def __init__(self, status=200, body=SAMPLE_HTML, headers=None):
        self.status_code = status
        self._body = body.encode() if isinstance(body, str) else body
        self.headers = {"Content-Type": "text/html; charset=utf-8", **(headers or {})}
        self.encoding = "utf-8"

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        r = self.routes.get(url)
        if isinstance(r, Exception):
            raise r
        return r or FakeResp(404)


class FakeLLM:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, system, user, attempts=2):
        self.calls += 1
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return LLMResult(text=text, provider="anthropic", model="claude-haiku-5-5",
                         input_tokens=7000, output_tokens=2500, latency_ms=1200)


def make_service(llm, html=SAMPLE_HTML):
    s = get_settings()
    from app.services.website_extractor import ExtractedPage, WebsiteDigest, build_context
    page = extract_content(html, "https://acme.test/")

    def extractor(url, settings):
        return WebsiteDigest(final_url="https://acme.test/", pages=[page], context=build_context([page], 4000))

    return AnalysisService(settings=s, extractor=extractor, llm_factory=lambda p, m, st: llm, cache=TTLCache(60, 10))


# ----------------------------------------------------------------------------- validators / SSRF


@pytest.mark.parametrize("bad", [
    "http://127.0.0.1", "http://localhost", "http://10.1.2.3", "http://192.168.1.1/admin",
    "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://2130706433/",
    "http://0.0.0.0", "https://internal.example.com", "https://meta.example.com",
    "https://loop.example.com", "https://v6.example.com", "https://mapped.example.com",
    "http://user:pass@example.com", "http://example.com:22", "http://printer.local",
])
def test_blocks_private_and_internal(bad):
    with pytest.raises((BlockedURLError, InvalidURLError)):
        validate_public_url(bad)


@pytest.mark.parametrize("bad", ["", "   ", "ftp://example.com", "file:///etc/passwd", "http://", "javascript:alert(1)",
                                 "http://exa mple.com", "http://intranet", "a" * 3000])
def test_rejects_invalid(bad):
    with pytest.raises(InvalidURLError):
        validate_public_url(bad)


def test_accepts_public_and_adds_scheme():
    assert validate_public_url("example.com") == "https://example.com/"
    assert validate_public_url("  http://Example.com/path?q=1#frag ") == "http://example.com/path?q=1"
    assert validate_public_url("https://example.com:8443/x").endswith(":8443/x")


# ----------------------------------------------------------------------------- extractor


def test_extract_content_cleans_and_dedupes():
    p = extract_content(SAMPLE_HTML, "https://acme.test/")
    assert p.title.startswith("Acme Web Studio")
    assert p.meta_description == "We design affordable websites for small businesses."
    assert p.site_name == "Acme Web Studio"
    assert "alert" not in p.text and "color:red" not in p.text and "Copyright" not in p.text
    assert p.text.count("builds fast, affordable websites") == 1  # duplicate paragraph removed
    assert "Web Design" in p.headings and "E-commerce" in p.headings
    assert p.nav_items[:2] == ["Services", "About"]
    assert any("Organization" in j for j in p.json_ld)


def test_pick_extra_pages_prefers_services_and_skips_query_urls():
    p = extract_content(SAMPLE_HTML, "https://acme.test/")
    picked = pick_extra_pages(p, 2)
    assert picked == ["https://acme.test/services", "https://acme.test/about"]


def test_fetch_follows_redirect_and_blocks_private_redirect():
    s = get_settings()
    ok = FakeSession({
        "https://a.example.com/": FakeResp(301, "", {"Location": "https://b.example.com/home"}),
        "https://b.example.com/home": FakeResp(200),
    })
    assert fetch_html("https://a.example.com/", s, ok).final_url == "https://b.example.com/home"

    evil = FakeSession({"https://a.example.com/": FakeResp(302, "", {"Location": "http://169.254.169.254/latest"})})
    with pytest.raises(FetchError, match="disallowed"):
        fetch_html("https://a.example.com/", s, evil)
    assert evil.calls == ["https://a.example.com/"]  # the internal address was never requested


@pytest.mark.parametrize("resp,match", [
    (FakeResp(403), "refused automated access"),
    (FakeResp(404), "HTTP 404"),
    (FakeResp(200, "%PDF", {"Content-Type": "application/pdf"}), "not point to an HTML"),
    (requests.exceptions.Timeout(), "too long"),
    (requests.exceptions.ConnectionError(), "Could not connect"),
    (requests.exceptions.SSLError(), "SSL"),
])
def test_fetch_errors_are_friendly(resp, match):
    with pytest.raises(FetchError, match=match):
        fetch_html("https://x.example.com/", get_settings(), FakeSession({"https://x.example.com/": resp}))


def test_fetch_redirect_loop_and_size_cap():
    s = get_settings()
    loop = FakeSession({"https://x.example.com/": FakeResp(302, "", {"Location": "https://x.example.com/"})})
    with pytest.raises(FetchError, match="too many times"):
        fetch_html("https://x.example.com/", s, loop)
    big = FakeSession({"https://x.example.com/": FakeResp(200, "<p>" + "a" * 5_000_000)})
    assert len(fetch_html("https://x.example.com/", s, big).html) <= s.fetch_max_bytes


def test_extract_website_best_effort_extra_pages():
    s = get_settings()
    session = FakeSession({
        "https://acme.test/": FakeResp(200),
        "https://acme.test/services": FakeResp(200, "<html><title>Services</title><body><h1>Our Services</h1><p>SEO and hosting packages.</p></body></html>"),
        "https://acme.test/about": requests.exceptions.ConnectionError(),  # failure must not break the request
    })
    # extra pages use their own Session, so route them through the same fake
    import app.services.website_extractor as we
    orig = we.requests.Session
    we.requests.Session = lambda: session
    try:
        d = extract_website("https://acme.test/", s, session)
    finally:
        we.requests.Session = orig
    assert len(d.pages) == 2 and "SEO and hosting" in d.context
    assert len(d.context) <= s.max_content_chars


# ----------------------------------------------------------------------------- LLM helpers


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": {"b": 2}} Hope it helps') == {"a": {"b": 2}}
    for bad in ["", "no json here", "{broken", "[1,2]"]:
        with pytest.raises(LLMOutputError):
            extract_json(bad)


def test_cost_estimate():
    assert estimate_cost_usd("claude-haiku-5-5", 7000, 2500) == pytest.approx(0.00195)
    assert estimate_cost_usd("unknown-model", 1, 1) is None
    assert estimate_cost_usd("claude-haiku-5-5", None, 5) is None


def test_llm_client_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        LLMClient("anthropic", "claude-haiku-5-5", get_settings())
    with pytest.raises(LLMConfigError):
        LLMClient("nope", "m", get_settings())


class PostSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class JResp:
    def __init__(self, status, payload=None):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload or {})

    def json(self):
        return self._p


@pytest.mark.parametrize("provider,payload", [
    ("anthropic", {"content": [{"type": "text", "text": "{}"}], "usage": {"input_tokens": 10, "output_tokens": 5}}),
    ("openai", {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
    ("gemini", {"candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}),
])
def test_llm_client_parses_each_provider(monkeypatch, provider, payload):
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "test-key")
    sess = PostSession([JResp(200, payload)])
    r = LLMClient(provider, "m", get_settings(), sess).complete("sys", "user")
    assert (r.text, r.input_tokens, r.output_tokens) == ("{}", 10, 5)
    url, kw = sess.calls[0]
    assert "test-key" not in url  # keys must never be placed in the URL


def test_llm_client_retries_then_fails_cleanly(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    ok = {"content": [{"type": "text", "text": "{}"}], "usage": {}}
    assert LLMClient("anthropic", "m", get_settings(), PostSession([JResp(529), JResp(200, ok)])).complete("s", "u").text == "{}"
    from app.errors import LLMError, LLMTimeoutError
    with pytest.raises(LLMTimeoutError):
        LLMClient("anthropic", "m", get_settings(), PostSession([requests.exceptions.Timeout()] * 2)).complete("s", "u")
    with pytest.raises(LLMConfigError):
        LLMClient("anthropic", "m", get_settings(), PostSession([JResp(401)])).complete("s", "u")
    with pytest.raises(LLMError):
        LLMClient("anthropic", "m", get_settings(), PostSession([JResp(500)] * 2)).complete("s", "u")
    trunc = {"content": [{"type": "text", "text": "{"}], "stop_reason": "max_tokens"}
    with pytest.raises(LLMOutputError, match="cut off"):
        LLMClient("anthropic", "m", get_settings(), PostSession([JResp(200, trunc)])).complete("s", "u")


# ----------------------------------------------------------------------------- analysis service


def test_analysis_shapes_and_cleans_llm_output():
    resp = make_service(FakeLLM([json.dumps(LLM_JSON)])).analyze("https://acme.test/")
    a = resp.analysis
    assert resp.success and resp.brand.name == "Acme Web Studio"
    # competitors: brand itself + duplicate removed, domains normalized, labelled as unverified AI suggestions
    names = [c.name for c in a.competitor_analysis.competitors]
    assert names == ["Wix", "Squarespace", "Webflow"]
    assert all(c.source == "ai_suggested" and not c.verified for c in a.competitor_analysis.competitors)
    assert a.competitor_analysis.competitors[0].website == "wix.com"
    table = {r.comparison: r for r in a.competitor_analysis.comparison_table}
    assert table["Main Service"].your_brand == "Web design" and len(table["Main Service"].competitors) == 3
    # queries capped at 5, platforms normalized (twitter -> X, unknown dropped)
    assert len(a.target_search_queries) == 5
    assert a.target_search_queries[0].platforms == ["X", "Google"]
    # services capped at 5 with provenance labels; lenient coercion of str -> list and None -> ""
    assert len(a.top_services) == 5 and a.top_services[0].source == "inferred" and a.top_services[1].source == "website"
    assert a.seo_improvements.secondary_keywords == ["single string"]
    assert a.target_audience.geographic_focus == ""
    assert resp.meta.llm.estimated_cost_usd == pytest.approx(0.00195)
    assert "search-volume" in resp.meta.disclaimer
    dumped = resp.model_dump_json().lower()
    assert "search_volume" not in dumped  # we never fabricate volumes


def test_analysis_retries_on_bad_json_then_succeeds():
    llm = FakeLLM(["not json at all", json.dumps(LLM_JSON)])
    resp = make_service(llm).analyze("https://acme.test/")
    assert llm.calls == 2 and resp.brand.name == "Acme Web Studio"
    assert resp.meta.llm.input_tokens == 14000  # usage of both attempts is counted


def test_analysis_fails_cleanly_when_llm_never_returns_json():
    with pytest.raises(LLMOutputError):
        make_service(FakeLLM(["nope"])).analyze("https://acme.test/")


def test_cache_avoids_second_llm_call():
    llm = FakeLLM([json.dumps(LLM_JSON)])
    svc = make_service(llm)
    first, second = svc.analyze("https://acme.test/"), svc.analyze("https://acme.test/")
    assert llm.calls == 1 and not first.meta.cached and second.meta.cached


def test_thin_content_is_rejected():
    with pytest.raises(InsufficientContentError):
        make_service(FakeLLM(["{}"]), html="<html><body><div id='root'></div></body></html>").analyze("https://acme.test/")


def test_prompt_injection_text_is_treated_as_data():
    from app.services.analysis_service import SYSTEM_PROMPT
    assert "untrusted data" in SYSTEM_PROMPT


def test_competitor_warning_when_too_few():
    from app.schemas import Competitor
    cleaned, warnings = competitor_service.normalize_competitors([Competitor(name="Only One")], "Me", "https://me.com")
    assert len(cleaned) == 1 and warnings


# ----------------------------------------------------------------------------- HTTP endpoint


def _client(service):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import app.main as main
    main._service = service
    return TestClient(main.app, raise_server_exceptions=False)


def test_endpoint_success():
    r = _client(make_service(FakeLLM([json.dumps(LLM_JSON)]))).post("/link-analysis", json={"url": "https://acme.test/"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and set(body["analysis"]) == {
        "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}


@pytest.mark.parametrize("payload,status,code", [
    ({"url": "http://127.0.0.1"}, 400, "blocked_url"),
    ({"url": "ftp://example.com"}, 400, "invalid_url"),
    ({}, 422, "invalid_request"),
    ({"url": 123}, 422, "invalid_request"),
])
def test_endpoint_errors(payload, status, code):
    r = _client(make_service(FakeLLM(["{}"]))).post("/link-analysis", json=payload)
    assert r.status_code == status and r.json() == {"success": False, "error": {"code": code, "message": r.json()["error"]["message"]}}
