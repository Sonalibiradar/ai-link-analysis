# AI Website Link Analysis API

FastAPI service that takes a website URL, fetches and cleans the page content, asks an LLM to analyse it, and returns five structured sections: **competitors**, **top 5 search queries**, **SEO improvements**, **top 5 services** and **target audience**.

```
POST http://localhost:4000/link-analysis      {"url": "https://example.com"}
GET  http://localhost:4000/docs               (Swagger UI)
GET  http://localhost:4000/health
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
cp .env.example .env                                   # add ONE provider API key (see below)
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

```bash
curl -s -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}' | python -m json.tool
```

`.env` (never committed — `.gitignore` excludes it):

```
LLM_PROVIDER=anthropic            # anthropic | openai | gemini
LLM_MODEL=claude-haiku-5-5
FALLBACK_LLM_PROVIDER=anthropic
FALLBACK_LLM_MODEL=claude-sonnet-5-5
ANTHROPIC_API_KEY=...             # or OPENAI_API_KEY / GEMINI_API_KEY
```

Run the tests (fully offline — DNS, HTTP and the LLM are faked): `python -m pytest -q`

## How it works

```
URL ─► validate + SSRF check ─► fetch HTML (manual redirects, size cap) ─► + up to 2 same-site pages (services/about)
    ─► strip scripts/nav/footer, dedupe lines, cap to 12k chars ─► ONE LLM call (JSON-only prompt)
    ─► parse JSON (+1 repair retry) ─► Pydantic validation ─► post-process ─► response
```

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, endpoint, uniform error responses |
| `app/schemas.py` | Request, LLM-output contract (lenient coercion) and response models |
| `app/utils/validators.py` | URL validation and **SSRF protection** |
| `app/services/website_extractor.py` | Safe fetching, HTML → compact text digest |
| `app/services/llm_service.py` | Anthropic / OpenAI / Gemini REST adapters, retries, JSON extraction, cost estimate |
| `app/services/analysis_service.py` | Prompt, orchestration, validation, caching, fallback model |
| `app/services/competitor_service.py` | Competitor clean-up (dedupe, drop the brand itself) and comparison table |
| `app/utils/cache.py` | In-memory TTL cache |
| `scripts/benchmark_llms.py` | Measures latency / tokens / cost / completeness across models |

### Design decisions

- **One LLM call** for all five sections keeps cost and latency low; the website digest is sent once.
- **Grounding:** the prompt requires the model to use only the supplied content, labels each service `website` or `inferred`, and tells it to ignore instructions embedded in the page (prompt-injection defence).
- **Competitors are AI suggestions, not research.** There is no market-data API behind them, so every competitor carries `source: "ai_suggested"` and `verified: false`, the brand itself is removed, and a warning is added when fewer than 3 are returned.
- **No invented search volumes.** The schema has no volume field, the prompt forbids statistics, and `meta.disclaimer` says so.
- **Graceful failures:** every error returns `{"success": false, "error": {"code", "message"}}` with a meaningful status (below).
- **Fallback model:** if the primary model errors or returns unusable JSON, `FALLBACK_LLM_*` is tried; `meta.llm.used_fallback` shows when that happened.

### Error responses

| Status | `error.code` | When |
|---|---|---|
| 400 | `invalid_url` / `blocked_url` | Malformed URL, unsupported scheme, unresolvable host, private/internal target |
| 422 | `invalid_request` / `insufficient_content` | Bad body; page has no readable text (e.g. JS-only app) |
| 502 | `website_unreachable` / `llm_error` / `llm_invalid_output` | Site down, 403/bot-blocked, non-HTML, SSL error; provider error; unusable JSON |
| 503 | `llm_not_configured` | Missing/invalid API key or unknown model |
| 504 | `llm_timeout` | Provider timed out |

### Security

- **SSRF:** only http/https on ports 80/443/8080/8443, no credentials in URLs, no internal hostnames (`localhost`, `.local`, `.internal`…). The hostname is resolved and **every** returned address must be globally routable, which blocks loopback, RFC1918, link-local/cloud-metadata (`169.254.169.254`), CGNAT, IPv6 ULA/loopback, IPv4-mapped IPv6 and obfuscated forms like `http://2130706433`. Redirects are followed manually and **each hop is re-validated**.
- Responses are streamed with a 2 MB cap; only HTML content types are accepted; timeouts on connect/read.
- API keys come from environment variables only; the Gemini key is sent in a header, never in the URL.
- **Known limitation:** DNS rebinding (record changes between the check and the connection) is not fully closed. For production, pin the connection to the validated IP or run the fetcher in an egress-restricted network.
- No rate limiting or auth is included — add both before exposing publicly, since each request spends LLM money.

## Sample request / response

Request:

```json
{"url": "https://example.com"}
```

Response shape (trimmed). The full file is in [`docs/sample_response.json`](docs/sample_response.json).

> This is a real response from a live run (provider: Gemini, model: gemini-3.5-flash-lite) for https://www.zoho.com.

```json
{
  "success": true,
  "website": "https://example.com/",
  "brand": {"name": "...", "category": "...", "summary": "...", "main_products_services": ["..."]},
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "...",
      "brand_profile": {"main_service": "...", "target_market": "...", "key_strength": "...", "opportunity": "..."},
      "competitors": [{"name": "...", "website": "domain.com", "main_service": "...", "target_market": "...",
                       "strengths": ["..."], "weaknesses_or_gaps": ["..."], "source": "ai_suggested", "verified": false}],
      "comparison_table": [{"comparison": "Main Service", "your_brand": "...", "competitors": [{"name": "...", "value": "..."}]}],
      "strengths": ["..."], "opportunities": ["..."], "note": "..."
    },
    "target_search_queries": [{"query": "...", "platforms": ["Google", "LinkedIn"], "intent": "...", "business_relevance": "...", "target_audience": "..."}],
    "seo_improvements": {"primary_keyword": "...", "secondary_keywords": [], "long_tail_keywords": [], "business_keywords": [],
                         "recommended_title": "...", "title_structure": "...", "meta_description": "...",
                         "supporting_content": [{"type": "FAQ", "title": "...", "rationale": "..."}]},
    "top_services": [{"name": "...", "description": "...", "target_industry": "...", "target_customer": "...", "main_benefit": "...", "source": "website"}],
    "target_audience": {"primary_audience": [], "secondary_audience": [], "industries": [], "business_types": [], "customer_profiles": [],
                        "geographic_focus": "...", "customer_needs": [], "customer_challenges": [], "customer_interests": [], "recommended_segments": []}
  },
  "meta": {
    "llm": {"provider": "anthropic", "model": "claude-haiku-5-5", "input_tokens": 5400, "output_tokens": 2400,
            "latency_ms": 0, "estimated_cost_usd": 0.0017, "used_fallback": false},
    "cached": false, "pages_analyzed": ["https://example.com/"], "content_chars_sent": 11800,
    "total_time_ms": 0, "warnings": [], "disclaimer": "..."
  }
}
```

Error example (`{"url": "http://169.254.169.254/latest/meta-data"}` → HTTP 400):

```json
{"success": false, "error": {"code": "blocked_url", "message": "This address points to a private or internal network and cannot be analyzed."}}
```

Test run log: [`docs/test_run.txt`](docs/test_run.txt).

---

# LLM platform & model research

> **Pricing snapshot: 21 Nov 2026**, standard tier, USD per 1M tokens, from the providers' pricing pages ([Anthropic](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI](https://developers.openai.com/api/docs/pricing), [Google](https://ai.google.dev/gemini-api/docs/pricing)). Prices change; re-check before committing to a model.
>
> **What is measured vs. estimated:** prices are from the pages above. Everything about speed and accuracy in this section is **not yet measured** — it needs API keys. Run `scripts/benchmark_llms.py` and paste the output into section E. I do not claim any model has "maximum accuracy".

## A. Comparison

| Provider | Model | Input / Output ($/1M) | Cached input | Context window | JSON support | Speed | Accuracy |
|---|---|---|---|---|---|---|---|
| Anthropic | **Claude Haiku 5.5** (`claude-haiku-5-5`) | $0.10 / $0.50 for prompts ≤100K; **$0.50 / $2.50 above 100K** | $0.01 | Not stated on pricing page — verify | Prompt-enforced JSON + our parser/validator/retry | Not measured | Not measured |
| Anthropic | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | $2.00 / $10.00 | $0.10 | 1M (standard pricing) | same | Not measured | Not measured |
| OpenAI | `gpt-6-luna` (cheapest listed tier) | $0.05 / $0.25 | $0.005 | Not shown on pricing page — verify | `response_format: json_object` | Not measured | Not measured |
| Google | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | $0.30 / $2.50 | $0.03 (+ storage) | Not listed on pricing page — verify | `responseMimeType: application/json` | Not measured | Not measured |
| Google | Gemini 3.6 Flash (`gemini-3.6-flash`) | $0.75 / $3.75 until 31 Dec 2026, then $1.50 / $7.50 | $0.075 | Not listed — verify | same | Not measured | Not measured |

Notes: Anthropic models from 4.7 onward use a tokenizer that produces ~30% more tokens for the same text, so compare *cost per analysis*, not just per-token price. The OpenAI page I could read did not list "mini/nano" models; `gpt-6-luna` is the cheapest tier it showed — confirm that it suits this task and that the ID is right before use. The default `LLM_MODEL` names live in `app/config.py`.

## B. Selection criteria → current pick

1. **Cost:** a single analysis sends ≈5–6K input tokens (≤12k chars of page text + ≈1.5K of prompt/schema) and returns ≈2–2.5K tokens. Output tokens dominate the bill (5× the input price), so the output price matters most.
2. **Large pages:** the digest is capped at 12k chars, so even a 100K-token context is far more than needed; Haiku 5.5's >100K surcharge **never triggers** here.
3. **Structured output:** all three providers can return JSON; the service additionally strips fences, validates with Pydantic, and retries once with the validation error, so occasional format slips are recoverable.
4. **Insight quality** (competitors, SEO) benefits from a larger model's world knowledge — this is exactly what the benchmark should test before committing.

**Current recommendation (to be confirmed by benchmark): Claude Haiku 5.5 as primary, Claude Sonnet 5.5 as fallback.**
Reasons: lowest-tier Anthropic price with the long-prompt surcharge irrelevant for our prompt size; same provider/API shape for primary and fallback (one key, one adapter); Sonnet gives a quality step-up when Haiku fails or returns unusable output. `gpt-6-luna` is the cheaper challenger on paper (~half the cost) and Gemini 3.5 Flash-Lite is a viable third provider; neither has been tested here, so neither is ruled out.

## C. Cost optimisation (implemented unless noted)

- ✅ HTML → text: scripts, styles, SVG, nav, footer, forms and aside removed; duplicate lines dropped (`website_extractor.py`).
- ✅ Hard cap of `MAX_CONTENT_CHARS` (12k) with high-signal metadata first (title, meta description, nav, headings, JSON-LD), then body text.
- ✅ Only 2 extra same-site pages (services/about) — enough grounding for the "top services" section without paying for the whole site.
- ✅ One call for all five sections instead of five calls that would each repeat the page content.
- ✅ Compact prompt, concise-string instruction and `LLM_MAX_OUTPUT_TOKENS` ceiling to bound output cost.
- ✅ Cache keyed by `sha256(provider|model|digest)` (24h TTL): repeated requests for unchanged content cost $0 (`meta.cached: true`).
- ✅ Model tiering: cheap model first, bigger model only on failure.
- 🔜 Production ideas: Anthropic prompt caching for the static system prompt + schema (cache reads are 0.1× input price); Batch API (50% off) for non-interactive bulk analysis; Redis cache shared across workers; per-IP rate limiting.

## D. Final recommendation (estimates, assumptions stated)

| Item | Value |
|---|---|
| Primary | Anthropic **Claude Haiku 5.5** |
| Fallback | Anthropic **Claude Sonnet 5.5** (then optionally a second provider, e.g. Gemini 3.5 Flash-Lite, for provider-outage resilience) |
| Input tokens / analysis | **≈5,500** *(estimate: 12k chars ≈ 3–3.9K tokens + ≈1.5K prompt/schema)* |
| Output tokens / analysis | **≈2,500** *(estimate from the response schema size)* |
| Cost / analysis | Haiku 5.5 **≈ $0.0018** (≈ $1.80 per 1,000) · Sonnet 5.5 ≈ $0.036 · gpt-6-luna ≈ $0.0009 · Gemini 3.5 Flash-Lite ≈ $0.0079 · Gemini 3.6 Flash ≈ $0.0135 *(computed: tokens × list price; replace with the measured `meta.llm.estimated_cost_usd`)* |
| Response time | **Not measured.** Output generation (~2.5K tokens) dominates; fetching adds roughly 1–3 s. Measure with the benchmark script. |
| Balancing accuracy / speed / cost | Run cheap-first; escalate to the larger model on parse failure; if benchmark shows Haiku's competitor/SEO quality is weak, move only the competitor + SEO reasoning to Sonnet and keep extraction on Haiku. |

## E. Measured benchmark (fill in after running)

```bash
python scripts/benchmark_llms.py https://site-one.com https://site-two.com https://site-three.com \
  --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5 gemini:gemini-3.5-flash-lite openai:gpt-6-luna --runs 2
```

It fetches each site once, runs every model on the identical digest, and writes `docs/benchmark_results.json` plus a Markdown table (latency, tokens, cost, structural completeness). Completeness only checks that the sections are filled (5 queries, 5 services, ≥3 competitors, SEO fields…); **judge factual accuracy by reading a few outputs yourself** and note the result here.

<!-- PASTE MEASURED TABLE HERE -->

## Limitations

- JavaScript-rendered sites (SPAs) and bot-protected sites often return no readable text; the API reports `insufficient_content` / `website_unreachable` rather than guessing. A headless-browser fallback would be the next step.
- Competitor lists reflect the model's general knowledge and can be outdated or wrong — they are explicitly flagged as unverified.
- The in-memory cache is per process and lost on restart.
