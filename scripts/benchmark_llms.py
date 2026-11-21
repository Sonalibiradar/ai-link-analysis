"""Measure latency, token usage, cost and output quality for several LLMs on the same websites.

Usage (needs the relevant API keys in your environment / .env):

    python scripts/benchmark_llms.py https://stripe.com https://www.zoho.com \
        --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5 gemini:gemini-3.5-flash-lite openai:gpt-6-luna

For every (website, model) it records: wall-clock latency, input/output tokens, estimated cost,
whether the JSON validated first time, and a simple completeness score (does the output contain the
5 queries / 5 services / >=3 competitors / SEO title+meta / audience). Results go to docs/benchmark_results.json
and a Markdown table is printed so you can paste *measured* numbers into the README.

Completeness is a structural check only. It does NOT judge factual accuracy -- review a few outputs by hand.
"""

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.services.analysis_service import AnalysisService  # noqa: E402
from app.services.website_extractor import extract_website  # noqa: E402
from app.utils.cache import TTLCache  # noqa: E402
import requests  # noqa: E402


def completeness(resp) -> float:
    a = resp.analysis
    checks = [
        bool(resp.brand.name and resp.brand.category and resp.brand.summary),
        len(a.target_search_queries) == 5,
        len(a.top_services) == 5,
        len(a.competitor_analysis.competitors) >= 3,
        bool(a.seo_improvements.recommended_title and a.seo_improvements.meta_description),
        len(a.seo_improvements.secondary_keywords) >= 3 and len(a.seo_improvements.long_tail_keywords) >= 3,
        len(a.seo_improvements.supporting_content) >= 3,
        bool(a.target_audience.primary_audience and a.target_audience.customer_needs),
    ]
    return sum(checks) / len(checks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--models", nargs="+", required=True, help="provider:model pairs")
    ap.add_argument("--runs", type=int, default=1, help="repeats per (url, model) to average latency")
    args = ap.parse_args()

    base = get_settings()
    rows = []
    for url in args.urls:
        digest = extract_website(url, base, requests.Session())  # fetch once, reuse for every model
        for spec in args.models:
            provider, model = spec.split(":", 1)
            settings = dataclasses.replace(base, llm_provider=provider, llm_model=model,
                                           fallback_provider=provider, fallback_model=model, cache_ttl_s=0)
            svc = AnalysisService(settings, extractor=lambda u, s, d=digest: d, cache=TTLCache(0, 1))
            for run in range(args.runs):
                t0 = time.perf_counter()
                row = {"url": url, "model": spec, "run": run + 1}
                try:
                    resp = svc.analyze(url)
                    llm = resp.meta.llm
                    row.update(ok=True, latency_s=round(time.perf_counter() - t0, 2), llm_latency_ms=llm.latency_ms,
                               input_tokens=llm.input_tokens, output_tokens=llm.output_tokens,
                               cost_usd=llm.estimated_cost_usd, completeness=round(completeness(resp), 2),
                               warnings=resp.meta.warnings)
                except AppError as e:
                    row.update(ok=False, error=f"{e.code}: {e.message}")
                rows.append(row)
                print(json.dumps(row))

    out = Path(__file__).resolve().parent.parent / "docs" / "benchmark_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    print("\n| Model | Runs ok | Avg latency (s) | Avg in tok | Avg out tok | Avg cost (USD) | Avg completeness |")
    print("|---|---|---|---|---|---|---|")
    for spec in args.models:
        ok = [r for r in rows if r["model"] == spec and r.get("ok")]
        total = len([r for r in rows if r["model"] == spec])
        if not ok:
            print(f"| {spec} | 0/{total} | - | - | - | - | - |")
            continue
        avg = lambda k: statistics.mean(r[k] for r in ok if r.get(k) is not None) if any(r.get(k) is not None for r in ok) else float("nan")
        print(f"| {spec} | {len(ok)}/{total} | {avg('latency_s'):.1f} | {avg('input_tokens'):.0f} | "
              f"{avg('output_tokens'):.0f} | {avg('cost_usd'):.4f} | {avg('completeness'):.0%} |")


if __name__ == "__main__":
    main()
