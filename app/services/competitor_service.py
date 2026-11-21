"""Post-processing for the competitor section.

Competitors come from the LLM's general knowledge plus the site content -- there is no live
market-research API behind them -- so they are always labelled `ai_suggested` / `verified: false`.
This module cleans the list (dedupe, drop the brand itself, cap length) and builds the
"Your Brand vs Competitor" comparison table.
"""

import re
from urllib.parse import urlsplit

from app.schemas import BrandProfile, Competitor, ComparisonCell, ComparisonRow

MIN_COMPETITORS = 3
MAX_COMPETITORS = 5


def _domain(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    host = urlsplit(value).hostname or ""
    return host.removeprefix("www.")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def normalize_competitors(
    competitors: list[Competitor], brand_name: str, brand_url: str
) -> tuple[list[Competitor], list[str]]:
    own_domain = _domain(brand_url)
    own_slug = _slug(brand_name)
    cleaned: list[Competitor] = []
    seen: set[str] = set()
    for c in competitors:
        slug = _slug(c.name)
        dom = _domain(c.website)
        if not slug or slug in seen:
            continue
        if (own_slug and slug == own_slug) or (own_domain and dom == own_domain):
            continue  # the model listed the brand itself
        seen.add(slug)
        c.website = dom  # keep only the domain; we never claim it was visited
        c.source = "ai_suggested"
        c.verified = False
        cleaned.append(c)
        if len(cleaned) >= MAX_COMPETITORS:
            break

    warnings = []
    if len(cleaned) < MIN_COMPETITORS:
        warnings.append(
            f"Only {len(cleaned)} competitor(s) could be identified with reasonable confidence "
            f"(target: at least {MIN_COMPETITORS})."
        )
    return cleaned, warnings


def build_comparison_table(
    brand_name: str, profile: BrandProfile, competitors: list[Competitor]
) -> list[ComparisonRow]:
    if not competitors:
        return []

    def row(label: str, own: str, getter) -> ComparisonRow:
        return ComparisonRow(
            comparison=label,
            your_brand=own or "-",
            competitors=[ComparisonCell(name=c.name, value=getter(c) or "-") for c in competitors],
        )

    return [
        row("Brand Name", brand_name, lambda c: c.name),
        row("Main Service", profile.main_service, lambda c: c.main_service),
        row("Target Market", profile.target_market, lambda c: c.target_market),
        row("Key Strength", profile.key_strength, lambda c: c.strengths[0] if c.strengths else ""),
        row(
            "Opportunity",
            profile.opportunity,
            lambda c: c.weaknesses_or_gaps[0] if c.weaknesses_or_gaps else "",
        ),
    ]
