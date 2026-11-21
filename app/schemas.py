"""Pydantic models for the request, the LLM output contract, and the API response.

The LLM output is validated with *lenient* coercion (None -> "", str -> [str], dict -> "k: v")
so a slightly sloppy model response does not fail the request, while the shape that
reaches the client is always strict and predictable.
"""

import json
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, Field


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        return ", ".join(_to_str(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_to_str(val)}" for k, val in v.items())
    return str(v)


def _to_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (str, int, float)):
        s = _to_str(v)
        return [s] if s else []
    if isinstance(v, dict):
        v = [v]
    if isinstance(v, (list, tuple)):
        return [s for s in (_to_str(x) for x in v) if s]
    return [_to_str(v)]


Text = Annotated[str, BeforeValidator(_to_str)]
TextList = Annotated[list[str], BeforeValidator(_to_list)]


def _obj(v: Any) -> Any:
    return {} if v is None else v


def _lst(v: Any) -> Any:
    if v is None:
        return []
    if isinstance(v, dict):
        return [v]
    return v


# ----------------------------------------------------------------------------- request


class LinkAnalysisRequest(BaseModel):
    url: str = Field(..., description="Website URL to analyze", examples=["https://example.com"])


# ----------------------------------------------------------------------------- sections


class Brand(BaseModel):
    name: Text = ""
    category: Text = ""
    summary: Text = ""
    main_products_services: TextList = []


class BrandProfile(BaseModel):
    main_service: Text = ""
    target_market: Text = ""
    key_strength: Text = ""
    opportunity: Text = ""


class Competitor(BaseModel):
    name: Text = ""
    website: Text = ""
    main_service: Text = ""
    target_market: Text = ""
    strengths: TextList = []
    weaknesses_or_gaps: TextList = []
    source: str = "ai_suggested"  # competitors come from the LLM, not from live research
    verified: bool = False


class ComparisonCell(BaseModel):
    name: str
    value: str


class ComparisonRow(BaseModel):
    comparison: str
    your_brand: str
    competitors: list[ComparisonCell]


class CompetitorAnalysis(BaseModel):
    brand_overview: Text = ""
    brand_profile: Annotated[BrandProfile, BeforeValidator(_obj)] = BrandProfile()
    competitors: Annotated[list[Competitor], BeforeValidator(_lst)] = []
    comparison_table: list[ComparisonRow] = []  # built server-side from the fields above
    strengths: TextList = []  # your brand's competitive advantages
    opportunities: TextList = []  # opportunities to improve market positioning
    note: str = ""


class SearchQuery(BaseModel):
    query: Text = ""
    platforms: TextList = []
    intent: Text = ""
    business_relevance: Text = ""
    target_audience: Text = ""


class SupportContent(BaseModel):
    type: Text = ""  # e.g. "Service landing page", "FAQ", "Blog topic"
    title: Text = ""
    rationale: Text = ""


class SeoImprovements(BaseModel):
    primary_keyword: Text = ""
    secondary_keywords: TextList = []
    long_tail_keywords: TextList = []
    business_keywords: TextList = []
    recommended_title: Text = ""
    title_structure: Text = ""
    meta_description: Text = ""
    supporting_content: Annotated[list[SupportContent], BeforeValidator(_lst)] = []


class Service(BaseModel):
    name: Text = ""
    description: Text = ""
    target_industry: Text = ""
    target_customer: Text = ""
    main_benefit: Text = ""
    source: Text = "website"  # "website" (found on the site) or "inferred"


class TargetAudience(BaseModel):
    primary_audience: TextList = []
    secondary_audience: TextList = []
    industries: TextList = []
    business_types: TextList = []
    customer_profiles: TextList = []
    geographic_focus: Text = ""
    customer_needs: TextList = []
    customer_challenges: TextList = []
    customer_interests: TextList = []
    recommended_segments: TextList = []


class AnalysisSections(BaseModel):
    competitor_analysis: Annotated[CompetitorAnalysis, BeforeValidator(_obj)] = CompetitorAnalysis()
    target_search_queries: Annotated[list[SearchQuery], BeforeValidator(_lst)] = []
    seo_improvements: Annotated[SeoImprovements, BeforeValidator(_obj)] = SeoImprovements()
    top_services: Annotated[list[Service], BeforeValidator(_lst)] = []
    target_audience: Annotated[TargetAudience, BeforeValidator(_obj)] = TargetAudience()


class LLMOutput(BaseModel):
    """What we require the LLM to return."""

    brand: Annotated[Brand, BeforeValidator(_obj)] = Brand()
    analysis: Annotated[AnalysisSections, BeforeValidator(_obj)] = AnalysisSections()


# ----------------------------------------------------------------------------- response


class LLMMeta(BaseModel):
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[int] = None  # measured wall-clock time of the LLM call(s)
    estimated_cost_usd: Optional[float] = None  # computed from reported token usage x list price
    used_fallback: bool = False


class ResponseMeta(BaseModel):
    llm: LLMMeta
    cached: bool = False
    pages_analyzed: list[str] = []
    content_chars_sent: int = 0
    total_time_ms: int = 0
    warnings: list[str] = []
    disclaimer: str = (
        "Competitors and search queries are AI-generated suggestions based on the website content; "
        "no search-volume or ranking data is included because none was sourced from a data provider."
    )


class LinkAnalysisResponse(BaseModel):
    success: bool = True
    website: str
    brand: Brand
    analysis: AnalysisSections
    meta: ResponseMeta


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody


def dump_json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(), ensure_ascii=False)
