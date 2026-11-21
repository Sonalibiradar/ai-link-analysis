"""Configuration loaded from environment variables (optionally from a local .env file)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-6-luna",
    "gemini": "gemini-3.5-flash-lite",
}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    # Fallback used when the primary model fails (provider errors / unusable output).
    fallback_provider: str
    fallback_model: str
    anthropic_api_key: str
    openai_api_key: str
    gemini_api_key: str
    llm_timeout_s: float
    llm_max_output_tokens: int
    fetch_connect_timeout_s: float
    fetch_read_timeout_s: float
    fetch_max_bytes: int
    fetch_max_redirects: int
    extra_pages: int
    max_content_chars: int
    cache_ttl_s: int
    cache_max_items: int
    user_agent: str

    def api_key_for(self, provider: str) -> str:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
        }.get(provider, "")


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    fb_provider = os.getenv("FALLBACK_LLM_PROVIDER", "anthropic").strip().lower()
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(provider, ""),
        fallback_provider=fb_provider,
        fallback_model=os.getenv("FALLBACK_LLM_MODEL", "").strip()
        or ("claude-sonnet-5-5" if fb_provider == "anthropic" else DEFAULT_MODELS.get(fb_provider, "")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        llm_timeout_s=_float("LLM_TIMEOUT_SECONDS", 60),
        llm_max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 4096),
        fetch_connect_timeout_s=_float("FETCH_CONNECT_TIMEOUT_SECONDS", 5),
        fetch_read_timeout_s=_float("FETCH_READ_TIMEOUT_SECONDS", 10),
        fetch_max_bytes=_int("FETCH_MAX_BYTES", 2_000_000),
        fetch_max_redirects=_int("FETCH_MAX_REDIRECTS", 5),
        extra_pages=_int("EXTRA_PAGES", 2),
        max_content_chars=_int("MAX_CONTENT_CHARS", 12_000),
        cache_ttl_s=_int("CACHE_TTL_SECONDS", 86_400),
        cache_max_items=_int("CACHE_MAX_ITEMS", 256),
        user_agent=os.getenv(
            "FETCH_USER_AGENT",
            "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://example.com/bot)",
        ),
    )
