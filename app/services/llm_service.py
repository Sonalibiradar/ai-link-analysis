"""Provider-agnostic LLM client (Anthropic / OpenAI / Gemini) using plain REST calls.

No vendor SDKs are required: one small `requests`-based adapter per provider keeps the
dependency footprint low and makes the providers easy to swap or mock in tests.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from app.config import Settings
from app.errors import LLMConfigError, LLMError, LLMOutputError, LLMTimeoutError

# List prices in USD per 1M tokens (input, output), standard tier, prompts <=100K tokens.
# Snapshot taken from the providers' pricing pages on 2026-11-21 -- verify before relying on them.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-6-luna": (0.05, 0.25),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
}

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: int


def estimate_cost_usd(model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    price = PRICING_USD_PER_MTOK.get(model)
    if not price or input_tokens is None or output_tokens is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 6)


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating ```json fences and stray prose."""
    if not text or not text.strip():
        raise LLMOutputError("The model returned an empty response.")
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S | re.I)
    if fenced:
        cleaned = fenced.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("The model response did not contain a JSON object.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            raise LLMOutputError(f"The model returned malformed JSON ({e.msg}).")
    if not isinstance(data, dict):
        raise LLMOutputError("The model returned JSON that is not an object.")
    return data


class LLMClient:
    """One provider + model. Call `complete()` to get raw text plus usage."""

    def __init__(self, provider: str, model: str, settings: Settings, session: Optional[requests.Session] = None):
        self.provider = provider.lower()
        self.model = model
        self.settings = settings
        self.session = session or requests.Session()
        if self.provider not in ("anthropic", "openai", "gemini"):
            raise LLMConfigError(f"Unsupported LLM provider '{provider}'. Use anthropic, openai or gemini.")
        if not model:
            raise LLMConfigError(f"No model configured for provider '{provider}'.")
        self.api_key = settings.api_key_for(self.provider)
        if not self.api_key:
            raise LLMConfigError(
                f"No API key configured for provider '{self.provider}'. "
                f"Set {self.provider.upper()}_API_KEY in the environment."
            )

    # -- request builders ---------------------------------------------------------------------

    def _build_request(self, system: str, user: str) -> tuple[str, dict, dict]:
        max_out = self.settings.llm_max_output_tokens
        if self.provider == "anthropic":
            return (
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                {
                    "model": self.model,
                    "max_tokens": max_out,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
        if self.provider == "openai":
            return (
                "https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
                {
                    "model": self.model,
                    "max_completion_tokens": max_out,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
            )
        # gemini: key goes in a header (not the URL) so it is never written to access logs
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.api_key, "content-type": "application/json"},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "maxOutputTokens": max_out,
                    "temperature": 0.2,
                },
            },
        )

    # -- response parsers ---------------------------------------------------------------------

    def _parse_response(self, data: dict) -> tuple[str, Optional[int], Optional[int]]:
        try:
            if self.provider == "anthropic":
                if data.get("stop_reason") == "max_tokens":
                    raise LLMOutputError("The model response was cut off (output token limit reached).")
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                usage = data.get("usage", {})
                return text, usage.get("input_tokens"), usage.get("output_tokens")
            if self.provider == "openai":
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise LLMOutputError("The model response was cut off (output token limit reached).")
                usage = data.get("usage", {})
                return choice["message"]["content"] or "", usage.get("prompt_tokens"), usage.get("completion_tokens")
            cand = data["candidates"][0]
            if cand.get("finishReason") == "MAX_TOKENS":
                raise LLMOutputError("The model response was cut off (output token limit reached).")
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            usage = data.get("usageMetadata", {})
            return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
        except (KeyError, IndexError, TypeError):
            raise LLMOutputError("Unexpected response structure from the LLM provider.")

    # -- public -------------------------------------------------------------------------------

    def complete(self, system: str, user: str, attempts: int = 2) -> LLMResult:
        url, headers, body = self._build_request(system, user)
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(attempts):
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=self.settings.llm_timeout_s)
            except requests.exceptions.Timeout as e:
                last_error = LLMTimeoutError("The LLM provider timed out.")
                last_error.__cause__ = e
            except requests.exceptions.RequestException as e:
                last_error = LLMError("Could not reach the LLM provider.")
                last_error.__cause__ = e
            else:
                if resp.status_code == 200:
                    try:
                        payload = resp.json()
                    except ValueError:
                        raise LLMOutputError("The LLM provider returned a non-JSON response.")
                    text, tin, tout = self._parse_response(payload)
                    return LLMResult(
                        text=text,
                        provider=self.provider,
                        model=self.model,
                        input_tokens=tin,
                        output_tokens=tout,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                if resp.status_code in (401, 403):
                    raise LLMConfigError(f"The {self.provider} API rejected the API key (HTTP {resp.status_code}).")
                if resp.status_code == 404:
                    raise LLMConfigError(f"Model '{self.model}' was not found on {self.provider} (HTTP 404).")
                if resp.status_code == 400:
                    raise LLMError(f"The {self.provider} API rejected the request (HTTP 400): {_short(resp)}")
                last_error = LLMError(f"The {self.provider} API returned HTTP {resp.status_code}.")
                if resp.status_code not in RETRYABLE_STATUS:
                    raise last_error
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error


def _short(resp: requests.Response) -> str:
    try:
        return json.dumps(resp.json())[:300]
    except ValueError:
        return (resp.text or "")[:300]
