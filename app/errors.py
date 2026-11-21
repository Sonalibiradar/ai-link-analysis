"""Application errors. Each carries an HTTP status and a stable machine-readable code."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidURLError(AppError):
    status_code = 400
    code = "invalid_url"


class BlockedURLError(AppError):
    """URL points to a private/internal address (SSRF protection)."""

    status_code = 400
    code = "blocked_url"


class FetchError(AppError):
    """The website could not be fetched (down, blocked, timeout, not HTML ...)."""

    status_code = 502
    code = "website_unreachable"


class InsufficientContentError(AppError):
    status_code = 422
    code = "insufficient_content"


class LLMConfigError(AppError):
    status_code = 503
    code = "llm_not_configured"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMTimeoutError(AppError):
    status_code = 504
    code = "llm_timeout"


class LLMOutputError(AppError):
    status_code = 502
    code = "llm_invalid_output"
