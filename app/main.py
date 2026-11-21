"""FastAPI entry point.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 4000
Docs: http://localhost:4000/docs
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app.errors import AppError
from app.schemas import ErrorBody, ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("link-analysis")

app = FastAPI(
    title="AI Website Link Analysis API",
    version="1.0.0",
    description=(
        "Analyzes a website with an LLM and returns competitor, search-query, SEO, services "
        "and target-audience insights as structured JSON."
    ),
)

_service: AnalysisService | None = None


def get_service() -> AnalysisService:
    """Lazy singleton so the in-memory cache is shared across requests."""
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service


def _error(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s: %s", exc.code, exc.message)
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "invalid_request", 'Request body must be JSON like {"url": "https://example.com"}.')


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return _error(500, "internal_error", "An unexpected error occurred.")

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or blocked URL"},
        422: {"model": ErrorResponse, "description": "Bad request body or not enough page content"},
        502: {"model": ErrorResponse, "description": "Website unreachable or LLM failure"},
        503: {"model": ErrorResponse, "description": "LLM API key not configured"},
        504: {"model": ErrorResponse, "description": "LLM timeout"},
    },
    tags=["analysis"],
    summary="Analyze a website",
)
def link_analysis(payload: LinkAnalysisRequest) -> LinkAnalysisResponse:
    # Plain `def` on purpose: the work is blocking I/O (requests), so FastAPI runs it in its threadpool.
    return get_service().analyze(payload.url)
