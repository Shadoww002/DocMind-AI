import os
import sys
import time
import logging
import logging.config
from contextlib import asynccontextmanager

# Must be set before any chromadb import is triggered anywhere in the tree
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.endpoints import router
from backend.app.core.config import DomainConfig


# ── Logging setup ─────────────────────────────────────────────────────────────
# IMPROVEMENT: use dictConfig so every module in the project inherits the same
# format without each file calling basicConfig (which is a no-op if called twice).
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    # Silence noisy third-party loggers that spam on CPU
    "loggers": {
        "httpx":          {"level": "WARNING", "propagate": True},
        "httpcore":       {"level": "WARNING", "propagate": True},
        "transformers":   {"level": "WARNING", "propagate": True},
        "chromadb":       {"level": "WARNING", "propagate": True},
        "urllib3":        {"level": "WARNING", "propagate": True},
    },
})

logger = logging.getLogger("backend.main")


# ── Startup validation helpers ────────────────────────────────────────────────

def _ensure_storage_dir() -> bool:
    """
    Creates the ChromaDB persistence directory if it does not exist.
    Returns True on success, False on failure.
    """
    try:
        os.makedirs(DomainConfig.VECTOR_DB_DIR, exist_ok=True)
        # IMPROVEMENT: verify we can actually write — not just that the dir exists
        probe = os.path.join(DomainConfig.VECTOR_DB_DIR, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        logger.info(f"[Startup] Storage directory ready: {DomainConfig.VECTOR_DB_DIR}")
        return True
    except OSError as e:
        logger.critical(f"[Startup] Storage directory not writable: {e}")
        return False


def _check_required_env_vars():
    """
    IMPROVEMENT: fail fast at startup if critical env vars are missing.
    Add any API keys or secrets your project needs to REQUIRED_VARS.
    """
    REQUIRED_VARS = [
        # e.g. "OPENAI_API_KEY", "HF_TOKEN"
        # Leave empty for now; add vars as your project grows.
    ]
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        logger.critical(f"[Startup] Missing required environment variables: {missing}")
        sys.exit(1)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("[Startup] DocMind AI — Multi-Domain Intelligence Engine")
    logger.info("=" * 60)

    _check_required_env_vars()

    storage_ok = _ensure_storage_dir()
    if not storage_ok:
        # IMPROVEMENT: abort startup instead of silently continuing with a
        # broken storage path — every subsequent request would fail anyway.
        logger.critical("[Startup] Cannot start: storage directory is not writable. Exiting.")
        sys.exit(1)

    logger.info(f"[Startup] Domains loaded: {list(DomainConfig.DOMAINS.keys())}")
    logger.info("[Startup] Ready — accepting requests.")

    yield  # ── application runs here ──

    logger.info("[Shutdown] Releasing pipeline resources.")


# ── App instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocMind AI — Multi-Domain Document Intelligence",
    description=(
        "Domain-aware document intelligence for Medical, Indian Legal, and Resume PDFs. "
        "Provides summarization, entity extraction, and ChromaDB-backed RAG Q&A."
    ),
    version="2.1.0",
    lifespan=lifespan,
    # IMPROVEMENT: disable the default /docs and /redoc in production by
    # reading an env var — safe to leave enabled locally.
    docs_url="/docs" if os.getenv("ENV", "development") != "production" else None,
    redoc_url=None,
)


# ── CORS ──────────────────────────────────────────────────────────────────────

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")

# IMPROVEMENT: wildcard "*" and allow_credentials=True is an invalid
# combination — browsers reject it. If no origins are set, default to
# localhost only. Use "*" only when credentials are not needed.
_allow_credentials = True
if _raw_origins.strip() == "*":
    _allow_credentials = False   # credentials must be False with wildcard
    _origins_list = ["*"]
else:
    _origins_list = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST"],   # IMPROVEMENT: explicit verbs only — not ["*"]
    allow_headers=["Content-Type", "Authorization"],
)


# ── Global exception middleware ───────────────────────────────────────────────

@app.middleware("http")
async def global_exception_middleware(request: Request, call_next):
    """
    Last-resort exception handler.

    IMPROVEMENT: also measures and logs request duration so slow endpoints
    are visible in logs — useful when profiling CPU inference times.
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = round((time.perf_counter() - start) * 1000)
        # Only log non-health-check routes to avoid log spam
        if request.url.path not in ("/", "/health"):
            logger.info(f"[HTTP] {request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
        return response
    except Exception as exc:
        elapsed = round((time.perf_counter() - start) * 1000)
        logger.error(
            f"[HTTP] Unhandled exception on {request.method} {request.url.path} "
            f"after {elapsed}ms: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "detail": "Internal server error."},
        )


# ── Router ────────────────────────────────────────────────────────────────────

app.include_router(router)


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], include_in_schema=False)
def root_redirect():
    """Redirect-friendly root — returns minimal JSON so curl works cleanly."""
    return {"status": "ok", "docs": "/docs"}


@app.get("/health", tags=["Health"])
def health_check():
    """
    Dedicated health route with storage and collection stats.
    Streamlit polls this before submitting a document.
    """
    from backend.app.api.endpoints import _vdb_manager
    storage_accessible = os.path.isdir(DomainConfig.VECTOR_DB_DIR)

    # FIX 5: wire collection_stats() so /health shows per-domain chunk counts
    try:
        collection_counts = _vdb_manager.collection_stats()
    except Exception:
        collection_counts = {}

    return {
        "status": "healthy" if storage_accessible else "degraded",
        "version": app.version,
        "domains": list(DomainConfig.DOMAINS.keys()),
        "storage_path": DomainConfig.VECTOR_DB_DIR,
        "storage_accessible": storage_accessible,
        "collection_stats": collection_counts,
        "env": os.getenv("ENV", "development"),
    }