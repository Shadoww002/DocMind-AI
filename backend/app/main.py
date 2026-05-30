import os
from contextlib import asynccontextmanager
import logging

# Force ChromaDB telemetry off globally before initialization dependencies trigger
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.endpoints import router
from backend.app.core.config import DomainConfig

# Setup enterprise-level core system logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("backend.core")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle sequences.
    Verifies that system environment states and storage folders are ready.
    """
    logger.info("Initializing Multi-Domain Document Intelligence Engine...")
    
    # Verify local persistence database directories are accessible
    try:
        os.makedirs(DomainConfig.VECTOR_DB_DIR, exist_ok=True)
        logger.info(f"Vector storage verified at destination path: {DomainConfig.VECTOR_DB_DIR}")
    except Exception as e:
        logger.critical(f"Failed to establish persistent storage volumes: {str(e)}")
        
    yield  # Application runtime happens here
    
    logger.info("Shutting down Multi-Domain Document Intelligence Engine... Releasing pipeline locks.")


# Core FastAPI Application Instance Definition
app = FastAPI(
    title="Multi-Domain Document Intelligence Enterprise Platform",
    description="Production-Grade Modular Architecture for Medical, Legal, and Resume extraction pipelines.",
    version="2.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------
# Production CORS Configuration Core
# -----------------------------------------------------------------
# Dynamically pulls allowed origins from environment variables, defaulting to wildcard for local dev
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permits standard GET, POST, OPTIONS, and DELETE verbs
    allow_headers=["*"],  # Safeguards custom authentication tokens and file headers
)

# -----------------------------------------------------------------
# Global Exception Catch-All Middleware
# -----------------------------------------------------------------
@app.middleware("http")
async def global_exception_handler(request: Request, call_next):
    """
    Acts as a final defensive line. Intercepts structural processing anomalies 
    and converts them into structured JSON error arrays.
    """
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Unhandled app exception captured at server edge: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "fail",
                "detail": "An unexpected error occurred within the host pipeline processing engine."
            }
        )

# Register high-performance routing matrices
app.include_router(router)


@app.get("/", tags=["Health"])
def system_root():
    """System health monitoring checkpoint endpoint."""
    return {
        "status": "operational",
        "system": "Multi-Domain Intelligence Engine",
        "storage_path": DomainConfig.VECTOR_DB_DIR
    }