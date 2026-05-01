"""
api/main.py — FastAPI Application Entry Point

WHY THIS FILE EXISTS:
    This is the root of the FastAPI app.  It:
    - Creates the FastAPI instance
    - Registers all routers (auth, targets, jobs, websocket)
    - Sets up CORS for browser clients
    - Adds startup/shutdown lifespan events (DB init, health checks)
    - Exposes /health for Docker and load balancer checks

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/main.py ───────────────────────────────────────────────────────────┐
    │  Imports all routers:                                                   │
    │    api/auth.py          → /auth/token                                  │
    │    api/routers/targets  → /api/v1/targets/*                            │
    │    api/routers/jobs     → /api/v1/jobs/*                               │
    │    api/routers/websocket → /ws/jobs/{id}                              │
    │  Run with: uvicorn api.main:app --reload --port 8000                   │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import router as auth_router
from api.routers.jobs import router as jobs_router
from api.routers.targets import router as targets_router
from api.routers.websocket import router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    On startup:
    - Verify Ollama is reachable (warn if not, don't crash)
    - Run any pending Alembic migrations (optional — safe for dev)

    On shutdown:
    - Close DB connection pool cleanly
    """
    logger.info("EvalForge starting up...")

    # Optionally check Ollama is running
    try:
        import httpx
        from config import settings
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama OK — available models: {models}")
            if settings.OLLAMA_MODEL not in models:
                logger.warning(
                    f"Model {settings.OLLAMA_MODEL} not found in Ollama. "
                    f"Run: ollama pull {settings.OLLAMA_MODEL}"
                )
    except Exception as e:
        logger.warning(f"Ollama not reachable at startup: {e}. Run: ollama serve")

    yield  # ← Server runs here

    logger.info("EvalForge shutting down...")


app = FastAPI(
    title="EvalForge",
    description="LLM Evaluation & Reliability Platform — powered by local Ollama",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────────
# Allow browser clients from localhost during development.
# Restrict origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(targets_router)
app.include_router(jobs_router)
app.include_router(ws_router)


@app.get("/health", tags=["meta"])
async def health():
    """
    Health check endpoint.
    Used by Docker Compose healthcheck and load balancers.
    Returns 200 if the API is running.
    """
    return {"status": "ok", "service": "evalforge"}


@app.get("/", tags=["meta"])
async def root():
    return {
        "service": "EvalForge",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
