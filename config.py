"""
config.py — Central settings object (Pydantic-Settings)

WHY THIS FILE EXISTS:
    Every module in EvalForge needs configuration (DB URL, Ollama host, JWT
    secret, etc.).  Rather than scattering os.getenv() calls everywhere, we
    use a single Pydantic Settings class.

    Pydantic-Settings reads from (in priority order):
      1. Real environment variables
      2. The .env file
      3. Declared defaults

    Any module that needs a setting does:
        from config import settings
        url = settings.DATABASE_URL

    This makes the code testable — tests can monkeypatch `settings` rather
    than juggling real env vars.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://evalforge:evalforge@localhost:5432/evalforge"

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── Ollama ────────────────────────────────────────────────────────────
    # OLLAMA_BASE_URL is passed to ChatOllama(base_url=...) and to the raw
    # ollama Python client.  All LLM calls in this project hit this address —
    # no OpenAI / Anthropic keys required.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma4:26b"

    # ── JWT ───────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = "change-me-before-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440

    # ── LangSmith ─────────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "evalforge"

    # ── HITL ──────────────────────────────────────────────────────────────
    HITL_HALLUCINATION_THRESHOLD: float = 0.3

    # ── Scoring ───────────────────────────────────────────────────────────
    # When LoRA classifier confidence exceeds this, we skip the expensive
    # Ollama LLM-as-Judge call and trust the classifier alone.
    LORA_CONFIDENCE_THRESHOLD: float = 0.85


# Module-level singleton — import this everywhere
settings = Settings()
