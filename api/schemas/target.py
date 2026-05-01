"""
api/schemas/target.py — Target system schemas

WHY THIS FILE EXISTS:
    The `Target` is the system EvalForge evaluates (e.g. your RAG Stock
    Analyzer).  These Pydantic models define:
      - What the API *accepts* when registering a target (TargetCreate)
      - What the API *returns* about a target (TargetRead)
      - How the Execution Agent wraps a question into an HTTP request (TargetConfig)

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/schemas/target.py ─────────────────────────────────────────────────┐
    │  TargetConfig  → used by agents/execution_agent.py to call the target  │
    │  TargetCreate  → used by api/routers/targets.py (POST /targets)        │
    │  TargetRead    → used by api/routers/targets.py (GET /targets/{id})    │
    │  Both mirror db/models.py::Target but are decoupled from SQLAlchemy    │
    └─────────────────────────────────────────────────────────────────────────┘

    KEY DESIGN: TargetConfig.request_template
    The target system might expect the question in different shapes:
      {"query": "..."}  or  {"messages": [{"role": "user", "content": "..."}]}
    request_template is a dict where the special token "__QUESTION__" gets
    replaced with the actual test question at runtime.

    Example:
      request_template = {"query": "__QUESTION__", "top_k": 5}
      → sent as {"query": "What is the capital of France?", "top_k": 5}
"""

from typing import Any, Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, HttpUrl, Field


class TargetConfig(BaseModel):
    """
    Black-box HTTP configuration for calling the system under evaluation.

    This is the 'adapter' between EvalForge and any external LLM app.
    As long as the target has an HTTP endpoint that accepts JSON and
    returns JSON, EvalForge can evaluate it.
    """
    endpoint: str = Field(..., description="HTTP endpoint of the target system")
    auth_header: Optional[str] = Field(
        None,
        description="Value for Authorization header, e.g. 'Bearer sk-...'",
    )
    request_template: dict[str, Any] = Field(
        default={"query": "__QUESTION__"},
        description="JSON body sent to target. __QUESTION__ is replaced at runtime.",
    )
    response_path: str = Field(
        default="$.answer",
        description="JSONPath to extract the answer from the response body.",
    )
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class TargetCreate(BaseModel):
    """Request body for POST /api/v1/targets"""
    name: str = Field(..., min_length=1, max_length=255)
    config: TargetConfig


class TargetRead(BaseModel):
    """Response body for GET /api/v1/targets/{id}"""
    model_config = {"from_attributes": True}   # Enable ORM → Pydantic conversion

    id: UUID
    name: str
    endpoint: str
    config: dict[str, Any]
    created_at: datetime
