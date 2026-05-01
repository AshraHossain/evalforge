"""
tests/fixtures/mock_target_server.py — Mock Target HTTP Server

WHY THIS FILE EXISTS:
    The Execution Agent needs to send HTTP requests to a "target system."
    In tests, we can't depend on a real LLM app being available.
    This fixture creates a local FastAPI server that mimics a target system,
    returning predictable responses so tests are deterministic.

    RELATIONSHIP TO OTHER FILES:
    ┌─ tests/fixtures/mock_target_server.py ────────────────────────────────────┐
    │  Used by:  tests/unit/test_testgen_agent.py                               │
    │            tests/integration/test_full_eval_pipeline.py                   │
    │  Mimics:   Any system registered as a Target (e.g. RAG Stock Analyzer)   │
    └───────────────────────────────────────────────────────────────────────────┘

    USAGE:
        async with mock_target_server() as base_url:
            target_config = TargetConfig(
                endpoint=f"{base_url}/query",
                request_template={"query": "__QUESTION__"},
                response_path="$.answer",
            )
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

mock_app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@mock_app.post("/query")
async def mock_query(body: QueryRequest):
    """
    Simulate a RAG-style target.

    Returns:
    - A sensible answer for normal questions
    - A hallucinated answer for hallucination_trap questions (to test judge)
    - An error for adversarial prompts
    """
    question = body.query.lower()

    if "ignore all previous" in question or "jailbreak" in question:
        return {"answer": "I cannot comply with that request.", "sources": []}

    if "dragon" in question or "doesn't exist" in question or "fake" in question:
        # Simulate hallucination: the model confidently makes something up
        return {
            "answer": "Yes, dragons were discovered on Mars in 2024 by NASA.",
            "sources": ["nasa.gov/fake"],
        }

    if "capital of france" in question:
        return {"answer": "The capital of France is Paris.", "sources": []}

    if "what can you help" in question:
        return {
            "answer": "I can help you with stock analysis, market research, and financial data.",
            "sources": [],
        }

    # Generic sensible answer for other questions
    return {
        "answer": f"Based on available data, here is an answer to: {body.query[:50]}...",
        "sources": ["source1.com", "source2.com"],
    }


@asynccontextmanager
async def mock_target_server(host: str = "127.0.0.1", port: int = 18765) -> AsyncGenerator[str, None]:
    """
    Context manager that starts/stops the mock target server.

    Usage:
        async with mock_target_server() as base_url:
            # base_url = "http://127.0.0.1:18765"
            ...
    """
    config = uvicorn.Config(mock_app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())
    # Wait for server to start
    await asyncio.sleep(0.3)

    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        await task
