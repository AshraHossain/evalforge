"""
agents/execution_agent.py — Execution Agent

WHY THIS FILE EXISTS:
    The Execution Agent is the *only* part of EvalForge that talks to the
    system under evaluation.  It treats the target as a complete black box —
    just an HTTP endpoint — and sends each test case question to it.

    NO LLM IS CALLED HERE.
    This agent is pure async HTTP.  It uses httpx for async requests so
    multiple test cases can be in-flight simultaneously (asyncio.gather).

    RELATIONSHIP TO OTHER FILES:
    ┌─ agents/execution_agent.py ─────────────────────────────────────────────┐
    │  Reads from EvalState:   test_cases, target_config                     │
    │  Writes to EvalState:    execution_results                             │
    │  Uses:                   api/schemas/target.py (TargetConfig)         │
    │                          api/schemas/job.py (Result)                  │
    │  Called by:              agents/orchestrator.py node "execute"        │
    │  Streams via:            Redis pub/sub (partial results to WebSocket)  │
    └─────────────────────────────────────────────────────────────────────────┘

    CONCURRENCY DESIGN:
    asyncio.gather(*tasks) fires all HTTP calls concurrently up to
    MAX_CONCURRENT requests.  This means a 20-question eval run finishes in
    ~1× the target's p95 latency, not 20×.

    REQUEST TEMPLATE:
    The target_config.request_template dict may contain "__QUESTION__" as a
    value — we do a deep-replace so you can nest it:
      {"messages": [{"role": "user", "content": "__QUESTION__"}]}
    becomes:
      {"messages": [{"role": "user", "content": "What is 2+2?"}]}
"""

import asyncio
import json
import logging
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import httpx
from jsonpath_ng import parse as jsonpath_parse

from api.schemas.job import Result, TestCase
from api.schemas.target import TargetConfig

if TYPE_CHECKING:
    from agents.orchestrator import EvalState

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 5   # Max simultaneous HTTP calls to the target system


def _inject_question(template: dict, question: str) -> dict:
    """
    Deep-replace the string "__QUESTION__" with the actual question text.
    Works at any nesting level of the dict.
    """
    def replace(obj: Any) -> Any:
        if isinstance(obj, str):
            return obj.replace("__QUESTION__", question)
        if isinstance(obj, dict):
            return {k: replace(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace(i) for i in obj]
        return obj

    return replace(deepcopy(template))


def _extract_answer(response_body: dict, response_path: str) -> str:
    """
    Extract the answer text from the target's JSON response using JSONPath.

    Example:
      response_path = "$.answer"
      response_body = {"answer": "Paris", "sources": [...]}
      → "Paris"
    """
    try:
        expr = jsonpath_parse(response_path)
        matches = expr.find(response_body)
        if matches:
            return str(matches[0].value)
    except Exception:
        pass
    # Fallback: stringify the whole response
    return json.dumps(response_body)


async def _call_target(
    client: httpx.AsyncClient,
    test_case: TestCase,
    cfg: TargetConfig,
    redis_client=None,
    job_id: str = "",
) -> Result:
    """
    Call the target system for a single test case.

    Publishes partial results to Redis so the WebSocket endpoint can
    stream them to the browser in real time.
    """
    body = _inject_question(cfg.request_template, test_case.question)
    headers = {"Content-Type": "application/json"}
    if cfg.auth_header:
        headers["Authorization"] = cfg.auth_header

    start = time.monotonic()
    try:
        resp = await client.post(
            cfg.endpoint,
            json=body,
            headers=headers,
            timeout=cfg.timeout_seconds,
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - start) * 1000)
        answer = _extract_answer(resp.json(), cfg.response_path)
        result = Result(
            test_case_id=test_case.id,
            response_text=answer,
            latency_ms=latency_ms,
            status="success",
        )
    except httpx.TimeoutException:
        latency_ms = int(cfg.timeout_seconds * 1000)
        result = Result(
            test_case_id=test_case.id,
            response_text="",
            latency_ms=latency_ms,
            status="timeout",
            error_detail=f"Target timed out after {cfg.timeout_seconds}s",
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        result = Result(
            test_case_id=test_case.id,
            response_text="",
            latency_ms=latency_ms,
            status="error",
            error_detail=str(e),
        )

    # ── Publish partial result to Redis pub/sub ────────────────────────────
    # The WebSocket router subscribes to this channel and streams each result
    # to the browser as soon as it arrives, giving a live progress view.
    if redis_client and job_id:
        try:
            await redis_client.publish(
                f"evalforge:job:{job_id}:results",
                json.dumps({
                    "type": "result",
                    "test_case_id": result.test_case_id,
                    "status": result.status,
                    "latency_ms": result.latency_ms,
                }),
            )
        except Exception:
            pass  # Redis being down should not stop the eval

    return result


async def execution_agent_async(state: "EvalState", redis_client=None) -> dict:
    """
    Async version — used in production (FastAPI async context).
    """
    logger.info(
        f"[Execute] Running {len(state['test_cases'])} test cases "
        f"against {state['target_config']['endpoint']}"
    )

    cfg = TargetConfig(**state["target_config"])
    test_cases: list[TestCase] = state["test_cases"]
    job_id = state["job_id"]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded_call(tc: TestCase) -> Result:
        async with semaphore:
            return await _call_target(tc_client, tc, cfg, redis_client, job_id)

    async with httpx.AsyncClient() as tc_client:
        results = await asyncio.gather(*[bounded_call(tc) for tc in test_cases])

    logger.info(f"[Execute] Completed {len(results)} calls")
    return {"execution_results": list(results)}


def execution_agent(state: "EvalState") -> dict:
    """
    Sync wrapper — called by LangGraph which may run in a sync context.
    Uses asyncio.run() to drive the async implementation.
    """
    return asyncio.run(execution_agent_async(state))
