"""
api/routers/websocket.py — Real-time WebSocket streaming

WHY THIS FILE EXISTS:
    Eval jobs take 30–120 seconds.  Without streaming, the user stares at a
    spinner until the whole job completes.  With WebSocket streaming, each
    test case result appears in real time as it's scored — like a live test
    runner output.

    ARCHITECTURE:
    ┌─ Worker ──────────────────────────────────────────────────────────────┐
    │  execution_agent runs test case → publishes to Redis pub/sub channel  │
    │  Channel name: "evalforge:job:{job_id}:results"                      │
    └───────────────────────────────────────┬───────────────────────────────┘
                                             │ Redis SUBSCRIBE
    ┌─ WebSocket Handler ───────────────────▼───────────────────────────────┐
    │  Client connects to WS /ws/jobs/{job_id}                             │
    │  Handler subscribes to Redis channel                                 │
    │  For each message received → forwards to WebSocket client            │
    └───────────────────────────────────────────────────────────────────────┘

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/routers/websocket.py ──────────────────────────────────────────────┐
    │  Subscribes to:  Redis pub/sub (published by execution_agent.py)       │
    │  Used by:        api/main.py (router registration)                     │
    │  Client usage:   ws = new WebSocket("ws://localhost:8000/ws/jobs/{id}")│
    └─────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """
    Stream real-time eval progress for a job.

    Connect from the browser:
        const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${jobId}`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "result") {
                // Update UI with partial result
            } else if (data.type === "complete") {
                // Eval finished — fetch full report
            }
        };

    Message types sent to client:
      {"type": "result", "test_case_id": "...", "status": "success", "latency_ms": 342}
      {"type": "complete", "badge": "RELIABLE", "overall_score": 87.5}
      {"type": "error", "detail": "..."}
    """
    await websocket.accept()
    logger.info(f"[WebSocket] Client connected for job {job_id}")

    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    channel = f"evalforge:job:{job_id}:results"

    try:
        await pubsub.subscribe(channel)

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            data = json.loads(message["data"])
            await websocket.send_json(data)

            # Close WebSocket when job signals completion
            if data.get("type") in ("complete", "failed", "hitl_pending"):
                break

    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client disconnected from job {job_id}")
    except Exception as e:
        logger.error(f"[WebSocket] Error for job {job_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        await pubsub.unsubscribe(channel)
        await redis_client.aclose()
        logger.info(f"[WebSocket] Connection closed for job {job_id}")
