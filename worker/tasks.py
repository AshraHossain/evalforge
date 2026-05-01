"""
worker/tasks.py — ARQ Job Task Definitions

WHY THIS FILE EXISTS:
    ARQ (Async Redis Queue) needs task functions defined here.
    These are the actual functions the worker process executes when it
    picks up a job from the Redis queue.

    WHY ARQ OVER CELERY:
    - ARQ is async-native (asyncio), plays perfectly with FastAPI's event loop
    - Celery requires a separate sync/threading layer for async tasks
    - ARQ has a simpler API for our use case (eval jobs, not distributed compute)
    - ARQ supports job health signals (in_progress flag, retry, cancellation)

    RELATIONSHIP TO OTHER FILES:
    ┌─ worker/tasks.py ───────────────────────────────────────────────────────┐
    │  Called by:    worker/worker.py (ARQ picks up and runs these)          │
    │  Enqueued by:  api/routers/jobs.py (POST /jobs)                       │
    │  Calls:        agents/orchestrator.py (run_eval_job, resume_eval_job)  │
    │  Updates:      db/models.py (EvalJob status)                          │
    │  Publishes to: Redis (job completion signals for WebSocket)            │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import json
import logging
from typing import Optional


from db.models import JobStatus

logger = logging.getLogger(__name__)


async def run_eval_job(
    ctx: dict,
    job_id: str,
    target_id: str,
    target_config: dict,
    num_test_cases: int = 10,
    seed_questions: Optional[list[str]] = None,
) -> dict:
    """
    ARQ task: Execute a complete eval job.

    `ctx` is provided by ARQ and contains the Redis connection pool.
    This function is called by the worker when it dequeues a job.

    FLOW:
    1. Update job status to RUNNING in DB
    2. Run LangGraph orchestrator (testgen → execute → judge → memory → report)
    3. Save report to DB
    4. Publish completion signal to Redis (WebSocket clients receive it)
    5. Return summary dict (stored in ARQ job result)
    """
    from agents.orchestrator import run_eval_job as orchestrate
    from db.session import AsyncSessionLocal
    from db.models import EvalJob
    from sqlalchemy import select
    from datetime import datetime, timezone

    redis_client = ctx.get("redis")

    logger.info(f"[Worker] Starting eval job {job_id}")

    # ── 1. Mark job as running ─────────────────────────────────────────────
    async with AsyncSessionLocal() as session:
        async with session.begin():
            stmt = select(EvalJob).where(EvalJob.id == job_id)
            job = (await session.execute(stmt)).scalar_one_or_none()
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now(timezone.utc)

    # ── 2. Run orchestrator ────────────────────────────────────────────────
    try:
        report = await orchestrate(
            job_id=job_id,
            target_id=target_id,
            target_config=target_config,
            num_test_cases=num_test_cases,
            seed_questions=seed_questions,
        )

        # ── 3. Save report to DB ───────────────────────────────────────────
        from memory.store import EvalStore
        store = EvalStore()
        await store.save_report(
            job_id=job_id,
            report_dict=report.model_dump(mode="json") if report else {},
        )

        # ── 4. Publish completion to Redis ─────────────────────────────────
        if redis_client and report:
            await redis_client.publish(
                f"evalforge:job:{job_id}:results",
                json.dumps({
                    "type": "complete",
                    "job_id": job_id,
                    "badge": report.badge,
                    "overall_score": report.overall_reliability_score,
                }),
            )

        logger.info(f"[Worker] Job {job_id} completed. Badge={report.badge if report else 'N/A'}")
        return {"job_id": job_id, "status": "complete"}

    except Exception as e:
        logger.error(f"[Worker] Job {job_id} failed: {e}", exc_info=True)

        # Check if this was a HITL interrupt (not an error)
        from langgraph.errors import NodeInterrupt
        if isinstance(e, NodeInterrupt):
            # Update job status to hitl_pending
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    stmt = select(EvalJob).where(EvalJob.id == job_id)
                    job = (await session.execute(stmt)).scalar_one_or_none()
                    if job:
                        job.status = JobStatus.HITL_PENDING

            if redis_client:
                await redis_client.publish(
                    f"evalforge:job:{job_id}:results",
                    json.dumps({"type": "hitl_pending", "job_id": job_id, "message": str(e)}),
                )
            return {"job_id": job_id, "status": "hitl_pending"}

        # Real failure — mark job as failed
        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(EvalJob).where(EvalJob.id == job_id)
                job = (await session.execute(stmt)).scalar_one_or_none()
                if job:
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)[:1000]

        if redis_client:
            await redis_client.publish(
                f"evalforge:job:{job_id}:results",
                json.dumps({"type": "failed", "job_id": job_id, "error": str(e)[:200]}),
            )

        raise  # Re-raise so ARQ marks the job as failed


async def resume_eval_job(ctx: dict, job_id: str) -> dict:
    """
    ARQ task: Resume a HITL-paused job after human approval.

    Called by POST /api/v1/jobs/{id}/approve → jobs.py → ARQ enqueue.
    """
    from agents.orchestrator import resume_eval_job as resume
    from memory.store import EvalStore

    redis_client = ctx.get("redis")
    logger.info(f"[Worker] Resuming HITL job {job_id}")

    report = await resume(job_id=job_id)

    if report:
        store = EvalStore()
        await store.save_report(
            job_id=job_id,
            report_dict=report.model_dump(mode="json"),
        )

        if redis_client:
            await redis_client.publish(
                f"evalforge:job:{job_id}:results",
                json.dumps({
                    "type": "complete",
                    "job_id": job_id,
                    "badge": report.badge,
                    "overall_score": report.overall_reliability_score,
                }),
            )

    return {"job_id": job_id, "status": "complete"}
