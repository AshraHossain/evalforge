"""
api/routers/jobs.py — Eval Job Management

WHY THIS FILE EXISTS:
    This is the primary control surface for EvalForge.
    - POST /jobs   → submit a new eval job (enqueues to ARQ/Redis worker)
    - GET /jobs/{id} → poll job status + get final report
    - POST /jobs/{id}/approve → approve a HITL-paused job to continue

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/routers/jobs.py ───────────────────────────────────────────────────┐
    │  Uses:     worker/tasks.py (ARQ job enqueue)                           │
    │            api/schemas/job.py (JobCreate, JobRead)                     │
    │            db/models.py (EvalJob, Target)                              │
    │            agents/orchestrator.py (resume_eval_job for HITL)          │
    └─────────────────────────────────────────────────────────────────────────┘

    WHY ASYNC QUEUE:
    Running an eval can take 30–120 seconds (10+ LLM calls to Ollama).
    If we ran it synchronously in the HTTP handler, the client would time out.
    Instead: POST /jobs returns immediately with job_id, client polls or
    subscribes via WebSocket for progress.
"""

from uuid import UUID

import arq
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import User, get_current_user
from api.schemas.job import JobCreate, JobRead
from db.models import EvalJob, JobStatus, Target
from db.session import get_db
from config import settings

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


async def get_arq_pool():
    """Create an ARQ Redis connection pool for job enqueueing."""
    return await arq.create_pool(arq.connections.RedisSettings.from_dsn(settings.REDIS_URL))


@router.post("/", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    body: JobCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Submit a new eval job.

    1. Validates the target exists
    2. Creates an EvalJob row with status=queued
    3. Enqueues the job to ARQ (Redis) for async execution
    4. Returns immediately with job_id — don't wait for results

    Client should then poll GET /jobs/{id} or connect to WS /ws/jobs/{id}.
    """
    # Validate target exists
    stmt = select(Target).where(Target.id == body.target_id)
    target = (await db.execute(stmt)).scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")

    # Create job record
    job = EvalJob(
        target_id=body.target_id,
        status=JobStatus.QUEUED,
        triggered_by=body.triggered_by or user.username,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Enqueue to ARQ worker
    try:
        pool = await get_arq_pool()
        await pool.enqueue_job(
            "run_eval_job",
            job_id=str(job.id),
            target_id=str(body.target_id),
            target_config=target.config,
            num_test_cases=body.num_test_cases,
            seed_questions=body.seed_questions or [],
        )
        await pool.close()
    except Exception as e:
        # If Redis is down, mark job as failed
        job.status = JobStatus.FAILED
        job.error_message = f"Failed to enqueue: {str(e)}"
        await db.commit()
        raise HTTPException(status_code=503, detail=f"Job queue unavailable: {e}")

    return job


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get job status and final report (if complete).

    Poll this endpoint until status == "complete" or "failed".
    The `report` field is populated once status == "complete".
    """
    stmt = select(EvalJob).where(EvalJob.id == job_id)
    job = (await db.execute(stmt)).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/results")
async def get_job_results(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get all individual test case results for a completed job."""
    from db.models import Result, TestCase

    stmt = (
        select(Result, TestCase)
        .join(TestCase, TestCase.id == Result.test_case_id)
        .where(Result.job_id == job_id)
    )
    rows = (await db.execute(stmt)).all()

    return {
        "job_id": str(job_id),
        "results": [
            {
                "test_case_id": str(r.Result.test_case_id),
                "question": r.TestCase.question,
                "category": r.TestCase.category,
                "response": r.Result.response_text,
                "latency_ms": r.Result.latency_ms,
                "factual_consistency": r.Result.factual_consistency,
                "relevance": r.Result.relevance,
                "completeness": r.Result.completeness,
                "hallucination_detected": r.Result.hallucination_detected,
                "judge_reasoning": r.Result.judge_reasoning,
            }
            for r in rows
        ],
    }


@router.post("/{job_id}/approve", response_model=JobRead)
async def approve_hitl_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Approve a HITL-paused job and resume graph execution.

    When a job's hallucination rate exceeds the threshold, the LangGraph
    pauses at the hitl_gate node.  Calling this endpoint resumes it.
    """
    stmt = select(EvalJob).where(EvalJob.id == job_id)
    job = (await db.execute(stmt)).scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.HITL_PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting approval (status: {job.status})"
        )

    # Update status and re-enqueue the resume task
    job.status = JobStatus.RUNNING
    await db.commit()

    try:
        pool = await get_arq_pool()
        await pool.enqueue_job("resume_eval_job", job_id=str(job_id))
        await pool.close()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Job queue unavailable: {e}")

    await db.refresh(job)
    return job
