"""
api/routers/targets.py — Target System CRUD

WHY THIS FILE EXISTS:
    Before you can run an eval, you register the system you want to evaluate.
    This router handles creating targets and fetching their eval history.

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/routers/targets.py ────────────────────────────────────────────────┐
    │  Uses:     api/schemas/target.py (TargetCreate, TargetRead)            │
    │            db/models.py (Target ORM model)                             │
    │            db/session.py (get_db dependency)                           │
    │            api/auth.py (get_current_user)                              │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import User, get_current_user
from api.schemas.target import TargetCreate, TargetRead
from db.models import Target
from db.session import get_db

router = APIRouter(prefix="/api/v1/targets", tags=["targets"])


@router.post("/", response_model=TargetRead, status_code=status.HTTP_201_CREATED)
async def register_target(
    body: TargetCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Register a new target system.

    Request body example:
    {
      "name": "RAG Stock Analyzer",
      "config": {
        "endpoint": "http://localhost:8001/query",
        "request_template": {"query": "__QUESTION__"},
        "response_path": "$.answer",
        "timeout_seconds": 30
      }
    }
    """
    target = Target(
        name=body.name,
        endpoint=body.config.endpoint,
        config=body.config.model_dump(),
    )
    db.add(target)
    await db.commit()
    await db.refresh(target)
    return target


@router.get("/{target_id}", response_model=TargetRead)
async def get_target(
    target_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get target details."""
    stmt = select(Target).where(Target.id == target_id)
    target = (await db.execute(stmt)).scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    return target


@router.get("/{target_id}/trends")
async def get_target_trends(
    target_id: UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Get reliability trend data for a target across all eval runs.

    Returns time-series data suitable for charting pass_rate over time.
    """
    from db.models import EvalJob
    from sqlalchemy import desc

    stmt = (
        select(EvalJob)
        .where(EvalJob.target_id == target_id)
        .where(EvalJob.status == "complete")
        .order_by(desc(EvalJob.completed_at))
        .limit(20)
    )
    jobs = (await db.execute(stmt)).scalars().all()

    trend_data = []
    for job in reversed(jobs):
        if job.report:
            trend_data.append({
                "job_id": str(job.id),
                "timestamp": job.completed_at.isoformat() if job.completed_at else None,
                "overall_score": job.report.get("overall_reliability_score"),
                "pass_rate": job.report.get("pass_rate"),
                "hallucination_rate": job.report.get("hallucination_rate"),
                "badge": job.report.get("badge"),
            })

    return {"target_id": str(target_id), "trend": trend_data}
