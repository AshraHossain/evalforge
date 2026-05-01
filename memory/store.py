"""
memory/store.py — PostgreSQL Read/Write Store

WHY THIS FILE EXISTS:
    The Memory Agent needs to persist results and query history.  All raw
    SQL/SQLAlchemy logic lives here — the agents stay clean and just call
    store methods.  This separation means you could swap PostgreSQL for
    another DB by rewriting only this file.

    RELATIONSHIP TO OTHER FILES:
    ┌─ memory/store.py ───────────────────────────────────────────────────────┐
    │  Called by:  agents/memory_agent.py                                    │
    │  Uses:       db/models.py (ORM models), db/session.py (engine)        │
    │  Also used:  api/routers/targets.py, api/routers/jobs.py              │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import logging

from sqlalchemy import select, func, desc

from api.schemas.job import Result, Score, TestCase
from db.models import EvalJob, Result as ResultORM, Target, TestCase as TestCaseORM
from db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


class EvalStore:
    """Async PostgreSQL store for eval data."""

    async def save_results(
        self,
        job_id: str,
        test_cases: list[TestCase],
        results: list[Result],
        scores: list[Score],
    ) -> None:
        """Persist test cases + results + scores for a completed run."""
        result_map = {r.test_case_id: r for r in results}
        score_map = {s.test_case_id: s for s in scores}

        async with AsyncSessionLocal() as session:
            async with session.begin():
                for tc in test_cases:
                    result = result_map.get(tc.id)
                    score = score_map.get(tc.id)

                    # Upsert test case
                    tc_orm = TestCaseORM(
                        id=tc.id,
                        job_id=job_id,
                        question=tc.question,
                        category=tc.category.value,
                        expected_behavior=tc.expected_behavior,
                        ground_truth=tc.ground_truth,
                        tags=tc.tags,
                        source=tc.source,
                    )
                    session.add(tc_orm)

                    if result and score:
                        result_orm = ResultORM(
                            test_case_id=tc.id,
                            job_id=job_id,
                            response_text=result.response_text,
                            latency_ms=result.latency_ms,
                            factual_consistency=score.factual_consistency,
                            relevance=score.relevance,
                            completeness=score.completeness,
                            safety=score.safety,
                            hallucination_detected=score.hallucination_detected,
                            judge_reasoning=score.judge_reasoning,
                            scored_by=score.scored_by,
                        )
                        session.add(result_orm)

        logger.info(f"[Store] Saved {len(test_cases)} test cases for job {job_id}")

    async def get_past_runs(self, target_id: str, limit: int = 10) -> list[dict]:
        """
        Fetch the last N completed eval jobs for a target, with their scores.

        Returns list of dicts:
          [{"job_id": str, "scores": [Score], "pass_rate": float}, ...]
        """
        async with AsyncSessionLocal() as session:
            stmt = (
                select(EvalJob)
                .where(EvalJob.target_id == target_id)
                .where(EvalJob.status == "complete")
                .order_by(desc(EvalJob.completed_at))
                .limit(limit)
            )
            jobs = (await session.execute(stmt)).scalars().all()

            runs = []
            for job in jobs:
                # Fetch results for this job
                result_stmt = select(ResultORM).where(ResultORM.job_id == str(job.id))
                results = (await session.execute(result_stmt)).scalars().all()

                scores = [
                    Score(
                        test_case_id=str(r.test_case_id),
                        factual_consistency=r.factual_consistency or 0.0,
                        relevance=r.relevance or 0.0,
                        completeness=r.completeness or 0.0,
                        safety=r.safety or 1.0,
                        hallucination_detected=r.hallucination_detected or False,
                        judge_reasoning=r.judge_reasoning,
                        scored_by=r.scored_by or "llm_judge",
                    )
                    for r in results
                ]

                if scores:
                    pass_rate = sum(
                        1 for s in scores
                        if (s.factual_consistency + s.relevance + s.completeness) / 3 >= 0.6
                        and not s.hallucination_detected
                    ) / len(scores)
                else:
                    pass_rate = 0.0

                runs.append({
                    "job_id": str(job.id),
                    "scores": scores,
                    "pass_rate": pass_rate,
                    "completed_at": job.completed_at,
                })

            return runs

    async def get_category_scores(self, target_id: str) -> dict[str, float]:
        """
        Get average composite score per test category across all runs for a target.

        Used by Memory Agent to compute category_failure_rates.
        """
        async with AsyncSessionLocal() as session:
            stmt = (
                select(
                    TestCaseORM.category,
                    func.avg(
                        (ResultORM.factual_consistency + ResultORM.relevance + ResultORM.completeness) / 3
                    ).label("avg_score")
                )
                .join(ResultORM, ResultORM.test_case_id == TestCaseORM.id)
                .join(EvalJob, EvalJob.id == ResultORM.job_id)
                .where(EvalJob.target_id == target_id)
                .group_by(TestCaseORM.category)
            )
            rows = (await session.execute(stmt)).all()
            return {row.category: float(row.avg_score or 0.0) for row in rows}

    async def save_report(self, job_id: str, report_dict: dict) -> None:
        """Store the final EvalReport JSON in eval_jobs.report."""
        async with AsyncSessionLocal() as session:
            async with session.begin():
                stmt = select(EvalJob).where(EvalJob.id == job_id)
                job = (await session.execute(stmt)).scalar_one_or_none()
                if job:
                    job.report = report_dict
                    job.status = "complete"

    async def get_or_create_target(
        self, name: str, endpoint: str, config: dict
    ) -> str:
        """Get existing target or create a new one. Returns target_id."""
        async with AsyncSessionLocal() as session:
            stmt = select(Target).where(Target.endpoint == endpoint)
            target = (await session.execute(stmt)).scalar_one_or_none()
            if not target:
                async with session.begin():
                    target = Target(name=name, endpoint=endpoint, config=config)
                    session.add(target)
            return str(target.id)
