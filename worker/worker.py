"""
worker/worker.py — ARQ Worker Process

WHY THIS FILE EXISTS:
    This is the worker process that sits separately from the API server.
    It listens to the Redis queue and executes eval jobs asynchronously.

    RUN WITH:
        python worker/worker.py
        # or
        arq worker.worker.WorkerSettings

    RELATIONSHIP TO OTHER FILES:
    ┌─ worker/worker.py ──────────────────────────────────────────────────────┐
    │  Registers:  worker/tasks.py functions (run_eval_job, resume_eval_job) │
    │  Connects to: Redis (REDIS_URL from config)                            │
    │  Uses:        ctx["redis"] in tasks for pub/sub                        │
    └─────────────────────────────────────────────────────────────────────────┘

    CONCURRENCY:
    max_jobs=5 means the worker handles up to 5 eval jobs simultaneously.
    Each job runs asyncio-concurrently (not in separate threads).
    Since our jobs are mostly I/O-bound (HTTP calls to target, calls to Ollama),
    asyncio concurrency is appropriate and efficient.

    WORKER HEALTH:
    ARQ tracks job start/end times in Redis.  You can inspect queued jobs with:
        redis-cli hgetall arq:job:<job_id>
"""

import logging
import sys
import os

# Add project root to path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arq.connections import RedisSettings

from config import settings
from worker.tasks import resume_eval_job, run_eval_job

logger = logging.getLogger(__name__)


class WorkerSettings:
    """
    ARQ worker configuration.

    ARQ discovers this class when you run `arq worker.worker.WorkerSettings`.
    """

    # ── Functions this worker can execute ─────────────────────────────────
    # These must match the function names enqueued by api/routers/jobs.py
    functions = [run_eval_job, resume_eval_job]

    # ── Redis connection ───────────────────────────────────────────────────
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)

    # ── Concurrency ────────────────────────────────────────────────────────
    max_jobs = 5              # Max concurrent eval jobs
    job_timeout = 600         # 10 minutes max per job
    keep_result = 3600        # Keep job result in Redis for 1 hour

    # ── Health logging ────────────────────────────────────────────────────
    queue_read_limit = 50

    @staticmethod
    async def on_startup(ctx: dict):
        """
        Called when the worker starts.
        Set up any shared resources that tasks can access via ctx.
        """
        import redis.asyncio as aioredis

        logger.info("EvalForge worker starting up...")
        # Attach a Redis client to ctx so tasks can use it for pub/sub
        ctx["redis"] = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("Worker ready. Waiting for jobs...")

    @staticmethod
    async def on_shutdown(ctx: dict):
        """Clean up on worker shutdown."""
        if "redis" in ctx:
            await ctx["redis"].aclose()
        logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    import asyncio
    import arq

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    asyncio.run(arq.run_worker(WorkerSettings))
