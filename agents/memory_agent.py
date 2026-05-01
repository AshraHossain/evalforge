"""
agents/memory_agent.py — Memory Agent

WHY THIS FILE EXISTS:
    The Memory Agent does two things:
    1. PERSIST — Save the current run's test cases and results to PostgreSQL.
    2. ANALYSE — Query past runs to build a MemoryContext (regressions, trends).

    "Memory" here is NOT in-context LLM memory.  It's persistent, structured
    storage.  The LLM in this project doesn't remember across runs — PostgreSQL
    does.  This is the right architecture for production systems.

    RELATIONSHIP TO OTHER FILES:
    ┌─ agents/memory_agent.py ────────────────────────────────────────────────┐
    │  Reads from EvalState:   job_id, test_cases, execution_results, scores  │
    │  Writes to EvalState:    memory_context                                │
    │  Calls:                  memory/store.py        (read/write to DB)     │
    │                          memory/regression_detector.py                 │
    │                          memory/trend_analyzer.py                      │
    │  Called by:              agents/orchestrator.py node "memory"          │
    │  Feeds into:             agents/report_agent.py (via memory_context)   │
    │                          orchestrator route_hitl() (HITL trigger)      │
    └─────────────────────────────────────────────────────────────────────────┘

    WHY BEFORE THE REPORT:
    The Report Agent needs historical context (is this a regression? is the
    target improving or degrading?) to generate useful recommendations.  The
    Memory Agent gathers that context first, then the Report Agent synthesises
    it.  This separation keeps each agent's responsibility clean.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from api.schemas.report import MemoryContext

if TYPE_CHECKING:
    from agents.orchestrator import EvalState

logger = logging.getLogger(__name__)


async def memory_agent_async(state: "EvalState") -> dict:
    from memory.store import EvalStore
    from memory.regression_detector import RegressionDetector
    from memory.trend_analyzer import TrendAnalyzer

    job_id = state["job_id"]
    target_id = state["target_id"]
    test_cases = state["test_cases"]
    results = state["execution_results"]
    scores = state["scores"]

    logger.info(f"[Memory] Persisting results for job {job_id}")

    # ── 1. Persist current run ─────────────────────────────────────────────
    # Saves test_cases and results to PostgreSQL so future runs can compare.
    store = EvalStore()
    await store.save_results(
        job_id=job_id,
        test_cases=test_cases,
        results=results,
        scores=scores,
    )

    # ── 2. Query past runs for this target ─────────────────────────────────
    past_runs = await store.get_past_runs(target_id=target_id, limit=10)

    if not past_runs:
        logger.info(f"[Memory] No past runs found for target {target_id}")
        return {
            "memory_context": MemoryContext(
                target_id=str(target_id),
                total_past_runs=0,
            )
        }

    # ── 3. Regression detection ────────────────────────────────────────────
    # Compares current run scores against previous run scores to find cases
    # that regressed (passed before, fail now) or are flapping.
    detector = RegressionDetector()
    regression_cases, flapping_cases = detector.detect(
        current_scores=scores,
        past_runs=past_runs,
    )

    # ── 4. Category failure rate analysis ─────────────────────────────────
    # Aggregates scores across ALL past runs to build per-category failure rates.
    # e.g. {"hallucination_trap": 0.78} means 78% of hallucination tests fail.
    all_scores_by_category = await store.get_category_scores(target_id=target_id)
    category_failure_rates = {
        cat: 1.0 - avg_score
        for cat, avg_score in all_scores_by_category.items()
    }
    worst_cats = sorted(
        category_failure_rates, key=lambda c: category_failure_rates[c], reverse=True
    )[:3]

    # ── 5. Trend analysis ──────────────────────────────────────────────────
    analyzer = TrendAnalyzer()
    trend = analyzer.compute_trend(past_runs=past_runs)

    memory_ctx = MemoryContext(
        target_id=str(target_id),
        total_past_runs=len(past_runs),
        regression_cases=[str(tc_id) for tc_id in regression_cases],
        flapping_cases=[str(tc_id) for tc_id in flapping_cases],
        category_failure_rates=category_failure_rates,
        worst_performing_categories=worst_cats,
        trend=trend,
    )

    logger.info(
        f"[Memory] Past runs: {len(past_runs)}, "
        f"Regressions: {len(regression_cases)}, Trend: {trend}"
    )
    return {"memory_context": memory_ctx}


def memory_agent(state: "EvalState") -> dict:
    """Sync wrapper for LangGraph."""
    return asyncio.run(memory_agent_async(state))
