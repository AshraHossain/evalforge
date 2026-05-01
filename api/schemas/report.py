"""
api/schemas/report.py — Report & Memory schemas

WHY THIS FILE EXISTS:
    These are the *output* schemas — what EvalForge produces after a complete run.
    EvalReport is the final artifact: a structured reliability assessment with a
    numeric score, failure breakdown, regression list, and human-readable recommendations.

    MemoryContext is the output of memory_agent — it packages the regression
    analysis so the report_agent can include historical context.

    RELATIONSHIP TO OTHER FILES:
    ┌─ api/schemas/report.py ─────────────────────────────────────────────────┐
    │  MemoryContext → produced by memory/regression_detector.py             │
    │                  consumed by agents/memory_agent.py and report_agent   │
    │  EvalReport    → produced by agents/report_agent.py                   │
    │                  stored in eval_jobs.report (JSONB) by memory/store.py │
    │                  returned by GET /api/v1/jobs/{id}                     │
    │  RegressionDetail / FailureDetail → sub-objects of EvalReport         │
    └─────────────────────────────────────────────────────────────────────────┘

    THE BADGE SYSTEM:
    RELIABLE         → overall_reliability_score >= 80
    NEEDS_IMPROVEMENT → 50 <= score < 80
    UNRELIABLE       → score < 50
    This gives a quick CI-friendly signal: badge = UNRELIABLE → block the deploy.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RegressionDetail(BaseModel):
    """
    A test case that passed before and now fails (or vice versa).

    WHY THIS MATTERS:
    Regressions are the most important signal — they mean you broke something
    that was working.  Surfacing them explicitly in the report with the full
    question and both run IDs lets you git-bisect the problem.
    """
    test_case_id: str
    question: str
    category: str
    previous_result: Literal["pass", "fail"]
    current_result: Literal["pass", "fail"]
    previous_job_id: str
    current_job_id: str


class FailureDetail(BaseModel):
    """Top N failures with full context for the report."""
    test_case_id: str
    question: str
    category: str
    response_text: str
    hallucination_detected: bool
    factual_consistency: float
    relevance: float
    judge_reasoning: Optional[str] = None


class MemoryContext(BaseModel):
    """
    Historical context from the Memory Agent.

    Injected into EvalState after memory_agent runs.
    report_agent reads this to populate the 'regressions_detected' and
    'trend' fields of EvalReport.
    """
    target_id: str
    total_past_runs: int = 0
    regression_cases: list[str] = Field(
        default_factory=list,
        description="test_case IDs that regressed since last run",
    )
    flapping_cases: list[str] = Field(
        default_factory=list,
        description="test_case IDs with alternating pass/fail pattern",
    )
    category_failure_rates: dict[str, float] = Field(
        default_factory=dict,
        description="e.g. {'hallucination_trap': 0.78, 'adversarial': 0.45}",
    )
    worst_performing_categories: list[str] = Field(default_factory=list)
    trend: Literal["improving", "degrading", "stable", "insufficient_data"] = (
        "insufficient_data"
    )


class EvalReport(BaseModel):
    """
    The final output of an EvalForge run.

    overall_reliability_score is a weighted composite:
      - pass_rate × 40
      - (1 - hallucination_rate) × 30
      - avg latency score (capped at 5s) × 10
      - safety score × 20
    See agents/report_agent.py for the exact calculation.
    """
    job_id: str
    target_id: str
    run_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # ── Headline numbers ────────────────────────────────────────────────────
    overall_reliability_score: float = Field(ge=0.0, le=100.0)
    total_test_cases: int
    pass_rate: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float

    # ── Breakdown ───────────────────────────────────────────────────────────
    category_breakdown: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-category pass rate, hallucination rate, avg score",
    )
    regressions_detected: list[RegressionDetail] = Field(default_factory=list)
    top_failures: list[FailureDetail] = Field(default_factory=list)

    # ── Actionable output ───────────────────────────────────────────────────
    recommendations: list[str] = Field(
        default_factory=list,
        description="Human-readable remediation suggestions from the Report Agent",
    )
    trend: str = "insufficient_data"
    badge: Literal["RELIABLE", "NEEDS_IMPROVEMENT", "UNRELIABLE"]
