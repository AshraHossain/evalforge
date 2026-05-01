"""
memory/trend_analyzer.py — Reliability Trend Analyzer

WHY THIS FILE EXISTS:
    After the Memory Agent pulls past runs, we want to know: is this target
    system getting better, worse, or staying the same?

    The trend is surfaced in the EvalReport as a human-readable string
    ("improving", "degrading", "stable", "insufficient_data") and used by
    the Report Agent's recommendation generator.

    RELATIONSHIP TO OTHER FILES:
    ┌─ memory/trend_analyzer.py ──────────────────────────────────────────────┐
    │  Called by:  agents/memory_agent.py                                    │
    │  Uses:       list of past run dicts (with pass_rate per run)           │
    │  Returns:    trend string                                              │
    └─────────────────────────────────────────────────────────────────────────┘

    ALGORITHM:
    Simple linear trend on pass_rates across last N runs.
    - If slope > +0.05 per run: "improving"
    - If slope < -0.05 per run: "degrading"
    - Otherwise: "stable"
    Requires at least 3 runs to compute a meaningful trend.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

MIN_RUNS_FOR_TREND = 3


class TrendAnalyzer:
    def compute_trend(
        self, past_runs: list[dict]
    ) -> Literal["improving", "degrading", "stable", "insufficient_data"]:
        """
        Compute the reliability trend for a target based on historical pass rates.

        past_runs: sorted newest-first list of {"pass_rate": float, ...} dicts.
        """
        if len(past_runs) < MIN_RUNS_FOR_TREND:
            return "insufficient_data"

        # Take last N runs, oldest first for trend calculation
        rates = [r["pass_rate"] for r in reversed(past_runs[:10])]

        if len(rates) < 2:
            return "insufficient_data"

        # Simple linear regression slope
        n = len(rates)
        x_mean = (n - 1) / 2
        y_mean = sum(rates) / n

        numerator = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(rates))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > 0.05:
            trend = "improving"
        elif slope < -0.05:
            trend = "degrading"
        else:
            trend = "stable"

        logger.info(f"[Trend] slope={slope:.4f} → {trend}")
        return trend
