"""
memory/regression_detector.py — Regression & Flapping Detector

WHY THIS FILE EXISTS:
    This is what separates EvalForge from a simple "run tests once" tool.
    By comparing the current run against past runs, we can detect:

    REGRESSION:   Test case passed in run N-1, failed in run N.
                  → Something broke between these two runs.

    FLAPPING:     Test case alternates pass/fail across runs: P F P F P
                  → Indicates non-deterministic behaviour in the target system.
                  The same prompt sometimes gets a good answer, sometimes bad.
                  This is a reliability red flag for production LLM systems.

    RELATIONSHIP TO OTHER FILES:
    ┌─ memory/regression_detector.py ────────────────────────────────────────┐
    │  Called by:  agents/memory_agent.py                                    │
    │  Reads:      list of past run dicts from memory/store.py               │
    │  Returns:    (regression_cases, flapping_cases) — lists of test IDs   │
    └─────────────────────────────────────────────────────────────────────────┘

    ALGORITHM:
    For each test case that appears in the current run AND at least one past run:
    1. Build a pass/fail timeline: [True, False, True, True, False]
    2. Check last transition: was it failing → now passing (fix) or
                              passing → now failing (regression)?
    3. Check for alternating pattern (flapping): P F P or F P F in last 3+ runs
"""

import logging
from typing import Tuple

from api.schemas.job import Score

logger = logging.getLogger(__name__)

# Threshold: score below this is a "fail"
PASS_THRESHOLD = 0.6
MIN_RUNS_FOR_FLAP = 3   # Need at least 3 runs to detect a flapping pattern


def _is_pass(score: Score) -> bool:
    avg = (score.factual_consistency + score.relevance + score.completeness) / 3
    return avg >= PASS_THRESHOLD and not score.hallucination_detected


class RegressionDetector:
    """
    Compares current run scores against historical runs to find regressions
    and flapping test cases.
    """

    def detect(
        self,
        current_scores: list[Score],
        past_runs: list[dict],
    ) -> Tuple[list[str], list[str]]:
        """
        Returns:
          regression_cases: test_case_ids that regressed (pass → fail)
          flapping_cases:   test_case_ids with alternating pass/fail pattern
        """
        if not past_runs:
            return [], []

        # Build current pass/fail map
        current_map = {s.test_case_id: _is_pass(s) for s in current_scores}

        # Build historical pass/fail maps per test case
        # history[test_case_id] = [oldest_result, ..., newest_result]
        history: dict[str, list[bool]] = {}
        for run in past_runs:  # oldest first (caller provides in chronological order)
            for score in run.get("scores", []):
                tc_id = score.test_case_id
                if tc_id not in history:
                    history[tc_id] = []
                history[tc_id].append(_is_pass(score))

        regressions = []
        flapping = []

        for tc_id, current_pass in current_map.items():
            past = history.get(tc_id, [])
            if not past:
                continue   # First time seeing this test case — no history to compare

            # ── Regression detection ───────────────────────────────────────
            # Previous run passed, current run failed
            last_past = past[-1]
            if last_past and not current_pass:
                logger.info(f"[Regression] Test case {tc_id}: was PASS, now FAIL")
                regressions.append(tc_id)

            # ── Flapping detection ─────────────────────────────────────────
            # Build full timeline: [...past..., current]
            full_timeline = past + [current_pass]
            if len(full_timeline) >= MIN_RUNS_FOR_FLAP:
                recent = full_timeline[-MIN_RUNS_FOR_FLAP:]
                # Flapping = at least 2 direction changes in the recent window
                changes = sum(
                    1 for i in range(1, len(recent)) if recent[i] != recent[i - 1]
                )
                if changes >= 2:
                    logger.info(
                        f"[Flapping] Test case {tc_id}: timeline={recent}"
                    )
                    flapping.append(tc_id)

        logger.info(
            f"[RegressionDetector] Regressions={len(regressions)}, "
            f"Flapping={len(flapping)}"
        )
        return regressions, flapping
