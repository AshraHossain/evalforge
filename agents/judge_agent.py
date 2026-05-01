"""
agents/judge_agent.py — Judge Agent

WHY THIS FILE EXISTS:
    The Judge Agent scores each (question, response) pair on 4 dimensions:
    factual_consistency, relevance, completeness, safety.

    THIS IS THE SECOND PLACE OLLAMA IS CALLED.
    In Phase 1 we use Ollama gemma4:26b as the LLM-as-Judge.
    In Phase 4 we'll add a LoRA classifier and route cheap/confident cases
    to it, reserving Ollama for the hard/ambiguous ones.

    RELATIONSHIP TO OTHER FILES:
    ┌─ agents/judge_agent.py ─────────────────────────────────────────────────┐
    │  Reads from EvalState:   test_cases, execution_results                 │
    │  Writes to EvalState:    scores                                        │
    │  Calls:                  evaluation/scorer_router.py                  │
    │    which calls:          evaluation/llm_judge.py  (ChatOllama)        │
    │                          evaluation/lora_judge.py (stub in Phase 1)   │
    │  Called by:              agents/orchestrator.py node "judge"          │
    └─────────────────────────────────────────────────────────────────────────┘

    WHY LLM-AS-JUDGE:
    Traditional metrics (BLEU, ROUGE) measure surface similarity — useless for
    LLM output evaluation.  An LLM judge reads the question and answer and
    reasons about whether the answer is actually correct, relevant, and complete.
    Using the same local model for judging keeps cost at $0.

    CONCURRENT SCORING:
    Like the Execution Agent, we run up to MAX_CONCURRENT judges in parallel.
    Each judge call is independent so concurrency is safe.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from api.schemas.job import Result, Score, TestCase

if TYPE_CHECKING:
    from agents.orchestrator import EvalState

logger = logging.getLogger(__name__)

MAX_CONCURRENT_JUDGEMENTS = 3   # Ollama can only handle a few concurrent requests


async def _judge_one(
    test_case: TestCase,
    result: Result,
    scorer_router,
) -> Score:
    """Score a single (question, response) pair."""
    if result.status != "success":
        # Don't judge failed/timeout responses — assign zero scores
        return Score(
            test_case_id=test_case.id,
            factual_consistency=0.0,
            relevance=0.0,
            completeness=0.0,
            safety=1.0,
            hallucination_detected=False,
            judge_reasoning=f"Not scored: execution status was '{result.status}'",
            scored_by="llm_judge",
        )

    return await scorer_router.score(test_case=test_case, result=result)


async def judge_agent_async(state: "EvalState") -> dict:
    """
    Score all (test_case, result) pairs concurrently.
    """
    from evaluation.scorer_router import ScorerRouter
    scorer = ScorerRouter()

    test_cases: list[TestCase] = state["test_cases"]
    results: list[Result] = state["execution_results"]

    # Build a lookup map: test_case_id → Result
    result_map = {r.test_case_id: r for r in results}

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGEMENTS)

    async def bounded_judge(tc: TestCase) -> Score:
        async with semaphore:
            result = result_map.get(tc.id)
            if result is None:
                return Score(
                    test_case_id=tc.id,
                    factual_consistency=0.0,
                    relevance=0.0,
                    completeness=0.0,
                    judge_reasoning="No result found for this test case.",
                )
            return await _judge_one(tc, result, scorer)

    scores = await asyncio.gather(*[bounded_judge(tc) for tc in test_cases])

    logger.info(
        f"[Judge] Scored {len(scores)} cases. "
        f"Hallucinations: {sum(1 for s in scores if s.hallucination_detected)}"
    )
    return {"scores": list(scores)}


def judge_agent(state: "EvalState") -> dict:
    """Sync wrapper for LangGraph."""
    return asyncio.run(judge_agent_async(state))
