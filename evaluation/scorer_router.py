"""
evaluation/scorer_router.py — Scoring Strategy Router

WHY THIS FILE EXISTS:
    This is the decision point between the two scoring strategies.
    It hides the routing logic from the Judge Agent — the agent just calls
    `scorer.score()` and gets back a Score object.

    ROUTING LOGIC:
    1. Try LoRA classifier first (fast, free, local)
    2. If confidence >= LORA_CONFIDENCE_THRESHOLD → trust it, done
    3. If confidence < threshold → fall back to Ollama LLM judge
    4. Mark the score's `scored_by` field accordingly

    In Phase 1: LoRA always returns confidence=0.0, so every case goes to
    the LLM judge.  Swap in Phase 4 by setting LORA_MODEL_AVAILABLE=True.

    RELATIONSHIP TO OTHER FILES:
    ┌─ evaluation/scorer_router.py ───────────────────────────────────────────┐
    │  Called by:  agents/judge_agent.py                                     │
    │  Uses:       evaluation/lora_judge.py  (fast path)                    │
    │              evaluation/llm_judge.py   (fallback)                     │
    │  Returns:    api/schemas/job.Score                                     │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import logging

from api.schemas.job import Score, TestCase, Result
from config import settings

logger = logging.getLogger(__name__)


class ScorerRouter:
    """
    Routes scoring to LoRA or LLM judge based on LoRA confidence.

    Instantiated once per judge_agent invocation.
    """

    def __init__(self):
        from evaluation.lora_judge import LoRAJudge
        from evaluation.llm_judge import OllamaLLMJudge

        # Try loading LoRA model (returns stub if not available)
        self.lora = LoRAJudge(model_path="models/artifacts/lora_judge")
        self.llm_judge = OllamaLLMJudge()

    async def score(self, test_case: TestCase, result: Result) -> Score:
        """
        Score one (test_case, result) pair using the best available scorer.

        PHASE 1 FLOW:
          LoRA.score() → confidence=0.0 → always fall through to LLM judge

        PHASE 4 FLOW:
          LoRA.score() → confidence=0.92 → trust LoRA, return (no LLM call)
          LoRA.score() → confidence=0.60 → escalate to LLM judge
        """
        # ── Step 1: Try LoRA ───────────────────────────────────────────────
        lora_score, confidence = await self.lora.score(test_case, result)

        if confidence >= settings.LORA_CONFIDENCE_THRESHOLD:
            logger.debug(
                f"[Router] LoRA confident ({confidence:.2f}) for {test_case.id}"
            )
            lora_score.scored_by = "lora"
            return lora_score

        # ── Step 2: Fall back to Ollama LLM judge ─────────────────────────
        logger.debug(
            f"[Router] LoRA confidence {confidence:.2f} < "
            f"{settings.LORA_CONFIDENCE_THRESHOLD} → using LLM judge"
        )
        llm_score = await self.llm_judge.score(test_case, result)

        # If LoRA had some confidence, mark as hybrid
        if confidence > 0.0:
            llm_score.scored_by = "hybrid"

        return llm_score
