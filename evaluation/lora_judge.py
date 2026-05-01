"""
evaluation/lora_judge.py — Fine-tuned LoRA Classifier (Phase 4 stub)

WHY THIS FILE EXISTS:
    In Phase 4, we fine-tune a DeBERTa-v3 model with LoRA for multi-label
    classification.  This classifier replaces ~80% of LLM-as-Judge calls with
    a fast, cheap local inference that runs in milliseconds vs seconds.

    IN PHASE 1: This is a stub that always returns confidence=0.0, which tells
    the scorer_router to always use the LLM judge instead.

    RELATIONSHIP TO OTHER FILES:
    ┌─ evaluation/lora_judge.py ──────────────────────────────────────────────┐
    │  Called by:  evaluation/scorer_router.py                               │
    │  Returns:    (Score, confidence: float)                                │
    │  Weights:    models/artifacts/ (gitignored, not yet trained)           │
    │  Training:   models/training/train.py (Phase 4)                       │
    └─────────────────────────────────────────────────────────────────────────┘

    PHASE 4 IMPLEMENTATION PLAN:
    Base model:   microsoft/deberta-v3-base
    Task:         Multi-label binary classification
    Labels:       [factually_consistent, relevant, complete, safe]
    Training set: TruthfulQA + HaluEval + your own labeled outputs
    Method:       LoRA (PEFT) — fine-tunes <1% of params, fast training
    Threshold:    confidence >= 0.85 → use LoRA score (skip LLM judge)

    WHY DEBERTA:
    DeBERTa uses disentangled attention (separate position + content matrices)
    which gives it better text understanding than BERT/RoBERTa at the same size.
    deberta-v3-base is ~180M params — fast inference, low VRAM.
"""

import logging
from typing import Optional, Tuple

from api.schemas.job import Score, TestCase, Result

logger = logging.getLogger(__name__)

# Set to True once you've trained and saved the LoRA weights in Phase 4
LORA_MODEL_AVAILABLE = False


class LoRAJudge:
    """
    Fast local classifier for scoring.

    Phase 1: stub — always returns confidence=0.0 so scorer_router falls
             through to the LLM judge.
    Phase 4: loads fine-tuned DeBERTa LoRA weights and runs actual inference.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None

        if LORA_MODEL_AVAILABLE and model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load fine-tuned LoRA weights. Only runs in Phase 4."""
        try:
            # Phase 4 implementation:
            # from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # from peft import PeftModel
            # base = AutoModelForSequenceClassification.from_pretrained(
            #     "microsoft/deberta-v3-base", num_labels=4
            # )
            # self.model = PeftModel.from_pretrained(base, model_path)
            # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            # self.model.eval()
            logger.info(f"[LoRAJudge] Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"[LoRAJudge] Failed to load model: {e}")

    async def score(self, test_case: TestCase, result: Result) -> Tuple[Score, float]:
        """
        Score a (question, response) pair.

        Returns (Score, confidence) where confidence ∈ [0.0, 1.0].
        scorer_router uses confidence to decide whether to trust this score
        or escalate to the LLM judge.

        Phase 1: returns confidence=0.0 (always escalate to LLM judge).
        Phase 4: returns real scores with calibrated confidence.
        """
        if not self.model:
            # Phase 1 stub: confidence=0.0 means "I don't know, use LLM judge"
            return (
                Score(
                    test_case_id=test_case.id,
                    factual_consistency=0.5,
                    relevance=0.5,
                    completeness=0.5,
                    safety=1.0,
                    scored_by="lora",
                ),
                0.0,  # confidence
            )

        # Phase 4 implementation:
        # tokens = self.tokenizer(
        #     f"Question: {test_case.question}\nAnswer: {result.response_text}",
        #     return_tensors="pt", truncation=True, max_length=512
        # )
        # with torch.no_grad():
        #     logits = self.model(**tokens).logits
        # probs = torch.sigmoid(logits).squeeze().tolist()
        # confidence = min(probs)   # worst-case confidence across all labels
        # return Score(...), confidence
        raise NotImplementedError("Phase 4 not yet implemented")
