"""
evaluation/lora_judge.py — Fine-tuned LoRA Classifier

Loads a DeBERTa-v3-base model with a LoRA adapter trained in models/training/train.py.
Returns (Score, confidence) where confidence drives the scorer_router decision:
  >= LORA_CONFIDENCE_THRESHOLD  → trust this score, skip LLM judge
  <  threshold                  → escalate to Ollama LLM judge

Confidence = min over labels of (how far the sigmoid prob is from 0.5),
normalised to [0, 1].  Any label near 0.5 → low overall confidence.

Activated by setting LORA_MODEL_AVAILABLE=true in .env after training.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

from api.schemas.job import Result, Score, TestCase
from config import settings

logger = logging.getLogger(__name__)

MAX_LENGTH = 512


class LoRAJudge:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self._device = None

        if settings.LORA_MODEL_AVAILABLE and model_path and Path(model_path).exists():
            self._load_model(model_path)
        elif settings.LORA_MODEL_AVAILABLE:
            logger.warning(
                f"[LoRAJudge] LORA_MODEL_AVAILABLE=true but path not found: {model_path}. "
                "Run models/training/train.py first."
            )

    def _load_model(self, model_path: str) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"[LoRAJudge] Loading adapter from {model_path} on {self._device}")

            base = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v3-base",
                num_labels=4,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )
            self.model = PeftModel.from_pretrained(base, model_path).to(self._device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("[LoRAJudge] Model loaded.")
        except Exception as e:
            logger.error(f"[LoRAJudge] Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def _infer(self, text: str) -> list[float]:
        import torch

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(self._device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
        if isinstance(probs, float):
            probs = [probs] * 4
        return probs

    async def score(self, test_case: TestCase, result: Result) -> Tuple[Score, float]:
        if not self.model:
            return (
                Score(
                    test_case_id=test_case.id,
                    factual_consistency=0.5,
                    relevance=0.5,
                    completeness=0.5,
                    safety=1.0,
                    scored_by="lora",
                ),
                0.0,
            )

        text = f"Question: {test_case.question}\nAnswer: {result.response_text or ''}"

        # Run CPU-bound inference off the event loop
        loop = asyncio.get_event_loop()
        probs = await loop.run_in_executor(None, self._infer, text)

        # Confidence: how far is each label from 0.5?  Normalise to [0, 1].
        # min() picks the most uncertain label — if any label is unsure, we escalate.
        raw_confidence = min(max(p, 1.0 - p) for p in probs)  # in [0.5, 1.0]
        confidence = (raw_confidence - 0.5) * 2.0              # in [0.0, 1.0]

        fc, rel, comp, safe = probs
        return (
            Score(
                test_case_id=test_case.id,
                factual_consistency=fc,
                relevance=rel,
                completeness=comp,
                safety=safe,
                hallucination_detected=fc < 0.5,
                scored_by="lora",
            ),
            confidence,
        )
