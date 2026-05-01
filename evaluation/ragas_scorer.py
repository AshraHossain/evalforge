"""
evaluation/ragas_scorer.py — RAGAS Metrics for RAG Targets

WHY THIS FILE EXISTS:
    RAGAS is a framework specifically designed for evaluating RAG (Retrieval
    Augmented Generation) pipelines.  It computes metrics that require:
    - The question
    - The generated answer
    - The retrieved context documents
    - The ground truth answer (for some metrics)

    RAGAS METRICS COMPUTED:
    ┌───────────────────┬──────────────────────────────────────────────────────┐
    │ answer_relevancy  │ Is the answer relevant to the question?              │
    │ faithfulness      │ Is the answer grounded in the retrieved context?     │
    │ context_recall    │ Did retrieval return the context needed to answer?   │
    │ context_precision │ Is the retrieved context precise (low noise)?        │
    └───────────────────┴──────────────────────────────────────────────────────┘

    RELATIONSHIP TO OTHER FILES:
    ┌─ evaluation/ragas_scorer.py ────────────────────────────────────────────┐
    │  Called by:     agents/judge_agent.py (optional, only for RAG targets)  │
    │  Requires:      Target response must include 'contexts' field           │
    │  Returns:       dict of RAGAS metric scores                             │
    │  Uses Ollama:   RAGAS's LLM evaluator is replaced with ChatOllama      │
    └─────────────────────────────────────────────────────────────────────────┘

    OLLAMA INTEGRATION WITH RAGAS:
    RAGAS uses LangChain under the hood.  We provide it our ChatOllama instance
    so all RAGAS LLM calls also go to the local Ollama server — no external APIs.

    WHEN TO USE:
    Only call this scorer if the target system returns context documents
    alongside its answer.  For a simple chatbot with no retrieval, skip it.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGASScorer:
    """
    Wrapper around RAGAS that uses local Ollama for all LLM evaluations.
    """

    def __init__(self):
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.warning("[RAGAS] ragas package not installed. RAGAS scoring disabled.")
            return False

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Compute RAGAS metrics for one RAG response.

        Returns dict like:
          {
            "answer_relevancy": 0.85,
            "faithfulness": 0.92,
            "context_recall": 0.78,     # only if ground_truth provided
            "context_precision": 0.88,
          }
        """
        if not self._ragas_available:
            return {}

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from langchain_ollama import ChatOllama
            from ragas.llms import LangchainLLMWrapper
            from config import settings

            # Point RAGAS at local Ollama
            ollama_llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.0,
            )
            ragas_llm = LangchainLLMWrapper(ollama_llm)

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            metrics = [answer_relevancy, faithfulness, context_precision]
            if ground_truth:
                metrics.append(context_recall)

            # Pass Ollama LLM to each metric
            for metric in metrics:
                if hasattr(metric, "llm"):
                    metric.llm = ragas_llm

            result = evaluate(dataset, metrics=metrics)
            return dict(result)

        except Exception as e:
            logger.error(f"[RAGAS] Scoring failed: {e}")
            return {}
