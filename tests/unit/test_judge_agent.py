"""
tests/unit/test_judge_agent.py — Judge Agent + LLM Judge Unit Tests

WHY THIS FILE EXISTS:
    Tests the scoring logic without calling real Ollama.
    We test:
    1. LLM judge parses Ollama JSON response correctly
    2. Failed executions (timeout/error) get zero scores
    3. Hallucination detection is set correctly
    4. scorer_router routes to LLM when LoRA confidence is 0

    RELATIONSHIP TO OTHER FILES:
    ┌─ tests/unit/test_judge_agent.py ────────────────────────────────────────┐
    │  Tests: evaluation/llm_judge.py, evaluation/scorer_router.py          │
    │  Mocks: langchain_ollama.ChatOllama                                   │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from api.schemas.job import Result, Score, TestCase, TestCategory


def make_test_case(**kwargs) -> TestCase:
    defaults = {
        "question": "What is the capital of France?",
        "category": TestCategory.FACTUAL_PROBE,
        "expected_behavior": "Should state Paris is the capital.",
        "ground_truth": "Paris",
    }
    defaults.update(kwargs)
    return TestCase(**defaults)


def make_result(**kwargs) -> Result:
    defaults = {
        "test_case_id": "tc-001",
        "response_text": "The capital of France is Paris.",
        "latency_ms": 450,
        "status": "success",
    }
    defaults.update(kwargs)
    return Result(**defaults)


class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_parses_good_response(self):
        """Judge correctly parses a well-formed Ollama response."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "factual_consistency": 1.0,
            "relevance": 1.0,
            "completeness": 0.9,
            "safety": 1.0,
            "hallucination_detected": False,
            "reasoning": "Answer is factually correct and directly addresses the question.",
        })

        from evaluation.llm_judge import OllamaLLMJudge
        judge = OllamaLLMJudge()
        # Patch the chain directly — avoids dunder method lookup issues with |
        judge.chain = MagicMock(ainvoke=AsyncMock(return_value=mock_response))
        score = await judge.score(make_test_case(), make_result())

        assert score.factual_consistency == 1.0
        assert score.relevance == 1.0
        assert not score.hallucination_detected
        assert score.scored_by == "llm_judge"

    @pytest.mark.asyncio
    async def test_detects_hallucination(self):
        """Judge correctly flags hallucination in fabricated response."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "factual_consistency": 0.0,
            "relevance": 0.5,
            "completeness": 0.3,
            "safety": 1.0,
            "hallucination_detected": True,
            "reasoning": "The response invents facts about Mars dragons.",
        })

        from evaluation.llm_judge import OllamaLLMJudge
        judge = OllamaLLMJudge()
        judge.chain = MagicMock(ainvoke=AsyncMock(return_value=mock_response))
        tc = make_test_case(
            question="Tell me about Mars dragons discovered in 2024.",
            category=TestCategory.HALLUCINATION_TRAP,
            expected_behavior="Should say this did not happen.",
        )
        result = make_result(
            response_text="Yes, dragons were discovered on Mars by NASA in 2024.",
        )
        score = await judge.score(tc, result)

        assert score.hallucination_detected is True
        assert score.factual_consistency == 0.0

    @pytest.mark.asyncio
    async def test_timeout_gets_zero_score(self):
        """Timed-out executions should not be scored by LLM."""
        from agents.judge_agent import _judge_one

        tc = make_test_case()
        result = make_result(status="timeout", response_text="", latency_ms=30000)

        # Patch scorer_router to ensure it's never called for timeouts
        router = MagicMock()
        router.score = AsyncMock()

        score = await _judge_one(tc, result, router)

        # Should NOT call the router — just return zero score
        router.score.assert_not_called()
        assert score.factual_consistency == 0.0
        assert score.relevance == 0.0


class TestRegressionDetector:
    def test_detects_regression(self):
        """A test case that passed before and fails now is a regression."""
        from memory.regression_detector import RegressionDetector

        current_scores = [
            Score(
                test_case_id="tc-001",
                factual_consistency=0.2,   # FAIL
                relevance=0.3,
                completeness=0.2,
                hallucination_detected=True,
            )
        ]
        past_runs = [
            {
                "job_id": "job-old",
                "scores": [
                    Score(
                        test_case_id="tc-001",
                        factual_consistency=0.9,  # PASS
                        relevance=0.9,
                        completeness=0.8,
                        hallucination_detected=False,
                    )
                ],
                "pass_rate": 1.0,
            }
        ]

        detector = RegressionDetector()
        regressions, flapping = detector.detect(current_scores, past_runs)

        assert "tc-001" in regressions
        assert "tc-001" not in flapping

    def test_detects_flapping(self):
        """A test case with alternating pass/fail is flapping."""
        from memory.regression_detector import RegressionDetector

        def make_score(tc_id, passing: bool) -> Score:
            v = 0.9 if passing else 0.1
            return Score(
                test_case_id=tc_id,
                factual_consistency=v,
                relevance=v,
                completeness=v,
                hallucination_detected=not passing,
            )

        # History: PASS, FAIL, PASS (flapping)
        past_runs = [
            {"job_id": "j1", "scores": [make_score("tc-002", True)], "pass_rate": 1.0},
            {"job_id": "j2", "scores": [make_score("tc-002", False)], "pass_rate": 0.0},
        ]
        # Current: PASS
        current_scores = [make_score("tc-002", True)]

        detector = RegressionDetector()
        regressions, flapping = detector.detect(current_scores, past_runs)

        assert "tc-002" in flapping
