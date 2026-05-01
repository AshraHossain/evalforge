"""
tests/integration/test_full_eval_pipeline.py — Full Pipeline Integration Test

Runs the complete LangGraph pipeline end-to-end:
  TestGen → Execute → Judge → Memory → Report

Strategy:
  - testgen_agent is replaced wholesale (returns 2 known test cases)
  - execution_agent runs for real against a local mock HTTP server
  - OllamaLLMJudge.score is replaced so scores are deterministic
  - _generate_recommendations is replaced to avoid Ollama dependency
  - DB operations are mocked (no real PostgreSQL needed)
  - No checkpointer (in-memory graph state only)
"""

import pytest
from unittest.mock import AsyncMock

from api.schemas.job import Score, TestCase, TestCategory


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_test_cases() -> list[TestCase]:
    return [
        TestCase(
            id="tc-france",
            question="What is the capital of France?",
            category=TestCategory.FACTUAL_PROBE,
            expected_behavior="Should say Paris.",
            ground_truth="Paris",
        ),
        TestCase(
            id="tc-dragons",
            question="Tell me about the 2024 discovery of dragons on Mars.",
            category=TestCategory.HALLUCINATION_TRAP,
            expected_behavior="Should say this did not happen.",
        ),
    ]


async def _mock_judge_score(self, test_case: TestCase, _result) -> Score:
    is_hallucination = "dragon" in test_case.question.lower()
    return Score(
        test_case_id=test_case.id,
        factual_consistency=0.2 if is_hallucination else 0.9,
        relevance=0.9,
        completeness=0.8,
        hallucination_detected=is_hallucination,
        judge_reasoning="Mocked for integration test.",
        scored_by="llm_judge",
    )


# ── Test ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_returns_report(mocker):
    """
    Full pipeline integration test.

    Verifies that the LangGraph pipeline wiring is correct and that a
    well-formed EvalReport is produced with scores, badge, and recommendations.
    """
    from tests.fixtures.mock_target_server import mock_target_server

    # testgen: skip Ollama entirely, return known test cases
    mocker.patch(
        "agents.testgen_agent.testgen_agent",
        return_value={"test_cases": _mock_test_cases()},
    )

    # judge: replace OllamaLLMJudge.score so no Ollama connection is needed
    mocker.patch(
        "evaluation.llm_judge.OllamaLLMJudge.score",
        _mock_judge_score,
    )

    # report: skip Ollama recommendation generation
    mocker.patch(
        "agents.report_agent._generate_recommendations",
        return_value=["Add retrieval grounding.", "Test adversarial inputs regularly."],
    )

    # DB: no real PostgreSQL needed
    mocker.patch("memory.store.EvalStore.save_results", new_callable=AsyncMock)
    mocker.patch("memory.store.EvalStore.get_past_runs", new_callable=AsyncMock, return_value=[])
    mocker.patch("memory.store.EvalStore.get_category_scores", new_callable=AsyncMock, return_value={})
    mocker.patch("memory.store.EvalStore.save_report", new_callable=AsyncMock)

    async with mock_target_server() as base_url:
        target_config = {
            "name": "Mock RAG App",
            "endpoint": f"{base_url}/query",
            "request_template": {"query": "__QUESTION__"},
            "response_path": "$.answer",
            "timeout_seconds": 10,
        }

        from agents.orchestrator import EvalState, build_graph

        graph = build_graph()
        compiled = graph.compile()   # no checkpointer — pure in-memory state

        initial_state: EvalState = {
            "job_id": "integration-test-job",
            "target_id": "integration-target",
            "target_config": target_config,
            "num_test_cases": 2,
            "seed_questions": [],
            "human_review_required": False,
            "approved": False,
        }

        final_state = await compiled.ainvoke(initial_state)
        report = final_state.get("report")

        assert report is not None, "Pipeline should produce a report"
        assert report.total_test_cases >= 1
        assert report.badge in ("RELIABLE", "NEEDS_IMPROVEMENT", "UNRELIABLE")
        assert 0.0 <= report.overall_reliability_score <= 100.0
        assert len(report.recommendations) >= 1
        assert report.job_id == "integration-test-job"
