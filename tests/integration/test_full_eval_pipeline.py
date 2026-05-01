"""
tests/integration/test_full_eval_pipeline.py — Full Pipeline Integration Test

WHY THIS FILE EXISTS:
    This test runs the ENTIRE LangGraph pipeline end-to-end:
    TestGen → Execute → Judge → Memory → Report

    It uses:
    - mock_target_server  for the HTTP target (no real LLM app needed)
    - Mocked Ollama       for TestGen + Judge (no Ollama needed)
    - In-memory state     no real PostgreSQL (memory_agent skips DB)

    MILESTONE TEST:
    This is the Phase 1 milestone test from the tech spec:
    "Run EvalForge against your RAG Stock Analyzer from CLI, get a report"

    If this test passes, Phase 1 is functionally complete.

    RELATIONSHIP TO OTHER FILES:
    ┌─ tests/integration/test_full_eval_pipeline.py ──────────────────────────┐
    │  Uses:    tests/fixtures/mock_target_server.py                         │
    │  Calls:   agents/orchestrator.py (indirectly via run_eval_job)        │
    │  Mocks:   All Ollama calls + DB operations                            │
    └─────────────────────────────────────────────────────────────────────────┘
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


def make_mock_testgen_response():
    """Minimal test cases for the full pipeline."""
    return json.dumps({
        "test_cases": [
            {
                "question": "What is the capital of France?",
                "category": "factual_probe",
                "expected_behavior": "Should say Paris.",
                "ground_truth": "Paris",
                "tags": ["geography"],
                "source": "generated",
            },
            {
                "question": "Tell me about dragons on Mars.",
                "category": "hallucination_trap",
                "expected_behavior": "Should say this did not happen.",
                "ground_truth": None,
                "tags": ["hallucination"],
                "source": "generated",
            },
        ]
    })


def make_mock_judge_response(hallucination: bool = False):
    return json.dumps({
        "factual_consistency": 0.2 if hallucination else 0.9,
        "relevance": 0.5 if hallucination else 0.9,
        "completeness": 0.3 if hallucination else 0.8,
        "safety": 1.0,
        "hallucination_detected": hallucination,
        "reasoning": "Test reasoning.",
    })


def make_mock_recommendation_response():
    return json.dumps([
        "Add retrieval grounding to reduce hallucinations.",
        "Test adversarial inputs regularly.",
    ])


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_returns_report(mocker):
    """
    Full pipeline integration test.

    Mocks Ollama but uses real mock HTTP target server.
    Verifies that the pipeline produces an EvalReport with a badge.
    """
    from tests.fixtures.mock_target_server import mock_target_server

    # ── Mock Ollama responses ──────────────────────────────────────────────
    call_count = 0

    async def mock_ollama_invoke(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = MagicMock()

        # First call: testgen
        if call_count == 1:
            mock_response.content = make_mock_testgen_response()
        # Calls 2-3: judge (one per test case)
        elif call_count <= 3:
            is_hallucination = call_count == 3  # second test case is hallucination_trap
            mock_response.content = make_mock_judge_response(hallucination=is_hallucination)
        # Final call: report recommendations
        else:
            mock_response.content = make_mock_recommendation_response()

        return mock_response

    mocker.patch(
        "langchain_ollama.ChatOllama",
        return_value=MagicMock(
            ainvoke=AsyncMock(side_effect=mock_ollama_invoke),
            invoke=MagicMock(side_effect=lambda *a, **kw: MagicMock(
                content=make_mock_testgen_response()
            )),
        ),
    )

    # ── Mock DB operations ─────────────────────────────────────────────────
    mocker.patch("memory.store.EvalStore.save_results", new_callable=AsyncMock)
    mocker.patch(
        "memory.store.EvalStore.get_past_runs",
        new_callable=AsyncMock,
        return_value=[],
    )
    mocker.patch(
        "memory.store.EvalStore.get_category_scores",
        new_callable=AsyncMock,
        return_value={},
    )
    mocker.patch("memory.store.EvalStore.save_report", new_callable=AsyncMock)

    # ── Mock LangGraph checkpointer ────────────────────────────────────────
    from unittest.mock import AsyncMock as AM

    mock_checkpointer = MagicMock()
    mock_checkpointer.__aenter__ = AM(return_value=mock_checkpointer)
    mock_checkpointer.__aexit__ = AM(return_value=None)
    mocker.patch(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string",
        new_callable=AsyncMock,
        return_value=mock_checkpointer,
    )

    # ── Run mock target server + pipeline ──────────────────────────────────
    async with mock_target_server() as base_url:
        target_config = {
            "name": "Mock RAG App",
            "endpoint": f"{base_url}/query",
            "request_template": {"query": "__QUESTION__"},
            "response_path": "$.answer",
            "timeout_seconds": 10,
        }

        # Run the graph directly (skip ARQ for integration test)
        from agents.orchestrator import build_graph
        from agents.orchestrator import EvalState

        graph = build_graph()
        compiled = graph.compile()

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

        # ── Assertions ────────────────────────────────────────────────────
        assert report is not None, "Pipeline should produce a report"
        assert report.total_test_cases >= 1
        assert report.badge in ("RELIABLE", "NEEDS_IMPROVEMENT", "UNRELIABLE")
        assert 0.0 <= report.overall_reliability_score <= 100.0
        assert len(report.recommendations) >= 1
        assert report.job_id == "integration-test-job"
