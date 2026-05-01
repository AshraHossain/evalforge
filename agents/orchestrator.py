"""
agents/orchestrator.py — LangGraph StateGraph Orchestrator

WHY THIS FILE EXISTS:
    This is the central nervous system of EvalForge.  It:
    1. Defines EvalState — the shared dict passed between all agent nodes
    2. Registers each agent as a named graph node
    3. Wires edges (control flow) between nodes
    4. Sets up PostgreSQL checkpointing for long-running / HITL jobs
    5. Exposes `run_eval_job()` — the single entrypoint called by the worker

    RELATIONSHIP TO OTHER FILES:
    ┌─ agents/orchestrator.py ────────────────────────────────────────────────┐
    │  Imports ALL agents:                                                    │
    │    testgen_agent.py, execution_agent.py, judge_agent.py,              │
    │    memory_agent.py, report_agent.py, hitl_node.py                     │
    │  Uses schemas from:                                                    │
    │    api/schemas/job.py    (TestCase, Result, Score)                    │
    │    api/schemas/report.py (EvalReport, MemoryContext)                  │
    │    api/schemas/target.py (TargetConfig)                               │
    │  Called by: worker/tasks.py (async job runner)                        │
    └─────────────────────────────────────────────────────────────────────────┘

    THE GRAPH FLOW:
    ┌──────────┐   ┌──────────┐   ┌───────┐   ┌────────┐
    │ testgen  │──▶│ execute  │──▶│ judge │──▶│ memory │
    └──────────┘   └──────────┘   └───────┘   └───┬────┘
                                                    │
                                     ┌──────────────┼──────────────┐
                                     │ hallucination │              │
                                     │ rate > thresh │  auto path   │
                                     ▼              ▼              │
                                ┌──────────┐  ┌────────┐          │
                                │hitl_gate │  │ report │◀─────────┘
                                └────┬─────┘  └────────┘
                                     │ (after approval)
                                     └──────▶ report

    CHECKPOINTING:
    PostgreSQLSaver persists EvalState after every node completes.  This means:
    - If the worker crashes mid-run, the job resumes from the last completed node
    - HITL jobs survive server restarts — state is in Postgres, not memory
    - You can replay a specific node for debugging
"""

import logging
from typing import Annotated, Any, Literal, Optional
from uuid import UUID

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from api.schemas.job import Result, Score, TestCase
from api.schemas.report import EvalReport, MemoryContext
from config import settings

logger = logging.getLogger(__name__)


# ── EvalState ──────────────────────────────────────────────────────────────────
# This TypedDict is the "bus" that all nodes read from and write to.
# LangGraph passes the full state to each node function.
# Each node returns a dict with only the keys it modified — LangGraph merges
# the returned dict into the state (reducer pattern).
class EvalState(TypedDict, total=False):
    # ── Inputs (set before graph runs) ────────────────────────────────────
    job_id: str
    target_id: str
    target_config: dict          # TargetConfig serialised to dict
    num_test_cases: int
    seed_questions: list[str]

    # ── TestGen output ─────────────────────────────────────────────────────
    test_cases: list[TestCase]

    # ── Execution output ───────────────────────────────────────────────────
    execution_results: list[Result]

    # ── Judge output ───────────────────────────────────────────────────────
    scores: list[Score]

    # ── Memory output ──────────────────────────────────────────────────────
    memory_context: MemoryContext

    # ── Report output ──────────────────────────────────────────────────────
    report: EvalReport

    # ── HITL control ───────────────────────────────────────────────────────
    human_review_required: bool
    approved: bool               # Set to True by POST /jobs/{id}/approve


def route_hitl(state: EvalState) -> Literal["review", "auto"]:
    """
    Conditional edge function called after the memory node.

    Decides whether human review is required.  Currently based on
    hallucination rate, but you could add: regression spike, safety failures,
    first-run-ever (always review), etc.

    Returns "review" → graph goes to hitl_gate node (pauses)
    Returns "auto"   → graph goes directly to report node
    """
    scores = state.get("scores", [])
    if not scores:
        return "auto"

    hallucination_rate = sum(1 for s in scores if s.hallucination_detected) / len(scores)

    if hallucination_rate > settings.HITL_HALLUCINATION_THRESHOLD:
        logger.warning(
            f"[Router] Hallucination rate {hallucination_rate:.0%} > "
            f"threshold {settings.HITL_HALLUCINATION_THRESHOLD:.0%}. "
            "Routing to HITL gate."
        )
        return "review"

    # Also check if an explicit flag was set (e.g. by the memory agent)
    if state.get("human_review_required", False):
        return "review"

    return "auto"


def build_graph():
    """
    Construct the LangGraph StateGraph.

    Called once at application startup.  The compiled graph is thread-safe
    and can be invoked concurrently for multiple jobs.
    """
    # Import here to avoid circular imports at module load time
    from agents.testgen_agent import testgen_agent
    from agents.execution_agent import execution_agent
    from agents.judge_agent import judge_agent
    from agents.memory_agent import memory_agent
    from agents.report_agent import report_agent
    from agents.hitl_node import human_review_node

    graph = StateGraph(EvalState)

    # ── Register nodes ─────────────────────────────────────────────────────
    # Each node is a Python function: (EvalState) → dict
    graph.add_node("testgen",   testgen_agent)
    graph.add_node("execute",   execution_agent)
    graph.add_node("judge",     judge_agent)
    graph.add_node("memory",    memory_agent)
    graph.add_node("hitl_gate", human_review_node)
    graph.add_node("report",    report_agent)

    # ── Wire edges ─────────────────────────────────────────────────────────
    graph.set_entry_point("testgen")
    graph.add_edge("testgen",  "execute")
    graph.add_edge("execute",  "judge")
    graph.add_edge("judge",    "memory")

    # Conditional: after memory, either go to HITL gate or directly to report
    graph.add_conditional_edges(
        "memory",
        route_hitl,
        {"review": "hitl_gate", "auto": "report"},
    )

    # After HITL approval, proceed to report
    graph.add_edge("hitl_gate", "report")
    graph.add_edge("report",    END)

    return graph


async def run_eval_job(
    job_id: str,
    target_id: str,
    target_config: dict,
    num_test_cases: int = 10,
    seed_questions: Optional[list[str]] = None,
    db_url: Optional[str] = None,
) -> EvalReport:
    """
    Main entrypoint called by worker/tasks.py.

    Sets up the PostgreSQL checkpointer and invokes the graph.
    Returns the completed EvalReport.

    CHECKPOINTER SETUP:
    AsyncPostgresSaver connects to PostgreSQL and creates a `checkpoints` table
    (if not exists) to store graph state after each node.  thread_id is the
    job_id — each job gets its own checkpoint namespace.
    """
    # Lazy import — keeps CLI mode (which has no libpq) from failing at import time
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    graph = build_graph()

    db_url = db_url or settings.DATABASE_URL.replace("+asyncpg", "")

    async with await AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        # interrupt_before=["hitl_gate"] tells LangGraph to checkpoint and
        # pause execution when it reaches the hitl_gate node.
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["hitl_gate"],
        )

        initial_state: EvalState = {
            "job_id": job_id,
            "target_id": target_id,
            "target_config": target_config,
            "num_test_cases": num_test_cases,
            "seed_questions": seed_questions or [],
            "human_review_required": False,
            "approved": False,
        }

        config = {"configurable": {"thread_id": job_id}}

        logger.info(f"[Orchestrator] Starting eval job {job_id}")
        final_state = await compiled.ainvoke(initial_state, config=config)

        report: EvalReport = final_state.get("report")
        logger.info(
            f"[Orchestrator] Job {job_id} complete. "
            f"Badge={report.badge if report else 'N/A'}"
        )
        return report


async def resume_eval_job(job_id: str, db_url: Optional[str] = None) -> EvalReport:
    """
    Resume a HITL-paused job after human approval.

    Called by POST /api/v1/jobs/{id}/approve.
    Resumes from the hitl_gate checkpoint by providing approved=True in state.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # lazy

    graph = build_graph()
    db_url = db_url or settings.DATABASE_URL.replace("+asyncpg", "")

    async with await AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["hitl_gate"],
        )

        config = {"configurable": {"thread_id": job_id}}

        # Update state with approval flag before resuming
        await compiled.aupdate_state(
            config,
            {"approved": True},
        )

        # Resume from checkpoint — None input means "continue from where we paused"
        final_state = await compiled.ainvoke(None, config=config)
        return final_state.get("report")
