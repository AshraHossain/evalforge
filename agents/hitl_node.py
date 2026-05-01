"""
agents/hitl_node.py — Human-in-the-Loop (HITL) Gate Node

WHY THIS FILE EXISTS:
    Some eval runs surface results so alarming (e.g. 40%+ hallucination rate)
    that we don't want to auto-generate a report — we want a human to review
    the raw results first and decide whether to continue or abort.

    LangGraph supports this via a special "interrupt" mechanism: the graph
    pauses at this node, serialises its state to PostgreSQL (via the
    LangGraph checkpoint system), and waits.  When a human calls
    POST /api/v1/jobs/{id}/approve, the graph resumes from exactly this point.

    RELATIONSHIP TO OTHER FILES:
    ┌─ agents/hitl_node.py ───────────────────────────────────────────────────┐
    │  Called by:  agents/orchestrator.py via conditional edge "route_hitl"  │
    │  Triggered when: hallucination_rate > settings.HITL_HALLUCINATION_THRESHOLD │
    │  Resume path: POST /api/v1/jobs/{id}/approve  (api/routers/jobs.py)   │
    │  After approval: → report node                                         │
    │  DB state:   eval_jobs.status = "hitl_pending" while paused           │
    └─────────────────────────────────────────────────────────────────────────┘

    HOW LANGGRAPH HITL WORKS:
    1. orchestrator.py adds `interrupt_before=["hitl_gate"]` to the graph compiler.
    2. When execution reaches hitl_gate, LangGraph raises NodeInterrupt.
    3. The ARQ worker catches NodeInterrupt, sets job status = hitl_pending.
    4. Human calls /approve → worker resumes graph with `approved=True` in state.
    5. Graph continues to report node.
"""

import logging
from typing import TYPE_CHECKING

from langgraph.errors import NodeInterrupt

if TYPE_CHECKING:
    from agents.orchestrator import EvalState

logger = logging.getLogger(__name__)


def human_review_node(state: "EvalState") -> dict:
    """
    HITL gate node.

    If this node is reached, it means the orchestrator's route_hitl() function
    decided human review is required.  We raise NodeInterrupt to pause the graph.

    LangGraph serialises the full EvalState to the PostgreSQL checkpointer at
    this point.  The state persists until the job is approved via the API.

    WHY RAISE NOT RETURN:
    Returning from this node would let the graph continue.  NodeInterrupt is
    LangGraph's mechanism to *pause* execution and persist state — it's not
    an error, it's a checkpoint.
    """
    job_id = state["job_id"]
    scores = state.get("scores", [])

    # Compute hallucination rate for the interrupt message
    if scores:
        hallucination_rate = sum(1 for s in scores if s.hallucination_detected) / len(scores)
    else:
        hallucination_rate = 0.0

    logger.warning(
        f"[HITL] Job {job_id} paused for human review. "
        f"Hallucination rate: {hallucination_rate:.0%}"
    )

    # This message is displayed in the UI / returned by GET /jobs/{id}
    raise NodeInterrupt(
        f"Job {job_id} requires human review. "
        f"Hallucination rate ({hallucination_rate:.0%}) exceeds threshold. "
        f"Call POST /api/v1/jobs/{job_id}/approve to continue."
    )
