"""
run_eval.py — Phase 1 CLI Runner

WHY THIS FILE EXISTS:
    This is the Phase 1 milestone entry point.  Run an eval directly from the
    command line — no FastAPI server, no Redis, no Docker needed.

    It drives the LangGraph orchestrator directly (bypassing the HTTP API and
    ARQ worker) so you can validate the full pipeline on your machine right now.

    RELATIONSHIP TO OTHER FILES:
    ┌─ run_eval.py ───────────────────────────────────────────────────────────┐
    │  Calls:   agents/orchestrator.py::build_graph() directly              │
    │  Skips:   api/, worker/, Redis (Phase 1 shortcut)                     │
    │  Uses:    In-memory stub for memory_agent (no DB required)            │
    │  Outputs: EvalReport printed as JSON + saved to report_{job_id}.json  │
    └─────────────────────────────────────────────────────────────────────────┘

    USAGE:
        # Evaluate a local target (must be running)
        python run_eval.py \\
          --endpoint http://localhost:8001/query \\
          --name "My RAG App" \\
          --num-cases 5

        # With a custom request template
        python run_eval.py \\
          --endpoint http://localhost:8001/chat \\
          --template '{"messages": [{"role": "user", "content": "__QUESTION__"}]}' \\
          --response-path "$.reply"

    PREREQUISITES:
        1. Ollama running: ollama serve
        2. Model available: ollama pull gemma4:26b
        3. (Optional) Target system running

    OUTPUT:
        Prints progress as test cases complete.
        Saves final report to ./report_{job_id}.json
"""

import argparse
import asyncio
import json
import sys
import uuid
import logging
from datetime import datetime

# Force UTF-8 output on Windows — prevents UnicodeEncodeError from emoji chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EvalForge — CLI eval runner (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py --endpoint http://localhost:8001/query --num-cases 5
  python run_eval.py --endpoint http://localhost:8001/query --name "RAG App" --num-cases 10 --domain finance
        """
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="HTTP endpoint of the target system",
    )
    parser.add_argument(
        "--name", default="Target System",
        help="Human-readable name for the target",
    )
    parser.add_argument(
        "--domain", default="general",
        help="Domain/topic area (e.g. finance, healthcare, legal)",
    )
    parser.add_argument(
        "--num-cases", type=int, default=5,
        help="Number of test cases to generate (default: 5)",
    )
    parser.add_argument(
        "--template", default='{"query": "__QUESTION__"}',
        help='JSON request template. Use __QUESTION__ as placeholder.',
    )
    parser.add_argument(
        "--response-path", default="$.answer",
        help="JSONPath to extract answer from response (default: $.answer)",
    )
    parser.add_argument(
        "--auth", default=None,
        help="Authorization header value (e.g. 'Bearer sk-...')",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file for report JSON (default: report_{job_id}.json)",
    )
    return parser.parse_args()


async def run(args):
    """Run a complete eval pipeline from CLI."""
    job_id = f"cli-{str(uuid.uuid4())[:8]}"

    target_config = {
        "name": args.name,
        "endpoint": args.endpoint,
        "domain": args.domain,
        "request_template": json.loads(args.template),
        "response_path": args.response_path,
        "timeout_seconds": 30,
    }
    if args.auth:
        target_config["auth_header"] = args.auth

    print(f"\n{'='*60}")
    print(f"  EvalForge — Evaluation Run")
    print(f"  Job ID:   {job_id}")
    print(f"  Target:   {args.name}")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Cases:    {args.num_cases}")
    print(f"  Model:    Ollama gemma4:26b (local)")
    print(f"{'='*60}\n")

    # Check if Ollama is running before starting
    try:
        import httpx
        from config import settings
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            if settings.OLLAMA_MODEL not in models:
                print(f"⚠️  WARNING: Model '{settings.OLLAMA_MODEL}' not found in Ollama.")
                print(f"   Run: ollama pull {settings.OLLAMA_MODEL}")
                print()
    except Exception:
        print("⚠️  WARNING: Cannot connect to Ollama at localhost:11434")
        print("   Run: ollama serve")
        print()

    # Build and run the graph
    from agents.orchestrator import build_graph, EvalState

    # Use in-memory stub for memory (skip PostgreSQL in Phase 1 CLI mode)
    _patch_memory_for_cli()

    graph = build_graph()
    compiled = graph.compile()   # No checkpointer for CLI mode

    initial_state: EvalState = {
        "job_id": job_id,
        "target_id": "cli-target",
        "target_config": target_config,
        "num_test_cases": args.num_cases,
        "seed_questions": [],
        "human_review_required": False,
        "approved": False,
    }

    print("📋 Phase 1: Generating test cases...")
    start = datetime.now()

    try:
        final_state = await compiled.ainvoke(initial_state)
        elapsed = (datetime.now() - start).total_seconds()

        report = final_state.get("report")
        if not report:
            print("❌ No report generated.")
            return

        # ── Print summary ──────────────────────────────────────────────────
        badge_emoji = {"RELIABLE": "✅", "NEEDS_IMPROVEMENT": "⚠️", "UNRELIABLE": "❌"}
        print(f"\n{'='*60}")
        print(f"  EVAL COMPLETE  ({elapsed:.1f}s)")
        print(f"{'='*60}")
        print(f"  {badge_emoji.get(report.badge, '?')} Badge:              {report.badge}")
        print(f"  📊 Reliability Score: {report.overall_reliability_score:.1f}/100")
        print(f"  ✓  Pass Rate:         {report.pass_rate:.0%}")
        print(f"  🧪 Test Cases:        {report.total_test_cases}")
        print(f"  🌀 Hallucinations:    {report.hallucination_rate:.0%}")
        print(f"  ⚡ Avg Latency:       {report.avg_latency_ms:.0f}ms")
        print(f"{'='*60}")

        if report.category_breakdown:
            print("\n📂 Category Breakdown:")
            for cat, stats in report.category_breakdown.items():
                print(f"   {cat:25s}  pass={stats.get('pass_rate', 0):.0%}")

        if report.recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"   {i}. {rec}")

        if report.top_failures:
            print(f"\n❌ Top Failures ({len(report.top_failures)}):")
            for f in report.top_failures[:3]:
                print(f"   [{f.category}] {f.question[:60]}")
                if f.judge_reasoning:
                    print(f"   → {f.judge_reasoning[:80]}")

        # ── Save report ────────────────────────────────────────────────────
        output_file = args.output or f"report_{job_id}.json"
        with open(output_file, "w") as fp:
            json.dump(report.model_dump(mode="json"), fp, indent=2, default=str)
        print(f"\n💾 Full report saved to: {output_file}")
        print()

    except Exception as e:
        print(f"\n❌ Eval failed: {e}")
        raise


def _patch_memory_for_cli():
    """
    Patch the memory agent to skip PostgreSQL in CLI mode.
    Uses in-memory stub so Phase 1 works without a running database.
    """
    import agents.memory_agent as mem_module
    from api.schemas.report import MemoryContext

    async def stub_memory_agent_async(state):
        """CLI stub: just return empty memory context (no DB)."""
        return {
            "memory_context": MemoryContext(
                target_id=state.get("target_id", "cli"),
                total_past_runs=0,
            )
        }

    def stub_memory_agent(state):
        return asyncio.run(stub_memory_agent_async(state))

    mem_module.memory_agent = stub_memory_agent


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
