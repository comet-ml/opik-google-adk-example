"""Evaluate the support agent on the test set and log it as an Opik experiment.

    uv run evaluate.py                          # the default model
    uv run evaluate.py --model gemini-3.5-flash-lite

Each test case (eval_cases.py) runs in a fresh ADK session and is scored on:
  - expected_tools: did the agent call the tools this case needs?
  - correct_handoff: was the case handled by the right agent?
  - policy_safe_refunds: did it ever try a refund the policy forbids?
  - grounded_order_ids: did it only use order IDs the customer gave?
  - answer_relevance_metric: an LLM judge (Gemini) rates whether the reply
    answers the customer.

After the run it reads each case's cost back from Opik and prints a summary.
compare.py runs this for two models and prints them side by side.
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import opik
from opik.evaluation import evaluate
from opik.evaluation.models import LiteLLMChatModel

from eval_cases import TEST_CASES
from eval_metrics import (
    CorrectHandoff,
    ExpectedToolsCalled,
    GroundedOrderIds,
    JudgedAnswerRelevance,
    PolicySafeRefunds,
)
from support_agent.agent import MODEL, OPIK_PROJECT, build_agent
from support_agent.conversation import (
    check_credentials,
    make_runner,
    new_session,
    run_turn,
)

# Opik skips items that are already in the dataset, matching on content. If you
# edit a case, the old version stays too: delete it in the Opik UI or rename
# the dataset.
DATASET_NAME = "trailhead-support-cases"

# The judge stays fixed when you change the agent's model, so two experiments
# are scored the same way. LiteLLM model name.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini/gemini-3.8-flash")

METRIC_NAMES = [
    "expected_tools",
    "correct_handoff",
    "policy_safe_refunds",
    "grounded_order_ids",
    "answer_relevance_metric",
]

# A case "misses" on a metric when it scores below this. The code metrics are
# 0 or 1 (expected_tools can be a fraction). The LLM judge gives 0.9 to 0.95
# to answers that are fine, so those don't count as misses.
MISS_BELOW = 0.8


def make_task(agent):
    """Return the function Opik calls once per dataset item, bound to one agent."""

    def run_case(item: dict) -> dict:
        async def _run():
            runner = make_runner(agent)
            try:
                session_id = await new_session(runner, "eval-user", f"eval-{uuid.uuid4().hex[:8]}")
                return await run_turn(runner, "eval-user", session_id, item["input"], as_thread=False)
            finally:
                await runner.close()

        started = time.perf_counter()
        try:
            turn = asyncio.run(_run())
        except Exception as error:  # noqa: BLE001 (any failure, see below)
            # A case that still fails after the retries (a daily quota, say)
            # would otherwise abort the whole run. Record it and move on. The
            # custom metrics mark it as not scored rather than guess, and the
            # summary lists it as an error.
            return {
                "output": "",
                "tool_calls": [],
                "agents": [],
                "model_errors": [],
                "latency_s": time.perf_counter() - started,
                "trace_id": None,
                "span_id": None,
                "error": f"{type(error).__name__}: {error}",
            }
        # These keys are what the metrics' score() methods ask for by name.
        return {
            "output": turn.reply,
            "tool_calls": turn.tool_calls,
            "agents": turn.agents,
            # Unlike `error`, these are the model's own failures, so they are
            # scored like any other answer and listed in the summary.
            "model_errors": turn.model_errors,
            "latency_s": turn.latency_s,
            "trace_id": turn.trace_id,
            "span_id": turn.span_id,
            "error": None,
        }

    return run_case


def fetch_spans(client: opik.Opik, since: datetime, trace_ids: set[str]) -> dict[str, list]:
    """Read back every span logged since `since`, grouped by trace ID.

    One search for the whole run instead of one per case keeps this under
    Opik Cloud's rate limit. Spans show up a moment after flush, so poll until
    every trace we ran is there and the span count has stopped changing.
    """
    since_filter = f'start_time >= "{since.strftime("%Y-%m-%dT%H:%M:%SZ")}"'
    spans, previous = [], -1
    for _ in range(15):
        spans = client.search_spans(project_name=OPIK_PROJECT, filter_string=since_filter, max_results=10_000)
        all_there = trace_ids <= {span.trace_id for span in spans}
        if all_there and len(spans) == previous:
            break
        previous = len(spans)
        time.sleep(3)
    by_trace = defaultdict(list)
    for span in spans:
        by_trace[span.trace_id].append(span)
    return by_trace


def agent_cost(spans: list, turn_span_id: str | None) -> tuple[float | None, int]:
    """Total the cost and tokens of the agent's own LLM calls in one trace.

    Opik prices every Gemini span from its token usage. The evaluation trace
    also holds the judge's LLM call, so only spans under the customer_turn
    span count. The cost is None unless Opik priced every one of the agent's
    LLM calls, so a partial total is never reported as the whole.
    """
    children = defaultdict(list)
    for span in spans:
        children[span.parent_span_id].append(span)

    llm_spans = []
    stack = list(children[turn_span_id])  # top-level spans when turn_span_id is None
    while stack:
        span = stack.pop()
        stack.extend(children[span.id])
        if span.type == "llm":
            llm_spans.append(span)

    tokens = sum((span.usage or {}).get("total_tokens", 0) for span in llm_spans)
    if not llm_spans or any(span.total_estimated_cost is None for span in llm_spans):
        return None, tokens
    return sum(span.total_estimated_cost for span in llm_spans), tokens


def run_experiment(model: str = MODEL) -> dict:
    """Run the whole test set on one model, log it to Opik and return a summary."""
    client = opik.Opik(project_name=OPIK_PROJECT)
    dataset = client.get_or_create_dataset(DATASET_NAME)
    dataset.insert(TEST_CASES)

    started = datetime.now(UTC) - timedelta(seconds=5)
    judge = LiteLLMChatModel(
        model_name=JUDGE_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )
    try:
        result = evaluate(
            dataset=dataset,
            task=make_task(build_agent(model)),
            scoring_metrics=[
                ExpectedToolsCalled(),
                CorrectHandoff(),
                PolicySafeRefunds(),
                GroundedOrderIds(),
                JudgedAnswerRelevance(model=judge, require_context=False),
            ],
            # The model is in the experiment name and its config, so two runs
            # are easy to tell apart and compare in Opik.
            experiment_name_prefix=f"support-agent-{model}",
            experiment_config={
                "model": model,
                "judge_model": JUDGE_MODEL,
                "agents": ["support_agent", "refunds_agent"],
            },
            task_threads=2,
        )
    finally:
        opik.flush_tracker()
    trace_ids = {test.test_case.task_output["trace_id"] for test in result.test_results}
    trace_ids.discard(None)  # cases that errored before a trace existed
    return summarize(result, model, fetch_spans(client, started, trace_ids))


def summarize(result, model: str, spans_by_trace: dict[str, list]) -> dict:
    """Mean score per metric, plus cost, tokens and latency per case."""
    stats = result.aggregate_evaluation_scores().aggregated_scores
    cases = []
    for test in result.test_results:
        output = test.test_case.task_output
        cost, tokens = agent_cost(spans_by_trace.get(output["trace_id"], []), output["span_id"])
        # A judge call that failed is not the agent's fault, so it is left out
        # of the scores here (Opik leaves it out of its averages too).
        scored = [score for score in test.score_results if not score.scoring_failed]
        cases.append({
            "case": test.test_case.dataset_item_content.get("case", ""),
            "scores": {score.name: score.value for score in scored},
            "reasons": {score.name: score.reason for score in scored},
            "failed_scores": [score.name for score in test.score_results if score.scoring_failed],
            "error": output.get("error"),
            "model_errors": output.get("model_errors", []),
            "cost_usd": cost,
            "tokens": tokens,
            "latency_s": output["latency_s"],
        })
    ran = [case for case in cases if not case["error"]] or cases
    costs = [case["cost_usd"] for case in ran if case["cost_usd"] is not None]
    return {
        "model": model,
        "experiment": result.experiment_name,
        "scores": {name: stats[name].mean for name in METRIC_NAMES if name in stats},
        "cost_per_case_usd": statistics.mean(costs) if costs else None,
        "priced_cases": len(costs),
        "tokens_per_case": statistics.mean(case["tokens"] for case in ran),
        "median_latency_s": statistics.median(case["latency_s"] for case in ran),
        "errors": [(case["case"], case["error"]) for case in cases if case["error"]],
        "model_errors": [(case["case"], err) for case in cases for err in case["model_errors"]],
        "failed_scores": [(case["case"], name) for case in cases for name in case["failed_scores"]],
        "cases": cases,
    }


def print_summary(summary: dict) -> None:
    print(f"\nExperiment '{summary['experiment']}' ({summary['model']}), dataset '{DATASET_NAME}':")
    for name in METRIC_NAMES:
        value = summary["scores"].get(name)
        print(f"  {name:<26} " + (f"{value:.2f}" if value is not None else "n/a (scoring failed)"))
    cost = summary["cost_per_case_usd"]
    priced = f"{summary['priced_cases']}/{len(summary['cases'])} cases priced"
    print(f"  {'cost per case (Opik)':<26} " + (f"${cost:.4f} ({priced})" if cost is not None else "not priced"))
    print(f"  {'tokens per case':<26} {summary['tokens_per_case']:,.0f}")
    print(f"  {'median latency per case':<26} {summary['median_latency_s']:.1f}s")

    # The averages say how good the model is. The misses say where it slipped.
    print(f"\nMisses (any score below {MISS_BELOW}):")
    misses = [
        (case["case"], name, value, case["reasons"].get(name))
        for case in summary["cases"]
        for name, value in case["scores"].items()
        if value < MISS_BELOW
    ]
    for case, name, value, reason in misses:
        print(f"  - {case}: {name} {value:.2f} ({reason})")
    if not misses:
        print("  none")

    for case, error in summary["model_errors"]:
        print(f"  - {case}: the model returned an error ({error})")

    # Errors are the run's fault, not the model's: rerun before trusting the scores.
    for case, error in summary["errors"]:
        print(f"  ! {case}: the agent run failed ({error}). It was not scored; rerun it.")
    for case, name in summary["failed_scores"]:
        print(f"  ! {case}: {name} could not be scored (see the experiment in Opik).")


def exit_if_missing_credentials() -> None:
    missing = check_credentials()
    if missing:
        print("Missing before this can run:\n  - " + "\n  - ".join(missing))
        print("Copy .env.example to .env and fill it in.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL, help=f"Gemini model for the agent (default {MODEL})")
    args = parser.parse_args()
    exit_if_missing_credentials()
    print_summary(run_experiment(args.model))
