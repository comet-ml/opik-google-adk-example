"""Run the same evaluation on two Gemini models and print the results side by side.

    uv run compare.py                                       # the two defaults below
    uv run compare.py --models gemini-3.8-flash gemini-3.1-flash-lite

Each model gets its own Opik experiment on the same dataset, so you can also
open both in Opik and compare them case by case. The tables printed here are
Markdown, ready to paste into a PR or a doc.
"""

import argparse

from evaluate import (
    METRIC_NAMES,
    MISS_BELOW,
    exit_if_missing_credentials,
    run_experiment,
)
from support_agent.agent import DEFAULT_MODEL

# The current GA Flash model against the cheaper Flash-Lite.
DEFAULT_MODELS = [DEFAULT_MODEL, "gemini-3.5-flash-lite"]


def comparison_table(summaries: list[dict]) -> str:
    """One row per metric, one column per model."""
    header = "| | " + " | ".join(s["model"] for s in summaries) + " |"
    lines = [header, "|---" * (len(summaries) + 1) + "|"]
    for name in METRIC_NAMES:
        cells = [s["scores"].get(name) for s in summaries]
        lines.append(f"| {name} | " + " | ".join("n/a" if v is None else f"{v:.2f}" for v in cells) + " |")

    def money(value):
        return f"${value:.4f}" if value is not None else "not priced"

    lines.append("| agent cost per case (Opik) | " + " | ".join(money(s["cost_per_case_usd"]) for s in summaries) + " |")
    lines.append("| tokens per case | " + " | ".join(f"{s['tokens_per_case']:,.0f}" for s in summaries) + " |")
    lines.append("| median latency per case | " + " | ".join(f"{s['median_latency_s']:.1f}s" for s in summaries) + " |")
    return "\n".join(lines)


def misses_table(summaries: list[dict]) -> str:
    """Every case and metric where at least one model scored below MISS_BELOW.

    The averages say which model is better. This says where, which is what
    tells you whether the cheaper model is safe to ship.
    """
    by_case = [{case["case"]: case["scores"] for case in s["cases"]} for s in summaries]
    lines = [
        "| case | metric | " + " | ".join(s["model"] for s in summaries) + " |",
        "|---|---" + "|---" * len(summaries) + "|",
    ]
    for case in by_case[0]:
        for name in METRIC_NAMES:
            values = [scores.get(case, {}).get(name) for scores in by_case]
            if any(value is not None and value < MISS_BELOW for value in values):
                cells = " | ".join("" if v is None else f"{v:.2f}" for v in values)
                lines.append(f"| {case} | {name} | {cells} |")
    return "\n".join(lines) if len(lines) > 2 else f"No case scored below {MISS_BELOW} on any metric."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Gemini models to compare")
    args = parser.parse_args()
    exit_if_missing_credentials()

    summaries = [run_experiment(model) for model in args.models]
    print("\n## Scores, cost and latency\n")
    print(comparison_table(summaries))
    print("\n## Where the models missed\n")
    print(misses_table(summaries))
    for summary in summaries:
        for case, error in summary["errors"]:
            print(f"\n! {summary['model']}, {case}: the agent run failed ({error}). Rerun before trusting it.")
        for case, name in summary["failed_scores"]:
            print(f"\n! {summary['model']}, {case}: {name} could not be scored.")
    print("\nExperiments in Opik: " + ", ".join(s["experiment"] for s in summaries))
