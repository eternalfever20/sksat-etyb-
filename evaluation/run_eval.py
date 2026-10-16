"""Deterministic offline checks; --live verifies OpenRouter tool calling."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent.orchestrator import AgentError, ask
from dataset.loader import load_dataset
from tools.analysis import run_tool

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")


def offline():
    frame, info = load_dataset(
        "demo_sales.csv", (ROOT / "data/demo_sales.csv").read_bytes()
    )
    cases = [
        ("profile", "describe_dataset", {}, lambda r: r["rows"] == len(frame)),
        ("missingness", "analyze_missingness", {}, lambda r: "rows_with_missing" in r),
        (
            "filter",
            "filter_rows",
            {
                "conditions": [
                    {"column": "returned", "operator": "eq", "value": False}
                ],
                "count_only": True,
            },
            lambda r: r["matched_rows"] == 33,
        ),
        (
            "group aggregate",
            "group_aggregate",
            {"group_by": ["region"], "metric": "revenue", "agg": "mean"},
            lambda r: bool(r["groups"]),
        ),
        (
            "year trend",
            "trend_over_time",
            {
                "time_column": "order_date",
                "metric": "revenue",
                "frequency": "month",
                "agg": "sum",
            },
            lambda r: bool(r["periods"]),
        ),
    ]
    iris, iris_info = load_dataset("iris.csv", (ROOT / "data/iris.csv").read_bytes())
    cases.extend(
        [
            (
                "UCI Iris profile",
                "describe_dataset",
                {},
                lambda r: r["rows"] == 150 and r["columns"] == 5,
                iris,
                iris_info,
            ),
            (
                "UCI Iris species aggregation",
                "group_aggregate",
                {"group_by": ["species"], "metric": "petal_length_cm", "agg": "mean"},
                lambda r: (
                    len(r["groups"]) == 3
                    and next(
                        g["value"]
                        for g in r["groups"]
                        if g["species"] == "Iris-virginica"
                    )
                    > next(
                        g["value"] for g in r["groups"] if g["species"] == "Iris-setosa"
                    )
                ),
                iris,
                iris_info,
            ),
            (
                "UCI Iris correlation",
                "correlation",
                {"columns": ["sepal_length_cm", "petal_length_cm"]},
                lambda r: r["pairs"][0]["coefficient"] > 0.8,
                iris,
                iris_info,
            ),
        ]
    )
    passed = 0
    for case in cases:
        label, name, args, check = case[:4]
        case_frame, case_info = case[4:] or (frame, info)
        result = run_tool(
            name,
            case_frame,
            case_info["types"],
            args,
            roles=case_info["roles"],
            conversions=case_info.get("conversions"),
        )
        ok = bool(result.get("ok") and check(result))
        passed += ok
        print(
            f"{'PASS' if ok else 'FAIL'} {label}"
            + (f" — {result.get('error', {}).get('message')}" if not ok else "")
        )
    print(f"Offline evaluation: {passed}/{len(cases)}")
    return passed == len(cases)


def live():
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY before --live.")
    frame, info = load_dataset(
        "demo_sales.csv", (ROOT / "data/demo_sales.csv").read_bytes()
    )
    try:
        answer = ask(
            "What is the total revenue? Calculate it and cite evidence.",
            frame,
            info["types"],
            roles=info["roles"],
            conversions=info.get("conversions"),
        )
    except AgentError as exc:
        print(f"Live OpenRouter evaluation failed: {exc}")
        return False
    print(json.dumps(answer, indent=2, default=str))
    return bool(answer["trace"] and not answer.get("limit_reached"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if (live() if args.live else offline()) else 1)
