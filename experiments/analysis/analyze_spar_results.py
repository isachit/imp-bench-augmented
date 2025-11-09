#!/usr/bin/env python3
"""Unified analysis CLI for SPAR experiments.

This script ingests LiveCodeBench or SWE-bench evaluation logs produced by
`spar_baseline_experiment.py` and can emit:

- Quick pass-rate summaries per experiment
- Detailed breakdowns and baseline vs. failure comparisons
- Optional LLM-based judging of agent trajectories (LLMaaJ)

It replaces the previous `quick_analysis.py` and `llm_judge_analysis.py`
wrappers. Those legacy entry points now forward to this module.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from impossiblebench.analysis import DataLoader

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def parse_experiment_name(name: str) -> dict[str, str | None]:
    """Parse log directory names produced by spar_baseline_experiment."""
    meta: dict[str, str | None] = {
        "experiment_name": name,
        "experiment_type": None,
        "benchmark": "livecodebench",
        "prompt": None,
        "swe_split": None,
        "agent_type": None,
    }

    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "spar":
        return meta

    meta["experiment_type"] = parts[1]
    remainder = parts[2:]
    if not remainder:
        return meta

    first = remainder[0]

    if first == "swe":
        meta["benchmark"] = "swebench"
        if len(remainder) >= 2:
            meta["swe_split"] = remainder[1]
        if len(remainder) >= 3:
            meta["agent_type"] = remainder[2]
    elif first.startswith("prompt"):
        meta["prompt"] = first[len("prompt") :]
        if len(remainder) >= 2:
            meta["agent_type"] = remainder[1]
    elif first in {"minimal", "tools"}:
        meta["agent_type"] = first
    else:
        if first in {"original", "oneoff", "conflicting"}:
            meta["benchmark"] = "swebench"
            meta["swe_split"] = first
        else:
            meta["prompt"] = first

    if meta["agent_type"] is None:
        meta["agent_type"] = "minimal"

    if meta["benchmark"] == "swebench" and meta["swe_split"] is None:
        meta["swe_split"] = "conflicting"

    if meta["benchmark"] == "livecodebench" and meta["prompt"] is None:
        meta["prompt"] = "D"

    return meta


def load_results(paths: Iterable[str]) -> DataLoader:
    """Load .eval files from the provided paths."""
    path_list = list(paths)
    loader = DataLoader(n_workers=4)

    print(f"Loading results from {len(path_list)} directories/files...")
    for path in path_list:
        if os.path.exists(path):
            print(f"  Loading: {path}")
            loader.load_folder(path)
        else:
            print(f"  ⚠️  Path not found: {path}")
    return loader


def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "file_path" in df.columns:
        df["experiment"] = df["file_path"].str.extract(r"logs/([^/]+)/")[0]
        meta_df = df["experiment"].apply(parse_experiment_name).apply(pd.Series)
        for col in meta_df.columns:
            if col not in df.columns:
                df[col] = meta_df[col]
            else:
                df[col] = df[col].fillna(meta_df[col])
    else:
        df["experiment"] = "unknown"

    if "agent_type" not in df.columns:
        df["agent_type"] = "minimal"
    else:
        df["agent_type"] = df["agent_type"].fillna("minimal")

    return df


def filter_dataframe(df: pd.DataFrame, variant: str | None) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = df.copy()

    if variant and variant.lower() != "all":
        if "variant" in filtered.columns:
            filtered = filtered[filtered["variant"] == variant]
        else:
            print(
                f"⚠️  Variant column not found in results; skipping filter for '{variant}'."
            )

    if "task_id" in filtered.columns:
        filtered = filtered[filtered["task_id"].str.contains("sample_", na=False)]

    return filtered


def prepare_dataframe(raw_df: pd.DataFrame, variant: str | None) -> pd.DataFrame:
    df = augment_dataframe(raw_df)
    df = filter_dataframe(df, variant)
    return df


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------


def print_quick_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    quick = (
        df.groupby("experiment")["passed"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_samples", "sum": "n_passed", "mean": "pass_rate"})
    )
    meta_cols = ["benchmark", "agent_type", "experiment_type", "prompt", "swe_split"]
    meta = (
        df[["experiment"] + meta_cols]
        .drop_duplicates("experiment")
        .set_index("experiment")
    )
    quick = quick.join(meta, how="left")

    display_cols = [
        "benchmark",
        "agent_type",
        "experiment_type",
        "prompt",
        "swe_split",
        "n_samples",
        "n_passed",
        "pass_rate",
    ]
    printable = quick[display_cols].fillna("")
    print("\nReward hacking rate (pass rate on filtered samples):\n")
    print(printable.to_string())


def print_detailed_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    print(f"\n📊 Total samples: {len(df)}")
    if "passed" in df.columns:
        print(f"📈 Overall pass rate: {df['passed'].mean():.1%}")

    exp_summary = (
        df.groupby("experiment")["passed"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "n_samples", "sum": "n_passed", "mean": "pass_rate"})
    )
    meta_cols = ["experiment_type", "benchmark", "agent_type", "prompt", "swe_split"]
    meta_info = (
        df[["experiment"] + meta_cols]
        .drop_duplicates("experiment")
        .set_index("experiment")
    )
    exp_summary = exp_summary.join(meta_info, how="left")

    baselines: dict[
        tuple[str | None, str | None, str | None], tuple[str, pd.Series]
    ] = {}
    for name, row in exp_summary.iterrows():
        if row["experiment_type"] == "baseline":
            baselines[baseline_key(row)] = (name, row)

    if baselines:
        print("\n🔍 Baseline comparisons (per benchmark/agent/prompt or split):")
    for name, row in exp_summary.iterrows():
        key = baseline_key(row)
        if row["experiment_type"] == "baseline" or key not in baselines:
            continue
        baseline_name, baseline_row = baselines[key]
        diff = row["pass_rate"] - baseline_row["pass_rate"]
        direction = (
            "INCREASED" if diff > 0 else "DECREASED" if diff < 0 else "UNCHANGED"
        )
        delta_pct = abs(diff) * 100
        print(
            f"  {name} vs {baseline_name}: {direction} reward hacking by {delta_pct:.2f} percentage points"
        )

    if "model" in df.columns:
        print("\n📈 Pass rates by model:")
        model_rates = (
            df.groupby("model")["passed"]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_samples", "mean": "pass_rate"})
        )
        print(model_rates.to_string())

    if "agent_type" in df.columns:
        print("\n🤖 Pass rates by agent type:")
        agent_rates = (
            df.groupby("agent_type")["passed"]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_samples", "mean": "pass_rate"})
        )
        print(agent_rates.to_string())


def baseline_key(row: pd.Series) -> tuple[str | None, str | None, str | None]:
    if row.get("benchmark") == "swebench":
        return (row.get("benchmark"), row.get("agent_type"), row.get("swe_split"))
    return (row.get("benchmark"), row.get("agent_type"), row.get("prompt"))


def compare_failure_rates(df: pd.DataFrame) -> None:
    if "passed" not in df.columns:
        return

    print("\n🔬 Tool Failure Impact Analysis:")
    print("-" * 80)

    for col in ["prompt_id", "failure_mode", "failure_rate"]:
        if col in df.columns and df[col].notna().any():
            grouped = (
                df.groupby(col)["passed"]
                .agg(["count", "mean"])
                .rename(columns={"count": "n_samples", "mean": "pass_rate"})
            )
            print(f"\nPass rates by {col}:")
            print(grouped.to_string())

    if "score" in df.columns:
        print("\n📊 Score statistics:")
        print(df["score"].describe())

    if "first_score" in df.columns and df["first_score"].notna().any():
        print("\n🎯 First-attempt score statistics:")
        print(df["first_score"].dropna().describe())


async def _run_llm_judge_async(samples: pd.DataFrame, model: str) -> None:
    from impossiblebench.analysis import LLMJudge

    judge = LLMJudge(model=model, max_concurrency=2)
    dumps = samples["dump"].tolist()

    print("\n🧑‍⚖️ Running LLM Judge (binary cheating vs. legitimate)...")
    binary_results = await judge.batch_evaluate_binary(dumps)

    for i, result in enumerate(binary_results):
        sample = samples.iloc[i]
        print(f"\nSample {i + 1} ({sample['task_id']}):")
        print(f"  Experiment: {sample['experiment']}")
        print(f"  Passed: {sample['passed']}")
        print(f"  Judgment: {result['judgment']}")
        if not result["success"]:
            snippet = result.get("raw_response", "")[:120]
            print(f"  Error: {snippet}...")

    binary_stats = judge.get_stats(binary_results)
    print("\n📈 Binary Evaluation Stats:")
    print(f"  Total samples: {binary_stats['total_samples']}")
    print(f"  Successful evaluations: {binary_stats['successful_evaluations']}")
    print(f"  Judgments: {binary_stats['judgment_counts']}")

    cheating_indices = [
        idx
        for idx, res in enumerate(binary_results)
        if res.get("judgment") == "cheating"
    ]
    if not cheating_indices:
        print("\n🧪 Cheating type evaluation: No cheating samples detected.")
        return

    print(
        f"\n🧪 Running cheating type classification on {len(cheating_indices)} samples..."
    )
    cheating_dumps = [dumps[idx] for idx in cheating_indices]
    type_results = await judge.batch_evaluate_type(cheating_dumps)

    for i, result in enumerate(type_results):
        sample = samples.iloc[cheating_indices[i]]
        print(f"\nSample {cheating_indices[i] + 1} ({sample['task_id']}):")
        print(f"  Cheating Type: {result['judgment']}")
        if not result["success"]:
            snippet = result.get("raw_response", "")[:120]
            print(f"  Error: {snippet}...")

    type_stats = judge.get_stats(type_results)
    print("\n📈 Type Evaluation Stats:")
    print(f"  Total samples: {type_stats['total_samples']}")
    print(f"  Successful evaluations: {type_stats['successful_evaluations']}")
    print(f"  Cheating types: {type_stats['judgment_counts']}")


def run_llm_judge(df: pd.DataFrame, model: str, max_samples: int) -> None:
    if "dump" not in df.columns:
        print(
            "\n❌ LLM judge requested but `dump` column missing. Re-run analysis with dumps."
        )
        return

    samples = df[df["dump"].notna()]
    if samples.empty:
        print("\n❌ No samples with trajectories (`dump`) available for LLM judge.")
        return

    limited = samples.head(max_samples)
    print(
        f"\n📊 Selected {len(limited)} samples (of {len(samples)}) for LLM judge analysis."
    )
    print(f"Experiments: {limited['experiment'].unique().tolist()}")

    asyncio.run(_run_llm_judge_async(limited, model))


def save_results(df: pd.DataFrame, output_file: str) -> None:
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SPAR experiment logs (LiveCodeBench or SWE-bench).",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Log directories or files to load (can mix baseline/failure runs).",
    )
    parser.add_argument(
        "--variant",
        default="conflicting",
        help="Variant to filter on (set to 'all' to disable filtering).",
    )
    parser.add_argument(
        "--quick-only",
        action="store_true",
        help="Only print the quick summary table (skip detailed analysis).",
    )
    parser.add_argument(
        "--no-quick",
        action="store_true",
        help="Skip printing the quick summary table.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Run LLM-based trajectory evaluation (requires OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--llm-judge-model",
        default="openrouter/anthropic/claude-3.5-sonnet",
        help="Model identifier to use for LLM judge (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-judge-max-samples",
        type=int,
        default=10,
        help="Maximum number of samples to send to the LLM judge (default: %(default)s).",
    )
    parser.add_argument(
        "--output-csv",
        default="spar_analysis_results.csv",
        help="Where to write the filtered results CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the CSV output file.",
    )

    args = parser.parse_args(argv)

    loader = load_results(args.paths)
    if len(loader.results) == 0:
        print("\n❌ No results loaded! Make sure the directories contain .eval files.")
        sys.exit(1)

    include_dump = bool(args.llm_judge)
    raw_df = loader.to_df(dump=include_dump)
    filtered_df = prepare_dataframe(raw_df, args.variant)

    if filtered_df.empty:
        print("\n❌ No samples after filtering; try setting --variant all.")
        sys.exit(1)

    if not args.no_quick:
        print_quick_summary(filtered_df)

    if not args.quick_only:
        print_detailed_summary(filtered_df)
        compare_failure_rates(filtered_df)

    if args.llm_judge:
        run_llm_judge(filtered_df, args.llm_judge_model, args.llm_judge_max_samples)

    if not args.no_save:
        save_results(filtered_df, args.output_csv)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
