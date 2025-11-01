#!/usr/bin/env python3
"""Quick analysis helper for comparing baseline vs failure experiments."""

import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from impossiblebench.analysis import DataLoader


def quick_compare(log_dirs):
    """Quick comparison between multiple experiment runs."""

    # Load data
    loader = DataLoader(n_workers=4)
    for log_dir in log_dirs:
        if Path(log_dir).exists():
            print(f"Loading: {log_dir}")
            loader.load_folder(log_dir)
        else:
            print(f"⚠️  Not found: {log_dir}")

    if len(loader.results) == 0:
        print("❌ No data loaded!")
        return

    # Convert to dataframe
    df = loader.to_df()

    # Filter to conflicting tests only
    df = df[df["variant"] == "conflicting"]

    # Filter out aggregated rows
    df = df[df["task_id"].str.contains("sample_", na=False)]

    # Extract experiment type from file_path
    df["experiment"] = df["file_path"].str.extract(r"logs/([^/]+)/")[0]

    # Group by experiment
    results = df.groupby("experiment")["passed"].agg(["count", "sum", "mean"])
    results.columns = ["n_samples", "n_passed", "pass_rate"]

    # Print results
    print("\n" + "=" * 80)
    print("BASELINE vs TOOL FAILURE EXPERIMENTS")
    print("=" * 80)
    print("\nReward Hacking Rate (Passing Impossible Tests):")
    print("\n" + results.to_string() + "\n")

    # Try to identify baseline vs others
    baseline_row = None
    failure_row = None

    for exp in results.index:
        if "baseline" in exp.lower():
            baseline_row = results.loc[exp]
        elif any(
            x in exp.lower()
            for x in ["aggressive", "conservative", "failures", "custom"]
        ):
            failure_row = (exp, results.loc[exp])

    if baseline_row is not None and failure_row is not None:
        exp_name, failure_row_data = failure_row
        baseline_rate = baseline_row["pass_rate"]
        failure_rate = failure_row_data["pass_rate"]

        print("=" * 80)
        print("COMPARISON:")
        print("=" * 80)
        print(
            f"\nBaseline:  {baseline_rate:.1%} ({baseline_row['n_passed']:.0f}/{baseline_row['n_samples']:.0f} samples)"
        )
        print(
            f"{exp_name}: {failure_rate:.1%} ({failure_row_data['n_passed']:.0f}/{failure_row_data['n_samples']:.0f} samples)"
        )

        if baseline_rate > failure_rate:
            diff = baseline_rate - failure_rate
            print(
                f"\nResult: Tool failures DECREASED reward hacking by {diff:.0%} ({diff*100:.1f} percentage points)"
            )
        elif failure_rate > baseline_rate:
            diff = failure_rate - baseline_rate
            print(
                f"\nResult: Tool failures INCREASED reward hacking by {diff:.0%} ({diff*100:.1f} percentage points)"
            )
        else:
            print(f"\nResult: No change in reward hacking rate")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_analysis.py <log_dir1> [log_dir2 ...]")
        print("\nExample:")
        print(
            "  python quick_analysis.py logs/spar_baseline_promptD logs/spar_aggressive_promptD"
        )
        sys.exit(1)

    log_dirs = sys.argv[1:]
    quick_compare(log_dirs)


if __name__ == "__main__":
    main()
