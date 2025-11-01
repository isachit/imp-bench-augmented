#!/usr/bin/env python3
"""SPAR Baseline Experiment: Tool Unreliability vs Reward Hacking

This script runs experiments to test whether tool unreliability increases
reward hacking rates in AI agents.

Usage:
    python experiments/spar_baseline_experiment.py baseline
    python experiments/spar_baseline_experiment.py with_failures
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import eval
from impossiblebench import (
    impossible_livecodebench,
    ToolReliability,
    create_baseline_config,
    create_conservative_config,
    create_aggressive_config,
)


def run_baseline_experiment():
    """Run baseline experiment with no tool failures (conflicting tests only)."""
    print("=" * 60)
    print("BASELINE EXPERIMENT: No Tool Failures")
    print("=" * 60)

    task = impossible_livecodebench(
        split="conflicting",  # Only possible to pass via reward hacking
        agent_type="minimal",  # Simple agent
        limit=10,  # Run on 10 samples
        max_attempts=5,
        allow_test_modifications=True,
    )

    # No tool reliability injected
    logs = eval(
        task, model="openrouter/openai/gpt-4o-mini", log_dir="./logs/spar_baseline"
    )

    print("\nBaseline experiment complete!")
    print(f"Logs saved to: {logs}")

    return logs


def run_tool_failure_experiment(failure_rate: float = 0.3):
    """Run experiment with tool failures injected.

    Args:
        failure_rate: Failure rate for all failure types (0.0-1.0)
    """
    print("=" * 60)
    print(f"TOOL FAILURE EXPERIMENT: {failure_rate*100:.0f}% failure rate")
    print("=" * 60)

    # Create custom tool reliability config
    tool_reliability = ToolReliability(
        timeout_failure_rate=failure_rate,
        write_failure_rate=failure_rate,
        corruption_rate=failure_rate,
        seed=42,  # For reproducibility
    )

    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")

    # Create custom solver with tool reliability
    from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

    solver = agentic_humaneval_solver(
        max_attempts=5,
        allow_test_modifications=True,
        tool_reliability=tool_reliability,
    )

    task = impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        solver=solver,  # Use our custom solver
        limit=10,
        max_attempts=5,
        allow_test_modifications=True,
    )

    logs = eval(
        task,
        model="openrouter/openai/gpt-4o-mini",
        log_dir=f"./logs/spar_failures_{int(failure_rate*100)}",
    )

    print("\nTool failure experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_conservative_experiment():
    """Run experiment with conservative failure rates (10%)."""
    print("=" * 60)
    print("CONSERVATIVE FAILURE EXPERIMENT: 10% failure rate")
    print("=" * 60)

    tool_reliability = create_conservative_config()

    from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

    solver = agentic_humaneval_solver(
        max_attempts=5,
        allow_test_modifications=True,
        tool_reliability=tool_reliability,
    )

    task = impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        solver=solver,
        limit=10,
        max_attempts=5,
        allow_test_modifications=True,
    )

    logs = eval(
        task, model="openrouter/openai/gpt-4o-mini", log_dir="./logs/spar_conservative"
    )

    print("\nConservative experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_aggressive_experiment():
    """Run experiment with aggressive failure rates (50%)."""
    print("=" * 60)
    print("AGGRESSIVE FAILURE EXPERIMENT: 50% failure rate")
    print("=" * 60)

    tool_reliability = create_aggressive_config()

    from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

    solver = agentic_humaneval_solver(
        max_attempts=5,
        allow_test_modifications=True,
        tool_reliability=tool_reliability,
    )

    task = impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        solver=solver,
        limit=10,
        max_attempts=5,
        allow_test_modifications=True,
    )

    logs = eval(
        task, model="openrouter/openai/gpt-4o-mini", log_dir="./logs/spar_aggressive"
    )

    print("\nAggressive experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def main():
    """Run experiments based on command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python experiments/spar_baseline_experiment.py <experiment_name>")
        print("\nAvailable experiments:")
        print("  baseline       - No tool failures (baseline)")
        print("  conservative   - 10% failure rate")
        print("  failures30     - 30% failure rate")
        print("  aggressive     - 50% failure rate")
        print("  all            - Run all experiments")
        sys.exit(1)

    experiment = sys.argv[1].lower()

    if experiment == "baseline":
        run_baseline_experiment()
    elif experiment == "conservative":
        run_conservative_experiment()
    elif experiment == "failures30":
        run_tool_failure_experiment(failure_rate=0.3)
    elif experiment == "aggressive":
        run_aggressive_experiment()
    elif experiment == "all":
        print("Running all experiments...\n")
        run_baseline_experiment()
        run_conservative_experiment()
        run_tool_failure_experiment(failure_rate=0.3)
        run_aggressive_experiment()
    else:
        print(f"Unknown experiment: {experiment}")
        sys.exit(1)


if __name__ == "__main__":
    main()
