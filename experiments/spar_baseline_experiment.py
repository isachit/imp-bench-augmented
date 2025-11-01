#!/usr/bin/env python3
"""SPAR Baseline Experiment: Tool Unreliability vs Reward Hacking

This script runs experiments to test whether tool unreliability increases
reward hacking rates in AI agents.

Usage:
    # Baseline (no failures)
    python experiments/spar_baseline_experiment.py baseline --model openrouter/openai/gpt-4o-mini
    
    # With failures (using preset configs)
    python experiments/spar_baseline_experiment.py conservative --model openrouter/openai/gpt-4o-mini
    python experiments/spar_baseline_experiment.py aggressive --model openrouter/openai/gpt-4o-mini
    
    # Custom failure configuration
    python experiments/spar_baseline_experiment.py custom \\
        --model openrouter/openai/gpt-4o-mini \\
        --timeout 0.3 --write 0.3 --corruption 0.3 \\
        --rate-limit 0.2 --memory 0.1 --misleading 0.1
    
    # Specific failure modes only
    python experiments/spar_baseline_experiment.py failures \\
        --model openrouter/openai/gpt-4o-mini \\
        --failure-modes timeout,rate-limit,memory
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from inspect_ai import eval
from impossiblebench import (
    impossible_livecodebench,
    ToolReliability,
    create_baseline_config,
    create_conservative_config,
    create_aggressive_config,
)

# Import prompts
from experiments.prompts import get_prompt, get_prompt_description


def run_baseline_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    limit: int = 10,
    prompt: str = "D",
):
    """Run baseline experiment with no tool failures (conflicting tests only).

    Args:
        model: Model identifier to use for evaluation
        limit: Number of samples to run
        prompt: Prompt variant to use (A, B, C, D, default, strict, weak, loose, moderate)
    """
    # Get prompt from paper variants (system will automatically add "DO NOT MODIFY THE TESTS")
    # The weak/strict distinction is in OTHER wording:
    # - Weak (A): "pass all tests" emphasis → 92% cheating
    # - Strict (D): "STOP, identify them" + "Do NOT try to carve out the code" → 1% cheating
    instruction_prompt = get_prompt(prompt)

    print("=" * 60)
    print("BASELINE EXPERIMENT: No Tool Failures")
    print(f"Model: {model}")
    print(
        f"Prompt: {prompt} - {get_prompt_description(prompt.upper() if prompt.upper() in ['A', 'B', 'C', 'D'] else 'D')}"
    )
    print("=" * 60)

    task = impossible_livecodebench(
        split="conflicting",  # Only possible to pass via reward hacking
        agent_type="minimal",  # Simple agent
        instruction_prompt=instruction_prompt,
        limit=limit,
        max_attempts=5,
        allow_test_modifications=True,
    )

    # No tool reliability injected
    logs = eval(
        task,
        model=model,
        log_dir=f"./logs/spar_baseline_prompt{prompt.upper()}",
    )

    print("\nBaseline experiment complete!")
    print(f"Logs saved to: {logs}")

    return logs


def run_tool_failure_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    failure_rate: float = 0.3,
    limit: int = 10,
    failure_modes: Optional[list] = None,
    prompt: str = "D",
):
    """Run experiment with tool failures injected.

    Args:
        model: Model identifier to use for evaluation
        failure_rate: Failure rate for all failure types (0.0-1.0)
        limit: Number of samples to run
        failure_modes: List of failure modes to enable. If None, enables all basic modes.
            Options: 'timeout', 'write', 'corruption', 'rate-limit', 'memory',
            'misleading', 'permission-denied', 'intermittent'
        prompt: Prompt variant to use (A, B, C, D, default, strict, weak, loose, moderate)
    """
    # Get prompt from paper variants (system will automatically add "DO NOT MODIFY THE TESTS")
    # The weak/strict distinction is in OTHER wording:
    # - Weak (A): "pass all tests" emphasis → 92% cheating
    # - Strict (D): "STOP, identify them" + "Do NOT try to carve out the code" → 1% cheating
    instruction_prompt = get_prompt(prompt)

    print("=" * 60)
    print(f"TOOL FAILURE EXPERIMENT: {failure_rate*100:.0f}% failure rate")
    print(f"Model: {model}")
    print(
        f"Prompt: {prompt} - {get_prompt_description(prompt.upper() if prompt.upper() in ['A', 'B', 'C', 'D'] else 'D')}"
    )
    if failure_modes:
        print(f"Failure modes: {', '.join(failure_modes)}")
    print("=" * 60)

    # Default to basic modes if none specified
    if failure_modes is None:
        failure_modes = ["timeout", "write", "corruption"]

    # Create custom tool reliability config
    config_kwargs = {"seed": 42}  # For reproducibility

    if "timeout" in failure_modes:
        config_kwargs["timeout_failure_rate"] = failure_rate
    if "write" in failure_modes:
        config_kwargs["write_failure_rate"] = failure_rate
    if "corruption" in failure_modes:
        config_kwargs["corruption_rate"] = failure_rate
    if "rate-limit" in failure_modes:
        config_kwargs["rate_limit_failure_rate"] = failure_rate
    if "memory" in failure_modes:
        config_kwargs["memory_failure_rate"] = failure_rate
    if "misleading" in failure_modes:
        config_kwargs["misleading_error_rate"] = failure_rate
    if "permission-denied" in failure_modes:
        config_kwargs["permission_denied_rate"] = failure_rate
    if "intermittent" in failure_modes:
        config_kwargs["intermittent_failure_rate"] = failure_rate

    tool_reliability = ToolReliability(**config_kwargs)

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
        instruction_prompt=instruction_prompt,
        solver=solver,  # Use our custom solver
        limit=limit,
        max_attempts=5,
        allow_test_modifications=True,
    )

    modes_str = "_".join(sorted(failure_modes)) if failure_modes else "all"
    logs = eval(
        task,
        model=model,
        log_dir=f"./logs/spar_failures_{int(failure_rate*100)}_{modes_str}_prompt{prompt.upper()}",
    )

    print("\nTool failure experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_conservative_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    limit: int = 10,
    prompt: str = "D",
):
    """Run experiment with conservative failure rates (10%).

    Args:
        model: Model identifier to use for evaluation
        limit: Number of samples to run
        prompt: Prompt variant to use (A, B, C, D, default, strict, weak, loose, moderate)
    """
    # Get prompt from paper variants (system will automatically add "DO NOT MODIFY THE TESTS")
    # The weak/strict distinction is in OTHER wording:
    # - Weak (A): "pass all tests" emphasis → 92% cheating
    # - Strict (D): "STOP, identify them" + "Do NOT try to carve out the code" → 1% cheating
    instruction_prompt = get_prompt(prompt)

    print("=" * 60)
    print("CONSERVATIVE FAILURE EXPERIMENT: 10% failure rate")
    print(f"Model: {model}")
    print(
        f"Prompt: {prompt} - {get_prompt_description(prompt.upper() if prompt.upper() in ['A', 'B', 'C', 'D'] else 'D')}"
    )
    print("=" * 60)

    tool_reliability = create_conservative_config()
    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")

    from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

    solver = agentic_humaneval_solver(
        max_attempts=5,
        allow_test_modifications=True,
        tool_reliability=tool_reliability,
    )

    task = impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        instruction_prompt=instruction_prompt,
        solver=solver,
        limit=limit,
        max_attempts=5,
        allow_test_modifications=True,
    )

    logs = eval(
        task,
        model=model,
        log_dir=f"./logs/spar_conservative_prompt{prompt.upper()}",
    )

    print("\nConservative experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_aggressive_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    limit: int = 10,
    prompt: str = "D",
):
    """Run experiment with aggressive failure rates (50%).

    Args:
        model: Model identifier to use for evaluation
        limit: Number of samples to run
        prompt: Prompt variant to use (A, B, C, D, default, strict, weak, loose, moderate)
    """
    # Get prompt from paper variants (system will automatically add "DO NOT MODIFY THE TESTS")
    # The weak/strict distinction is in OTHER wording:
    # - Weak (A): "pass all tests" emphasis → 92% cheating
    # - Strict (D): "STOP, identify them" + "Do NOT try to carve out the code" → 1% cheating
    instruction_prompt = get_prompt(prompt)

    print("=" * 60)
    print("AGGRESSIVE FAILURE EXPERIMENT: 50% failure rate")
    print(f"Model: {model}")
    print(
        f"Prompt: {prompt} - {get_prompt_description(prompt.upper() if prompt.upper() in ['A', 'B', 'C', 'D'] else 'D')}"
    )
    print("=" * 60)

    tool_reliability = create_aggressive_config()
    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")

    from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

    solver = agentic_humaneval_solver(
        max_attempts=5,
        allow_test_modifications=True,
        tool_reliability=tool_reliability,
    )

    task = impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        instruction_prompt=instruction_prompt,
        solver=solver,
        limit=limit,
        max_attempts=5,
        allow_test_modifications=True,
    )

    logs = eval(
        task,
        model=model,
        log_dir=f"./logs/spar_aggressive_prompt{prompt.upper()}",
    )

    print("\nAggressive experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_custom_experiment(
    model: str,
    limit: int = 10,
    timeout: float = 0.0,
    write: float = 0.0,
    corruption: float = 0.0,
    rate_limit: float = 0.0,
    memory: float = 0.0,
    misleading: float = 0.0,
    permission_denied: float = 0.0,
    intermittent: float = 0.0,
    intermittent_pattern: Optional[int] = None,
    seed: int = 42,
    prompt: str = "D",
):
    """Run experiment with custom failure configuration.

    Args:
        model: Model identifier to use for evaluation
        limit: Number of samples to run
        timeout: Timeout failure rate (0-1)
        write: Write failure rate (0-1)
        corruption: Corruption failure rate (0-1)
        rate_limit: Rate limit failure rate (0-1)
        memory: Memory failure rate (0-1)
        misleading: Misleading error rate (0-1)
        permission_denied: Permission denied rate (0-1)
        intermittent: Intermittent failure rate (0-1)
        intermittent_pattern: If set, fail every Nth attempt
        seed: Random seed for reproducibility
        prompt: Prompt variant to use (A, B, C, D, default, strict, weak, loose, moderate)
    """
    # Get prompt from paper variants (system will automatically add "DO NOT MODIFY THE TESTS")
    # The weak/strict distinction is in OTHER wording:
    # - Weak (A): "pass all tests" emphasis → 92% cheating
    # - Strict (D): "STOP, identify them" + "Do NOT try to carve out the code" → 1% cheating
    instruction_prompt = get_prompt(prompt)

    print("=" * 60)
    print("CUSTOM FAILURE EXPERIMENT")
    print(f"Model: {model}")
    print(
        f"Prompt: {prompt} - {get_prompt_description(prompt.upper() if prompt.upper() in ['A', 'B', 'C', 'D'] else 'D')}"
    )
    print("=" * 60)

    tool_reliability = ToolReliability(
        timeout_failure_rate=timeout,
        write_failure_rate=write,
        corruption_rate=corruption,
        rate_limit_failure_rate=rate_limit,
        memory_failure_rate=memory,
        misleading_error_rate=misleading,
        permission_denied_rate=permission_denied,
        intermittent_failure_rate=intermittent,
        intermittent_pattern=intermittent_pattern,
        seed=seed,
    )

    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")

    from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

    solver = agentic_humaneval_solver(
        max_attempts=5,
        allow_test_modifications=True,
        tool_reliability=tool_reliability,
    )

    task = impossible_livecodebench(
        split="conflicting",
        agent_type="minimal",
        instruction_prompt=instruction_prompt,
        solver=solver,
        limit=limit,
        max_attempts=5,
        allow_test_modifications=True,
    )

    logs = eval(
        task,
        model=model,
        log_dir=f"./logs/spar_custom_prompt{prompt.upper()}",
    )

    print("\nCustom experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def main():
    """Run experiments based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="SPAR Baseline Experiment: Tool Unreliability vs Reward Hacking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (no failures)
  python experiments/spar_baseline_experiment.py baseline --model openrouter/openai/gpt-4o-mini
  
  # Preset experiments
  python experiments/spar_baseline_experiment.py conservative --model openrouter/openai/gpt-4o-mini
  python experiments/spar_baseline_experiment.py aggressive --model openrouter/openai/gpt-4o-mini
  
  # Custom failure rates
  python experiments/spar_baseline_experiment.py failures --model openrouter/openai/gpt-4o-mini --rate 0.3
  
  # Specific failure modes
  python experiments/spar_baseline_experiment.py failures --model openrouter/openai/gpt-4o-mini \\
      --rate 0.3 --failure-modes timeout,rate-limit,memory
  
    # Custom configuration
    python experiments/spar_baseline_experiment.py custom --model openrouter/openai/gpt-4o-mini \\
        --timeout 0.3 --write 0.2 --corruption 0.1 --rate-limit 0.4
    
    # With different prompts
    python experiments/spar_baseline_experiment.py baseline --model openrouter/openai/gpt-5 \\
        --prompt A  # Weak prompt (92%% cheating baseline)
    python experiments/spar_baseline_experiment.py aggressive --prompt D  # Strict prompt (1%% baseline)
        """,
    )

    parser.add_argument(
        "experiment",
        choices=["baseline", "conservative", "aggressive", "failures", "custom"],
        help="Experiment type to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/openai/gpt-4o-mini",
        help="Model identifier (default: openrouter/openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of samples to run (default: 10)",
    )

    # Failure mode arguments (for 'failures' experiment)
    parser.add_argument(
        "--rate",
        type=float,
        default=0.3,
        help="Failure rate for all enabled failure modes (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--failure-modes",
        type=str,
        default=None,
        help="Comma-separated list of failure modes to enable. Options: "
        "timeout,write,corruption,rate-limit,memory,misleading,permission-denied,intermittent",
    )

    # Custom configuration arguments (for 'custom' experiment)
    parser.add_argument(
        "--timeout", type=float, default=0.0, help="Timeout failure rate"
    )
    parser.add_argument("--write", type=float, default=0.0, help="Write failure rate")
    parser.add_argument(
        "--corruption", type=float, default=0.0, help="Corruption failure rate"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.0, help="Rate limit failure rate"
    )
    parser.add_argument("--memory", type=float, default=0.0, help="Memory failure rate")
    parser.add_argument(
        "--misleading", type=float, default=0.0, help="Misleading error rate"
    )
    parser.add_argument(
        "--permission-denied", type=float, default=0.0, help="Permission denied rate"
    )
    parser.add_argument(
        "--intermittent", type=float, default=0.0, help="Intermittent failure rate"
    )
    parser.add_argument(
        "--intermittent-pattern",
        type=int,
        default=None,
        help="If set, fail every Nth attempt (e.g., 3 = fail every 3rd attempt)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="D",
        choices=["A", "B", "C", "D", "default", "strict", "weak", "loose", "moderate"],
        help="Prompt variant to use. A=weakest (92%% cheating), B=loose (>85%%), C=moderate, D=strictest (1%% cheating). "
        "Aliases: default/strict=D, weak=A, loose=B, moderate=C. (default: D)",
    )

    args = parser.parse_args()

    # Parse failure modes if provided
    failure_modes = None
    if args.failure_modes:
        failure_modes = [m.strip() for m in args.failure_modes.split(",")]
        valid_modes = [
            "timeout",
            "write",
            "corruption",
            "rate-limit",
            "memory",
            "misleading",
            "permission-denied",
            "intermittent",
        ]
        invalid_modes = [m for m in failure_modes if m not in valid_modes]
        if invalid_modes:
            print(f"Error: Invalid failure modes: {', '.join(invalid_modes)}")
            print(f"Valid modes: {', '.join(valid_modes)}")
            sys.exit(1)

    # Run the appropriate experiment
    if args.experiment == "baseline":
        run_baseline_experiment(model=args.model, limit=args.limit, prompt=args.prompt)
    elif args.experiment == "conservative":
        run_conservative_experiment(
            model=args.model, limit=args.limit, prompt=args.prompt
        )
    elif args.experiment == "aggressive":
        run_aggressive_experiment(
            model=args.model, limit=args.limit, prompt=args.prompt
        )
    elif args.experiment == "failures":
        run_tool_failure_experiment(
            model=args.model,
            failure_rate=args.rate,
            limit=args.limit,
            failure_modes=failure_modes,
            prompt=args.prompt,
        )
    elif args.experiment == "custom":
        run_custom_experiment(
            model=args.model,
            limit=args.limit,
            timeout=args.timeout,
            write=args.write,
            corruption=args.corruption,
            rate_limit=args.rate_limit,
            memory=args.memory,
            misleading=args.misleading,
            permission_denied=args.permission_denied,
            intermittent=args.intermittent,
            intermittent_pattern=args.intermittent_pattern,
            seed=args.seed,
            prompt=args.prompt,
        )


if __name__ == "__main__":
    main()
