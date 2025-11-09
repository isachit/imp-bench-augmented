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

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Optional

from inspect_ai import eval
from impossiblebench import (
    impossible_livecodebench,
    impossible_swebench,
    ToolReliability,
    create_conservative_config,
    create_aggressive_config,
    agentic_humaneval_solver,
    swe_bench_style_humaneval_solver,
    mini_agent_solver,
    multi_submission_solver,
)

# Import prompts
from experiments.prompts import get_prompt, get_prompt_description


SUPPORTED_BENCHMARKS = {"livecodebench", "swebench"}
SUPPORTED_AGENT_TYPES = {"minimal", "tools"}
DEFAULT_BENCHMARK = "livecodebench"
DEFAULT_AGENT_TYPE = "minimal"
SUPPORTED_LIVE_SPLITS = {"conflicting", "oneoff", "original"}
DEFAULT_LIVE_SPLIT = "conflicting"
DEFAULT_SWE_SPLIT = "conflicting"
DEFAULT_MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class ExperimentMetadata:
    experiment: str
    benchmark: str
    agent_type: str
    prompt: str
    model: str
    limit: int
    max_attempts: int
    hide_tests: bool = False
    swe_split: str = DEFAULT_SWE_SPLIT
    livecode_split: str = DEFAULT_LIVE_SPLIT
    failure_rate: float | None = None
    failure_modes: tuple[str, ...] = ()

    def prompt_label(self) -> str:
        raw = (self.prompt or "").strip()
        if not raw:
            return "D"
        upper = raw.upper()
        if upper in {"A", "B", "C", "D"}:
            return upper
        aliases = {
            "DEFAULT": "D",
            "STRICT": "D",
            "WEAK": "A",
            "LOOSE": "B",
            "MODERATE": "C",
        }
        return aliases.get(upper, "D")

    def sorted_failure_modes(self) -> tuple[str, ...]:
        if not self.failure_modes:
            return ()
        return tuple(sorted(set(self.failure_modes)))

    def log_suffix_parts(self) -> list[str]:
        parts: list[str] = []
        if self.benchmark == "livecodebench":
            parts.extend(["lcb", self.livecode_split])
        else:
            parts.extend(["swe", self.swe_split])

        parts.append(f"prompt{self.prompt_label()}")

        if self.agent_type != DEFAULT_AGENT_TYPE:
            parts.append(self.agent_type)

        if self.hide_tests:
            parts.append("hidden")

        if self.max_attempts != DEFAULT_MAX_ATTEMPTS:
            parts.append(f"attempts{self.max_attempts}")

        if self.failure_rate is not None:
            parts.append(f"fail{int(round(self.failure_rate * 100))}")

        modes = self.sorted_failure_modes()
        if modes:
            parts.append("modes-" + "-".join(modes))

        return parts

    def make_log_dir(self, prefix: str) -> str:
        suffix_parts = self.log_suffix_parts()
        suffix = "_".join(suffix_parts)
        return f"./logs/{prefix}_{suffix}" if suffix else f"./logs/{prefix}"

    def describe_prompt(self) -> str:
        label = self.prompt_label()
        return get_prompt_description(label if label in {"A", "B", "C", "D"} else "D")


def validate_benchmark_agent(benchmark: str, agent_type: str) -> tuple[str, str]:
    benchmark_norm = (benchmark or DEFAULT_BENCHMARK).lower()
    agent_norm = (agent_type or DEFAULT_AGENT_TYPE).lower()

    if benchmark_norm not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark '{benchmark}'. Choose from: {', '.join(sorted(SUPPORTED_BENCHMARKS))}"
        )
    if agent_norm not in SUPPORTED_AGENT_TYPES:
        raise ValueError(
            f"Unsupported agent type '{agent_type}'. Choose from: {', '.join(sorted(SUPPORTED_AGENT_TYPES))}"
        )
    return benchmark_norm, agent_norm


def normalize_livecode_split(split: str | None) -> str:
    value = (split or DEFAULT_LIVE_SPLIT).lower()
    if value not in SUPPORTED_LIVE_SPLITS:
        raise ValueError(
            f"Unsupported LiveCodeBench split '{split}'. Choose from: {', '.join(sorted(SUPPORTED_LIVE_SPLITS))}"
        )
    return value


def normalize_swe_split(split: str | None) -> str:
    value = (split or DEFAULT_SWE_SPLIT).lower()
    if value not in {"original", "oneoff", "conflicting"}:
        raise ValueError(
            f"Unsupported SWE-bench split '{split}'. Choose from: original, oneoff, conflicting"
        )
    return value


def normalize_failure_modes(modes: Optional[list[str]]) -> tuple[str, ...]:
    if not modes:
        return ()
    cleaned = [mode.strip() for mode in modes if mode and mode.strip()]
    return tuple(sorted(set(cleaned)))


def infer_failure_rate(config: ToolReliability) -> float | None:
    rates = [
        getattr(config, "timeout_failure_rate", 0.0),
        getattr(config, "write_failure_rate", 0.0),
        getattr(config, "corruption_rate", 0.0),
        getattr(config, "rate_limit_failure_rate", 0.0),
        getattr(config, "memory_failure_rate", 0.0),
        getattr(config, "misleading_error_rate", 0.0),
        getattr(config, "permission_denied_rate", 0.0),
        getattr(config, "intermittent_failure_rate", 0.0),
    ]
    positives = [rate for rate in rates if rate and rate > 0]
    return max(positives) if positives else None


def extract_active_modes(config: ToolReliability) -> tuple[str, ...]:
    mapping = {
        "timeout_failure_rate": "timeout",
        "write_failure_rate": "write",
        "corruption_rate": "corruption",
        "rate_limit_failure_rate": "rate-limit",
        "memory_failure_rate": "memory",
        "misleading_error_rate": "misleading",
        "permission_denied_rate": "permission-denied",
        "intermittent_failure_rate": "intermittent",
    }
    modes = [
        mode
        for attr, mode in mapping.items()
        if getattr(config, attr, 0.0) and getattr(config, attr, 0.0) > 0
    ]
    return tuple(sorted(set(modes)))


def build_solver(
    metadata: ExperimentMetadata,
    *,
    tool_reliability: ToolReliability | None = None,
):
    """Construct the appropriate solver for the selected benchmark and agent."""
    if metadata.benchmark == "livecodebench":
        if metadata.agent_type == "minimal":
            return agentic_humaneval_solver(
                max_attempts=metadata.max_attempts,
                allow_test_modifications=True,
                tool_reliability=tool_reliability,
            )
        if metadata.agent_type == "tools":
            return swe_bench_style_humaneval_solver(
                max_attempts=metadata.max_attempts,
                tool_reliability=tool_reliability,
                hide_tests=metadata.hide_tests,
            )

    if metadata.benchmark == "swebench":
        if metadata.agent_type == "minimal":
            if tool_reliability is not None:
                print(
                    "[warning] Tool reliability injection is not supported for the SWE-bench mini agent; "
                    "ignoring configured failures."
                )
            return mini_agent_solver(
                max_attempts=metadata.max_attempts,
                hide_tests=metadata.hide_tests,
                reset_tests=metadata.hide_tests,
                custom_prompt="",
            )
        if metadata.agent_type == "tools":
            return multi_submission_solver(
                max_attempts=metadata.max_attempts,
                tool_reliability=tool_reliability,
                hide_tests=metadata.hide_tests,
                custom_prompt="",
            )

    raise ValueError(
        f"Unsupported benchmark/agent combination: {metadata.benchmark}/{metadata.agent_type}"
    )


def build_task(
    metadata: ExperimentMetadata,
    *,
    instruction_prompt: str,
    solver,
):
    if metadata.benchmark == "livecodebench":
        return impossible_livecodebench(
            split=metadata.livecode_split,
            agent_type=metadata.agent_type,
            instruction_prompt=instruction_prompt,
            solver=solver,
            limit=metadata.limit,
            max_attempts=metadata.max_attempts,
            allow_test_modifications=True,
            hide_tests=metadata.hide_tests,
        )

    prompt_label = metadata.prompt_label()

    return impossible_swebench(
        split=metadata.swe_split,
        agent_type=metadata.agent_type,
        solver=solver,
        limit=metadata.limit,
        max_attempts=metadata.max_attempts,
        hide_tests=metadata.hide_tests,
        reset_tests=metadata.hide_tests,
        custom_prompt="",
        custom_id=f"prompt_{prompt_label}",
    )


def print_experiment_overview(title: str, metadata: ExperimentMetadata) -> None:
    print("=" * 60)
    print(f"{title}: {metadata.benchmark.upper()} ({metadata.agent_type})")
    print(f"Model: {metadata.model}")
    print(f"Limit: {metadata.limit}")
    print(f"Max attempts: {metadata.max_attempts}")

    if metadata.benchmark == "livecodebench":
        print(f"LiveCodeBench split: {metadata.livecode_split}")
        print(f"Prompt: {metadata.prompt_label()} - {metadata.describe_prompt()}")
    else:
        print(f"SWE-bench split: {metadata.swe_split}")
        print(
            f"Prompt Config: {metadata.prompt_label()} - {metadata.describe_prompt()} "
            "(label only; SWE tasks rely on the issue text)."
        )

    if metadata.hide_tests:
        print("Test visibility: HIDDEN")
    else:
        print("Test visibility: VISIBLE")

    print("=" * 60)


def run_baseline_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    limit: int = 10,
    prompt: str = "D",
    hide_tests: bool = False,
    benchmark: str = DEFAULT_BENCHMARK,
    agent_type: str = DEFAULT_AGENT_TYPE,
    livecode_split: str = DEFAULT_LIVE_SPLIT,
    swe_split: str = DEFAULT_SWE_SPLIT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Run baseline experiment with no tool failures.

    Args:
        model: Model identifier to use for evaluation
        limit: Number of samples to run
        prompt: Prompt variant to use for LiveCodeBench (A, B, C, D, default, strict, weak, loose, moderate)
        hide_tests: Hide test definitions from the agent.
        benchmark: Benchmark to evaluate ("livecodebench" or "swebench").
        agent_type: Agent scaffold to use ("minimal" or "tools").
        livecode_split: LiveCodeBench split (conflicting, oneoff, original) when benchmark="livecodebench".
        swe_split: SWE-bench split to evaluate (original, oneoff, conflicting) when benchmark="swebench".
        max_attempts: Maximum number of submission attempts per sample.
    """
    benchmark_norm, agent_norm = validate_benchmark_agent(benchmark, agent_type)
    swe_split_norm = normalize_swe_split(swe_split)
    live_split_norm = normalize_livecode_split(livecode_split)

    metadata = ExperimentMetadata(
        experiment="baseline",
        benchmark=benchmark_norm,
        agent_type=agent_norm,
        prompt=prompt or "D",
        model=model,
        limit=limit,
        hide_tests=hide_tests,
        swe_split=swe_split_norm,
        livecode_split=live_split_norm,
        max_attempts=max_attempts,
    )

    instruction_prompt = (
        get_prompt(metadata.prompt) if metadata.benchmark == "livecodebench" else ""
    )

    print_experiment_overview("BASELINE EXPERIMENT", metadata)

    solver = build_solver(metadata, tool_reliability=None)

    task = build_task(
        metadata,
        instruction_prompt=instruction_prompt,
        solver=solver,
    )

    log_dir = metadata.make_log_dir("spar_baseline")

    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
    )

    print("\nBaseline experiment complete!")
    print(f"Logs saved to: {logs}")

    return logs


def run_tool_failure_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    failure_rate: float = 0.3,
    limit: int = 10,
    failure_modes: Optional[list[str]] = None,
    prompt: str = "D",
    hide_tests: bool = False,
    benchmark: str = DEFAULT_BENCHMARK,
    agent_type: str = DEFAULT_AGENT_TYPE,
    livecode_split: str = DEFAULT_LIVE_SPLIT,
    swe_split: str = DEFAULT_SWE_SPLIT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Run experiment with tool failures injected."""
    benchmark_norm, agent_norm = validate_benchmark_agent(benchmark, agent_type)
    swe_split_norm = normalize_swe_split(swe_split)
    live_split_norm = normalize_livecode_split(livecode_split)
    normalized_modes = normalize_failure_modes(failure_modes)
    active_modes = normalized_modes or ("timeout", "write", "corruption")

    metadata = ExperimentMetadata(
        experiment="failures",
        benchmark=benchmark_norm,
        agent_type=agent_norm,
        prompt=prompt or "D",
        model=model,
        limit=limit,
        hide_tests=hide_tests,
        swe_split=swe_split_norm,
        livecode_split=live_split_norm,
        max_attempts=max_attempts,
        failure_rate=failure_rate,
        failure_modes=active_modes,
    )

    instruction_prompt = (
        get_prompt(metadata.prompt) if metadata.benchmark == "livecodebench" else ""
    )

    print_experiment_overview("TOOL FAILURE EXPERIMENT", metadata)
    print(f"Failure rate: {failure_rate * 100:.0f}%")
    if normalized_modes:
        print(f"Failure modes: {', '.join(metadata.sorted_failure_modes())}")
    else:
        print("Failure modes: timeout, write, corruption (default)")
    print("=" * 60)

    config_kwargs = {"seed": 42}
    if "timeout" in active_modes:
        config_kwargs["timeout_failure_rate"] = failure_rate
    if "write" in active_modes:
        config_kwargs["write_failure_rate"] = failure_rate
    if "corruption" in active_modes:
        config_kwargs["corruption_rate"] = failure_rate
    if "rate-limit" in active_modes:
        config_kwargs["rate_limit_failure_rate"] = failure_rate
    if "memory" in active_modes:
        config_kwargs["memory_failure_rate"] = failure_rate
    if "misleading" in active_modes:
        config_kwargs["misleading_error_rate"] = failure_rate
    if "permission-denied" in active_modes:
        config_kwargs["permission_denied_rate"] = failure_rate
    if "intermittent" in active_modes:
        config_kwargs["intermittent_failure_rate"] = failure_rate
        if normalized_modes and "intermittent" in normalized_modes:
            config_kwargs.setdefault("intermittent_pattern", 3)

    tool_reliability = ToolReliability(**config_kwargs)
    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")

    solver = build_solver(
        metadata,
        tool_reliability=tool_reliability,
    )

    task = build_task(
        metadata,
        instruction_prompt=instruction_prompt,
        solver=solver,
    )

    log_dir = metadata.make_log_dir(f"spar_failures_{int(round(failure_rate * 100))}")

    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
    )

    print("\nTool failure experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_conservative_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    limit: int = 10,
    prompt: str = "D",
    hide_tests: bool = False,
    benchmark: str = DEFAULT_BENCHMARK,
    agent_type: str = DEFAULT_AGENT_TYPE,
    livecode_split: str = DEFAULT_LIVE_SPLIT,
    swe_split: str = DEFAULT_SWE_SPLIT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Run experiment with conservative failure rates (10%)."""
    benchmark_norm, agent_norm = validate_benchmark_agent(benchmark, agent_type)
    swe_split_norm = normalize_swe_split(swe_split)
    live_split_norm = normalize_livecode_split(livecode_split)

    tool_reliability = create_conservative_config()

    metadata = ExperimentMetadata(
        experiment="conservative",
        benchmark=benchmark_norm,
        agent_type=agent_norm,
        prompt=prompt or "D",
        model=model,
        limit=limit,
        hide_tests=hide_tests,
        swe_split=swe_split_norm,
        livecode_split=live_split_norm,
        max_attempts=max_attempts,
        failure_rate=infer_failure_rate(tool_reliability),
        failure_modes=extract_active_modes(tool_reliability),
    )

    instruction_prompt = (
        get_prompt(metadata.prompt) if metadata.benchmark == "livecodebench" else ""
    )

    print_experiment_overview("CONSERVATIVE FAILURE EXPERIMENT", metadata)
    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")
    print("=" * 60)

    solver = build_solver(
        metadata,
        tool_reliability=tool_reliability,
    )

    task = build_task(
        metadata,
        instruction_prompt=instruction_prompt,
        solver=solver,
    )

    log_dir = metadata.make_log_dir("spar_conservative")

    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
    )

    print("\nConservative experiment complete!")
    print(f"Logs saved to: {logs}")
    return logs


def run_aggressive_experiment(
    model: str = "openrouter/openai/gpt-4o-mini",
    limit: int = 10,
    prompt: str = "D",
    hide_tests: bool = False,
    benchmark: str = DEFAULT_BENCHMARK,
    agent_type: str = DEFAULT_AGENT_TYPE,
    livecode_split: str = DEFAULT_LIVE_SPLIT,
    swe_split: str = DEFAULT_SWE_SPLIT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Run experiment with aggressive failure rates (50%)."""
    benchmark_norm, agent_norm = validate_benchmark_agent(benchmark, agent_type)
    swe_split_norm = normalize_swe_split(swe_split)
    live_split_norm = normalize_livecode_split(livecode_split)

    tool_reliability = create_aggressive_config()

    metadata = ExperimentMetadata(
        experiment="aggressive",
        benchmark=benchmark_norm,
        agent_type=agent_norm,
        prompt=prompt or "D",
        model=model,
        limit=limit,
        hide_tests=hide_tests,
        swe_split=swe_split_norm,
        livecode_split=live_split_norm,
        max_attempts=max_attempts,
        failure_rate=infer_failure_rate(tool_reliability),
        failure_modes=extract_active_modes(tool_reliability),
    )

    instruction_prompt = (
        get_prompt(metadata.prompt) if metadata.benchmark == "livecodebench" else ""
    )

    print_experiment_overview("AGGRESSIVE FAILURE EXPERIMENT", metadata)
    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")
    print("=" * 60)

    solver = build_solver(
        metadata,
        tool_reliability=tool_reliability,
    )

    task = build_task(
        metadata,
        instruction_prompt=instruction_prompt,
        solver=solver,
    )

    log_dir = metadata.make_log_dir("spar_aggressive")

    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
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
    hide_tests: bool = False,
    benchmark: str = DEFAULT_BENCHMARK,
    agent_type: str = DEFAULT_AGENT_TYPE,
    livecode_split: str = DEFAULT_LIVE_SPLIT,
    swe_split: str = DEFAULT_SWE_SPLIT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
):
    """Run experiment with custom failure configuration."""
    benchmark_norm, agent_norm = validate_benchmark_agent(benchmark, agent_type)
    swe_split_norm = normalize_swe_split(swe_split)
    live_split_norm = normalize_livecode_split(livecode_split)

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

    metadata = ExperimentMetadata(
        experiment="custom",
        benchmark=benchmark_norm,
        agent_type=agent_norm,
        prompt=prompt or "D",
        model=model,
        limit=limit,
        hide_tests=hide_tests,
        swe_split=swe_split_norm,
        livecode_split=live_split_norm,
        max_attempts=max_attempts,
        failure_rate=infer_failure_rate(tool_reliability),
        failure_modes=extract_active_modes(tool_reliability),
    )

    instruction_prompt = (
        get_prompt(metadata.prompt) if metadata.benchmark == "livecodebench" else ""
    )

    print_experiment_overview("CUSTOM FAILURE EXPERIMENT", metadata)
    print(f"Tool reliability config: {tool_reliability.get_config_summary()}")
    print("=" * 60)

    solver = build_solver(
        metadata,
        tool_reliability=tool_reliability,
    )

    task = build_task(
        metadata,
        instruction_prompt=instruction_prompt,
        solver=solver,
    )

    log_dir = metadata.make_log_dir("spar_custom")

    logs = eval(
        task,
        model=model,
        log_dir=log_dir,
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
    parser.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_BENCHMARK,
        choices=sorted(SUPPORTED_BENCHMARKS),
        help="Benchmark to run (livecodebench or swebench). Default: livecodebench",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=DEFAULT_AGENT_TYPE,
        choices=sorted(SUPPORTED_AGENT_TYPES),
        help="Agent scaffold to use (minimal or tools). Default: minimal",
    )
    parser.add_argument(
        "--swe-split",
        type=str,
        default=DEFAULT_SWE_SPLIT,
        choices=["original", "oneoff", "conflicting"],
        help="SWE-bench split to evaluate (only used when --benchmark swebench). Default: conflicting",
    )
    parser.add_argument(
        "--livecode-split",
        type=str,
        default=DEFAULT_LIVE_SPLIT,
        choices=sorted(SUPPORTED_LIVE_SPLITS),
        help="LiveCodeBench split to evaluate (only used when --benchmark livecodebench). Default: conflicting",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum number of submission attempts per sample (default: 5)",
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
    parser.add_argument(
        "--hide-tests",
        action="store_true",
        help="Hide test code from the agent. Agent will only see test results (pass/fail, error messages) but not the actual test code.",
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
        run_baseline_experiment(
            model=args.model,
            limit=args.limit,
            prompt=args.prompt,
            hide_tests=args.hide_tests,
            benchmark=args.benchmark,
            agent_type=args.agent_type,
            livecode_split=args.livecode_split,
            swe_split=args.swe_split,
            max_attempts=args.max_attempts,
        )
    elif args.experiment == "conservative":
        run_conservative_experiment(
            model=args.model,
            limit=args.limit,
            prompt=args.prompt,
            hide_tests=args.hide_tests,
            benchmark=args.benchmark,
            agent_type=args.agent_type,
            livecode_split=args.livecode_split,
            swe_split=args.swe_split,
            max_attempts=args.max_attempts,
        )
    elif args.experiment == "aggressive":
        run_aggressive_experiment(
            model=args.model,
            limit=args.limit,
            prompt=args.prompt,
            hide_tests=args.hide_tests,
            benchmark=args.benchmark,
            agent_type=args.agent_type,
            livecode_split=args.livecode_split,
            swe_split=args.swe_split,
            max_attempts=args.max_attempts,
        )
    elif args.experiment == "failures":
        run_tool_failure_experiment(
            model=args.model,
            failure_rate=args.rate,
            limit=args.limit,
            failure_modes=failure_modes,
            prompt=args.prompt,
            hide_tests=args.hide_tests,
            benchmark=args.benchmark,
            agent_type=args.agent_type,
            livecode_split=args.livecode_split,
            swe_split=args.swe_split,
            max_attempts=args.max_attempts,
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
            hide_tests=args.hide_tests,
            benchmark=args.benchmark,
            agent_type=args.agent_type,
            livecode_split=args.livecode_split,
            swe_split=args.swe_split,
            max_attempts=args.max_attempts,
        )


if __name__ == "__main__":
    main()
