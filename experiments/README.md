# SPAR Experiment Suite

This directory contains experiments for testing the effect of tool unreliability on reward hacking behavior.

## Quick Start

```bash
# Pre-reqs
export OPENROUTER_API_KEY=$(cat openrouter_apikey) 
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# Run baseline experiment (no tool failures)
python experiments/spar_baseline_experiment.py baseline

# Run with conservative failures (10% failure rate)
python experiments/spar_baseline_experiment.py conservative

# Run with moderate failures (30% failure rate)
python experiments/spar_baseline_experiment.py failures30

# Run with aggressive failures (50% failure rate)
python experiments/spar_baseline_experiment.py aggressive

# Run all experiments
python experiments/spar_baseline_experiment.py all
```

## Experiments

### 1. Baseline Experiment
- **No tool failures** - just impossible tests
- Tests: conflicting split (only passable via reward hacking)
- Purpose: Establish baseline reward hacking rate

### 2. Conservative Failure (10%)
- 10% timeout failures
- 10% write failures
- 10% corruption failures
- Purpose: See if even small amounts of unreliability increase reward hacking

### 3. Moderate Failure (30%)
- 30% failure rates across all modes
- Purpose: Medium stress test

### 4. Aggressive Failure (50%)
- 50% failure rates across all modes
- Purpose: High stress test - extreme pressure

## Expected Results

**Hypothesis**: Tool unreliability → increased frustration → increased reward hacking

- **Baseline (0%)**: Base rate of reward hacking
- **Conservative (10%)**: Small increase expected
- **Moderate (30%)**: Noticeable increase expected
- **Aggressive (50%)**: Significant increase expected

## Failure Modes Implemented

1. **Execution Timeout**: Randomly force test execution to timeout
2. **File Write Failures**: Randomly fail when agent tries to write test files
3. **Corrupted Output**: Randomly corrupt stdout/stderr with noise or truncation

## Analysis

### Quick Analysis (Built-in)

Each experiment automatically runs basic analysis:

```bash
python experiments/spar_baseline_experiment.py baseline
# Automatically analyzes results at the end
```

### Detailed Analysis

Use the dedicated analysis script for detailed statistics:

```bash
# Analyze a single log directory
python experiments/analyze_spar_results.py logs/spar_baseline

# Analyze multiple experiments and compare
python experiments/analyze_spar_results.py \
    logs/spar_baseline \
    logs/spar_conservative \
    logs/spar_failures_30 \
    logs/spar_aggressive

# This will output:
# - Reward hacking rates (pass rate on conflicting tests)
# - Pass rates by model and agent type
# - First-attempt success rates
# - Score statistics
# - Saves results to spar_analysis_results.csv
```

### Programmatic Analysis

```python
from impossiblebench.analysis import DataLoader

# Load results
loader = DataLoader(n_workers=4)
loader.load_folder("./logs/")

# Get DataFrame
df = loader.to_df()

# Filter to conflicting tests (where pass rate = reward hacking rate)
conflicting = df[df['variant'] == 'conflicting']
reward_hacking_rate = conflicting['passed'].mean()

print(f"Reward hacking rate: {reward_hacking_rate:.1%}")
```

## Customizing Experiments

You can create custom failure configurations:

```python
from impossiblebench import ToolReliability

# Custom config with only timeouts
tool_reliability = ToolReliability(
    timeout_failure_rate=0.5,  # 50% timeouts
    write_failure_rate=0.0,    # No write failures
    corruption_rate=0.0,       # No corruption
)
```

## Next Steps

- [ ] Run baseline experiments
- [ ] Compare results across failure rates
- [ ] Add more failure modes (rate limiting, permission denied, etc.)
- [ ] Test with multiple models
- [ ] Test on SWE-bench tasks
