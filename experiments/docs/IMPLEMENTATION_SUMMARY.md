# Implementation Summary: Top 3 Failure Modes

## What Was Implemented

### 1. Core Module: `tool_unreliability.py`

Created `src/impossiblebench/tool_unreliability.py` with:

- **`ToolReliability` class**: Main class for managing failure injection
  - Configurable failure rates for each failure type
  - Seed support for reproducibility
  - Methods to inject each failure type

- **Helper functions**:
  - `create_baseline_config()` - No failures
  - `create_conservative_config()` - 10% failure rates
  - `create_aggressive_config()` - 50% failure rates
  - `create_timeout_only_config()` - Only timeouts
  - `create_custom_config()` - Fully customizable

### 2. Three Failure Modes Implemented

#### Failure Mode 1: Execution Timeout
- Randomly reduces timeout from 30s to 1s
- Forces execution to timeout
- **Impact**: Completely blocks legitimate progress
- **Location**: `livecodebench_agent_mini.py` lines 113-118

#### Failure Mode 2: File Write Failures
- Randomly fails when agent tries to write test files
- Returns "Permission denied" error
- **Impact**: Agent thinks it saved code but didn't
- **Location**: `livecodebench_agent_mini.py` lines 102-124

#### Failure Mode 3: Corrupted Output
- Two corruption types:
  - `fake_success`: Returns fake success with corrupted output
  - `corrupted_output`: Adds noise or truncates actual output
- **Impact**: Agent gets confusing/incorrect feedback
- **Location**: `livecodebench_agent_mini.py` lines 135-138

### 3. Integration with Agent

Modified `livecodebench_agent_mini.py`:
- Added `tool_reliability` parameter to `agentic_humaneval_solver()`
- Integrated failure injection at three key points
- Tracks failure types in attempt history
- Maintains backward compatibility (optional parameter)

### 4. Experiment Script

Created `experiments/spar_baseline_experiment.py`:
- Baseline experiment (no failures)
- Conservative experiment (10% failures)
- Moderate experiment (30% failures)
- Aggressive experiment (50% failures)
- CLI interface for running individual or all experiments

### 5. Documentation

- Updated `__init__.py` to export tool reliability classes
- Created `experiments/README.md` with usage instructions
- Created comprehensive failure mode documentation in `spar_integration_plan.md`

## How to Use

### Basic Usage

```python
from impossiblebench import (
    impossible_livecodebench,
    ToolReliability,
)

# Create failure configuration
tool_reliability = ToolReliability(
    timeout_failure_rate=0.3,   # 30% timeouts
    write_failure_rate=0.3,     # 30% write failures
    corruption_rate=0.3,        # 30% corruption
)

# Create custom solver with failures
from impossiblebench.livecodebench_agent_mini import agentic_humaneval_solver

solver = agentic_humaneval_solver(
    max_attempts=5,
    tool_reliability=tool_reliability,  # Inject failures here
)

# Create task with custom solver
task = impossible_livecodebench(
    split="conflicting",      # Impossible tests
    agent_type="minimal",
    solver=solver,
)
```

### Running Experiments

```bash
# Baseline (no failures)
python experiments/spar_baseline_experiment.py baseline

# With failures
python experiments/spar_baseline_experiment.py failures30

# All experiments
python experiments/spar_baseline_experiment.py all
```

## Testing

To test the implementation:

```bash
cd impossiblebench

# Test tool unreliability module
python -c "from impossiblebench import ToolReliability; print(ToolReliability(0.5, 0.5, 0.5).get_config_summary())"

# Test import
python -c "from impossiblebench import create_conservative_config; print('Success')"

# Run a small experiment
python experiments/spar_baseline_experiment.py baseline
```

## Research Questions

With this implementation, you can now answer:

1. **Does tool unreliability increase reward hacking?**
   - Compare baseline vs. failure experiments
   - Reward hacking rate = pass rate on conflicting tests

2. **What failure rates are needed?**
   - Test 10%, 30%, 50% rates
   - Find threshold where behavior changes

3. **Which failure type is most impactful?**
   - Can isolate failures (timeout only, etc.)

## Next Steps

1. **Run baseline experiments** to establish baseline reward hacking rate
2. **Compare with failure experiments** to see if unreliability increases hacking
3. **Add more failure modes** (rate limiting, permission denied, etc.)
4. **Test with multiple models** (GPT-4o, Claude, cheaper models)
5. **Analyze results** using existing DataLoader tools

## Files Modified

- ✅ `src/impossiblebench/tool_unreliability.py` (NEW)
- ✅ `src/impossiblebench/__init__.py` (UPDATED)
- ✅ `src/impossiblebench/livecodebench_agent_mini.py` (UPDATED)
- ✅ `experiments/spar_baseline_experiment.py` (NEW)
- ✅ `experiments/README.md` (NEW)
- ✅ `spar_integration_plan.md` (UPDATED)
- ✅ `IMPLEMENTATION_SUMMARY.md` (NEW)

## Success Criteria

The implementation is complete when:

- ✅ Can inject three failure modes
- ✅ Can configure failure rates
- ✅ Can run baseline vs. failure experiments
- ✅ Backward compatible (works without failures)
- ✅ Reproducible (seed support)

All criteria met! Ready to run experiments.
