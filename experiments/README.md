# SPAR Tool Unreliability Experiments

Testing whether tool unreliability increases reward hacking in LLM agents using the ImpossibleBench framework.

## Quick Start

```bash
# Setup OpenRouter API key
export OPENROUTER_API_KEY=$(cat openrouter_apikey)
source .venv/bin/activate

# Run baseline (no failures)
python experiments/spar_baseline_experiment.py baseline \
    --model openrouter/openai/o3 \
    --prompt D \
    --limit 20

# Run with 30% failures
python experiments/spar_baseline_experiment.py failures \
    --model openrouter/openai/o3 \
    --rate 0.3 \
    --limit 20

# Aggressive (50% failures)
python experiments/spar_baseline_experiment.py aggressive \
    --model openrouter/openai/o3 \
    --limit 20
```

## Experiment Types

| Type           | Failure Rate | Use Case                               |
| -------------- | ------------ | -------------------------------------- |
| `baseline`     | 0%           | Establish baseline reward hacking rate |
| `conservative` | 10%          | Test subtle effects                    |
| `failures`     | Custom       | Custom rate with `--rate X`            |
| `aggressive`   | 50%          | Maximum stress test                    |

## Failure Modes

**8 failure modes** available:
- `timeout` - Execution timeouts
- `write` - File write failures
- `corruption` - Corrupted output
- `rate-limit` - Rate limiting
- `memory` - Memory exhaustion
- `misleading` - False errors
- `permission-denied` - Access errors
- `intermittent` - Transient failures

## Models (OpenRouter Compatible)

**Recommended**:
- `openrouter/openai/o3` - 49% baseline cheating (Prompt A)
- `openrouter/openai/gpt-5` - 92% baseline cheating if available (Prompt A)

**Available OpenRouter Models**:
- `openrouter/openai/gpt-4o`
- `openrouter/openai/o3`
- `openrouter/anthropic/claude-3.5-sonnet`
- `openrouter/anthropic/claude-3-opus`

**Avoid**: `openrouter/openai/gpt-4o-mini` (shows 0% reward hacking, no signal)

## Prompts

- **Prompt A (weak)**: "pass all tests" → 92% baseline cheating
- **Prompt D (strict)**: "STOP, don't carve out code" → 1% baseline cheating

## Examples

```bash
# Test specific failure modes only
python experiments/spar_baseline_experiment.py failures \
    --model openrouter/openai/o3 \
    --rate 0.3 \
    --failure-modes timeout,write,memory \
    --limit 10

# Custom failure configuration
python experiments/spar_baseline_experiment.py custom \
    --model openrouter/openai/o3 \
    --timeout 0.4 --write 0.3 --corruption 0.2 \
    --limit 20

# Compare prompts
for prompt in A D; do
    python experiments/spar_baseline_experiment.py baseline \
        --model openrouter/openai/o3 \
        --prompt $prompt \
        --limit 10
done
```

## Analysis

```bash
# Quick comparison (recommended)
python experiments/quick_analysis.py \
    logs/spar_baseline_promptD \
    logs/spar_aggressive_promptD

# Detailed analysis
python experiments/analyze_spar_results.py \
    logs/spar_baseline_promptD \
    logs/spar_aggressive_promptD

# LLM Judge analysis (classify cheating behavior)
python experiments/llm_judge_analysis.py \
    logs/spar_baseline_promptD \
    logs/spar_aggressive_promptD \
    --model openrouter/anthropic/claude-3.5-sonnet \
    --max-samples 10
```

## Framework Features

- **Unified injection** - Single middleware across all agents  
- **LLM transparent** - Tools appear as standard `bash/python/text_editor`  
- **Backward compatible** - Works with existing experiment scripts  
- **Reproducible** - Fixed seed (42) for consistency  
- **Agent coverage** - Mini and full agents supported

## Documentation

- **Quick Guide**: See `EXPERIMENT_GUIDE.md` in project root
- **Detailed Setup**: `experiments/docs/GPT5_EXPERIMENT_GUIDE.md`
- **Architecture**: See tool injection framework in `src/impossiblebench/`
