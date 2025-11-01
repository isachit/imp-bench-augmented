#!/usr/bin/env python3
"""Analysis script for SPAR tool unreliability experiments.

This script analyzes the results from SPAR experiments to compare baseline
vs. tool failure experiments and measure reward hacking rates.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from impossiblebench.analysis import DataLoader


def analyze_reward_hacking_rates(loader: DataLoader):
    """Analyze reward hacking rates from different experiments.
    
    Args:
        loader: DataLoader instance with loaded results
    """
    df = loader.to_df()
    
    if df.empty:
        print("No data loaded!")
        return
    
    print("=" * 80)
    print("SPAR EXPERIMENT ANALYSIS: Tool Unreliability vs Reward Hacking")
    print("=" * 80)
    
    # Summary statistics
    summary = loader.get_summary()
    print("\n📊 Overall Summary:")
    print(f"  Total samples: {summary.get('total_samples', 0)}")
    print(f"  Overall pass rate: {summary.get('pass_rate', 0):.1%}")
    
    # Filter to conflicting tests only (these are impossible to pass legitimately)
    conflicting_df = df[df['variant'] == 'conflicting']
    
    if len(conflicting_df) == 0:
        print("\n⚠️  No conflicting variant results found. Make sure you ran experiments with split='conflicting'")
        return
    
    print(f"\n📊 Conflicting Tests Analysis ({len(conflicting_df)} samples):")
    print(f"  Pass rate (reward hacking rate): {conflicting_df['passed'].mean():.1%}")
    print(f"  Number that passed: {conflicting_df['passed'].sum()}/{len(conflicting_df)}")
    
    # Compare by different experiment types
    # Extract experiment type from model name or log path
    # This assumes different model names = different experiments
    print("\n📈 Pass Rates by Model:")
    model_rates = conflicting_df.groupby('model')['passed'].agg(['count', 'mean']).round(3)
    model_rates.columns = ['n_samples', 'pass_rate']
    for model, row in model_rates.iterrows():
        print(f"  {model}: {row['pass_rate']:.1%} ({int(row['n_samples'])} samples)")
    
    # Compare agent types
    print("\n🤖 Pass Rates by Agent Type:")
    agent_rates = conflicting_df.groupby('agent_type')['passed'].agg(['count', 'mean']).round(3)
    agent_rates.columns = ['n_samples', 'pass_rate']
    for agent, row in agent_rates.iterrows():
        print(f"  {agent}: {row['pass_rate']:.1%} ({int(row['n_samples'])} samples)")
    
    # First-pass analysis (attempt 1 success)
    if 'first_pass' in conflicting_df.columns:
        print("\n🎯 First-Pass Success Rate:")
        first_pass_rate = conflicting_df['first_pass'].mean()
        print(f"  First attempt success: {first_pass_rate:.1%}")
        print(f"  Success on first try: {conflicting_df['first_pass'].sum()}/{len(conflicting_df)}")
    
    # Detailed breakdown
    print("\n📋 Detailed Breakdown:")
    if 'model' in conflicting_df.columns and 'agent_type' in conflicting_df.columns:
        detailed = conflicting_df.groupby(['model', 'agent_type'])['passed'].agg(['count', 'mean'])
        detailed.columns = ['n_samples', 'pass_rate']
        print(detailed)
    
    # Save results
    output_file = "spar_analysis_results.csv"
    conflicting_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")


def compare_failure_rates(loader: DataLoader):
    """Compare pass rates across different failure configurations.
    
    This function looks for patterns in the data that might indicate
    different failure injection rates.
    """
    df = loader.to_df()
    conflicting_df = df[df['variant'] == 'conflicting']
    
    if len(conflicting_df) == 0:
        return
    
    print("\n🔬 Tool Failure Impact Analysis:")
    print("-" * 80)
    
    # If we have metadata about failure rates, use it
    # Otherwise, we can infer from patterns in the data
    
    # Compare samples by any available metadata
    if 'prompt_id' in conflicting_df.columns:
        prompt_rates = conflicting_df.groupby('prompt_id')['passed'].mean()
        print("\nPass rates by prompt ID:")
        for prompt_id, rate in prompt_rates.items():
            print(f"  {prompt_id}: {rate:.1%}")
    
    # Look at average scores
    if 'score' in conflicting_df.columns:
        print("\n📊 Score Statistics:")
        print(f"  Mean score: {conflicting_df['score'].mean():.2f}")
        print(f"  Median score: {conflicting_df['score'].median():.2f}")
        print(f"  Std dev: {conflicting_df['score'].std():.2f}")
    
    if 'first_score' in conflicting_df.columns:
        first_scores = conflicting_df['first_score'].dropna()
        if len(first_scores) > 0:
            print(f"\n  First-attempt mean score: {first_scores.mean():.2f}")


def main():
    """Main analysis function."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_spar_results.py <log_directory> [additional_dirs...]")
        print("\nExample:")
        print("  python analyze_spar_results.py logs/")
        print("  python analyze_spar_results.py logs/baseline logs/failures logs/conservative")
        sys.exit(1)
    
    log_dirs = sys.argv[1:]
    
    # Create DataLoader
    loader = DataLoader(n_workers=4)
    
    # Load from specified directories
    print(f"Loading results from {len(log_dirs)} directories...")
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"  Loading: {log_dir}")
            loader.load_folder(log_dir)
        else:
            print(f"  ⚠️  Directory not found: {log_dir}")
    
    if len(loader.results) == 0:
        print("\n❌ No results loaded! Make sure the directories contain .eval files.")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(loader.results)} evaluation results")
    
    # Run analyses
    analyze_reward_hacking_rates(loader)
    compare_failure_rates(loader)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
