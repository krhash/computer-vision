# tasks/task5_experiment.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Orchestrates Task 5 — Design your own experiment.
#              Runs a hyperparameter sweep on NetTransformer using
#              Fashion MNIST across three dimensions:
#
#   Dimension 1: patch_size  [2, 4, 7, 14]
#   Dimension 2: embed_dim   [16, 32, 48, 96]
#   Dimension 3: depth       [1, 2, 4, 6]
#
#   Strategy: round-robin linear search (hold 2 fixed, sweep 1)
#   Target:   50-100 total runs saved to outputs/task5_results.csv
#
# Hypotheses (5B):
#   H1 (patch_size): Medium patches (4-7) will outperform extremes.
#                    Patch=2 produces too many noisy tokens; patch=14
#                    loses spatial detail critical for clothing textures.
#   H2 (embed_dim):  Larger embeddings (48-96) will perform better.
#                    Fashion MNIST needs more capacity than digits to
#                    distinguish subtle texture/shape differences.
#   H3 (depth):      Depth 2-4 will be the sweet spot. Shallow (1)
#                    underfits; deep (6) overfits on this dataset size
#                    and adds training time with diminishing returns.
#
# Usage:
#   python tasks/task5_experiment.py
#   python tasks/task5_experiment.py --epochs 8 --output-dir ./outputs

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.experiment.experiment_config import ExperimentConfig, ExperimentResult
from src.experiment.experiment_runner import ExperimentRunner
from src.visualization.plotter        import Plotter
from src.utils.device_utils           import get_device
from typing import List


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_OUTPUT_DIR  = "./outputs"
DEFAULT_DATA_DIR    = "./data"
DEFAULT_EPOCHS      = 10     # per run — reduce to 8 if sweep is too slow


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def task5a_print_plan(configs: list) -> None:
    """
    Subtask 5A: Print the experiment plan to stdout.

    Summarises the sweep dimensions, total runs, and strategy
    before execution begins.

    Args:
        configs (list): Generated list of experiment configs.
    """
    print("\n[Task 5A] Experiment Plan")
    print("=" * 60)
    print("  Dataset:    Fashion MNIST (10 clothing categories)")
    print("  Model:      NetTransformer (Vision Transformer)")
    print("  Strategy:   Round-robin linear search")
    print()
    print("  Dimensions:")
    print(f"    1. patch_size  : {ExperimentConfig.PATCH_SIZES}")
    print(f"    2. embed_dim   : {ExperimentConfig.EMBED_DIMS}")
    print(f"    3. depth       : {ExperimentConfig.DEPTHS}")
    print()
    print("  Hypotheses:")
    print("    H1: Medium patches (4-7) will outperform extremes.")
    print("    H2: Larger embed_dim (48-96) will perform better.")
    print("    H3: Depth 2-4 is the sweet spot — deeper overfits.")
    print()

    # Count runs per phase
    phase_counts: dict = {}
    for c in configs:
        phase_counts[c["phase"]] = phase_counts.get(c["phase"], 0) + 1

    print("  Runs per phase:")
    for phase, count in phase_counts.items():
        print(f"    {phase:<25} {count:>3} runs")
    print(f"    {'TOTAL':<25} {len(configs):>3} runs")
    print("=" * 60)


def task5b_print_hypotheses() -> None:
    """
    Subtask 5B: Print formal hypotheses before execution.

    These are discussed in the report after results are collected.
    """
    print("\n[Task 5B] Hypotheses")
    print("=" * 60)
    print("  H1 — Patch Size:")
    print("    Patch=4 or patch=7 will achieve the highest accuracy.")
    print("    Rationale: patch=2 creates 169 tokens — too many for")
    print("    the encoder to find meaningful patterns. patch=14 creates")
    print("    only 4 tokens — too coarse for 10-class clothing detail.")
    print()
    print("  H2 — Embedding Dimension:")
    print("    embed_dim=96 will outperform smaller dimensions.")
    print("    Rationale: Fashion MNIST requires distinguishing subtle")
    print("    textures (shirt vs pullover vs coat). Larger embedding")
    print("    space allows richer feature representations.")
    print()
    print("  H3 — Transformer Depth:")
    print("    depth=2 or depth=4 will be optimal.")
    print("    Rationale: 28x28 images with simple clothing shapes do")
    print("    not require deep attention stacks. depth=6 will overfit")
    print("    and train significantly slower with minimal accuracy gain.")
    print("=" * 60)


def task5c_run_sweep(
    runner:  ExperimentRunner,
    configs: list,
) -> List[ExperimentResult]:
    """
    Subtask 5C: Execute the full hyperparameter sweep.

    Args:
        runner  (ExperimentRunner): Configured runner instance.
        configs (list):             List of experiment config dicts.

    Returns:
        List[ExperimentResult]: All results from the sweep.
    """
    print("\n[Task 5C] Executing sweep...")
    return runner.run(configs)


def print_summary(results: List[ExperimentResult], baseline: float) -> None:
    """
    Prints a summary of sweep results to stdout.

    Shows top 5 configurations and discusses whether hypotheses
    were supported.

    Args:
        results  (List[ExperimentResult]): All sweep results.
        baseline (float):                  Baseline accuracy (%).
    """
    sorted_results = sorted(results, key=lambda r: r.test_accuracy, reverse=True)

    print("\n" + "=" * 60)
    print("  Task 5 — Sweep Summary")
    print("=" * 60)
    print(f"  Baseline accuracy: {baseline:.2f}%")
    print(f"  Best accuracy:     {sorted_results[0].test_accuracy:.2f}%  "
          f"(patch={sorted_results[0].patch_size}, "
          f"embed={sorted_results[0].embed_dim}, "
          f"depth={sorted_results[0].depth})")
    print(f"  Improvement:       "
          f"+{sorted_results[0].test_accuracy - baseline:.2f}%")
    print()
    print("  Top 5 configurations:")
    print(f"  {'Rank':<5} {'patch':>6} {'embed':>6} {'depth':>6} "
          f"{'accuracy':>10} {'time(s)':>8}")
    print("  " + "-" * 45)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i:<5} {r.patch_size:>6} {r.embed_dim:>6} {r.depth:>6} "
              f"{r.test_accuracy:>9.2f}% {r.train_time_s:>8.1f}")
    print("=" * 60)


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args(argv: list) -> argparse.Namespace:
    """
    Parses CLI arguments for standalone execution.

    Args:
        argv (list): sys.argv

    Returns:
        argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Task 5 — Hyperparameter sweep on Fashion MNIST transformer."
    )
    parser.add_argument(
        "--epochs",
        dest    = "epochs",
        type    = int,
        default = DEFAULT_EPOCHS,
        help    = f"Epochs per run (default: {DEFAULT_EPOCHS}). "
                  f"Reduce to 6-8 if sweep is too slow.",
    )
    parser.add_argument(
        "--output-dir",
        dest    = "output_dir",
        default = DEFAULT_OUTPUT_DIR,
        help    = f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        dest    = "data_dir",
        default = DEFAULT_DATA_DIR,
        help    = f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    return parser.parse_args(argv[1:])


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main(argv: list) -> None:
    """
    Main function for Task 5. Runs 5A → 5B → 5C in sequence.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Task 5: Design Your Own Experiment")
    print(f"  Epochs per run: {args.epochs}")
    print("=" * 60)

    device  = get_device()
    plotter = Plotter(output_dir=args.output_dir, show=True)

    # Generate all sweep configurations
    exp_config = ExperimentConfig()
    exp_config.DEFAULT_EPOCHS = args.epochs   # allow CLI override
    configs = exp_config.generate_sweep_configs()

    # 5A: print plan
    task5a_print_plan(configs)

    # 5B: print hypotheses
    task5b_print_hypotheses()

    # Initialise runner
    runner = ExperimentRunner(
        device      = device,
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
    )

    # 5C: run sweep
    results = task5c_run_sweep(runner, configs)

    # Extract baseline accuracy (run_id=0, phase="baseline")
    baseline_result = next(r for r in results if r.phase == "baseline")
    baseline_acc    = baseline_result.test_accuracy

    # Print summary
    print_summary(results, baseline_acc)

    # Plot results
    plotter.plot_experiment_results(results, baseline=baseline_acc)
    plotter.plot_top_configs(results, baseline=baseline_acc)

    print("\n[Task 5] Complete.")
    print(f"  Results CSV: {args.output_dir}/task5_results.csv")
    print(f"  Plots saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main(sys.argv)