# tasks/task5b_cnn_optimizer.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: CNN optimizer hyperparameter sweep on Fashion MNIST.
#              Complements the transformer sweep in task5_experiment.py.
#
# Three dimensions:
#   1. optimizer    — SGD, Adam, AdamW, RMSprop
#   2. lr           — [0.1, 0.01, 0.001, 0.0001]
#   3. momentum     — [0.0, 0.5, 0.9, 0.99]  (SGD only)
#
# Hypotheses:
#   H1 (optimizer): Adam/AdamW will converge faster and achieve higher
#                   accuracy than SGD. RMSprop will be competitive.
#   H2 (lr):        Each optimizer has a different optimal LR.
#                   SGD needs lr=0.01-0.1; Adam/AdamW work best at
#                   lr=0.001; too high LR will diverge.
#   H3 (momentum):  SGD momentum=0.9 will outperform momentum=0.5
#                   (baseline). Very high momentum (0.99) may oscillate.
#
# Usage:
#   python tasks/task5b_cnn_optimizer.py
#   python tasks/task5b_cnn_optimizer.py --epochs 5 --output-dir ./outputs

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment.cnn_experiment_config import CNNExperimentConfig, CNNExperimentResult
from src.experiment.cnn_experiment_runner import CNNExperimentRunner
from src.visualization.plotter            import Plotter
from src.utils.device_utils               import get_device
from typing import List


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_DATA_DIR   = "./data"
DEFAULT_EPOCHS     = 5      # CNN trains fast — 5 epochs sufficient per run


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def print_plan(configs: list) -> None:
    """
    Prints the CNN sweep plan to stdout before execution.

    Args:
        configs (list): Generated list of experiment configs.
    """
    print("\n[Task 5B] CNN Optimizer Sweep — Plan")
    print("=" * 60)
    print("  Dataset:    Fashion MNIST")
    print("  Model:      DigitNetwork (CNN)")
    print("  Strategy:   Round-robin linear search")
    print()
    print("  Dimensions:")
    print(f"    1. optimizer   : {CNNExperimentConfig.OPTIMIZERS}")
    print(f"    2. lr          : {CNNExperimentConfig.LEARNING_RATES}")
    print(f"    3. momentum    : {CNNExperimentConfig.MOMENTUMS} (SGD only)")
    print()
    print("  Hypotheses:")
    print("    H1: Adam/AdamW will outperform SGD in accuracy and speed.")
    print("    H2: Optimal LR differs per optimizer (SGD=0.01, Adam=0.001).")
    print("    H3: SGD momentum=0.9 beats baseline momentum=0.5.")
    print()

    phase_counts: dict = {}
    for c in configs:
        phase_counts[c["phase"]] = phase_counts.get(c["phase"], 0) + 1

    print("  Runs per phase:")
    for phase, count in phase_counts.items():
        print(f"    {phase:<25} {count:>3} runs")
    print(f"    {'TOTAL':<25} {len(configs):>3} runs")
    print("=" * 60)


def print_summary(
    results:  List[CNNExperimentResult],
    baseline: float,
) -> None:
    """
    Prints sweep summary with top configs and hypothesis discussion.

    Args:
        results  (list):  All CNNExperimentResult objects.
        baseline (float): Baseline test accuracy (%).
    """
    sorted_acc = sorted(results, key=lambda r: r.test_accuracy, reverse=True)
    sorted_eff = sorted(
        results,
        key=lambda r: r.test_accuracy / r.train_time_s,
        reverse=True,
    )
    best      = sorted_acc[0]
    best_eff  = sorted_eff[0]
    base_time = next(r.train_time_s for r in results if r.phase == "baseline")

    print("\n" + "=" * 60)
    print("  Task 5B — CNN Optimizer Sweep Summary")
    print("=" * 60)
    print(f"  Baseline: SGD lr=0.01 mom=0.5 → {baseline:.2f}% in {base_time:.0f}s")
    print(f"  Best accuracy:    {best.test_accuracy:.2f}%  "
          f"({best.optimizer_name} lr={best.lr} mom={best.momentum})")
    print(f"  Improvement:      +{best.test_accuracy - baseline:.2f}%")
    print()
    print(f"  Best efficiency (accuracy/time):")
    time_saved = (1 - best_eff.train_time_s / base_time) * 100
    print(f"    {best_eff.optimizer_name} lr={best_eff.lr} "
          f"→ {best_eff.test_accuracy:.2f}% in {best_eff.train_time_s:.0f}s "
          f"({time_saved:+.1f}% time vs baseline)")
    print()
    print("  Top 10 configurations:")
    print(f"  {'Rank':<5} {'optimizer':<10} {'lr':>8} {'mom':>6} "
          f"{'accuracy':>10} {'time(s)':>8}")
    print("  " + "-" * 52)
    for i, r in enumerate(sorted_acc[:10], 1):
        print(f"  {i:<5} {r.optimizer_name:<10} {r.lr:>8.4f} "
              f"{r.momentum:>6.2f} "
              f"{r.test_accuracy:>9.2f}% {r.train_time_s:>8.1f}")
    print()

    # Per-optimizer best
    print("  Best per optimizer:")
    for opt in CNNExperimentConfig.OPTIMIZERS:
        opt_results = [r for r in results if r.optimizer_name == opt]
        if opt_results:
            best_opt = max(opt_results, key=lambda r: r.test_accuracy)
            print(f"    {opt:<10} → {best_opt.test_accuracy:.2f}%  "
                  f"lr={best_opt.lr}  mom={best_opt.momentum}  "
                  f"time={best_opt.train_time_s:.0f}s")
    print()
    print("  Hypothesis discussion:")
    print("  (Fill in after reviewing results)")
    print("=" * 60)


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args(argv: list) -> argparse.Namespace:
    """
    Parses CLI arguments.

    Args:
        argv (list): sys.argv

    Returns:
        argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Task 5B — CNN optimizer sweep on Fashion MNIST."
    )
    parser.add_argument(
        "--epochs",
        dest    = "epochs",
        type    = int,
        default = DEFAULT_EPOCHS,
        help    = f"Epochs per run (default: {DEFAULT_EPOCHS})",
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
    Main function for Task 5B CNN optimizer sweep.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Task 5B: CNN Optimizer Sweep on Fashion MNIST")
    print(f"  Epochs per run: {args.epochs}")
    print("=" * 60)

    device  = get_device()
    plotter = Plotter(output_dir=args.output_dir, show=True)

    # Generate sweep configurations
    exp_config = CNNExperimentConfig()
    exp_config.DEFAULT_EPOCHS = args.epochs
    configs = exp_config.generate_sweep_configs()

    # Print plan and hypotheses
    print_plan(configs)

    # Run sweep
    runner = CNNExperimentRunner(
        device      = device,
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
    )
    results = runner.run(configs)

    # Baseline accuracy
    baseline_acc = next(
        r.test_accuracy for r in results if r.phase == "baseline"
    )

    # Print summary
    print_summary(results, baseline_acc)

    # Plot results
    plotter.plot_cnn_experiment_results(results, baseline=baseline_acc)
    plotter.plot_cnn_top_configs(results, baseline=baseline_acc)

    print("\n[Task 5B] Complete.")
    print(f"  Results CSV : {args.output_dir}/task5b_cnn_results.csv")
    print(f"  Plots saved : {args.output_dir}/\n")


if __name__ == "__main__":
    main(sys.argv)