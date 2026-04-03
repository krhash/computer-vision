# tasks/task2_examine.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Orchestrates all subtasks of Task 2:
#   2A — Load trained model, print structure, extract and visualise
#        the 10 conv1 filter weights as a 3x4 grid
#   2B — Apply the 10 filters to the first MNIST training image using
#        cv2.filter2D and visualise the 10 filtered outputs
#
# Requires: models/mnist_cnn.pth (produced by task1_build_train.py)
#
# Usage:
#   python tasks/task2_examine.py
#   python tasks/task2_examine.py --model mnist_cnn.pth --model-dir ./models

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.mnist_loader        import MNISTDataLoader
from src.evaluation.filter_analyzer import FilterAnalyzer
from src.visualization.plotter    import Plotter
from tasks.read_network           import load_trained_model


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

DEFAULT_MODEL_FILE = "mnist_cnn.pth"
DEFAULT_MODEL_DIR  = "./models"
DEFAULT_DATA_DIR   = "./data"
DEFAULT_OUTPUT_DIR = "./outputs"


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def task2a_analyse_filters(analyzer: FilterAnalyzer, plotter: Plotter) -> list:
    """
    Subtask 2A: Extract and visualise the 10 conv1 filter weights.

    Accesses model.conv1.weight (shape [10, 1, 5, 5]), prints the weight
    values and shape of each filter to stdout, then plots all 10 in a
    3x4 grid (Plot 1).

    Args:
        analyzer (FilterAnalyzer): Analyser bound to the loaded model.
        plotter  (Plotter):        For saving the filter weight grid.

    Returns:
        list: The 10 extracted filter arrays (each 5x5 numpy array).
    """
    print("\n[Task 2A] Extracting and visualising conv1 filter weights...")

    _, filters = analyzer.get_filters()

    # Plot 1: all 10 filters in a 3x4 grid
    plotter.plot_filters(filters)

    print(f"  Extracted {len(filters)} filters, each shape: {filters[0].shape}")
    return filters


def task2b_show_filter_effects(
    analyzer:    FilterAnalyzer,
    filters:     list,
    data_dir:    str,
    plotter:     Plotter,
) -> None:
    """
    Subtask 2B: Apply the 10 conv1 filters to the first MNIST test image
    using cv2.filter2D and plot filter + response side by side.

    Uses the test loader (unshuffled) so index 0 is always the same image,
    matching the deterministic behaviour expected by the assignment.

    Args:
        analyzer  (FilterAnalyzer): Analyser with apply_filters method.
        filters   (list):           The 10 extracted 5x5 filter arrays.
        data_dir  (str):            MNIST data directory.
        plotter   (Plotter):        For saving the combined filter/response grid.
    """
    print("\n[Task 2B] Applying filters to first MNIST test image...")

    # Use test loader (unshuffled) — index 0 is always the same image
    data_loader = MNISTDataLoader(data_dir=data_dir, batch_size=1000)
    images, labels = next(iter(data_loader.test_loader))

    first_image = images[0]   # shape (1, 28, 28)
    print(f"  Using test image at index 0 with label: {labels[0].item()}")

    # Apply all 10 filters via cv2.filter2D
    filtered_images = analyzer.apply_filters(first_image, filters)

    # Plot 2: filter (left) | response (right) for each of the 10 filters
    plotter.plot_filter_responses(
        filters         = filters,
        filtered_images = filtered_images,
    )


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args(argv: list) -> argparse.Namespace:
    """
    Parses CLI arguments for standalone execution.

    Args:
        argv (list): sys.argv

    Returns:
        argparse.Namespace with model_file, model_dir, data_dir, output_dir.
    """
    parser = argparse.ArgumentParser(
        description="Task 2 — Examine network filters and filter effects."
    )
    parser.add_argument(
        "--model",
        dest    = "model_file",
        default = DEFAULT_MODEL_FILE,
        help    = f"Model filename in model_dir (default: {DEFAULT_MODEL_FILE})",
    )
    parser.add_argument(
        "--model-dir",
        dest    = "model_dir",
        default = DEFAULT_MODEL_DIR,
        help    = f"Directory containing saved models (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        dest    = "data_dir",
        default = DEFAULT_DATA_DIR,
        help    = f"MNIST data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        dest    = "output_dir",
        default = DEFAULT_OUTPUT_DIR,
        help    = f"Directory to save output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv[1:])


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main(argv: list) -> None:
    """
    Main function for Task 2. Runs subtasks 2A and 2B in sequence.

    Loads the pre-trained MNIST model via read_network.load_trained_model(),
    then analyses the conv1 filters and their effects on a training image.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Task 2: Examine Network Filters")
    print("=" * 60)

    # Load pre-trained model — also prints model structure to stdout
    model, device = load_trained_model(
        model_file = args.model_file,
        model_dir  = args.model_dir,
    )

    # Shared utility instances
    analyzer = FilterAnalyzer(model=model)
    plotter  = Plotter(output_dir=args.output_dir, show=True)

    # --- 2A: Extract weights + Plot 1: filter grid ---
    filters = task2a_analyse_filters(analyzer, plotter)

    # --- 2B: Apply filters + Plot 2: filter | response side by side ---
    task2b_show_filter_effects(
        analyzer  = analyzer,
        filters   = filters,
        data_dir  = args.data_dir,
        plotter   = plotter,
    )

    print("\n[Task 2] Complete.\n")


if __name__ == "__main__":
    main(sys.argv)