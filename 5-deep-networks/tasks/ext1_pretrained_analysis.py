# tasks/ext1_pretrained_analysis.py
# Project 5: Recognition using Deep Networks — Extension 1
# Author: Krushna Sanjay Sharma
# Description: Loads a pre-trained torchvision model (default: ResNet18)
#              and analyses its first convolutional layer, mirroring the
#              approach in Task 2 but on ImageNet-trained filters.
#
# Steps:
#   1. Load pre-trained ResNet18 with ImageNet weights
#   2. Print model structure and first conv layer info
#   3. Extract and visualise all 64 filters (vs our 10 MNIST filters)
#   4. Apply filters to first MNIST test image via cv2.filter2D
#   5. Plot side-by-side comparison: MNIST CNN filters vs ResNet filters
#
# Requires: models/mnist_cnn.pth  (for the comparison plot)
#
# Usage:
#   python tasks/ext1_pretrained_analysis.py
#   python tasks/ext1_pretrained_analysis.py --model resnet18
#   python tasks/ext1_pretrained_analysis.py --model vgg16

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.evaluation.pretrained_analyzer import PretrainedAnalyzer
from src.evaluation.filter_analyzer     import FilterAnalyzer
from src.data.mnist_loader              import MNISTDataLoader
from src.visualization.plotter          import Plotter
from src.utils.device_utils             import get_device
from tasks.read_network                 import load_trained_model


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

DEFAULT_MODEL_NAME = "resnet18"
DEFAULT_MNIST_MODEL = "mnist_cnn.pth"
DEFAULT_MODEL_DIR   = "./models"
DEFAULT_DATA_DIR    = "./data"
DEFAULT_OUTPUT_DIR  = "./outputs"
MAX_RESPONSE_SHOW   = 32    # show first 32 filter responses (out of 64)


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def ext1_load_and_inspect(model_name: str) -> PretrainedAnalyzer:
    """
    Step 1-2: Load the pre-trained model and print its structure.

    Downloads ImageNet weights automatically if not cached.
    Prints the full model structure and first conv layer info.

    Args:
        model_name (str): torchvision model name (e.g. 'resnet18').

    Returns:
        PretrainedAnalyzer: Initialised analyser for the loaded model.
    """
    print(f"\n[Ext 1] Loading pre-trained {model_name}...")

    analyzer = PretrainedAnalyzer(model_name=model_name)
    info     = analyzer.get_model_info()

    print(f"\n  Model:        {info['model_name']}")
    print(f"  Num filters:  {info['num_filters']}")
    print(f"  Kernel size:  {info['kernel_size']}")
    print(f"  In channels:  {info['in_channels']} "
          f"({'RGB' if info['in_channels'] == 3 else 'greyscale'})")

    print(f"\n  Comparison with DigitNetwork conv1:")
    print(f"    DigitNetwork: 10 filters, 5x5, 1 channel  (MNIST greyscale)")
    print(f"    {model_name:<13}: {info['num_filters']} filters, "
          f"{info['kernel_size'][0]}x{info['kernel_size'][1]}, "
          f"{info['in_channels']} channels (ImageNet RGB)")

    return analyzer


def ext1_visualise_filters(
    analyzer: PretrainedAnalyzer,
    plotter:  Plotter,
) -> list:
    """
    Step 3: Extract and visualise all filters from the first conv layer.

    For RGB models, filters are averaged across colour channels to produce
    2D greyscale representations suitable for cv2.filter2D.

    Args:
        analyzer (PretrainedAnalyzer): Analyser with loaded model.
        plotter  (Plotter):            For saving filter grid.

    Returns:
        list: Extracted 2D filter arrays.
    """
    print(f"\n[Ext 1] Extracting and visualising filters...")

    _, filters = analyzer.get_filters()

    plotter.plot_pretrained_filters(
        filters    = filters,
        model_name = analyzer.model_name,
    )

    print(f"  Extracted {len(filters)} filters.")
    return filters


def ext1_apply_filters(
    analyzer:  PretrainedAnalyzer,
    filters:   list,
    data_dir:  str,
    plotter:   Plotter,
) -> list:
    """
    Step 4: Apply all filters to the first MNIST test image.

    Uses the unshuffled test loader so index 0 is always the same
    image (digit 7) — consistent with Task 2B.

    Args:
        analyzer  (PretrainedAnalyzer): Analyser with apply_filters().
        filters   (list):               Extracted filter arrays.
        data_dir  (str):                MNIST data directory.
        plotter   (Plotter):            For saving response grid.

    Returns:
        list: Filtered image arrays.
    """
    print(f"\n[Ext 1] Applying filters to first MNIST test image...")

    # Load MNIST test set — index 0 is always the same (unshuffled)
    data_loader = MNISTDataLoader(data_dir=data_dir, batch_size=1000)
    images, labels = next(iter(data_loader.test_loader))

    # Convert tensor to float32 numpy (H, W)
    first_image = images[0].squeeze().numpy().astype(np.float32)
    print(f"  Using test image at index 0, label: {labels[0].item()}")

    filtered_images = analyzer.apply_filters(first_image, filters)

    plotter.plot_pretrained_responses(
        filters         = filters,
        filtered_images = filtered_images,
        model_name      = analyzer.model_name,
        max_show        = MAX_RESPONSE_SHOW,
    )

    return filtered_images


def ext1_compare_with_mnist(
    pretrained_analyzer: PretrainedAnalyzer,
    pretrained_filters:  list,
    device:              torch.device,
    plotter:             Plotter,
) -> None:
    """
    Step 5: Load our trained MNIST CNN and plot a side-by-side filter
    comparison — MNIST conv1 (top) vs pre-trained conv1 (bottom).

    Args:
        pretrained_analyzer (PretrainedAnalyzer): For model name.
        pretrained_filters  (list):               Pre-trained filter arrays.
        device              (torch.device):        Target device.
        plotter             (Plotter):             For saving comparison plot.
    """
    print(f"\n[Ext 1] Comparing with MNIST CNN filters...")

    # Load our trained MNIST model
    try:
        mnist_model, _ = load_trained_model(
            model_file = DEFAULT_MNIST_MODEL,
            model_dir  = DEFAULT_MODEL_DIR,
            device     = device,
        )
        mnist_analyzer = FilterAnalyzer(model=mnist_model)
        _, mnist_filters = mnist_analyzer.get_filters()

        plotter.plot_filter_comparison(
            mnist_filters       = mnist_filters,
            pretrained_filters  = pretrained_filters,
            pretrained_name     = pretrained_analyzer.model_name,
        )

    except FileNotFoundError:
        print(f"  [SKIP] {DEFAULT_MNIST_MODEL} not found — "
              f"skipping comparison plot. Run task1 first.")


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
        description="Extension 1 — Pre-trained network conv layer analysis."
    )
    parser.add_argument(
        "--model",
        dest    = "model_name",
        default = DEFAULT_MODEL_NAME,
        choices = list(PretrainedAnalyzer.SUPPORTED_MODELS.keys()),
        help    = f"Pre-trained model to analyse (default: {DEFAULT_MODEL_NAME})",
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
        help    = f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv[1:])


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main(argv: list) -> None:
    """
    Main function for Extension 1.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Extension 1: Pre-trained Network Conv Layer Analysis")
    print(f"  Model: {args.model_name}")
    print("=" * 60)

    device  = get_device()
    plotter = Plotter(output_dir=args.output_dir, show=True)

    # Step 1-2: Load and inspect
    analyzer = ext1_load_and_inspect(args.model_name)

    # Step 3: Extract and visualise all filters
    filters = ext1_visualise_filters(analyzer, plotter)

    # Step 4: Apply filters to MNIST test image
    ext1_apply_filters(analyzer, filters, args.data_dir, plotter)

    # Step 5: Compare with MNIST CNN filters
    ext1_compare_with_mnist(analyzer, filters, device, plotter)

    print("\n[Extension 1] Complete.")
    print(f"  Outputs saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main(sys.argv)