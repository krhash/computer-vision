# tasks/read_network.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Reusable utility for loading a saved DigitNetwork from disk.
#              Satisfies Task 1E requirement of a separate file for reading
#              the network. load_trained_model() is imported by Tasks 2, 3,
#              and 4 which all begin by loading the pre-trained MNIST model.
#              Task 1E-specific evaluation logic lives in task1_build_train.py.
#
# Usage (standalone — prints model structure only):
#   python tasks/read_network.py
#   python tasks/read_network.py --model mnist_cnn.pth --model-dir ./models

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.network.digit_network import DigitNetwork
from src.utils.model_io        import ModelIO
from src.utils.device_utils    import get_device


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_MODEL_FILE = "mnist_cnn.pth"
DEFAULT_MODEL_DIR  = "./models"
DEFAULT_DATA_DIR   = "./data"
DEFAULT_OUTPUT_DIR = "./outputs"


# ------------------------------------------------------------------
# Core reusable function — imported by other task files
# ------------------------------------------------------------------

def load_trained_model(
    model_file: str = DEFAULT_MODEL_FILE,
    model_dir:  str = DEFAULT_MODEL_DIR,
    device:     torch.device | None = None,
) -> tuple[DigitNetwork, torch.device]:
    """
    Loads a saved DigitNetwork from disk and sets it to evaluation mode.

    This is the shared entry point used by Tasks 1E, 2, 3, and 4.
    The model is always put into eval() mode so that dropout behaves
    deterministically (scales by 1 - dropout_rate rather than zeroing).

    Args:
        model_file (str):              Filename of the saved .pth file.
        model_dir  (str):              Directory containing the .pth file.
        device     (torch.device|None): Device to load onto. Auto-detected
                                        if None.

    Returns:
        Tuple of (DigitNetwork in eval() mode, torch.device).
    """
    if device is None:
        device = get_device()

    model_io = ModelIO(model_dir=model_dir)
    model    = DigitNetwork()
    model_io.load(model, model_file)   # also calls model.eval() internally
    model    = model.to(device)

    print(f"\n  Loaded model: {model_file}")
    print(f"  Model structure:\n{model}")

    return model, device



# ------------------------------------------------------------------
# Main entry point (standalone execution)
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
        description="Task 1E — Read network and evaluate on MNIST test set."
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


def main(argv: list) -> None:
    """
    Standalone entry point. Loads the model and prints its structure.
    Task 1E evaluation logic (predict + plot) runs via task1_build_train.py.

    Args:
        argv (list): sys.argv
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  read_network — Load & Inspect Saved Model")
    print("=" * 60)

    model, _ = load_trained_model(
        model_file = args.model_file,
        model_dir  = args.model_dir,
    )

    print("\n[read_network] Model loaded and ready.\n")


if __name__ == "__main__":
    main(sys.argv)