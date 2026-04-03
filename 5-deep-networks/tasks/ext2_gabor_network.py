# tasks/ext2_gabor_network.py
# Project 5: Recognition using Deep Networks — Extension 2
# Author: Krushna Sanjay Sharma
# Description: Replaces the first conv layer of DigitNetwork with a fixed
#              Gabor filter bank and retrains only conv2 and FC layers.
#              Compares accuracy and convergence against the fully learned
#              DigitNetwork from Task 1.
#
# Hypothesis:
#   Gabor filters capture orientation/frequency features similar to
#   what conv1 learns on MNIST. The Gabor network should achieve
#   close to the same accuracy (~97-99%) as the fully trained CNN,
#   demonstrating that hand-crafted filters can substitute for learned
#   ones in the first layer of a digit recognition network.
#
# Steps:
#   1. Build GaborDigitNetwork — conv1 pre-loaded with Gabor filters, frozen
#   2. Visualise Gabor filters and compare with learned MNIST filters
#   3. Train on MNIST (only conv2 + FC layers update)
#   4. Evaluate and compare with original DigitNetwork
#   5. Plot training curves, filter comparison, and test results
#
# Usage:
#   python tasks/ext2_gabor_network.py
#   python tasks/ext2_gabor_network.py --epochs 10 --sigma 1.5 --gamma 0.5

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from src.network.gabor_network   import GaborDigitNetwork
from src.network.digit_network   import DigitNetwork
from src.data.mnist_loader       import MNISTDataLoader
from src.training.trainer        import Trainer
from src.evaluation.evaluator    import Evaluator
from src.evaluation.filter_analyzer import FilterAnalyzer
from src.visualization.plotter   import Plotter
from src.utils.model_io          import ModelIO
from src.utils.device_utils      import get_device
from tasks.read_network          import load_trained_model


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

DEFAULT_EPOCHS      = 10
DEFAULT_BATCH_SIZE  = 64
DEFAULT_LR          = 0.01
DEFAULT_MOMENTUM    = 0.5
DEFAULT_SIGMA       = 1.5
DEFAULT_GAMMA       = 0.5
DEFAULT_DATA_DIR    = "./data"
DEFAULT_OUTPUT_DIR  = "./outputs"
DEFAULT_MODEL_DIR   = "./models"
GABOR_MODEL_FILE    = "gabor_network.pth"
MNIST_MODEL_FILE    = "mnist_cnn.pth"


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def ext2_build_and_visualise(
    sigma:   float,
    gamma:   float,
    device:  torch.device,
    plotter: Plotter,
) -> tuple:
    """
    Step 1-2: Build GaborDigitNetwork, visualise Gabor filters, and
    compare them with the learned MNIST conv1 filters.

    Args:
        sigma   (float):        Gabor filter sigma parameter.
        gamma   (float):        Gabor filter gamma parameter.
        device  (torch.device): Target device.
        plotter (Plotter):      For saving filter plots.

    Returns:
        Tuple of (GaborDigitNetwork, gabor_filter_arrays).
    """
    print("\n[Ext 2] Building GaborDigitNetwork...")
    model = GaborDigitNetwork(sigma=sigma, gamma=gamma).to(device)

    print(f"\n  Network structure:")
    print(model)

    # Count trainable vs frozen parameters
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    frozen    = total - trainable
    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen (Gabor conv1): {frozen:,}")

    # Extract Gabor filters for visualisation
    gabor_filters = model.get_gabor_filters()   # (10, 5, 5)

    # Plot Gabor filter bank
    plotter.plot_gabor_filters(gabor_filters)

    # Compare with learned MNIST filters (if model exists)
    try:
        learned_model, _ = load_trained_model(
            model_file = MNIST_MODEL_FILE,
            model_dir  = DEFAULT_MODEL_DIR,
            device     = device,
        )
        learned_analyzer = FilterAnalyzer(model=learned_model)
        _, learned_filters = learned_analyzer.get_filters()

        plotter.plot_gabor_vs_learned(
            gabor_filters   = gabor_filters,
            learned_filters = learned_filters,
        )
    except FileNotFoundError:
        print(f"  [SKIP] {MNIST_MODEL_FILE} not found — skipping comparison.")

    return model, gabor_filters


def ext2_train(
    model:       GaborDigitNetwork,
    data_loader: MNISTDataLoader,
    device:      torch.device,
    num_epochs:  int,
) -> Trainer:
    """
    Step 3: Train GaborDigitNetwork on MNIST.

    Only conv2, fc1, and fc2 are updated — conv1 (Gabor) is frozen.
    Uses SGD with momentum matching Task 1 for fair comparison.

    Args:
        model       (GaborDigitNetwork): Network with frozen Gabor conv1.
        data_loader (MNISTDataLoader):   MNIST train/test loaders.
        device      (torch.device):      Target device.
        num_epochs  (int):               Number of training epochs.

    Returns:
        Trainer: With full loss/accuracy history.
    """
    print(f"\n[Ext 2] Training Gabor network for {num_epochs} epochs...")
    print(f"  (conv1 frozen — only conv2, fc1, fc2 update)")

    # Pass only trainable parameters to the optimiser
    optimiser = optim.SGD(
        model.get_trainable_params(),
        lr       = DEFAULT_LR,
        momentum = DEFAULT_MOMENTUM,
    )

    trainer = Trainer(
        model        = model,
        optimiser    = optimiser,
        device       = device,
        log_interval = 100,
    )

    for epoch in range(1, num_epochs + 1):
        print(f"\n  --- Epoch {epoch}/{num_epochs} ---")
        trainer.train_epoch(data_loader.train_loader, epoch_num=epoch)

    return trainer


def ext2_evaluate_and_compare(
    gabor_model:    GaborDigitNetwork,
    gabor_trainer:  Trainer,
    data_loader:    MNISTDataLoader,
    device:         torch.device,
    plotter:        Plotter,
    num_epochs:     int,
) -> None:
    """
    Step 4-5: Evaluate Gabor network, load the original learned network,
    retrain it for the same number of epochs, and compare results.

    Plots training curves side by side and prints a comparison table.

    Args:
        gabor_model   (GaborDigitNetwork): Trained Gabor network.
        gabor_trainer (Trainer):           Gabor training history.
        data_loader   (MNISTDataLoader):   MNIST test loader.
        device        (torch.device):      Target device.
        plotter       (Plotter):           For saving comparison plots.
        num_epochs    (int):               Epochs used (for comparison).
    """
    print("\n[Ext 2] Evaluating Gabor network on test set...")

    gabor_evaluator = Evaluator(model=gabor_model, device=device)
    gabor_acc       = gabor_evaluator.evaluate(data_loader.test_loader)

    # Train a fresh DigitNetwork for the same number of epochs for comparison
    print(f"\n[Ext 2] Training fresh DigitNetwork for {num_epochs} epochs "
          f"(comparison baseline)...")

    learned_model = DigitNetwork().to(device)
    learned_opt   = optim.SGD(
        learned_model.parameters(),
        lr       = DEFAULT_LR,
        momentum = DEFAULT_MOMENTUM,
    )
    learned_trainer   = Trainer(
        model        = learned_model,
        optimiser    = learned_opt,
        device       = device,
        log_interval = 999,   # suppress per-batch logs
    )
    learned_evaluator = Evaluator(model=learned_model, device=device)

    for epoch in range(1, num_epochs + 1):
        learned_trainer.train_epoch(data_loader.train_loader, epoch_num=epoch)

    learned_acc = learned_evaluator.evaluate(data_loader.test_loader)

    # Plot training curves comparison
    plotter.plot_gabor_training_curves(
        gabor_losses   = gabor_trainer.train_losses,
        gabor_accs     = gabor_trainer.train_accuracies,
        learned_losses = learned_trainer.train_losses,
        learned_accs   = learned_trainer.train_accuracies,
    )

    # Print comparison summary
    print("\n" + "=" * 55)
    print("  Extension 2 — Results Summary")
    print("=" * 55)
    print(f"  {'Network':<30} {'Test Accuracy':>15}")
    print("  " + "-" * 47)
    print(f"  {'GaborDigitNetwork (frozen conv1)':<30} "
          f"{gabor_acc:>14.2f}%")
    print(f"  {'DigitNetwork (fully learned)':<30} "
          f"{learned_acc:>14.2f}%")
    print(f"  {'Difference':<30} "
          f"{gabor_acc - learned_acc:>+14.2f}%")
    print("=" * 55)

    total_params     = sum(p.numel() for p in gabor_model.parameters())
    trainable_params = sum(p.numel() for p in gabor_model.parameters()
                           if p.requires_grad)
    print(f"\n  Gabor network trainable params: {trainable_params:,} "
          f"/ {total_params:,} total")
    print(f"  Reduction: "
          f"{100 * (1 - trainable_params/total_params):.1f}% fewer "
          f"learnable parameters")


def ext2_save_model(model: GaborDigitNetwork, model_io: ModelIO) -> None:
    """
    Saves the trained Gabor network to disk.

    Args:
        model    (GaborDigitNetwork): Trained model.
        model_io (ModelIO):           Handles .pth file I/O.
    """
    print("\n[Ext 2] Saving Gabor network...")
    model_io.save(model, GABOR_MODEL_FILE)


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
        description="Extension 2 — Gabor filter bank as first conv layer."
    )
    parser.add_argument(
        "--epochs",
        dest    = "epochs",
        type    = int,
        default = DEFAULT_EPOCHS,
        help    = f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--sigma",
        dest    = "sigma",
        type    = float,
        default = DEFAULT_SIGMA,
        help    = f"Gabor sigma parameter (default: {DEFAULT_SIGMA})",
    )
    parser.add_argument(
        "--gamma",
        dest    = "gamma",
        type    = float,
        default = DEFAULT_GAMMA,
        help    = f"Gabor gamma parameter (default: {DEFAULT_GAMMA})",
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
    Main function for Extension 2.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Extension 2: Gabor Filter Bank as First Conv Layer")
    print(f"  Epochs: {args.epochs}  sigma: {args.sigma}  gamma: {args.gamma}")
    print("=" * 60)

    device   = get_device()
    plotter  = Plotter(output_dir=args.output_dir, show=True)
    model_io = ModelIO(model_dir=DEFAULT_MODEL_DIR)

    # Load MNIST data
    data_loader = MNISTDataLoader(
        data_dir   = args.data_dir,
        batch_size = DEFAULT_BATCH_SIZE,
    )

    # Steps 1-2: build network and visualise filters
    model, _ = ext2_build_and_visualise(
        sigma   = args.sigma,
        gamma   = args.gamma,
        device  = device,
        plotter = plotter,
    )

    # Step 3: train
    trainer = ext2_train(
        model       = model,
        data_loader = data_loader,
        device      = device,
        num_epochs  = args.epochs,
    )

    # Steps 4-5: evaluate and compare
    ext2_evaluate_and_compare(
        gabor_model   = model,
        gabor_trainer = trainer,
        data_loader   = data_loader,
        device        = device,
        plotter       = plotter,
        num_epochs    = args.epochs,
    )

    # Save model
    ext2_save_model(model, model_io)

    print("\n[Extension 2] Complete.")
    print(f"  Outputs saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main(sys.argv)