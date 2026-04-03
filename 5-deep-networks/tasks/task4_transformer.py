# tasks/task4_transformer.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Orchestrates Task 4 — Re-implement the network using
#              transformer layers. NetTransformer is a drop-in replacement
#              for DigitNetwork with identical input/output interface.
#
#   Steps:
#     1. Build NetTransformer with default NetConfig settings
#     2. Train on MNIST using the same Trainer/Evaluator pipeline as Task 1
#     3. Evaluate after each epoch, plot training curves
#     4. Save the trained transformer model
#     5. Run on first 10 test examples (same as Task 1E for comparison)
#
# Usage:
#   python tasks/task4_transformer.py
#   python tasks/task4_transformer.py --epochs 15 --batch-size 64

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from src.network.transformer_network import NetTransformer, NetConfig
from src.data.mnist_loader           import MNISTDataLoader
from src.training.trainer            import Trainer
from src.evaluation.evaluator        import Evaluator
from src.visualization.plotter       import Plotter
from src.utils.model_io              import ModelIO
from src.utils.device_utils          import get_device


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

DEFAULT_MODEL_FILE  = "mnist_transformer.pth"
DEFAULT_MODEL_DIR   = "./models"
DEFAULT_DATA_DIR    = "./data"
DEFAULT_OUTPUT_DIR  = "./outputs"
DEFAULT_NUM_EPOCHS  = 15
DEFAULT_BATCH_SIZE  = 64


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def task4_build_transformer(device: torch.device) -> tuple:
    """
    Step 1: Instantiate NetTransformer with default NetConfig settings.

    Uses the default configuration from the provided template:
        patch_size=4, stride=2, embed_dim=48, depth=4,
        num_heads=8, mlp_dim=128, dropout=0.1

    Prints the full model structure and config string to stdout.

    Args:
        device (torch.device): Target device.

    Returns:
        Tuple of (NetTransformer, NetConfig).
    """
    print("\n[Task 4] Building NetTransformer with default config...")

    config = NetConfig(
        name          = "vit_base",
        dataset       = "mnist",
        patch_size    = 4,
        stride        = 2,
        embed_dim     = 48,
        depth         = 4,
        num_heads     = 8,
        mlp_dim       = 128,
        dropout       = 0.1,
        use_cls_token = False,   # use mean pooling
        epochs        = DEFAULT_NUM_EPOCHS,
        batch_size    = DEFAULT_BATCH_SIZE,
        lr            = 1e-3,
        weight_decay  = 1e-4,
        seed          = 0,
        optimizer     = "adamw",
        device        = str(device),
    )

    print(f"\n  Config:\n  {config.config_string}")

    model = NetTransformer(config).to(device)

    print(f"\n  Model structure:\n{model}")

    return model, config


def task4_train(
    model:       NetTransformer,
    config:      NetConfig,
    data_loader: MNISTDataLoader,
    device:      torch.device,
    plotter:     Plotter,
    num_epochs:  int,
) -> tuple:
    """
    Step 2-3: Train the transformer on MNIST, evaluate after each epoch,
    and plot training/test curves.

    Uses AdamW optimiser (better suited for transformers than SGD)
    with weight decay regularisation as specified in NetConfig.

    Args:
        model       (NetTransformer):  The transformer model.
        config      (NetConfig):       Hyperparameter config.
        data_loader (MNISTDataLoader): Provides train and test loaders.
        device      (torch.device):    Target device.
        plotter     (Plotter):         For saving training curves.
        num_epochs  (int):             Number of epochs to train.

    Returns:
        Tuple of (Trainer, Evaluator) with full history.
    """
    print(f"\n[Task 4] Training NetTransformer for {num_epochs} epochs...")

    # AdamW is the standard optimiser for transformer models
    optimiser = optim.AdamW(
        model.parameters(),
        lr           = config.lr,
        weight_decay = config.weight_decay,
    )

    trainer   = Trainer(model=model, optimiser=optimiser, device=device)
    evaluator = Evaluator(model=model, device=device)

    for epoch in range(1, num_epochs + 1):
        print(f"\n  --- Epoch {epoch}/{num_epochs} ---")
        trainer.train_epoch(data_loader.train_loader, epoch_num=epoch)
        evaluator.evaluate(data_loader.test_loader)

    # Plot training and test curves
    plotter.plot_transformer_curves(
        train_losses     = trainer.train_losses,
        test_losses      = evaluator.test_losses,
        train_accuracies = trainer.train_accuracies,
        test_accuracies  = evaluator.test_accuracies,
    )

    return trainer, evaluator


def task4_save_model(
    model:    NetTransformer,
    model_io: ModelIO,
) -> None:
    """
    Step 4: Save the trained transformer model to disk.

    Args:
        model    (NetTransformer): Trained model.
        model_io (ModelIO):        Handles .pth file I/O.
    """
    print("\n[Task 4] Saving transformer model...")
    model_io.save(model, DEFAULT_MODEL_FILE)


def task4_evaluate_test_samples(
    model:       NetTransformer,
    data_loader: MNISTDataLoader,
    device:      torch.device,
    plotter:     Plotter,
) -> None:
    """
    Step 5: Run the trained transformer on the first 10 test examples
    and plot the first 9 as a 3x3 prediction grid.

    Mirrors Task 1E output for direct CNN vs transformer comparison.

    Args:
        model       (NetTransformer):  Trained transformer in eval() mode.
        data_loader (MNISTDataLoader): Provides unshuffled test loader.
        device      (torch.device):    Target device.
        plotter     (Plotter):         For saving the prediction grid.
    """
    print("\n[Task 4] Running transformer on first 10 test samples...")

    model.eval()
    images, labels = next(iter(data_loader.test_loader))

    evaluator = Evaluator(model=model, device=device)
    predictions, _ = evaluator.predict_samples(
        data   = images,
        labels = labels,
        n      = 10,
    )

    plotter.plot_predictions_grid(
        images      = images,
        predictions = predictions,
        labels      = labels.tolist(),
        n           = 9,
        filename    = "task4_predictions.png",
        title       = "Task 4 — Transformer Predictions on Test Set",
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
        argparse.Namespace with epochs, batch_size, model_dir,
        data_dir, output_dir.
    """
    parser = argparse.ArgumentParser(
        description="Task 4 — Transformer network for MNIST digit recognition."
    )
    parser.add_argument(
        "--epochs",
        dest    = "num_epochs",
        type    = int,
        default = DEFAULT_NUM_EPOCHS,
        help    = f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        dest    = "batch_size",
        type    = int,
        default = DEFAULT_BATCH_SIZE,
        help    = f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--model-dir",
        dest    = "model_dir",
        default = DEFAULT_MODEL_DIR,
        help    = f"Directory to save model (default: {DEFAULT_MODEL_DIR})",
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
    Main function for Task 4. Runs all steps in sequence.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Task 4: MNIST Transformer Network")
    print(f"  Epochs: {args.num_epochs}  |  Batch size: {args.batch_size}")
    print("=" * 60)

    device   = get_device()
    plotter  = Plotter(output_dir=args.output_dir, show=True)
    model_io = ModelIO(model_dir=args.model_dir)

    # Load MNIST data
    data_loader = MNISTDataLoader(
        data_dir   = args.data_dir,
        batch_size = args.batch_size,
    )

    # Step 1: build transformer
    model, config = task4_build_transformer(device)

    # Steps 2-3: train and evaluate
    task4_train(
        model       = model,
        config      = config,
        data_loader = data_loader,
        device      = device,
        plotter     = plotter,
        num_epochs  = args.num_epochs,
    )

    # Step 4: save model
    task4_save_model(model, model_io)

    # Step 5: evaluate on first 10 test samples
    task4_evaluate_test_samples(
        model       = model,
        data_loader = data_loader,
        device      = device,
        plotter     = plotter,
    )

    print("\n[Task 4] Complete.\n")


if __name__ == "__main__":
    main(sys.argv)