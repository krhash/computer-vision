# tasks/task3_greek.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Orchestrates Task 3 — Transfer Learning on Greek Letters.
#
#   Steps:
#     1. Load pre-trained MNIST DigitNetwork from file
#     2. Freeze all network weights
#     3. Replace fc2 with a new Linear(50, 3) layer for alpha/beta/gamma
#     4. Train on 27 Greek letter examples, plotting loss per epoch
#     5. Save the transfer-learned model
#     6. Evaluate on custom Greek letter photos
#
# Requires:
#   - models/mnist_cnn.pth        (produced by task1_build_train.py)
#   - data/greek_letters/alpha/   (provided dataset)
#   - data/greek_letters/beta/
#   - data/greek_letters/gamma/
#
# Usage:
#   python tasks/task3_greek.py
#   python tasks/task3_greek.py --epochs 50 --greek-dir data/greek_letters

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.data.greek_loader        import GreekDataLoader
from src.data.handwritten_loader  import HandwrittenLoader
from src.training.trainer         import Trainer
from src.training.transfer_trainer import TransferTrainer
from src.evaluation.evaluator     import Evaluator
from src.visualization.plotter    import Plotter
from src.utils.model_io           import ModelIO
from src.utils.device_utils       import get_device
from tasks.read_network           import load_trained_model


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

DEFAULT_MNIST_MODEL    = "mnist_cnn.pth"
DEFAULT_GREEK_MODEL    = "greek_transfer.pth"
DEFAULT_MODEL_DIR      = "./models"
DEFAULT_GREEK_DIR      = "./data/greek_train"
DEFAULT_CUSTOM_DIR     = "./data/greek_custom"
DEFAULT_OUTPUT_DIR     = "./outputs"
DEFAULT_NUM_EPOCHS     = 50
DEFAULT_LEARNING_RATE  = 0.01
DEFAULT_MOMENTUM       = 0.5

# Greek class names in ImageFolder alphabetical order
GREEK_CLASSES = ["alpha", "beta", "gamma"]


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def task3_setup_transfer_model(
    device:     torch.device,
    model_dir:  str,
    model_file: str,
) -> tuple:
    """
    Steps 1-3: Load pre-trained MNIST model, freeze weights, replace fc2.

    Loads the saved DigitNetwork, passes it to TransferTrainer which:
        - Freezes all parameters (requires_grad = False)
        - Replaces fc2: Linear(50, 10) -> Linear(50, 3)

    Prints the modified network structure to stdout.

    Args:
        device     (torch.device): Target device.
        model_dir  (str):          Directory containing mnist_cnn.pth.
        model_file (str):          Filename of the pre-trained model.

    Returns:
        Tuple of (modified DigitNetwork, TransferTrainer).
    """
    print("\n[Task 3] Loading pre-trained MNIST model...")

    # Load pre-trained weights via shared utility
    model, device = load_trained_model(
        model_file = model_file,
        model_dir  = model_dir,
        device     = device,
    )

    # Freeze all weights and replace final layer with 3-class output
    transfer = TransferTrainer(
        model       = model,
        num_classes = 3,
        device      = device,
    )

    print("\n  Modified network structure:")
    print(transfer.model)

    return transfer.model, transfer


def task3_train(
    model:      torch.nn.Module,
    transfer:   TransferTrainer,
    greek_dir:  str,
    device:     torch.device,
    plotter:    Plotter,
    num_epochs: int,
) -> Trainer:
    """
    Step 4: Train the transfer model on 27 Greek letter examples.

    Only the new fc2 layer is updated — all other weights are frozen.
    Plots the training loss curve after all epochs complete.

    Args:
        model      (nn.Module):       Modified model with frozen weights.
        transfer   (TransferTrainer): Provides the filtered optimiser.
        greek_dir  (str):             Path to greek_letters/ directory.
        device     (torch.device):    Target device.
        plotter    (Plotter):         For saving the training curve.
        num_epochs (int):             Number of training epochs.

    Returns:
        Trainer: Contains full loss history.
    """
    print(f"\n[Task 3] Training on Greek letters for {num_epochs} epochs...")

    # Load Greek letter dataset with GreekTransform
    greek_loader = GreekDataLoader(
        training_set_path = greek_dir,
        batch_size        = 5,
        shuffle           = True,
    )

    # Optimiser only updates trainable parameters (new fc2)
    optimiser = transfer.get_optimiser(
        lr       = DEFAULT_LEARNING_RATE,
        momentum = DEFAULT_MOMENTUM,
    )

    trainer = Trainer(
        model        = model,
        optimiser    = optimiser,
        device       = device,
        log_interval = 5,
    )

    for epoch in range(1, num_epochs + 1):
        trainer.train_epoch(greek_loader.loader, epoch_num=epoch)

    # Plot training loss curve
    plotter.plot_greek_training_curve(losses=trainer.train_losses)

    print(
        f"\n  Final training loss: {trainer.train_losses[-1]:.4f}"
        f"  |  Final accuracy:   {trainer.train_accuracies[-1]:.2f}%"
    )

    return trainer


def task3_save_model(
    model:    torch.nn.Module,
    model_io: ModelIO,
) -> None:
    """
    Step 5: Save the transfer-learned model to disk.

    Args:
        model    (nn.Module): Trained transfer model.
        model_io (ModelIO):   Handles .pth file I/O.
    """
    print("\n[Task 3] Saving transfer model...")
    model_io.save(model, DEFAULT_GREEK_MODEL)


def task3_evaluate_custom(
    model:      torch.nn.Module,
    device:     torch.device,
    custom_dir: str,
    plotter:    Plotter,
) -> None:
    """
    Step 6: Load custom Greek letter photos and run them through the
    transfer-learned network.

    Custom images should be ~128x128 and named by class:
        alpha_1.jpg, beta_1.jpg, gamma_1.jpg etc.
    The label is inferred from the first character of the filename:
        'a' -> alpha (0), 'b' -> beta (1), 'g' -> gamma (2)

    Skips gracefully if the custom directory does not exist.

    Args:
        model      (nn.Module):    Trained transfer model in eval() mode.
        device     (torch.device): Target device.
        custom_dir (str):          Path to custom Greek letter images.
        plotter    (Plotter):      For saving prediction results.
    """
    print("\n[Task 3] Running inference on custom Greek letter images...")

    if not os.path.isdir(custom_dir):
        print(
            f"  [SKIP] Custom Greek directory not found: {custom_dir}\n"
            f"  Place your alpha/beta/gamma photos there and re-run.\n"
            f"  Expected filenames: alpha_1.jpg, beta_1.jpg, gamma_1.jpg ..."
        )
        return

    # Reuse HandwrittenLoader — same pre-processing pipeline
    # (greyscale, Otsu threshold, invert, resize to 28x28, normalise)
    loader = HandwrittenLoader(image_dir=custom_dir)
    print(f"  Loaded {len(loader)} custom Greek images.")

    # Map filename prefix to class index: a->0, b->1, g->2
    raw_labels = loader.get_labels()   # these are digit labels from filename[0]
    # Re-map: filename starts with 'a','b','g' but HandwrittenLoader reads
    # first char as digit. We store the correct mapping in the filenames
    # directly as 0,1,2 (alpha=0.jpg, beta=1.jpg, gamma=2.jpg).
    # Users should name files: 0_alpha.jpg, 1_beta.jpg, 2_gamma.jpg

    model.eval()
    evaluator  = Evaluator(model=model, device=device)
    batch      = loader.as_batch()

    predictions, _ = evaluator.predict_samples(
        data   = batch,
        labels = raw_labels,
        n      = len(loader),
    )

    # Compute and print accuracy (skip samples with unknown label -1)
    correct = sum(
        p == t for p, t in zip(predictions, raw_labels) if t != -1
    )
    total   = sum(1 for t in raw_labels if t != -1)
    print(f"\n  Custom Greek accuracy: {correct}/{total} = {100.0 * correct / total:.1f}%")

    # Plot up to 15 samples in a 3x5 grid
    plot_n = min(15, len(loader))
    plotter.plot_greek_predictions(
        images      = loader.get_tensors()[:plot_n],
        predictions = predictions[:plot_n],
        labels      = raw_labels[:plot_n],
        class_names = GREEK_CLASSES,
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
        argparse.Namespace with all configurable parameters.
    """
    parser = argparse.ArgumentParser(
        description="Task 3 — Transfer Learning on Greek Letters."
    )
    parser.add_argument(
        "--model",
        dest    = "model_file",
        default = DEFAULT_MNIST_MODEL,
        help    = f"Pre-trained MNIST model filename (default: {DEFAULT_MNIST_MODEL})",
    )
    parser.add_argument(
        "--model-dir",
        dest    = "model_dir",
        default = DEFAULT_MODEL_DIR,
        help    = f"Directory containing saved models (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--greek-dir",
        dest    = "greek_dir",
        default = DEFAULT_GREEK_DIR,
        help    = f"Greek letter dataset directory (default: {DEFAULT_GREEK_DIR})",
    )
    parser.add_argument(
        "--custom-dir",
        dest    = "custom_dir",
        default = DEFAULT_CUSTOM_DIR,
        help    = f"Custom Greek letter images directory (default: {DEFAULT_CUSTOM_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        dest    = "output_dir",
        default = DEFAULT_OUTPUT_DIR,
        help    = f"Directory to save output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--epochs",
        dest    = "num_epochs",
        type    = int,
        default = DEFAULT_NUM_EPOCHS,
        help    = f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    return parser.parse_args(argv[1:])


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main(argv: list) -> None:
    """
    Main function for Task 3. Runs all steps in sequence.

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    args = parse_args(argv)

    print("=" * 60)
    print("  Task 3: Transfer Learning on Greek Letters")
    print(f"  Epochs: {args.num_epochs}  |  Greek dir: {args.greek_dir}")
    print("=" * 60)

    device   = get_device()
    plotter  = Plotter(output_dir=args.output_dir, show=True)
    model_io = ModelIO(model_dir=args.model_dir)

    # Steps 1-3: load, freeze, replace layer
    model, transfer = task3_setup_transfer_model(
        device     = device,
        model_dir  = args.model_dir,
        model_file = args.model_file,
    )

    # Step 4: train on Greek letters
    task3_train(
        model      = model,
        transfer   = transfer,
        greek_dir  = args.greek_dir,
        device     = device,
        plotter    = plotter,
        num_epochs = args.num_epochs,
    )

    # Step 5: save transfer model
    task3_save_model(model, model_io)

    # Step 6: evaluate on custom Greek photos
    task3_evaluate_custom(
        model      = model,
        device     = device,
        custom_dir = args.custom_dir,
        plotter    = plotter,
    )

    print("\n[Task 3] Complete.\n")


if __name__ == "__main__":
    main(sys.argv)