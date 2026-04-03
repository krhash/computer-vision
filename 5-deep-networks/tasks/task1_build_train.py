# tasks/task1_build_train.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Orchestrates all subtasks of Task 1:
#   1A — Load MNIST and plot the first 6 test digits
#   1B — Build the DigitNetwork CNN (architecture defined in src/network)
#   1C — Train for N epochs, evaluate after each, plot loss/accuracy curves
#   1D — Save the trained model to models/mnist_cnn.pth
#   1E — Reload model, run on first 10 test samples, plot 3x3 prediction grid
#   1F — Load handwritten digit images, run through network, display results

import sys
import os

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

from src.network.digit_network   import DigitNetwork
from src.utils.device_utils      import get_device
from tasks.read_network          import load_trained_model
from src.data.mnist_loader       import MNISTDataLoader
from src.data.handwritten_loader import HandwrittenLoader
from src.training.trainer        import Trainer
from src.evaluation.evaluator    import Evaluator
from src.visualization.plotter   import Plotter
from src.utils.model_io          import ModelIO


# ------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------

# Training hyper-parameters
NUM_EPOCHS      = 5
BATCH_SIZE      = 64
LEARNING_RATE   = 0.01
MOMENTUM        = 0.5           # Used with SGD optimiser

# File paths
MODEL_FILENAME       = "mnist_cnn.pth"
DATA_DIR             = "./data"
HANDWRITTEN_DIR      = "./data/handwritten"
OUTPUT_DIR           = "./outputs"
MODEL_DIR            = "./models"


# ------------------------------------------------------------------
# Subtask functions
# ------------------------------------------------------------------

def task1a_load_and_visualise(
    data_loader: MNISTDataLoader,
    plotter:     Plotter,
) -> tuple:
    """
    Subtask 1A: Fetch the first 6 digits from the test set and plot them.

    The test loader is unshuffled so these are always the same 6 digits.

    Args:
        data_loader (MNISTDataLoader): Configured MNIST data loader.
        plotter     (Plotter):         Plotter instance for saving figures.

    Returns:
        Tuple of (images tensor, labels list) for the first batch.
    """
    print("\n[Task 1A] Loading MNIST and plotting first 6 test digits...")

    # Grab the first batch from the test loader (batch_size=1000, unshuffled)
    images, labels = next(iter(data_loader.test_loader))

    plotter.plot_sample_grid(
        images = images,
        labels = labels.tolist(),
        n      = 6,
    )

    print(f"  First 6 test labels: {labels[:6].tolist()}")
    return images, labels


def task1b_build_network(device: torch.device) -> DigitNetwork:
    """
    Subtask 1B: Instantiate the DigitNetwork CNN and move it to device.

    The architecture is defined entirely in src/network/digit_network.py.
    This function simply constructs it and prints a summary.

    Args:
        device (torch.device): Target device (cpu or cuda).

    Returns:
        DigitNetwork: Initialised model on the given device.
    """
    print("\n[Task 1B] Building DigitNetwork CNN...")

    model = DigitNetwork().to(device)

    print("  Network architecture:")
    print(model)

    return model


def task1c_train(
    model:       DigitNetwork,
    data_loader: MNISTDataLoader,
    device:      torch.device,
    plotter:     Plotter,
    num_epochs:  int = NUM_EPOCHS,
) -> tuple:
    """
    Subtask 1C: Train the model for num_epochs, evaluating after each epoch.

    Uses SGD with momentum as the optimiser and NLL loss (via Trainer).
    After training, plots loss and accuracy curves.

    Args:
        model       (DigitNetwork):    The network to train.
        data_loader (MNISTDataLoader): Provides train and test loaders.
        device      (torch.device):    Target device.
        plotter     (Plotter):         For saving training curves.
        num_epochs  (int):             Number of epochs to train.

    Returns:
        Tuple of (Trainer, Evaluator) with full history attached.
    """
    print(f"\n[Task 1C] Training for {num_epochs} epochs...")

    # SGD optimiser with momentum (common choice for MNIST)
    optimiser = optim.SGD(
        model.parameters(),
        lr       = LEARNING_RATE,
        momentum = MOMENTUM,
    )

    trainer   = Trainer(model=model, optimiser=optimiser, device=device)
    evaluator = Evaluator(model=model, device=device)

    for epoch in range(1, num_epochs + 1):
        print(f"\n  --- Epoch {epoch}/{num_epochs} ---")

        # One full pass through training data
        trainer.train_epoch(data_loader.train_loader, epoch_num=epoch)

        # Evaluate on test set after each epoch
        evaluator.evaluate(data_loader.test_loader)

    # Plot combined training and test curves
    plotter.plot_training_curves(
        train_losses     = trainer.train_losses,
        test_losses      = evaluator.test_losses,
        train_accuracies = trainer.train_accuracies,
        test_accuracies  = evaluator.test_accuracies,
    )

    return trainer, evaluator


def task1d_save_model(model: DigitNetwork, model_io: ModelIO) -> None:
    """
    Subtask 1D: Save the trained model weights to disk.

    Args:
        model    (DigitNetwork): Trained model.
        model_io (ModelIO):      Handles file I/O for .pth files.
    """
    print("\n[Task 1D] Saving trained model...")
    model_io.save(model, MODEL_FILENAME)


def task1e_evaluate_test_samples(
    device:      torch.device,
    data_loader: MNISTDataLoader,
    plotter:     Plotter,
) -> None:
    """
    Subtask 1E: Reloads the saved model and runs it on the first 10 test
    examples. Prints per-sample output values, predicted index, and true
    label. Plots the first 9 as a 3x3 grid with predictions above each.

    Uses load_trained_model() from tasks/read_network.py to load the model.
    run_on_test_samples logic lives here as it is specific to Task 1E.

    Args:
        device      (torch.device):    Target device.
        data_loader (MNISTDataLoader): Provides the unshuffled test loader.
        plotter     (Plotter):         For saving the prediction grid.
    """
    print("\n[Task 1E] Reloading model and running on first 10 test samples...")

    # Load model via shared utility (also prints model structure)
    model, device = load_trained_model(
        model_file = MODEL_FILENAME,
        model_dir  = MODEL_DIR,
        device     = device,
    )

    # Grab the first batch — test loader is unshuffled so always same samples
    images, labels = next(iter(data_loader.test_loader))

    # Print per-sample output values, predicted index, and true label
    evaluator = Evaluator(model=model, device=device)
    predictions, _ = evaluator.predict_samples(
        data   = images,
        labels = labels,
        n      = 10,
    )

    # Plot first 9 as a 3x3 grid with predictions above each image
    plotter.plot_predictions_grid(
        images      = images,
        predictions = predictions,
        labels      = labels.tolist(),
        n           = 9,
        filename    = "task1e_predictions.png",
        title       = "Task 1E — First 9 Test Digits with Predictions",
    )


def task1f_evaluate_handwritten(
    device:   torch.device,
    model_io: ModelIO,
    plotter:  Plotter,
) -> None:
    """
    Subtask 1F: Load handwritten digit images and classify them.

    Reads images from HANDWRITTEN_DIR, applies the MNIST pre-processing
    pipeline (greyscale, resize, invert, normalise), runs them through the
    network, and displays the results.

    Skips gracefully if the handwritten directory does not exist.

    Args:
        device   (torch.device): Target device.
        model_io (ModelIO):      For loading the saved model.
        plotter  (Plotter):      For displaying results.
    """
    print("\n[Task 1F] Running handwritten digit inference...")

    if not os.path.isdir(HANDWRITTEN_DIR):
        print(
            f"  [SKIP] Handwritten directory not found: {HANDWRITTEN_DIR}\n"
            f"  Place digit images (0.png – 9.png) in that directory and re-run."
        )
        return

    # Load and pre-process handwritten images
    hw_loader = HandwrittenLoader(image_dir=HANDWRITTEN_DIR)
    print(f"  Loaded {len(hw_loader)} handwritten digit images.")

    # Reload model
    model = DigitNetwork()
    model_io.load(model, MODEL_FILENAME)
    model = model.to(device)

    evaluator = Evaluator(model=model, device=device)
    batch     = hw_loader.as_batch()

    predictions, _ = evaluator.predict_samples(
        data   = batch,
        labels = hw_loader.get_labels(),
        n      = len(hw_loader),
    )

    plotter.plot_handwritten_results(
        images      = hw_loader.get_tensors(),
        predictions = predictions,
        labels      = hw_loader.get_labels(),
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main(argv: list) -> None:
    """
    Main function for Task 1. Runs subtasks 1A through 1F in sequence.

    Command-line arguments (optional):
        argv[1]: Number of epochs  (default: 5)
        argv[2]: Batch size        (default: 64)

    Args:
        argv (list): sys.argv passed from the if __name__ guard.
    """
    # Parse optional CLI overrides
    num_epochs = int(argv[1]) if len(argv) > 1 else NUM_EPOCHS
    batch_size = int(argv[2]) if len(argv) > 2 else BATCH_SIZE

    print("=" * 60)
    print("  Task 1: Build and Train MNIST Digit Recognition Network")
    print(f"  Epochs: {num_epochs}  |  Batch size: {batch_size}")
    print("=" * 60)

    # Auto-detect the best available accelerator (CUDA, MPS, or CPU)
    device = get_device()

    # Shared utility instances
    plotter  = Plotter(output_dir=OUTPUT_DIR, show=True)
    model_io = ModelIO(model_dir=MODEL_DIR)

    # Load MNIST data
    data_loader = MNISTDataLoader(data_dir=DATA_DIR, batch_size=batch_size)

    # --- 1A: Visualise first 6 test digits ---
    task1a_load_and_visualise(data_loader, plotter)

    # --- 1B: Build network ---
    model = task1b_build_network(device)

    # --- 1C: Train and evaluate ---
    task1c_train(model, data_loader, device, plotter, num_epochs)

    # --- 1D: Save model ---
    task1d_save_model(model, model_io)

    # --- 1E: Reload and run on first 10 test samples ---
    task1e_evaluate_test_samples(device, data_loader, plotter)

    # --- 1F: Run on handwritten digits ---
    task1f_evaluate_handwritten(device, model_io, plotter)

    print("\n[Task 1] Complete.\n")


if __name__ == "__main__":
    main(sys.argv)