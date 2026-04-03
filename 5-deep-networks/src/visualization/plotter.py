# src/visualization/plotter.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Plotter class responsible for all matplotlib visualisations
#              required by Tasks 1–5. Each method is a single, focused plot.

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


class Plotter:
    """
    Centralises all matplotlib plotting for the project.

    Each public method produces one figure, saves it to output_dir
    (if provided), and optionally displays it with plt.show().

    Methods cover:
        - plot_sample_grid        : Task 1A — first 6 MNIST test digits
        - plot_training_curves    : Task 1C — train/test loss & accuracy
        - plot_predictions_grid   : Task 1E — 3x3 grid with predictions
        - plot_handwritten_results: Task 1F — handwritten digits + results

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, output_dir: str = "./outputs", show: bool = True):
        """
        Initialises the Plotter.

        Args:
            output_dir (str):  Directory where figures are saved as PNG files.
                               Created automatically if it does not exist.
            show       (bool): If True, call plt.show() after each plot.
        """
        self._output_dir = Path(output_dir)
        self._show       = show
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Task 1A — Sample digit grid
    # ------------------------------------------------------------------

    def plot_sample_grid(
        self,
        images:   torch.Tensor,
        labels:   List[int],
        n:        int   = 6,
        filename: str   = "task1a_sample_digits.png",
        title:    str   = "First 6 MNIST Test Digits",
    ) -> None:
        """
        Plots the first n MNIST test digits in a 2-row grid.

        Args:
            images   (Tensor):   Batch tensor of shape (N, 1, 28, 28).
            labels   (List[int]): Corresponding ground-truth labels.
            n        (int):      Number of digits to display (default 6).
            filename (str):      Output filename saved to output_dir.
            title    (str):      Overall figure title.
        """
        cols = min(n, 6)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        fig.suptitle(title, fontsize=14)

        # Flatten axes for uniform indexing when rows=1
        axes_flat = np.array(axes).flatten()

        for i in range(n):
            ax  = axes_flat[i]
            img = images[i].squeeze().numpy()  # (28, 28)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Label: {labels[i]}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide any unused subplot axes
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 1C — Training & test curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        train_losses:      List[float],
        test_losses:       List[float],
        train_accuracies:  List[float],
        test_accuracies:   List[float],
        filename:          str = "task1c_training_curves.png",
    ) -> None:
        """
        Plots training/test loss and training/test accuracy side by side.

        Args:
            train_losses     (List[float]): Per-epoch training loss.
            test_losses      (List[float]): Per-epoch test loss.
            train_accuracies (List[float]): Per-epoch training accuracy (%).
            test_accuracies  (List[float]): Per-epoch test accuracy (%).
            filename         (str):         Output filename.
        """
        epochs = range(1, len(train_losses) + 1)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Training Progress", fontsize=14)

        # --- Loss subplot ---
        ax_loss.plot(epochs, train_losses,     color="steelblue",  label="Train Loss")
        ax_loss.plot(epochs, test_losses,      color="darkorange",  label="Test Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Average NLL Loss")
        ax_loss.set_title("Loss per Epoch")
        ax_loss.legend()
        ax_loss.grid(True, linestyle="--", alpha=0.5)

        # --- Accuracy subplot ---
        ax_acc.plot(epochs, train_accuracies,  color="steelblue",  label="Train Accuracy")
        ax_acc.plot(epochs, test_accuracies,   color="darkorange",  label="Test Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title("Accuracy per Epoch")
        ax_acc.legend()
        ax_acc.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 1E — 3x3 prediction grid
    # ------------------------------------------------------------------

    def plot_predictions_grid(
        self,
        images:      torch.Tensor,
        predictions: List[int],
        labels:      Optional[List[int]] = None,
        n:           int  = 9,
        filename:    str  = "task1e_predictions.png",
        title:       str  = "Network Predictions on Test Set",
    ) -> None:
        """
        Plots a 3x3 grid of images with the predicted class above each.

        Args:
            images      (Tensor):        Images tensor (N, 1, 28, 28).
            predictions (List[int]):     Predicted labels for first n.
            labels      (List[int]|None): Ground-truth labels (optional).
            n           (int):           Number of images (default 9 → 3x3).
            filename    (str):           Output filename.
            title       (str):           Figure title.
        """
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        fig.suptitle(title, fontsize=13)

        axes_flat = np.array(axes).flatten()

        for i in range(n):
            ax  = axes_flat[i]
            img = images[i].squeeze().numpy()

            ax.imshow(img, cmap="gray")

            # Build subtitle: prediction (and true label if available)
            subtitle = f"Pred: {predictions[i]}"
            if labels is not None:
                match = "✓" if predictions[i] == labels[i] else "✗"
                subtitle += f"  True: {labels[i]} {match}"

            ax.set_title(subtitle, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 1F — Handwritten digit results
    # ------------------------------------------------------------------

    def plot_handwritten_results(
        self,
        images:      List[torch.Tensor],
        predictions: List[int],
        labels:      Optional[List[int]] = None,
        filename:    str = "task1f_handwritten.png",
        title:       str = "Handwritten Digit Predictions",
    ) -> None:
        """
        Plots all handwritten digit images with their predicted labels.

        Args:
            images      (List[Tensor]):   List of (1, 28, 28) tensors.
            predictions (List[int]):      Predicted class for each image.
            labels      (List[int]|None): True labels (if known).
            filename    (str):            Output filename.
            title       (str):            Figure title.
        """
        n    = len(images)
        rows, cols = 3, 5

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        fig.suptitle(title, fontsize=13)

        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i in range(n):
            ax  = axes_flat[i]
            img = images[i].squeeze().numpy()

            ax.imshow(img, cmap="gray")

            subtitle = f"Pred: {predictions[i]}"
            if labels is not None and labels[i] != -1:
                match    = "✓" if predictions[i] == labels[i] else "✗"
                subtitle += f"  True: {labels[i]} {match}"

            ax.set_title(subtitle, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 2A — Filter weight visualisation
    # ------------------------------------------------------------------

    def plot_filters(
        self,
        filters:  list,
        filename: str = "task2a_filters.png",
        title:    str = "Task 2A — conv1 Filter Weights",
    ) -> None:
        """
        Visualises the 10 conv1 filter weight arrays in a 3x4 grid.
        The two unused cells in the grid are hidden.

        Args:
            filters  (list): List of 10 numpy arrays, each (5, 5).
            filename (str):  Output filename.
            title    (str):  Figure title.
        """
        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        fig.suptitle(title, fontsize=13)

        axes_flat = np.array(axes).flatten()

        for i, f in enumerate(filters):
            ax = axes_flat[i]
            ax.imshow(f, cmap="viridis")
            ax.set_title(f"Filter {i}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide the two unused cells in the 3x4 grid (10 filters, 12 cells)
        for j in range(len(filters), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    def plot_filter_responses(
        self,
        filters:         list,
        filtered_images: list,
        filename:        str = "task2b_filter_effects.png",
        title:           str = "Task 2B — conv1 Filter | Response on First Test Image",
    ) -> None:
        """
        Plots filters and responses in a 5-row x 4-col layout.

        Layout (each row shows two filter+response pairs):
            Col 0: filter i      Col 1: response i
            Col 2: filter i+5    Col 3: response i+5

        Width ratios make filter columns narrower than response columns
        since filters are 5x5 and responses are 28x28.

        Args:
            filters         (list): List of 10 numpy arrays, each (5, 5).
            filtered_images (list): List of 10 filtered images, each (28, 28).
            filename        (str):  Output filename.
            title           (str):  Figure title.
        """
        rows = 5
        fig, axes = plt.subplots(
            rows, 4,
            figsize = (10, rows * 2.2),
            gridspec_kw = {"width_ratios": [1, 3, 1, 3]},
        )
        fig.suptitle(title, fontsize=11)

        # Column headers
        for col, lbl in enumerate(["Filter", "Response", "Filter", "Response"]):
            axes[0, col].set_title(lbl, fontsize=9)

        for row in range(rows):
            i = row          # filters 0-4  in left pair
            j = row + 5      # filters 5-9  in right pair

            # --- Left pair: filter i ---
            axes[row, 0].imshow(filters[i], cmap="viridis")
            axes[row, 0].set_ylabel(f"F{i}", fontsize=8, rotation=0, labelpad=12)
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(filtered_images[i], cmap="gray")
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

            # --- Right pair: filter j ---
            axes[row, 2].imshow(filters[j], cmap="viridis")
            axes[row, 2].set_ylabel(f"F{j}", fontsize=8, rotation=0, labelpad=12)
            axes[row, 2].set_xticks([])
            axes[row, 2].set_yticks([])

            axes[row, 3].imshow(filtered_images[j], cmap="gray")
            axes[row, 3].set_xticks([])
            axes[row, 3].set_yticks([])

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 3 — Greek letter training curve + results
    # ------------------------------------------------------------------

    def plot_greek_training_curve(
        self,
        losses:   list,
        filename: str = "task3_training_curve.png",
        title:    str = "Task 3 — Greek Letter Transfer Learning Loss",
    ) -> None:
        """
        Plots the per-epoch training loss for the Greek letter transfer
        learning task.

        Args:
            losses   (list): Per-epoch average training loss values.
            filename (str):  Output filename.
            title    (str):  Figure title.
        """
        epochs = range(1, len(losses) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, losses, color="steelblue", marker="o", label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average NLL Loss")
        ax.set_title(title, fontsize=13)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    def plot_greek_predictions(
        self,
        images:      list,
        predictions: list,
        labels:      list,
        class_names: list,
        filename:    str = "task3_custom_greek.png",
        title:       str = "Task 3 — Custom Greek Letter Predictions",
    ) -> None:
        """
        Plots custom Greek letter images with their predicted class labels.

        Args:
            images      (list):  List of tensors (1, 28, 28).
            predictions (list):  Predicted class indices.
            labels      (list):  True class indices (-1 if unknown).
            class_names (list):  Class name strings e.g. ['alpha','beta','gamma'].
            filename    (str):   Output filename.
            title       (str):   Figure title.
        """
        n    = len(images)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        fig.suptitle(title, fontsize=13)

        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i in range(n):
            ax  = axes_flat[i]
            img = images[i].squeeze().numpy()
            ax.imshow(img, cmap="gray")

            pred_name = class_names[predictions[i]]
            subtitle  = f"Pred: {pred_name}"
            if labels[i] != -1:
                true_name = class_names[labels[i]]
                match     = "✓" if predictions[i] == labels[i] else "✗"
                subtitle += f"\nTrue: {true_name} {match}"

            ax.set_title(subtitle, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 4 — Transformer training curves
    # ------------------------------------------------------------------

    def plot_transformer_curves(
        self,
        train_losses:     list,
        test_losses:      list,
        train_accuracies: list,
        test_accuracies:  list,
        filename:         str = "task4_transformer_curves.png",
    ) -> None:
        """
        Plots training/test loss and accuracy curves for the transformer.
        Reuses the same layout as plot_training_curves (Task 1C).

        Args:
            train_losses     (list): Per-epoch training loss.
            test_losses      (list): Per-epoch test loss.
            train_accuracies (list): Per-epoch training accuracy (%).
            test_accuracies  (list): Per-epoch test accuracy (%).
            filename         (str):  Output filename.
        """
        self.plot_training_curves(
            train_losses     = train_losses,
            test_losses      = test_losses,
            train_accuracies = train_accuracies,
            test_accuracies  = test_accuracies,
            filename         = filename,
        )

    # ------------------------------------------------------------------
    # Task 3+ plots added below
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_and_show(self, fig: plt.Figure, filename: str) -> None:
        """
        Saves a figure to output_dir and optionally displays it.

        Args:
            fig      (Figure): The matplotlib figure to save.
            filename (str):    Output filename (PNG).
        """
        save_path = self._output_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Plotter] Saved: {save_path}")

        if self._show:
            plt.show()

        plt.close(fig)