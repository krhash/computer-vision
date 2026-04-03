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
    # Task 5 — Experiment sweep results
    # ------------------------------------------------------------------

    def plot_experiment_results(
        self,
        results:  list,
        baseline: float,
        filename: str = "task5_experiment_results.png",
    ) -> None:
        """
        Plots four subplots summarising the hyperparameter sweep:
            1. Accuracy vs patch_size   (dim 1 sweep)
            2. Accuracy vs embed_dim    (dim 2 sweep)
            3. Accuracy vs depth        (dim 3 sweep)
            4. Training time vs accuracy scatter (all runs)

        Args:
            results  (list):  List of ExperimentResult objects.
            baseline (float): Baseline accuracy (%) for reference lines.
            filename (str):   Output filename.
        """
        import numpy as np

        # Extract fields
        patch_sizes  = sorted(set(r.patch_size  for r in results))
        embed_dims   = sorted(set(r.embed_dim   for r in results))
        depths       = sorted(set(r.depth       for r in results))

        def best_acc_for(field, val):
            """Returns the best accuracy across all runs with field==val."""
            matches = [r.test_accuracy for r in results
                       if getattr(r, field) == val]
            return max(matches) if matches else 0.0

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle("Task 5 — Hyperparameter Sweep on Fashion MNIST", fontsize=14)

        # --- Plot 1: Accuracy vs patch_size ---
        ax = axes[0, 0]
        accs = [best_acc_for("patch_size", p) for p in patch_sizes]
        ax.plot(patch_sizes, accs, marker="o", color="steelblue")
        ax.axhline(baseline, color="darkorange", linestyle="--", label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Patch Size")
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("Dimension 1 — Patch Size")
        ax.set_xticks(patch_sizes)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # --- Plot 2: Accuracy vs embed_dim ---
        ax = axes[0, 1]
        accs = [best_acc_for("embed_dim", e) for e in embed_dims]
        ax.plot(embed_dims, accs, marker="s", color="seagreen")
        ax.axhline(baseline, color="darkorange", linestyle="--", label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("Dimension 2 — Embedding Dimension")
        ax.set_xticks(embed_dims)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # --- Plot 3: Accuracy vs depth ---
        ax = axes[1, 0]
        accs = [best_acc_for("depth", d) for d in depths]
        ax.plot(depths, accs, marker="^", color="mediumpurple")
        ax.axhline(baseline, color="darkorange", linestyle="--", label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Transformer Depth (layers)")
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("Dimension 3 — Transformer Depth")
        ax.set_xticks(depths)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # --- Plot 4: Training time vs accuracy scatter ---
        ax = axes[1, 1]
        times = [r.train_time_s   for r in results]
        accs  = [r.test_accuracy  for r in results]
        ax.scatter(times, accs, alpha=0.6, color="steelblue", s=40)
        ax.axhline(baseline, color="darkorange", linestyle="--", label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Training Time (seconds)")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Accuracy vs Training Time (all runs)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    def plot_top_configs(
        self,
        results:  list,
        baseline: float,
        top_n:    int = 10,
        filename: str = "task5_top_configs.png",
    ) -> None:
        """
        Horizontal bar chart of the top N configurations by accuracy.

        Args:
            results  (list):  List of ExperimentResult objects.
            baseline (float): Baseline accuracy for reference line.
            top_n    (int):   Number of top configs to show.
            filename (str):   Output filename.
        """
        # Sort by accuracy descending, take top N
        sorted_results = sorted(results, key=lambda r: r.test_accuracy, reverse=True)
        top = sorted_results[:top_n]

        labels = [f"p={r.patch_size} e={r.embed_dim} d={r.depth}" for r in top]
        accs   = [r.test_accuracy for r in top]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(labels[::-1], accs[::-1], color="steelblue", alpha=0.8)
        ax.axvline(baseline, color="darkorange", linestyle="--",
                   label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Test Accuracy (%)")
        ax.set_title(f"Task 5 — Top {top_n} Configurations (Fashion MNIST)", fontsize=12)
        ax.legend()
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Annotate bars with accuracy value
        for bar, acc in zip(bars[::-1], accs[::-1]):
            ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.2f}%", va="center", ha="right",
                    color="white", fontsize=8, fontweight="bold")

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Task 5B — CNN optimizer sweep plots
    # ------------------------------------------------------------------

    def plot_cnn_experiment_results(
        self,
        results:  list,
        baseline: float,
        filename: str = "task5b_cnn_results.png",
    ) -> None:
        """
        Plots four subplots summarising the CNN optimizer sweep:
            1. Accuracy vs optimizer (bar chart)
            2. Accuracy vs learning rate per optimizer (line plot)
            3. Accuracy vs momentum (SGD only)
            4. Training time vs accuracy scatter (all runs)

        Args:
            results  (list):  List of CNNExperimentResult objects.
            baseline (float): Baseline accuracy (%) for reference lines.
            filename (str):   Output filename.
        """
        optimizers = ["sgd", "adam", "adamw", "rmsprop"]
        colors     = {"sgd": "steelblue", "adam": "seagreen",
                      "adamw": "mediumpurple", "rmsprop": "darkorange"}

        def best_acc_for(field, val):
            matches = [r.test_accuracy for r in results
                       if getattr(r, field) == val]
            return max(matches) if matches else 0.0

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle("Task 5B — CNN Optimizer Sweep on Fashion MNIST", fontsize=14)

        # --- Plot 1: Best accuracy per optimizer ---
        ax = axes[0, 0]
        opt_accs = [best_acc_for("optimizer_name", o) for o in optimizers]
        bars = ax.bar(optimizers, opt_accs,
                      color=[colors[o] for o in optimizers], alpha=0.8)
        ax.axhline(baseline, color="red", linestyle="--",
                   label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Optimizer")
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("Dimension 1 — Optimizer")
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        for bar, acc in zip(bars, opt_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5,
                    f"{acc:.2f}%", ha="center", va="top",
                    color="white", fontsize=8, fontweight="bold")

        # --- Plot 2: Accuracy vs LR per optimizer ---
        ax = axes[0, 1]
        lrs = sorted(set(r.lr for r in results))
        for opt in optimizers:
            opt_results = [r for r in results if r.optimizer_name == opt]
            lr_accs = []
            for lr in lrs:
                matches = [r.test_accuracy for r in opt_results if r.lr == lr]
                lr_accs.append(max(matches) if matches else None)
            valid = [(lr, acc) for lr, acc in zip(lrs, lr_accs) if acc is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, marker="o", label=opt,
                        color=colors[opt], alpha=0.8)
        ax.axhline(baseline, color="red", linestyle="--",
                   label=f"Baseline {baseline:.2f}%")
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("Dimension 2 — Learning Rate per Optimizer")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

        # --- Plot 3: Accuracy vs momentum (SGD only) ---
        ax = axes[1, 0]
        sgd_results = [r for r in results if r.optimizer_name == "sgd"]
        momentums   = sorted(set(r.momentum for r in sgd_results))
        mom_accs    = [best_acc_for("momentum", m) for m in momentums
                       if any(r.optimizer_name == "sgd" and r.momentum == m
                              for r in results)]
        mom_vals    = [m for m in momentums
                       if any(r.optimizer_name == "sgd" and r.momentum == m
                              for r in results)]
        ax.plot(mom_vals, mom_accs, marker="s", color="steelblue")
        ax.axhline(baseline, color="red", linestyle="--",
                   label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Momentum (SGD only)")
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("Dimension 3 — SGD Momentum")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # --- Plot 4: Training time vs accuracy scatter ---
        ax = axes[1, 1]
        for opt in optimizers:
            opt_r = [r for r in results if r.optimizer_name == opt]
            if opt_r:
                ax.scatter(
                    [r.train_time_s  for r in opt_r],
                    [r.test_accuracy for r in opt_r],
                    label=opt, color=colors[opt], alpha=0.7, s=50,
                )
        ax.axhline(baseline, color="red", linestyle="--",
                   label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Training Time (seconds)")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Accuracy vs Training Time (all runs)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    def plot_cnn_top_configs(
        self,
        results:  list,
        baseline: float,
        top_n:    int = 10,
        filename: str = "task5b_cnn_top_configs.png",
    ) -> None:
        """
        Horizontal bar chart of the top N CNN configurations by accuracy.

        Args:
            results  (list):  List of CNNExperimentResult objects.
            baseline (float): Baseline accuracy for reference line.
            top_n    (int):   Number of top configs to show.
            filename (str):   Output filename.
        """
        sorted_results = sorted(
            results, key=lambda r: r.test_accuracy, reverse=True
        )[:top_n]

        labels = [
            f"{r.optimizer_name}  lr={r.lr}  mom={r.momentum}"
            for r in sorted_results
        ]
        accs = [r.test_accuracy for r in sorted_results]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels[::-1], accs[::-1], color="steelblue", alpha=0.8)
        ax.axvline(baseline, color="red", linestyle="--",
                   label=f"Baseline {baseline:.2f}%")
        ax.set_xlabel("Test Accuracy (%)")
        ax.set_title(
            f"Task 5B — Top {top_n} CNN Configs (Fashion MNIST)", fontsize=12
        )
        ax.legend()
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        for bar, acc in zip(bars[::-1], accs[::-1]):
            ax.text(
                bar.get_width() - 0.2, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va="center", ha="right",
                color="white", fontsize=8, fontweight="bold",
            )

        plt.tight_layout()
        self._save_and_show(fig, filename)

    # ------------------------------------------------------------------
    # Extension 1 — Pre-trained network filter analysis
    # ------------------------------------------------------------------

    def plot_pretrained_filters(
        self,
        filters:    list,
        model_name: str,
        filename:   str = "ext1_pretrained_filters.png",
    ) -> None:
        """
        Visualises all filters from a pre-trained model's first conv layer.

        Arranges filters in an 8-column grid (padded to fit).
        Uses a diverging colormap (RdBu) to show positive/negative weights.

        Args:
            filters    (list): List of 2D numpy filter arrays.
            model_name (str):  Model name for the figure title.
            filename   (str):  Output filename.
        """
        n    = len(filters)
        cols = 8
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        fig.suptitle(
            f"Extension 1 — {model_name} First Conv Layer Filters ({n} total)",
            fontsize=12,
        )

        axes_flat = np.array(axes).flatten()

        for i, f in enumerate(filters):
            ax = axes_flat[i]
            # Normalise each filter to [-1, 1] for consistent colour scaling
            vmax = max(abs(f.max()), abs(f.min())) + 1e-8
            ax.imshow(f, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"F{i}", fontsize=6)

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    def plot_pretrained_responses(
        self,
        filters:         list,
        filtered_images: list,
        model_name:      str,
        max_show:        int = 32,
        filename:        str = "ext1_pretrained_responses.png",
    ) -> None:
        """
        Plots filter + response pairs for a pre-trained model.

        Shows up to max_show filters to keep the figure manageable.
        Layout: 8 cols, each col pair = [filter | response].

        Args:
            filters         (list): List of 2D filter arrays.
            filtered_images (list): List of filtered image arrays.
            model_name      (str):  Model name for title.
            max_show        (int):  Max filters to show (default 32).
            filename        (str):  Output filename.
        """
        n    = min(len(filters), max_show)
        cols = 8
        rows = (n + cols - 1) // cols

        # 2 subrows per row: filter on top, response on bottom
        fig, axes = plt.subplots(
            rows * 2, cols,
            figsize = (cols * 2, rows * 3),
        )
        fig.suptitle(
            f"Extension 1 — {model_name} Filter Responses (first {n})",
            fontsize=12,
        )

        for i in range(n):
            col       = i % cols
            group_row = (i // cols) * 2

            # Filter weight
            ax_f = axes[group_row, col]
            vmax = max(abs(filters[i].max()), abs(filters[i].min())) + 1e-8
            ax_f.imshow(filters[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax_f.set_title(f"F{i}", fontsize=7)
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            # Filter response
            ax_r = axes[group_row + 1, col]
            ax_r.imshow(filtered_images[i], cmap="gray")
            ax_r.set_xticks([])
            ax_r.set_yticks([])

        # Hide unused cells
        for i in range(n, rows * cols):
            col       = i % cols
            group_row = (i // cols) * 2
            if group_row < rows * 2:
                axes[group_row, col].set_visible(False)
            if group_row + 1 < rows * 2:
                axes[group_row + 1, col].set_visible(False)

        plt.tight_layout()
        self._save_and_show(fig, filename)

    def plot_filter_comparison(
        self,
        mnist_filters:     list,
        pretrained_filters: list,
        pretrained_name:   str,
        filename:          str = "ext1_filter_comparison.png",
    ) -> None:
        """
        Side-by-side comparison of MNIST CNN filters vs pre-trained filters.

        Shows the first 10 filters from each model for direct comparison.

        Args:
            mnist_filters      (list): Our 10 MNIST conv1 filters (5x5).
            pretrained_filters (list): Pre-trained model filters (kHxkW).
            pretrained_name    (str):  Pre-trained model name.
            filename           (str):  Output filename.
        """
        n = min(10, len(pretrained_filters))

        fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))
        fig.suptitle(
            f"Filter Comparison — DigitNetwork (top) vs {pretrained_name} (bottom)",
            fontsize=11,
        )

        for i in range(n):
            # Top row: MNIST filters
            ax_top = axes[0, i]
            ax_top.imshow(mnist_filters[i], cmap="viridis")
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            if i == 0:
                ax_top.set_ylabel("MNIST\nconv1", fontsize=8)

            # Bottom row: pre-trained filters
            ax_bot = axes[1, i]
            vmax = max(abs(pretrained_filters[i].max()),
                       abs(pretrained_filters[i].min())) + 1e-8
            ax_bot.imshow(pretrained_filters[i], cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])
            if i == 0:
                ax_bot.set_ylabel(f"{pretrained_name}\nconv1", fontsize=8)

            axes[0, i].set_title(f"F{i}", fontsize=8)

        plt.tight_layout()
        self._save_and_show(fig, filename)

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