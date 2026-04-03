# src/evaluation/evaluator.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Evaluator class for computing test-set accuracy and running
#              per-sample inference. Handles subtasks 1E (first 10 test
#              examples) and 1F (handwritten digit images).

import sys
from typing import List, Tuple

import torch
import torch.nn as nn


class Evaluator:
    """
    Evaluates a trained PyTorch model on a dataset or individual samples.

    Responsibilities:
        - Compute accuracy over a full DataLoader (evaluate)
        - Run inference on a fixed number of samples (predict_samples)
        - Print per-sample output values, predicted label, and true label

    The model is always put into eval() mode before inference so that
    dropout behaves deterministically (scales by 1 - drop_rate).

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialises the Evaluator.

        Args:
            model  (nn.Module):    The trained network.
            device (torch.device): CPU or CUDA device.
        """
        self._model  = model
        self._device = device

        # NLL loss matches the log_softmax output of DigitNetwork
        self._criterion = nn.NLLLoss()

        # History accumulated across evaluate() calls (one entry per call)
        self._test_losses:      List[float] = []
        self._test_accuracies:  List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> float:
        """
        Computes loss and accuracy over the full test DataLoader.

        The model is set to eval() mode for this pass and restored to
        train() mode afterwards only if it was in train() mode before.

        Args:
            test_loader (DataLoader): Loader over the test set.

        Returns:
            float: Test accuracy as a percentage (0–100).
        """
        self._model.eval()

        test_loss = 0.0
        correct   = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self._device), target.to(self._device)
                output       = self._model(data)
                test_loss   += self._criterion(output, target).item()
                pred         = output.argmax(dim=1, keepdim=True)
                correct     += pred.eq(target.view_as(pred)).sum().item()

        # Average loss across all batches
        avg_loss = test_loss / len(test_loader)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        self._test_losses.append(avg_loss)
        self._test_accuracies.append(accuracy)

        print(
            f"  --> Test set: avg loss = {avg_loss:.4f}, "
            f"accuracy = {correct}/{len(test_loader.dataset)} "
            f"({accuracy:.2f}%)"
        )

        return accuracy

    def predict_samples(
        self,
        data:   torch.Tensor,
        labels: torch.Tensor | List[int] | None = None,
        n:      int = 10,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Runs inference on the first n samples and prints a detailed report.

        For each sample the method prints:
            - The 10 raw network output values (2 decimal places)
            - The index of the maximum output (predicted class)
            - The ground-truth label (if provided)

        Args:
            data   (Tensor):       Batch tensor of shape (N, 1, 28, 28).
            labels (Tensor|list):  Ground-truth labels (optional).
            n      (int):          Number of samples to evaluate (default 10).

        Returns:
            Tuple of:
                - List[int]:            Predicted class indices for first n.
                - List[torch.Tensor]:   Raw output tensors for first n.
        """
        self._model.eval()

        # Take first n samples
        data_n   = data[:n].to(self._device)
        labels_n = None
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels_n = labels[:n].tolist()
            else:
                labels_n = list(labels)[:n]

        predictions:   List[int]          = []
        output_tensors: List[torch.Tensor] = []

        with torch.no_grad():
            outputs = self._model(data_n)   # shape (n, 10)

        print("\n" + "=" * 60)
        print(f"  Predictions on first {n} samples")
        print("=" * 60)

        for i in range(n):
            out      = outputs[i]                       # shape (10,)
            pred_idx = int(out.argmax().item())
            vals_str = "  ".join(f"{v:.2f}" for v in out.tolist())

            true_label_str = ""
            if labels_n is not None:
                true_label_str = f"  True: {labels_n[i]}"

            print(
                f"  Sample {i:2d} | Output: [{vals_str}] "
                f"| Pred: {pred_idx}{true_label_str}"
            )

            predictions.append(pred_idx)
            output_tensors.append(out.cpu())

        print("=" * 60 + "\n")

        return predictions, output_tensors

    # ------------------------------------------------------------------
    # History properties
    # ------------------------------------------------------------------

    @property
    def test_losses(self) -> List[float]:
        """Returns per-evaluation test loss history."""
        return self._test_losses

    @property
    def test_accuracies(self) -> List[float]:
        """Returns per-evaluation test accuracy history (percentage)."""
        return self._test_accuracies