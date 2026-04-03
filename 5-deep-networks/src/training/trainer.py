# src/training/trainer.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Trainer class that manages the training loop for a PyTorch
#              model. Responsible for a single epoch of training and
#              accumulates loss/accuracy history for plotting.

import sys
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    Manages the training loop for a PyTorch nn.Module.

    Responsibilities:
        - Execute one training epoch at a time (train_epoch)
        - Track per-epoch training loss and accuracy
        - Keep the model and optimiser as owned state

    This class is intentionally decoupled from evaluation (see Evaluator)
    so that train and test metrics can be collected independently.

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        model:          nn.Module,
        optimiser:      optim.Optimizer,
        device:         torch.device,
        log_interval:   int = 100,
    ):
        """
        Initialises the Trainer.

        Args:
            model        (nn.Module):         The network to train.
            optimiser    (optim.Optimizer):   Optimiser (e.g. Adam, SGD).
            device       (torch.device):      CPU or CUDA device.
            log_interval (int):               Print progress every N batches.
        """
        self._model        = model
        self._optimiser    = optimiser
        self._device       = device
        self._log_interval = log_interval

        # NLL loss matches the log_softmax output of DigitNetwork
        self._criterion = nn.NLLLoss()

        # History lists — appended once per epoch
        self._train_losses:    List[float] = []
        self._train_accuracies: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch_num:    int,
    ) -> float:
        """
        Runs one complete pass through the training data.

        After each batch the gradients are zeroed, loss is back-propagated,
        and the optimiser takes a step. Training loss and accuracy for this
        epoch are appended to the internal history.

        Args:
            train_loader (DataLoader): Loader over the training set.
            epoch_num    (int):        Current epoch index (for logging).

        Returns:
            float: Average training loss for this epoch.
        """
        self._model.train()

        epoch_loss     = 0.0
        correct        = 0
        total          = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to the configured device
            data, target = data.to(self._device), target.to(self._device)

            # Zero gradients from the previous step
            self._optimiser.zero_grad()

            # Forward pass
            output = self._model(data)

            # Compute NLL loss
            loss = self._criterion(output, target)

            # Backward pass and optimiser step
            loss.backward()
            self._optimiser.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            pred        = output.argmax(dim=1, keepdim=True)
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total      += len(data)

            # Periodic console log
            if (batch_idx + 1) % self._log_interval == 0:
                print(
                    f"  Epoch {epoch_num} "
                    f"[{(batch_idx + 1) * len(data):>5d}/{len(train_loader.dataset)}] "
                    f"  Loss: {loss.item():.4f}",
                    file=sys.stdout,
                )

        # Average loss and accuracy over the full epoch
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        self._train_losses.append(avg_loss)
        self._train_accuracies.append(accuracy)

        print(
            f"  --> Epoch {epoch_num} training complete: "
            f"avg loss = {avg_loss:.4f}, accuracy = {accuracy:.2f}%"
        )

        return avg_loss

    @property
    def model(self) -> nn.Module:
        """Returns the model being trained."""
        return self._model

    @property
    def train_losses(self) -> List[float]:
        """Returns per-epoch training loss history."""
        return self._train_losses

    @property
    def train_accuracies(self) -> List[float]:
        """Returns per-epoch training accuracy history (percentage)."""
        return self._train_accuracies