# src/training/transfer_trainer.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: TransferTrainer class that modifies a pre-trained DigitNetwork
#              for transfer learning. Freezes all existing weights and replaces
#              the final fully-connected layer with a new layer sized for the
#              target number of output classes.

import torch
import torch.nn as nn
import torch.optim as optim

from src.network.digit_network import DigitNetwork


class TransferTrainer:
    """
    Adapts a pre-trained DigitNetwork for transfer learning.

    Responsibilities:
        - Freeze all network parameters (no gradient updates)
        - Replace fc2 (the final layer) with a new Linear layer
          sized for the target number of output classes
        - Expose the modified model ready for training

    Only the new final layer is trainable. All other weights remain
    frozen at their pre-trained MNIST values.

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        model:       DigitNetwork,
        num_classes: int = 3,
        device:      torch.device = torch.device("cpu"),
    ):
        """
        Initialises the TransferTrainer.

        Freezes all parameters and replaces fc2 with a new Linear layer.

        Args:
            model       (DigitNetwork):  Pre-trained model to adapt.
            num_classes (int):           Number of output classes for the
                                         new final layer (default 3 for
                                         alpha, beta, gamma).
            device      (torch.device): Target device.
        """
        self._model       = model
        self._num_classes = num_classes
        self._device      = device

        self._freeze_all_parameters()
        self._replace_final_layer()
        self._model = self._model.to(device)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_all_parameters(self) -> None:
        """
        Freezes all network parameters so they are not updated during
        back-propagation.

        Sets requires_grad=False on every parameter in the network.
        The new final layer added afterwards will have requires_grad=True
        by default since it is a freshly constructed nn.Linear.
        """
        for param in self._model.parameters():
            param.requires_grad = False

        print("  [TransferTrainer] All parameters frozen.")

    def _replace_final_layer(self) -> None:
        """
        Replaces fc2 (the 10-class output layer) with a new Linear layer
        sized for num_classes.

        The new layer is initialised with default PyTorch weight init
        (Kaiming uniform) and has requires_grad=True automatically.

        fc2 input features: 50  (output of fc1)
        fc2 output features: num_classes (3 for Greek letters)
        """
        in_features = self._model.fc2.in_features   # 50
        self._model.fc2 = nn.Linear(in_features, self._num_classes)

        print(
            f"  [TransferTrainer] Replaced fc2: "
            f"Linear({in_features}, {self._num_classes})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model(self) -> DigitNetwork:
        """Returns the modified model with frozen weights and new fc2."""
        return self._model

    def get_optimiser(
        self,
        lr: float = 0.01,
        momentum: float = 0.5,
    ) -> optim.Optimizer:
        """
        Returns an SGD optimiser configured to update only the new fc2 layer.

        Since all other parameters have requires_grad=False, passing
        model.parameters() is equivalent to passing only fc2's parameters.

        Args:
            lr       (float): Learning rate (default 0.01).
            momentum (float): SGD momentum (default 0.5).

        Returns:
            optim.SGD configured for the trainable parameters only.
        """
        # filter() ensures only trainable params are passed to the optimiser
        trainable = filter(
            lambda p: p.requires_grad,
            self._model.parameters()
        )
        return optim.SGD(trainable, lr=lr, momentum=momentum)