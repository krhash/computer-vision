"""
Author: Krushna Sanjay Sharma
Description: Utilities for transfer-learning training phases.
"""

import torch.nn as nn
from typing import Iterator

class TransferTrainer:
    """
    Orchestrates the two-phase training (fine-tuning) strategy by wrapping the model
    and controlling gradient requirements for progressive unfreezing.
    """

    def __init__(self, model: nn.Module):
        """
        Initializes the TransferTrainer.

        Args:
            model (nn.Module): The model to manage.
        """
        self.model = model

    def freeze_all(self):
        """Freezes all parameters in the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_head(self):
        """
        Unfreezes only the final classification head.
        Works across various model architectures (ViT, ResNet, DenseNet) by inspecting
        common network head namings mapping to torchvision structures.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "fc"):
            # ResNet
            for param in self.model.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "model") and hasattr(self.model.model, "classifier"):
            # DenseNet
            for param in self.model.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "model") and hasattr(self.model.model, "heads"):
            # ViT
            for param in self.model.model.heads.parameters():
                param.requires_grad = True
        else:
            # Fallback
            print("[WARNING] Could not automatically unfreeze head based on standard names.")

    def unfreeze_last_n(self, n: int):
        """
        Progressively unfreezes the last n blocks of the model.
        Usually relies on a custom proxy method defined on the wrapped model.

        Args:
            n (int): Number of blocks to unfreeze.
        """
        if hasattr(self.model, "unfreeze_last_n_blocks"):
            self.model.unfreeze_last_n_blocks(n)
        else:
            print(f"[WARNING] Model {type(self.model).__name__} does not implement `unfreeze_last_n_blocks`. Ignoring.")

    def get_trainable_params(self) -> Iterator[nn.Parameter]:
        """
        Retrieves parameters that currently require gradients.
        Crucial for passing dynamically updating parameters to the optimizer when phases change.

        Returns:
            Iterator[nn.Parameter]: Generator of trainable parameters.
        """
        return filter(lambda p: p.requires_grad, self.model.parameters())
