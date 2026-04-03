# src/utils/model_io.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: ModelIO utility class for saving and loading PyTorch model
#              state dictionaries to/from .pth files.

import sys
from pathlib import Path

import torch
import torch.nn as nn


class ModelIO:
    """
    Handles serialisation and deserialisation of PyTorch model weights.

    Saves and loads the model's state_dict (weights only, not architecture).
    The calling code must supply a model instance with the correct
    architecture before loading weights.

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, model_dir: str = "./models"):
        """
        Initialises ModelIO and ensures the models directory exists.

        Args:
            model_dir (str): Directory where .pth files are stored.
        """
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, model: nn.Module, filename: str) -> Path:
        """
        Saves a model's state_dict to a .pth file.

        Args:
            model    (nn.Module): The trained model whose weights to save.
            filename (str):       Filename (e.g. 'mnist_cnn.pth').

        Returns:
            Path: Full path to the saved file.
        """
        save_path = self._model_dir / filename
        torch.save(model.state_dict(), save_path)
        print(f"  [ModelIO] Model saved to: {save_path}", file=sys.stdout)
        return save_path

    def load(self, model: nn.Module, filename: str) -> nn.Module:
        """
        Loads weights from a .pth file into an existing model instance.

        The model architecture must already match the saved weights.
        Sets the model to eval() mode after loading.

        Args:
            model    (nn.Module): Empty model with the correct architecture.
            filename (str):       Filename to load (e.g. 'mnist_cnn.pth').

        Returns:
            nn.Module: The same model with loaded weights, in eval() mode.

        Raises:
            FileNotFoundError: If the .pth file does not exist.
        """
        load_path = self._model_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(
                f"[ModelIO] Model file not found: {load_path}\n"
                f"Run task1_build_train.py first to generate it."
            )

        # Load weights — map_location ensures CPU loading even if saved on GPU
        state_dict = torch.load(load_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()

        print(f"  [ModelIO] Model loaded from: {load_path}", file=sys.stdout)
        return model

    def exists(self, filename: str) -> bool:
        """
        Checks whether a saved model file exists.

        Args:
            filename (str): Filename to check.

        Returns:
            bool: True if the file exists in model_dir.
        """
        return (self._model_dir / filename).exists()