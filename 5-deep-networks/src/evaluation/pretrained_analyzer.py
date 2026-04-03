# src/evaluation/pretrained_analyzer.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: PretrainedAnalyzer extracts and applies the first
#              convolutional layer filters from a pre-trained torchvision
#              model (e.g. ResNet18). Extends the Task 2 filter analysis
#              to ImageNet-trained networks for comparison with our MNIST CNN.
#
# Supports any torchvision model whose first conv layer is accessible.
# Default: ResNet18 — 64 filters, 7x7 kernel, 3 input channels (RGB).

import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class PretrainedAnalyzer:
    """
    Extracts and applies the first convolutional layer filters from a
    pre-trained torchvision model.

    Responsibilities:
        - Load a named pre-trained model from torchvision
        - Extract the first conv layer weights
        - Convert RGB filters to greyscale for cv2.filter2D application
        - Apply each filter to a sample image

    Comparison with Task 2 (DigitNetwork):
        DigitNetwork conv1:  10 filters,  5x5, 1 channel  (MNIST greyscale)
        ResNet18    conv1:   64 filters,  7x7, 3 channels  (ImageNet RGB)

    Author: Krushna Sanjay Sharma
    """

    # Supported model names and their first conv layer attribute paths
    SUPPORTED_MODELS = {
        "resnet18":  "conv1",
        "resnet50":  "conv1",
        "vgg16":     "features.0",
        "alexnet":   "features.0",
    }

    def __init__(self, model_name: str = "resnet18"):
        """
        Loads a pre-trained torchvision model and prepares it for analysis.

        Args:
            model_name (str): Name of the torchvision model to load.
                              Supported: resnet18, resnet50, vgg16, alexnet.

        Raises:
            ValueError: If model_name is not in SUPPORTED_MODELS.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self._model_name = model_name
        self._model      = self._load_model(model_name)
        self._conv_layer = self._get_first_conv_layer()

        print(f"  [PretrainedAnalyzer] Loaded {model_name}")
        print(f"  First conv layer: {self._conv_layer}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_name: str) -> nn.Module:
        """
        Downloads and loads a pre-trained torchvision model.

        Uses the new torchvision weights API (weights='DEFAULT') which
        loads the best available pre-trained weights for the model.

        Args:
            model_name (str): Model identifier string.

        Returns:
            nn.Module: Pre-trained model in eval() mode.
        """
        print(f"  [PretrainedAnalyzer] Loading {model_name} with ImageNet weights...")

        model_fn = getattr(models, model_name)
        model    = model_fn(weights="DEFAULT")
        model.eval()
        return model

    def _get_first_conv_layer(self) -> nn.Conv2d:
        """
        Navigates the model attribute path to retrieve the first conv layer.

        Uses the SUPPORTED_MODELS mapping to find the layer by attribute path.
        Supports nested paths like 'features.0' via recursive getattr.

        Returns:
            nn.Conv2d: The first convolutional layer of the model.
        """
        attr_path = self.SUPPORTED_MODELS[self._model_name]
        layer     = self._model

        for attr in attr_path.split("."):
            try:
                layer = getattr(layer, attr)
            except AttributeError:
                # Fall back to integer index for Sequential layers (e.g. features.0)
                layer = layer[int(attr)]

        return layer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_filters(self) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Extracts the first conv layer filter weights.

        For RGB models (3 input channels), converts each filter to
        greyscale by averaging across the 3 colour channels so filters
        can be visualised and applied to greyscale images consistently.

        Returns:
            Tuple of:
                - weight_tensor (torch.Tensor): Full weights
                  shape [out_channels, in_channels, kH, kW]
                - filters (List[ndarray]): List of 2D greyscale filter
                  arrays, one per output filter.
        """
        with torch.no_grad():
            weights = self._conv_layer.weight   # [out, in, kH, kW]

        print(f"\n  {self._model_name} conv1 weight shape: {weights.shape}")
        print(f"  (filters, channels, height, width) = {tuple(weights.shape)}")

        filters: List[np.ndarray] = []

        for i in range(weights.shape[0]):
            f = weights[i].cpu().detach().numpy()  # (in_channels, kH, kW)

            if f.shape[0] == 3:
                # RGB filter — average across colour channels for visualisation
                f_grey = f.mean(axis=0)            # (kH, kW)
            else:
                # Single channel filter (e.g. greyscale MNIST)
                f_grey = f[0]

            filters.append(f_grey)

        return weights, filters

    def apply_filters(
        self,
        image:   np.ndarray,
        filters: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Applies each filter to a greyscale image using cv2.filter2D.

        Args:
            image   (ndarray): Greyscale image as float32 array (H, W).
            filters (list):    List of 2D filter arrays from get_filters().

        Returns:
            List[ndarray]: Filtered images, one per filter.
        """
        filtered: List[np.ndarray] = []

        for i, kernel in enumerate(filters):
            result = cv2.filter2D(
                image.astype(np.float32),
                ddepth = -1,
                kernel = kernel.astype(np.float32),
            )
            filtered.append(result)

        print(f"  Applied {len(filters)} filters via cv2.filter2D")
        return filtered

    def get_model_info(self) -> dict:
        """
        Returns a summary of the model and its first conv layer.

        Returns:
            dict with model_name, num_filters, kernel_size, in_channels.
        """
        w = self._conv_layer.weight
        return {
            "model_name":  self._model_name,
            "num_filters": w.shape[0],
            "kernel_size": (w.shape[2], w.shape[3]),
            "in_channels": w.shape[1],
        }

    @property
    def model_name(self) -> str:
        """Returns the loaded model name."""
        return self._model_name