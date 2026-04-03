# src/evaluation/filter_analyzer.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: FilterAnalyzer class responsible for extracting the conv1
#              layer weights from a trained DigitNetwork and applying them
#              to an input image using OpenCV's filter2D function.
#              Covers Task 2A (weight extraction) and Task 2B (filter effects).

import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.network.digit_network import DigitNetwork


class FilterAnalyzer:
    """
    Extracts and applies the first convolutional layer filters of a
    trained DigitNetwork.

    Responsibilities:
        - Extract conv1 weights as numpy arrays  (Task 2A)
        - Apply each filter to an image via cv2.filter2D  (Task 2B)

    The conv1 layer has shape [10, 1, 5, 5]:
        10 filters, 1 input channel, 5x5 kernel each.

    All weight access is wrapped in torch.no_grad() to prevent
    unnecessary gradient computation.

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, model: DigitNetwork):
        """
        Initialises the FilterAnalyzer with a trained DigitNetwork.

        Args:
            model (DigitNetwork): Trained model. Must have a conv1 layer
                                  with weights of shape [10, 1, 5, 5].
        """
        self._model = model

    # ------------------------------------------------------------------
    # Task 2A — Weight extraction
    # ------------------------------------------------------------------

    def get_filters(self) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Extracts the 10 conv1 filter weight tensors.

        Accesses model.conv1.weight which has shape [10, 1, 5, 5].
        Each individual filter is a 5x5 numpy array at index [i, 0].

        Prints the shape and raw values of each filter to stdout.

        Returns:
            Tuple of:
                - weight_tensor (torch.Tensor): Full weights [10, 1, 5, 5].
                - filters       (List[ndarray]): List of 10 arrays, each (5,5).
        """
        with torch.no_grad():
            # Access conv1 weights — shape [10, 1, 5, 5]
            weights = self._model.conv1.weight

        print(f"\n  conv1 weight tensor shape: {weights.shape}")
        print(f"  (filters, input_channels, height, width) = {tuple(weights.shape)}\n")

        filters: List[np.ndarray] = []

        for i in range(weights.shape[0]):
            # Extract the i-th 5x5 filter (single input channel)
            f = weights[i, 0].cpu().detach().numpy()
            filters.append(f)

            print(f"  Filter {i:2d} | shape: {f.shape}")
            print(np.array2string(f, precision=4, suppress_small=True))
            print()

        return weights, filters

    # ------------------------------------------------------------------
    # Task 2B — Filter effects via cv2.filter2D
    # ------------------------------------------------------------------

    def apply_filters(
        self,
        image_tensor: torch.Tensor,
        filters:      List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Applies each of the 10 conv1 filters to an input image using
        OpenCV's filter2D function.

        The image tensor is converted to a float32 numpy array before
        filtering. filter2D performs a 2D convolution (cross-correlation)
        with the given kernel.

        Args:
            image_tensor (torch.Tensor): Single image tensor (1, 28, 28)
                                         or (28, 28).
            filters      (List[ndarray]): List of 10 filter arrays (5x5).

        Returns:
            List[ndarray]: 10 filtered images, each of shape (28, 28).
        """
        with torch.no_grad():
            # Convert tensor to (28, 28) float32 numpy array
            img = image_tensor.squeeze().cpu().numpy().astype(np.float32)

        filtered_images: List[np.ndarray] = []

        for i, kernel in enumerate(filters):
            # Apply filter using OpenCV filter2D
            # ddepth=-1 preserves the source image depth (float32)
            filtered = cv2.filter2D(img, ddepth=-1, kernel=kernel)
            filtered_images.append(filtered)
            print(f"  Applied filter {i:2d} -> output range: "
                  f"[{filtered.min():.3f}, {filtered.max():.3f}]",
                  file=sys.stdout)

        return filtered_images