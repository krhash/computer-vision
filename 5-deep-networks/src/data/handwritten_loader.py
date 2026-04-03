# src/data/handwritten_loader.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Loads user-supplied handwritten digit images (0-9), converts
#              them to greyscale, resizes to 28x28, inverts intensities to
#              match the MNIST format (white digit on black background), and
#              normalises using the same MNIST statistics.
#
# Input: one file per digit placed in data/handwritten/
#        Named by digit: 0.png, 1.jpg, 2.png ... 9.png (any size accepted)

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# MNIST normalisation constants (must match training transform)
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)

# Expected pixel size after resize
TARGET_SIZE = (28, 28)


class HandwrittenLoader:
    """
    Loads and pre-processes handwritten digit images for inference.

    Pre-processing pipeline (per project spec):
        1. Read image via OpenCV (any input size accepted)
        2. Convert to greyscale
        3. Resize to 28x28 using area interpolation
        4. Invert intensities  (photos: dark digit on white background;
           MNIST format: white digit on black background)
        5. Convert to normalised float tensor using MNIST mean/std

    Expected directory layout:
        handwritten_dir/
            0.png  (or .jpg / .jpeg / .bmp)
            1.png
            ...
            9.png

    Files are loaded in sorted order so indices match digit labels.
    The label is inferred from the first character of the filename stem.

    Author: Krushna Sanjay Sharma
    """

    # Supported image file extensions
    _SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, image_dir: str):
        """
        Initialises the loader and reads all digit images from image_dir.

        Args:
            image_dir (str): Path to the directory containing digit images.

        Raises:
            FileNotFoundError: If image_dir does not exist.
            ValueError:        If no supported images are found.
        """
        self._image_dir = Path(image_dir)

        if not self._image_dir.exists():
            raise FileNotFoundError(
                f"Handwritten image directory not found: {image_dir}"
            )

        # Normalisation transform matching MNIST training pipeline
        self._normalise = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])

        # Load all images on construction
        self._images, self._labels = self._load_images()

        if len(self._images) == 0:
            raise ValueError(
                f"No supported images found in: {image_dir}\n"
                f"Supported extensions: {self._SUPPORTED_EXTENSIONS}\n"
                f"Expected filenames:   0.png, 1.png ... 9.png"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_images(self) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Scans image_dir, loads each image, and applies the full
        pre-processing pipeline.

        Files are sorted so that '0.png' -> label 0, '1.png' -> label 1, etc.
        The label is inferred from the filename stem (first character).

        Returns:
            Tuple of (list of tensors shape (1,28,28), list of int labels).
        """
        tensors: List[torch.Tensor] = []
        labels:  List[int]         = []

        # Collect and sort all supported files
        image_files = sorted([
            f for f in self._image_dir.iterdir()
            if f.suffix.lower() in self._SUPPORTED_EXTENSIONS
        ])

        for img_path in image_files:
            tensor, label = self._process_single_image(img_path)
            if tensor is not None:
                tensors.append(tensor)
                labels.append(label)

        return tensors, labels

    def _process_single_image(
        self, img_path: Path
    ) -> Tuple[torch.Tensor | None, int]:
        """
        Applies the full pre-processing pipeline to a single image file.

        Args:
            img_path (Path): Path to the image file.

        Returns:
            Tuple of (preprocessed tensor (1,28,28), int label).
            Returns (None, -1) if the image cannot be read.
        """
        # Read image in greyscale directly via OpenCV (handles any input size)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARNING] Could not read image: {img_path}", file=sys.stderr)
            return None, -1

        # Threshold BEFORE resizing — at full resolution Otsu cleanly separates
        # dark digit from white background before downscaling blurs the pixels.
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert: photos are dark digit on white background;
        # MNIST format is white digit on black background.
        img = cv2.bitwise_not(img)

        # Resize to 28x28 AFTER thresholding and inversion
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Convert to uint8 for ToTensor compatibility
        img_uint8 = img.astype(np.uint8)

        # Apply normalisation transform -> tensor shape (1, 28, 28)
        tensor = self._normalise(img_uint8)

        # Infer label from the filename stem (e.g. "3.png" -> label 3)
        try:
            label = int(img_path.stem[0])
        except ValueError:
            label = -1

        return tensor, label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tensors(self) -> List[torch.Tensor]:
        """Returns all pre-processed image tensors, each shape (1, 28, 28)."""
        return self._images

    def get_labels(self) -> List[int]:
        """Returns inferred labels (0-9, or -1 if filename has no digit prefix)."""
        return self._labels

    def as_batch(self) -> torch.Tensor:
        """Stacks all tensors into a single batch of shape (N, 1, 28, 28)."""
        return torch.stack(self._images)

    def __len__(self) -> int:
        """Returns the number of loaded handwritten images."""
        return len(self._images)