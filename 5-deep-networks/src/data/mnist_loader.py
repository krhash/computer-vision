# src/data/mnist_loader.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Handles downloading, transforming, and loading the MNIST digit
#              dataset into PyTorch DataLoader objects for training and testing.

import torch
import torchvision
import torchvision.transforms as transforms


# MNIST dataset mean and standard deviation (pre-computed constants)
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


class MNISTDataLoader:
    """
    Encapsulates all MNIST data loading logic.

    Responsibilities:
        - Download MNIST to a local data directory (if not already present)
        - Apply standard normalisation transforms
        - Expose train and test DataLoader objects

    The test set is intentionally NOT shuffled so that the first N examples
    are always the same digits across runs (required by subtasks 1A and 1E).

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        """
        Initialises the data loader and builds train/test DataLoaders.

        Args:
            data_dir   (str): Directory where MNIST will be stored/cached.
            batch_size (int): Mini-batch size for the training DataLoader.
                              The test loader always uses batch_size=1000.
        """
        self._data_dir   = data_dir
        self._batch_size = batch_size

        # Standard MNIST transform: convert PIL image to tensor, then normalise
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])

        # Build the underlying datasets and loaders on construction
        self._train_loader = self._build_train_loader()
        self._test_loader  = self._build_test_loader()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_train_loader(self) -> torch.utils.data.DataLoader:
        """
        Downloads (if needed) and wraps the MNIST training set.

        Returns:
            DataLoader over 60 000 training examples, shuffled each epoch.
        """
        train_dataset = torchvision.datasets.MNIST(
            root      = self._data_dir,
            train     = True,
            download  = True,
            transform = self._transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self._batch_size,
            shuffle    = True,
        )

    def _build_test_loader(self) -> torch.utils.data.DataLoader:
        """
        Downloads (if needed) and wraps the MNIST test set.

        The test set is NOT shuffled so example indices are deterministic.

        Returns:
            DataLoader over 10 000 test examples, unshuffled.
        """
        test_dataset = torchvision.datasets.MNIST(
            root      = self._data_dir,
            train     = False,
            download  = True,
            transform = self._transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 1000,
            shuffle    = False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        """Returns the training DataLoader (shuffled, mini-batched)."""
        return self._train_loader

    @property
    def test_loader(self) -> torch.utils.data.DataLoader:
        """Returns the test DataLoader (unshuffled, batch_size=1000)."""
        return self._test_loader

    @property
    def batch_size(self) -> int:
        """Returns the configured training batch size."""
        return self._batch_size