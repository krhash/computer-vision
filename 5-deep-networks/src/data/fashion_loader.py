# src/data/fashion_loader.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Data loader for the Fashion MNIST dataset. Mirrors
#              MNISTDataLoader interface exactly so it can be used as a
#              drop-in replacement in any training/evaluation pipeline.

import torch
import torchvision
import torchvision.transforms as transforms

# Fashion MNIST mean and std (pre-computed, similar to MNIST)
FASHION_MEAN = (0.2860,)
FASHION_STD  = (0.3530,)

# Human-readable class labels for Fashion MNIST
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMNISTLoader:
    """
    Encapsulates all Fashion MNIST data loading logic.

    Drop-in replacement for MNISTDataLoader — exposes identical
    train_loader and test_loader properties.

    Fashion MNIST is more challenging than digit MNIST (~91-93% baseline
    vs ~99%) making it better suited for hyperparameter experimentation
    as differences between configurations are more visible.

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        """
        Initialises the loader and builds train/test DataLoaders.

        Args:
            data_dir   (str): Directory where Fashion MNIST will be cached.
            batch_size (int): Mini-batch size for the training DataLoader.
        """
        self._data_dir   = data_dir
        self._batch_size = batch_size

        # Standard normalisation transform for Fashion MNIST
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MEAN, FASHION_STD),
        ])

        self._train_loader = self._build_train_loader()
        self._test_loader  = self._build_test_loader()

    def _build_train_loader(self) -> torch.utils.data.DataLoader:
        """
        Downloads (if needed) and wraps the Fashion MNIST training set.

        Returns:
            DataLoader over 60 000 training examples, shuffled.
        """
        dataset = torchvision.datasets.FashionMNIST(
            root      = self._data_dir,
            train     = True,
            download  = True,
            transform = self._transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = self._batch_size,
            shuffle    = True,
        )

    def _build_test_loader(self) -> torch.utils.data.DataLoader:
        """
        Downloads (if needed) and wraps the Fashion MNIST test set.

        Not shuffled so first-N examples are deterministic.

        Returns:
            DataLoader over 10 000 test examples, unshuffled.
        """
        dataset = torchvision.datasets.FashionMNIST(
            root      = self._data_dir,
            train     = False,
            download  = True,
            transform = self._transform,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size = 1000,
            shuffle    = False,
        )

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

    @property
    def class_names(self) -> list:
        """Returns the 10 Fashion MNIST class name strings."""
        return FASHION_CLASSES