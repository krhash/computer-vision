"""
Author: Krushna Sanjay Sharma
Description: PyTorch Dataset for loading and splitting UCEC histopathology patches.
"""

import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Callable
import torch

# Fix PIL image truncation behavior globally for the dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class UCECPatchDataset(Dataset):
    """
    Dataset class for loading pre-extracted UCEC patches.
    Provides methods to split into train, val, and test subsets.
    """

    CLASS_MAP = {"normal": 0, "cancerous": 1, "tumor": 1}

    def __init__(self, data_samples: List[Tuple[str, int]], transform: Optional[Callable] = None):
        """
        Initializes the dataset with a list of file paths and labels.
        (Usually created via the `create_splits` class method).

        Args:
            data_samples (List[Tuple[str, int]]): List of (file_path, label).
            transform (Optional[Callable]): Transforms to apply to the images.
        """
        self.samples = data_samples
        self.transform = transform
        self._print_class_distribution()

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, int]: The transformed image tensor and its integer label.
        """
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            # Try loading the next sample instead
            return self.__getitem__((idx + 1) % len(self))

    def _print_class_distribution(self):
        """Prints the underlying class distribution."""
        normal = sum(1 for _, label in self.samples if label == 0)
        cancerous = sum(1 for _, label in self.samples if label == 1)
        total = len(self.samples)
        if total == 0:
            print("Dataset is empty.")
            return

        print(f"Dataset initialized with {total} samples:")
        print(f" -> Normal (0):    {normal} ({normal/total*100:.1f}%)")
        print(f" -> Cancerous (1): {cancerous} ({cancerous/total*100:.1f}%)")

    @classmethod
    def create_splits(
        cls, 
        root_dir: str, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        random_state: int = 42
    ) -> Tuple['UCECPatchDataset', 'UCECPatchDataset', 'UCECPatchDataset']:
        """
        Reads the root directory, performs a stratified train/val/test split,
        and returns three UCECPatchDataset instances.

        Args:
            root_dir (str): Path to root directory containing 'normal' and 'cancerous' folders.
            train_ratio (float): Ratio of data for training.
            val_ratio (float): Ratio of data for validation.
            test_ratio (float): Ratio of data for testing.
            train_transform (Optional[Callable]): Transform for train split.
            val_transform (Optional[Callable]): Transform for val split.
            test_transform (Optional[Callable]): Transform for test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[UCECPatchDataset, UCECPatchDataset, UCECPatchDataset]: Train, Val, and Test datasets.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        all_samples = []
        for class_name, label in cls.CLASS_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Directory not found: {class_dir}")
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith('.png'):
                    filepath = os.path.join(class_dir, filename)
                    # Skip empty/corrupted files
                    if os.path.getsize(filepath) > 1024:
                        all_samples.append((filepath, label))
        
        if not all_samples:
            raise ValueError(f"No valid .png files found in {root_dir} under 'normal' or 'cancerous'.")

        # Extract labels for stratified splitting
        labels = [label for _, label in all_samples]

        # First split: separating out the train set
        train_samples, temp_samples, _, temp_labels = train_test_split(
            all_samples, labels, train_size=train_ratio, stratify=labels, random_state=random_state
        )

        # Second split: dividing the remainder into validation and test
        # Relative ratio for validation from the remaining data
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_samples, test_samples = train_test_split(
            temp_samples, train_size=relative_val_ratio, stratify=temp_labels, random_state=random_state
        )

        print("\n--- Creating Training Dataset ---")
        train_dataset = cls(train_samples, transform=train_transform)
        print("\n--- Creating Validation Dataset ---")
        val_dataset = cls(val_samples, transform=val_transform)
        print("\n--- Creating Testing Dataset ---")
        test_dataset = cls(test_samples, transform=test_transform)

        return train_dataset, val_dataset, test_dataset
