"""
Author: Krushna Sanjay Sharma
Description: Transform pipelines for UCEC histopathology patches.
"""

from torchvision import transforms
from typing import Callable


class PatchTransforms:
    """
    A factory class for obtaining training and validation image transformations.
    Applies standard ImageNet normalization and relevant augmentations.
    """
    # ImageNet normalization parameters
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    PATCH_SIZE = 224

    @classmethod
    def get_train_transforms(cls) -> Callable:
        """
        Returns the data augmentation pipeline for training.
        Includes random horizontal/vertical flips, rotation, color jitter, and normalization.

        Returns:
            Callable: A torchvision Compose object containing the transform pipeline.
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.MEAN, std=cls.STD)
        ])

    @classmethod
    def get_val_transforms(cls) -> Callable:
        """
        Returns the transform pipeline for validation/testing.
        Includes center cropping and normalization (no random augmentations).

        Returns:
            Callable: A torchvision Compose object containing the transform pipeline.
        """
        return transforms.Compose([
            transforms.CenterCrop(cls.PATCH_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=cls.MEAN, std=cls.STD)
        ])
