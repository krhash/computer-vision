# src/data/greek_loader.py
# Project 5: Recognition using Deep Networks
# Author: Krushna Sanjay Sharma
# Description: Data loader for the Greek letter dataset (alpha, beta, gamma).
#              Applies the required GreekTransform to convert 133x133 RGB
#              images to 28x28 greyscale inverted tensors matching MNIST format.
#
# Modified from source code provided in project specification.
# Original transform code by course instructor.

import torchvision
import torchvision.transforms as transforms
import torch.utils.data


# MNIST normalisation constants — must match training pipeline
GREEK_MEAN = (0.1307,)
GREEK_STD  = (0.3081,)

# Class index to label name mapping (ImageFolder sorts alphabetically)
# alpha=0, beta=1, gamma=2
GREEK_CLASSES = ["alpha", "beta", "gamma"]


class GreekTransform:
    """
    Transforms a 133x133 RGB Greek letter image into a 28x28 greyscale
    tensor that matches the MNIST digit format.

    Pipeline:
        1. Convert RGB to greyscale
        2. Affine scale: shrink by factor 36/128 to fit the letter into
           the center of the image
        3. Center crop to 28x28
        4. Invert intensities so the letter is white on black (MNIST style)

    Modified from source code provided in project specification.
    Original transform code by course instructor.

    Author: Krushna Sanjay Sharma
    """

    def __init__(self):
        """Initialises the GreekTransform (no parameters required)."""
        pass

    def __call__(self, x):
        """
        Applies the full transform pipeline to a single image tensor.

        Args:
            x (Tensor): Input image tensor from ToTensor() — shape (3, H, W).

        Returns:
            Tensor: Transformed image of shape (1, 28, 28), greyscale,
                    white letter on black background.
        """
        # Step 1: RGB -> greyscale, shape becomes (1, H, W)
        x = torchvision.transforms.functional.rgb_to_grayscale(x)

        # Step 2: Affine scale — shrink to fit letter in center
        # angle=0, translate=(0,0), scale=36/128, shear=0
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)

        # Step 3: Center crop to 28x28
        x = torchvision.transforms.functional.center_crop(x, (28, 28))

        # Step 4: Invert — letters are dark on white; MNIST is white on black
        return torchvision.transforms.functional.invert(x)


class GreekDataLoader:
    """
    Loads the Greek letter dataset using ImageFolder and applies
    GreekTransform + MNIST normalisation.

    Expected directory layout:
        training_set_path/
            alpha/   *.png (or .jpg)
            beta/    *.png
            gamma/   *.png

    ImageFolder assigns class indices alphabetically:
        alpha=0, beta=1, gamma=2

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        training_set_path: str,
        batch_size:        int  = 5,
        shuffle:           bool = True,
    ):
        """
        Initialises the GreekDataLoader.

        Args:
            training_set_path (str):  Path to directory containing
                                      alpha/, beta/, gamma/ subfolders.
            batch_size        (int):  Mini-batch size (default 5 per spec).
            shuffle           (bool): Shuffle training data each epoch.

        Raises:
            FileNotFoundError: If training_set_path does not exist.
        """
        self._training_set_path = training_set_path
        self._batch_size        = batch_size
        self._shuffle           = shuffle
        self._loader            = self._build_loader()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_loader(self) -> torch.utils.data.DataLoader:
        """
        Builds and returns the DataLoader for the Greek letter dataset.

        Applies ToTensor -> GreekTransform -> Normalize in sequence,
        matching the transform pipeline from the project specification.

        Returns:
            DataLoader over the Greek letter dataset.
        """
        dataset = torchvision.datasets.ImageFolder(
            root      = self._training_set_path,
            transform = transforms.Compose([
                transforms.ToTensor(),    # PIL -> tensor (3, H, W) in [0,1]
                GreekTransform(),          # RGB 133x133 -> greyscale 28x28 inverted
                transforms.Normalize(GREEK_MEAN, GREEK_STD),
            ]),
        )

        print(f"  [GreekDataLoader] Classes: {dataset.classes}")
        print(f"  [GreekDataLoader] Samples: {len(dataset)}")

        return torch.utils.data.DataLoader(
            dataset,
            batch_size = self._batch_size,
            shuffle    = self._shuffle,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def loader(self) -> torch.utils.data.DataLoader:
        """Returns the Greek letter DataLoader."""
        return self._loader

    @property
    def dataset(self) -> torchvision.datasets.ImageFolder:
        """Returns the underlying ImageFolder dataset."""
        return self._loader.dataset

    @property
    def class_names(self) -> list:
        """Returns class names in ImageFolder order (alphabetical)."""
        return self._loader.dataset.classes