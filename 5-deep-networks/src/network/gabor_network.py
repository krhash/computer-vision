# src/network/gabor_network.py
# Project 5: Recognition using Deep Networks — Extension 2
# Author: Krushna Sanjay Sharma
# Description: GaborFilterBank generates a bank of Gabor filters and
#              injects them into the first conv layer of DigitNetwork,
#              freezing conv1 so only conv2 and FC layers are trained.
#
# Gabor filters are sinusoidal gratings modulated by a Gaussian envelope.
# They respond selectively to edges at specific orientations and frequencies,
# similar to simple cells in the mammalian visual cortex. ResNet's learned
# filters (Extension 1) show visually similar patterns — Gabor filters are
# the hand-crafted equivalent.
#
# Architecture after substitution:
#   conv1 (Gabor, FROZEN) -> pool/relu -> conv2 (learned) -> dropout
#   -> pool/relu -> fc1 (learned) -> fc2 (learned)

import numpy as np
import torch
import torch.nn as nn

from src.network.digit_network import DigitNetwork


class GaborFilterBank:
    """
    Generates a bank of 2D Gabor filters for use as fixed conv1 weights.

    A Gabor filter is defined by:
        g(x,y) = exp(-(x'^2 + gamma^2 * y'^2) / (2*sigma^2))
                 * cos(2*pi*x'/lambda + psi)

    where x' and y' are coordinates rotated by angle theta.

    Parameters varied to produce a diverse filter bank:
        theta   — orientation of the filter (0 to pi)
        lambda_ — wavelength of the sinusoidal carrier
        psi     — phase offset (0 or pi/2 for even/odd filters)

    Author: Krushna Sanjay Sharma
    """

    def __init__(
        self,
        kernel_size: int   = 5,
        num_filters: int   = 10,
        sigma:       float = 1.5,
        gamma:       float = 0.5,
    ):
        """
        Initialises the filter bank parameters.

        Args:
            kernel_size (int):   Size of each filter (must match conv1 kernel).
                                 DigitNetwork conv1 uses 5x5.
            num_filters (int):   Number of filters to generate (must match
                                 conv1 out_channels = 10).
            sigma       (float): Standard deviation of the Gaussian envelope.
                                 Controls filter size/bandwidth.
            gamma       (float): Spatial aspect ratio (0.5 = elongated filter).
        """
        self._kernel_size = kernel_size
        self._num_filters = num_filters
        self._sigma       = sigma
        self._gamma       = gamma

    def generate(self) -> np.ndarray:
        """
        Generates num_filters Gabor filters covering orientations 0 to pi
        and two phase offsets (even/odd symmetric pairs).

        Filter design:
            - 5 orientations: 0, 36, 72, 108, 144 degrees
            - 2 phases per orientation: 0 (even) and pi/2 (odd)
            - lambda_ varies with orientation for frequency diversity

        Returns:
            ndarray of shape (num_filters, kernel_size, kernel_size).
            Ready to be reshaped to (10, 1, 5, 5) for conv1.
        """
        filters = []

        # 5 orientations × 2 phases = 10 filters
        n_orientations = self._num_filters // 2
        orientations   = [i * np.pi / n_orientations
                          for i in range(n_orientations)]
        phases         = [0, np.pi / 2]

        for theta in orientations:
            for psi in phases:
                # Lambda varies with orientation for frequency diversity
                lambda_ = self._kernel_size / (1.5 + theta / np.pi)
                f = self._make_gabor(theta, lambda_, psi)
                filters.append(f)

        return np.array(filters[:self._num_filters])   # (10, 5, 5)

    def _make_gabor(
        self,
        theta:   float,
        lambda_: float,
        psi:     float,
    ) -> np.ndarray:
        """
        Generates a single 2D Gabor filter kernel.

        Args:
            theta   (float): Filter orientation in radians.
            lambda_ (float): Wavelength of the sinusoidal carrier.
            psi     (float): Phase offset in radians.

        Returns:
            ndarray of shape (kernel_size, kernel_size).
        """
        half   = self._kernel_size // 2
        coords = np.arange(-half, half + 1)
        x, y   = np.meshgrid(coords, coords)

        # Rotate coordinates by theta
        x_prime =  x * np.cos(theta) + y * np.sin(theta)
        y_prime = -x * np.sin(theta) + y * np.cos(theta)

        # Gaussian envelope
        gaussian = np.exp(
            -(x_prime**2 + self._gamma**2 * y_prime**2)
            / (2 * self._sigma**2)
        )

        # Sinusoidal carrier
        carrier = np.cos(2 * np.pi * x_prime / lambda_ + psi)

        gabor = gaussian * carrier

        # Normalise to zero mean and unit std for stable conv output
        gabor -= gabor.mean()
        std = gabor.std()
        if std > 1e-8:
            gabor /= std

        return gabor.astype(np.float32)


class GaborDigitNetwork(DigitNetwork):
    """
    DigitNetwork with conv1 replaced by a fixed Gabor filter bank.

    Inherits the full DigitNetwork architecture. On construction:
        1. Generates 10 Gabor filters (5x5, matching conv1)
        2. Injects them as conv1 weights
        3. Freezes conv1 so it is never updated during training

    Only conv2, fc1, and fc2 are trainable — identical pattern to
    the transfer learning approach in Task 3.

    Comparison hypothesis:
        Hand-crafted Gabor filters capture orientation/frequency
        features similar to what conv1 learns. If true, the network
        should achieve close to the same accuracy as the fully trained
        DigitNetwork with fewer learnable parameters.

    Author: Krushna Sanjay Sharma
    """

    def __init__(self, sigma: float = 1.5, gamma: float = 0.5):
        """
        Initialises the network, injects Gabor filters, and freezes conv1.

        Args:
            sigma (float): Gaussian envelope std for Gabor filters.
            gamma (float): Spatial aspect ratio for Gabor filters.
        """
        super(GaborDigitNetwork, self).__init__()

        self._sigma = sigma
        self._gamma = gamma

        self._inject_gabor_filters()
        self._freeze_conv1()

    def _inject_gabor_filters(self) -> None:
        """
        Generates 10 Gabor filters and sets them as conv1 weights.

        Reshapes filters from (10, 5, 5) to (10, 1, 5, 5) to match
        conv1's expected weight shape (out_channels, in_channels, kH, kW).
        """
        bank    = GaborFilterBank(
            kernel_size = 5,
            num_filters = 10,
            sigma       = self._sigma,
            gamma       = self._gamma,
        )
        filters = bank.generate()               # (10, 5, 5)
        filters = filters[:, np.newaxis, :, :]  # (10, 1, 5, 5)

        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.from_numpy(filters)
            )

        print(f"  [GaborDigitNetwork] Injected 10 Gabor filters into conv1")
        print(f"  sigma={self._sigma}, gamma={self._gamma}")

    def _freeze_conv1(self) -> None:
        """
        Freezes conv1 weights so they are not updated during training.

        Sets requires_grad=False on conv1 parameters only.
        conv2, fc1, fc2 remain trainable.
        """
        for param in self.conv1.parameters():
            param.requires_grad = False

        print(f"  [GaborDigitNetwork] conv1 frozen (Gabor filters fixed)")

    def get_gabor_filters(self) -> np.ndarray:
        """
        Returns the current conv1 weights as numpy arrays.

        Returns:
            ndarray of shape (10, 5, 5) — the 10 Gabor filters.
        """
        with torch.no_grad():
            return self.conv1.weight.squeeze(1).cpu().numpy()

    def get_trainable_params(self) -> list:
        """
        Returns only the trainable (non-frozen) parameters.

        Used to pass the correct parameters to the optimiser so
        the frozen conv1 is not included in the update step.

        Returns:
            List of trainable nn.Parameter objects.
        """
        return [p for p in self.parameters() if p.requires_grad]