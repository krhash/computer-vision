"""
Author: Krushna Sanjay Sharma
Description: Gabor filter bank generator for injecting into CNN layers.
"""

import numpy as np
import cv2
from typing import List

class GaborFilterBank:
    """
    Generates a bank of Gabor filters structured to substitute early convolutional layers.
    
    Purpose:
        Bridges biological vision processing mathematical models with deep learning tensor convolutions,
        providing strict spatial-frequency wavelet textures bypassing standard CNN randomization.
        
    Steps Performed:
        1. Maps deterministic algorithmic parameter grids (sigma, lambda, theta, gamma).
        2. Actuates OpenCV (`cv2.getGaborKernel`) to explicitly render 2D spatial float tensors.
        3. Stacks 2D kernels homogeneously across RGB channel boundaries for 3D tensor volume creation.
        4. Provides dynamic visual normalization/rendering functions strictly for Plotter ingestion.
    """
    
    def __init__(self, n_filters: int = 64, kernel_size: int = 7, in_channels: int = 3):
        """
        Initializes the GaborFilterBank parameters.

        Args:
            n_filters (int): Number of total output filters exactly matching the target layer out_channels.
            kernel_size (int): Size of the square Gabor kernel.
            in_channels (int): Number of input channels matching target layer in_channels (e.g., 3 for RGB).
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.filters_generated = None
        
    def generate(self) -> np.ndarray:
        """
        Generates the Gabor filters constructed across multiple orientations, frequencies, 
        and scales. Spans across `in_channels` by duplicating the 2D filter.

        Returns:
            np.ndarray: Array shaped as (n_filters, in_channels, kH, kW).
        """
        filters = []
        
        # We vary orientation (theta) and frequency (lambda) to get a rich bank.
        thetas = np.linspace(0, np.pi, self.n_filters, endpoint=False)
        base_sigma = 4.0
        base_lam = 10.0
        gamma = 0.5
        psi = 0.0
        
        for i in range(self.n_filters):
            # Vary lambda and sigma slightly to give multiple scales
            scale_idx = i % 4
            current_sigma = base_sigma + (scale_idx * 0.5)
            current_lam = base_lam - (scale_idx * 1.5)
            theta = thetas[i]
            
            # cv2.getGaborKernel takes: ksize, sigma, theta, lambda, gamma, psi, ktype
            kernel_2d = cv2.getGaborKernel(
                (self.kernel_size, self.kernel_size), 
                current_sigma, 
                theta, 
                current_lam, 
                gamma, 
                psi, 
                ktype=cv2.CV_32F
            )
            
            # Duplicate the 2D kernel across `in_channels` to respond equally to all color channels.
            multi_channel_kernel = np.stack([kernel_2d] * self.in_channels, axis=0)
            filters.append(multi_channel_kernel)
            
        self.filters_generated = np.stack(filters, axis=0)
        return self.filters_generated

    def visualize(self) -> List[np.ndarray]:
        """
        Returns a list of 2D 8-bit arrays useful for visual plot rendering.
        Extracts the first channel of each 3D filter to visualize.

        Returns:
            List[np.ndarray]: Normalized 2D arrays ready for matplotlib plotting.
        """
        if self.filters_generated is None:
            self.generate()
            
        vis_kernels = []
        for i in range(self.n_filters):
            # We take the 0-th channel of the i-th filter (it's duplicated across channels anyway)
            k = self.filters_generated[i, 0, :, :]
            k_min, k_max = k.min(), k.max()
            if k_max - k_min > 1e-6:
                # Normalize to 0-255 range for image plotting directly
                k_normalized = ((k - k_min) / (k_max - k_min) * 255).astype(np.uint8)
            else:
                k_normalized = np.zeros_like(k, dtype=np.uint8)
                
            vis_kernels.append(k_normalized)
            
        return vis_kernels
