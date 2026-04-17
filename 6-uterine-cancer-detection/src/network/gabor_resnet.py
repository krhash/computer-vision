"""
Author: Krushna Sanjay Sharma
Description: ResNet-34 architecture with a structurally fixed Gabor Filter Bank in conv1.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from src.utils.gabor_bank import GaborFilterBank  

class GaborResNet(nn.Module):
    """
    ResNet-34 architecture injected with a hand-crafted Gabor filter bank in conv1.
    
    Purpose:
        Tests the core research hypothesis that biological/mathematical spatial orientation textures natively outperform 
        randomly initialized weights regarding early-stage dense medical histology ingestion.
        
    Steps Performed:
        1. Generates exactly 64 deterministic Gabor wavelet filters at deliberately staggering frequencies and polar orientations.
        2. Statically overwrites and binds `conv1.weight` of the standard ResNet-34, dropping the classic generic kernels.
        3. Locks `conv1` parameters seamlessly (`requires_grad=False`) converting the tensor block into a permanently fixed mathematical map.
        4. Sustains the subsequent residual chains explicitly to compute downstream projections purely off Gabor-processed feature volumes.
    """
    
    def __init__(self, num_classes: int = 2, num_filters: int = 64, kernel_size: int = 7):
        """
        Initializes the GaborResNet.

        Args:
            num_classes (int): Number of output classes.
            num_filters (int): Number of Gabor filters to generate (must match conv1 out_channels).
            kernel_size (int): Size of the Gabor filters (must match conv1 kernel_size).
        """
        super().__init__()
        
        # Load pre-trained model for the rest of the network weights
        self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Generate Gabor filter bank (returns shape: [n_filters, in_channels, kH, kW])
        gabor_bank = GaborFilterBank(n_filters=num_filters, kernel_size=kernel_size, in_channels=3)
        filters = gabor_bank.generate()
        
        # Inject Gabor filters into conv1
        with torch.no_grad():
            self.model.conv1.weight = nn.Parameter(torch.from_numpy(filters).float())
            
        # Freeze conv1
        for param in self.model.conv1.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)
