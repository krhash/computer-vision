"""
Author: Krushna Sanjay Sharma
Description: Transfer-learning enabled ResNet-34 architecture.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class PretrainedResNet(nn.Module):
    """
    ResNet-34 architecture modified for fine-tuning.
    Pre-trained on ImageNet. Freezes all layers except the final classification head.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initializes the PretrainedResNet.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        
        # Load pre-trained model
        self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer (automatically sets requires_grad=True)
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
