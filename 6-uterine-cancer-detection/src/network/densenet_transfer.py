"""
Author: Krushna Sanjay Sharma
Description: DenseNet-121 baseline architecture.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class PretrainedDenseNet(nn.Module):
    """
    DenseNet-121 architecture modified for classification.
    Used as the primary CNN baseline from the prior project.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initializes the PretrainedDenseNet.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        
        # Load pre-trained model
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        
        # Replace the classifier block
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)
