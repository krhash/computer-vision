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
        
    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreezes the last n macro layers of ResNet.
        
        Args:
            n (int): Number of blocks to unfreeze (from layer4 down to layer1).
        """
        blocks = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for i in range(len(blocks) - n, len(blocks)):
            if i >= 0:
                for param in blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)
