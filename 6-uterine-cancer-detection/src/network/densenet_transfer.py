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
        
    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreezes the last n denseblocks of DenseNet.
        
        Args:
            n (int): Number of denseblocks to unfreeze.
        """
        blocks = [
            self.model.features.denseblock1,
            self.model.features.denseblock2,
            self.model.features.denseblock3,
            self.model.features.denseblock4
        ]
        
        # Always unfreeze the final norm layer to be safe
        for param in self.model.features.norm5.parameters():
            param.requires_grad = True
            
        for i in range(len(blocks) - n, len(blocks)):
            if i >= 0:
                for param in blocks[i].parameters():
                    param.requires_grad = True
        
        transitions = [
            self.model.features.transition1,
            self.model.features.transition2,
            self.model.features.transition3
        ]
        for i in range(len(transitions) - n + 1, len(transitions)):
            if i >= 0:
                for param in transitions[i].parameters():
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
