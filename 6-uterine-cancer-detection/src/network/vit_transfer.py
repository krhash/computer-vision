"""
Author: Krushna Sanjay Sharma
Description: Transfer-learning enabled Vision Transformer architecture.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PretrainedViT(nn.Module):
    """
    Vision Transformer (ViT-B/16) architecture modified for fine-tuning.
    Pre-trained on ImageNet. Allows for progressive unfreezing of transformer blocks.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initializes the PretrainedViT.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        
        # Load pre-trained model
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Freeze all encoder layers initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the classification head (automatically sets requires_grad=True)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)
        
    def unfreeze_last_n_blocks(self, n: int):
        """
        Unfreezes the last n encoder blocks for fine-tuning.

        Args:
            n (int): Number of blocks from the end to unfreeze.
        """
        if n <= 0:
            return
            
        total_blocks = len(self.model.encoder.layers)
        start_idx = max(0, total_blocks - n)
        
        # Unfreeze specified blocks
        for i in range(start_idx, total_blocks):
            for param in self.model.encoder.layers[i].parameters():
                param.requires_grad = True
                
        # Unfreeze layer norm right before the head for stability
        for param in self.model.encoder.ln.parameters():
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
