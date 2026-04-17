"""
Author: Krushna Sanjay Sharma
Description: Grad-CAM implementation supporting both CNNs and ViTs.
"""

import numpy as np
import torch
import torch.nn as nn

# External packages built for efficient CAM mapping
from pytorch_grad_cam import GradCAM as PytorchGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAM:
    """
    Grad-CAM wrapper for generating Class Activation Maps efficiently across different model architectures.
    
    Purpose:
        Enables human-readable explainability mappings over "black box" neural network 
        decisions without requiring code structural hooks or architectural rewrites.
        
    Steps Performed:
        1. Dynamically navigates a given network's tree to locate target intermediate activation layers.
        2. Generates an automatic 2D reshaping fallback specifically designed for Vision Transformer (ViT) sequence tensors.
        3. Ingests raw tensors, traces their gradient backwards using PyTorch's execution graph safely.
        4. Overlays structural heat signatures natively onto unmodified histology inputs using JET color maps.
    """

    def __init__(self, model: nn.Module, target_layer_name: str):
        """
        Initializes GradCAM.

        Args:
            model (nn.Module): The model to inspect.
            target_layer_name (str): The name (via getattr chain) of the target convolutional or transformer layer.
        """
        self.model = model
        self.target_layer_name = target_layer_name
        
        target_layers = self._resolve_target_layer(model, target_layer_name)
        if not target_layers:
            raise ValueError(f"Could not find target layer: {target_layer_name}")
            
        reshape_transform = None
        # Provide the fallback 2D reshape transformation if inspecting a ViT architecture
        if "vit" in str(type(model)).lower() or "vit" in target_layer_name.lower():
            reshape_transform = self._vit_reshape_transform
            
        self.cam = PytorchGradCAM(
            model=self.model, 
            target_layers=target_layers, 
            reshape_transform=reshape_transform
        )

    def _resolve_target_layer(self, model: nn.Module, layer_path: str) -> list:
        """Dynamically navigates the model tree to get the list of target layers."""
        obj = model
        for part in layer_path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                try:
                    # attempt integer indexing, e.g., features.0
                    obj = obj[int(part)]
                except (ValueError, TypeError, IndexError, KeyError):
                    return []
        # pytorch-grad-cam expects an iterable of target layers
        return [obj]

    def _vit_reshape_transform(self, tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
        """
        Reshapes the Vision Transformer attention output sequence back into a spatial 2D grid matrix.
        Assumes standard ViT-B/16 parameters dividing 224x224 into 14x14 patches.
        """
        # Tensor shape is typically [batch, seq_length, hidden_dim]
        # Discard the [CLS] token at index 0
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        
        # Bring the channels to the first spatial dimension -> [batch, C, H, W]
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Executes GradCAM mapping to yield a normalized 2D heatmap.

        Args:
            image_tensor (torch.Tensor): A batch tensor (1, C, H, W) of the input.

        Returns:
            np.ndarray: The resultant normalized heatmap spatial float matrix (H, W) in range [0, 1].
        """
        # Output shape is [batch, H, W]. Takes the first batch item if queried directly.
        grayscale_cam = self.cam(input_tensor=image_tensor)
        heatmap = grayscale_cam[0, :]
        return heatmap

    def overlay(self, image_rgb_norm: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """
        Applies a JET colormap to the generated heatmap and blends it deeply with the original image.

        Args:
            image_rgb_norm (np.ndarray): HWC numpy array of the input image in [0, 1] range.
            heatmap (np.ndarray): The 2D heatmap spanning the [0, 1] range.

        Returns:
            np.ndarray: The overlaid RGB image matrix (H, W, 3).
        """
        # Automatically overlaps color mappings using JET color scaling
        visualization = show_cam_on_image(image_rgb_norm, heatmap, use_rgb=True)
        return visualization
