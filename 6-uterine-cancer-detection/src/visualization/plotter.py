"""
Author: Krushna Sanjay Sharma
Description: High-level visualization tools using matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Dict

class Plotter:
    """
    Stateless utility class encapsulating all plotting constraints for standard deliverables.
    Saves outputs directly to the parameterized filename.
    """

    @staticmethod
    def _ensure_dir(filename: str):
        """Helper to ensure the target output directory exists."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    @staticmethod
    def plot_training_curves(
        train_losses: List[float], val_losses: List[float], 
        train_accs: List[float], val_accs: List[float], 
        filename: str
    ):
        Plotter._ensure_dir(filename)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        if val_losses:
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        if val_accs:
            ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, model_name: str, filename: str):
        Plotter._ensure_dir(filename)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], filename: str):
        Plotter._ensure_dir(filename)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_gradcam_grid(
        images: List[np.ndarray], 
        heatmaps: List[np.ndarray], 
        overlays: List[np.ndarray], 
        labels: List[int], 
        predictions: List[int], 
        filename: str
    ):
        """
        Plots a 3xN grid overlaying the target patches.
        Images are expected to be properly un-normalized or processed RGB ready for matplotlib.
        """
        Plotter._ensure_dir(filename)
        n = len(images)
        if n == 0:
            return
            
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        # Handle 1D array of axes if n=1
        if n == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(n):
            # Column 1: Original Image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f"True: {labels[i]} | Pred: {predictions[i]}")
            axes[i, 0].axis('off')
            
            # Column 2: Raw Heatmap
            axes[i, 1].imshow(heatmaps[i], cmap='jet')
            axes[i, 1].set_title("Grad-CAM Heatmap")
            axes[i, 1].axis('off')
            
            # Column 3: Overlay
            axes[i, 2].imshow(overlays[i])
            axes[i, 2].set_title("Overlay")
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_gabor_filters(filters: List[np.ndarray], filename: str):
        """
        Plots up to 64 Gabor filters in an 8x8 grid.
        filters: List of 2D kernel arrays.
        """
        Plotter._ensure_dir(filename)
        n = min(len(filters), 64)
        grid_size = int(np.ceil(np.sqrt(n)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = [ax for row in axes for ax in row] if n > 1 else axes
        
        for i in range(len(axes)):
            if i < n:
                axes[i].imshow(filters[i], cmap='gray')
                axes[i].axis('off')
            else:
                axes[i].set_visible(False)
                
        plt.suptitle("Generated Gabor Filter Bank")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_gabor_vs_resnet(gabor_filters: List[np.ndarray], learned_filters: List[np.ndarray], filename: str):
        Plotter._ensure_dir(filename)
        n = min(len(gabor_filters), len(learned_filters), 8) # Plot top 8 for comparison
        
        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        if n == 1:
            axes = np.expand_dims(axes, axis=1)
            
        for i in range(n):
            axes[0, i].imshow(gabor_filters[i], cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title("Gabor")
                
            axes[1, i].imshow(learned_filters[i], cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title("Learned (ResNet)")
                
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    @staticmethod
    def plot_benchmark_table(results_dict: Dict[str, float], filename: str):
        Plotter._ensure_dir(filename)
        models = list(results_dict.keys())
        scores = list(results_dict.values())
        
        plt.figure(figsize=(8, 5))
        
        # We pass hue=models and legend=False strictly to bypass recent seaborn future warnings.
        sns.barplot(x=models, y=scores, hue=models, legend=False, palette='viridis')
        plt.ylim(0, 1.05)
        plt.ylabel('Score')
        plt.title('Model Benchmark Comparison')
        
        for i, score in enumerate(scores):
            plt.text(i, score + 0.02, f"{score:.3f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
