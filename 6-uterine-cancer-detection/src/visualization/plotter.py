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
    
    Purpose:
        Automates graphical data execution, standardizes chart visual styling rules (colors, scaling, markers), 
        and keeps pure matplotlib backend dependencies totally isolated from core network processing logic.
        
    Steps Performed:
        1. Automatically verifies/constructs the existence of target `outputs/` directory paths prior to physical block writes.
        2. Re-maps complex n-dimensional array results into `seaborn`/`matplotlib` compliant visual axes constraints.
        3. Safely executes raw file system dumps (`plt.savefig()`) immediately flushing internal Python memory contexts thereafter.
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
        """
        Generates and saves a dual-axis side-by-side plot for Training vs Validation Loss and Accuracy.
        
        Args:
            train_losses (List[float]): Historical array of training loss floats.
            val_losses (List[float]): Historical array of validation loss floats.
            train_accs (List[float]): Historical array of training accuracy percentages [0,1].
            val_accs (List[float]): Historical array of validation accuracy percentages [0,1].
            filename (str): The desired output file path for the .png visualization.
        """
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
        """
        Generates and saves a Receiver Operating Characteristic (ROC) curve graph.
        
        Args:
            fpr (np.ndarray): False Positive Rates from sklearn's roc_curve.
            tpr (np.ndarray): True Positive Rates from sklearn's roc_curve.
            auc (float): The calculated Area Under Curve scalar.
            model_name (str): Label identity of the model to display in the legend.
            filename (str): Target output filepath.
        """
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
        """
        Generates a visually annotated heatmap style Confusion Matrix plot.
        
        Args:
            cm (np.ndarray): 2x2 confusion matrix array from sklearn.
            class_names (List[str]): Categorical labels representing the axes (e.g. ['Normal', 'Cancerous']).
            filename (str): Target output filepath.
        """
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
        Plots a 3xN grid overlaying the target patches for robust Grad-CAM Explainability analysis.
        Images are inherently formatted as RGB arrays ready for matplotlib.
        
        Args:
            images (List[np.ndarray]): Original, un-normalized histology patches.
            heatmaps (List[np.ndarray]): Raw JET colormapped generated Grad-CAM heatmaps.
            overlays (List[np.ndarray]): Seamlessly blended heatmap/photo composite images.
            labels (List[int]): The true ground-truth binary labels per image.
            predictions (List[int]): The network's raw binary prediction values.
            filename (str): Target output filepath.
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
        Traces and visualizes up to 64 generated Gabor filters within a cohesive 8x8 tiled grid sequence.
        
        Args:
            filters (List[np.ndarray]): Spatial 2D kernel float arrays forming the Gabor bank.
            filename (str): Target output filepath.
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
        """
        Charts a side-by-side array isolating the organic data-driven kernels inferred natively by ResNet
        versus the strictly mapped mathematical orientation filters dictated by the Gabor bank.
        
        Args:
            gabor_filters (List[np.ndarray]): The engineered mathematical filters.
            learned_filters (List[np.ndarray]): The raw weights extracted directly from ResNet conv1.
            filename (str): Target output filepath.
        """
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
        """
        Generates the absolute final cross-model benchmark architecture chart.
        
        Args:
            results_dict (Dict[str, float]): A dictionary mapping the String model name to its maximal accuracy scalar.
            filename (str): Target output filepath.
        """
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
