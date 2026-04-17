"""
Author: Krushna Sanjay Sharma
Description: Model checkpoint saving and loading utilities.
"""

import torch
import torch.nn as nn
import os

class ModelIO:
    """
    Handles saving and loading of model weights gracefully.
    """
    
    @staticmethod
    def save_checkpoint(model: nn.Module, filepath: str):
        """Saves only the model state dictionary."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model.state_dict(), filepath)

    @staticmethod
    def load_checkpoint(model: nn.Module, filepath: str, device: torch.device):
        """Loads weights onto the specified device."""
        model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        
    @staticmethod
    def save_if_best(model: nn.Module, current_metric: float, best_metric: float, filepath: str) -> float:
        """
        Saves the model if the current metric is better than the historical best.
        
        Args:
            model (nn.Module): The PyTorch model.
            current_metric (float): The metric from the current epoch (e.g. valid accuracy).
            best_metric (float): The highest metric recorded so far.
            filepath (str): Checkpoint destination.
            
        Returns:
            float: The updated best metric.
        """
        if current_metric > best_metric:
            ModelIO.save_checkpoint(model, filepath)
            print(f"  [ModelIO] Metric improved ({best_metric:.4f} --> {current_metric:.4f}). Saved checkpoint to {filepath}")
            return current_metric
        return best_metric

    @staticmethod
    def save_resume_checkpoint(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scaler: torch.amp.GradScaler, 
        epoch: int, 
        best_metric: float, 
        filepath: str,
        metrics_history: dict = None
    ):
        """Saves a full state dictionary to resume training seamlessly."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_metric': best_metric,
            'metrics_history': metrics_history or {}
        }
        torch.save(checkpoint, filepath)

    @staticmethod
    def load_resume_checkpoint(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scaler: torch.amp.GradScaler, 
        filepath: str, 
        device: torch.device
    ) -> tuple[int, float, dict]:
        """Loads a full state dictionary and restores optimizer/scaler state. Returns (start_epoch, best_metric, metrics_history)."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            print(f"  [ModelIO] Resumed from checkpoint {filepath} at Epoch {checkpoint['epoch']} (Best Metric: {checkpoint['best_metric']:.4f})")
            return checkpoint['epoch'] + 1, checkpoint['best_metric'], checkpoint.get('metrics_history', {})
        return 1, 0.0, {}
