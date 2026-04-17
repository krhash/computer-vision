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
    
    Purpose:
        Serves as the centralized, fault-tolerant checkpointing engine for the training pipeline, 
        shielding the system against crashes and enabling long-term iteration continuity.
        
    Steps Performed:
        1. Validates tracking logic for storing only mathematically superior checkpoints via `best_metric`.
        2. Deep-saves `optimizer_state` and `scaler_state` into comprehensive generic `_resume.pth` artifacts.
        3. Restores complex dictionary architectures natively, catching missing tensor keys with robust `strict=False` mapping.
    """
    
    @staticmethod
    def save_checkpoint(model: nn.Module, filepath: str):
        """
        Saves only the pure model state dictionary (weights and biases).
        
        Args:
            model (nn.Module): The PyTorch neural network model to save.
            filepath (str): The absolute or relative path to the destination .pth file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model.state_dict(), filepath)

    @staticmethod
    def load_checkpoint(model: nn.Module, filepath: str, device: torch.device):
        """
        Loads weights safely onto the specified compute device.
        
        Args:
            model (nn.Module): The PyTorch neural network model architecture.
            filepath (str): The path to the stored .pth weights file.
            device (torch.device): The device (CPU/CUDA) to map the tensors to.
        """
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
        """
        Saves a comprehensive full-state dictionary to enable seamless mid-training interruption recovery.
        
        Args:
            model (nn.Module): The PyTorch model to preserve.
            optimizer (torch.optim.Optimizer): The optimizer (preserves momentum buffers).
            scaler (torch.amp.GradScaler): Mixed precision scaler (preserves scale factor).
            epoch (int): The integer index of the most recently completed epoch.
            best_metric (float): Best evaluation metric recorded thus far.
            filepath (str): Target destination .pth filepath.
            metrics_history (dict, optional): Ongoing history arrays of losses/accuracies. Defaults to None.
        """
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
        """
        Loads a full state dictionary, restoring model, optimizer, and scaler states perfectly.
        
        Args:
            model (nn.Module): The target architecture to load weights into.
            optimizer (torch.optim.Optimizer): Target optimizer to rehydrate momentum tensors into.
            scaler (torch.amp.GradScaler): Target scaler to restore.
            filepath (str): Path to the saved _resume.pth checkpoint.
            device (torch.device): Device mapping.
            
        Returns:
            tuple[int, float, dict]: (next_epoch_number, best_validation_metric, metrics_history_dict)
        """
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            print(f"  [ModelIO] Resumed from checkpoint {filepath} at Epoch {checkpoint['epoch']} (Best Metric: {checkpoint['best_metric']:.4f})")
            return checkpoint['epoch'] + 1, checkpoint['best_metric'], checkpoint.get('metrics_history', {})
        return 1, 0.0, {}
