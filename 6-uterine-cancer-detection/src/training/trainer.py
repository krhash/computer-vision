"""
Author: Krushna Sanjay Sharma
Description: Core training loop and epoch logic for deep learning models.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict

class Trainer:
    """
    Handles the standard training epoch loop, optimized with Automatic Mixed Precision (AMP).
    
    Purpose:
        Encapsulates the raw PyTorch backpropagation mechanisms, offloading gradient 
        scaling and hardware mapping from the task orchestrators.
        
    Steps Performed:
        1. Initializes the `CrossEntropyLoss` function (with optional dataset imbalance weights).
        2. Triggers the AMP `GradScaler` for lightning-fast memory-efficient `float16` forward passes.
        3. Iterates over the PyTorch `DataLoader`, casting tensors natively to GPU parameters.
        4. Executes `scaler.step(optimizer)` and securely aggregates tracking logic for historical metrics.
    """

    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        log_interval: int = 50
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch neural network model.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            device (torch.device): Device to compute on (CPU/GPU).
            class_weights (Optional[torch.Tensor]): Weights for classes to handle imbalance.
            log_interval (int): How often (in batches) to print progress.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        # Weighted CrossEntropyLoss helps combat highly imbalanced datasets
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.log_interval = log_interval
        
        # Scaler for mixed precision
        is_cuda = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda' if is_cuda else 'cpu')
        
        print(f"[Trainer] Initialized on device: {self.device.type.upper()} | AMP enabled: {is_cuda}", flush=True)

    def train_epoch(self, dataloader: DataLoader, epoch_num: int) -> Dict[str, float]:
        """
        Runs one complete pass over the training dataset.

        Args:
            dataloader (DataLoader): DataLoader providing the training batches.
            epoch_num (int): Current epoch number (for logging).

        Returns:
            Dict[str, float]: dictionary containing "loss", "accuracy", "time"
        """
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        start_time = time.time()
        num_batches = len(dataloader)
        
        print(f"[Trainer] Epoch {epoch_num} starting ({num_batches} batches to process)...", flush=True)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Automatic Mixed Precision for faster, memory-efficient training
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)
            
            # Compute batch accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if (batch_idx + 1) % self.log_interval == 0:
                cur_loss = running_loss / total_samples
                cur_acc = correct_preds / total_samples
                print(f"  --> Epoch [{epoch_num}] Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {cur_loss:.4f} Acc: {cur_acc:.4f}", flush=True)

        end_time = time.time()
        
        epoch_loss = running_loss / max(total_samples, 1)
        epoch_acc = correct_preds / max(total_samples, 1)
        
        print(f"--- Epoch {epoch_num} Summary: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Time={end_time - start_time:.2f}s ---", flush=True)
        
        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "time": end_time - start_time
        }
