"""
Author: Krushna Sanjay Sharma
Description: Model evaluation mechanisms, computing robust metrics and confusion matrices.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from typing import Dict, Tuple, List

class Evaluator:
    """
    Consolidates evaluation logic for standard deep learning classification.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initializes the Evaluator.

        Args:
            model (nn.Module): The PyTorch model to evaluate.
            device (torch.device): Device to compute on (CPU/GPU).
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def _get_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal generator for extracting logits and labels.

        Args:
            loader (DataLoader): DataLoader providing the evaluation batches.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: True labels, predicted labels, and probabilities.
        """
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                # Store class 1 probabilities for AUROC
                all_probs.extend(probs[:, 1].cpu().numpy())

        return np.array(all_targets), np.array(all_preds), np.array(all_probs)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Computes various classification metrics over the dataset.

        Args:
            loader (DataLoader): DataLoader for evaluation data.

        Returns:
            Dict[str, float]: accuracy, auroc, f1, precision, recall
        """
        y_true, y_pred, y_probs = self._get_predictions(loader)
        
        # Unique check for AUROC in case batch/class is pure (e.g. testing logic only)
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_probs)
        else:
            auroc = float('nan')

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auroc": auroc,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0)
        }

    def compute_confusion_matrix(self, loader: DataLoader) -> np.ndarray:
        """
        Computes the confusion matrix.

        Args:
            loader (DataLoader): DataLoader to evaluate.

        Returns:
            np.ndarray: Confusion matrix of shape (n_classes, n_classes).
        """
        y_true, y_pred, _ = self._get_predictions(loader)
        return confusion_matrix(y_true, y_pred)

    def predict_samples(self, images: torch.Tensor, labels: torch.Tensor, n: int) -> List[Dict]:
        """
        Runs prediction on a set number of samples for granular inspection.

        Args:
            images (torch.Tensor): A batch tensor of images.
            labels (torch.Tensor): The corresponding ground truth labels.
            n (int): Max number of samples to inspect.

        Returns:
            List[Dict]: List containing dictionaries of the image tensor, ground truth, and prediction.
        """
        results = []
        n = min(n, images.size(0))
        
        with torch.no_grad():
            img_subset = images[:n].to(self.device)
            lbl_subset = labels[:n]
            outputs = self.model(img_subset)
            _, preds = torch.max(outputs, 1)
            
            for i in range(n):
                results.append({
                    "image": images[i].clone().cpu(),
                    "true_label": lbl_subset[i].item(),
                    "predicted_label": preds[i].item()
                })
                
        return results
