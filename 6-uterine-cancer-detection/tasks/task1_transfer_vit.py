"""
Author: Krushna Sanjay Sharma
Description: Orchestrator for Task 1 (Transfer ViT).
"""

import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.patch_dataset import UCECPatchDataset
from src.data.patch_transforms import PatchTransforms
from src.network.vit_transfer import PretrainedViT
from src.training.trainer import Trainer
from src.training.transfer_trainer import TransferTrainer
from src.evaluation.evaluator import Evaluator
from src.visualization.plotter import Plotter
from src.utils.model_io import ModelIO

def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: Transfer ViT")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-dir", type=str, required=True)
    parser.add_argument("--unfreeze-blocks", type=int, default=2)
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_ds, val_ds, test_ds = UCECPatchDataset.create_splits(
        root_dir=args.patch_dir,
        train_transform=PatchTransforms.get_train_transforms(),
        val_transform=PatchTransforms.get_val_transforms(),
        test_transform=PatchTransforms.get_val_transforms()
    )
    
    # Class weights for training
    normal_count = sum(1 for _, label in train_ds.samples if label == 0)
    cancerous_count = sum(1 for _, label in train_ds.samples if label == 1)
    total = len(train_ds.samples)
    
    weight_0 = total / (2.0 * max(normal_count, 1))
    weight_1 = total / (2.0 * max(cancerous_count, 1))
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = PretrainedViT(num_classes=2)
    transfer_mgr = TransferTrainer(model)
    
    # Phase 1: Train Head Only
    print("\n--- Phase 1: Training Head Only ---")
    transfer_mgr.freeze_all()
    transfer_mgr.unfreeze_head()
    
    optimizer = optim.AdamW(transfer_mgr.get_trainable_params(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, class_weights=class_weights)
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    phase1_epochs = args.epochs // 2
    best_val_acc = 0.0
    
    for epoch in range(1, phase1_epochs + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        train_losses.append(metrics["loss"])
        train_accs.append(metrics["accuracy"])
        
        evaluator = Evaluator(model, device)
        val_metrics = evaluator.evaluate(val_loader)
        val_losses.append(0.0) 
        val_accs.append(val_metrics["accuracy"])
        best_val_acc = ModelIO.save_if_best(model, val_metrics["accuracy"], best_val_acc, "models/vit_transfer.pth")

    # Phase 2: Progressive Unfreeze
    print(f"\n--- Phase 2: Unfreezing Last {args.unfreeze_blocks} Blocks ---")
    transfer_mgr.unfreeze_last_n(args.unfreeze_blocks)
    
    optimizer = optim.AdamW(transfer_mgr.get_trainable_params(), lr=args.lr * 0.1)
    trainer.optimizer = optimizer
    
    for epoch in range(phase1_epochs + 1, args.epochs + 1):
        metrics = trainer.train_epoch(train_loader, epoch)
        train_losses.append(metrics["loss"])
        train_accs.append(metrics["accuracy"])
        
        evaluator = Evaluator(model, device)
        val_metrics = evaluator.evaluate(val_loader)
        val_losses.append(0.0)
        val_accs.append(val_metrics["accuracy"])
        best_val_acc = ModelIO.save_if_best(model, val_metrics["accuracy"], best_val_acc, "models/vit_transfer.pth")

    # Evaluation on Test Set using best loaded model
    print("\n--- Loading Best Validation Checkpoint for Testing ---")
    ModelIO.load_checkpoint(model, "models/vit_transfer.pth", device)
    
    evaluator = Evaluator(model, device)
    test_metrics = evaluator.evaluate(test_loader)
    print("\n--- Final Test Metrics ---")
    print(test_metrics)
    
    Plotter.plot_training_curves(train_losses, val_losses, train_accs, val_accs, "outputs/task1_learning_curves.png")
    
    y_true, y_pred, y_probs = evaluator._get_predictions(test_loader)
    
    from sklearn.metrics import roc_curve
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        Plotter.plot_roc_curve(fpr, tpr, test_metrics["auroc"], "PretrainedViT", "outputs/task1_roc_curve.png")

if __name__ == "__main__":
    main()
