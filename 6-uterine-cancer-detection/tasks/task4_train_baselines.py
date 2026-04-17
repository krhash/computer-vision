"""
Author: Krushna Sanjay Sharma
Description: Orchestrator to train baseline CNNs cleanly from scratch using Two-Phase auto-resume.
"""

import argparse
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.patch_dataset import UCECPatchDataset
from src.data.patch_transforms import PatchTransforms
from src.network.densenet_transfer import PretrainedDenseNet
from src.network.resnet_transfer import PretrainedResNet
from src.training.trainer import Trainer
from src.training.transfer_trainer import TransferTrainer
from src.evaluation.evaluator import Evaluator
from src.utils.model_io import ModelIO
from src.visualization.plotter import Plotter

def parse_args():
    parser = argparse.ArgumentParser(description="Task 4: Train CNN Baselines")
    parser.add_argument("--model", type=str, choices=["densenet", "resnet"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--patch-dir", type=str, required=True)
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Setup
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model definition
    if args.model == "densenet":
        model = PretrainedDenseNet(num_classes=2)
    else:
        model = PretrainedResNet(num_classes=2)
        
    transfer_mgr = TransferTrainer(model)
    
    resume_path = f"models/{args.model}_resume.pth"
    final_path = f"models/{args.model}_transfer.pth"
    
    # Phase 1: Train Head Only
    transfer_mgr.freeze_all()
    transfer_mgr.unfreeze_head()
    
    optimizer = optim.AdamW(transfer_mgr.get_trainable_params(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, class_weights=class_weights)
    
    phase1_epochs = args.epochs // 2
    
    # Check if we are resuming directly into the middle of Phase 2 logic (Epoch 6 or later saved)
    # If so, we Must pre-expand the optimizer groups so the loading mechanism doesn't crash mismatch!
    if os.path.exists(resume_path):
        chkpt = torch.load(resume_path, map_location=device, weights_only=False)
        if chkpt['epoch'] > phase1_epochs:
            transfer_mgr.unfreeze_last_n(args.unfreeze_blocks)
            trainer.optimizer = optim.AdamW(transfer_mgr.get_trainable_params(), lr=args.lr * 0.1)

    # Attempt Auto Resume
    start_epoch, best_val_acc, metrics_history = ModelIO.load_resume_checkpoint(model, trainer.optimizer, trainer.scaler, resume_path, device)
    
    train_losses = metrics_history.get("train_losses", [])
    val_losses = metrics_history.get("val_losses", [])
    train_accs = metrics_history.get("train_accs", [])
    val_accs = metrics_history.get("val_accs", [])
    
    # Run Phase 1 if not completed
    if start_epoch <= phase1_epochs:
        print("\n--- Phase 1: Training Head Only ---")
        for epoch in range(start_epoch, phase1_epochs + 1):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            evaluator = Evaluator(model, device)
            val_metrics = evaluator.evaluate(val_loader)
            
            train_losses.append(train_metrics["loss"])
            train_accs.append(train_metrics["accuracy"])
            val_losses.append(0.0)
            val_accs.append(val_metrics["accuracy"])
            
            metrics_history = {"train_losses": train_losses, "val_losses": val_losses, "train_accs": train_accs, "val_accs": val_accs}
            
            # Save Resume Checkpoint ALWAYS, and Save Final/Best .pth conditionally
            ModelIO.save_resume_checkpoint(model, trainer.optimizer, trainer.scaler, epoch, max(best_val_acc, val_metrics["accuracy"]), resume_path, metrics_history=metrics_history)
            best_val_acc = ModelIO.save_if_best(model, val_metrics["accuracy"], best_val_acc, final_path)
    
    # Move to Phase 2 logically over epochs if not resumed past it
    if start_epoch <= args.epochs:
        print(f"\n--- Phase 2: Unfreezing Last {args.unfreeze_blocks} Block(s) ---")
        transfer_mgr.unfreeze_last_n(args.unfreeze_blocks)
        
        # We only reinitialize optimizer if we literally just crossed the boundary from Phase 1, 
        # but if we resumed in phase 2, the optimizer is loaded already!
        if start_epoch <= phase1_epochs + 1:
            optimizer = optim.AdamW(transfer_mgr.get_trainable_params(), lr=args.lr * 0.1)
            trainer.optimizer = optimizer
        else:
            # We resumed strictly inside phase 2, so the parameters generator must be updated 
            # in optimizer, but PyTorch AdamW holds parameter references internally. 
            pass # Standard resume already loaded the Phase 2 optimizer state!
            
        phase2_start = max(start_epoch, phase1_epochs + 1)
        for epoch in range(phase2_start, args.epochs + 1):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            evaluator = Evaluator(model, device)
            val_metrics = evaluator.evaluate(val_loader)
            
            train_losses.append(train_metrics["loss"])
            train_accs.append(train_metrics["accuracy"])
            val_losses.append(0.0)
            val_accs.append(val_metrics["accuracy"])
            
            metrics_history = {"train_losses": train_losses, "val_losses": val_losses, "train_accs": train_accs, "val_accs": val_accs}
            
            ModelIO.save_resume_checkpoint(model, trainer.optimizer, trainer.scaler, epoch, max(best_val_acc, val_metrics["accuracy"]), resume_path, metrics_history=metrics_history)
            best_val_acc = ModelIO.save_if_best(model, val_metrics["accuracy"], best_val_acc, final_path)

    # Final Evaluation
    print(f"\n--- Loading Best Checkpoint ({final_path}) for Testing ---")
    ModelIO.load_checkpoint(model, final_path, device)
    
    evaluator = Evaluator(model, device)
    test_metrics = evaluator.evaluate(test_loader)
    print("\n--- Final Test Metrics ---")
    print(test_metrics)
    
    Plotter.plot_training_curves(train_losses, val_losses, train_accs, val_accs, f"outputs/task4_{args.model}_learning_curves.png")

    y_true, y_pred, y_probs = evaluator._get_predictions(test_loader)
    
    from sklearn.metrics import roc_curve
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        Plotter.plot_roc_curve(fpr, tpr, test_metrics["auroc"], args.model.capitalize(), f"outputs/task4_{args.model}_roc_curve.png")

if __name__ == "__main__":
    main()
