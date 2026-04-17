"""
Author: Krushna Sanjay Sharma
Description: Orchestrator for Task 3 (Gabor Filter ResNet).
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.patch_dataset import UCECPatchDataset
from src.data.patch_transforms import PatchTransforms
from src.network.gabor_resnet import GaborResNet
from src.network.resnet_transfer import PretrainedResNet
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from src.visualization.plotter import Plotter
from src.utils.gabor_bank import GaborFilterBank
from src.utils.model_io import ModelIO

def parse_args():
    parser = argparse.ArgumentParser(description="Task 3: Gabor Filter ResNet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-dir", type=str, required=True)
    args, _ = parser.parse_known_args()
    return args

def print_parameter_counts(model: torch.nn.Module, name: str):
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{name}] Parameters -> Trainable: {trainable:,} | Frozen: {frozen:,}")

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds = UCECPatchDataset.create_splits(
        root_dir=args.patch_dir,
        train_transform=PatchTransforms.get_train_transforms(),
        val_transform=PatchTransforms.get_val_transforms(),
        test_transform=PatchTransforms.get_val_transforms()
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    baseline_resnet = PretrainedResNet(num_classes=2).to(device)
    for param in baseline_resnet.parameters():
        param.requires_grad = True

    gabor_resnet = GaborResNet(num_classes=2, num_filters=64, kernel_size=7).to(device)

    print_parameter_counts(baseline_resnet, "Baseline ResNet-34")
    print_parameter_counts(gabor_resnet, "Gabor ResNet-34")

    gabor_bank = GaborFilterBank(n_filters=64, kernel_size=7)
    gabor_arrays = gabor_bank.visualize()
    Plotter.plot_gabor_filters(gabor_arrays, "outputs/task3_gabor_bank.png")

    print("\n--- Training Gabor ResNet ---")
    opt = optim.Adam(filter(lambda p: p.requires_grad, gabor_resnet.parameters()), lr=1e-4)
    trainer = Trainer(gabor_resnet, opt, device)
    
    best_gabor_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_acc = Evaluator(gabor_resnet, device).evaluate(val_loader)["accuracy"]
        
        train_losses.append(train_metrics["loss"])
        train_accs.append(train_metrics["accuracy"])
        val_losses.append(0.0) # Evaluator returns dict without loss, appending 0 for shape matching
        val_accs.append(val_acc)
        
        best_gabor_acc = ModelIO.save_if_best(gabor_resnet, val_acc, best_gabor_acc, "models/gabor_resnet.pth")

    print("\n--- Loading Best Validation Checkpoint for Testing ---")
    ModelIO.load_checkpoint(gabor_resnet, "models/gabor_resnet.pth", device)
    
    Plotter.plot_training_curves(train_losses, val_losses, train_accs, val_accs, "outputs/task3_gabor_learning_curves.png")

    evaluator = Evaluator(gabor_resnet, device)
    gabor_metrics = evaluator.evaluate(test_loader)
    print("\n[Gabor ResNet Test Metrics]:", gabor_metrics)
    
    y_true, y_pred, y_probs = evaluator._get_predictions(test_loader)
    from sklearn.metrics import roc_curve
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        Plotter.plot_roc_curve(fpr, tpr, gabor_metrics["auroc"], "Gabor ResNet", "outputs/task3_gabor_roc_curve.png")

    print("\n--- Loading Pre-Trained Baseline ResNet for Comparison ---")
    try:
        baseline_resnet.load_state_dict(torch.load("models/resnet_transfer.pth", map_location=device), strict=False)
        print("Successfully loaded Task 4 ResNet Baseline weights.")
    except FileNotFoundError:
        print("[WARNING] models/resnet_transfer.pth not found! Using untrained Baseline ResNet.")

    evaluator_base = Evaluator(baseline_resnet, device)
    base_metrics = evaluator_base.evaluate(test_loader)
    print("\n[Baseline ResNet Test Metrics]:", base_metrics)

    # Compare Filters Visually
    learned_conv1 = baseline_resnet.model.conv1.weight.data.cpu().numpy()
    learned_arrays = []
    import numpy as np
    for i in range(learned_conv1.shape[0]):
        k = learned_conv1[i, 0, :, :]
        # Normalize dynamically per kernel chunk to 0-255 map
        k_min = k.min()
        k_max = k.max()
        if k_max - k_min > 1e-6:
            k_norm = ((k - k_min) / (k_max - k_min) * 255).astype(np.uint8)
        else:
            k_norm = np.zeros_like(k, dtype=np.uint8)
        learned_arrays.append(k_norm)

    Plotter.plot_gabor_vs_resnet(gabor_arrays, learned_arrays, "outputs/task3_conv1_compare.png")
    
    results_dict = {
        "Baseline ResNet": base_metrics["accuracy"],
        "Gabor ResNet": gabor_metrics["accuracy"]
    }
    Plotter.plot_benchmark_table(results_dict, "outputs/task3_benchmark_table.png")

if __name__ == "__main__":
    main()
