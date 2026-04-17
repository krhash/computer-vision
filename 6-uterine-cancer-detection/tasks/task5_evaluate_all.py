"""
Author: Krushna Sanjay Sharma
Description: Final Benchmarking task evaluating all models against the test set.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from src.data.patch_dataset import UCECPatchDataset
from src.data.patch_transforms import PatchTransforms
from src.network.densenet_transfer import PretrainedDenseNet
from src.network.resnet_transfer import PretrainedResNet
from src.network.gabor_resnet import GaborResNet
from src.network.vit_transfer import PretrainedViT
from src.evaluation.evaluator import Evaluator
from src.visualization.plotter import Plotter

def parse_args():
    parser = argparse.ArgumentParser(description="Task 5: Final Benchmark Evaluation")
    parser.add_argument("--patch-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Task 5: Master Benchmark Evaluation ---")
    print(f"Using device: {device}")
    
    # Generate identical reproducible Test Set split!
    _, _, test_ds = UCECPatchDataset.create_splits(
        root_dir=args.patch_dir,
        test_ratio=0.05, val_ratio=0.05, train_ratio=0.9,
        test_transform=PatchTransforms.get_val_transforms()
    )
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    models_to_test = [
        ("ViT", PretrainedViT(num_classes=2), "models/vit_transfer.pth"),
        ("DenseNet", PretrainedDenseNet(num_classes=2), "models/densenet_transfer.pth"),
        ("ResNet", PretrainedResNet(num_classes=2), "models/resnet_transfer.pth"),
        ("Gabor ResNet", GaborResNet(num_classes=2, num_filters=64, kernel_size=7), "models/gabor_resnet.pth")
    ]
    
    master_results = {}

    for name, model, path in models_to_test:
        print(f"\n--- Benchmarking {name} ---")
        try:
            missing, unexpected = model.load_state_dict(torch.load(path, map_location=device), strict=False)
        except Exception as e:
            print(f"[WARNING] Failed to load {path}: {e}. Skipping {name}.")
            continue
            
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(test_loader)
        print(f"[{name} Metrics]: {metrics}")
        
        master_results[name] = metrics["accuracy"]
        
        cm = evaluator.compute_confusion_matrix(test_loader)
        print(f"Confusion Matrix:\n{cm}")
        
        name_clean = name.replace(" ", "_").lower()
        Plotter.plot_confusion_matrix(cm, ["Normal", "Cancerous"], f"outputs/task5_{name_clean}_cm.png")

    # Generate the Ultimate Master Benchmark Plot if we successfully tested multiple models!
    if len(master_results) > 1:
        Plotter.plot_benchmark_table(master_results, "outputs/task5_master_benchmark.png")
        print(f"\n[Task 5 Success] Generated Master Benchmark Plot at outputs/task5_master_benchmark.png!")

if __name__ == '__main__':
    main()
