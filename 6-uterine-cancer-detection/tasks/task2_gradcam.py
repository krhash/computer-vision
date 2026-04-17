"""
Author: Krushna Sanjay Sharma
Description: Orchestrator for Task 2 (Grad-CAM Explainability).
"""

import argparse
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.patch_dataset import UCECPatchDataset
from src.data.patch_transforms import PatchTransforms
from src.network.densenet_transfer import PretrainedDenseNet
from src.network.vit_transfer import PretrainedViT
from src.network.resnet_transfer import PretrainedResNet
from src.evaluation.gradcam import GradCAM
from src.visualization.plotter import Plotter

def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Grad-CAM Explainability")
    parser.add_argument("--model", type=str, choices=["densenet", "vit", "resnet", "gabor_resnet"], required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--patch-dir", type=str, required=True)
    args, _ = parser.parse_known_args()
    return args

def inv_normalize(tensor: torch.Tensor) -> np.ndarray:
    """Helper to un-normalize ImageNet tensors back to [0,1] RGB numpy arrays."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        _, _, test_ds = UCECPatchDataset.create_splits(
            root_dir=args.patch_dir,
            val_transform=PatchTransforms.get_val_transforms(),
            test_transform=PatchTransforms.get_val_transforms()
        )
    except ValueError:
        print(f"Error: Could not load data from {args.patch_dir}")
        sys.exit(1)
        
    loader = DataLoader(test_ds, batch_size=args.n_samples * 2, shuffle=True)

    if args.model == "densenet":
        model = PretrainedDenseNet(num_classes=2)
        target_layer = "model.features.denseblock4.denselayer16.conv2" # Common final conv in DenseNet121
    elif args.model == "resnet":
        model = PretrainedResNet(num_classes=2)
        target_layer = "model.layer4.2.conv2" # Final conv in ResNet34
    elif args.model == "gabor_resnet":
        from src.network.gabor_resnet import GaborResNet
        model = GaborResNet(num_classes=2, num_filters=64, kernel_size=7)
        target_layer = "model.layer4.2.conv2" # Same underlying backbone as ResNet34
    else:
        model = PretrainedViT(num_classes=2)
        target_layer = "model.encoder.layers.11.ln_1" # Last block layernorm in ViT-B/16
        
    weight_path = f"models/{args.model}_transfer.pth"
    if args.model == "gabor_resnet":
        weight_path = "models/gabor_resnet.pth"
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        print(f"Loaded weights from {weight_path}")
    except FileNotFoundError:
        print(f"[WARNING] Weights not found at {weight_path}. Running with pretrained base.")
        
    model = model.to(device)
    model.eval()

    cam_analyzer = GradCAM(model, target_layer)

    normal_images, normal_labels, normal_preds = [], [], []
    cancer_images, cancer_labels, cancer_preds = [], [], []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        for i in range(inputs.size(0)):
            lbl = targets[i].item()
            if lbl == 0 and len(normal_images) < args.n_samples:
                normal_images.append(inputs[i])
                normal_labels.append(lbl)
                normal_preds.append(preds[i].item())
            elif lbl == 1 and len(cancer_images) < args.n_samples:
                cancer_images.append(inputs[i])
                cancer_labels.append(lbl)
                cancer_preds.append(preds[i].item())
                
        if len(normal_images) == args.n_samples and len(cancer_images) == args.n_samples:
            break

    all_tensors = normal_images + cancer_images
    all_labels = normal_labels + cancer_labels
    all_preds = normal_preds + cancer_preds

    orig_imgs = []
    heatmaps = []
    overlays = []

    for t in all_tensors:
        t_batch = t.unsqueeze(0)
        hm = cam_analyzer.generate(t_batch)
        
        rgb = inv_normalize(t.cpu())
        orig_imgs.append(rgb)
        heatmaps.append(hm)
        
        overlay = cam_analyzer.overlay(rgb, hm)
        overlays.append(overlay)

    out_file = f"outputs/gradcam_{args.model}.png"
    Plotter.plot_gradcam_grid(orig_imgs, heatmaps, overlays, all_labels, all_preds, out_file)
    print(f"Saved Grad-CAM grid to {out_file}")

if __name__ == "__main__":
    main()
