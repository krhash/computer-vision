"""
Author: Krushna Sanjay Sharma
Description: Utility script to generate a cross-architecture Grad-CAM comparison grid.

Scans the test split for a verified True Positive (cancerous, correctly predicted) and
a True Negative (normal, correctly predicted) using the ViT model, then runs Grad-CAM
across all four architectures on those patches and saves a 2x5 comparison figure.

Usage (run from project root):
    python -m src.utils.plot_gradcam_comparison --patch-dir <path>
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# All imports are relative to the package — no sys.path hacks needed when
# run as a module from the project root via `python -m src.utils.plot_gradcam_comparison`
from src.data.patch_dataset import UCECPatchDataset
from src.data.patch_transforms import PatchTransforms
from src.network.vit_transfer import PretrainedViT
from src.network.resnet_transfer import PretrainedResNet
from src.network.densenet_transfer import PretrainedDenseNet
from src.network.gabor_resnet import GaborResNet
from src.evaluation.gradcam import GradCAM
from src.utils.model_io import ModelIO

# Project root is two levels up from this file (src/utils/ -> src/ -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_WEIGHTS = {
    "ViT":        os.path.join(PROJECT_ROOT, "models", "vit_transfer.pth"),
    "DenseNet":   os.path.join(PROJECT_ROOT, "models", "densenet_transfer.pth"),
    "ResNet":     os.path.join(PROJECT_ROOT, "models", "resnet_transfer.pth"),
    "GaborResNet":os.path.join(PROJECT_ROOT, "models", "gabor_resnet.pth"),
}

TARGET_LAYERS = {
    "ViT":        "model.encoder.layers.11.ln_1",
    "DenseNet":   "model.features.norm5",
    "ResNet":     "model.layer4.2.conv2",
    "GaborResNet":"model.layer4.2.conv2",
}

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "combined_gradcam_benchmark.png")


def _inverse_transform(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized ImageNet tensor back to a [0, 1] HWC numpy array.

    Args:
        tensor: A (C, H, W) float tensor normalized with ImageNet statistics.

    Returns:
        A (H, W, C) numpy float32 array in the range [0, 1].
    """
    mean = torch.tensor(PatchTransforms.MEAN).view(3, 1, 1)
    std = torch.tensor(PatchTransforms.STD).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()


def _find_representative_patches(
    test_ds: UCECPatchDataset,
    vit_model: torch.nn.Module,
    device: torch.device,
    cancer_min_index: int = 250,
    normal_min_index: int = 150,
) -> tuple:
    """Scans the test set to find a verified True Positive and True Negative.

    Uses the ViT model's predictions to guarantee that the cancerous patch is
    truly detected as cancerous, and the normal patch is truly detected as normal,
    ensuring the Grad-CAM heatmaps are clinically meaningful.

    Args:
        test_ds: The test split of UCECPatchDataset.
        vit_model: A loaded, eval-mode ViT model used for pre-filtering patches.
        device: The torch device to run inference on.
        cancer_min_index: Minimum dataset index to search for a cancer patch (skips
            early patches that tend to be mostly background).
        normal_min_index: Minimum dataset index to search for a normal patch.

    Returns:
        A tuple of (cancer_tensor, cancer_rgb, normal_tensor, normal_rgb) where
        tensors are (1, C, H, W) on device and rgb arrays are (H, W, C) numpy [0,1].

    Raises:
        RuntimeError: If either a cancerous or normal qualifying patch cannot be found.
    """
    cancer_tensor, cancer_rgb = None, None
    normal_tensor, normal_rgb = None, None

    with torch.no_grad():
        for i in range(len(test_ds)):
            img, label = test_ds[i]
            img_batch = img.unsqueeze(0).to(device)
            pred = torch.argmax(vit_model(img_batch), dim=1).item()

            if label == 1 and pred == 1 and cancer_tensor is None and i > cancer_min_index:
                cancer_tensor = img_batch
                cancer_rgb = _inverse_transform(img.cpu())
                print(f"  [Cancerous] Selected patch at index {i} (ViT: TP)")

            if label == 0 and pred == 0 and normal_tensor is None and i > normal_min_index:
                normal_tensor = img_batch
                normal_rgb = _inverse_transform(img.cpu())
                print(f"  [Normal]    Selected patch at index {i} (ViT: TN)")

            if cancer_tensor is not None and normal_tensor is not None:
                break

    if cancer_tensor is None or normal_tensor is None:
        raise RuntimeError(
            "Could not find qualifying patches. Try lowering --cancer-min-index or --normal-min-index."
        )

    return cancer_tensor, cancer_rgb, normal_tensor, normal_rgb


def _run_gradcam_all_models(
    models: dict,
    cancer_tensor: torch.Tensor,
    cancer_rgb: np.ndarray,
    normal_tensor: torch.Tensor,
    normal_rgb: np.ndarray,
) -> tuple:
    """Runs Grad-CAM for all models on both the cancerous and normal patches.

    Args:
        models: Dict mapping model name to loaded nn.Module instance.
        cancer_tensor: (1, C, H, W) tensor of the cancerous patch on device.
        cancer_rgb: (H, W, C) numpy array of the cancerous patch in [0, 1].
        normal_tensor: (1, C, H, W) tensor of the normal patch on device.
        normal_rgb: (H, W, C) numpy array of the normal patch in [0, 1].

    Returns:
        A tuple (overlays_cancer, overlays_normal), each a dict mapping model
        name to an (H, W, 3) uint8 numpy overlay array.
    """
    overlays_cancer, overlays_normal = {}, {}

    for name, model in models.items():
        print(f"  Running Grad-CAM for {name}...")
        try:
            for param in model.parameters():
                param.requires_grad = True
            model.eval()
            cam = GradCAM(model, TARGET_LAYERS[name])

            overlays_cancer[name] = cam.overlay(cancer_rgb, cam.generate(cancer_tensor))
            overlays_normal[name] = cam.overlay(normal_rgb, cam.generate(normal_tensor))
        except Exception as exc:
            print(f"  [WARNING] {name} failed: {exc}")
            overlays_cancer[name] = np.zeros_like(cancer_rgb)
            overlays_normal[name] = np.zeros_like(normal_rgb)

    return overlays_cancer, overlays_normal


def _save_comparison_grid(
    cancer_rgb: np.ndarray,
    normal_rgb: np.ndarray,
    overlays_cancer: dict,
    overlays_normal: dict,
    out_path: str,
) -> None:
    """Renders and saves the 2x5 Grad-CAM comparison figure.

    Args:
        cancer_rgb: (H, W, C) numpy array of the original cancerous patch.
        normal_rgb: (H, W, C) numpy array of the original normal patch.
        overlays_cancer: Dict of model name -> (H, W, 3) uint8 overlay for cancer row.
        overlays_normal: Dict of model name -> (H, W, 3) uint8 overlay for normal row.
        out_path: Absolute path to save the output PNG file.
    """
    model_names = ["DenseNet", "ResNet", "GaborResNet", "ViT"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    axes[0, 0].imshow(cancer_rgb)
    axes[0, 0].set_title("Original\n(Cancerous)", fontsize=12)
    axes[0, 0].axis("off")

    axes[1, 0].imshow(normal_rgb)
    axes[1, 0].set_title("Original\n(Normal)", fontsize=12)
    axes[1, 0].axis("off")

    for col, name in enumerate(model_names, start=1):
        axes[0, col].imshow(overlays_cancer[name])
        axes[0, col].set_title(name, fontsize=12)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlays_normal[name])
        axes[1, col].set_title(name, fontsize=12)
        axes[1, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a cross-architecture Grad-CAM comparison grid."
    )
    parser.add_argument("--patch-dir", type=str, required=True,
                        help="Path to the root patch directory (same as --patch-dir in main.py).")
    parser.add_argument("--cancer-min-index", type=int, default=250,
                        help="Minimum dataset index for cancerous patch selection (default: 250).")
    parser.add_argument("--normal-min-index", type=int, default=150,
                        help="Minimum dataset index for normal patch selection (default: 150).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pre-load ViT for patch filtering
    print("\nLoading ViT for patch pre-filtering...")
    vit = PretrainedViT(num_classes=2)
    ModelIO.load_checkpoint(vit, MODEL_WEIGHTS["ViT"], device)
    vit = vit.to(device)
    vit.eval()

    # Build test dataset
    print("\nBuilding dataset...")
    val_transform = PatchTransforms.get_val_transforms()
    _, _, test_ds = UCECPatchDataset.create_splits(
        args.patch_dir,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        train_transform=val_transform,
        val_transform=val_transform,
        test_transform=val_transform,
    )

    torch.manual_seed(42)

    # Find representative patches
    print("\nScanning for representative patches...")
    cancer_tensor, cancer_rgb, normal_tensor, normal_rgb = _find_representative_patches(
        test_ds, vit, device,
        cancer_min_index=args.cancer_min_index,
        normal_min_index=args.normal_min_index,
    )

    # Load all models
    print("\nLoading remaining models...")
    models = {
        "ViT":         vit,
        "DenseNet":    PretrainedDenseNet(num_classes=2).to(device),
        "ResNet":      PretrainedResNet(num_classes=2).to(device),
        "GaborResNet": GaborResNet(num_classes=2).to(device),
    }
    for name in ["DenseNet", "ResNet", "GaborResNet"]:
        ModelIO.load_checkpoint(models[name], MODEL_WEIGHTS[name], device)

    # Run Grad-CAM
    print("\nRunning Grad-CAM...")
    overlays_cancer, overlays_normal = _run_gradcam_all_models(
        models, cancer_tensor, cancer_rgb, normal_tensor, normal_rgb
    )

    # Save figure
    _save_comparison_grid(cancer_rgb, normal_rgb, overlays_cancer, overlays_normal, OUTPUT_PATH)


if __name__ == "__main__":
    main()
