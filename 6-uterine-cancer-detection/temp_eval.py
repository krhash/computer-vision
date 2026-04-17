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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    patch_dir = r"C:\Users\krush\Documents\GitHub\vit-ucec-detection\processed_slides"
    _, _, test_ds = UCECPatchDataset.create_splits(
        root_dir=patch_dir,
        test_ratio=0.05, val_ratio=0.05, train_ratio=0.9,
        test_transform=PatchTransforms.get_val_transforms()
    )
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    models_to_test = [
        ("ViT", PretrainedViT(num_classes=2), "models/vit_transfer.pth"),
        ("DenseNet", PretrainedDenseNet(num_classes=2), "models/densenet_transfer.pth"),
        ("ResNet", PretrainedResNet(num_classes=2), "models/resnet_transfer.pth"),
        ("Gabor ResNet", GaborResNet(num_classes=2, num_filters=64, kernel_size=7), "models/gabor_resnet.pth")
    ]

    for name, model, path in models_to_test:
        print(f"\n--- Evaluating {name} ---")
        try:
            missing, unexpected = model.load_state_dict(torch.load(path, map_location=device), strict=False)
            print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"Sample missing: {missing[:3]}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
            
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(test_loader)
        print(f"[{name} Metrics]: {metrics}")
        
        cm = evaluator.compute_confusion_matrix(test_loader)
        print(f"Confusion Matrix:\n{cm}")
        
        name_clean = name.replace(" ", "_").lower()
        Plotter.plot_confusion_matrix(cm, ["Normal", "Cancerous"], f"outputs/temp_eval_{name_clean}_cm.png")

if __name__ == '__main__':
    main()
