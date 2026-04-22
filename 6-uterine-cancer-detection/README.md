# Beyond Local Convolution: Vision Transformers and Gabor Priors for Interpretable Uterine Cancer Histopathology

**Author:** Krushna Sanjay Sharma
**Recording:** [https://drive.google.com/file/d/1Jn7TysXmK7UEDK5XidVOyCjuQxqUNJt5/view?usp=drive_link](https://drive.google.com/file/d/1Jn7TysXmK7UEDK5XidVOyCjuQxqUNJt5/view?usp=drive_link)

---

## Project Description

This project extends our prior UCEC (Uterine Corpus Endometrial Carcinoma) cancer detection work with a comprehensive suite of methodological improvements to address underperformance, resolve class imbalance collapses, and boost baseline stability:

1. **Transfer-learned Vision Transformers (ViT):** Utilizing progressive unfreezing techniques to adapt large ViT structures to histopathology data without catastrophic forgetting.
2. **Two-Phase CNN Baselines (DenseNet / ResNet):** A robust orchestrator that stabilizes legacy ResNet and DenseNet models using a frozen-backbone warm-up phase, followed by deeply unfrozen fine-tuning.
3. **Gabor Filter Bank Injection:** Engineering mathematically derived Gabor filters into the initial convolution layer of a ResNet architecture to natively bypass learning fundamental textures from scratch.
4. **Grad-CAM Explainability:** Integration of dynamic Grad-CAM heatmaps across all four models (ViT, ResNet, DenseNet, GaborResNet), unlocking white-box transparency for predictions using deterministic, 1-to-1 patch comparisons.
5. **Stateful Auto-Resume Architecture:** Deep-level checkpointing allowing seamless mid-epoch train resuming (safeguarding gradients, AMP scaler, and historical metrics) when expanding batch limits.

## Dataset Information

- **Name:** CPTAC-UCEC dataset
- **Format:** 234 GB of pre-extracted 224x224 PNG patches across `cancerous` and `normal` subsets.
- **URL / DOI:** [Cancer Imaging Archive CPTAC-UCEC](https://www.cancerimagingarchive.net/collection/cptac-ucec/)

---

## Results

All v2 models were evaluated on a held-out test split of **25,280 patches** from the CPTAC-UCEC dataset with a strict **patient-level stratified 70/15/15 split** to prevent data leakage.

### v2 Results (This Work)

| Model                           | Accuracy   | Precision  | Recall     | F1 Score   | AUROC      |
| ------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| DenseNet-121 (Transfer)         | 95.44%     | —          | —          | 0.9669     | 0.9886     |
| ResNet-34 (Transfer)            | 95.65%     | **0.9838** | 0.9559     | 0.9697     | 0.9907     |
| GaborResNet (Ours)              | 96.08%     | 0.9737     | **0.9723** | **0.9730** | 0.9918     |
| ViT-B/16 (Progressive Transfer) | **96.40%** | —          | —          | **0.9759** | **0.9931** |

### v1 vs v2 Comparison

| Model        | v1 Accuracy | v1 AUROC | v2 Accuracy      | v2 AUROC   | Δ Accuracy  |
| ------------ | ----------- | -------- | ---------------- | ---------- | ----------- |
| DenseNet     | 99.15%\*    | 1.000\*  | 95.44%           | 0.9886     | −3.71%      |
| ResNet-34    | 99.10%\*    | 1.000\*  | 95.65%           | 0.9907     | −3.45%      |
| EfficientNet | 27.40%      | 0.528    | _(not included)_ | —          | —           |
| ViT          | 27.35%      | 0.508    | **96.40%**       | **0.9931** | **+68.65%** |

> **v1 CNN results marked \* are likely affected by data leakage.** An AUROC of 1.000 is not achievable on real medical histopathology data. v1 did not enforce patient-level stratification, causing slides from the same patient to appear in both train and test splits. v2 corrects this with strict patient-ID-based splitting, yielding lower but far more **reliable and generalizable** accuracy estimates.

> **The ViT's 27% → 96.40% recovery is the most significant v2 achievement.** The v1 ViT collapsed entirely due to catastrophic forgetting when fine-tuned with all layers unfrozen simultaneously. v2 resolves this with two-phase progressive unfreezing, transforming the ViT from a complete failure into the best-performing model in the benchmark.

---

## Installation Setup

### 1. Create Virtual Environment

Depending on your OS, create and activate a Python virtual environment to cleanly isolate your packages.

_(Windows)_

```bash
python -m venv venv
venv\Scripts\activate
```

_(macOS/Linux)_

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install all requisite frameworks including PyTorch, OpenCV, SciKit-Learn, and Grad-CAM.

```bash
pip install -r requirements.txt
```

### 3. CUDA GPU Configuration

To leverage the local 8 GB VRAM RTX 2080 MAX-Q hardware optimally, ensure Torch is synchronized with CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Project Execution & Commands

The central pipeline is governed entirely through `main.py`. Ensure your dataset patches are structured appropriately across `normal` and `cancerous` subdirectories. Data loaders are deeply optimized with `pin_memory=True` and Automatic Mixed Precision (AMP), so ensure you utilize a large `--batch-size` parameter (e.g. `128` or `256`) to saturate GPU throughput.

### Global Command

To sequentially execute all tasks internally, simply specify `all`:

```bash
python main.py --task all --patch-dir <path/to/patches>
```

### Task Execution Breakdown

| Task Number | Overview                                                             | Default Execution Command Example                                                            |
| ----------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Task 1**  | Train & benchmark progressive transfer-learning ViT.                 | `python main.py --task 1 --epochs 10 --lr 0.0001 --patch-dir patches/ --batch-size 128`      |
| **Task 2**  | Grad-CAM heatmap visualization over trained models.                  | `python main.py --task 2 --model <vit\|resnet\|densenet\|gabor_resnet> --patch-dir patches/` |
| **Task 3**  | Train mathematical Gabor filters into ResNet conv1.                  | `python main.py --task 3 --patch-dir patches/ --batch-size 128`                              |
| **Task 4**  | Train CNN Baselines (DenseNet/ResNet) using Two-Phase stabilization. | `python main.py --task 4 --model resnet --patch-dir patches/ --batch-size 128`               |
| **Task 5**  | Final Benchmark Evaluation generating universal Confusion Matrices.  | `python main.py --task 5 --patch-dir patches/ --batch-size 128`                              |

_(Note: Task 1, Task 3, and Task 4 gracefully support `Ctrl+C` Auto-Resume functionality. You can cancel and restart the command anytime and it will safely pick up right where the loss function left off!)_

---

## Utility Scripts

### Cross-Architecture Grad-CAM Comparison

Generates a full `2x5` comparison grid showing Grad-CAM heatmaps for all four models on a verified True Positive (cancerous) and True Negative (normal) patch side-by-side. Output is saved to `outputs/combined_gradcam_benchmark.png`.

> **Must be run as a module from the project root.**

```bash
python -m src.utils.plot_gradcam_comparison --patch-dir <path/to/patches>
```

| Argument             | Type  | Description                                                                                                                  |
| -------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--patch-dir`        | `str` | Path to the pre-extracted patch directory.                                                                                   |
| `--cancer-min-index` | `int` | Minimum dataset index to scan for a cancerous patch (default: `250`). Increase to select patches from deeper within the WSI. |
| `--normal-min-index` | `int` | Minimum dataset index to scan for a normal patch (default: `150`).                                                           |

## Application Arguments Reference

| Argument       | Type    | Applicable Tasks | Description                                                                                 |
| -------------- | ------- | ---------------- | ------------------------------------------------------------------------------------------- |
| `--task`       | `str`   | All              | Specifies which computational task to trigger (`1`, `2`, `3`, `4`, `5`, `all`).             |
| `--patch-dir`  | `str`   | All              | Absolute or relative path to the pre-extracted patch directory.                             |
| `--model`      | `str`   | 2, 4             | Target model parameter (`vit`, `densenet`, `resnet`, `gabor_resnet`).                       |
| `--batch-size` | `int`   | 1, 3, 4, 5       | Active batch size loader magnitude (Optimized to `128` via AMP).                            |
| `--epochs`     | `int`   | 1, 3, 4          | Training cycle iterations (default `10`).                                                   |
| `--lr`         | `float` | 1, 4             | Base learning rate applied prior to Phase 2 unfreezing drop (default `1e-4`).               |
| `--n-samples`  | `int`   | 2                | Number of exact normal and cancerous tissue samples to visualize dynamically (default `5`). |

---

## Output Architecture Tracker

Running the tools naturally delegates file writes autonomously under `models/` and `outputs/`. All graphical outputs are generated natively via our custom `Plotter` class.

| File / Object                         | Type       | Description                                                                                                                                         |
| ------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `taskX_learning_curves.png`           | Plot       | Loss and accuracy dual-axis parameter traces for ViT, ResNet, DenseNet, and GaborResNet.                                                            |
| `taskX_roc_curve.png`                 | Plot       | AUROC graphing to quantify model discriminative capability.                                                                                         |
| `gradcam_<model>.png`                 | Grid Map   | 10-patch deterministic `JET` heatmaps mapping interpretability for 1:1 model comparison.                                                            |
| `combined_gradcam_benchmark.png`      | Grid Map   | 2x5 cross-architecture Grad-CAM comparison on a verified TP (cancerous) and TN (normal) patch. Generated by `src/utils/plot_gradcam_comparison.py`. |
| `task3_gabor_bank.png`                | Plot       | 8x8 tiled display documenting the generated spatial convolutions.                                                                                   |
| `task3_conv1_compare.png`             | Plot       | Side-by-side array plotting engineered mathematical `Gabor` against dynamically learned `ResNet`.                                                   |
| `task5_master_benchmark.png`          | Plot       | Final comparative score bar chart across the 4 final trained models.                                                                                |
| `task5_<model>_cm.png`                | Matrix     | Classical 2x2 categorical True/False Positive mapping per model.                                                                                    |
| `*_transfer.pth` / `gabor_resnet.pth` | Checkpoint | State dictionaries storing finalized metrics and the optimal prediction epoch parameters.                                                           |
| `*_resume.pth`                        | Checkpoint | Raw optimizer layout, scaler blocks, and step traces used purely for pipeline interrupt protection.                                                 |
