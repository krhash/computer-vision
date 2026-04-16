# Pixels to Prognosis v2: Explainable Deep Learning for Uterine Cancer Detection

**Author:** Krushna Sanjay Sharma

---

## Project Description
This project extends our prior UCEC (Uterine Corpus Endometrial Carcinoma) cancer detection work with three massive methodological improvements to address underperformance on limited data:
1. **Transfer-learned Vision Transformers (ViT):** Utilizing progressive unfreezing techniques to adapt large ViT structures to histopathology data without overfitting.
2. **Grad-CAM Explainability:** Integration of dynamic Grad-CAM heatmaps across both our best CNN and ViT models, unlocking white-box transparency for predictions.
3. **Gabor Filter Bank Injection:** Engineering mathematically derived Gabor filters into the initial convolution layer of a ResNet architecture to bypass learning fundamental textures from scratch.

## Dataset Information
- **Name:** CPTAC-UCEC dataset
- **Format:** 234 GB of pre-extracted 224x224 PNG patches across `cancerous` and `normal` subsets (from prior project iterations).
- **URL / DOI:** [Cancer Imaging Archive CPTAC-UCEC](https://www.cancerimagingarchive.net/collection/cptac-ucec/)

---

## Installation Setup

### 1. Create Virtual Environment
Depending on your OS, create and activate a Python virtual environment to cleanly isolate your packages.

*(Windows)*
```bash
python -m venv venv
venv\Scripts\activate
```

*(macOS/Linux)*
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all requisite frameworks including PyTorch, OpenCV, SciKit-Learn, and Grad-CAM.
```bash
pip install -r requirements.txt
```

### 3. CUDA GPU Configuration (NVIDIA RTX 2080 MAX-Q)
To leverage the local 8 GB VRAM RTX 2080 MAX-Q hardware optimally, ensure Torch is synchronized with CUDA 12.4. You can override the pip torch distributions via:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Project Execution & Commands

The central pipeline is governed entirely through `main.py`. Ensure your dataset patches are structured appropriately before executing.

### Global Command
To sequentially execute all tasks internally without intervention, simply specify `all`:
```bash
python main.py --task all --patch-dir <path/to/patches>
```

### Task Execution Breakdown
| Task Number | Overview | Default Execution Command Example |
|---|---|---|
| **Task 1** | Progressive transfer-learning ViT benchmark. | `python main.py --task 1 --epochs 10 --lr 0.0001 --patch-dir patches/ --unfreeze-blocks 2` |
| **Task 2** | Grad-CAM heatmap visualization over top models. | `python main.py --task 2 --model densenet --n-samples 5 --patch-dir patches/` |
| **Task 3** | Train mathematical Gabor filters into ResNet conv1. | `python main.py --task 3 --epochs 10 --sigma 4.0 --gamma 0.5 --patch-dir patches/` |

---

## Application Arguments Reference

| Argument | Type | Applicable Tasks | Description |
|---|---|---|---|
| `--task` | `str` | All | Specifies which computational task to trigger (`1`, `2`, `3`, `all`). |
| `--patch-dir` | `str` | All | Absolute or relative path to the pre-extracted patch directory. |
| `--epochs` | `int` | 1, 3 | Training cycle iterations (default depends on task logic). |
| `--lr` | `float` | 1 | Base learning rate parameter applied universally before LR reduction phases (default `1e-4`). |
| `--batch-size` | `int` | 1, 3 | Active batch size loader magnitude (default `32`). |
| `--unfreeze-blocks` | `int` | 1 | Designates how many trailing ViT block layers to unfreeze in Phase 2 (default `2`). |
| `--model` | `str` | 2 | Target model for explainability (`vit` or `densenet`). |
| `--n-samples` | `int` | 2 | Number of exact normal and cancerous tissue samples to visualize dynamically (default `5`). |
| `--sigma` | `float` | 3 | Base standard deviation for Gabor kernel envelope modulation. |
| `--gamma` | `float` | 3 | Gabor filter spatial aspect ratio envelope dimension. |

---

## Output Architecture Tracker

Running the tools naturally delegates file writes autonomously under `models/` and `outputs/`.

| File/Object | Type | Description |
|---|---|---|
| `task1_learning_curves.png` | Plot | Loss and accuracy dual-axis visualization tracing ViT transfer phases. |
| `task1_roc_curve.png` | Plot | AUC validation parameters graphing test-set robustness. |
| `gradcam_densenet.png` | Grid Map | 3xN overlay detailing true patches against GradCAM heatmaps on DenseNet. |
| `gradcam_vit.png` | Grid Map | 3xN overlay detailing true patches against GradCAM heatmaps on ViT. |
| `task3_gabor_bank.png` | Plot | 8x8 tiled display documenting the generated spatial convolutions. |
| `task3_conv1_compare.png` | Plot | Side-by-side array plotting engineered `Gabor` against `ResNet`. |
| `task3_benchmark_table.png` | Graph Diagram| End-to-end bar plot charting ResNet benchmark versus GaborResNet. |
| `vit_transfer.pth` | Checkpoint | Saved best weights for the ViT classification model. |
| `gabor_resnet.pth` | Checkpoint | Saved best weights for the engineered ResNet modification. |

---

## Relevant Links
- **Demo Video URL**: [Insert YouTube/Vimeo Link Here]
- **Custom Image Assets**: Generated plots will reside securely in the standard `outputs/` folder upon task completion. 
