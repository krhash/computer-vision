# Pixels to Prognosis v2: Explainable Deep Learning for Uterine Cancer Detection

**Author:** Krushna Sanjay Sharma

## Project Description
Extends prior UCEC cancer detection work with transfer-learned ViT, Grad-CAM explainability heatmaps (for best CNN and ViT), and a custom Gabor filter bank replacing ResNet conv1 to overcome data limitations and increase interpretability.

## Dataset
- **Name:** CPTAC-UCEC dataset
- **URL/DOI:** [Dataset Link Placeholder]
- Note: Uses 234 GB pre-extracted patches from the prior project.

## Installation Instructions

1. **Create Virtual Environment:**
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

2. **Install Dependencies:**
   Install the required Python packages from the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For Windows with NVIDIA RTX 2080 MAX-Q, install Torch compiled with CUDA 12.4+ if needed by running:*
   `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`

## Execution Commands

### Run Everything
```bash
python main.py --task all
```

### Task 1: Transfer-learned ViT
```bash
python main.py --task 1 --epochs 10 --lr 0.0001 --batch-size 32 --patch-dir patches/ --unfreeze-blocks 2
```

### Task 2: Grad-CAM Explainability
```bash
python main.py --task 2 --model densenet --n-samples 5 --patch-dir patches/
```

### Task 3: Gabor Filter Bank ResNet
```bash
python main.py --task 3 --epochs 10 --sigma 4.0 --gamma 0.5 --patch-dir patches/
```

## Relevant Links
- **Demo Video URL**: [TBA]
- **Custom Images/Assets**: Located in `outputs/`
