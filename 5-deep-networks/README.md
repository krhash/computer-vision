# Project 5: Recognition using Deep Networks
**Author:** Krushna Sanjay Sharma  
**Course:** CS 5330 — Pattern Recognition & Computer Vision  
**Date:** 2026

MNIST digit recognition using CNNs and Transformers in PyTorch. Covers training, filter analysis, transfer learning on Greek letters, transformer-based classification, and automated hyperparameter experimentation on Fashion MNIST.

---

## Project Structure

```
project5_deep_networks/
├── main.py                            # Master pipeline (run all or selected tasks)
├── requirements.txt
│
├── tasks/
│   ├── task1_build_train.py           # 1A–1F: build, train, save, evaluate CNN
│   ├── task2_examine.py               # 2A–2B: filter analysis + cv2.filter2D
│   ├── task3_greek.py                 # Transfer learning on Greek letters
│   ├── task4_transformer.py           # Transformer network drop-in replacement
│   ├── task5_experiment.py            # Transformer hyperparameter sweep (Fashion MNIST)
│   ├── task5b_cnn_optimizer.py        # CNN optimizer sweep (Fashion MNIST)
│   └── read_network.py                # Shared utility: load_trained_model()
│
├── src/
│   ├── network/
│   │   ├── digit_network.py           # DigitNetwork — CNN architecture
│   │   └── transformer_network.py    # NetTransformer + NetConfig + PatchEmbedding
│   ├── data/
│   │   ├── mnist_loader.py            # MNISTDataLoader
│   │   ├── fashion_loader.py          # FashionMNISTLoader
│   │   ├── greek_loader.py            # GreekDataLoader + GreekTransform
│   │   └── handwritten_loader.py      # HandwrittenLoader (your digit photos)
│   ├── training/
│   │   ├── trainer.py                 # Trainer — epoch-level training loop
│   │   └── transfer_trainer.py        # TransferTrainer — freeze + replace layer
│   ├── evaluation/
│   │   ├── evaluator.py               # Evaluator — accuracy + per-sample output
│   │   └── filter_analyzer.py         # FilterAnalyzer — conv weights + filter2D
│   ├── experiment/
│   │   ├── experiment_config.py       # ExperimentConfig — transformer sweep dims
│   │   ├── experiment_runner.py       # ExperimentRunner — transformer sweep executor
│   │   ├── cnn_experiment_config.py   # CNNExperimentConfig — optimizer sweep dims
│   │   └── cnn_experiment_runner.py   # CNNExperimentRunner — optimizer sweep executor
│   ├── visualization/
│   │   └── plotter.py                 # Plotter — all matplotlib figures
│   └── utils/
│       ├── model_io.py                # ModelIO — save/load .pth files
│       └── device_utils.py            # get_device() — auto-detect CUDA/MPS/CPU
│
├── data/
│   ├── greek_train/                   # alpha/, beta/, gamma/ subdirectories
│   ├── greek_custom/                  # Your own Greek letter photos
│   └── handwritten/                   # Your digit photos: 0.jpg – 9.jpg
├── models/                            # Saved .pth files (auto-created)
└── outputs/                           # Saved plots as PNG + CSV results (auto-created)
```

---

## Build & Execution

### 1. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

**CPU only:**
```bash
pip install -r requirements.txt
```

**CUDA-enabled (recommended — check version with `nvidia-smi`):**
```bash
# CUDA 12.4 / 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Verify CUDA is detected:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3. Run individual tasks

```bash
# Task 1 — Build, train, save, evaluate CNN (subtasks 1A–1F)
python tasks/task1_build_train.py                  # defaults: 5 epochs, batch=64
python tasks/task1_build_train.py 10 128           # override: 10 epochs, batch=128

# Task 1E — Read network and run on test set  [requires models/mnist_cnn.pth]
python tasks/read_network.py
python tasks/read_network.py --model mnist_cnn.pth --model-dir ./models

# Task 2 — Examine filters and filter effects  [requires models/mnist_cnn.pth]
python tasks/task2_examine.py

# Task 3 — Transfer learning on Greek letters  [requires models/mnist_cnn.pth]
python tasks/task3_greek.py
python tasks/task3_greek.py --epochs 150

# Task 4 — Transformer network
python tasks/task4_transformer.py
python tasks/task4_transformer.py --epochs 15 --batch-size 64

# Task 5A — Transformer hyperparameter sweep on Fashion MNIST
python tasks/task5_experiment.py
python tasks/task5_experiment.py --epochs 10

# Task 5B — CNN optimizer sweep on Fashion MNIST
python tasks/task5b_cnn_optimizer.py
python tasks/task5b_cnn_optimizer.py --epochs 5
```

### 4. Run the full pipeline
```bash
python main.py                     # runs all tasks (1–5) in order
python main.py --task 1            # runs Task 1 only
python main.py --task 1 2          # runs Tasks 1 and 2
```

> **Task dependencies:**
> - Tasks 2 and 3 require `models/mnist_cnn.pth` — run Task 1 first
> - Tasks 4, 5A, 5B are fully standalone

---

## Arguments

| File | Argument | Type | Default | Description |
|---|---|---|---|---|
| `task1_build_train.py` | `argv[1]` | int | `5` | Number of training epochs |
| `task1_build_train.py` | `argv[2]` | int | `64` | Training batch size |
| `read_network.py` | `--model` | str | `mnist_cnn.pth` | Model filename |
| `read_network.py` | `--model-dir` | str | `./models` | Model directory |
| `task2_examine.py` | `--model` | str | `mnist_cnn.pth` | Model filename |
| `task2_examine.py` | `--model-dir` | str | `./models` | Model directory |
| `task2_examine.py` | `--data-dir` | str | `./data` | MNIST data directory |
| `task3_greek.py` | `--model` | str | `mnist_cnn.pth` | Pre-trained model filename |
| `task3_greek.py` | `--greek-dir` | str | `./data/greek_train` | Greek letter dataset directory |
| `task3_greek.py` | `--custom-dir` | str | `./data/greek_custom` | Your Greek letter photos |
| `task3_greek.py` | `--epochs` | int | `50` | Number of training epochs |
| `task4_transformer.py` | `--epochs` | int | `15` | Number of training epochs |
| `task4_transformer.py` | `--batch-size` | int | `64` | Training batch size |
| `task4_transformer.py` | `--model-dir` | str | `./models` | Model save directory |
| `task5_experiment.py` | `--epochs` | int | `10` | Epochs per sweep run |
| `task5_experiment.py` | `--output-dir` | str | `./outputs` | Output directory |
| `task5b_cnn_optimizer.py` | `--epochs` | int | `5` | Epochs per sweep run |
| `task5b_cnn_optimizer.py` | `--output-dir` | str | `./outputs` | Output directory |
| `main.py` | `--task` | int(s) | `1 2 3 4 5` | Task number(s) to run |

---

## Key Classes

### Networks — `src/network/`
| Class | File | Description |
|---|---|---|
| `DigitNetwork` | `digit_network.py` | CNN: conv→pool→conv→dropout→pool→fc→fc. Input `(N,1,28,28)`, output log-probs `(N,10)` |
| `NetTransformer` | `transformer_network.py` | Patch-based ViT-style transformer. Drop-in replacement for `DigitNetwork` |
| `NetConfig` | `transformer_network.py` | Dataclass holding all transformer hyperparameters |
| `PatchEmbedding` | `transformer_network.py` | Divides image into patches, projects each to embedding space |

### Data — `src/data/`
| Class | File | Description |
|---|---|---|
| `MNISTDataLoader` | `mnist_loader.py` | Downloads MNIST, returns shuffled train + unshuffled test loader |
| `FashionMNISTLoader` | `fashion_loader.py` | Same interface as MNISTDataLoader but for Fashion MNIST |
| `HandwrittenLoader` | `handwritten_loader.py` | Reads digit photos: greyscale → Otsu threshold → invert → resize 28×28 |
| `GreekDataLoader` | `greek_loader.py` | Loads `alpha/beta/gamma` folders via `ImageFolder` |
| `GreekTransform` | `greek_loader.py` | Converts 133×133 RGB Greek images to 28×28 greyscale MNIST-format tensors |

### Training — `src/training/`
| Class | File | Description |
|---|---|---|
| `Trainer` | `trainer.py` | Runs one epoch, accumulates loss/accuracy history |
| `TransferTrainer` | `transfer_trainer.py` | Freezes all weights, replaces `fc2` with `Linear(50, N)` |

### Evaluation — `src/evaluation/`
| Class | File | Description |
|---|---|---|
| `Evaluator` | `evaluator.py` | Test accuracy over DataLoader; `predict_samples()` prints per-sample output values |
| `FilterAnalyzer` | `filter_analyzer.py` | Extracts `conv1` weights `[10,1,5,5]`; applies filters via `cv2.filter2D` |

### Experiment — `src/experiment/`
| Class | File | Description |
|---|---|---|
| `ExperimentConfig` | `experiment_config.py` | Generates transformer sweep configs (patch_size × embed_dim × depth) |
| `ExperimentRunner` | `experiment_runner.py` | Trains each transformer variant, saves CSV incrementally |
| `CNNExperimentConfig` | `cnn_experiment_config.py` | Generates CNN optimizer sweep configs |
| `CNNExperimentRunner` | `cnn_experiment_runner.py` | Trains each CNN variant, saves CSV incrementally |

### Utilities
| Class | File | Description |
|---|---|---|
| `ModelIO` | `utils/model_io.py` | `save(model, filename)` / `load(model, filename)` for `.pth` state dicts |
| `get_device()` | `utils/device_utils.py` | Auto-detects CUDA / MPS / CPU via `torch.accelerator` |
| `Plotter` | `visualization/plotter.py` | All figures: training curves, filter grids, prediction grids, sweep results |

---

## Data Setup

### Handwritten digits (Task 1F)
```
data/handwritten/
    0.jpg  1.jpg  2.jpg  ...  9.jpg    ← one file per digit, any size
```
Use thick marker lines, white background. Images are auto-resized to 28×28 and inverted.

### Greek letters (Task 3)
```
data/greek_train/
    alpha/   *.png    (provided dataset — 9 images each)
    beta/    *.png
    gamma/   *.png

data/greek_custom/                     ← your own photos (~128×128)
    0_1.jpg  0_2.jpg  ...              ← 0 = alpha
    1_1.jpg  1_2.jpg  ...              ← 1 = beta
    2_1.jpg  2_2.jpg  ...              ← 2 = gamma
```

## Extensions (Extra Credit)

### Extension 1 — Pre-trained Network Conv Layer Analysis
Loads a pre-trained torchvision model and analyses its first conv layer,
mirroring Task 2 but on ImageNet-trained filters.

```bash
python tasks/ext1_pretrained_analysis.py                    # ResNet18 (default)
python tasks/ext1_pretrained_analysis.py --model vgg16
python tasks/ext1_pretrained_analysis.py --model alexnet
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet18` | Model: resnet18, vgg16, alexnet, resnet50 |
| `--data-dir` | `./data` | MNIST data directory |
| `--output-dir` | `./outputs` | Output directory |

Outputs: `ext1_pretrained_filters.png`, `ext1_pretrained_responses.png`, `ext1_filter_comparison.png`

---

### Extension 2 — Gabor Filter Bank as First Conv Layer
Replaces `conv1` with hand-crafted Gabor filters, freezes it, and retrains
only `conv2` + FC layers. Compares accuracy against the fully learned network.

```bash
python tasks/ext2_gabor_network.py                          # default settings
python tasks/ext2_gabor_network.py --epochs 10 --sigma 1.5 --gamma 0.5
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `10` | Training epochs |
| `--sigma` | `1.5` | Gabor Gaussian envelope std |
| `--gamma` | `0.5` | Gabor spatial aspect ratio |
| `--data-dir` | `./data` | MNIST data directory |
| `--output-dir` | `./outputs` | Output directory |

Result: Gabor network **99.02%** vs Learned **99.15%** — only 0.13% difference.
Outputs: `ext2_gabor_filters.png`, `ext2_gabor_vs_learned.png`, `ext2_training_curves.png`

---

### Extension 3 — Live Webcam Digit Recognition
Real-time digit recognition using a trained network and your webcam.
Hold a digit inside the green box. The network predicts in real time.

```bash
python tasks/ext3_live_recognition.py                       # default webcam
python tasks/ext3_live_recognition.py --camera 1            # alternate camera
python tasks/ext3_live_recognition.py --model gabor_network.pth  # use Gabor network
```

| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Webcam index |
| `--model` | `mnist_cnn.pth` | Model file to use |
| `--model-dir` | `./models` | Model directory |
| `--output-dir` | `./outputs` | Screenshot save directory |

Controls: `Q` quit · `S` save screenshot · `R` reset display



All figures saved to `outputs/`, models saved to `models/`:

| File | Task | Description |
|---|---|---|
| `task1a_sample_digits.png` | 1A | First 6 MNIST test digits |
| `task1c_training_curves.png` | 1C | CNN loss and accuracy per epoch |
| `task1e_predictions.png` | 1E | 3×3 grid with predicted labels |
| `task1f_handwritten.png` | 1F | Handwritten digits + predictions |
| `task2a_filters.png` | 2A | 10 conv1 filter weight visualisations |
| `task2b_filter_effects.png` | 2B | Filter + response pairs (5×4 grid) |
| `task3_training_curve.png` | 3 | Greek letter transfer learning loss |
| `task3_custom_greek.png` | 3 | Custom Greek letter predictions |
| `task4_transformer_curves.png` | 4 | Transformer training curves |
| `task4_predictions.png` | 4 | Transformer 3×3 prediction grid |
| `task5_experiment_results.png` | 5A | Transformer sweep — 4-subplot figure |
| `task5_top_configs.png` | 5A | Top 10 transformer configs bar chart |
| `task5_results.csv` | 5A | Full transformer sweep log (55 runs) |
| `task5b_cnn_results.png` | 5B | CNN optimizer sweep — 4-subplot figure |
| `task5b_cnn_top_configs.png` | 5B | Top 10 CNN configs bar chart |
| `task5b_cnn_results.csv` | 5B | Full CNN optimizer sweep log (26 runs) |
| `models/mnist_cnn.pth` | 1D | Trained CNN weights |
| `models/greek_transfer.pth` | 3 | Transfer-learned Greek letter model |
| `models/mnist_transformer.pth` | 4 | Trained transformer weights |

---

## Experiment Results Summary

### Task 5A — Transformer Sweep on Fashion MNIST (55 runs)

| Dimension | Range | Best Value | Finding |
|---|---|---|---|
| patch_size | 2, 4, 7, 14 | **4** | patch=2 worst (too many noisy tokens) |
| embed_dim | 16, 32, 48, 96 | **48** | embed=96 adds time, no accuracy gain |
| depth | 1, 2, 4, 6 | **4** (baseline) | depth=2 nearly as good, 31% faster |

Baseline: **89.49%** — Best found: **89.49%** (baseline already well-tuned)
Best efficiency: depth=2 → 88.96% in 260s vs baseline 375s

### Task 5B — CNN Optimizer Sweep on Fashion MNIST (26 runs)

| Optimizer | Best LR | Best Accuracy |
|---|---|---|
| SGD | 0.01, mom=0.9 | 87.94% |
| Adam | 0.001 | 88.93% |
| **AdamW** | **0.001** | **89.32%** |
| RMSprop | 0.001 | 87.74% |

Baseline (SGD lr=0.01 mom=0.5): **86.29%** — Best: **AdamW lr=0.001 → 89.32%** (+3.03%)
Key finding: lr=0.1 causes complete divergence for all adaptive optimizers. SGD momentum=0.99 diverges catastrophically.