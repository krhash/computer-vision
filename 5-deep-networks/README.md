# Project 5: Recognition using Deep Networks
**Author:** Krushna Sanjay Sharma  
**Course:** CS 5330 — Pattern Recognition & Computer Vision  
**Date:** 2026

MNIST digit recognition using CNNs and Transformers in PyTorch. Covers training, filter analysis, transfer learning on Greek letters, transformer-based classification, and automated hyperparameter experimentation.

---

## Project Structure

```
project5_deep_networks/
├── main.py                        # Master pipeline (run all or selected tasks)
├── requirements.txt
│
├── tasks/                         # Thin orchestrators — one file per task
│   ├── task1_build_train.py       # 1A–1F: build, train, save, evaluate
│   ├── task2_examine.py           # 2A–2B: filter analysis + cv2.filter2D
│   ├── task3_greek.py             # Transfer learning on Greek letters
│   ├── task4_transformer.py       # Drop-in transformer replacement
│   └── task5_experiment.py        # Hyperparameter sweep (50–100 variants)
│
├── src/
│   ├── network/
│   │   ├── digit_network.py       # DigitNetwork — CNN architecture
│   │   └── transformer_network.py # NetTransformer — ViT-style architecture
│   ├── data/
│   │   ├── mnist_loader.py        # MNISTDataLoader
│   │   ├── greek_loader.py        # GreekDataLoader + GreekTransform
│   │   └── handwritten_loader.py  # HandwrittenLoader (your digit photos)
│   ├── training/
│   │   ├── trainer.py             # Trainer — epoch-level training loop
│   │   └── transfer_trainer.py    # TransferTrainer — freeze + replace layer
│   ├── evaluation/
│   │   ├── evaluator.py           # Evaluator — accuracy + per-sample output
│   │   └── filter_analyzer.py     # FilterAnalyzer — conv weights + filter2D
│   ├── visualization/
│   │   └── plotter.py             # Plotter — all matplotlib figures
│   ├── experiment/
│   │   ├── experiment_config.py   # ExperimentConfig — sweep dimensions
│   │   └── experiment_runner.py   # ExperimentRunner — executes + logs sweep
│   └── utils/
│       └── model_io.py            # ModelIO — save/load .pth files
│
├── data/
│   ├── greek_letters/             # alpha/, beta/, gamma/ subdirectories
│   └── handwritten/               # Your digit photos: 0.png – 9.png
├── models/                        # Saved .pth files (auto-created)
└── outputs/                       # Saved plots as PNG (auto-created)
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
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Verify CUDA is detected:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 3. Run individual tasks
```bash
# Task 1 — Build, train, save, evaluate (subtasks 1A–1F)
python tasks/task1_build_train.py                  # defaults: 5 epochs, batch=64
python tasks/task1_build_train.py 10 128           # override: 10 epochs, batch=128

# Task 2 — Examine filters and filter effects  [requires models/mnist_cnn.pth]
python tasks/task2_examine.py

# Task 3 — Transfer learning on Greek letters  [requires models/mnist_cnn.pth]
python tasks/task3_greek.py

# Task 4 — Transformer network                 [standalone]
python tasks/task4_transformer.py

# Task 5 — Hyperparameter experiment sweep     [standalone]
python tasks/task5_experiment.py
```

### 4. Run the full pipeline
```bash
python main.py                     # runs all tasks (1–5) in order
python main.py --task 1            # runs Task 1 only
python main.py --task 1 2          # runs Tasks 1 and 2
```

> **Note:** Tasks 2 and 3 depend on `models/mnist_cnn.pth` produced by Task 1. Always run Task 1 first.

---

## Arguments

| File | Argument | Type | Default | Description |
|---|---|---|---|---|
| `task1_build_train.py` | `argv[1]` | int | `5` | Number of training epochs |
| `task1_build_train.py` | `argv[2]` | int | `64` | Training batch size |
| `main.py` | `--task` | int(s) | `1 2 3 4 5` | Task number(s) to run |

---

## Key Classes

### Networks — `src/network/`
| Class | File | Description |
|---|---|---|
| `DigitNetwork` | `digit_network.py` | CNN: conv→pool→conv→dropout→pool→fc→fc. Input `(N,1,28,28)`, output log-probs `(N,10)` |
| `NetTransformer` | `transformer_network.py` | Patch-based transformer. Drop-in replacement for `DigitNetwork` with identical I/O |

### Data — `src/data/`
| Class | File | Description |
|---|---|---|
| `MNISTDataLoader` | `mnist_loader.py` | Downloads MNIST, returns shuffled train loader and unshuffled test loader |
| `HandwrittenLoader` | `handwritten_loader.py` | Reads your digit photos, applies greyscale→resize→invert→normalise pipeline |
| `GreekDataLoader` | `greek_loader.py` | Loads `alpha/beta/gamma` folders via `ImageFolder`; applies `GreekTransform` |
| `GreekTransform` | `greek_loader.py` | Converts 133×133 RGB Greek images to 28×28 greyscale inverted MNIST-format tensors |

### Training — `src/training/`
| Class | File | Description |
|---|---|---|
| `Trainer` | `trainer.py` | Runs one epoch, accumulates loss/accuracy history, logs per-batch progress |
| `TransferTrainer` | `transfer_trainer.py` | Freezes all weights, replaces the final `fc2` layer, trains only the new layer |

### Evaluation — `src/evaluation/`
| Class | File | Description |
|---|---|---|
| `Evaluator` | `evaluator.py` | Computes test accuracy over a full DataLoader; `predict_samples()` prints per-sample output values, predicted index, and true label |
| `FilterAnalyzer` | `filter_analyzer.py` | Extracts `conv1` weights `[10,1,5,5]`; applies each filter via `cv2.filter2D` |

### Utilities — `src/utils/` and `src/visualization/`
| Class | File | Description |
|---|---|---|
| `ModelIO` | `model_io.py` | `save(model, filename)` / `load(model, filename)` for `.pth` state dicts |
| `Plotter` | `plotter.py` | All figures: sample grid, training curves, prediction grid, filter plots, experiment results. Saves PNG to `outputs/` |

### Experiment — `src/experiment/`
| Class | File | Description |
|---|---|---|
| `ExperimentConfig` | `experiment_config.py` | Defines sweep dimensions (epochs, batch size, dropout, filter count, etc.) |
| `ExperimentRunner` | `experiment_runner.py` | Iterates configs, trains each variant, logs accuracy/time, plots summary |

---

## Data Setup

### Handwritten digits (Task 1F)
Place your own digit photos in `data/handwritten/`. Name files by digit:
```
data/handwritten/0.png  1.png  2.png  ... 9.png
```
Use thick lines (marker/sharpie), white background. Images are auto-resized to 28×28 and inverted.

### Greek letters (Task 3)
```
data/greek_letters/
    alpha/   *.png
    beta/    *.png
    gamma/   *.png
```
Images should be ~128×128. The `GreekTransform` handles all preprocessing.

---

## Output Files

All figures are saved to `outputs/` automatically:

| File | Generated by | Description |
|---|---|---|
| `task1a_sample_digits.png` | Task 1A | First 6 MNIST test digits |
| `task1c_training_curves.png` | Task 1C | Loss and accuracy per epoch |
| `task1e_predictions.png` | Task 1E | 3×3 grid with predicted labels |
| `task1f_handwritten.png` | Task 1F | Handwritten digits + predictions |
| `task2a_filters.png` | Task 2A | 10 conv1 filter weight visualisations |
| `task2b_filter_effects.png` | Task 2B | 10 filtered versions of first training image |
| `task3_training_curve.png` | Task 3 | Greek letter transfer learning loss |
| `task4_transformer_curves.png` | Task 4 | Transformer training curves |
| `task5_experiment_results.png` | Task 5 | Sweep results across all dimensions |

Saved models go to `models/`:
- `mnist_cnn.pth` — trained CNN (Task 1D)
- `greek_transfer.pth` — fine-tuned transfer model (Task 3)
- `mnist_transformer.pth` — trained transformer (Task 4)