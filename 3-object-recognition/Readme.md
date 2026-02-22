# 2D Object Recognition System

**Author:** Krushna Sanjay Sharma  
**Language:** C++17  
**Platform:** Windows 10/11  

Real-time object recognition system that identifies objects placed on a uniform background using shape feature analysis and CNN embeddings. Supports live camera, image, and video input.

---

## Environment

| Dependency | Version | Location |
|-----------|---------|----------|
| Windows SDK | 10.0+ | System |
| MSVC | 2022 (v143) | Visual Studio Build Tools |
| CMake | 3.15+ | System PATH |
| OpenCV | 4.14 | `C:\lib\build_opencv` |
| ONNX Runtime | 1.x | `C:\lib\onnxruntime` |
| GLFW3 | 3.x | `C:\lib\glfw` |
| Dear ImGui | Latest | `third_party/imgui/` |

---

## Project Structure

```
ObjectRecognition/
├── CMakeLists.txt
├── build.bat
├── include/              # Header files
├── src/                  # Source files
├── third_party/imgui/    # Dear ImGui (git clone)
├── data/
│   ├── images/           # Test/dev images
│   ├── models/
│   │   └── resnet18-v2-7.onnx
│   └── db/
│       ├── objects.csv        # Shape feature training DB
│       ├── embeddings.csv     # CNN embedding training DB
│       └── confusion_matrix.csv
└── bin/Release/
    └── objectRecognition.exe
```

---

## Build

```bat
git clone https://github.com/ocornut/imgui third_party/imgui
build.bat
```

`build.bat` auto-detects OpenCV, ONNX Runtime, and GLFW. All are optional except OpenCV. Check the CMake output for:
```
-- GLFW found   : TRUE
-- ImGui found  : TRUE
-- ONNX Runtime : TRUE
```

---

## Running

```bat
cd bin\Release

# Live camera (default)
objectRecognition.exe --mode live --camera 0

# Single image
objectRecognition.exe --mode image --input ..\..\data\images\test.jpg

# Video file
objectRecognition.exe --mode live --input ..\..\data\video.mp4

# Custom DB path
objectRecognition.exe --mode live --db ..\..\data\db\objects.csv
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `live` | `live`, `image`, `train`, `eval`, `embed` |
| `--camera` | `0` | Camera index for live mode |
| `--input` | — | Path to image or video file |
| `--db` | `data/db/objects.csv` | Path to shape feature DB |

---

## GUI

Built with **Dear ImGui** (GLFW + OpenGL3 backend). Opens as a separate control window alongside the OpenCV video windows.

### Panels

**Pipeline** *(open by default)*
- Threshold value, blur kernel, threshold mode (Global / ISODATA / Sat+Intensity)
- Morphology mode (Open/Close/Erode/Dilate), kernel size, iterations
- Min region area, max regions
- Confidence threshold, K neighbours, distance metric
- Window visibility checkboxes: Threshold, Cleaned, Regions, Crop, Axes, BBox, Features, Overlay

**Training** *(open by default)*
- Type label name → click **Set**
- **+ Shape Sample** — captures shape features to `objects.csv`
- **+ Embedding Sample** — captures CNN embedding to `embeddings.csv`
- CNN Mode checkbox — toggles between shape feature and CNN classifiers

**Databases** *(collapsed)*
- **Shape Feature DB** — table of labels + sample counts + delete per label
- **Embedding DB (CNN)** — table of labels + sample counts

**Confusion Matrix** *(collapsed)*
- Color-coded matrix (green = correct, red = incorrect)
- Overall accuracy %, Save CSV, Clear buttons

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `t / T` | Threshold -/+5 |
| `b / B` | Blur kernel -/+2 |
| `k` | Toggle ISODATA threshold |
| `s` | Toggle Sat+Intensity threshold |
| `m / M` | Morph kernel -/+2 |
| `i / I` | Morph iterations -/+1 |
| `o` | Cycle morph mode |
| `r / R` | Min region area -/+100 |
| `n` | Set training label (terminal prompt) |
| `c` | Capture shape feature sample |
| `C` | Capture CNN embedding sample |
| `x / X` | Confidence threshold -/+0.1 |
| `d` | Toggle Euclidean / Cosine metric |
| `j / J` | K neighbours -/+1 |
| `e` | Record evaluation sample (terminal prompt) |
| `p` | Print + save confusion matrix |
| `1` | Toggle Threshold window |
| `2` | Toggle Cleaned window |
| `3` | Toggle Regions window |
| `4` | Toggle axes overlay |
| `5` | Toggle oriented bounding box |
| `6` | Toggle feature text |
| `7` | Toggle confusion matrix window |
| `8` | Toggle CNN crop debug windows |
| `9` | Toggle CNN / Shape Feature mode |
| `q / ESC` | Quit |

---

## Training Steps

### Shape Features (classical classifier)

1. Place one object on white background under camera
2. Verify object is detected — bounding box visible in main window
3. In GUI Training panel, type label name → click **Set**
4. Click **+ Shape Sample** 5–10 times in different positions/orientations
5. Repeat for each object
6. Data saved to: `data/db/objects.csv`

### CNN Embeddings (one-shot classifier)

1. Set label as above
2. Click **+ Embedding Sample** 1–3 times (at 0°, 90°, 180° orientations)
3. Press `9` to switch to CNN mode — labels appear from embeddings
4. Press `8` to verify crop window shows correct isolated object
5. Data saved to: `data/db/embeddings.csv`

> **Note:** Delete and retrain if you change threshold/blur/morph settings — feature vectors must be captured under the same pipeline settings used at runtime.

---

## Data Files

### `data/db/objects.csv`
Shape feature training database. One row per captured sample.
```
label,fillRatio,bboxRatio,hu0,hu1,hu2,hu3,hu4,hu5,hu6
scissors,0.823,3.241,-1.452,-4.123,-6.234,-7.891,...
```

### `data/db/embeddings.csv`
CNN embedding training database. One row per captured sample.
```
label,e0,e1,...,e511
scissors,0.231,-1.423,0.891,...
```
Each row contains 512 float values from ResNet18's penultimate layer.

### `data/db/confusion_matrix.csv`
Evaluation results. Generated by pressing `p` or clicking Save in GUI.
```
true/predicted,scissors,hat,...,accuracy
scissors,5,0,...,1.00
hat,0,4,...,0.80
```

---

## Pipeline Overview

```
Camera / Image
      │
      ▼
Threshold      → binary mask (object white, background black)
      │
      ▼
Morphology     → clean mask (remove noise, fill holes)
      │
      ▼
Connected      → labeled regions (two-pass algorithm — from scratch)
Components
      │
      ▼
Region         → shape features per region
Features         (fillRatio, bboxRatio, 7 Hu moments)
      │
      ├──► Shape Classifier  → scaled Euclidean KNN → label
      │
      └──► CNN Classifier    → ResNet18 → 512-dim embedding → SSD → label
```

### From-Scratch Implementations
- **Morphology** — erosion and dilation using raw pixel loops
- **Connected Components** — two-pass algorithm with Union-Find data structure

---