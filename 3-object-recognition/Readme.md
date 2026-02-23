# 2D Object Recognition System

**Author:** Krushna Sanjay Sharma  
**Language:** C++17  
**Platform:** Windows 10/11  

---

## License

MIT License

Copyright (c) 2026 Krushna Sanjay Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Attribution

Portions of this project are modified from third-party sources:

- **`src/utilities.cpp`** — Modified from course utilities code by Prof. Bruce Maxwell (CS 5330). Original functions: `prepEmbeddingImage()`, `getEmbedding()`.
- **Dear ImGui** — Copyright (c) 2014-2024 Omar Cornut. MIT License. https://github.com/ocornut/imgui
- **GLFW** — Copyright (c) 2002-2006 Marcus Geelnard, 2006-2019 Camilla Löwy. zlib License. https://www.glfw.org
- **OpenCV** — Apache 2.0 License. https://opencv.org
- **ONNX Runtime** — MIT License. https://github.com/microsoft/onnxruntime
- **ResNet18 model** — ONNX Model Zoo. Apache 2.0 License. https://github.com/onnx/models

---

Real-time object recognition system that identifies objects placed on a uniform background using shape feature analysis and CNN embeddings. Supports live camera, image, and video input.

---

## Dependencies — Installation

### OpenCV
1. Download pre-built Windows binaries from https://opencv.org/releases
2. Extract and build, or use pre-built:
   - Headers + libs at: `C:\lib\build_opencv`
   - Confirm: `C:\lib\build_opencv\OpenCVConfig.cmake` exists
3. Add OpenCV DLLs to PATH or copy to `bin\Release\`:
   ```bat
   set PATH=%PATH%;C:\lib\build_opencv\bin\Release
   ```

### ONNX Runtime *(optional — required for CNN embeddings)*
1. Download from https://github.com/microsoft/onnxruntime/releases
2. Extract to `C:\lib\onnxruntime\`
3. Confirm layout:
   ```
   C:\lib\onnxruntime\include\onnxruntime_cxx_api.h
   C:\lib\onnxruntime\lib\onnxruntime.lib
   C:\lib\onnxruntime\lib\onnxruntime.dll
   ```

### GLFW3 *(optional — required for ImGui GUI)*
1. Download pre-built Windows binaries from https://www.glfw.org/download
2. Extract to `C:\lib\glfw\`
3. Confirm layout:
   ```
   C:\lib\glfw\include\GLFW\glfw3.h
   C:\lib\glfw\lib-vc2022\glfw3.lib
   ```

### Dear ImGui *(optional — required for GUI)*
```bat
git clone https://github.com/ocornut/imgui third_party/imgui
```

### ResNet18 ONNX Model *(optional — required for CNN embeddings)*
1. Download `resnet18-v2-7.onnx` from https://github.com/onnx/models
2. Place at: `data/models/resnet18-v2-7.onnx`

---

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
- Morphology mode, kernel size, iterations

| Mode | Operations | Use when |
|------|-----------|----------|
| **Open** | Erode → Dilate | Speckle noise around object |
| **Close** | Dilate → Erode | Holes or gaps inside object |
| **Erode** | Shrinks foreground | Separate nearly-touching objects |
| **Dilate** | Grows foreground | Object appears fragmented |
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
