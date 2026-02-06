# Video Special Effects

**Author:** Krushna Sanjay Sharma  
**Date:** January 2026  
**Course:** CS 5330 Computer Vision

A real-time video processing application featuring image filters, face detection, depth estimation, and creative visual effects implemented in C++ with OpenCV.

---

## Video Demonstrations

| Demo | Link |
|------|------|
| **Video Filters (All Effects)** | [Watch on Google Drive](https://drive.google.com/file/d/1XBqkbhCQQOuZFDnaUM-dyIKFYuknacvz/view?usp=drive_link) |
| **Image Display Application** | [Watch on Google Drive](https://drive.google.com/file/d/1eT8P9yf3R84965zDIJr1GJgub5-5JJXR/view?usp=drive_link) |
| **Cartoon App (Paper-Based)** | [Watch on Google Drive](https://drive.google.com/file/d/12Np8GFC_UBf6tMNjAvD3sduPNuTfoWt3/view?usp=drive_link) |
| **Sparkles Animation Effect** | [Watch on Google Drive](https://drive.google.com/file/d/1as42R9Wdcs_gLnH74BZ8di4EoOZUC-V_/view?usp=drive_link) |

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Building the Project](#building-the-project)
- [Running the Applications](#running-the-applications)
- [Application Controls](#application-controls)
- [Task Descriptions](#task-descriptions)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a comprehensive suite of real-time video processing effects, progressing from basic image operations to advanced computer vision techniques including face detection and monocular depth estimation. The application processes live webcam feed and applies various filters and effects in real-time.

---

## Features

### Core Filters
- Custom greyscale conversion with channel difference method
- Sepia tone with optional vignetting
- 5x5 Gaussian blur (naive and optimized separable implementations)
- Sobel edge detection (X and Y gradients)
- Gradient magnitude computation
- Color quantization with blur

### Face Detection
- Haar cascade-based face detection
- Face region highlighting
- Face-centered special effects

### Depth Estimation (GPU Accelerated)
- Depth Anything V2 neural network integration
- Real-time monocular depth estimation
- Depth-based fog effect
- Portrait mode (depth-based focus blur)

### Creative Effects
- Emboss (3D relief effect)
- Negative/inverse
- Cartoon rendering
- Bulge/fisheye distortion
- Wave ripple effect
- Swirl/twirl distortion
- Face bulge caricature
- Animated sparkles around faces

### Additional Features
- Real-time video capture from webcam
- Frame saving with timestamps
- Adjustable effect parameters
- Temporal smoothing for video stability

---

## Project Structure

```
VisionProject/
├── CMakeLists.txt          # CMake build configuration
├── build.bat               # Windows build script
├── README.md               # This file
├── include/
│   ├── filters.hpp         # Filter function declarations
│   ├── faceDetect.h        # Face detection declarations
│   ├── cartoonVideo.hpp    # Cartoon effect class
│   └── DA2Network.hpp      # Depth estimation network wrapper
├── src/
│   ├── imgDisplay.cpp      # Image display application (Task 1)
│   ├── vidDisplay.cpp      # Video display application (Tasks 2-12+)
│   ├── filters.cpp         # Filter implementations
│   ├── faceDetect.cpp      # Face detection implementation
│   ├── cartoonApp.cpp      # Standalone cartoon application
│   └── cartoonVideo.cpp    # Cartoon effect implementation
├── data/
│   ├── haarcascade_frontalface_alt2.xml  # Face detection model
│   ├── model_fp16.onnx     # Depth Anything V2 model
│   └── images/             # Test images
├── cmake/
│   └── FindONNXRuntime.cmake  # ONNX Runtime finder
├── bin/                    # Output executables
└── lib/                    # Output libraries
```

---

## Dependencies

### Required
| Dependency | Version | Purpose |
|------------|---------|---------|
| OpenCV | 4.x | Core image processing |
| CMake | 3.15+ | Build system |
| Visual Studio | 2019/2022 | C++ compiler (Windows) |

### Optional (for GPU-accelerated depth estimation)
| Dependency | Version | Purpose |
|------------|---------|---------|
| ONNX Runtime | 1.16+ | Neural network inference |
| CUDA Toolkit | 11.x/12.x | GPU acceleration |
| cuDNN | 8.x | Deep learning primitives |

### Windows Installation Paths (Expected)
```
C:\lib\build_opencv\       # OpenCV build directory
C:\lib\install\            # OpenCV install directory
C:\lib\onnxruntime\        # ONNX Runtime
C:\lib\cudnn\bin\x64\      # cuDNN DLLs
```

---

## Building the Project

### Using the Build Script (Recommended)

The `build.bat` script automates the build process on Windows:

```batch
# Run from project root directory
.\build.bat
```

The script performs the following:
1. Verifies OpenCV installation at `C:\lib\build_opencv`
2. Locates `OpenCVConfig.cmake`
3. Creates build directory
4. Runs CMake configuration
5. Compiles all targets
6. Reports DLL locations for runtime

### Manual Build with CMake

```batch
# Create and enter build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DOpenCV_DIR=C:\lib\build_opencv -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release
```

### Build Targets

| Target | Description |
|--------|-------------|
| `imgDisplay` | Static image viewer with basic controls |
| `vidDisplay` | Real-time video processor with all effects |
| `cartoonApp` | Standalone cartoon video application |
| `filters` | Static library containing all filter functions |

### CMake Configuration Options

```cmake
# Set custom OpenCV path
-DOpenCV_DIR=<path_to_opencv>

# Set custom ONNX Runtime path
-DONNXRUNTIME_ROOT=<path_to_onnxruntime>

# Build type
-DCMAKE_BUILD_TYPE=Release  # or Debug
```

---

## Running the Applications

### Setting Up Runtime Environment

Before running, ensure OpenCV DLLs are accessible:

```batch
# Option 1: Add to PATH
set PATH=%PATH%;C:\lib\build_opencv\bin\Release

# Option 2: Copy DLLs to executable directory
copy C:\lib\build_opencv\bin\Release\*.dll bin\Release\
```

### imgDisplay - Image Viewer

```batch
cd bin\Release
imgDisplay.exe <image_path>

# Example
imgDisplay.exe ..\..\data\images\test.jpg
```

### vidDisplay - Video Processor

```batch
cd bin\Release
vidDisplay.exe [camera_index]

# Default camera (index 0)
vidDisplay.exe

# Specific camera
vidDisplay.exe 1
```

### cartoonApp - Cartoon Video

```batch
cd bin\Release
cartoonApp.exe [camera_index]
```

---

## Application Controls

### imgDisplay Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit application |
| `s` | Save current image |
| `i` | Display image information |
| `r` | Reset to original image |
| `g` | Convert to greyscale |
| `h` | Show help menu |
| `+` / `=` | Increase brightness |
| `-` / `_` | Decrease brightness |

### vidDisplay Controls

#### General Controls
| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit application |
| `s` | Save current frame |
| `i` | Display video information |
| `p` | Print help menu |

#### Display Modes
| Key | Mode | Task |
|-----|------|------|
| `c` | Color (default) | Task 2 |
| `g` | OpenCV Greyscale | Task 3 |
| `h` | Custom Greyscale | Task 4 |
| `t` | Sepia Tone | Task 5 |
| `b` | Gaussian Blur | Task 6 |
| `x` | Sobel X (vertical edges) | Task 7 |
| `y` | Sobel Y (horizontal edges) | Task 7 |
| `m` | Gradient Magnitude | Task 8 |
| `l` | Blur & Quantize | Task 9 |
| `f` | Face Detection | Task 10 |
| `d` | Depth Estimation | Task 11 |

#### Custom Effects (Task 12)
| Key | Effect |
|-----|--------|
| `1` | Depth Fog |
| `2` | Emboss |
| `3` | Negative |
| `4` | Face Highlight |
| `5` | Cartoon |
| `6` | Depth Focus (Portrait Mode) |

#### Extension Effects
| Key | Effect |
|-----|--------|
| `7` | Bulge Warp |
| `8` | Wave Warp |
| `9` | Swirl Warp |
| `0` | Face Bulge |
| `[` | Sparkles |

#### Adjustments
| Key | Action |
|-----|--------|
| `+` / `=` | Increase effect strength / quantize levels |
| `-` / `_` | Decrease effect strength / quantize levels |

---

## Task Descriptions

### Task 1: Image Display (`imgDisplay`)
Basic image loading and display with OpenCV. Supports keyboard controls for brightness adjustment, greyscale conversion, and image saving.

### Task 2: Video Capture (`vidDisplay`)
Real-time video capture from webcam with display window. Foundation for all subsequent video processing tasks.

### Task 3: OpenCV Greyscale
Standard greyscale conversion using OpenCV's `cvtColor` with `COLOR_BGR2GRAY`.

### Task 4: Custom Greyscale
Unique greyscale algorithm using channel difference method:
```
grey = |R - B| + G/2
```
Emphasizes color boundaries and creates a distinctive look.

### Task 5: Sepia Tone
Classic sepia transformation with optional vignetting effect:
```
R' = 0.393*R + 0.769*G + 0.189*B
G' = 0.349*R + 0.686*G + 0.168*B
B' = 0.272*R + 0.534*G + 0.131*B
```

### Task 6: Gaussian Blur
Two implementations of 5x5 Gaussian blur:
- **Naive:** Direct 2D convolution (25 operations per pixel)
- **Optimized:** Separable 1D filters (10 operations per pixel)

### Task 7: Sobel Edge Detection
Separable Sobel filters for edge detection:
- **Sobel X:** Detects vertical edges (horizontal brightness changes)
- **Sobel Y:** Detects horizontal edges (vertical brightness changes)

### Task 8: Gradient Magnitude
Combines Sobel X and Y using Euclidean distance:
```
magnitude = sqrt(sobelX² + sobelY²)
```

### Task 9: Blur and Quantize
Color posterization effect combining Gaussian blur with uniform quantization.

### Task 10: Face Detection
Haar cascade-based face detection with bounding box visualization.

### Task 11: Depth Estimation
Real-time monocular depth estimation using Depth Anything V2 neural network with ONNX Runtime and CUDA acceleration.

### Task 12: Custom Effects
Multiple creative effects including depth fog, emboss, negative, face highlight, cartoon rendering, and depth-based focus blur.

### Extensions: Warp Effects
Advanced image warping effects:
- **Bulge:** Radial distortion using polar coordinates
- **Wave:** Sinusoidal displacement creating ripple patterns
- **Swirl:** Rotation decreasing with distance from center
- **Face Bulge:** Localized bulge effect on detected faces
- **Sparkles:** Animated 3D particles orbiting faces

---

## Troubleshooting

### OpenCV Not Found
```
Error: C:\lib\build_opencv not found
```
**Solution:** Verify OpenCV is installed at `C:\lib\build_opencv` or update the path in `build.bat`.

### Missing DLLs at Runtime
```
The code execution cannot proceed because opencv_world4xx.dll was not found
```
**Solution:** Add OpenCV bin directory to PATH or copy DLLs to executable location.

### Depth Estimation Not Working
```
Warning: Could not load depth network
```
**Solution:** 
1. Verify `model_fp16.onnx` exists in `data/` directory
2. Check ONNX Runtime installation
3. Verify CUDA and cuDNN are properly installed

### Face Detection Failing
```
Unable to load face cascade file
```
**Solution:** Ensure `haarcascade_frontalface_alt2.xml` is in the executable directory or `data/` folder.

### Build Fails with ONNX Runtime
**Solution:** ONNX Runtime is optional. The project will build without it, but depth estimation features will be disabled.

---

## References

- OpenCV Documentation: https://docs.opencv.org/
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Winnemoller et al. (2006) "Real-time video abstraction" - Cartoon effect algorithm
- ONNX Runtime: https://onnxruntime.ai/

---

## License

This project is for educational purposes as part of CS 5330 Computer Vision coursework.