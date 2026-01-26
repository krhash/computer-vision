### Extension 2: Additional Artistic Effects

\begin{figure}[H]
    \centering
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{ext_depth_focus.jpg}
        \caption{Depth focus (portrait mode)}
    \end{subfigure}
    \caption{Extension 2: Depth Focus Effect}
\end{figure}

% CAPTURE: Depth focus effect

\textbf{Depth Focus:} Portrait mode effect using depth map. Heavily blurs background while keeping focus region sharp. Calculates blur amount based on distance from target depth, creating smooth transition between sharp and blurred regions.

\subsection{Extension 3: Paper-Based Cartoon Implementation}

Implemented video cartoonization as a standalone application based on Winnemöller et al. (2006).

\begin{figure}[H]
    \centering
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{ext_cartoon_simple.jpg}
        \caption{Simple cartoon (vidDisplay)}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.45\textwidth}
        \includegraphics[width=\textwidth]{ext_cartoon_paper.jpg}
        \caption{Paper-based cartoon (cartoonApp)}
    \end{subfigure}
    \caption{Extension 3: Cartoon Effect Comparison}
\end{figure}

% CAPTURE: Side-by-side comparison of simple vs paper-based cartoon

\textbf{Separate Application:} Implemented as standalone \texttt{cartoonApp} executable for clean demonstration of the academic paper's algorithm without mixing with other effects.

\textbf{Algorithm Comparison:}

\begin{table}[H]
\centering
\begin{tabular}{|l|p{5cm}|p{5cm}|}
\hline
\textbf{Component} & \textbf{Simple (vidDisplay)} & \textbf{Paper (cartoonApp)} \\
\hline
Smoothing & Gaussian blur & Bilateral filter (edge-preserving) \\
\hline
Edge Detection & Sobel gradients & Difference-of-Gaussians (DoG) \\
\hline
Temporal Stability & None (flickering) & Exponential moving average \\
\hline
Edge Quality & Noisy, variable & Clean, consistent \\
\hline
Processing Speed & ~5-10 ms/frame & ~20-30 ms/frame \\
\hline
Video Stability & Moderate flickering & Stable, smooth transitions \\
\hline
\end{tabular}
\caption{Comparison of Cartoon Implementations}
\end{table}

\textbf{Key Differences:}
\begin{itemize}
    \item \textbf{Bilateral Filter:} Preserves edges while smoothing flat regions, creating characteristic cartoon appearance
    \item \textbf{DoG Edge Detection:} Produces cleaner, more consistent edges than Sobel gradients
    \item \textbf{Temporal Coherence:} Reduces frame-to-frame flickering through exponential moving average blending
    \item \textbf{Quality vs Speed:} Paper implementation prioritizes quality and stability over raw processing speed
\end{itemize}

\textbf{Implementation:} Created separate \texttt{cartoonVideo.hpp/cpp} class and \texttt{cartoonApp} executable. This separation allows focused# Video Special Effects - Computer Vision Project

**Author:** Krushna Sanjay Sharma  
**Date:** January 24, 2026  
**Course:** CS 5330 - Computer Vision

## Project Overview

This project implements a real-time video processing application with various image filters and special effects. The application captures live video from a camera and applies filters including edge detection, blur, artistic effects, face detection, and animated sparkle effects.

## Features

### Basic Filters (Tasks 1-9)
- **Custom Greyscale** - Unique greyscale conversion emphasizing color differences
- **Sepia Tone** - Antique photo effect with optional vignetting
- **Gaussian Blur** - Both naive and optimized (separable) implementations
- **Sobel Edge Detection** - X, Y, and magnitude for comprehensive edge detection
- **Blur & Quantize** - Posterization effect with color quantization

### Face Detection & Effects (Task 10-12)
- **Face Detection** - Real-time face detection using Haar cascades
- **Face Highlight** - Color faces with grayscale background
- **Face Bulge** - Caricature "big head" effect on detected faces

### Depth-Based Effects (Task 11-12)
- **Depth Estimation** - Using Depth Anything V2 neural network (optional)
- **Depth Fog** - Atmospheric fog based on distance
- **Depth Focus** - Portrait mode with background blur

### Artistic Effects (Task 12)
- **Emboss** - 3D relief appearance
- **Negative** - Color inversion
- **Cartoon** - Edge detection + color quantization

### Warp Effects (Extensions)
- **Bulge** - Fisheye/pinch distortion
- **Wave** - Sinusoidal ripple effect
- **Swirl** - Rotating twirl distortion
- **Face Bulge** - Per-face distortion effect

### Sparkle Animation (Extension)
- **Sparkles** - Animated magical particles orbiting faces in 3D
- Features depth sorting, glow effects, and smooth animation
- Sparkles appear to move in front of and behind faces

## File Structure

```
project/
├── src/
│   ├── imgDisplay.cpp       # Static image display (Task 1)
│   ├── vidDisplay.cpp       # Live video with filters (Tasks 2-12)
│   ├── filters.cpp          # Filter implementations
│   ├── filters.hpp          # Filter declarations
│   ├── faceDetect.cpp       # Face detection functions
│   ├── faceDetect.h         # Face detection header
│   └── DA2Network.hpp       # Depth network wrapper (optional)
├── data/
|   ├──haarcascade_frontalface_alt2.xml  # Face detection model
|   ├── model_fp16.onnx          # Depth estimation model (optional)
└── CMakeLists.txt           # Build configuration
```

## Building the Project

### Requirements
- C++11 or later
- OpenCV 4.x
- CMake 3.10+
- Optional: ONNX Runtime with CUDA for depth estimation

### Build Commands

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release

# Or using make
make
```

### Optional: Enable Depth Estimation

```bash
cmake -DUSE_ONNXRUNTIME=ON ..
```

## Usage

### Static Image Display

```bash
./imgDisplay <image_path>
```

**Controls:**
- `q/ESC` - Quit
- `s` - Save image
- `i` - Display image info
- `r` - Reset to original
- `g` - Convert to greyscale
- `+/-` - Adjust brightness

### Live Video Display

```bash
./vidDisplay [camera_index]
```

**Basic Modes:**
- `c` - Color (default)
- `g` - OpenCV Greyscale
- `h` - Custom Greyscale
- `t` - Sepia Tone
- `b` - Blur
- `x` - Sobel X (vertical edges)
- `y` - Sobel Y (horizontal edges)
- `m` - Gradient Magnitude
- `l` - Blur & Quantize
- `f` - Face Detection
- `d` - Depth Estimation

**Special Effects:**
- `1` - Depth Fog
- `2` - Emboss
- `3` - Negative
- `4` - Face Highlight
- `5` - Cartoon
- `6` - Depth Focus

**Warp Effects:**
- `7` - Bulge
- `8` - Wave
- `9` - Swirl
- `0` - Face Bulge
- `[` - **Sparkles**

**Adjustments:**
- `+/-` - Adjust effect strength or quantize levels

**Utilities:**
- `s` - Save current frame
- `i` - Display video info
- `p` - Show help
- `q/ESC` - Quit

## Technical Details

### Edge Detection
The Sobel filters compute image gradients:
- **Sobel X**: Detects vertical edges (horizontal gradient)
- **Sobel Y**: Detects horizontal edges (vertical gradient)
- **Magnitude**: Combines both using √(Gx² + Gy²)

The magnitude filter provides direction-independent edge detection, showing all edges regardless of orientation.

### Gaussian Blur Optimization
Two implementations provided:
1. **Naive**: Full 5x5 convolution (9 multiplications/pixel)
2. **Separable**: Two 1x3 passes (6 multiplications/pixel)

The separable version is ~33% faster while producing identical results.

### Sparkle Animation
The sparkle effect uses 3D mathematics:
- **Position**: x = r·cos(θ), y = r·sin(θ)·cos(φ), z = r·sin(θ)·sin(φ)·0.7
- **Depth Sorting**: Separates sparkles by z-coordinate
- **Rendering**: Back-to-front with depth-based scaling
- **Animation**: Time-based angle updates with individual speeds
- **Effects**: Glow layers, pulsing brightness, smooth transitions

### Performance Notes
- Separable filters are significantly faster for blur operations
- Sobel filters use signed short (16-bit) for gradient values
- Face detection uses half-resolution processing for speed
- Depth estimation requires GPU for real-time performance


## Algorithm References

- **Sobel Edge Detection**: Sobel, I., & Feldman, G. (1968)
- **Gaussian Blur**: Separable filters optimization
- **Image Warping**: Wolberg, G. "Digital Image Warping" (1990)
- **Face Detection**: Viola-Jones cascade classifier
- **Depth Estimation**: Depth Anything V2 (Yang et al., 2024)

## Known Limitations

- Sparkle effect requires face detection to be active
- Depth-based effects require ONNX Runtime build
- Some filters may be slower on high-resolution video
- Face detection works best with frontal faces


## License

Educational project for CS 5330 - Computer Vision

## Acknowledgments

- Face detection code adapted from Prof. Bruce Maxwell
- Depth Anything V2 network wrapper by Prof. Bruce Maxwell
- OpenCV community for excellent documentation