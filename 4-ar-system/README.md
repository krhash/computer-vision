# AR Calibration System
**Author:** Krushna Sanjay Sharma  
**Date:** March 2026  
**Course:** Computer Vision — Project 4

---

## Overview

A complete Camera Calibration and Augmented Reality system built in C++ using OpenCV. The system detects a chessboard target, calibrates the camera, estimates pose in real time using `solvePnP`, and renders 3D virtual objects (rocket, eagle) on physical targets. Extends to SIFT-based AR on a dollar bill without a chessboard target.

---

## System Environment

| Component | Details |
|---|---|
| OS | Windows 10/11 (x64) |
| Compiler | MSVC 2022 (Visual Studio Build Tools 17.x) |
| Build System | CMake 3.15+ |
| Language Standard | C++17 |
| OpenCV | 4.x (built from source at `C:\lib\build_opencv`) |
| Camera | Any USB webcam or built-in camera (device ID 0) |
| Target patterns | 9×6 internal corner chessboard, US dollar bill |

---

## Project Structure

```
4-ar-system/
├── include/                    # Class headers
│   ├── CameraCalibration.h     # Tasks 1-3: chessboard detection + calibration
│   ├── PoseEstimator.h         # Tasks 4-5: solvePnP + projectPoints
│   ├── VirtualObject.h         # Task 6: 3D rocket line object
│   ├── VirtualEagleObject.h    # Extension: 3D eagle line object (dollar bill)
│   ├── SIFTTracker.h           # Uber Ext 2: SIFT feature matching + pose
│   ├── FeatureDetector.h       # Task 7: SIFT keypoint visualization
│   └── Utils.h                 # Shared helpers
│
├── src/                        # Class implementations
│   ├── CameraCalibration.cpp
│   ├── PoseEstimator.cpp
│   ├── VirtualObject.cpp
│   ├── VirtualEagleObject.cpp
│   ├── SIFTTracker.cpp
│   └── FeatureDetector.cpp
│
├── apps/                       # Application entry points
│   ├── calibrateCamera.cpp     # App 1: Tasks 1-3
│   ├── augmentedReality.cpp    # App 2: Tasks 4-6
│   ├── featureDetector.cpp     # App 3: Task 7
│   ├── siftAR.cpp              # App 4: Uber Extension 2
│   └── multiTargetAR.cpp       # App 5: Multi-target extension
│
├── data/
│   └── calibration/
│       └── calibration.xml     # Saved camera intrinsics (generated at runtime)
│
├── bin/Release/                # Compiled executables (generated)
├── lib/Release/                # Static library ar_core (generated)
├── build/                      # CMake build files (generated)
├── CMakeLists.txt
└── build.bat
```

---

## Core Library — `ar_core`

All classes compile into a single static library `ar_core.lib` which all executables link against.

### `CameraCalibration`
Manages the full calibration pipeline.
- Detects internal chessboard corners using `cv::findChessboardCorners`
- Refines to sub-pixel accuracy with `cv::cornerSubPix`
- Collects calibration frames (user presses `s`)
- Runs `cv::calibrateCamera` to compute the 3×3 camera matrix and distortion coefficients
- Saves intrinsic parameters to an XML file via `cv::FileStorage`
- **World convention:** `(col, -row, 0)` — +Y away from board, +Z toward camera

### `PoseEstimator`
Estimates the chessboard's 6-DOF pose each frame.
- Loads calibration XML written by `CameraCalibration`
- Detects chessboard corners each frame (reuses same pipeline as calibration)
- Calls `cv::solvePnP` (ITERATIVE method) to compute `rvec` and `tvec`
- Projects outer board corners and 3D axes onto the image using `cv::projectPoints`
- Applies exponential moving average smoothing to reduce pose jitter
- Supports three display modes: axes only, corners only, corners + axes

### `VirtualObject`
Defines and renders a 3D wireframe rocket above the chessboard.
- Geometry defined as `LineSegment` pairs in world space (square units)
- Five parts: body (blue), nose cone (cyan), fins (amber), exhaust (red), window (green)
- `buildRocket(offset, scale, flipY, flipZ)` — supports any target coordinate system
- `draw()` collects all 3D endpoints, calls `cv::projectPoints` once, draws with `cv::line`

### `VirtualEagleObject`
Defines and renders a 3D wireframe eagle above the dollar bill.
- Standalone class — independent of `VirtualObject`
- Four parts: body (blue), head + beak (dark blue + amber), wings (light blue), tail (amber)
- `build(offset, scale, flipY, flipZ)` — `flipY=true, flipZ=true` for bill convention
- Same `draw()` batch projection pattern as `VirtualObject`

### `SIFTTracker`
Tracks a dollar bill using SIFT feature matching instead of a chessboard.
- Loads reference image of the bill and computes SIFT keypoints + 128-dim descriptors once
- Applies CLAHE contrast enhancement to both reference and live frames
- Each frame: detects SIFT, matches with `cv::BFMatcher` + Lowe's ratio test (0.75)
- Filters outliers with `cv::findHomography` RANSAC (5.0px threshold)
- Maps inlier 2D reference points to 3D world points in cm on the bill surface
- Runs `cv::solvePnP` for pose; applies EMA smoothing (α=0.35) for stability
- **Bill world convention:** `(xCm, yCm, 0)` — +Y downward, +Z toward camera

### `FeatureDetector`
Visualizes SIFT keypoints in a live video stream.
- Caches `cv::SIFT` detector — only recreates when trackbar value changes
- `detectSIFT()` draws keypoints with `DRAW_RICH_KEYPOINTS`: circle size = scale, line = orientation
- Trackbar controls max feature count in real time

---

## Applications

### `calibrateCamera.exe` — Tasks 1–3
**Purpose:** Calibrate the camera using a physical chessboard target.

**Usage:**
```
calibrateCamera.exe [cameraId] [outputFile]
```

**Controls:**

| Key | Action |
|---|---|
| `s` | Save current frame (only when green corners visible) |
| `c` | Run calibration (requires ≥ 5 saved frames) |
| `q` / ESC | Quit |

**Output:** `bin/Release/data/calibration/calibration.xml` containing `camera_matrix` and `distortion_coefficients`. Run this once per camera before using any other app.

---

### `augmentedReality.exe` — Tasks 4–6
**Purpose:** Detect the chessboard in real time, estimate its pose, and render a 3D rocket above it.

**Usage:**
```
augmentedReality.exe [calibrationFile] [cameraId]
```

**Controls:**

| Key | Action |
|---|---|
| `SPACE` | Cycle display mode: Axes → Corners → Corners+Axes |
| `r` | Toggle rocket on/off |
| `q` / ESC | Quit |

**Display:** Live `tvec` and `rvec` values overlaid on frame. Red Z axis points toward camera, green Y axis points away from board, blue X axis points right.

---

### `featureDetector.exe` — Task 7
**Purpose:** Detect and visualize SIFT features on a textured pattern (dollar bill or any richly textured surface).

**Usage:**
```
featureDetector.exe [cameraId] [maxFeatures] [contrastThreshold]
```

**Controls:**

| Key | Action |
|---|---|
| Trackbar | Adjust max features in real time |
| `q` / ESC | Quit |

**Display:** Orange circles on detected keypoints. Circle size = feature scale, line = orientation. Feature count shown in top-left. Point at a dollar bill or any richly textured pattern.

---

### `siftAR.exe` — Uber Extension 2
**Purpose:** AR on a dollar bill using SIFT feature matching — no chessboard required. Uses the same camera calibration from `calibrateCamera.exe`.

**Usage:**
```
siftAR.exe [calibrationFile] [referenceImage] [cameraId]
```

**Setup:** Place a flat photo of the dollar bill at `bin/Release/data/bill.jpg`. Take the photo directly overhead with even lighting for best results.

**Controls:**

| Key | Action |
|---|---|
| `r` | Toggle rocket on/off |
| `d` | Toggle SIFT debug keypoints |
| `q` / ESC | Quit |

**Display:** Inlier count shown live. Tracking confirmed when ≥ 8 inliers detected.

---

### `multiTargetAR.exe` — Multi-Target Extension
**Purpose:** Track chessboard and dollar bill simultaneously in the same scene. Renders a rocket on the chessboard and an eagle on the dollar bill independently.

**Usage:**
```
multiTargetAR.exe [calibrationFile] [billImage] [cameraId]
```

**Controls:**

| Key | Action |
|---|---|
| `r` | Toggle rocket (chessboard) |
| `e` | Toggle eagle (dollar bill) |
| `d` | Toggle SIFT debug keypoints |
| `SPACE` | Cycle chessboard display mode |
| `q` / ESC | Quit |

**Display:** Two status lines at top — `Chess: tracked/searching` and `Bill: tracked (N inliers)/searching`. Each target tracks independently; objects appear only when their respective target is detected.

---

## Build Instructions

### Prerequisites
- OpenCV built at `C:\lib\build_opencv` (with `OpenCVConfig.cmake` present)
- CMake 3.15+
- Visual Studio Build Tools 2022

### Build
```batch
build.bat
```
This configures with CMake and builds all targets in Release mode. Executables land in `bin\Release\`.

### Manual build
```batch
mkdir build
cd build
cmake .. -DOpenCV_DIR=C:\lib\build_opencv -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### OpenCV DLLs
Add OpenCV to PATH or copy DLLs to `bin\Release\`:
```batch
set PATH=%PATH%;C:\lib\build_opencv\bin\Release
```

---

## Output Artifacts

| File | Description |
|---|---|
| `bin/Release/calibrateCamera.exe` | Camera calibration app |
| `bin/Release/augmentedReality.exe` | Chessboard AR app |
| `bin/Release/featureDetector.exe` | SIFT feature visualizer |
| `bin/Release/siftAR.exe` | Dollar bill SIFT AR app |
| `bin/Release/multiTargetAR.exe` | Multi-target AR app |
| `lib/Release/ar_core.lib` | Static library (all classes) |
| `bin/Release/data/calibration/calibration.xml` | Camera intrinsics (runtime) |

---

## Recommended Workflow

```
1. Run calibrateCamera.exe
   → Collect ≥ 5 frames from different angles
   → Press 'c' to calibrate
   → Verify reprojection error < 1.0 pixel

2. Run augmentedReality.exe
   → Point at chessboard
   → Verify axes orientation (Z toward camera)
   → Verify rocket floats above board

3. Run featureDetector.exe
   → Point at dollar bill
   → Adjust trackbar to ~50-100 features
   → Observe keypoint locations

4. Place bill.jpg in bin/Release/data/
   → Run siftAR.exe
   → Point at dollar bill
   → Verify ≥ 8 inliers for stable tracking

5. Run multiTargetAR.exe
   → Place chessboard AND dollar bill in view
   → Both objects appear simultaneously
```

---

## Key OpenCV Functions Used

| Function | Used in | Purpose |
|---|---|---|
| `cv::findChessboardCorners` | CameraCalibration, PoseEstimator | Detect internal corners |
| `cv::cornerSubPix` | CameraCalibration, PoseEstimator | Sub-pixel corner refinement |
| `cv::calibrateCamera` | CameraCalibration | Compute intrinsic matrix |
| `cv::solvePnP` | PoseEstimator, SIFTTracker | 6-DOF pose from 3D-2D correspondences |
| `cv::projectPoints` | PoseEstimator, VirtualObject, VirtualEagleObject | Project 3D points to image |
| `cv::SIFT::create` | SIFTTracker, FeatureDetector | SIFT feature detection |
| `cv::BFMatcher` | SIFTTracker | Descriptor matching |
| `cv::findHomography` | SIFTTracker | RANSAC outlier rejection |
| `cv::FileStorage` | CameraCalibration, PoseEstimator | Read/write calibration XML |
| `cv::createCLAHE` | SIFTTracker | Contrast enhancement |
