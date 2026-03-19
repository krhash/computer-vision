////////////////////////////////////////////////////////////////////////////////
// augmentedReality.cpp - Augmented Reality Application Entry Point
// Author:      Krushna Sanjay Sharma
// Description: Main application for Tasks 4-6. Currently implements Task 4:
//              loads camera calibration, detects the chessboard target each
//              frame, and uses solvePnP to compute and print the board's
//              real-time rotation and translation vectors.
//
//              Tasks 5 and 6 (projectPoints, virtual object) will be added
//              to this same app in subsequent tasks.
//
// Usage:
//   augmentedReality.exe [calibrationFile] [cameraId]
//
//   calibrationFile : path to calibration XML (default: uses exe-relative path)
//   cameraId        : webcam index (default: 0)
//
// Controls:
//   'q' / ESC - quit
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "PoseEstimator.h"
#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " AR Calibration System\n";
    std::cout << " Augmented Reality App (Tasks 4-6)\n";
    std::cout << "========================================\n\n";

    // ── Resolve calibration file path relative to the executable ─────────────
    // Mirrors the same approach used in calibrateCamera.cpp so both apps
    // read/write from the same location automatically.
    std::filesystem::path exeDir =
        std::filesystem::path(argv[0]).parent_path();

    std::string calibFile =
        (exeDir / "data" / "calibration" / "calibration.xml").string();
    int cameraId = 0;

    // Allow overrides from command line
    if (argc >= 2) calibFile  = argv[1];
    if (argc >= 3)
    {
        try   { cameraId = std::stoi(argv[2]); }
        catch (...) { std::cerr << "[WARN] Invalid camera ID, using 0.\n"; }
    }

    std::cout << "Calibration file : " << calibFile << "\n";
    std::cout << "Camera ID        : " << cameraId  << "\n\n";

    // ── Run Task 4: pose estimation ───────────────────────────────────────────
    PoseEstimator estimator(calibFile, cameraId);

    if (!estimator.run())
    {
        std::cerr << "[FAILED] Pose estimation did not complete.\n";
        return 1;
    }

    return 0;
}
