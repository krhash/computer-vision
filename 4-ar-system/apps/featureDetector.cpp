////////////////////////////////////////////////////////////////////////////////
// featureDetector.cpp - Feature Detector Application Entry Point
// Author:      Krushna Sanjay Sharma
// Description: Main application for Task 7. Runs SIFT feature detection
//              on a live webcam feed pointed at a dollar bill pattern.
//              A trackbar controls the max number of features detected.
//              Circle size shows feature scale, line shows orientation.
//
// Usage:
//   featureDetector.exe [cameraId] [maxFeatures] [contrastThreshold]
//
//   cameraId          : webcam index (default: 0)
//   maxFeatures       : max SIFT features (default: 500, 0 = unlimited)
//   contrastThreshold : SIFT sensitivity (default: 0.04, lower = more features)
//
// Controls:
//   Trackbar  - drag to adjust max features in real time
//   'q' / ESC - quit
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "FeatureDetector.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " AR Calibration System\n";
    std::cout << " SIFT Feature Detector App (Task 7)\n";
    std::cout << "========================================\n\n";

    int    cameraId          = 0;
    int    maxFeatures       = 500;
    double contrastThreshold = 0.04;

    if (argc >= 2)
    {
        try { cameraId = std::stoi(argv[1]); }
        catch (...) { std::cerr << "[WARN] Invalid cameraId, using 0\n"; }
    }
    if (argc >= 3)
    {
        try { maxFeatures = std::stoi(argv[2]); }
        catch (...) { std::cerr << "[WARN] Invalid maxFeatures, using 500\n"; }
    }
    if (argc >= 4)
    {
        try { contrastThreshold = std::stod(argv[3]); }
        catch (...) { std::cerr << "[WARN] Invalid contrastThreshold, using 0.04\n"; }
    }

    std::cout << "Camera ID           : " << cameraId          << "\n";
    std::cout << "Max features        : " << maxFeatures        << "\n";
    std::cout << "Contrast threshold  : " << contrastThreshold  << "\n\n";
    std::cout << "Point camera at a dollar bill or any richly textured pattern.\n";
    std::cout << "Drag the trackbar to control how many features are shown.\n\n";

    FeatureDetector detector(cameraId, maxFeatures, contrastThreshold);

    if (!detector.run())
    {
        std::cerr << "[FAILED] Feature detection did not complete.\n";
        return 1;
    }

    return 0;
}
