////////////////////////////////////////////////////////////////////////////////
// multiTargetAR.cpp - Multi-Target Augmented Reality Application
// Author:      Krushna Sanjay Sharma
// Description: Tracks two targets simultaneously in the same scene:
//              1. Chessboard → 3D Rocket  (PoseEstimator + VirtualObject)
//              2. Dollar bill → 3D Eagle  (SIFTTracker + VirtualEagleObject)
//
//              Both targets are always active. Place chessboard and dollar
//              bill in the same camera view to see both AR objects at once.
//
// Extension:   Multiple targets in the scene
//
// Usage:
//   multiTargetAR.exe [calibrationFile] [billImage] [cameraId]
//
// Controls:
//   'r'     - toggle rocket visibility (chessboard)
//   'e'     - toggle eagle visibility  (dollar bill)
//   'd'     - toggle SIFT debug keypoints
//   SPACE   - cycle chessboard display mode (axes/corners/both)
//   'q'/ESC - quit
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "PoseEstimator.h"
#include "SIFTTracker.h"
#include "VirtualObject.h"
#include "VirtualEagleObject.h"
#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " AR Calibration System\n";
    std::cout << " Multi-Target AR\n";
    std::cout << "   Chessboard → Rocket\n";
    std::cout << "   Dollar bill  → Eagle\n";
    std::cout << "========================================\n\n";

    // ── Resolve paths ─────────────────────────────────────────────────────────
    std::filesystem::path exeDir =
        std::filesystem::path(argv[0]).parent_path();

    std::string calibFile =
        (exeDir / "data" / "calibration" / "calibration.xml").string();
    std::string billImage =
        (exeDir / "data" / "bill.jpg").string();
    int cameraId = 0;

    if (argc >= 2) calibFile = argv[1];
    if (argc >= 3) billImage = argv[2];
    if (argc >= 4)
    {
        try { cameraId = std::stoi(argv[3]); }
        catch (...) { std::cerr << "[WARN] Invalid camera ID, using 0.\n"; }
    }

    std::cout << "Calibration : " << calibFile << "\n";
    std::cout << "Bill image  : " << billImage  << "\n";
    std::cout << "Camera ID   : " << cameraId   << "\n\n";

    // ── Target 1: Chessboard tracker ─────────────────────────────────────────
    PoseEstimator chessTracker(calibFile, cameraId);
    if (!chessTracker.loadCalibration())
    {
        std::cerr << "[FAILED] Cannot load calibration.\n";
        return 1;
    }

    // Rocket on chessboard — same geometry as augmentedReality.exe
    VirtualObject rocket;
    rocket.buildRocket();

    // ── Target 2: Dollar bill tracker ────────────────────────────────────────
    SIFTTracker billTracker(calibFile, billImage, 300);
    if (!billTracker.initialize())
    {
        std::cerr << "[FAILED] Cannot initialize bill tracker.\n";
        std::cerr << "         Ensure bill.jpg exists at: " << billImage << "\n";
        return 1;
    }

    // Eagle on dollar bill
    // scale=2.5: makes wingspan ~15cm which fills the bill nicely
    // Z values scale too so eagle floats visibly above the bill
    // Eagle on dollar bill
    // scale=1.2: wingspan ~7cm — fits inside bill without covering it
    // Z offset=2.0cm: floats visibly above bill surface
    VirtualEagleObject eagle;
    eagle.build(
        cv::Vec3f(7.8f, 3.3f, 2.0f),   // center of bill, 2cm above surface
        1.2f,                            // scale: 1 unit = 1.2cm
        true,                            // flipY
        true                             // flipZ
    );

    // ── Open webcam ───────────────────────────────────────────────────────────
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[FAILED] Cannot open camera " << cameraId << "\n";
        return 1;
    }

    // ── Controls ──────────────────────────────────────────────────────────────
    bool showRocket = true;
    bool showEagle  = true;
    bool showDebug  = false;

    std::cout << "Controls:\n";
    std::cout << "  'r'     - toggle rocket  (chessboard)\n";
    std::cout << "  'e'     - toggle eagle   (dollar bill)\n";
    std::cout << "  'd'     - toggle SIFT debug keypoints\n";
    std::cout << "  SPACE   - cycle chessboard axes/corners mode\n";
    std::cout << "  'q'/ESC - quit\n\n";
    std::cout << "Place chessboard AND dollar bill in view simultaneously.\n\n";

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat display = frame.clone();

        // ── Target 1: Chessboard → Rocket ────────────────────────────────────
        {
            std::vector<cv::Point2f> corners;
            bool found = chessTracker.detectCorners(frame, corners);

            if (found && chessTracker.estimatePose(corners))
            {
                // Axes / corners overlay
                if (chessTracker.getDisplayMode() == PoseEstimator::DisplayMode::CORNERS_ONLY ||
                    chessTracker.getDisplayMode() == PoseEstimator::DisplayMode::CORNERS_AXES)
                    chessTracker.projectOuterCorners(display);

                if (chessTracker.getDisplayMode() == PoseEstimator::DisplayMode::AXES_ONLY ||
                    chessTracker.getDisplayMode() == PoseEstimator::DisplayMode::CORNERS_AXES)
                    chessTracker.projectAxes(display);

                // Rocket
                if (showRocket)
                    rocket.draw(display,
                                chessTracker.getRvec(),
                                chessTracker.getTvec(),
                                chessTracker.getCameraMatrix(),
                                chessTracker.getDistCoeffs());
            }

            // Status top-left
            cv::Scalar col = (found && chessTracker.isPoseValid())
                ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::putText(display,
                        found ? "Chess: tracked" : "Chess: searching...",
                        cv::Point(10, 28),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 3);
            cv::putText(display,
                        found ? "Chess: tracked" : "Chess: searching...",
                        cv::Point(10, 28),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1);
        }

        // ── Target 2: Dollar bill → Eagle ────────────────────────────────────
        {
            bool tracked = billTracker.track(frame);

            if (showDebug)
                billTracker.drawDebug(display);

            if (tracked && showEagle)
                eagle.draw(display,
                           billTracker.getRvec(),
                           billTracker.getTvec(),
                           billTracker.getCameraMatrix(),
                           billTracker.getDistCoeffs());

            // Status below chess status
            cv::Scalar col = tracked
                ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
            std::string billStr = tracked
                ? "Bill:  tracked (" + std::to_string(billTracker.getInlierCount()) + " inliers)"
                : "Bill:  searching...";
            cv::putText(display, billStr, cv::Point(10, 54),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,0), 3);
            cv::putText(display, billStr, cv::Point(10, 54),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 1);
        }

        // ── Bottom hints ──────────────────────────────────────────────────────
        cv::putText(display,
                    std::string("Rocket:") + (showRocket ? "ON" : "OFF")
                    + "('r')  Eagle:" + (showEagle ? "ON" : "OFF") + "('e')",
                    cv::Point(10, display.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(180, 180, 180), 1);

        cv::imshow("Multi-Target AR", display);

        // ── Keys ──────────────────────────────────────────────────────────────
        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) break;
        if (key == 'r') { showRocket = !showRocket; }
        if (key == 'e') { showEagle  = !showEagle;  }
        if (key == 'd') { showDebug  = !showDebug;  }
        if (key == ' ') { chessTracker.cycleDisplayMode(); }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
