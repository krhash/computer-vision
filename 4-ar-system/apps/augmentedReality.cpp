////////////////////////////////////////////////////////////////////////////////
// augmentedReality.cpp - Augmented Reality Application Entry Point
// Author:      Krushna Sanjay Sharma
// Description: Main application for Tasks 4-6.
//   Task 4 - solvePnP: compute and print real-time rotation/translation
//   Task 5 - projectPoints: overlay outer corners and 3D axes on the board
//   Task 6 - VirtualObject: render a 3D rocket floating above the board
//
// Usage:
//   augmentedReality.exe [calibrationFile] [cameraId]
//
// Controls:
//   SPACE   - cycle Task 5 display mode (corners / axes / both)
//   'r'     - toggle rocket visibility (Task 6)
//   'q'/ESC - quit
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "PoseEstimator.h"
#include "VirtualObject.h"
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
    std::filesystem::path exeDir =
        std::filesystem::path(argv[0]).parent_path();

    std::string calibFile =
        (exeDir / "data" / "calibration" / "calibration.xml").string();
    int cameraId = 0;

    if (argc >= 2) calibFile = argv[1];
    if (argc >= 3)
    {
        try   { cameraId = std::stoi(argv[2]); }
        catch (...) { std::cerr << "[WARN] Invalid camera ID, using 0.\n"; }
    }

    std::cout << "Calibration file : " << calibFile << "\n";
    std::cout << "Camera ID        : " << cameraId  << "\n\n";

    // ── Build the rocket once ─────────────────────────────────────────────────
    VirtualObject rocket;
    rocket.buildRocket();

    // ── Set up the pose estimator ─────────────────────────────────────────────
    PoseEstimator estimator(calibFile, cameraId);

    if (!estimator.loadCalibration())
    {
        std::cerr << "[FAILED] Could not load calibration. "
                  << "Run calibrateCamera.exe first.\n";
        return 1;
    }

    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[FAILED] Cannot open camera " << cameraId << "\n";
        return 1;
    }

    std::cout << "Controls:\n";
    std::cout << "  SPACE  - cycle axes/corners display mode\n";
    std::cout << "  'r'    - toggle rocket on/off\n";
    std::cout << "  'q'/ESC - quit\n\n";

    bool showRocket = true;

    // Pre-build world points (reused every frame)
    std::vector<cv::Vec3f> worldPoints;
    // (re-use PoseEstimator's buildWorldPoints via a local copy)
    for (int row = 0; row < PoseEstimator::BOARD_HEIGHT; ++row)
        for (int col = 0; col < PoseEstimator::BOARD_WIDTH; ++col)
            worldPoints.emplace_back(float(col), float(-row), 0.0f);

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // ── Tasks 4 + 5: detect, solve pose, draw axes/corners ───────────────
        std::vector<cv::Point2f> corners;
        bool found = estimator.detectCorners(frame, corners);

        if (found && estimator.estimatePose(corners))
        {
            estimator.printPose();

            // Task 5 — project outer corners and/or axes
            if (estimator.getDisplayMode() == PoseEstimator::DisplayMode::CORNERS_ONLY ||
                estimator.getDisplayMode() == PoseEstimator::DisplayMode::CORNERS_AXES)
                estimator.projectOuterCorners(frame);

            if (estimator.getDisplayMode() == PoseEstimator::DisplayMode::AXES_ONLY ||
                estimator.getDisplayMode() == PoseEstimator::DisplayMode::CORNERS_AXES)
                estimator.projectAxes(frame);

            // ── Task 6: draw the rocket ───────────────────────────────────────
            if (showRocket)
                rocket.draw(frame,
                            estimator.getRvec(),
                            estimator.getTvec(),
                            estimator.getCameraMatrix(),
                            estimator.getDistCoeffs());
        }

        estimator.overlayStatus(frame, found && estimator.isPoseValid());

        // Rocket visibility indicator
        cv::putText(frame,
                    showRocket ? "Rocket: ON  ('r' to toggle)"
                               : "Rocket: OFF ('r' to toggle)",
                    cv::Point(10, frame.rows - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    showRocket ? cv::Scalar(0, 255, 100)
                               : cv::Scalar(100, 100, 100), 1);

        cv::imshow("Augmented Reality", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) break;
        if (key == 'r') { showRocket = !showRocket; }
        if (key == ' ') { estimator.cycleDisplayMode(); }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
