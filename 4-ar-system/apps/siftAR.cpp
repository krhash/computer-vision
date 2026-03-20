/*
 * siftAR.cpp - SIFT-based Augmented Reality Application
 * Author:      Krushna Sanjay Sharma
 * Description: Uber Extension 2 — AR with SIFT feature points.
 *              Replaces the chessboard target with a dollar bill tracked
 *              via SIFT feature matching. Uses the same camera calibration
 *              from Tasks 1-3 and the same rocket VirtualObject from Task 6.
 *
 * Usage:
 *   siftAR.exe [calibrationFile] [referenceImage] [cameraId]
 *
 *   calibrationFile : path to calibration XML (default: exe-relative path)
 *   referenceImage  : flat photo of the dollar bill (default: data/bill.jpg)
 *   cameraId        : webcam index (default: 0)
 *
 * Controls:
 *   'r'     - toggle rocket on/off
 *   'd'     - toggle debug overlay (inlier keypoints)
 *   'q'/ESC - quit
 *
 * How to prepare the reference image:
 *   1. Lay the dollar bill flat on a table
 *   2. Take a photo from directly above (as flat/square as possible)
 *   3. Crop to just the bill with minimal border
 *   4. Save as data/bill.jpg
 *
 * Date: March 2026
 */

#include "SIFTTracker.h"
#include "VirtualObject.h"
#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " AR Calibration System\n";
    std::cout << " SIFT AR App (Uber Extension 2)\n";
    std::cout << "========================================\n\n";

    /* Resolve default paths relative to executable */
    std::filesystem::path exeDir =
        std::filesystem::path(argv[0]).parent_path();

    std::string calibFile =
        (exeDir / "data" / "calibration" / "calibration.xml").string();
    std::string refImage  =
        (exeDir / "data" / "bill.jpg").string();
    int cameraId = 0;

    if (argc >= 2) calibFile = argv[1];
    if (argc >= 3) refImage  = argv[2];
    if (argc >= 4)
    {
        try { cameraId = std::stoi(argv[3]); }
        catch (...) { std::cerr << "[WARN] Invalid camera ID, using 0.\n"; }
    }

    std::cout << "Calibration : " << calibFile << "\n";
    std::cout << "Reference   : " << refImage  << "\n";
    std::cout << "Camera ID   : " << cameraId  << "\n\n";

    SIFTTracker tracker(calibFile, refImage, 300);
    if (!tracker.initialize())
    {
        std::cerr << "[FAILED] Tracker initialization failed.\n";
        std::cerr << "Make sure bill.jpg exists at: " << refImage << "\n";
        return 1;
    }

    /* Build rocket for dollar bill
     * Bill: X right (0-15.6cm), Y down (0-6.6cm).
     * flipY=true: bill Y is positive downward (opposite to chessboard -row).
     * flipZ=true: bill solvePnP Z toward camera is negative (standard OpenCV).
     * scale=1.2: 1 unit = 1.2cm */
    VirtualObject rocket;
    rocket.buildRocket(
        cv::Vec3f(7.8f - 0.75f, 3.3f - 0.75f, 0.0f),  // center on bill
        1.2f,                                            // scale
        true,                                            // flipY
        true                                             // flipZ
    );

    /* Open webcam */
    cv::VideoCapture cap(cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[FAILED] Cannot open camera " << cameraId << "\n";
        return 1;
    }

    std::cout << "Controls:\n";
    std::cout << "  'r' - toggle rocket on/off\n";
    std::cout << "  'd' - toggle debug keypoints\n";
    std::cout << "  'q'/ESC - quit\n\n";
    std::cout << "Point camera at the dollar bill reference image.\n\n";

    bool showRocket = true;
    bool showDebug  = true;
    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat display = frame.clone();

        /* Track bill and estimate pose */
        bool tracked = tracker.track(frame);

        /* Draw debug overlay */
        if (showDebug)
            tracker.drawDebug(display);

        /* Draw rocket if tracking */
        if (tracked && showRocket)
        {
            rocket.draw(display,
                        tracker.getRvec(),
                        tracker.getTvec(),
                        tracker.getCameraMatrix(),
                        tracker.getDistCoeffs());
        }

        // Rocket toggle indicator
        cv::putText(display,
                    showRocket ? "Rocket: ON  ('r')"
                               : "Rocket: OFF ('r')",
                    cv::Point(10, display.rows - 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    showRocket ? cv::Scalar(0, 255, 100)
                               : cv::Scalar(100, 100, 100), 1);

        cv::imshow("SIFT AR - Dollar Bill", display);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) break;
        if (key == 'r') showRocket = !showRocket;
        if (key == 'd') showDebug  = !showDebug;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
