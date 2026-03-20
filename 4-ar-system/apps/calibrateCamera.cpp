/*
 * calibrateCamera.cpp - Camera Calibration Application Entry Point
 * Author:      Krushna Sanjay Sharma
 * Description: Main application for Tasks 1-3. Opens a live webcam feed,
 *              detects chessboard corners, collects calibration frames,
 *              runs camera calibration, and saves the intrinsic parameters
 *              to an XML file for use in the augmented reality tasks.
 *
 * Usage:
 *   calibrateCamera.exe [cameraId] [outputFile]
 *
 *   cameraId   : optional, webcam index (default: 0)
 *   outputFile : optional, path for calibration XML
 *                (default: data/calibration/calibration.xml)
 *
 * Controls (shown in the live video window):
 *   's'      - Save current frame (only works when green corners are visible)
 *   'c'      - Run calibration (requires >= 5 saved frames)
 *   'q'/ESC  - Quit
 *
 * Date: March 2026
 */

#include "CameraCalibration.h"
#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " AR Calibration System\n";
    std::cout << " Camera Calibration App (Tasks 1-3)\n";
    std::cout << "========================================\n\n";

    /* Resolve output path relative to the executable location.
     * argv[0] gives us the full path to the exe. We derive a sibling
     * data/calibration/ folder next to it so the path always resolves
     * regardless of what the working directory is. */
    std::filesystem::path exeDir = 
        std::filesystem::path(argv[0]).parent_path();

    int         cameraId   = 0;
    std::string outputFile = (exeDir / "data" / "calibration" / "calibration.xml")
                                .string();

    if (argc >= 2)
    {
        try
        {
            cameraId = std::stoi(argv[1]);
        }
        catch (...)
        {
            std::cerr << "[WARN] Invalid camera ID '" << argv[1]
                      << "'. Using default: 0\n";
        }
    }

    if (argc >= 3)
    {
        outputFile = argv[2];
    }

    std::cout << "Camera ID   : " << cameraId   << "\n";
    std::cout << "Output file : " << outputFile << "\n\n";

    /* Run the calibration pipeline */
    CameraCalibration calib(cameraId, outputFile);
    bool success = calib.run();

    /* Report result */
    if (success)
    {
        std::cout << "\n[SUCCESS] Calibration complete.\n";
        std::cout << "  Reprojection error : "
                  << calib.getReprojError() << " pixels\n";
        std::cout << "  Parameters saved to: " << outputFile << "\n";
        std::cout << "\nNext step: run augmentedReality.exe\n";
        return 0;
    }
    else
    {
        std::cerr << "\n[FAILED] Calibration was not completed.\n";
        return 1;
    }
}
