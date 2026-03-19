////////////////////////////////////////////////////////////////////////////////
// CameraCalibration.cpp - Camera Calibration Class Implementation
// Author:      Krushna Sanjay Sharma
// Description: Implements the CameraCalibration class. Handles live video
//              corner detection, frame collection, camera calibration, and
//              saving intrinsic parameters to XML.
//
// Modified from:
//   OpenCV Tutorial - Camera calibration With OpenCV
//   Original Author: Bernát Gábor
//   Source: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "CameraCalibration.h"
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
CameraCalibration::CameraCalibration(int cameraId, const std::string& outputFile)
    : m_cameraId(cameraId)
    , m_outputFile(outputFile)
    , m_reprojError(0.0)
    , m_calibrated(false)
    , m_boardSize(BOARD_WIDTH, BOARD_HEIGHT)
{
    // Initialize camera matrix to identity-like starting estimate.
    // Focal lengths start at 1.0; principal point will be set per frame size.
    // cv::calibrateCamera updates this fully during calibration.
    m_cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    // Initialize distortion to zero (5 coefficients: k1, k2, p1, p2, k3)
    m_distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
}

// ----------------------------------------------------------------------------
// run() - Main video loop
// ----------------------------------------------------------------------------
bool CameraCalibration::run()
{
    // Open the webcam
    cv::VideoCapture cap(m_cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open camera with ID: " << m_cameraId << std::endl;
        return false;
    }

    std::cout << "========================================\n";
    std::cout << " Camera Calibration - Tasks 1, 2, 3\n";
    std::cout << "========================================\n";
    std::cout << " Board: " << BOARD_WIDTH << " x " << BOARD_HEIGHT
              << " internal corners\n";
    std::cout << " Controls:\n";
    std::cout << "   's'   - Save current frame (when corners detected)\n";
    std::cout << "   'c'   - Calibrate (need >= " << MIN_FRAMES << " frames)\n";
    std::cout << "   'q'/ESC - Quit\n";
    std::cout << "========================================\n\n";

    cv::Mat frame;
    cv::Size imageSize;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "[ERROR] Empty frame received from camera.\n";
            break;
        }

        // Record image size from the first valid frame.
        // cv::calibrateCamera requires a consistent image size.
        if (imageSize.empty())
        {
            imageSize = frame.size();

            // Set a reasonable initial principal point estimate at image center.
            // This is the recommended starting point per the OpenCV tutorial.
            m_cameraMatrix.at<double>(0, 2) = imageSize.width  / 2.0;
            m_cameraMatrix.at<double>(1, 2) = imageSize.height / 2.0;
        }

        // ── Task 1: Detect and draw chessboard corners ────────────────────────
        std::vector<cv::Point2f> corners;
        bool found = detectCorners(frame, corners);

        // Overlay status info (frame count, instructions, calibration state)
        printStatus(frame, found, static_cast<int>(m_cornerList.size()));

        cv::imshow("Camera Calibration", frame);

        // ── Keyboard controls ─────────────────────────────────────────────────
        char key = static_cast<char>(cv::waitKey(30));

        if (key == 'q' || key == 27) // 27 = ESC
        {
            std::cout << "[INFO] Quitting.\n";
            break;
        }

        // ── Task 2: Save frame for calibration ───────────────────────────────
        if (key == 's')
        {
            if (found)
            {
                saveFrame(corners);
                std::cout << "[INFO] Frame saved. Total frames: "
                          << m_cornerList.size() << "\n";
                // Brief white flash to acknowledge save (like the OpenCV tutorial)
                cv::bitwise_not(frame, frame);
                cv::imshow("Camera Calibration", frame);
                cv::waitKey(200);
            }
            else
            {
                std::cout << "[WARN] No corners detected in this frame. Not saved.\n";
            }
        }

        // ── Task 3: Run calibration ───────────────────────────────────────────
        if (key == 'c')
        {
            int n = static_cast<int>(m_cornerList.size());
            if (n < MIN_FRAMES)
            {
                std::cout << "[WARN] Need at least " << MIN_FRAMES
                          << " frames. Currently have: " << n << "\n";
            }
            else
            {
                std::cout << "\n[INFO] Running calibration with "
                          << n << " frames...\n";
                if (calibrate(imageSize))
                {
                    printCalibrationResults();
                    saveCalibration(imageSize);
                    m_calibrated = true;
                }
                else
                {
                    std::cerr << "[ERROR] Calibration failed.\n";
                }
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return m_calibrated;
}

// ----------------------------------------------------------------------------
// detectCorners() - Task 1
// Finds internal chessboard corners in the given frame.
// Uses cornerSubPix to refine corner positions to sub-pixel accuracy.
// Draws corners on the frame for visual feedback.
// ----------------------------------------------------------------------------
bool CameraCalibration::detectCorners(const cv::Mat& frame,
                                       std::vector<cv::Point2f>& corners)
{
    // Convert to grayscale - findChessboardCorners works on grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Flags recommended by the OpenCV tutorial for reliable detection:
    //   CALIB_CB_ADAPTIVE_THRESH - adapts threshold for uneven lighting
    //   CALIB_CB_NORMALIZE_IMAGE - normalizes image before thresholding
    //   CALIB_CB_FAST_CHECK      - quick rejection of frames without a board
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH
              | cv::CALIB_CB_NORMALIZE_IMAGE
              | cv::CALIB_CB_FAST_CHECK;

    bool found = cv::findChessboardCorners(gray, m_boardSize, corners, flags);

    if (found)
    {
        // Refine corner positions to sub-pixel accuracy.
        cv::cornerSubPix(
            gray, corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                30, 0.001
            )
        );

        // Print info about the first corner
        if (!corners.empty())
        {
            std::cout << "[Task 1] Corners found: " << corners.size()
                      << " | First corner: ("
                      << std::fixed << std::setprecision(1)
                      << corners[0].x << ", " << corners[0].y << ")\n";
        }

        // Draw small green dots at each corner — no rainbow grid lines
        for (const auto& pt : corners)
            cv::circle(const_cast<cv::Mat&>(frame), pt, 3,
                       cv::Scalar(0, 255, 0), -1);
    }

    return found;
}

// ----------------------------------------------------------------------------
// saveFrame() - Task 2
// Saves the detected 2D corners and the corresponding 3D world points.
// ----------------------------------------------------------------------------
void CameraCalibration::saveFrame(const std::vector<cv::Point2f>& corners)
{
    // Store the 2D image corners for this frame
    m_cornerList.push_back(corners);

    // Build and store the 3D world point set for this frame.
    // The world coordinates are always the same regardless of board orientation.
    std::vector<cv::Vec3f> pointSet;
    buildWorldPoints(pointSet);
    m_pointList.push_back(pointSet);
}

// ----------------------------------------------------------------------------
// buildWorldPoints()
// Creates 3D world coordinates for all internal chessboard corners.
//
// Convention (per assignment description):
//   - Squares treated as 1x1 units
//   - (0,0,0) at upper-left internal corner
//   - X increases rightward along columns
//   - Y increases downward along rows  (positive Y = down)
//   - Z negative = toward the viewer   (per assignment: "first point on next
//     row is (0,-1,0) if Z-axis comes towards the viewer" — meaning +Y is down)
//   - Z = 0 for all points (planar target)
//
// This results in: (0,0,0), (1,0,0), ..., (8,0,0),
//                  (0,1,0), (1,1,0), ..., (8,1,0), ...
// ----------------------------------------------------------------------------
void CameraCalibration::buildWorldPoints(std::vector<cv::Vec3f>& pointSet) const
{
    pointSet.clear();
    pointSet.reserve(BOARD_WIDTH * BOARD_HEIGHT);

    for (int row = 0; row < BOARD_HEIGHT; ++row)
    {
        for (int col = 0; col < BOARD_WIDTH; ++col)
        {
            // Y positive downward, Z = 0 (planar board)
            pointSet.emplace_back(
                static_cast<float>(col),   // X = column index
                static_cast<float>(row),   // Y = row index (positive downward)
                0.0f                       // Z = 0 (planar)
            );
        }
    }
}

// ----------------------------------------------------------------------------
// calibrate() - Task 3
// Runs cv::calibrateCamera with all collected frames.
// Fills m_cameraMatrix, m_distCoeffs, m_reprojError.
// ----------------------------------------------------------------------------
bool CameraCalibration::calibrate(const cv::Size& imageSize)
{
    // These will hold per-frame rotation and translation vectors.
    // rvecs[i] and tvecs[i] describe the board pose for calibration frame i.
    std::vector<cv::Mat> rvecs, tvecs;

    // Use CALIB_FIX_ASPECT_RATIO to assume square pixels (fx = fy).
    // This is appropriate for modern cameras as noted in the OpenCV tutorial.
    int calibFlags = cv::CALIB_FIX_ASPECT_RATIO;

    try
    {
        // cv::calibrateCamera returns the RMS re-projection error.
        // A value < 1.0 pixel indicates a good calibration.
        m_reprojError = cv::calibrateCamera(
            m_pointList,    // 3D world points (same set repeated per frame)
            m_cornerList,   // 2D image points collected from each frame
            imageSize,      // size of the calibration images
            m_cameraMatrix, // OUTPUT: 3x3 intrinsic matrix
            m_distCoeffs,   // OUTPUT: distortion coefficients
            rvecs,          // OUTPUT: rotation vectors per frame
            tvecs,          // OUTPUT: translation vectors per frame
            calibFlags
        );
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[ERROR] cv::calibrateCamera threw: " << e.what() << "\n";
        return false;
    }

    // Sanity check: reprojection error should be < 1 pixel for a good result
    if (m_reprojError > 5.0)
    {
        std::cout << "[WARN] High reprojection error: " << m_reprojError
                  << " px. Consider recapturing calibration frames.\n";
    }

    return true;
}

// ----------------------------------------------------------------------------
// saveCalibration() - Task 3
// Writes intrinsic parameters to an XML file using cv::FileStorage.
// The output file is read by the augmentedReality app (Tasks 4-6).
// ----------------------------------------------------------------------------
void CameraCalibration::saveCalibration(const cv::Size& imageSize) const
{
    // Directory is guaranteed to exist — created by CMake at configure time
    // via file(MAKE_DIRECTORY) in CMakeLists.txt. No runtime dir creation needed.
    cv::FileStorage fs(m_outputFile, cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "[ERROR] Cannot open output file: " << m_outputFile << "\n";
        return;
    }

    // Write metadata
    fs << "calibration_date"  << cv::format("%s", __DATE__);
    fs << "image_width"       << imageSize.width;
    fs << "image_height"      << imageSize.height;
    fs << "board_width"       << BOARD_WIDTH;
    fs << "board_height"      << BOARD_HEIGHT;
    fs << "num_frames"        << static_cast<int>(m_cornerList.size());
    fs << "reprojection_error" << m_reprojError;

    // Write the two key matrices
    fs << "camera_matrix"          << m_cameraMatrix;
    fs << "distortion_coefficients" << m_distCoeffs;

    fs.release();
    std::cout << "[INFO] Calibration saved to: " << m_outputFile << "\n";
}

// ----------------------------------------------------------------------------
// printCalibrationResults() - Task 3
// Prints camera matrix, distortion coefficients, and reprojection error.
// ----------------------------------------------------------------------------
void CameraCalibration::printCalibrationResults() const
{
    std::cout << "\n========================================\n";
    std::cout << " Calibration Results\n";
    std::cout << "========================================\n";

    std::cout << "\nCamera Matrix (intrinsics):\n";
    std::cout << "  [fx,  0, cx]   [" << m_cameraMatrix.at<double>(0,0) << ",  0, "
              << m_cameraMatrix.at<double>(0,2) << "]\n";
    std::cout << "  [ 0, fy, cy] = [0,  " << m_cameraMatrix.at<double>(1,1) << ", "
              << m_cameraMatrix.at<double>(1,2) << "]\n";
    std::cout << "  [ 0,  0,  1]   [0,  0,  1]\n";

    std::cout << "\nDistortion Coefficients [k1, k2, p1, p2, k3]:\n  ";
    for (int i = 0; i < m_distCoeffs.rows; ++i)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << m_distCoeffs.at<double>(i, 0);
        if (i < m_distCoeffs.rows - 1) std::cout << ",  ";
    }
    std::cout << "\n";

    std::cout << "\nRe-projection Error: " << std::fixed << std::setprecision(4)
              << m_reprojError << " pixels";

    if (m_reprojError < 1.0)
        std::cout << "  [GOOD - < 1 pixel]\n";
    else if (m_reprojError < 3.0)
        std::cout << "  [ACCEPTABLE]\n";
    else
        std::cout << "  [POOR - consider recapturing]\n";

    std::cout << "========================================\n\n";
}

// ----------------------------------------------------------------------------
// printStatus()
// Overlays frame count, corner detection status, and key hints onto the frame.
// ----------------------------------------------------------------------------
void CameraCalibration::printStatus(cv::Mat& frame,
                                     bool cornersFound,
                                     int savedFrames) const
{
    // Choose color based on corner detection state
    cv::Scalar color = cornersFound
        ? cv::Scalar(0, 255, 0)    // green when corners found
        : cv::Scalar(0, 0, 255);   // red when not found

    // Detection status line
    std::string statusMsg = cornersFound
        ? "Corners detected - press 's' to save"
        : "No corners - aim at chessboard";
    cv::putText(frame, statusMsg, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

    // Frame count line
    std::string countMsg = "Saved frames: " + std::to_string(savedFrames)
                         + " / " + std::to_string(MIN_FRAMES) + " min";
    cv::putText(frame, countMsg, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // Calibration hint
    if (savedFrames >= MIN_FRAMES)
    {
        cv::putText(frame, "Press 'c' to calibrate",
                    cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
    }

    // Calibrated state
    if (m_calibrated)
    {
        cv::putText(frame, "CALIBRATED",
                    cv::Point(10, frame.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
}
