////////////////////////////////////////////////////////////////////////////////
// PoseEstimator.cpp - Pose Estimator Class Implementation
// Author:      Krushna Sanjay Sharma
// Description: Implements pose estimation using cv::solvePnP. Loads camera
//              intrinsics from calibration XML, detects chessboard corners
//              each frame, and computes the board's 6-DOF pose in real time.
//
// Reference:
//   OpenCV Documentation - solvePnP
//   https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "PoseEstimator.h"
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
PoseEstimator::PoseEstimator(const std::string& calibrationFile, int cameraId)
    : m_calibrationFile(calibrationFile)
    , m_cameraId(cameraId)
    , m_poseValid(false)
    , m_boardSize(BOARD_WIDTH, BOARD_HEIGHT)
{
    // Initialize rvec and tvec as empty 3x1 double matrices.
    // solvePnP will fill these each frame.
    m_rvec = cv::Mat::zeros(3, 1, CV_64F);
    m_tvec = cv::Mat::zeros(3, 1, CV_64F);
}

// ----------------------------------------------------------------------------
// run() - Main video loop for Task 4
// ----------------------------------------------------------------------------
bool PoseEstimator::run()
{
    // ── Load calibration from XML ─────────────────────────────────────────────
    if (!loadCalibration())
    {
        std::cerr << "[ERROR] Failed to load calibration. "
                  << "Run calibrateCamera.exe first.\n";
        return false;
    }

    // ── Open webcam ───────────────────────────────────────────────────────────
    cv::VideoCapture cap(m_cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open camera ID: " << m_cameraId << "\n";
        return false;
    }

    std::cout << "========================================\n";
    std::cout << " Pose Estimator - Task 4\n";
    std::cout << "========================================\n";
    std::cout << " Calibration: " << m_calibrationFile << "\n";
    std::cout << " Point camera at the chessboard target.\n";
    std::cout << " Rotation and translation printed each frame.\n";
    std::cout << " Press 'q' or ESC to quit.\n";
    std::cout << "========================================\n\n";

    // Pre-build the fixed 3D world point set once — it never changes
    std::vector<cv::Vec3f> worldPoints;
    buildWorldPoints(worldPoints);

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        // ── Task 4: Detect corners then solve for pose ────────────────────────
        std::vector<cv::Point2f> corners;
        bool found = detectCorners(frame, corners);

        if (found)
        {
            // solvePnP: given known 3D world points and their detected 2D image
            // positions, compute the rotation and translation that maps world
            // space into camera space.
            if (estimatePose(corners))
            {
                // Print rvec and tvec to console every frame (Task 4 requirement)
                printPose();
            }
        }

        // Overlay status and current pose values on the frame
        overlayStatus(frame, found && m_poseValid);

        cv::imshow("Pose Estimator", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return true;
}

// ----------------------------------------------------------------------------
// loadCalibration()
// Reads camera_matrix and distortion_coefficients from the calibration XML.
// ----------------------------------------------------------------------------
bool PoseEstimator::loadCalibration()
{
    cv::FileStorage fs(m_calibrationFile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "[ERROR] Cannot open calibration file: "
                  << m_calibrationFile << "\n";
        return false;
    }

    fs["camera_matrix"]           >> m_cameraMatrix;
    fs["distortion_coefficients"] >> m_distCoeffs;
    fs.release();

    // Validate that we actually read something
    if (m_cameraMatrix.empty() || m_distCoeffs.empty())
    {
        std::cerr << "[ERROR] Calibration file is missing required keys "
                  << "(camera_matrix / distortion_coefficients).\n";
        return false;
    }

    std::cout << "[INFO] Calibration loaded from: " << m_calibrationFile << "\n";
    std::cout << "[INFO] Camera matrix:\n"          << m_cameraMatrix    << "\n";
    std::cout << "[INFO] Distortion coefficients:\n"<< m_distCoeffs      << "\n\n";

    return true;
}

// ----------------------------------------------------------------------------
// detectCorners()
// Finds and refines chessboard corners. Reuses the same approach as Task 1.
// ----------------------------------------------------------------------------
bool PoseEstimator::detectCorners(const cv::Mat& frame,
                                   std::vector<cv::Point2f>& corners)
{
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    int flags = cv::CALIB_CB_ADAPTIVE_THRESH
              | cv::CALIB_CB_NORMALIZE_IMAGE
              | cv::CALIB_CB_FAST_CHECK;

    bool found = cv::findChessboardCorners(gray, m_boardSize, corners, flags);

    if (found)
    {
        // Refine to sub-pixel accuracy for better pose accuracy
        cv::cornerSubPix(
            gray, corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
        );

        cv::drawChessboardCorners(
            const_cast<cv::Mat&>(frame), m_boardSize,
            cv::Mat(corners), found);
    }

    return found;
}

// ----------------------------------------------------------------------------
// buildWorldPoints()
// Fixed 3D world coordinates for all internal chessboard corners.
// Convention: (col, -row, 0) — same as CameraCalibration.
// ----------------------------------------------------------------------------
void PoseEstimator::buildWorldPoints(std::vector<cv::Vec3f>& pointSet) const
{
    pointSet.clear();
    pointSet.reserve(BOARD_WIDTH * BOARD_HEIGHT);

    for (int row = 0; row < BOARD_HEIGHT; ++row)
        for (int col = 0; col < BOARD_WIDTH; ++col)
            pointSet.emplace_back(
                static_cast<float>(col),
                static_cast<float>(-row),
                0.0f
            );
}

// ----------------------------------------------------------------------------
// estimatePose() - Core of Task 4
//
// cv::solvePnP parameters:
//   objectPoints  : 3D world coords of board corners (Vec3f vector)
//   imagePoints   : detected 2D corner positions     (Point2f vector)
//   cameraMatrix  : 3x3 intrinsic matrix from calibration
//   distCoeffs    : distortion coefficients from calibration
//   rvec          : OUTPUT rotation vector (Rodrigues, 3x1)
//   tvec          : OUTPUT translation vector (3x1, in world units = squares)
//   useExtrinsicGuess : false — compute from scratch each frame
//   flags         : SOLVEPNP_ITERATIVE — Levenberg-Marquardt, best for planar
//
// What the outputs mean (Task 4 requirement):
//   rvec: axis-angle rotation. ||rvec|| = angle in radians.
//         Rotating the board changes rvec direction.
//   tvec: position of the world origin (0,0,0 = upper-left board corner)
//         expressed in camera coordinates (in square units).
//         tvec[2] ~ distance from camera to board.
//         Move camera right → tvec[0] becomes more negative.
//         Move camera closer → tvec[2] decreases.
// ----------------------------------------------------------------------------
bool PoseEstimator::estimatePose(const std::vector<cv::Point2f>& corners)
{
    std::vector<cv::Vec3f> worldPoints;
    buildWorldPoints(worldPoints);

    try
    {
        bool ok = cv::solvePnP(
            worldPoints,      // 3D object points in world space
            corners,          // 2D image points detected this frame
            m_cameraMatrix,   // intrinsic matrix (from calibration XML)
            m_distCoeffs,     // distortion coefficients (from calibration XML)
            m_rvec,           // OUTPUT: rotation vector  (Rodrigues)
            m_tvec,           // OUTPUT: translation vector
            false,            // useExtrinsicGuess: false = solve from scratch
            cv::SOLVEPNP_ITERATIVE  // method: Levenberg-Marquardt
        );

        m_poseValid = ok;
        return ok;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[ERROR] solvePnP failed: " << e.what() << "\n";
        m_poseValid = false;
        return false;
    }
}

// ----------------------------------------------------------------------------
// printPose()
// Prints rvec and tvec to console. Required by Task 4 to verify values
// change correctly as the camera or board is moved.
//
// Expected behaviour to verify:
//   Move camera LEFT  → tvec[0] increases  (board origin moves right in cam)
//   Move camera RIGHT → tvec[0] decreases
//   Move camera CLOSER→ tvec[2] decreases
//   Move camera AWAY  → tvec[2] increases
//   Tilt board        → rvec direction changes noticeably
// ----------------------------------------------------------------------------
void PoseEstimator::printPose() const
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[Pose] "
              << "rvec: ["
              << m_rvec.at<double>(0) << ", "
              << m_rvec.at<double>(1) << ", "
              << m_rvec.at<double>(2) << "]  "
              << "tvec: ["
              << m_tvec.at<double>(0) << ", "
              << m_tvec.at<double>(1) << ", "
              << m_tvec.at<double>(2) << "]\n";
}

// ----------------------------------------------------------------------------
// overlayStatus()
// Draws current rvec/tvec values and instructions onto the video frame.
// ----------------------------------------------------------------------------
void PoseEstimator::overlayStatus(cv::Mat& frame, bool poseFound) const
{
    cv::Scalar color = poseFound
        ? cv::Scalar(0, 255, 0)
        : cv::Scalar(0, 0, 255);

    std::string status = poseFound ? "Pose estimated" : "No board detected";
    cv::putText(frame, status, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);

    if (poseFound)
    {
        // Display tvec (most intuitive: represents camera-to-board distance)
        auto fmt = [](double v) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << v;
            return ss.str();
        };

        std::string tvecStr = "tvec: ["
            + fmt(m_tvec.at<double>(0)) + ", "
            + fmt(m_tvec.at<double>(1)) + ", "
            + fmt(m_tvec.at<double>(2)) + "]";

        std::string rvecStr = "rvec: ["
            + fmt(m_rvec.at<double>(0)) + ", "
            + fmt(m_rvec.at<double>(1)) + ", "
            + fmt(m_rvec.at<double>(2)) + "]";

        cv::putText(frame, tvecStr, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1);
        cv::putText(frame, rvecStr, cv::Point(10, 85),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(255, 255, 255), 1);
    }

    cv::putText(frame, "q/ESC: quit",
                cv::Point(10, frame.rows - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(180, 180, 180), 1);
}
