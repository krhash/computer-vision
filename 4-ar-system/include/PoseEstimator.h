////////////////////////////////////////////////////////////////////////////////
// PoseEstimator.h - Pose Estimator Class Header
// Author:      Krushna Sanjay Sharma
// Description: Declares the PoseEstimator class responsible for:
//              - Loading camera intrinsics from a calibration XML file
//              - Detecting chessboard corners each frame
//              - Running cv::solvePnP to get board rotation and translation
//              - Printing real-time rotation and translation to the console
//
// Reference:
//   OpenCV Documentation - solvePnP
//   https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
//
// Task coverage:
//   Task 4 - Calculate current position of the camera using solvePnP
//
// Key OpenCV functions used:
//   cv::solvePnP       - compute rotation (rvec) and translation (tvec)
//                        from 3D-2D point correspondences
//   cv::FileStorage    - read calibration XML written by CameraCalibration
//   cv::findChessboardCorners / cv::cornerSubPix - reused from Task 1
//
// solvePnP signature:
//   bool cv::solvePnP(
//       InputArray  objectPoints,   // 3D world points  (Vec3f vector)
//       InputArray  imagePoints,    // 2D image corners (Point2f vector)
//       InputArray  cameraMatrix,   // 3x3 intrinsic matrix from calibration
//       InputArray  distCoeffs,     // distortion coefficients from calibration
//       OutputArray rvec,           // OUTPUT: rotation vector    (Rodrigues)
//       OutputArray tvec,           // OUTPUT: translation vector
//       bool        useExtrinsicGuess = false,
//       int         flags = SOLVEPNP_ITERATIVE
//   )
//
// Coordinate interpretation:
//   rvec (rotation vector, Rodrigues form):
//     - direction = axis of rotation
//     - magnitude = angle in radians
//     - Use cv::Rodrigues(rvec, R) to get the 3x3 rotation matrix
//   tvec (translation vector):
//     - position of the world origin (board corner 0,0,0) in camera space
//     - tvec[0] = X right,  tvec[1] = Y down,  tvec[2] = Z into scene
//     - Moving camera right  → tvec[0] becomes more negative
//     - Moving camera closer → tvec[2] decreases
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// PoseEstimator
//
// Loads intrinsic calibration, then runs a live video loop where each frame
// with a detected chessboard target is solved for pose using solvePnP.
// Rotation and translation are printed to the console in real time.
//
// Designed to be extended by Tasks 5 and 6 (projectPoints, virtual objects).
// The rvec and tvec are stored as member variables and accessible via getters
// so derived work in augmentedReality.cpp can reuse them directly.
////////////////////////////////////////////////////////////////////////////////
class PoseEstimator
{
public:
    // Board dimensions — must match CameraCalibration constants
    static constexpr int BOARD_WIDTH  = 9;
    static constexpr int BOARD_HEIGHT = 6;

    // -------------------------------------------------------------------------
    // Constructor
    // calibrationFile : path to the XML written by CameraCalibration::run()
    //                   (default: same location the calibrateCamera app saves to)
    // cameraId        : webcam index (default 0)
    // -------------------------------------------------------------------------
    explicit PoseEstimator(
        const std::string& calibrationFile =
            "data/calibration/calibration.xml",
        int cameraId = 0);

    // -------------------------------------------------------------------------
    // run()
    // Opens the webcam and loops:
    //   1. Detect chessboard corners
    //   2. Call solvePnP with the board's 3D world points
    //   3. Print rvec and tvec to console each frame
    // Controls:
    //   'q' / ESC - quit
    // Returns true if the loop ran without errors.
    // -------------------------------------------------------------------------
    bool run();

    // -------------------------------------------------------------------------
    // Accessors — valid after at least one successful solvePnP call
    // -------------------------------------------------------------------------
    const cv::Mat& getRvec()          const { return m_rvec; }
    const cv::Mat& getTvec()          const { return m_tvec; }
    const cv::Mat& getCameraMatrix()  const { return m_cameraMatrix; }
    const cv::Mat& getDistCoeffs()    const { return m_distCoeffs; }
    bool           isPoseValid()      const { return m_poseValid; }

protected:
    // -------------------------------------------------------------------------
    // loadCalibration()
    // Reads camera_matrix and distortion_coefficients from the XML file.
    // Returns false if the file cannot be opened or keys are missing.
    // -------------------------------------------------------------------------
    bool loadCalibration();

    // -------------------------------------------------------------------------
    // detectCorners()
    // Finds and refines chessboard corners in 'frame'.
    // Draws corners onto 'frame' when found.
    // Returns true and fills 'corners' on success.
    // -------------------------------------------------------------------------
    bool detectCorners(const cv::Mat& frame,
                       std::vector<cv::Point2f>& corners);

    // -------------------------------------------------------------------------
    // buildWorldPoints()
    // Returns the fixed 3D world coordinates for all internal board corners.
    // Identical convention to CameraCalibration::buildWorldPoints():
    //   (col, -row, 0) in square units, Z = 0 (planar target)
    // -------------------------------------------------------------------------
    void buildWorldPoints(std::vector<cv::Vec3f>& pointSet) const;

    // -------------------------------------------------------------------------
    // estimatePose()
    // Calls cv::solvePnP using the detected 2D corners and the fixed 3D world
    // points. Stores results in m_rvec and m_tvec.
    // Returns true on success.
    // -------------------------------------------------------------------------
    bool estimatePose(const std::vector<cv::Point2f>& corners);

    // -------------------------------------------------------------------------
    // printPose()
    // Prints the current rvec and tvec values to stdout with labels.
    // Called every frame when pose is valid — helps verify Task 4 behaviour.
    // -------------------------------------------------------------------------
    void printPose() const;

    // -------------------------------------------------------------------------
    // overlayStatus()
    // Draws pose values and instructions onto the display frame.
    // -------------------------------------------------------------------------
    void overlayStatus(cv::Mat& frame, bool poseFound) const;

    // =========================================================================
    // Member variables
    // =========================================================================

    std::string m_calibrationFile;  // path to calibration XML
    int         m_cameraId;         // webcam index

    // Intrinsics loaded from calibration file
    cv::Mat m_cameraMatrix;         // 3x3 intrinsic matrix
    cv::Mat m_distCoeffs;           // distortion coefficients

    // Pose output from solvePnP (updated every frame)
    cv::Mat m_rvec;                 // rotation vector    (3x1, Rodrigues)
    cv::Mat m_tvec;                 // translation vector (3x1)
    bool    m_poseValid;            // true after at least one successful solve

    // Board size as cv::Size
    cv::Size m_boardSize;
};
