////////////////////////////////////////////////////////////////////////////////
// CameraCalibration.h - Camera Calibration Class Header
// Author:      Krushna Sanjay Sharma
// Description: Declares the CameraCalibration class responsible for detecting
//              chessboard corners in a live video stream, collecting calibration
//              frames, and running OpenCV camera calibration.
//
// Modified from:
//   OpenCV Tutorial - Camera calibration With OpenCV
//   Original Author: Bernát Gábor
//   Source: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
//
// Task coverage:
//   Task 1 - Detect and extract chessboard corners from live video
//   Task 2 - Let user select calibration frames and store corner/world points
//   Task 3 - Run calibrateCamera, print results, save intrinsic parameters
//
// Key OpenCV functions used:
//   cv::findChessboardCorners  - detect internal corners of the chessboard
//   cv::cornerSubPix           - refine corner locations to sub-pixel accuracy
//   cv::drawChessboardCorners  - visualize detected corners on the frame
//   cv::calibrateCamera        - compute camera matrix + distortion coefficients
//   cv::FileStorage            - save/load calibration results to XML/YAML
//
// Chessboard configuration (matches checkerboard.png):
//   - 9 columns x 6 rows of internal corners  (BOARD_WIDTH x BOARD_HEIGHT)
//   - World coordinates treat each square as 1x1 unit
//   - Z = 0 (planar target), X right, Y down, Z toward viewer
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// CameraCalibration
//
// Manages the full calibration pipeline:
//   1. Opens a webcam and streams live video
//   2. Detects chessboard corners each frame (Task 1)
//   3. On keypress 's', saves that frame's corners + world points (Task 2)
//   4. On keypress 'c' (with >= MIN_FRAMES saved), runs calibration (Task 3)
//   5. Prints camera matrix, distortion coefficients, and reprojection error
//   6. Saves intrinsic parameters to an XML file for use in later tasks
////////////////////////////////////////////////////////////////////////////////
class CameraCalibration
{
public:
    // -------------------------------------------------------------------------
    // Board dimensions: internal corners of the checkerboard.
    // The 9x6 checkerboard.png has 9 columns and 6 rows of internal corners.
    // -------------------------------------------------------------------------
    static constexpr int BOARD_WIDTH  = 9;   // number of internal corners horizontally
    static constexpr int BOARD_HEIGHT = 6;   // number of internal corners vertically

    // Minimum number of frames required before calibration is allowed
    static constexpr int MIN_FRAMES = 5;

    // -------------------------------------------------------------------------
    // Constructor
    // cameraId     : index of the webcam to open (default 0)
    // outputFile   : path to save the calibration XML (e.g. "calibration.xml")
    // -------------------------------------------------------------------------
    CameraCalibration(int cameraId = 0,
                      const std::string& outputFile = "data/calibration/calibration.xml");

    // -------------------------------------------------------------------------
    // run()
    // Starts the main video loop.
    // Controls:
    //   's' - save current frame for calibration (only when corners detected)
    //   'c' - run calibration (requires >= MIN_FRAMES saved)
    //   'q' / ESC - quit
    // Returns true if calibration was successfully completed.
    // -------------------------------------------------------------------------
    bool run();

    // -------------------------------------------------------------------------
    // Accessors - available after a successful calibration
    // -------------------------------------------------------------------------
    const cv::Mat& getCameraMatrix()    const { return m_cameraMatrix; }
    const cv::Mat& getDistCoeffs()      const { return m_distCoeffs; }
    double         getReprojError()     const { return m_reprojError; }
    bool           isCalibrated()       const { return m_calibrated; }

private:
    // -------------------------------------------------------------------------
    // detectCorners()
    // Attempts to find chessboard corners in 'frame'.
    // On success: refines with cornerSubPix and draws corners onto 'frame'.
    // Returns true if corners were found; fills 'corners' output.
    // -------------------------------------------------------------------------
    bool detectCorners(const cv::Mat& frame,
                       std::vector<cv::Point2f>& corners);

    // -------------------------------------------------------------------------
    // saveFrame()
    // Stores the most recently detected corner_set and the corresponding
    // 3D world point_set into corner_list and point_list respectively.
    // World coordinates: (col, -row, 0) in square units (Z=0, planar target).
    // -------------------------------------------------------------------------
    void saveFrame(const std::vector<cv::Point2f>& corners);

    // -------------------------------------------------------------------------
    // buildWorldPoints()
    // Fills a point_set with the fixed 3D world coordinates of the chessboard
    // internal corners. Called once per saved frame (coordinates are identical
    // for every frame since they are defined on the target, not the camera).
    //
    // Convention (Z-axis toward viewer, squares = 1 unit):
    //   (0,0,0), (1,0,0), ..., (8,0,0),
    //   (0,-1,0), (1,-1,0), ..., (8,-1,0), ...
    // -------------------------------------------------------------------------
    void buildWorldPoints(std::vector<cv::Vec3f>& pointSet) const;

    // -------------------------------------------------------------------------
    // calibrate()
    // Calls cv::calibrateCamera with the collected point_list and corner_list.
    // Fills m_cameraMatrix, m_distCoeffs, m_reprojError.
    // Returns true on success.
    // -------------------------------------------------------------------------
    bool calibrate(const cv::Size& imageSize);

    // -------------------------------------------------------------------------
    // saveCalibration()
    // Writes camera matrix and distortion coefficients to m_outputFile (XML).
    // Also writes image size and reprojection error for reference.
    // -------------------------------------------------------------------------
    void saveCalibration(const cv::Size& imageSize) const;

    // -------------------------------------------------------------------------
    // printCalibrationResults()
    // Prints camera matrix, distortion coefficients, and reprojection error
    // to stdout in a readable format.
    // -------------------------------------------------------------------------
    void printCalibrationResults() const;

    // -------------------------------------------------------------------------
    // printStatus()
    // Overlays status text onto the display frame (frame count, instructions).
    // -------------------------------------------------------------------------
    void printStatus(cv::Mat& frame,
                     bool cornersFound,
                     int savedFrames) const;

    // =========================================================================
    // Member variables
    // =========================================================================

    int         m_cameraId;     // webcam device index
    std::string m_outputFile;   // XML output path for calibration data

    // Calibration data collected across frames (Task 2)
    // corner_list[i] = 2D image corners found in calibration frame i
    // point_list[i]  = corresponding 3D world points (same for all frames)
    std::vector<std::vector<cv::Point2f>> m_cornerList;  // 2D image points
    std::vector<std::vector<cv::Vec3f>>   m_pointList;   // 3D world points

    // Results (Task 3)
    cv::Mat m_cameraMatrix;   // 3x3 intrinsic matrix [fx,0,cx; 0,fy,cy; 0,0,1]
    cv::Mat m_distCoeffs;     // distortion coefficients [k1,k2,p1,p2,k3]
    double  m_reprojError;    // average re-projection error in pixels
    bool    m_calibrated;     // true after a successful calibration run

    // Chessboard size as cv::Size (width x height of internal corners)
    cv::Size m_boardSize;
};
