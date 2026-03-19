////////////////////////////////////////////////////////////////////////////////
// FeatureDetector.h - Feature Detector Class Header
// Author:      Krushna Sanjay Sharma
// Description: Declares the FeatureDetector class responsible for detecting
//              SIFT features in a live video stream and visualizing them.
//              Uses a dollar bill / banknote as the target pattern.
//              Allows real-time feature count adjustment via a trackbar.
//
// Task coverage:
//   Task 7 - Detect robust SIFT features in a live video stream.
//            Show where features appear on a dollar bill pattern.
//            Experiment with max features and contrast threshold.
//
// Key OpenCV functions used:
//   cv::SIFT::create()           - creates SIFT detector instance
//   sift->detectAndCompute()     - detects keypoints + 128-dim descriptors
//   cv::drawKeypoints()          - draws keypoints with scale + orientation
//   cv::createTrackbar()         - runtime threshold adjustment
//
// cv::SIFT::create() parameters:
//   nfeatures         : max features to detect (0 = unlimited)
//   nOctaveLayers     : layers per octave (default 3)
//   contrastThreshold : filters weak low-contrast features (default 0.04)
//   edgeThreshold     : filters edge-like responses (default 10)
//   sigma             : initial Gaussian blur (default 1.6)
//
// Why SIFT over SURF:
//   SURF is patented and requires opencv_contrib (not in standard OpenCV).
//   SIFT patent expired in 2020 — available in standard OpenCV 4.x.
//
// How SIFT features could enable AR (for report):
//   1. Detect SIFT keypoints on a reference dollar bill image
//   2. Match keypoints from reference to live frame (BFMatcher/FLANN)
//   3. Use cv::findHomography on matched pairs to get transform
//   4. Project virtual object onto the bill — AR without chessboard
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>

////////////////////////////////////////////////////////////////////////////////
// FeatureDetector
//
// Detects SIFT keypoints on a live video stream pointed at a dollar bill.
// Draws keypoints with scale and orientation indicators.
// A trackbar controls the number of features detected in real time.
////////////////////////////////////////////////////////////////////////////////
class FeatureDetector
{
public:
    // -------------------------------------------------------------------------
    // Constructor
    // cameraId          : webcam device index (default 0)
    // nfeatures         : max SIFT features to detect (default 500)
    // contrastThreshold : filters weak features (default 0.04, lower = more)
    // -------------------------------------------------------------------------
    explicit FeatureDetector(int    cameraId          = 0,
                             int    nfeatures         = 500,
                             double contrastThreshold = 0.04);

    // -------------------------------------------------------------------------
    // run()
    // Opens webcam, detects SIFT keypoints each frame, draws results.
    // Trackbar controls max number of features (0 = unlimited).
    // Controls:
    //   'q' / ESC - quit
    // Returns true if loop ran without errors.
    // -------------------------------------------------------------------------
    bool run();

private:
    // -------------------------------------------------------------------------
    // detectSIFT()
    // Runs SIFT detection on the current frame.
    // Draws keypoints with DRAW_RICH_KEYPOINTS:
    //   - Circle size = keypoint scale
    //   - Line inside = keypoint orientation
    // Returns the number of keypoints detected.
    // -------------------------------------------------------------------------
    int detectSIFT(const cv::Mat& frame, cv::Mat& displayFrame);

    // -------------------------------------------------------------------------
    // overlayInfo()
    // Draws keypoint count, settings, and hints on displayFrame.
    // -------------------------------------------------------------------------
    void overlayInfo(cv::Mat& frame, int keypointCount) const;

    // =========================================================================
    // Member variables
    // =========================================================================
    int    m_cameraId;           // webcam index
    int    m_nfeatures;          // max features — controlled by trackbar
    double m_contrastThreshold;  // SIFT contrast threshold
};
