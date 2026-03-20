/*
 * FeatureDetector.h - Feature Detector Class Header
 * Author:      Krushna Sanjay Sharma
 * Description: Declares the FeatureDetector class responsible for detecting
 *              SIFT features in a live video stream and visualizing them.
 *              Uses a dollar bill / banknote as the target pattern.
 *              SIFT detector is cached and only recreated when trackbar changes.
 *
 * Task coverage:
 *   Task 7 - Detect robust SIFT features in a live video stream.
 *            Show where features appear on a dollar bill pattern.
 *            Experiment with max features via trackbar.
 *
 * Key OpenCV functions used:
 *   cv::SIFT::create()           - creates SIFT detector instance
 *   sift->detectAndCompute()     - detects keypoints + 128-dim descriptors
 *   cv::drawKeypoints()          - draws keypoints with scale + orientation
 *   cv::createTrackbar()         - runtime feature count adjustment
 *
 * Date: March 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>

/*
 * FeatureDetector
 *
 * Detects SIFT keypoints on a live video stream pointed at a dollar bill.
 * Draws keypoints with scale and orientation indicators.
 * A trackbar controls the number of features detected in real time.
 * SIFT detector is cached and only recreated when trackbar value changes.
 */
class FeatureDetector
{
public:
    explicit FeatureDetector(int    cameraId          = 0,
                             int    nfeatures         = 500,
                             double contrastThreshold = 0.04);

    bool run();

private:
    int  detectSIFT(const cv::Mat& frame, cv::Mat& displayFrame);
    void overlayInfo(cv::Mat& frame, int keypointCount) const;

    /* Member variables */
    int    m_cameraId;
    int    m_nfeatures;          // controlled by trackbar
    double m_contrastThreshold;

    // Cached SIFT detector — recreated only when m_nfeatures changes
    cv::Ptr<cv::SIFT> m_sift;
    int               m_lastNfeatures;  // last value used to build m_sift
};
