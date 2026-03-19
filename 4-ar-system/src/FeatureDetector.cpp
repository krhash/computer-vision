////////////////////////////////////////////////////////////////////////////////
// FeatureDetector.cpp - Feature Detector Class Implementation
// Author:      Krushna Sanjay Sharma
// Description: Implements SIFT feature detection on a live video stream.
//              Uses cv::SIFT::create() and detectAndCompute() to find
//              keypoints on a dollar bill pattern. Draws keypoints with
//              scale and orientation indicators using cv::drawKeypoints().
//
// Task coverage:
//   Task 7 - Detect and visualize SIFT features on a dollar bill
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "FeatureDetector.h"
#include <iostream>
#include <iomanip>
#include <sstream>

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
FeatureDetector::FeatureDetector(int cameraId, int nfeatures,
                                  double contrastThreshold)
    : m_cameraId(cameraId)
    , m_nfeatures(nfeatures)
    , m_contrastThreshold(contrastThreshold)
{
}

// ----------------------------------------------------------------------------
// run() - Main video loop for Task 7
// ----------------------------------------------------------------------------
bool FeatureDetector::run()
{
    cv::VideoCapture cap(m_cameraId);
    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] Cannot open camera ID: " << m_cameraId << "\n";
        return false;
    }

    std::cout << "========================================\n";
    std::cout << " Feature Detector - Task 7\n";
    std::cout << "========================================\n";
    std::cout << " Pattern : Dollar bill / banknote\n";
    std::cout << " Method  : SIFT\n";
    std::cout << " Controls:\n";
    std::cout << "   Trackbar - max features (0 = unlimited)\n";
    std::cout << "   'q'/ESC  - quit\n";
    std::cout << "========================================\n\n";

    const std::string windowName = "SIFT Feature Detector - Dollar Bill";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Trackbar: controls max SIFT features
    // 0 = unlimited, high values show only strongest features
    cv::createTrackbar("Max features", windowName, &m_nfeatures, 1000);

    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat display = frame.clone();

        int count = detectSIFT(frame, display);
        overlayInfo(display, count);

        cv::imshow(windowName, display);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 'q' || key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return true;
}

// ----------------------------------------------------------------------------
// detectSIFT() - Core of Task 7
//
// Pipeline:
//   1. Convert frame to grayscale (SIFT requires grayscale)
//   2. Create SIFT detector with current m_nfeatures setting
//   3. detectAndCompute → keypoints + 128-dim descriptors
//   4. drawKeypoints with DRAW_RICH_KEYPOINTS:
//      circle size = scale, line = orientation
// ----------------------------------------------------------------------------
int FeatureDetector::detectSIFT(const cv::Mat& frame, cv::Mat& displayFrame)
{
    // Step 1: grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Step 2: create SIFT detector
    // Re-created each frame so trackbar changes apply immediately
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(
        m_nfeatures,          // max features (0 = unlimited)
        3,                    // nOctaveLayers
        m_contrastThreshold,  // contrast threshold
        10,                   // edgeThreshold
        1.6                   // sigma
    );

    // Step 3: detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(
        gray,           // input
        cv::noArray(),  // no mask
        keypoints,      // OUTPUT: keypoints
        descriptors     // OUTPUT: 128-dim descriptors
    );

    // Step 4: draw with rich flags (scale + orientation)
    cv::drawKeypoints(
        frame,
        keypoints,
        displayFrame,
        cv::Scalar(0, 165, 255),                        // orange
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS       // circle=scale, line=angle
    );

    return static_cast<int>(keypoints.size());
}

// ----------------------------------------------------------------------------
// overlayInfo()
// ----------------------------------------------------------------------------
void FeatureDetector::overlayInfo(cv::Mat& frame, int keypointCount) const
{
    auto putText2 = [&](const std::string& text, cv::Point pos,
                        double scale, cv::Scalar color)
    {
        cv::putText(frame, text, pos,
                    cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0,0,0), 3);
        cv::putText(frame, text, pos,
                    cv::FONT_HERSHEY_SIMPLEX, scale, color, 1);
    };

    std::ostringstream ss;
    ss << "SIFT keypoints: " << keypointCount;
    putText2(ss.str(), cv::Point(10, 30), 0.65, cv::Scalar(0, 255, 255));

    std::ostringstream ms;
    ms << "Max features: "
       << (m_nfeatures == 0 ? "unlimited" : std::to_string(m_nfeatures));
    putText2(ms.str(), cv::Point(10, 55), 0.55, cv::Scalar(0, 165, 255));

    putText2("Circle = scale,  line = orientation",
             cv::Point(10, 78), 0.5, cv::Scalar(200, 200, 200));

    putText2("Point at dollar bill or any textured pattern",
             cv::Point(10, 100), 0.5, cv::Scalar(180, 220, 180));

    putText2("q/ESC: quit",
             cv::Point(10, frame.rows - 15), 0.5, cv::Scalar(180, 180, 180));
}
