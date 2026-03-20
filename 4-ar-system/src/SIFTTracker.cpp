////////////////////////////////////////////////////////////////////////////////
// SIFTTracker.cpp - SIFT-based AR Tracker Implementation
// Author:      Krushna Sanjay Sharma
// Description: Implements SIFT feature matching and pose estimation for AR
//              on a dollar bill target. Replaces chessboard with texture-based
//              tracking using the same camera calibration.
//
// Extension:   Uber Extension 2 — AR with SIFT feature points
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#include "SIFTTracker.h"
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
SIFTTracker::SIFTTracker(const std::string& calibrationFile,
                          const std::string& referenceImage,
                          int nfeatures)
    : m_calibrationFile(calibrationFile)
    , m_referenceImagePath(referenceImage)
    , m_nfeatures(nfeatures)
    , m_tracking(false)
    , m_inlierCount(0)
    , m_hasSmooth(false)
    , m_smoothAlpha(0.35f)   // 0.35 = responsive but smooth
                              // lower = smoother but more lag
                              // higher = less lag but shakier
{
    m_rvec       = cv::Mat::zeros(3, 1, CV_64F);
    m_tvec       = cv::Mat::zeros(3, 1, CV_64F);
    m_rvecSmooth = cv::Mat::zeros(3, 1, CV_64F);
    m_tvecSmooth = cv::Mat::zeros(3, 1, CV_64F);
}

// ----------------------------------------------------------------------------
// initialize()
// ----------------------------------------------------------------------------
bool SIFTTracker::initialize()
{
    if (!loadCalibration()) return false;
    if (!loadReference())   return false;
    return true;
}

// ----------------------------------------------------------------------------
// loadCalibration()
// ----------------------------------------------------------------------------
bool SIFTTracker::loadCalibration()
{
    cv::FileStorage fs(m_calibrationFile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "[ERROR] Cannot open calibration: " << m_calibrationFile << "\n";
        return false;
    }

    fs["camera_matrix"]           >> m_cameraMatrix;
    fs["distortion_coefficients"] >> m_distCoeffs;
    fs.release();

    if (m_cameraMatrix.empty() || m_distCoeffs.empty())
    {
        std::cerr << "[ERROR] Calibration file missing keys.\n";
        return false;
    }

    std::cout << "[INFO] Calibration loaded.\n";
    return true;
}

// ----------------------------------------------------------------------------
// loadReference()
// Loads reference bill image and computes SIFT keypoints + descriptors.
// These are computed once and reused every frame.
// ----------------------------------------------------------------------------
bool SIFTTracker::loadReference()
{
    m_refImage = cv::imread(m_referenceImagePath, cv::IMREAD_COLOR);
    if (m_refImage.empty())
    {
        std::cerr << "[ERROR] Cannot load reference image: "
                  << m_referenceImagePath << "\n";
        return false;
    }

    // More features + lower contrast = more candidates to match from
    // 300 features with threshold 0.04 gives good coverage of bill details
    m_sift  = cv::SIFT::create(m_nfeatures, 3, 0.04, 10, 1.6);
    m_clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    cv::Mat refGray, refEnhanced;
    cv::cvtColor(m_refImage, refGray, cv::COLOR_BGR2GRAY);
    m_clahe->apply(refGray, refEnhanced);

    m_sift->detectAndCompute(
        refEnhanced,
        cv::noArray(),
        m_refKeypoints,
        m_refDescriptors
    );

    std::cout << "[INFO] Reference image loaded: " << m_referenceImagePath << "\n";
    std::cout << "[INFO] Reference keypoints: "    << m_refKeypoints.size() << "\n";
    std::cout << "[INFO] Reference size: "
              << m_refImage.cols << " x " << m_refImage.rows << " px\n";

    return true;
}

// ----------------------------------------------------------------------------
// track()
// Main per-frame tracking function.
// ----------------------------------------------------------------------------
bool SIFTTracker::track(const cv::Mat& frame)
{
    m_tracking    = false;
    m_inlierCount = 0;

    // Convert to grayscale and apply CLAHE — reuse m_clahe created in initialize()
    cv::Mat gray, enhanced;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    m_clahe->apply(gray, enhanced);

    std::vector<cv::KeyPoint> liveKeypoints;
    cv::Mat                   liveDescriptors;
    m_sift->detectAndCompute(enhanced, cv::noArray(),
                              liveKeypoints, liveDescriptors);

    if (liveDescriptors.empty() || liveKeypoints.empty()) return false;

    // Step 2: match descriptors
    std::vector<cv::DMatch> goodMatches = matchFeatures(liveDescriptors);
    if (static_cast<int>(goodMatches.size()) < MIN_INLIERS) return false;

    // Save for debug drawing
    m_lastLiveKeypoints = liveKeypoints;
    m_lastGoodMatches   = goodMatches;

    // Step 3: estimate pose
    m_tracking = estimatePose(liveKeypoints, goodMatches);

    // Step 4: apply exponential moving average smoothing to reduce shakiness
    // Blends current raw pose with previous smoothed pose:
    //   smoothed = alpha * raw + (1 - alpha) * previous_smoothed
    // This reduces frame-to-frame jitter without introducing much lag.
    if (m_tracking)
    {
        if (!m_hasSmooth)
        {
            // First successful frame — initialize smoother with raw pose
            m_rvec.copyTo(m_rvecSmooth);
            m_tvec.copyTo(m_tvecSmooth);
            m_hasSmooth = true;
        }
        else
        {
            // Blend raw pose toward smoothed pose
            m_rvecSmooth = m_smoothAlpha * m_rvec
                         + (1.0 - m_smoothAlpha) * m_rvecSmooth;
            m_tvecSmooth = m_smoothAlpha * m_tvec
                         + (1.0 - m_smoothAlpha) * m_tvecSmooth;
        }
    }

    return m_tracking;
}

// ----------------------------------------------------------------------------
// matchFeatures()
// BFMatcher with L2 norm (correct for SIFT float descriptors).
// Applies Lowe's ratio test: keeps match only if best match is significantly
// better than second best (ratio < 0.75).
// ----------------------------------------------------------------------------
std::vector<cv::DMatch> SIFTTracker::matchFeatures(
    const cv::Mat& liveDescriptors) const
{
    // BFMatcher with L2 norm — appropriate for SIFT float descriptors
    cv::BFMatcher matcher(cv::NORM_L2);

    // knnMatch: find 2 best matches per descriptor
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(liveDescriptors, m_refDescriptors, knnMatches, 2);

    // Lowe's ratio test — 0.7 is standard, use 0.65 for stricter matching
    // Lower ratio = fewer but more reliable matches = less shakiness
    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches)
    {
        if (m.size() == 2 && m[0].distance < 0.65f * m[1].distance)
            goodMatches.push_back(m[0]);
    }

    return goodMatches;
}

// ----------------------------------------------------------------------------
// estimatePose()
// Uses findHomography+RANSAC to identify inliers, then builds 3D-2D
// correspondences and runs solvePnP.
//
// 3D world points: reference image pixels mapped to bill surface (cm units)
//   refPointTo3D() converts 2D reference pixel → 3D world cm coordinate
//   Z = 0 for all points (flat planar surface)
// ----------------------------------------------------------------------------
bool SIFTTracker::estimatePose(
    const std::vector<cv::KeyPoint>& liveKeypoints,
    const std::vector<cv::DMatch>&   goodMatches)
{
    // Build point correspondence vectors
    std::vector<cv::Point2f> refPts, livePts;
    for (const auto& m : goodMatches)
    {
        refPts.push_back(m_refKeypoints[m.trainIdx].pt);
        livePts.push_back(liveKeypoints[m.queryIdx].pt);
    }

    // RANSAC threshold 5.0px — more tolerant of camera noise
    // giving more inliers while still filtering wrong matches
    cv::Mat inlierMask;
    cv::Mat H = cv::findHomography(refPts, livePts,
                                    cv::RANSAC, 5.0, inlierMask);

    if (H.empty()) return false;

    // Count inliers
    std::vector<bool> inliers(inlierMask.rows);
    int inlierCount = 0;
    for (int i = 0; i < inlierMask.rows; ++i)
    {
        inliers[i] = (inlierMask.at<uchar>(i) != 0);
        if (inliers[i]) ++inlierCount;
    }
    m_lastInlierMask = inliers;
    m_inlierCount    = inlierCount;

    if (inlierCount < MIN_INLIERS) return false;

    // Build 3D-2D correspondences from inliers only
    // 3D: reference pixel → cm world coordinate on bill surface (Z=0)
    // 2D: corresponding live frame pixel
    std::vector<cv::Vec3f>   objPoints;
    std::vector<cv::Point2f> imgPoints;

    for (int i = 0; i < static_cast<int>(goodMatches.size()); ++i)
    {
        if (!inliers[i]) continue;
        objPoints.push_back(refPointTo3D(refPts[i]));
        imgPoints.push_back(livePts[i]);
    }

    // solvePnP: compute rotation and translation
    // Uses same calibration matrix as Tasks 4-6
    try
    {
        // Use previous pose as initial guess when available —
        // significantly improves stability at steep angles
        bool ok = cv::solvePnP(
            objPoints,
            imgPoints,
            m_cameraMatrix,
            m_distCoeffs,
            m_rvec,
            m_tvec,
            m_hasSmooth,              // useExtrinsicGuess = true after first frame
            cv::SOLVEPNP_ITERATIVE
        );
        return ok;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[WARN] solvePnP failed: " << e.what() << "\n";
        return false;
    }
}

// ----------------------------------------------------------------------------
// refPointTo3D()
// Maps a 2D reference image point to a 3D world point on the bill surface.
//
// Reference image covers the full bill:
//   pixel (0,0)              → world (0, 0, 0)        top-left corner
//   pixel (refW, 0)          → world (BILL_WIDTH, 0, 0)
//   pixel (0, refH)          → world (0, BILL_HEIGHT, 0)
//   pixel (refW/2, refH/2)   → world (BILL_WIDTH/2, BILL_HEIGHT/2, 0)
// ----------------------------------------------------------------------------
cv::Vec3f SIFTTracker::refPointTo3D(const cv::Point2f& refPt) const
{
    float xCm = (refPt.x / m_refImage.cols) * BILL_WIDTH_CM;
    float yCm = (refPt.y / m_refImage.rows) * BILL_HEIGHT_CM;
    return cv::Vec3f(xCm, yCm, 0.0f);
}

// ----------------------------------------------------------------------------
// drawDebug()
// Draws inlier matches and tracking status on the display frame.
// ----------------------------------------------------------------------------
void SIFTTracker::drawDebug(cv::Mat& displayFrame) const
{
    auto putText2 = [&](const std::string& t, cv::Point p,
                        double s, cv::Scalar c)
    {
        cv::putText(displayFrame, t, p, cv::FONT_HERSHEY_SIMPLEX, s,
                    cv::Scalar(0,0,0), 3);
        cv::putText(displayFrame, t, p, cv::FONT_HERSHEY_SIMPLEX, s, c, 1);
    };

    // Draw inlier keypoints as small circles
    if (m_tracking)
    {
        for (int i = 0; i < static_cast<int>(m_lastGoodMatches.size()); ++i)
        {
            if (i < static_cast<int>(m_lastInlierMask.size()) &&
                m_lastInlierMask[i])
            {
                cv::Point2f pt = m_lastLiveKeypoints[
                    m_lastGoodMatches[i].queryIdx].pt;
                cv::circle(displayFrame, pt, 4,
                           cv::Scalar(0, 165, 255), 1);
            }
        }
    }

    // Status line
    cv::Scalar statusColor = m_tracking
        ? cv::Scalar(0, 255, 0)
        : cv::Scalar(0, 0, 255);
    std::string statusStr = m_tracking
        ? "Tracking bill - pose estimated"
        : "No bill detected";
    putText2(statusStr, cv::Point(10, 30), 0.65, statusColor);

    // Inlier count
    std::ostringstream ss;
    ss << "Inliers: " << m_inlierCount
       << "  (need >= " << MIN_INLIERS << ")";
    putText2(ss.str(), cv::Point(10, 55), 0.55, cv::Scalar(0, 255, 255));

    // Instructions
    putText2("Point camera at the dollar bill reference image",
             cv::Point(10, 78), 0.5, cv::Scalar(180, 220, 180));
    putText2("'r' toggle rocket  |  'q' quit",
             cv::Point(10, displayFrame.rows - 15), 0.5,
             cv::Scalar(180, 180, 180));
}
