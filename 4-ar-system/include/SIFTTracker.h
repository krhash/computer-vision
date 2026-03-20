////////////////////////////////////////////////////////////////////////////////
// SIFTTracker.h - SIFT-based AR Tracker Header
// Author:      Krushna Sanjay Sharma
// Description: Declares the SIFTTracker class which replaces the chessboard
//              target with a dollar bill tracked via SIFT feature matching.
//              Uses the same calibration XML from Tasks 1-3.
//              Pose is estimated with cv::solvePnP on matched feature points.
//
// Extension:   Uber Extension 2 — AR with SIFT feature points
//
// Pipeline per frame:
//   1. Detect SIFT keypoints on live frame
//   2. Match against pre-computed reference keypoints (BFMatcher + ratio test)
//   3. Filter outliers with cv::findHomography (RANSAC)
//   4. Map inlier 2D reference points → 3D world points on bill surface
//   5. cv::solvePnP → rvec, tvec
//   6. cv::projectPoints → draw rocket via VirtualObject
//
// Key OpenCV functions:
//   cv::SIFT::create()          - SIFT detector
//   cv::BFMatcher               - brute-force descriptor matching
//   cv::findHomography()        - compute homography + RANSAC outlier rejection
//   cv::solvePnP()              - estimate pose from 3D-2D correspondences
//   cv::projectPoints()         - project 3D object onto image (in VirtualObject)
//
// World coordinate convention:
//   The dollar bill is treated as a flat plane (Z=0).
//   Origin (0,0,0) = top-left corner of the bill.
//   X increases rightward (bill width = BILL_WIDTH_CM units).
//   Y increases downward (bill height = BILL_HEIGHT_CM units).
//   Z positive = toward the camera (above the bill surface).
//   Units = centimeters.
//
// Date: March 2026
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// SIFTTracker
//
// Loads a reference image of the dollar bill and pre-computes its SIFT
// keypoints and descriptors. Each frame, detects SIFT on the live image,
// matches against the reference, and estimates pose with solvePnP.
////////////////////////////////////////////////////////////////////////////////
class SIFTTracker
{
public:
    // Real-world dimensions of a US dollar bill in centimeters
    static constexpr float BILL_WIDTH_CM  = 15.6f;
    static constexpr float BILL_HEIGHT_CM =  6.6f;

    // Minimum inlier matches required to accept a pose estimate
    static constexpr int MIN_INLIERS = 8;

    // -------------------------------------------------------------------------
    // Constructor
    // calibrationFile : path to calibration XML (from calibrateCamera app)
    // referenceImage  : path to a flat photo of the dollar bill
    // nfeatures       : max SIFT features per frame (0 = unlimited)
    //                   Higher = more matches but slower. 800 is a good balance.
    // -------------------------------------------------------------------------
    SIFTTracker(const std::string& calibrationFile,
                const std::string& referenceImage,
                int nfeatures = 300);

    // -------------------------------------------------------------------------
    // initialize()
    // Loads calibration and reference image, computes reference SIFT features.
    // Must be called before track(). Returns false on failure.
    // -------------------------------------------------------------------------
    bool initialize();

    // -------------------------------------------------------------------------
    // track()
    // Detects SIFT features in the live frame, matches to reference,
    // estimates pose via solvePnP. Updates rvec/tvec if successful.
    // Returns true if tracking succeeded this frame.
    // -------------------------------------------------------------------------
    bool track(const cv::Mat& frame);

    // -------------------------------------------------------------------------
    // drawDebug()
    // Draws matched keypoints and inlier count on displayFrame.
    // -------------------------------------------------------------------------
    void drawDebug(cv::Mat& displayFrame) const;

    // -------------------------------------------------------------------------
    // Accessors — return SMOOTHED pose for stable rendering
    // -------------------------------------------------------------------------
    const cv::Mat& getRvec()         const { return m_hasSmooth ? m_rvecSmooth : m_rvec; }
    const cv::Mat& getTvec()         const { return m_hasSmooth ? m_tvecSmooth : m_tvec; }
    const cv::Mat& getCameraMatrix() const { return m_cameraMatrix; }
    const cv::Mat& getDistCoeffs()   const { return m_distCoeffs; }
    bool           isTracking()      const { return m_tracking; }
    int            getInlierCount()  const { return m_inlierCount; }

private:
    // -------------------------------------------------------------------------
    // loadCalibration() - reads camera_matrix and distortion_coefficients
    // -------------------------------------------------------------------------
    bool loadCalibration();

    // -------------------------------------------------------------------------
    // loadReference() - loads reference image and computes SIFT features
    // -------------------------------------------------------------------------
    bool loadReference();

    // -------------------------------------------------------------------------
    // matchFeatures()
    // Matches live frame descriptors against reference descriptors.
    // Applies Lowe's ratio test to filter weak matches.
    // Returns good matches.
    // -------------------------------------------------------------------------
    std::vector<cv::DMatch> matchFeatures(
        const cv::Mat& liveDescriptors) const;

    // -------------------------------------------------------------------------
    // estimatePose()
    // Converts good matches to 3D-2D correspondences and runs solvePnP.
    // Uses findHomography+RANSAC to filter outliers first.
    // Returns true on success.
    // -------------------------------------------------------------------------
    bool estimatePose(
        const std::vector<cv::KeyPoint>& liveKeypoints,
        const std::vector<cv::DMatch>&   goodMatches);

    // -------------------------------------------------------------------------
    // refPointTo3D()
    // Maps a 2D reference image point to a 3D world point on the bill surface.
    // Reference image pixel → normalized bill coordinates → cm world coords.
    // Z = 0 (flat surface).
    // -------------------------------------------------------------------------
    cv::Vec3f refPointTo3D(const cv::Point2f& refPt) const;

    // =========================================================================
    // Member variables
    // =========================================================================
    std::string m_calibrationFile;
    std::string m_referenceImagePath;
    int         m_nfeatures;

    // Calibration
    cv::Mat m_cameraMatrix;
    cv::Mat m_distCoeffs;

    // Reference image data
    cv::Mat                      m_refImage;
    std::vector<cv::KeyPoint>    m_refKeypoints;
    cv::Mat                      m_refDescriptors;

    // SIFT detector and CLAHE enhancer — created once, reused every frame
    cv::Ptr<cv::SIFT>  m_sift;
    cv::Ptr<cv::CLAHE> m_clahe;

    // Current frame tracking state
    cv::Mat m_rvec;
    cv::Mat m_tvec;
    bool    m_tracking;
    int     m_inlierCount;

    // Smoothed pose — exponential moving average to reduce shakiness
    cv::Mat m_rvecSmooth;
    cv::Mat m_tvecSmooth;
    bool    m_hasSmooth;          // true after first successful pose
    float   m_smoothAlpha;        // blend factor: 0=no update, 1=raw pose

    // Last good matches for debug drawing
    std::vector<cv::KeyPoint> m_lastLiveKeypoints;
    std::vector<cv::DMatch>   m_lastGoodMatches;
    std::vector<bool>         m_lastInlierMask;
};
