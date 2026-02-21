/**
 * @file    RegionFeatures.cpp
 * @brief   Region-based feature extraction implementation.
 *
 *          Moments explanation:
 *            cv::moments() computes spatial moments m00, m10, m01 and
 *            central moments mu20, mu02, mu11, mu30 etc. from a binary mask.
 *
 *            We use:
 *              m00          — region area (zero-order moment)
 *              m10, m01     — used to verify centroid (first-order)
 *              mu20, mu02   — variance in x and y (second-order central)
 *              mu11         — covariance xy (second-order central)
 *
 *            Primary axis angle:
 *              theta = 0.5 * atan2(2 * mu11, mu20 - mu02)
 *              This is the axis of LEAST central moment (minimum inertia).
 *
 *            Hu moments (cv::HuMoments):
 *              7 invariants derived from normalised central moments.
 *              Invariant to translation, scale, and rotation.
 *              Log-scaled: sign(h) * log10(|h|) for numerical stability.
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#include "RegionFeatures.h"
#include <cmath>

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

/** Extract binary mask for a single region label from the label map. */
static cv::Mat extractRegionMask(const cv::Mat& labelMap, int regionId)
{
    cv::Mat mask = cv::Mat::zeros(labelMap.size(), CV_8UC1);
    for (int r = 0; r < labelMap.rows; r++) {
        const int* row = labelMap.ptr<int>(r);
        uchar*    mRow = mask.ptr<uchar>(r);
        for (int c = 0; c < labelMap.cols; c++)
            if (row[c] == regionId) mRow[c] = 255;
    }
    return mask;
}

/** Log-scale a Hu moment for numerical stability.
 *  Takes absolute value first to handle sign inconsistency across orientations.
 */
static double logScale(double h)
{
    if (h == 0.0) return 0.0;
    // Use abs value — sign can flip with orientation, abs is stable
    return std::log10(std::abs(h));
}

// -----------------------------------------------------------------------------
void computeRegionFeatures(const cv::Mat& labelMap, RegionInfo& reg)
{
    // --- Extract binary mask for this region ---------------------------------
    cv::Mat mask = extractRegionMask(labelMap, reg.id);

    // --- Compute moments on the region mask ----------------------------------
    // cv::moments computes spatial and central moments from a binary image.
    // We use central moments (mu**) which are translation invariant.
    cv::Moments m = cv::moments(mask, true); // true = treat as binary

    if (m.m00 < 1.0) return; // empty region guard

    // --- Centroid (verify / update from moments) -----------------------------
    reg.centroid = {
        static_cast<float>(m.m10 / m.m00),
        static_cast<float>(m.m01 / m.m00)
    };

    // --- Primary axis angle (axis of least central moment) -------------------
    // Derived from second-order central moments mu20, mu02, mu11.
    // theta = 0.5 * atan2(2*mu11, mu20-mu02)
    reg.angle = 0.5 * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02);

    // --- Oriented bounding box -----------------------------------------------
    // Find extent along primary axis and perpendicular axis.
    // Project all region pixels onto the two axes to find min/max extent.
    double cosA = std::cos(reg.angle);
    double sinA = std::sin(reg.angle);

    double minProj1 =  1e9, maxProj1 = -1e9; // along primary axis
    double minProj2 =  1e9, maxProj2 = -1e9; // along secondary axis

    for (int r = 0; r < mask.rows; r++) {
        const uchar* row = mask.ptr<uchar>(r);
        for (int c = 0; c < mask.cols; c++) {
            if (row[c] == 0) continue;
            double dx = c - reg.centroid.x;
            double dy = r - reg.centroid.y;
            double p1 =  dx * cosA + dy * sinA;
            double p2 = -dx * sinA + dy * cosA;
            minProj1 = std::min(minProj1, p1);
            maxProj1 = std::max(maxProj1, p1);
            minProj2 = std::min(minProj2, p2);
            maxProj2 = std::max(maxProj2, p2);
        }
    }

    float w = static_cast<float>(maxProj1 - minProj1); // width along primary
    float h = static_cast<float>(maxProj2 - minProj2); // height along secondary

    reg.orientedBox = cv::RotatedRect(reg.centroid,
                                      cv::Size2f(w, h),
                                      static_cast<float>(reg.angle * 180.0 / CV_PI));

    // --- Feature: area (normalised by image area for scale invariance) -------
    double imageArea = static_cast<double>(labelMap.rows * labelMap.cols);
    reg.area = m.m00 / imageArea;

    // --- Feature: fill ratio = region pixels / oriented bbox area ------------
    // Scale and rotation invariant: measures how "solid" the shape is.
    double bboxArea = static_cast<double>(w) * static_cast<double>(h);
    reg.fillRatio = (bboxArea > 0.0) ? m.m00 / bboxArea : 0.0;
    reg.fillRatio = std::min(reg.fillRatio, 1.0); // clamp to [0,1]

    // --- Feature: bbox aspect ratio = max/min side ---------------------------
    // Rotation invariant: measures elongation of the shape.
    float longSide  = std::max(w, h);
    float shortSide = std::min(w, h);
    reg.bboxRatio = (shortSide > 0.f) ? longSide / shortSide : 1.f;

    // --- Feature: Hu moments (7 invariants) ----------------------------------
    // cv::HuMoments derives 7 values from normalised central moments.
    // Invariant to translation, scale, and rotation.
    double hu[7];
    cv::HuMoments(m, hu);

    reg.huMoments.resize(7);
    for (int i = 0; i < 7; i++)
        reg.huMoments[i] = logScale(hu[i]); // log scale for usable range
}

// -----------------------------------------------------------------------------
void computeAllFeatures(const cv::Mat& labelMap, AppState& state)
{
    for (auto& reg : state.regions)
        computeRegionFeatures(labelMap, reg);
}

// -----------------------------------------------------------------------------
void drawFeatures(cv::Mat& frame, const AppState& state,
                  const PipelineParams& params)
{
    for (const auto& reg : state.regions) {
        cv::Scalar col = reg.displayColor;
        cv::Point  cx  = cv::Point(static_cast<int>(reg.centroid.x),
                                   static_cast<int>(reg.centroid.y));

        // --- Primary axis line -----------------------------------------------
        if (params.showAxes) {
            // Length = half the longer side of the oriented bbox
            float len = std::max(reg.orientedBox.size.width,
                                 reg.orientedBox.size.height) * 0.5f;
            double cosA = std::cos(reg.angle);
            double sinA = std::sin(reg.angle);

            cv::Point p1(static_cast<int>(reg.centroid.x - len * cosA),
                         static_cast<int>(reg.centroid.y - len * sinA));
            cv::Point p2(static_cast<int>(reg.centroid.x + len * cosA),
                         static_cast<int>(reg.centroid.y + len * sinA));

            cv::line(frame, p1, p2, col, 2);
            cv::circle(frame, cx, 5, col, -1); // centroid dot
        }

        // --- Oriented bounding box -------------------------------------------
        if (params.showOrientedBBox) {
            cv::Point2f corners[4];
            reg.orientedBox.points(corners);
            for (int i = 0; i < 4; i++)
                cv::line(frame, corners[i], corners[(i+1) % 4], col, 2);
        }

        // --- Feature text overlay --------------------------------------------
        if (params.showFeatureText && !reg.huMoments.empty()) {
            int tx = reg.boundingBox.x;
            int ty = reg.boundingBox.y - 10;
            if (ty < 15) ty = reg.boundingBox.y + reg.boundingBox.height + 20;

            auto putf = [&](const std::string& txt, int line) {
                cv::putText(frame, txt,
                            {tx, ty + line * 18},
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
            };

            putf("Fill:" + std::to_string(reg.fillRatio).substr(0,5), 0);
            putf("BBox:" + std::to_string(reg.bboxRatio).substr(0,5), 1);
            putf("Hu0:"  + std::to_string(reg.huMoments[0]).substr(0,6), 2);

            // Label + confidence in brackets e.g. "scissors (0.84)"
            std::string labelTxt = reg.label;
            if (!reg.label.empty() && reg.label != "unknown" && reg.label != "no DB") {
                std::string conf = std::to_string(reg.confidence);
                labelTxt += " (" + conf.substr(0, 4) + ")";
            }
            putf(labelTxt, 3);
        }
    }
}
