////////////////////////////////////////////////////////////////////////////////
// FaceAwareDistance.h
// Author: Krushna Sanjay Sharma
// Description: Custom distance metric for FaceAwareFeature.
//              Handles both face-mode and non-face mode features.
//
// Distance Computation:
//   Face Mode: [dnn(512), count(1), color(512), spatial(4)] = 1029
//   - 50% DNN (semantic)
//   - 10% Face count (number of people)
//   - 30% Face colors (context/clothing)
//   - 10% Spatial layout (positioning)
//
//   No-Face Mode: [dnn(512), color(512)] = 1024
//   - Uses ProductMatcherDistance (60% DNN, 40% color)
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef FACE_AWARE_DISTANCE_H
#define FACE_AWARE_DISTANCE_H

#include "DistanceMetric.h"

namespace cbir {

/**
 * @class FaceAwareDistance
 * @brief Adaptive distance metric for face-aware features
 * 
 * Handles two feature types:
 *   - Face features (1029D)
 *   - Non-face features (1024D)
 * 
 * @author Krushna Sanjay Sharma
 */
class FaceAwareDistance : public DistanceMetric {
public:
    FaceAwareDistance();
    virtual ~FaceAwareDistance() = default;
    
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;
    
    virtual std::string getMetricName() const override;

private:
    /**
     * Compute distance for face features (1029D)
     */
    double computeFaceDistance(const cv::Mat& f1, const cv::Mat& f2) const;
    
    /**
     * Compute distance for non-face features (1024D)
     */
    double computeNonFaceDistance(const cv::Mat& f1, const cv::Mat& f2) const;
    
    // Helper functions
    double computeCosineDistance(const cv::Mat& f1, const cv::Mat& f2, 
                                int start, int end) const;
    double computeHistogramIntersection(const cv::Mat& f1, const cv::Mat& f2,
                                       int start, int end) const;
    double computeEuclideanDistance(const cv::Mat& f1, const cv::Mat& f2,
                                   int start, int end) const;
};

} // namespace cbir

#endif // FACE_AWARE_DISTANCE_H
