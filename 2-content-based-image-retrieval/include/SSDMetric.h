////////////////////////////////////////////////////////////////////////////////
// SSDMetric.h
// Author: Krushna Sanjay Sharma
// Description: Sum of Squared Differences (SSD) distance metric for comparing
//              feature vectors. Lower SSD values indicate higher similarity.
//              Returns 0 for identical features.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef SSD_METRIC_H
#define SSD_METRIC_H

#include "DistanceMetric.h"

namespace cbir {

/**
 * @class SSDMetric
 * @brief Sum of Squared Differences (L2 distance) metric
 * 
 * Computes the sum of squared differences between two feature vectors:
 * 
 *   SSD = Σ(f1[i] - f2[i])²
 * 
 * Where f1 and f2 are the two feature vectors being compared.
 * 
 * Properties:
 * - Returns 0 when comparing identical features
 * - Always returns non-negative values
 * - Larger values indicate greater dissimilarity
 * - Equivalent to squared Euclidean distance
 * 
 * Usage example:
 * @code
 *   SSDMetric metric;
 *   cv::Mat features1 = extractor.extractFeatures(image1);
 *   cv::Mat features2 = extractor.extractFeatures(image2);
 *   double distance = metric.compute(features1, features2);
 * @endcode
 * 
 * @author Krushna Sanjay Sharma
 */
class SSDMetric : public DistanceMetric {
public:
    /**
     * @brief Default constructor
     */
    SSDMetric() = default;

    /**
     * @brief Destructor
     */
    virtual ~SSDMetric() = default;

    /**
     * @brief Compute sum of squared differences between two feature vectors
     * 
     * Calculates: SSD = Σ(f1[i] - f2[i])²
     * 
     * Implementation written manually without using OpenCV functions
     * as required by the project specifications.
     * 
     * @param features1 First feature vector
     * @param features2 Second feature vector
     * @return double Sum of squared differences (0 = identical, larger = more different)
     * 
     * @note Returns -1.0 if features are incompatible (different dimensions)
     * @note This implementation does NOT use OpenCV functions - written manually
     */
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;

    /**
     * @brief Get the name of this distance metric
     * 
     * @return std::string "SSD" (Sum of Squared Differences)
     */
    virtual std::string getMetricName() const override;
};

} // namespace cbir

#endif // SSD_METRIC_H
