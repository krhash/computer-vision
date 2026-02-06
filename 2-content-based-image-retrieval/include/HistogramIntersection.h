////////////////////////////////////////////////////////////////////////////////
// HistogramIntersection.h
// Author: Krushna Sanjay Sharma
// Description: Histogram intersection distance metric for comparing color
//              histograms. Computes overlap between normalized
//              histograms - higher intersection means more similar images.
////////////////////////////////////////////////////////////////////////////////

#ifndef HISTOGRAM_INTERSECTION_H
#define HISTOGRAM_INTERSECTION_H

#include "DistanceMetric.h"

namespace cbir {

/**
 * @class HistogramIntersection
 * @brief Histogram intersection distance metric
 * 
 * Computes the histogram intersection between two feature vectors:
 * 
 *   intersection = Σ min(H1[i], H2[i])
 *   distance = 1 - intersection
 * 
 * For normalized histograms (sum = 1.0):
 * - intersection ranges from 0 (no overlap) to 1 (identical)
 * - distance ranges from 0 (identical) to 1 (completely different)
 * 
 * Properties:
 * - Works well with color histograms
 * - Robust to lighting variations (especially with chromaticity)
 * - Returns 0 for identical histograms
 * - Lower distance = more similar color distributions
 * 
 * @author Krushna Sanjay Sharma
 */
class HistogramIntersection : public DistanceMetric {
public:
    HistogramIntersection() = default;
    virtual ~HistogramIntersection() = default;
    
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;
    
    virtual std::string getMetricName() const override;
    
private:
    /**
     * @brief Check if histogram is normalized (sum ≈ 1.0)
     */
    bool isNormalized(const cv::Mat& histogram) const;
    
    /**
     * @brief Normalize histogram if needed
     */
    cv::Mat normalizeIfNeeded(const cv::Mat& histogram) const;
};

} // namespace cbir

#endif // HISTOGRAM_INTERSECTION_H
