////////////////////////////////////////////////////////////////////////////////
// WeightedHistogramIntersection.h
// Author: Krushna Sanjay Sharma
// Description: Weighted histogram intersection for combined features.
//              Computes separate intersections for texture and color components
//              and combines them with equal weights (50% each).
//
// For TextureColorFeature:
//   - First N bins: texture histogram
//   - Remaining M bins: color histogram
//   - distance = 0.5 × texture_distance + 0.5 × color_distance
//
// This ensures texture and color contribute equally to the final distance.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef WEIGHTED_HISTOGRAM_INTERSECTION_H
#define WEIGHTED_HISTOGRAM_INTERSECTION_H

#include "DistanceMetric.h"

namespace cbir {

/**
 * @class WeightedHistogramIntersection
 * @brief Histogram intersection with separate weights for components
 * 
 * For TextureColorFeature with structure [texture(16), color(512)]:
 *   1. Compute intersection for texture component (bins 0-15)
 *   2. Compute intersection for color component (bins 16-527)
 *   3. Combine: distance = w1×d_texture + w2×d_color
 *   
 * Default: equal weights (0.5, 0.5)
 * 
 * @author Krushna Sanjay Sharma
 */
class WeightedHistogramIntersection : public DistanceMetric {
public:
    /**
     * Constructor with component dimensions
     * 
     * @param textureDim Dimension of texture component (default: 16)
     * @param colorDim Dimension of color component (default: 512)
     * @param textureWeight Weight for texture (default: 0.5)
     * @param colorWeight Weight for color (default: 0.5)
     */
    WeightedHistogramIntersection(int textureDim = 16,
                                 int colorDim = 512,
                                 double textureWeight = 0.5,
                                 double colorWeight = 0.5);
    
    virtual ~WeightedHistogramIntersection() = default;
    
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;
    
    virtual std::string getMetricName() const override;

private:
    int textureDim_;      ///< Dimension of texture component
    int colorDim_;        ///< Dimension of color component
    double textureWeight_; ///< Weight for texture distance
    double colorWeight_;   ///< Weight for color distance
    
    /**
     * Compute histogram intersection for a subrange of features
     */
    double computeIntersection(const cv::Mat& hist1, const cv::Mat& hist2,
                              int startIdx, int endIdx) const;
    
    /**
     * Check if histogram component is normalized
     */
    bool isNormalized(const cv::Mat& histogram, int startIdx, int endIdx) const;
};

} // namespace cbir

#endif // WEIGHTED_HISTOGRAM_INTERSECTION_H
