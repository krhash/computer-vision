////////////////////////////////////////////////////////////////////////////////
// ProductMatcherDistance.h
// Author: Krushna Sanjay Sharma
// Description: Custom distance metric for ProductMatcherFeature (Task 7).
//              Combines cosine distance (for DNN) with histogram intersection
//              (for color) using 60/40 weighting.
//
// Distance Computation:
//   Feature: [dnn(512), center_color(512)]
//   
//   1. DNN distance (cosine): Compare bins 0-511
//   2. Color distance (histogram intersection): Compare bins 512-1023
//   3. Weighted combination: 60% DNN + 40% color
//
// Weighting Rationale:
//   - DNN (60%): Object type is PRIMARY - ensures semantic correctness
//   - Color (40%): Appearance is SECONDARY - discriminates within category
//
// Example:
//   Red toy vs Blue toy: DNN matches (both toys), color differs → medium distance
//   Red toy vs Red wall: DNN differs (toy vs wall), color matches → high distance
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef PRODUCT_MATCHER_DISTANCE_H
#define PRODUCT_MATCHER_DISTANCE_H

#include "DistanceMetric.h"

namespace cbir {

/**
 * @class ProductMatcherDistance
 * @brief Custom weighted distance for product matching
 * 
 * Combines two distance metrics with different weights:
 *   - Cosine distance for DNN embeddings (semantic)
 *   - Histogram intersection for center-region color (appearance)
 * 
 * Weight ratio 60:40 prioritizes semantic correctness over color matching.
 * 
 * @author Krushna Sanjay Sharma
 */
class ProductMatcherDistance : public DistanceMetric {
public:
    /**
     * Constructor with default weights
     * 
     * Default: 60% DNN, 40% color
     */
    ProductMatcherDistance();
    
    /**
     * Constructor with custom weights
     * 
     * @param dnnWeight Weight for DNN component (0-1)
     * @param colorWeight Weight for color component (0-1)
     */
    ProductMatcherDistance(double dnnWeight, double colorWeight);
    
    virtual ~ProductMatcherDistance() = default;
    
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;
    
    virtual std::string getMetricName() const override;

private:
    static const int DNN_DIM = 512;      ///< DNN component dimension
    static const int COLOR_DIM = 512;    ///< Color component dimension
    static const int TOTAL_DIM = 1024;   ///< Total feature dimension
    
    double dnnWeight_;    ///< Weight for DNN distance
    double colorWeight_;  ///< Weight for color distance
    
    /**
     * Compute cosine distance for DNN component (bins 0-511)
     */
    double computeDNNDistance(const cv::Mat& f1, const cv::Mat& f2) const;
    
    /**
     * Compute histogram intersection for color component (bins 512-1023)
     */
    double computeColorDistance(const cv::Mat& f1, const cv::Mat& f2) const;
    
    /**
     * Normalize weights to sum to 1.0
     */
    void normalizeWeights();
    
    /**
     * Compute L2 norm for cosine distance
     */
    double computeL2Norm(const cv::Mat& features, int startIdx, int endIdx) const;
    
    /**
     * Compute dot product for cosine distance
     */
    double computeDotProduct(const cv::Mat& f1, const cv::Mat& f2, 
                            int startIdx, int endIdx) const;
};

} // namespace cbir

#endif // PRODUCT_MATCHER_DISTANCE_H
