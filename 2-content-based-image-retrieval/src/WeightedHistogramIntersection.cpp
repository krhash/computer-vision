////////////////////////////////////////////////////////////////////////////////
// WeightedHistogramIntersection.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of weighted histogram intersection for combined
//              texture and color features. Computes separate intersections
//              and combines with equal weights.
//
// Algorithm:
//   1. Split feature vector into texture and color components
//   2. Compute intersection for texture: I_texture = Σ min(T1[i], T2[i])
//   3. Compute intersection for color: I_color = Σ min(C1[i], C2[i])
//   4. Convert to distances: d = 1 - intersection
//   5. Combine: final_distance = 0.5×d_texture + 0.5×d_color
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "WeightedHistogramIntersection.h"
#include <iostream>
#include <cmath>

namespace cbir {

/**
 * Constructor
 * 
 * @param textureDim Number of texture bins
 * @param colorDim Number of color bins
 * @param textureWeight Weight for texture component (default 0.5 = 50%)
 * @param colorWeight Weight for color component (default 0.5 = 50%)
 */
WeightedHistogramIntersection::WeightedHistogramIntersection(int textureDim,
                                                           int colorDim,
                                                           double textureWeight,
                                                           double colorWeight)
    : textureDim_(textureDim),
      colorDim_(colorDim),
      textureWeight_(textureWeight),
      colorWeight_(colorWeight) {
    
    // Validate weights sum to 1.0
    double sum = textureWeight_ + colorWeight_;
    if (std::abs(sum - 1.0) > 0.01) {
        std::cerr << "Warning: Weights should sum to 1.0 (current: " << sum << ")" << std::endl;
        // Normalize weights
        textureWeight_ /= sum;
        colorWeight_ /= sum;
    }
}

/**
 * Compute weighted distance between texture-color features
 * 
 * Feature structure: [texture(16), color(512)]
 * 
 * Process:
 *   1. Extract texture components (first 16 values)
 *   2. Extract color components (next 512 values)
 *   3. Compute intersection for each component separately
 *   4. Convert to distances: d = 1 - intersection
 *   5. Combine with weights: final = 0.5×d_texture + 0.5×d_color
 * 
 * @param features1 First feature vector [texture, color]
 * @param features2 Second feature vector [texture, color]
 * @return Weighted combined distance
 */
double WeightedHistogramIntersection::compute(const cv::Mat& features1, 
                                             const cv::Mat& features2) {
    // Validate compatibility
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Features are not compatible" << std::endl;
        return -1.0;
    }
    
    // Validate total dimension matches texture + color
    int expectedDim = textureDim_ + colorDim_;
    if (features1.cols != expectedDim) {
        std::cerr << "Error: Feature dimension mismatch. Expected " << expectedDim
                  << ", got " << features1.cols << std::endl;
        return -1.0;
    }
    
    // Compute intersection for texture component (bins 0 to textureDim-1)
    double textureIntersection = computeIntersection(features1, features2, 0, textureDim_);
    double textureDistance = 1.0 - textureIntersection;
    
    // Compute intersection for color component (bins textureDim to end)
    double colorIntersection = computeIntersection(features1, features2, 
                                                   textureDim_, expectedDim);
    double colorDistance = 1.0 - colorIntersection;
    
    // Combine with equal weights (50% texture, 50% color)
    double combinedDistance = textureWeight_ * textureDistance + colorWeight_ * colorDistance;
    
    return combinedDistance;
}

/**
 * Get metric name
 */
std::string WeightedHistogramIntersection::getMetricName() const {
    return "WeightedHistogramIntersection";
}

/**
 * Compute histogram intersection for a subrange
 * 
 * Computes: Σ min(hist1[i], hist2[i]) for i in [startIdx, endIdx)
 * 
 * @param hist1 First histogram
 * @param hist2 Second histogram
 * @param startIdx Start index (inclusive)
 * @param endIdx End index (exclusive)
 * @return Intersection value
 */
double WeightedHistogramIntersection::computeIntersection(const cv::Mat& hist1, 
                                                         const cv::Mat& hist2,
                                                         int startIdx, 
                                                         int endIdx) const {
    double intersection = 0.0;
    
    // Sum of minimums for specified range
    for (int i = startIdx; i < endIdx; i++) {
        float val1 = hist1.at<float>(0, i);
        float val2 = hist2.at<float>(0, i);
        intersection += std::min(val1, val2);
    }
    
    return intersection;
}

/**
 * Check if histogram component is normalized
 * 
 * @param histogram Feature vector
 * @param startIdx Start index of component
 * @param endIdx End index of component
 * @return True if sum ≈ 1.0
 */
bool WeightedHistogramIntersection::isNormalized(const cv::Mat& histogram,
                                                int startIdx,
                                                int endIdx) const {
    double sum = 0.0;
    
    for (int i = startIdx; i < endIdx; i++) {
        sum += histogram.at<float>(0, i);
    }
    
    return std::abs(sum - 1.0) < 0.01;
}

} // namespace cbir
