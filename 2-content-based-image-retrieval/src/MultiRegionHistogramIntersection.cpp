////////////////////////////////////////////////////////////////////////////////
// MultiRegionHistogramIntersection.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of custom multi-region histogram intersection
//              distance metric for Task 3.
//
// This metric explicitly compares each spatial region's histogram separately
// and combines the results using weighted averaging. This is the "distance
// metric of your own design" required by Task 3.
//
// Algorithm:
//   1. Split feature vector into regions
//   2. For each region:
//      - Compute histogram intersection
//      - Convert to distance (1 - intersection)
//   3. Combine region distances with weights
//
// Example with 2 regions (top/bottom):
//   top_intersection = Σ min(top_query[i], top_db[i])
//   bottom_intersection = Σ min(bottom_query[i], bottom_db[i])
//   
//   final_distance = 0.5 × (1 - top_intersection) 
//                  + 0.5 × (1 - bottom_intersection)
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "MultiRegionHistogramIntersection.h"
#include <iostream>
#include <cmath>
#include <numeric>

namespace cbir {

/**
 * Constructor with default equal weights
 */
MultiRegionHistogramIntersection::MultiRegionHistogramIntersection(
    int numRegions,
    int binsPerRegion)
    : numRegions_(numRegions),
      binsPerRegion_(binsPerRegion) {
    
    // Initialize equal weights
    initializeEqualWeights();
}

/**
 * Constructor with custom weights
 */
MultiRegionHistogramIntersection::MultiRegionHistogramIntersection(
    int numRegions,
    int binsPerRegion,
    const std::vector<double>& weights)
    : numRegions_(numRegions),
      binsPerRegion_(binsPerRegion),
      weights_(weights) {
    
    // Validate and normalize weights
    if (weights_.empty()) {
        initializeEqualWeights();
    } else {
        normalizeWeights();
    }
}

/**
 * Compute multi-region histogram intersection distance
 * 
 * This is the core Task 3 custom distance metric implementation.
 * 
 * Process:
 *   1. Validate feature dimensions match expected (numRegions × binsPerRegion)
 *   2. For each region:
 *      - Extract region bounds (startIdx, endIdx)
 *      - Compute histogram intersection for that region
 *      - Convert to distance: d_region = 1 - intersection_region
 *      - Apply region weight
 *   3. Sum weighted region distances
 * 
 * @param features1 First multi-region histogram feature
 * @param features2 Second multi-region histogram feature
 * @return Combined weighted distance [0, 1]
 */
double MultiRegionHistogramIntersection::compute(const cv::Mat& features1, 
                                                const cv::Mat& features2) {
    // Validate feature compatibility
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Features not compatible for multi-region comparison" << std::endl;
        return -1.0;
    }
    
    // Validate feature dimension matches expected multi-region structure
    int expectedDim = numRegions_ * binsPerRegion_;
    if (features1.cols != expectedDim) {
        std::cerr << "Error: Feature dimension mismatch!" << std::endl;
        std::cerr << "  Expected: " << expectedDim 
                  << " (" << numRegions_ << " regions × " 
                  << binsPerRegion_ << " bins per region)" << std::endl;
        std::cerr << "  Got: " << features1.cols << std::endl;
        return -1.0;
    }
    
    // Compute distance for each region separately
    double combinedDistance = 0.0;
    
    for (int region = 0; region < numRegions_; region++) {
        // Define region bounds in feature vector
        int startIdx = region * binsPerRegion_;
        int endIdx = (region + 1) * binsPerRegion_;
        
        // Compute histogram intersection for this region
        double regionIntersection = computeRegionIntersection(
            features1, features2, startIdx, endIdx
        );
        
        // Convert intersection to distance
        double regionDistance = 1.0 - regionIntersection;
        
        // Add weighted contribution to combined distance
        combinedDistance += weights_[region] * regionDistance;
    }
    
    return combinedDistance;
}

/**
 * Get metric name
 */
std::string MultiRegionHistogramIntersection::getMetricName() const {
    return "MultiRegionHistogramIntersection";
}

/**
 * Set custom region weights
 * 
 * Allows prioritizing certain regions over others.
 * Example: {0.6, 0.4} weights top region 60%, bottom 40%
 */
void MultiRegionHistogramIntersection::setWeights(const std::vector<double>& weights) {
    if (weights.size() != static_cast<size_t>(numRegions_)) {
        std::cerr << "Error: Number of weights (" << weights.size() 
                  << ") must match number of regions (" << numRegions_ << ")" << std::endl;
        return;
    }
    
    weights_ = weights;
    normalizeWeights();
}

/**
 * Compute histogram intersection for a single region
 * 
 * Computes: intersection = Σ min(hist1[i], hist2[i]) for i in [startIdx, endIdx)
 * 
 * This is the same formula as regular histogram intersection,
 * but applied only to a specific region of the concatenated feature vector.
 * 
 * @param hist1 First feature vector (contains all regions)
 * @param hist2 Second feature vector (contains all regions)
 * @param startIdx Start index of this region
 * @param endIdx End index of this region (exclusive)
 * @return Intersection value for this region [0, 1]
 */
double MultiRegionHistogramIntersection::computeRegionIntersection(
    const cv::Mat& hist1, 
    const cv::Mat& hist2,
    int startIdx, 
    int endIdx) const {
    
    double intersection = 0.0;
    
    // Sum minimum values for this region's bins only
    for (int i = startIdx; i < endIdx; i++) {
        float val1 = hist1.at<float>(0, i);
        float val2 = hist2.at<float>(0, i);
        intersection += std::min(val1, val2);
    }
    
    return intersection;
}

/**
 * Initialize equal weights for all regions
 * 
 * Each region gets weight = 1.0 / numRegions
 * Example: 2 regions → {0.5, 0.5}
 */
void MultiRegionHistogramIntersection::initializeEqualWeights() {
    weights_.clear();
    double weight = 1.0 / numRegions_;
    
    for (int i = 0; i < numRegions_; i++) {
        weights_.push_back(weight);
    }
}

/**
 * Normalize weights to sum to 1.0
 * 
 * Ensures weighted combination produces valid distance metric.
 * If weights don't sum to 1.0, normalizes them proportionally.
 */
void MultiRegionHistogramIntersection::normalizeWeights() {
    // Compute sum of weights
    double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    
    // Check if already normalized
    if (std::abs(sum - 1.0) < 0.001) {
        return;  // Already normalized
    }
    
    // Normalize if sum is not 1.0
    if (sum > 1e-10) {
        std::cerr << "Warning: Weights sum to " << sum << ", normalizing to 1.0" << std::endl;
        for (auto& w : weights_) {
            w /= sum;
        }
    } else {
        // Invalid weights (sum is zero), reset to equal weights
        std::cerr << "Error: Invalid weights (sum is zero), using equal weights" << std::endl;
        initializeEqualWeights();
    }
}

} // namespace cbir
