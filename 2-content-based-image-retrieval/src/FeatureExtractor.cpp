////////////////////////////////////////////////////////////////////////////////
// FeatureExtractor.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of the FeatureExtractor base class providing
//              common functionality for all feature extraction methods.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "FeatureExtractor.h"
#include <iostream>

namespace cbir {

/**
 * @brief Normalize feature vector to [0, 1] range
 * 
 * This utility function normalizes a feature vector so that all values fall
 * within the range [0, 1]. This is useful for ensuring features from different
 * extractors have comparable ranges.
 * 
 * @param features Input feature vector
 * @return cv::Mat Normalized feature vector
 * 
 * @author Krushna Sanjay Sharma
 */
cv::Mat FeatureExtractor::normalizeFeatures(const cv::Mat& features) const {
    // Check if input is empty
    if (features.empty()) {
        std::cerr << "Warning: Cannot normalize empty feature vector" << std::endl;
        return cv::Mat();
    }
    
    // Create output matrix with same size and type
    cv::Mat normalized;
    
    // Find min and max values
    double minVal, maxVal;
    cv::minMaxLoc(features, &minVal, &maxVal);
    
    // Check for zero range (all values are the same)
    if (maxVal - minVal < 1e-10) {
        // All values are essentially the same, return zeros or ones
        normalized = cv::Mat::zeros(features.size(), features.type());
        return normalized;
    }
    
    // Normalize to [0, 1] range: (x - min) / (max - min)
    normalized = (features - minVal) / (maxVal - minVal);
    
    return normalized;
}

} // namespace cbir
