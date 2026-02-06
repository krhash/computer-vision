////////////////////////////////////////////////////////////////////////////////
// SSDMetric.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of Sum of Squared Differences distance metric.
//              Manually computed without using OpenCV built-in functions as
//              required by project specifications.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "SSDMetric.h"
#include <iostream>
#include <cmath>

namespace cbir {

/**
 * @brief Compute sum of squared differences between two feature vectors
 * 
 * This function manually computes SSD without using OpenCV's built-in
 * distance functions, as required by the project specifications.
 * 
 * Formula: SSD = Σ(f1[i] - f2[i])²
 * 
 * Returns 0.0 when comparing identical features (same image with itself).
 * 
 * @author Krushna Sanjay Sharma
 */
double SSDMetric::compute(const cv::Mat& features1, const cv::Mat& features2) {
    // Check if features are compatible
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Features are not compatible for SSD computation" 
                  << std::endl;
        std::cerr << "  Features1 size: " << features1.size() << std::endl;
        std::cerr << "  Features2 size: " << features2.size() << std::endl;
        return -1.0;
    }
    
    // Initialize sum
    double ssd = 0.0;
    
    // Get total number of elements
    int totalElements = features1.total();
    
    // Manual computation - do NOT use OpenCV functions
    // Iterate through all elements and compute squared differences
    for (int i = 0; i < totalElements; i++) {
        // Get feature values (assuming CV_32F type)
        float val1 = features1.at<float>(i);
        float val2 = features2.at<float>(i);
        
        // Compute difference
        float diff = val1 - val2;
        
        // Add squared difference to sum
        ssd += diff * diff;
    }
    
    return ssd;
}

/**
 * @brief Get metric name
 * 
 * @author Krushna Sanjay Sharma
 */
std::string SSDMetric::getMetricName() const {
    return "SSD";
}

} // namespace cbir
