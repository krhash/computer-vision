////////////////////////////////////////////////////////////////////////////////
// HistogramIntersection.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of histogram intersection distance metric.
//              Manually computes intersection without OpenCV compareHist.
////////////////////////////////////////////////////////////////////////////////

#include "HistogramIntersection.h"
#include <iostream>
#include <cmath>

namespace cbir {

double HistogramIntersection::compute(const cv::Mat& features1, 
                                      const cv::Mat& features2) {
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Histograms are not compatible" << std::endl;
        return -1.0;
    }
    
    // Normalize histograms if needed
    cv::Mat hist1 = normalizeIfNeeded(features1);
    cv::Mat hist2 = normalizeIfNeeded(features2);
    
    // Compute histogram intersection manually
    // intersection = Σ min(H1[i], H2[i])
    double intersection = 0.0;
    
    int totalBins = hist1.total();
    for (int i = 0; i < totalBins; i++) {
        float val1 = hist1.at<float>(i);
        float val2 = hist2.at<float>(i);
        
        // Add minimum of the two values
        intersection += std::min(val1, val2);
    }
    
    // Convert intersection to distance
    // distance = 1 - intersection
    // For normalized histograms: intersection ∈ [0, 1]
    // So distance ∈ [0, 1] where 0 = identical, 1 = completely different
    double distance = 1.0 - intersection;
    
    return distance;
}

std::string HistogramIntersection::getMetricName() const {
    return "HistogramIntersection";
}

bool HistogramIntersection::isNormalized(const cv::Mat& histogram) const {
    // Check if sum is approximately 1.0
    double sum = 0.0;
    int totalBins = histogram.total();
    
    for (int i = 0; i < totalBins; i++) {
        sum += histogram.at<float>(i);
    }
    
    // Allow small tolerance for floating point errors
    return std::abs(sum - 1.0) < 0.01;
}

cv::Mat HistogramIntersection::normalizeIfNeeded(const cv::Mat& histogram) const {
    if (isNormalized(histogram)) {
        return histogram;
    }
    
    // Compute sum
    double sum = 0.0;
    int totalBins = histogram.total();
    
    for (int i = 0; i < totalBins; i++) {
        sum += histogram.at<float>(i);
    }
    
    // Avoid division by zero
    if (sum < 1e-10) {
        std::cerr << "Warning: Histogram sum is zero" << std::endl;
        return histogram.clone();
    }
    
    // Normalize
    cv::Mat normalized = histogram.clone();
    normalized /= sum;
    
    return normalized;
}

} // namespace cbir
