////////////////////////////////////////////////////////////////////////////////
// HistogramIntersection.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of histogram intersection distance metric.
//              Manually computes intersection without OpenCV compareHist
//              as required by project specifications.
//
// Histogram Intersection Formula:
//   intersection = Σ min(H1[i], H2[i])
//   distance = 1 - intersection
//
// For normalized histograms (sum=1.0):
//   - intersection ∈ [0, 1] where 1 = identical, 0 = no overlap
//   - distance ∈ [0, 1] where 0 = identical, 1 = completely different
//
// This metric works well for comparing color distributions as it
// measures the overlap between two histograms.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "HistogramIntersection.h"
#include <iostream>
#include <cmath>

namespace cbir {

/**
 * Compute histogram intersection distance
 * 
 * Measures similarity between two histograms by computing their overlap.
 * Lower distance = more similar color distributions.
 * 
 * Algorithm:
 *   1. Validate histograms are compatible (same size/type)
 *   2. Normalize both histograms if needed (ensure sum=1.0)
 *   3. Compute intersection: sum of minimum values at each bin
 *   4. Convert to distance: distance = 1 - intersection
 * 
 * Example:
 *   H1 = [0.3, 0.2, 0.5]
 *   H2 = [0.4, 0.1, 0.5]
 *   intersection = min(0.3,0.4) + min(0.2,0.1) + min(0.5,0.5)
 *                = 0.3 + 0.1 + 0.5 = 0.9
 *   distance = 1.0 - 0.9 = 0.1 (very similar!)
 * 
 * @param features1 First histogram (normalized)
 * @param features2 Second histogram (normalized)
 * @return Distance in [0,1] where 0=identical, 1=completely different
 */
double HistogramIntersection::compute(const cv::Mat& features1, 
                                      const cv::Mat& features2) {
    // Validate that histograms are compatible for comparison
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Histograms are not compatible" << std::endl;
        return -1.0;
    }
    
    // Normalize histograms if needed (safety check)
    // This ensures both histograms sum to 1.0 for proper intersection
    cv::Mat hist1 = normalizeIfNeeded(features1);
    cv::Mat hist2 = normalizeIfNeeded(features2);
    
    // Compute histogram intersection manually
    // intersection = Σ min(H1[i], H2[i])
    // This measures the overlap between the two histograms
    double intersection = 0.0;
    
    int totalBins = static_cast<int>(hist1.total());
    for (int i = 0; i < totalBins; i++) {
        float val1 = hist1.at<float>(i);
        float val2 = hist2.at<float>(i);
        
        // Add minimum of the two values (the overlapping part)
        intersection += std::min(val1, val2);
    }
    
    // Convert intersection to distance
    // distance = 1 - intersection
    // 
    // For normalized histograms:
    //   - intersection = 1.0 → perfect match → distance = 0.0
    //   - intersection = 0.0 → no overlap → distance = 1.0
    double distance = 1.0 - intersection;
    
    return distance;
}

/**
 * Get metric name
 * 
 * @return "HistogramIntersection"
 */
std::string HistogramIntersection::getMetricName() const {
    return "HistogramIntersection";
}

/**
 * Check if histogram is normalized (sum ≈ 1.0)
 * 
 * Verifies that histogram bins sum to approximately 1.0,
 * allowing small floating point tolerance.
 * 
 * @param histogram Histogram to check
 * @return True if sum is close to 1.0
 */
bool HistogramIntersection::isNormalized(const cv::Mat& histogram) const {
    // Compute sum of all histogram bins
    double sum = 0.0;
    int totalBins = static_cast<int>(histogram.total());
    
    for (int i = 0; i < totalBins; i++) {
        sum += histogram.at<float>(i);
    }
    
    // Allow small tolerance for floating point errors
    // Sum should be approximately 1.0 for normalized histograms
    return std::abs(sum - 1.0) < 0.01;
}

/**
 * Normalize histogram if needed
 * 
 * Ensures histogram sums to 1.0 for proper intersection computation.
 * If already normalized, returns unchanged.
 * 
 * @param histogram Input histogram
 * @return Normalized histogram (sum=1.0)
 */
cv::Mat HistogramIntersection::normalizeIfNeeded(const cv::Mat& histogram) const {
    // Check if already normalized
    if (isNormalized(histogram)) {
        return histogram;  // Already normalized, no need to modify
    }
    
    // Compute sum of all bins
    double sum = 0.0;
    int totalBins = static_cast<int>(histogram.total());
    
    for (int i = 0; i < totalBins; i++) {
        sum += histogram.at<float>(i);
    }
    
    // Avoid division by zero (empty histogram)
    if (sum < 1e-10) {
        std::cerr << "Warning: Histogram sum is zero" << std::endl;
        return histogram.clone();
    }
    
    // Normalize: divide all bins by sum
    cv::Mat normalized = histogram.clone();
    normalized /= sum;
    
    return normalized;
}

} // namespace cbir
