////////////////////////////////////////////////////////////////////////////////
// MultiRegionHistogramIntersection.h
// Author: Krushna Sanjay Sharma
// Description: Custom distance metric for multi-region histogram features
//              (Task 3). Computes histogram intersection for each spatial
//              region separately and combines with weighted averaging.
//
// Algorithm:
//   For multi-region feature [region1_hist, region2_hist, ...]:
//   1. Extract each region's histogram from concatenated feature vector
//   2. Compute intersection for each region independently
//   3. Convert each intersection to distance: d = 1 - intersection
//   4. Combine region distances: final = Σ(weight[i] × distance[i])
//
// Default configuration:
//   - Equal weights for all regions (e.g., 0.5, 0.5 for 2 regions)
//   - Customizable for different region importance
//
// Example with 2 regions (top/bottom):
//   Feature: [top_histogram(512), bottom_histogram(512)]
//   d_top = 1 - intersection(top_query, top_db)
//   d_bottom = 1 - intersection(bottom_query, bottom_db)
//   final_distance = 0.5 × d_top + 0.5 × d_bottom
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef MULTI_REGION_HISTOGRAM_INTERSECTION_H
#define MULTI_REGION_HISTOGRAM_INTERSECTION_H

#include "DistanceMetric.h"
#include <vector>

namespace cbir {

/**
 * @class MultiRegionHistogramIntersection
 * @brief Distance metric for multi-region histogram features
 * 
 * This is the custom distance metric designed for Task 3. It treats
 * each spatial region's histogram independently, computes intersection
 * for each region, and combines the results using weighted averaging.
 * 
 * This approach allows:
 *   - Explicit per-region comparison (top matches top, bottom matches bottom)
 *   - Customizable region weights (e.g., sky more important than ground)
 *   - Better semantic understanding of spatial matching
 * 
 * Compared to regular HistogramIntersection:
 *   - Regular: Treats concatenated vector as single histogram
 *   - MultiRegion: Explicitly compares corresponding regions
 * 
 * With equal weights, both approaches are mathematically equivalent,
 * but MultiRegion is conceptually clearer and more flexible.
 * 
 * @author Krushna Sanjay Sharma
 */
class MultiRegionHistogramIntersection : public DistanceMetric {
public:
    /**
     * Constructor with default equal weights
     * 
     * @param numRegions Number of spatial regions in feature
     * @param binsPerRegion Histogram dimension per region (e.g., 512 for RGB)
     */
    MultiRegionHistogramIntersection(int numRegions = 2,
                                    int binsPerRegion = 512);
    
    /**
     * Constructor with custom weights
     * 
     * @param numRegions Number of spatial regions
     * @param binsPerRegion Histogram dimension per region
     * @param weights Custom weights for each region (must sum to 1.0)
     */
    MultiRegionHistogramIntersection(int numRegions,
                                    int binsPerRegion,
                                    const std::vector<double>& weights);
    
    virtual ~MultiRegionHistogramIntersection() = default;
    
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;
    
    virtual std::string getMetricName() const override;
    
    /**
     * Set custom weights for regions
     * 
     * @param weights Vector of weights (must sum to 1.0)
     */
    void setWeights(const std::vector<double>& weights);
    
    /**
     * Get current region weights
     */
    std::vector<double> getWeights() const { return weights_; }

private:
    int numRegions_;              ///< Number of spatial regions
    int binsPerRegion_;           ///< Histogram dimension per region
    std::vector<double> weights_; ///< Weight for each region (sum = 1.0)
    
    /**
     * Compute histogram intersection for a single region
     * 
     * @param hist1 First feature vector
     * @param hist2 Second feature vector
     * @param startIdx Start index of region (inclusive)
     * @param endIdx End index of region (exclusive)
     * @return Intersection value for this region
     */
    double computeRegionIntersection(const cv::Mat& hist1, 
                                    const cv::Mat& hist2,
                                    int startIdx, 
                                    int endIdx) const;
    
    /**
     * Initialize equal weights for all regions
     * Called when no custom weights provided
     */
    void initializeEqualWeights();
    
    /**
     * Validate and normalize weights to sum to 1.0
     */
    void normalizeWeights();
};

} // namespace cbir

#endif // MULTI_REGION_HISTOGRAM_INTERSECTION_H
