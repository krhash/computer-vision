////////////////////////////////////////////////////////////////////////////////
// MultiHistogramFeature.h
// Author: Krushna Sanjay Sharma
// Description: Multi-region histogram feature extractor for CBIR Task 3.
//              Extracts histograms from multiple spatial regions of the image
//              and concatenates them into a single feature vector.
//
// Strategy:
//   - Divide image into multiple regions (e.g., top half, bottom half)
//   - Compute histogram for each region independently
//   - Concatenate all histograms into single feature vector
//   - Compare regions separately using weighted distance combination
//
// This captures spatial information that single-histogram methods miss.
// For example: "blue sky on top, green grass on bottom" vs
//              "green grass on top, blue sky on bottom" will have
//              different multi-histograms but same single-histogram.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef MULTI_HISTOGRAM_FEATURE_H
#define MULTI_HISTOGRAM_FEATURE_H

#include "FeatureExtractor.h"
#include "HistogramFeature.h"
#include <vector>

namespace cbir {

/**
 * @class MultiHistogramFeature
 * @brief Extracts histograms from multiple spatial regions
 * 
 * Divides image into regions and computes a histogram for each region.
 * Regions can be:
 *   - Horizontal splits (top/bottom)
 *   - Vertical splits (left/right)
 *   - Grid (2×2, 3×3, etc.)
 *   - Custom regions
 * 
 * Default: Top and bottom halves (2 regions)
 * 
 * Feature vector = [histogram_region1, histogram_region2, ...]
 * 
 * For RGB histogram with 8 bins and 2 regions:
 *   Total dimension = 2 × 512 = 1024 values
 * 
 * @author Krushna Sanjay Sharma
 */
class MultiHistogramFeature : public FeatureExtractor {
public:
    /**
     * Region split type
     */
    enum class SplitType {
        HORIZONTAL,  ///< Split horizontally (top/bottom)
        VERTICAL,    ///< Split vertically (left/right)
        GRID         ///< Split into grid (2×2, 3×3, etc.)
    };
    
    /**
     * Constructor with default settings
     * 
     * Default: 2 horizontal regions (top/bottom), RGB histogram, 8 bins
     */
    MultiHistogramFeature();
    
    /**
     * Constructor with custom configuration
     * 
     * @param splitType How to divide the image
     * @param numRegions Number of regions to create
     * @param histType RGB or RG_CHROMATICITY
     * @param binsPerChannel Bins per color channel
     * @param normalize Normalize histograms
     */
    MultiHistogramFeature(SplitType splitType,
                         int numRegions,
                         HistogramFeature::HistogramType histType,
                         int binsPerChannel,
                         bool normalize);
    
    virtual ~MultiHistogramFeature() = default;
    
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    /**
     * Set split type
     */
    void setSplitType(SplitType type);
    
    /**
     * Set number of regions
     */
    void setNumRegions(int num);
    
    /**
     * Get weights for combining distances from each region
     * Can be used by distance metric for weighted averaging
     * 
     * @return Vector of weights (sum = 1.0)
     */
    std::vector<double> getRegionWeights() const;
    
    /**
     * Set custom weights for regions (must sum to 1.0)
     * Default: equal weights for all regions
     * 
     * @param weights Vector of weights
     */
    void setRegionWeights(const std::vector<double>& weights);

private:
    SplitType splitType_;                        ///< How to split image
    int numRegions_;                             ///< Number of regions
    HistogramFeature::HistogramType histType_;   ///< Histogram type
    int binsPerChannel_;                         ///< Bins per channel
    bool normalize_;                             ///< Normalize flag
    std::vector<double> regionWeights_;          ///< Weights for each region
    
    /**
     * Split image into regions based on split type
     * 
     * @param image Input image
     * @return Vector of image regions
     */
    std::vector<cv::Mat> splitImage(const cv::Mat& image) const;
    
    /**
     * Split horizontally (e.g., top half, bottom half)
     */
    std::vector<cv::Mat> splitHorizontal(const cv::Mat& image) const;
    
    /**
     * Split vertically (e.g., left half, right half)
     */
    std::vector<cv::Mat> splitVertical(const cv::Mat& image) const;
    
    /**
     * Split into grid (e.g., 2×2 quadrants)
     */
    std::vector<cv::Mat> splitGrid(const cv::Mat& image) const;
    
    /**
     * Initialize default region weights (equal weights)
     */
    void initializeWeights();
};

} // namespace cbir

#endif // MULTI_HISTOGRAM_FEATURE_H
