////////////////////////////////////////////////////////////////////////////////
// ProductMatcherFeature.h
// Author: Krushna Sanjay Sharma
// Description: Custom feature for Task 7 - Product/Object Image Matching.
//              Combines DNN embeddings for semantic understanding with
//              center-region color histogram for subject appearance matching.
//
// Design Rationale:
//   Problem: DNN finds similar objects but ignores appearance (color).
//            Histogram finds similar colors but includes backgrounds.
//   
//   Solution: DNN (60%) for object type + Center-color (40%) for subject color.
//             Center extraction (50% of image) focuses on subject, ignores background.
//
// Target Use Case: Product photography, object databases, e-commerce
//
// Feature Structure:
//   [dnn_embeddings(512), center_rgb_histogram(512)] = 1024 values
//
// Example:
//   Query: Red toy on blue background
//   Matches: Other red toys (any background)
//   Ignores: Blue backgrounds, red walls
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef PRODUCT_MATCHER_FEATURE_H
#define PRODUCT_MATCHER_FEATURE_H

#include "FeatureExtractor.h"
#include "DNNFeature.h"

namespace cbir {

/**
 * @class ProductMatcherFeature
 * @brief Custom feature combining DNN semantics with center-region color
 * 
 * Designed for product/object image matching where:
 *   - Subject is typically centered
 *   - Background should be ignored
 *   - Both object type AND appearance matter
 * 
 * Feature components:
 *   1. DNN embeddings (512D) - Object category, semantic content
 *   2. Center-region RGB histogram (512D) - Subject color only
 * 
 * Key innovation: Center-region extraction filters out background noise
 * 
 * @author Krushna Sanjay Sharma
 */
class ProductMatcherFeature : public FeatureExtractor {
public:
    /**
     * Constructor
     * 
     * @param dnnCsvPath Path to pre-computed DNN features CSV
     * @param centerRatio Ratio of center region to extract (0.5 = 50%)
     * @param colorBins Bins per color channel for histogram (default: 8)
     */
    ProductMatcherFeature(const std::string& dnnCsvPath,
                         double centerRatio = 0.5,
                         int colorBins = 8);
    
    virtual ~ProductMatcherFeature() = default;
    
    /**
     * Extract features - requires filename for DNN lookup
     * This override returns empty (use extractFeaturesWithFilename)
     */
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    
    /**
     * Extract combined DNN + center-color features
     * 
     * Process:
     *   1. Load DNN features by filename
     *   2. Extract center region from image
     *   3. Compute RGB histogram on center region only
     *   4. Concatenate [dnn, center_color]
     * 
     * @param image Input image
     * @param filename Image filename for DNN lookup
     * @return Combined feature vector [dnn(512), color(512)]
     */
    cv::Mat extractFeaturesWithFilename(const cv::Mat& image, 
                                       const std::string& filename);
    
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    /**
     * Get dimension of each component
     */
    int getDNNDimension() const { return 512; }
    int getCenterColorDimension() const { return colorBins_ * colorBins_ * colorBins_; }
    
    /**
     * Set center region ratio
     */
    void setCenterRatio(double ratio);

private:
    DNNFeature dnnExtractor_;   ///< DNN feature loader
    double centerRatio_;        ///< Center region size (0.5 = 50%)
    int colorBins_;            ///< Bins per color channel
    
    /**
     * Extract center region from image
     * 
     * For centerRatio=0.5, extracts center 50% of image
     * (25% margin on each side)
     * 
     * @param image Input image
     * @return Center region as cv::Mat
     */
    cv::Mat extractCenterRegion(const cv::Mat& image) const;
    
    /**
     * Compute RGB color histogram from image region
     * 
     * @param region Image region
     * @return Flattened RGB histogram (512 values for 8 bins)
     */
    cv::Mat computeRGBHistogram(const cv::Mat& region) const;
};

} // namespace cbir

#endif // PRODUCT_MATCHER_FEATURE_H
