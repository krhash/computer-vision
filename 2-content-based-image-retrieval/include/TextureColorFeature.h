////////////////////////////////////////////////////////////////////////////////
// TextureColorFeature.h
// Author: Krushna Sanjay Sharma
// Description: Combined texture and color feature extractor for CBIR Task 4.
//              Extracts both texture features (Sobel gradient magnitude) and
//              color features (RGB histogram) from an image and combines them
//              into a single feature vector.
//
// Feature Components:
//   1. Texture: Histogram of Sobel gradient magnitudes
//      - Captures edge/texture information
//      - Bins: configurable (default: 16)
//      - Represents image structure and patterns
//
//   2. Color: RGB color histogram
//      - Captures color distribution
//      - Bins: configurable (default: 8 per channel = 512 bins)
//      - Represents color composition
//
// Combined feature vector = [texture_histogram, color_histogram]
// Distance metric should weight both components equally (50/50).
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef TEXTURE_COLOR_FEATURE_H
#define TEXTURE_COLOR_FEATURE_H

#include "FeatureExtractor.h"

namespace cbir {

/**
 * @class TextureColorFeature
 * @brief Combines texture and color histograms into single feature vector
 * 
 * This feature extractor captures both:
 *   - Structural information (edges, patterns) via texture
 *   - Color distribution via color histogram
 * 
 * Feature vector structure:
 *   [texture_bin0, texture_bin1, ..., texture_binN,
 *    color_bin0, color_bin1, ..., color_binM]
 * 
 * Default configuration:
 *   - Texture: 16-bin histogram of gradient magnitudes (16 values)
 *   - Color: RGB 8-bin histogram (512 values)
 *   - Total: 528 values
 * 
 * The combined features should be compared using a distance metric
 * that weights texture and color components equally.
 * 
 * @author Krushna Sanjay Sharma
 */
class TextureColorFeature : public FeatureExtractor {
public:
    /**
     * Default constructor
     * 
     * Creates extractor with:
     *   - 16 bins for texture histogram
     *   - RGB histogram with 8 bins per channel
     *   - Normalized histograms
     */
    TextureColorFeature();
    
    /**
     * Constructor with custom configuration
     * 
     * @param textureBins Number of bins for gradient magnitude histogram
     * @param colorBinsPerChannel Number of bins per color channel
     * @param normalize Normalize histograms to sum=1.0
     */
    TextureColorFeature(int textureBins, 
                       int colorBinsPerChannel,
                       bool normalize);
    
    virtual ~TextureColorFeature() = default;
    
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    /**
     * Get dimension of texture component
     */
    int getTextureDimension() const { return textureBins_; }
    
    /**
     * Get dimension of color component
     */
    int getColorDimension() const { return colorBinsPerChannel_ * colorBinsPerChannel_ * colorBinsPerChannel_; }
    
    /**
     * Set number of texture bins
     */
    void setTextureBins(int bins);
    
    /**
     * Set number of color bins per channel
     */
    void setColorBinsPerChannel(int bins);

private:
    int textureBins_;         ///< Number of bins for texture histogram
    int colorBinsPerChannel_; ///< Number of bins per color channel
    bool normalize_;          ///< Normalize histograms flag
    
    /**
     * Compute Sobel gradient magnitude for entire image
     * 
     * Uses Sobel operators in X and Y directions:
     *   Gx = Sobel_X(image)
     *   Gy = Sobel_Y(image)
     *   magnitude = sqrt(Gx² + Gy²)
     * 
     * @param image Input image (grayscale or color)
     * @return Gradient magnitude image (CV_32F)
     */
    cv::Mat computeGradientMagnitude(const cv::Mat& image) const;
    
    /**
     * Compute histogram of gradient magnitudes (texture feature)
     * 
     * Bins gradient magnitudes into histogram representing texture.
     * Higher gradients = edges, lower gradients = smooth regions.
     * 
     * @param gradientMagnitude Gradient magnitude image
     * @return 1D texture histogram (normalized if configured)
     */
    cv::Mat computeTextureHistogram(const cv::Mat& gradientMagnitude) const;
    
    /**
     * Compute RGB color histogram (color feature)
     * 
     * Same as HistogramFeature, but integrated here.
     * 
     * @param image Color image
     * @return 1D color histogram (normalized if configured)
     */
    cv::Mat computeColorHistogram(const cv::Mat& image) const;
    
    /**
     * Normalize histogram to sum=1.0
     */
    cv::Mat normalizeHistogram(const cv::Mat& histogram) const;
};

} // namespace cbir

#endif // TEXTURE_COLOR_FEATURE_H
