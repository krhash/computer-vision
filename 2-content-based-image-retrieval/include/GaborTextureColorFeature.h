////////////////////////////////////////////////////////////////////////////////
// GaborTextureColorFeature.h
// Author: Krushna Sanjay Sharma
// Description: Advanced texture+color feature using Gabor filters for Task 4
//              extension. Gabor filters capture texture at multiple orientations
//              and frequencies, providing richer texture representation than
//              simple Sobel gradients.
//
// Gabor Filters:
//   - Oriented filters that detect edges/patterns at specific angles
//   - Multiple scales (frequencies) capture different texture details
//   - Commonly used for fabric, wood, water, natural texture analysis
//
// Configuration:
//   - 4 orientations (0°, 45°, 90°, 135°)
//   - 2 scales (fine and coarse texture)
//   - Total: 8 Gabor filter responses → 8-bin histogram each → 64 texture bins
//   - Plus RGB color histogram: 512 bins
//   - Total feature dimension: 64 + 512 = 576 values
//
// Compared to Sobel (Task 4):
//   - Sobel: Single gradient magnitude (16 bins)
//   - Gabor: Multiple orientations and scales (64 bins)
//   - Gabor provides richer texture discrimination
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef GABOR_TEXTURE_COLOR_FEATURE_H
#define GABOR_TEXTURE_COLOR_FEATURE_H

#include "FeatureExtractor.h"
#include <vector>

namespace cbir {

/**
 * @class GaborTextureColorFeature
 * @brief Advanced texture feature using Gabor filters + color histogram
 * 
 * Gabor filters are oriented sinusoidal gratings modulated by Gaussian:
 *   G(x,y) = exp(-(x'²+γ²y'²)/(2σ²)) × cos(2π(x'/λ) + ψ)
 * 
 * Where:
 *   - x', y' are rotated coordinates based on orientation θ
 *   - σ controls the Gaussian envelope size
 *   - λ is the wavelength (controls frequency/scale)
 *   - γ is the spatial aspect ratio
 *   - ψ is the phase offset
 * 
 * This implementation uses:
 *   - 4 orientations: 0°, 45°, 90°, 135°
 *   - 2 scales (wavelengths): fine and coarse
 *   - 8 total Gabor filters
 *   - Histogram of responses from each filter
 * 
 * Feature structure:
 *   [gabor_0°_fine(8), gabor_45°_fine(8), ..., gabor_135°_coarse(8), color(512)]
 *   = 64 texture bins + 512 color bins = 576 total
 * 
 * @author Krushna Sanjay Sharma
 */
class GaborTextureColorFeature : public FeatureExtractor {
public:
    /**
     * Default constructor
     * 
     * Uses:
     *   - 4 orientations
     *   - 2 scales
     *   - 8 bins per Gabor response histogram
     *   - RGB color histogram with 8 bins per channel
     */
    GaborTextureColorFeature();
    
    /**
     * Constructor with custom configuration
     * 
     * @param numOrientations Number of Gabor orientations (default: 4)
     * @param numScales Number of scales/frequencies (default: 2)
     * @param binsPerGabor Bins per Gabor response histogram (default: 8)
     * @param colorBinsPerChannel Bins per color channel (default: 8)
     * @param normalize Normalize histograms
     */
    GaborTextureColorFeature(int numOrientations,
                            int numScales,
                            int binsPerGabor,
                            int colorBinsPerChannel,
                            bool normalize);
    
    virtual ~GaborTextureColorFeature() = default;
    
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    /**
     * Get texture component dimension
     */
    int getTextureDimension() const { 
        return numOrientations_ * numScales_ * binsPerGabor_; 
    }
    
    /**
     * Get color component dimension
     */
    int getColorDimension() const { 
        return colorBinsPerChannel_ * colorBinsPerChannel_ * colorBinsPerChannel_; 
    }

private:
    int numOrientations_;      ///< Number of Gabor orientations (default: 4)
    int numScales_;            ///< Number of scales/frequencies (default: 2)
    int binsPerGabor_;         ///< Bins per Gabor response histogram (default: 8)
    int colorBinsPerChannel_;  ///< Color bins per channel (default: 8)
    bool normalize_;           ///< Normalize flag
    
    /**
     * Create Gabor filter kernel
     * 
     * @param ksize Kernel size (should be odd)
     * @param sigma Standard deviation of Gaussian envelope
     * @param theta Orientation in radians
     * @param lambda Wavelength (controls frequency/scale)
     * @param gamma Spatial aspect ratio
     * @param psi Phase offset
     * @return Gabor kernel
     */
    cv::Mat createGaborKernel(int ksize, double sigma, double theta, 
                             double lambda, double gamma, double psi) const;
    
    /**
     * Apply Gabor filter bank to image
     * 
     * @param image Grayscale image
     * @return Vector of filtered images (one per Gabor filter)
     */
    std::vector<cv::Mat> applyGaborFilters(const cv::Mat& image) const;
    
    /**
     * Compute histogram of Gabor filter responses
     * 
     * @param gaborResponses Vector of Gabor-filtered images
     * @return Concatenated histograms from all Gabor responses
     */
    cv::Mat computeGaborHistogram(const std::vector<cv::Mat>& gaborResponses) const;
    
    /**
     * Compute RGB color histogram (same as TextureColorFeature)
     */
    cv::Mat computeColorHistogram(const cv::Mat& image) const;
    
    /**
     * Normalize histogram to sum=1.0
     */
    cv::Mat normalizeHistogram(const cv::Mat& histogram) const;
};

} // namespace cbir

#endif // GABOR_TEXTURE_COLOR_FEATURE_H
