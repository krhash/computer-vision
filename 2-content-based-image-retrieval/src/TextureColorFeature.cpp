////////////////////////////////////////////////////////////////////////////////
// TextureColorFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of combined texture and color feature extraction.
//              Computes Sobel gradient magnitude histogram for texture and
//              RGB color histogram, then concatenates them.
//
// Texture Component:
//   - Apply Sobel filters in X and Y directions
//   - Compute magnitude = sqrt(Gx² + Gy²)
//   - Create histogram of magnitudes (16 bins, 0-255 range)
//   - Captures edges, patterns, and structure
//
// Color Component:
//   - RGB histogram (8 bins per channel = 512 bins)
//   - Captures color distribution
//
// Combined: [texture_hist(16), color_hist(512)] = 528 total values
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "TextureColorFeature.h"
#include <iostream>
#include <cmath>

namespace cbir {

/**
 * Default constructor
 * 
 * Configuration:
 *   - 16 bins for texture (gradient magnitude range 0-255)
 *   - 8 bins per channel for color (8×8×8 = 512 bins)
 *   - Normalized histograms
 */
TextureColorFeature::TextureColorFeature()
    : textureBins_(16),
      colorBinsPerChannel_(8),
      normalize_(true) {
}

/**
 * Constructor with custom configuration
 */
TextureColorFeature::TextureColorFeature(int textureBins, 
                                       int colorBinsPerChannel,
                                       bool normalize)
    : textureBins_(textureBins),
      colorBinsPerChannel_(colorBinsPerChannel),
      normalize_(normalize) {
    
    if (textureBins_ <= 0) {
        std::cerr << "Warning: Texture bins must be positive. Using 16." << std::endl;
        textureBins_ = 16;
    }
    
    if (colorBinsPerChannel_ <= 0) {
        std::cerr << "Warning: Color bins must be positive. Using 8." << std::endl;
        colorBinsPerChannel_ = 8;
    }
}

/**
 * Extract combined texture and color features
 * 
 * Process:
 *   1. Compute texture features:
 *      - Calculate Sobel gradient magnitude
 *      - Create histogram of magnitudes
 *   2. Compute color features:
 *      - Create RGB color histogram
 *   3. Concatenate both histograms into single feature vector
 * 
 * @param image Input image
 * @return Combined feature vector [texture_hist, color_hist]
 */
cv::Mat TextureColorFeature::extractFeatures(const cv::Mat& image) {
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for texture-color extraction" << std::endl;
        return cv::Mat();
    }
    
    // Convert to color if grayscale
    cv::Mat colorImage = image;
    if (image.channels() == 1) {
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
    }
    
    // Compute texture features
    cv::Mat gradientMag = computeGradientMagnitude(colorImage);
    cv::Mat textureHist = computeTextureHistogram(gradientMag);
    
    if (textureHist.empty()) {
        std::cerr << "Error: Failed to compute texture histogram" << std::endl;
        return cv::Mat();
    }
    
    // Compute color features
    cv::Mat colorHist = computeColorHistogram(colorImage);
    
    if (colorHist.empty()) {
        std::cerr << "Error: Failed to compute color histogram" << std::endl;
        return cv::Mat();
    }
    
    // Concatenate texture and color features
    int totalDim = textureHist.cols + colorHist.cols;
    cv::Mat combinedFeatures(1, totalDim, CV_32F);
    
    // Copy texture histogram (first part)
    for (int i = 0; i < textureHist.cols; i++) {
        combinedFeatures.at<float>(0, i) = textureHist.at<float>(0, i);
    }
    
    // Copy color histogram (second part)
    for (int i = 0; i < colorHist.cols; i++) {
        combinedFeatures.at<float>(0, textureHist.cols + i) = colorHist.at<float>(0, i);
    }
    
    return combinedFeatures;
}

/**
 * Get feature name
 */
std::string TextureColorFeature::getFeatureName() const {
    return "TextureColor_" + std::to_string(textureBins_) + "tex_" 
           + std::to_string(colorBinsPerChannel_) + "color";
}

/**
 * Get total feature dimension
 */
int TextureColorFeature::getFeatureDimension() const {
    return textureBins_ + (colorBinsPerChannel_ * colorBinsPerChannel_ * colorBinsPerChannel_);
}

/**
 * Set number of texture bins
 */
void TextureColorFeature::setTextureBins(int bins) {
    if (bins > 0) {
        textureBins_ = bins;
    }
}

/**
 * Set number of color bins per channel
 */
void TextureColorFeature::setColorBinsPerChannel(int bins) {
    if (bins > 0) {
        colorBinsPerChannel_ = bins;
    }
}

/**
 * Compute Sobel gradient magnitude
 * 
 * Algorithm:
 *   1. Convert to grayscale (if color)
 *   2. Apply Sobel filter in X direction → Gx
 *   3. Apply Sobel filter in Y direction → Gy
 *   4. Compute magnitude = sqrt(Gx² + Gy²)
 * 
 * Sobel kernels:
 *   Sobel_X:        Sobel_Y:
 *   [-1  0  1]      [-1 -2 -1]
 *   [-2  0  2]      [ 0  0  0]
 *   [-1  0  1]      [ 1  2  1]
 * 
 * @param image Input image
 * @return Gradient magnitude image (CV_32F, range ~0-255)
 */
cv::Mat TextureColorFeature::computeGradientMagnitude(const cv::Mat& image) const {
    // Convert to grayscale for gradient computation
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Convert to float for Sobel
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);
    
    // Apply Sobel filters
    cv::Mat gradX, gradY;
    cv::Sobel(grayFloat, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(grayFloat, gradY, CV_32F, 0, 1, 3);
    
    // Compute magnitude
    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);
    
    return magnitude;
}

/**
 * Compute texture histogram from gradient magnitudes
 * 
 * Creates histogram of gradient magnitudes where:
 *   - Low magnitudes (0-50): Smooth regions, uniform areas
 *   - Medium magnitudes (50-150): Texture, patterns
 *   - High magnitudes (150-255): Strong edges, boundaries
 * 
 * Algorithm:
 *   1. Create empty histogram with specified number of bins
 *   2. For each pixel, get gradient magnitude value
 *   3. Compute bin index based on magnitude
 *   4. Increment corresponding bin
 *   5. Normalize if configured
 * 
 * @param gradientMagnitude Gradient magnitude image (CV_32F)
 * @return 1D texture histogram
 */
cv::Mat TextureColorFeature::computeTextureHistogram(const cv::Mat& gradientMagnitude) const {
    // Create 1D histogram for texture
    cv::Mat histogram = cv::Mat::zeros(1, textureBins_, CV_32F);
    
    // Use fixed range [0, 255] for consistency
    float binSize = 255.0f / textureBins_;
    
    // Count gradient magnitudes into bins
    for (int row = 0; row < gradientMagnitude.rows; row++) {
        for (int col = 0; col < gradientMagnitude.cols; col++) {
            float magnitude = gradientMagnitude.at<float>(row, col);
            
            // Clamp to [0, 255]
            magnitude = std::min(std::max(magnitude, 0.0f), 255.0f);
            
            // Compute bin
            int bin = static_cast<int>(magnitude / binSize);
            bin = std::min(bin, textureBins_ - 1);
            
            // Increment
            histogram.at<float>(0, bin) += 1.0f;
        }
    }
    
    // Normalize if configured
    if (normalize_) {
        double sum = 0.0;
        for (int i = 0; i < histogram.cols; i++) {
            sum += histogram.at<float>(0, i);
        }
        
        if (sum > 1e-10) {
            histogram /= sum;
        }
    }
    
    return histogram;
}

/**
 * Compute RGB color histogram
 * 
 * Same algorithm as HistogramFeature::computeRGBHistogram()
 * 
 * @param image Color image (BGR format)
 * @return 1D color histogram (flattened from 3D)
 */
cv::Mat TextureColorFeature::computeColorHistogram(const cv::Mat& image) const {
    // Create 3D histogram for RGB
    int dims[3] = {colorBinsPerChannel_, colorBinsPerChannel_, colorBinsPerChannel_};
    cv::Mat histogram = cv::Mat::zeros(3, dims, CV_32F);
    
    // Compute bin size
    float binSize = 256.0f / colorBinsPerChannel_;
    
    // Count pixels into bins
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            
            int binB = static_cast<int>(b / binSize);
            int binG = static_cast<int>(g / binSize);
            int binR = static_cast<int>(r / binSize);
            
            binB = std::min(binB, colorBinsPerChannel_ - 1);
            binG = std::min(binG, colorBinsPerChannel_ - 1);
            binR = std::min(binR, colorBinsPerChannel_ - 1);
            
            histogram.at<float>(binR, binG, binB) += 1.0f;
        }
    }
    
    // Flatten 3D histogram to 1D FIRST (before normalization)
    int totalBins = colorBinsPerChannel_ * colorBinsPerChannel_ * colorBinsPerChannel_;
    cv::Mat flattened(1, totalBins, CV_32F);
    
    int idx = 0;
    const int* sizes = histogram.size.p;
    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                flattened.at<float>(0, idx++) = histogram.at<float>(i, j, k);
            }
        }
    }
    
    // THEN normalize the flattened 1D histogram
    if (normalize_) {
        double sum = 0.0;
        for (int i = 0; i < flattened.cols; i++) {
            sum += flattened.at<float>(0, i);
        }
        
        if (sum > 1e-10) {
            flattened /= sum;
        } else {
            std::cerr << "Warning: Color histogram sum is zero for an image" << std::endl;
        }
    }
    
    return flattened;
}

/**
 * Normalize histogram to sum=1.0
 * 
 * Works for both 1D (texture) and 3D (color) histograms
 * 
 * @param histogram Input histogram
 * @return Normalized histogram
 */
cv::Mat TextureColorFeature::normalizeHistogram(const cv::Mat& histogram) const {
    double sum = 0.0;
    
    // Handle 1D histogram (texture) or row vector
    if (histogram.dims == 1 || (histogram.dims == 2 && histogram.rows == 1)) {
        for (int i = 0; i < histogram.cols; i++) {
            sum += histogram.at<float>(0, i);
        }
    } else {
        std::cerr << "Error: normalizeHistogram only handles 1D histograms" << std::endl;
        return histogram.clone();
    }
    
    // Avoid division by zero
    if (sum < 1e-10) {
        std::cerr << "Warning: Histogram sum is zero or near-zero" << std::endl;
        return histogram.clone();
    }
    
    // Normalize
    cv::Mat normalized = histogram.clone();
    normalized /= sum;
    
    return normalized;
}

} // namespace cbir
