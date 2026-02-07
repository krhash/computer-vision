////////////////////////////////////////////////////////////////////////////////
// GaborTextureColorFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of Gabor filter-based texture features combined
//              with color histograms. Extension for Task 4.
//
// Gabor Filter Theory:
//   Gabor filters are band-pass filters that respond to specific orientations
//   and frequencies in an image. They model the receptive fields of simple
//   cells in the mammalian visual cortex.
//
// Implementation:
//   - Create bank of Gabor filters with different orientations and scales
//   - Apply each filter to grayscale image
//   - Compute histogram of filter responses
//   - Concatenate all Gabor histograms + color histogram
//
// Advantages over Sobel:
//   - Captures multiple orientations (not just X/Y)
//   - Multiple scales capture fine and coarse textures
//   - Better for natural textures (wood, fabric, water, grass)
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "GaborTextureColorFeature.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cbir {

/**
 * Default constructor
 * 
 * Configuration:
 *   - 4 orientations (0°, 45°, 90°, 135°)
 *   - 2 scales (wavelengths: 5 and 10)
 *   - 8 bins per Gabor histogram
 *   - 8 bins per color channel
 *   - Total: 4×2×8 + 512 = 64 + 512 = 576 values
 */
GaborTextureColorFeature::GaborTextureColorFeature()
    : numOrientations_(4),
      numScales_(2),
      binsPerGabor_(8),
      colorBinsPerChannel_(8),
      normalize_(true) {
}

/**
 * Constructor with custom configuration
 */
GaborTextureColorFeature::GaborTextureColorFeature(int numOrientations,
                                                 int numScales,
                                                 int binsPerGabor,
                                                 int colorBinsPerChannel,
                                                 bool normalize)
    : numOrientations_(numOrientations),
      numScales_(numScales),
      binsPerGabor_(binsPerGabor),
      colorBinsPerChannel_(colorBinsPerChannel),
      normalize_(normalize) {
}

/**
 * Extract Gabor texture + color features
 * 
 * Process:
 *   1. Convert to grayscale for texture analysis
 *   2. Apply Gabor filter bank (multiple orientations and scales)
 *   3. Compute histogram for each Gabor response
 *   4. Compute RGB color histogram
 *   5. Concatenate all histograms
 */
cv::Mat GaborTextureColorFeature::extractFeatures(const cv::Mat& image) {
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for Gabor texture extraction" << std::endl;
        return cv::Mat();
    }
    
    // Convert to color if grayscale
    cv::Mat colorImage = image;
    if (image.channels() == 1) {
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
    }
    
    // Convert to grayscale for Gabor filtering
    cv::Mat gray;
    cv::cvtColor(colorImage, gray, cv::COLOR_BGR2GRAY);
    
    // Apply Gabor filter bank
    std::vector<cv::Mat> gaborResponses = applyGaborFilters(gray);
    
    // Compute texture features from Gabor responses
    cv::Mat textureHist = computeGaborHistogram(gaborResponses);
    
    if (textureHist.empty()) {
        std::cerr << "Error: Failed to compute Gabor histogram" << std::endl;
        return cv::Mat();
    }
    
    // Compute color features
    cv::Mat colorHist = computeColorHistogram(colorImage);
    
    if (colorHist.empty()) {
        std::cerr << "Error: Failed to compute color histogram" << std::endl;
        return cv::Mat();
    }
    
    // Concatenate texture and color
    int totalDim = textureHist.cols + colorHist.cols;
    cv::Mat combinedFeatures(1, totalDim, CV_32F);
    
    for (int i = 0; i < textureHist.cols; i++) {
        combinedFeatures.at<float>(0, i) = textureHist.at<float>(0, i);
    }
    
    for (int i = 0; i < colorHist.cols; i++) {
        combinedFeatures.at<float>(0, textureHist.cols + i) = colorHist.at<float>(0, i);
    }
    
    return combinedFeatures;
}

std::string GaborTextureColorFeature::getFeatureName() const {
    return "GaborTextureColor_" + std::to_string(numOrientations_) + "orient_" 
           + std::to_string(numScales_) + "scales";
}

int GaborTextureColorFeature::getFeatureDimension() const {
    return getTextureDimension() + getColorDimension();
}

/**
 * Create Gabor filter kernel using OpenCV's built-in function
 * 
 * OpenCV's getGaborKernel implements:
 *   G(x,y) = exp(-(x'²+γ²y'²)/(2σ²)) × cos(2π(x'/λ) + ψ)
 * 
 * @param ksize Kernel size (odd number)
 * @param sigma Gaussian envelope standard deviation
 * @param theta Orientation in radians
 * @param lambda Wavelength (controls scale/frequency)
 * @param gamma Spatial aspect ratio
 * @param psi Phase offset
 * @return Gabor kernel (using OpenCV's implementation)
 */
cv::Mat GaborTextureColorFeature::createGaborKernel(int ksize, double sigma, 
                                                    double theta, double lambda, 
                                                    double gamma, double psi) const {
    // Use OpenCV's built-in Gabor kernel generator
    // This is allowed per project specifications for Task 4
    return cv::getGaborKernel(
        cv::Size(ksize, ksize),  // Kernel size
        sigma,                    // Standard deviation of Gaussian
        theta,                    // Orientation (radians)
        lambda,                   // Wavelength
        gamma,                    // Spatial aspect ratio
        psi,                      // Phase offset
        CV_32F                    // Kernel type
    );
}

/**
 * Apply Gabor filter bank to image
 * 
 * Creates and applies multiple Gabor filters with different orientations
 * and scales to capture comprehensive texture information.
 * 
 * Default configuration:
 *   - Orientations: 0°, 45°, 90°, 135° (4 orientations)
 *   - Scales: λ=5 (fine), λ=10 (coarse) (2 scales)
 *   - Total: 4 × 2 = 8 Gabor filters
 * 
 * @param image Grayscale image
 * @return Vector of 8 filtered images
 */
std::vector<cv::Mat> GaborTextureColorFeature::applyGaborFilters(const cv::Mat& image) const {
    std::vector<cv::Mat> responses;
    
    // Convert to float for filtering
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    
    // Gabor filter parameters
    int ksize = 31;        // Kernel size
    double sigma = 5.0;    // Gaussian envelope std dev
    double gamma = 0.5;    // Spatial aspect ratio
    double psi = 0.0;      // Phase offset
    
    // Wavelengths for different scales
    std::vector<double> wavelengths = {5.0, 10.0};  // Fine and coarse
    
    // Orientations (in radians)
    std::vector<double> orientations;
    for (int i = 0; i < numOrientations_; i++) {
        double theta = (M_PI * i) / numOrientations_;  // 0°, 45°, 90°, 135° for 4 orientations
        orientations.push_back(theta);
    }
    
    // Apply each combination of orientation and scale
    for (int s = 0; s < numScales_; s++) {
        double lambda = wavelengths[std::min(s, static_cast<int>(wavelengths.size()) - 1)];
        
        for (int o = 0; o < numOrientations_; o++) {
            double theta = orientations[o];
            
            // Create Gabor kernel
            cv::Mat kernel = createGaborKernel(ksize, sigma, theta, lambda, gamma, psi);
            
            // Apply filter using convolution
            cv::Mat filtered;
            cv::filter2D(imageFloat, filtered, CV_32F, kernel);
            
            // Take absolute value of response
            cv::Mat absFiltered = cv::abs(filtered);
            
            responses.push_back(absFiltered);
        }
    }
    
    return responses;
}

/**
 * Compute histogram from Gabor filter responses
 * 
 * For each Gabor-filtered image:
 *   1. Create histogram of response magnitudes
 *   2. Normalize histogram
 *   3. Concatenate all histograms
 * 
 * Result: [hist_gabor0, hist_gabor1, ..., hist_gabor7]
 * 
 * @param gaborResponses Vector of Gabor-filtered images
 * @return Concatenated Gabor texture histogram
 */
cv::Mat GaborTextureColorFeature::computeGaborHistogram(
    const std::vector<cv::Mat>& gaborResponses) const {
    
    std::vector<cv::Mat> histograms;
    
    // Compute histogram for each Gabor response
    for (const auto& response : gaborResponses) {
        cv::Mat histogram = cv::Mat::zeros(1, binsPerGabor_, CV_32F);
        
        // Use fixed range for consistency
        float maxRange = 100.0f;
        float binSize = maxRange / binsPerGabor_;
        
        // Compute histogram (RAW counts, don't normalize yet)
        for (int row = 0; row < response.rows; row++) {
            for (int col = 0; col < response.cols; col++) {
                float value = response.at<float>(row, col);
                value = std::min(value, maxRange);
                
                int bin = static_cast<int>(value / binSize);
                bin = std::min(bin, binsPerGabor_ - 1);
                bin = std::max(bin, 0);
                
                histogram.at<float>(0, bin) += 1.0f;
            }
        }
        
        // ⭐ DON'T normalize individual histograms here!
        histograms.push_back(histogram);
    }
    
    // Verify correct number of histograms
    int expectedHistograms = numOrientations_ * numScales_;
    if (static_cast<int>(histograms.size()) != expectedHistograms) {
        std::cerr << "ERROR: Expected " << expectedHistograms << " histograms, got " 
                  << histograms.size() << std::endl;
        return cv::Mat();
    }
    
    // Concatenate all Gabor histograms
    int totalTextureBins = binsPerGabor_ * static_cast<int>(histograms.size());
    cv::Mat combinedTexture(1, totalTextureBins, CV_32F);
    
    int offset = 0;
    for (const auto& hist : histograms) {
        for (int i = 0; i < hist.cols; i++) {
            combinedTexture.at<float>(0, offset + i) = hist.at<float>(0, i);
        }
        offset += hist.cols;
    }
    
    // ⭐ NOW normalize the CONCATENATED histogram (so sum = 1.0)
    if (normalize_) {
        double sum = 0.0;
        for (int i = 0; i < combinedTexture.cols; i++) {
            sum += combinedTexture.at<float>(0, i);
        }
        
        if (sum > 1e-10) {
            combinedTexture /= sum;
        } else {
            std::cerr << "Warning: Combined texture histogram sum is zero" << std::endl;
        }
    }
    
    return combinedTexture;
}

/**
 * Compute RGB color histogram
 * 
 * Same implementation as TextureColorFeature
 */
cv::Mat GaborTextureColorFeature::computeColorHistogram(const cv::Mat& image) const {
    // Create 3D histogram
    int dims[3] = {colorBinsPerChannel_, colorBinsPerChannel_, colorBinsPerChannel_};
    cv::Mat histogram = cv::Mat::zeros(3, dims, CV_32F);
    
    float binSize = 256.0f / colorBinsPerChannel_;
    
    // Count pixels
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
    
    // Flatten to 1D
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
    
    // Normalize
    if (normalize_) {
        double sum = 0.0;
        for (int i = 0; i < flattened.cols; i++) {
            sum += flattened.at<float>(0, i);
        }
        if (sum > 1e-10) {
            flattened /= sum;
        }
    }
    
    return flattened;
}

cv::Mat GaborTextureColorFeature::normalizeHistogram(const cv::Mat& histogram) const {
    double sum = 0.0;
    for (int i = 0; i < histogram.cols; i++) {
        sum += histogram.at<float>(0, i);
    }
    
    if (sum < 1e-10) {
        return histogram.clone();
    }
    
    cv::Mat normalized = histogram.clone();
    normalized /= sum;
    return normalized;
}

} // namespace cbir
