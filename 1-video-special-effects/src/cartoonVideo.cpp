/*
  Author: Krushna Sanjay Sharma
  Date: January 25, 2026
  Purpose: Implementation of video cartoonization using bilateral filtering and DoG
           Based on Winnemöller et al. (2006) "Real-time video abstraction"
*/

#include "cartoonVideo.hpp"
#include <iostream>

/**
 * @brief Default constructor with recommended parameters
 * 
 * Default values tuned for real-time video processing:
 * - Bilateral filter: d=9, sigmaColor=90, sigmaSpace=90
 * - DoG: sigma1=0.5, sigma2=2.0, threshold=0.01
 * - Quantization: 8 levels
 * - Temporal smoothing: enabled, alpha=0.7
 */
CartoonVideo::CartoonVideo() 
    : bilateralD_(9),
      bilateralSigmaColor_(90.0),
      bilateralSigmaSpace_(90.0),
      dogSigma1_(0.5),
      dogSigma2_(2.0),
      dogThreshold_(0.01),
      quantizeLevels_(8),
      useTemporalSmoothing_(true),
      temporalAlpha_(0.7),
      hasFirstFrame_(false) {
}

/**
 * @brief Constructor with custom parameters
 * 
 * Allows fine-tuning of all algorithm parameters.
 * 
 * @param bilateralD Bilateral filter diameter (odd number, typically 5-15)
 * @param bilateralSigmaColor Color sigma (higher = more smoothing, typically 50-150)
 * @param bilateralSigmaSpace Space sigma (higher = larger neighborhood, typically 50-150)
 * @param dogSigma1 Small Gaussian sigma (typically 0.3-1.0)
 * @param dogSigma2 Large Gaussian sigma (typically 1.5-4.0, should be > 2*sigma1)
 * @param dogThreshold Edge threshold (typically 0.005-0.02)
 * @param quantizeLevels Color levels (typically 4-16)
 */
CartoonVideo::CartoonVideo(int bilateralD, double bilateralSigmaColor, 
                           double bilateralSigmaSpace, double dogSigma1, 
                           double dogSigma2, double dogThreshold, int quantizeLevels)
    : bilateralD_(bilateralD),
      bilateralSigmaColor_(bilateralSigmaColor),
      bilateralSigmaSpace_(bilateralSigmaSpace),
      dogSigma1_(dogSigma1),
      dogSigma2_(dogSigma2),
      dogThreshold_(dogThreshold),
      quantizeLevels_(quantizeLevels),
      useTemporalSmoothing_(true),
      temporalAlpha_(0.7),
      hasFirstFrame_(false) {
}

/**
 * @brief Main processing function implementing the cartoon effect pipeline
 * 
 * Implements the full algorithm from Winnemöller et al. (2006):
 * 1. Edge-preserving smoothing via bilateral filter
 * 2. Edge detection using Difference-of-Gaussians
 * 3. Color quantization for posterization
 * 4. Combination of edges and colors
 * 5. Temporal smoothing for video stability
 * 
 * @param src Input frame (BGR color image)
 * @param dst Output cartoon frame
 * @return 0 on success, -1 on error
 */
int CartoonVideo::processFrame(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "CartoonVideo: Invalid input image" << std::endl;
        return -1;
    }
    
    // Step 1: Apply bilateral filter for edge-preserving smoothing
    cv::Mat smoothed;
    applyBilateralFilter(src, smoothed);
    
    // Step 2: Detect edges using Difference-of-Gaussians
    cv::Mat edges;
    detectEdgesDoG(smoothed, edges);
    
    // Step 3: Quantize colors
    cv::Mat quantized;
    quantizeColors(smoothed, quantized);
    
    // Step 4: Combine edges with quantized colors
    cv::Mat cartoon;
    combineEdgesAndColors(quantized, edges, cartoon);
    
    // Step 5: Apply temporal smoothing for video coherence
    if (useTemporalSmoothing_) {
        applyTemporalSmoothing(cartoon, dst);
    } else {
        dst = cartoon.clone();
    }
    
    return 0;
}

/**
 * @brief Apply bilateral filter for edge-preserving smoothing
 * 
 * Bilateral filter is key to the cartoon effect - it smooths flat regions
 * while preserving strong edges. This creates the characteristic flat-shaded
 * appearance of cartoons.
 * 
 * The filter considers both spatial proximity and color similarity:
 * - Spatial: pixels close together
 * - Color: pixels with similar colors
 * 
 * @param src Input image
 * @param dst Output smoothed image
 */
void CartoonVideo::applyBilateralFilter(const cv::Mat &src, cv::Mat &dst) {
    // Apply bilateral filter
    // d: Diameter of pixel neighborhood
    // sigmaColor: Filter sigma in color space (larger = more colors mixed)
    // sigmaSpace: Filter sigma in coordinate space (larger = farther pixels influence)
    cv::bilateralFilter(src, dst, bilateralD_, 
                       bilateralSigmaColor_, bilateralSigmaSpace_);
}

/**
 * @brief Detect edges using Difference-of-Gaussians (DoG)
 * 
 * DoG is used instead of traditional edge detectors (Sobel, Canny) because:
 * 1. It approximates Laplacian-of-Gaussian (LoG)
 * 2. Produces cleaner, more consistent edges
 * 3. Works well with bilateral-filtered images
 * 
 * Algorithm:
 * 1. Convert to grayscale
 * 2. Apply two Gaussians with different sigmas
 * 3. Subtract: DoG = Gaussian(σ1) - Gaussian(σ2)
 * 4. Threshold to binary edge map
 * 
 * @param src Input image (should be smoothed)
 * @param edges Output binary edge map
 */
void CartoonVideo::detectEdgesDoG(const cv::Mat &src, cv::Mat &edges) {
    // Convert to grayscale for edge detection
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Apply two Gaussian blurs with different sigma values
    cv::Mat gauss1, gauss2;
    
    // Calculate kernel size from sigma: size = 2 * round(3*sigma) + 1
    int ksize1 = 2 * static_cast<int>(std::round(3 * dogSigma1_)) + 1;
    int ksize2 = 2 * static_cast<int>(std::round(3 * dogSigma2_)) + 1;
    
    // Ensure odd kernel sizes
    if (ksize1 % 2 == 0) ksize1++;
    if (ksize2 % 2 == 0) ksize2++;
    
    cv::GaussianBlur(gray, gauss1, cv::Size(ksize1, ksize1), dogSigma1_);
    cv::GaussianBlur(gray, gauss2, cv::Size(ksize2, ksize2), dogSigma2_);
    
    // Compute Difference-of-Gaussians
    cv::Mat dog;
    cv::subtract(gauss1, gauss2, dog, cv::noArray(), CV_32F);
    
    // Normalize DoG to [0, 1] range
    double minVal, maxVal;
    cv::minMaxLoc(dog, &minVal, &maxVal);
    
    if (maxVal - minVal > 1e-6) {
        dog = (dog - minVal) / (maxVal - minVal);
    }
    
    // Threshold to create binary edge map
    // Edges are where DoG response is BELOW threshold (dark lines)
    edges = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    for (int i = 0; i < dog.rows; i++) {
        const float* dogRow = dog.ptr<float>(i);
        uchar* edgeRow = edges.ptr<uchar>(i);
        
        for (int j = 0; j < dog.cols; j++) {
            // Invert: edge where response is low (< threshold)
            if (dogRow[j] < dogThreshold_) {
                edgeRow[j] = 255;
            }
        }
    }
    
    // Optional: Apply morphological operations to clean up edges
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(edges, edges, kernel);
}

/**
 * @brief Quantize colors into discrete levels (posterization)
 * 
 * Reduces the number of colors in the image to create a cartoon appearance.
 * Uses uniform quantization across all color channels.
 * 
 * Formula for each channel:
 *   bucketSize = 255 / levels
 *   quantized = (value / bucketSize) * bucketSize
 * 
 * @param src Input image
 * @param dst Output quantized image
 */
void CartoonVideo::quantizeColors(const cv::Mat &src, cv::Mat &dst) {
    dst.create(src.size(), src.type());
    
    int bucketSize = 255 / quantizeLevels_;
    
    // Quantize each pixel
    for (int i = 0; i < src.rows; i++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int value = srcRow[j][c];
                int quantized = (value / bucketSize) * bucketSize;
                dstRow[j][c] = static_cast<uchar>(quantized);
            }
        }
    }
}

/**
 * @brief Combine edge map with quantized colors
 * 
 * Creates the final cartoon appearance by darkening pixels at edge locations.
 * This creates the characteristic black outlines of cartoons.
 * 
 * Edge pixels are darkened by multiplying with a factor (typically 0.2-0.4).
 * 
 * @param quantized Input quantized color image
 * @param edges Input binary edge map
 * @param dst Output cartoon image with outlines
 */
void CartoonVideo::combineEdgesAndColors(const cv::Mat &quantized, 
                                        const cv::Mat &edges, cv::Mat &dst) {
    dst.create(quantized.size(), quantized.type());
    
    const float edgeDarkeningFactor = 0.3f;  // How much to darken edges
    
    for (int i = 0; i < quantized.rows; i++) {
        const cv::Vec3b* quantRow = quantized.ptr<cv::Vec3b>(i);
        const uchar* edgeRow = edges.ptr<uchar>(i);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(i);
        
        for (int j = 0; j < quantized.cols; j++) {
            if (edgeRow[j] > 0) {
                // Darken edge pixels
                for (int c = 0; c < 3; c++) {
                    dstRow[j][c] = static_cast<uchar>(
                        quantRow[j][c] * edgeDarkeningFactor
                    );
                }
            } else {
                // Keep quantized color for non-edge pixels
                dstRow[j] = quantRow[j];
            }
        }
    }
}

/**
 * @brief Apply temporal smoothing for video coherence
 * 
 * Key contribution of Winnemöller et al. (2006): temporal coherence.
 * Blends current frame with previous frame to reduce flickering and
 * maintain stability across video frames.
 * 
 * Uses exponential moving average:
 *   output = alpha * current + (1 - alpha) * previous
 * 
 * Higher alpha = more weight on current frame (less smoothing)
 * Lower alpha = more weight on previous frames (more smoothing)
 * 
 * @param current Current cartoon frame
 * @param dst Output temporally smoothed frame
 */
void CartoonVideo::applyTemporalSmoothing(const cv::Mat &current, cv::Mat &dst) {
    if (!hasFirstFrame_) {
        // First frame - just copy
        dst = current.clone();
        previousFrame_ = current.clone();
        hasFirstFrame_ = true;
        return;
    }
    
    // Ensure previous frame matches current dimensions
    if (previousFrame_.size() != current.size()) {
        previousFrame_ = current.clone();
        dst = current.clone();
        return;
    }
    
    // Exponential moving average blend
    cv::addWeighted(current, temporalAlpha_, 
                   previousFrame_, 1.0 - temporalAlpha_, 
                   0, dst);
    
    // Update previous frame
    previousFrame_ = dst.clone();
}

/**
 * @brief Reset temporal buffer (call when video stream restarts)
 */
void CartoonVideo::resetTemporalBuffer() {
    hasFirstFrame_ = false;
    previousFrame_.release();
}

/**
 * @brief Set bilateral filter parameters
 * 
 * @param d Neighborhood diameter (5-15, odd number recommended)
 * @param sigmaColor Color sigma (50-150)
 * @param sigmaSpace Space sigma (50-150)
 */
void CartoonVideo::setBilateralParams(int d, double sigmaColor, double sigmaSpace) {
    bilateralD_ = d;
    bilateralSigmaColor_ = sigmaColor;
    bilateralSigmaSpace_ = sigmaSpace;
}

/**
 * @brief Set DoG edge detection parameters
 * 
 * @param sigma1 Small Gaussian sigma (0.3-1.0)
 * @param sigma2 Large Gaussian sigma (1.5-4.0, should be > 2*sigma1)
 * @param threshold Edge threshold (0.005-0.02)
 */
void CartoonVideo::setDoGParams(double sigma1, double sigma2, double threshold) {
    dogSigma1_ = sigma1;
    dogSigma2_ = sigma2;
    dogThreshold_ = threshold;
}

/**
 * @brief Set color quantization levels
 * 
 * @param levels Number of color levels per channel (4-16 typical)
 */
void CartoonVideo::setQuantizeLevels(int levels) {
    quantizeLevels_ = std::max(2, std::min(255, levels));
}

/**
 * @brief Enable or disable temporal smoothing
 * 
 * @param enable True to enable temporal coherence, false to disable
 */
void CartoonVideo::setTemporalSmoothing(bool enable) {
    useTemporalSmoothing_ = enable;
    if (!enable) {
        resetTemporalBuffer();
    }
}

/**
 * @brief Set temporal smoothing strength
 * 
 * @param alpha Blending factor [0.0-1.0]
 *              1.0 = no smoothing (current frame only)
 *              0.5 = equal blend of current and previous
 *              0.0 = maximum smoothing (mostly previous frames)
 *              Typical: 0.6-0.8
 */
void CartoonVideo::setTemporalAlpha(double alpha) {
    temporalAlpha_ = std::max(0.0, std::min(1.0, alpha));
}
