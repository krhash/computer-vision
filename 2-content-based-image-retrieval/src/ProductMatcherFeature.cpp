////////////////////////////////////////////////////////////////////////////////
// ProductMatcherFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of ProductMatcher custom feature for Task 7.
//              Combines DNN embeddings with center-region color histogram
//              to match objects by type and appearance while ignoring backgrounds.
//
// Key Innovation:
//   Center-region extraction focuses on centered subjects (typical in
//   product photography), filtering out background distractions.
//
// Algorithm:
//   1. Load pre-computed DNN features by filename
//   2. Extract center 50% region from image (subject focus)
//   3. Compute RGB histogram on center region only
//   4. Concatenate both components
//
// This addresses the observation that histogram matching includes
// backgrounds (like walls), while we want to match subjects only.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "ProductMatcherFeature.h"
#include <iostream>

namespace cbir {

/**
 * Constructor
 */
ProductMatcherFeature::ProductMatcherFeature(const std::string& dnnCsvPath,
                                           double centerRatio,
                                           int colorBins)
    : dnnExtractor_(dnnCsvPath),
      centerRatio_(centerRatio),
      colorBins_(colorBins) {
    
    // Validate parameters
    if (centerRatio_ <= 0.0 || centerRatio_ > 1.0) {
        std::cerr << "Warning: Center ratio should be (0, 1]. Using 0.5" << std::endl;
        centerRatio_ = 0.5;
    }
    
    if (colorBins_ <= 0) {
        std::cerr << "Warning: Color bins should be positive. Using 8" << std::endl;
        colorBins_ = 8;
    }
}

/**
 * Extract features (base class requirement)
 * 
 * For ProductMatcher, we need filename for DNN lookup.
 * This function returns empty - use extractFeaturesWithFilename instead.
 */
cv::Mat ProductMatcherFeature::extractFeatures(const cv::Mat& image) {
    std::cerr << "Warning: ProductMatcherFeature requires filename." << std::endl;
    std::cerr << "Use extractFeaturesWithFilename() instead." << std::endl;
    return cv::Mat();
}

/**
 * Extract combined DNN + center-color features
 * 
 * This is the main feature extraction function for Task 7.
 */
cv::Mat ProductMatcherFeature::extractFeaturesWithFilename(const cv::Mat& image, 
                                                          const std::string& filename) {
    // Validate image
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for ProductMatcher" << std::endl;
        return cv::Mat();
    }
    
    // Step 1: Get DNN features by filename lookup
    cv::Mat dnnFeatures = dnnExtractor_.getFeaturesByFilename(filename);
    
    if (dnnFeatures.empty()) {
        std::cerr << "Error: No DNN features found for " << filename << std::endl;
        return cv::Mat();
    }
    
    // Ensure DNN features are correct dimension
    if (dnnFeatures.cols != 512) {
        std::cerr << "Error: DNN features should be 512D, got " 
                  << dnnFeatures.cols << std::endl;
        return cv::Mat();
    }
    
    // Step 2: Extract center region (focus on subject, ignore background)
    cv::Mat centerRegion = extractCenterRegion(image);
    
    if (centerRegion.empty()) {
        std::cerr << "Error: Failed to extract center region" << std::endl;
        return cv::Mat();
    }
    
    // Step 3: Compute RGB histogram on center region only
    cv::Mat centerColorHist = computeRGBHistogram(centerRegion);
    
    if (centerColorHist.empty()) {
        std::cerr << "Error: Failed to compute center color histogram" << std::endl;
        return cv::Mat();
    }
    
    // Step 4: Concatenate DNN and center-color features
    int totalDim = dnnFeatures.cols + centerColorHist.cols;
    cv::Mat combinedFeatures(1, totalDim, CV_32F);
    
    // Copy DNN features (first 512 values)
    for (int i = 0; i < dnnFeatures.cols; i++) {
        combinedFeatures.at<float>(0, i) = dnnFeatures.at<float>(0, i);
    }
    
    // Copy center-color histogram (next 512 values)
    for (int i = 0; i < centerColorHist.cols; i++) {
        combinedFeatures.at<float>(0, dnnFeatures.cols + i) = centerColorHist.at<float>(0, i);
    }
    
    return combinedFeatures;
}

std::string ProductMatcherFeature::getFeatureName() const {
    return "ProductMatcher_DNN+CenterColor";
}

int ProductMatcherFeature::getFeatureDimension() const {
    return 512 + (colorBins_ * colorBins_ * colorBins_);
}

void ProductMatcherFeature::setCenterRatio(double ratio) {
    if (ratio > 0.0 && ratio <= 1.0) {
        centerRatio_ = ratio;
    }
}

/**
 * Extract center region from image
 * 
 * For product photos, subjects are typically centered.
 * This extracts the center portion to focus analysis on the subject.
 * 
 * Example with centerRatio=0.5:
 *   Original: 800×600
 *   Center:   400×300 (centered)
 *   Margins:  200px left/right, 150px top/bottom
 * 
 * Visual:
 *   ┌────────────────────┐
 *   │    Background      │ ← Excluded (top margin)
 *   │  ┌──────────────┐  │
 *   │  │    CENTER    │  │ ← Extracted (subject)
 *   │  │    REGION    │  │
 *   │  └──────────────┘  │
 *   │    Background      │ ← Excluded (bottom margin)
 *   └────────────────────┘
 */
cv::Mat ProductMatcherFeature::extractCenterRegion(const cv::Mat& image) const {
    // Calculate center region dimensions
    int centerWidth = static_cast<int>(image.cols * centerRatio_);
    int centerHeight = static_cast<int>(image.rows * centerRatio_);
    
    // Calculate starting position (to center the region)
    int startX = (image.cols - centerWidth) / 2;
    int startY = (image.rows - centerHeight) / 2;
    
    // Define ROI (Region of Interest)
    cv::Rect centerROI(startX, startY, centerWidth, centerHeight);
    
    // Extract and clone the region
    cv::Mat centerRegion = image(centerROI).clone();
    
    return centerRegion;
}

/**
 * Compute RGB color histogram from image region
 * 
 * Same algorithm as HistogramFeature, but applied only to
 * the extracted center region.
 */
cv::Mat ProductMatcherFeature::computeRGBHistogram(const cv::Mat& region) const {
    // Convert to color if grayscale
    cv::Mat colorRegion = region;
    if (region.channels() == 1) {
        cv::cvtColor(region, colorRegion, cv::COLOR_GRAY2BGR);
    }
    
    // Create 3D RGB histogram
    int dims[3] = {colorBins_, colorBins_, colorBins_};
    cv::Mat histogram = cv::Mat::zeros(3, dims, CV_32F);
    
    float binSize = 256.0f / colorBins_;
    
    // Count pixels into bins
    for (int row = 0; row < colorRegion.rows; row++) {
        for (int col = 0; col < colorRegion.cols; col++) {
            cv::Vec3b pixel = colorRegion.at<cv::Vec3b>(row, col);
            
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            
            int binB = static_cast<int>(b / binSize);
            int binG = static_cast<int>(g / binSize);
            int binR = static_cast<int>(r / binSize);
            
            binB = std::min(binB, colorBins_ - 1);
            binG = std::min(binG, colorBins_ - 1);
            binR = std::min(binR, colorBins_ - 1);
            
            histogram.at<float>(binR, binG, binB) += 1.0f;
        }
    }
    
    // Flatten to 1D
    int totalBins = colorBins_ * colorBins_ * colorBins_;
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
    
    // Normalize to sum=1.0
    double sum = 0.0;
    for (int i = 0; i < flattened.cols; i++) {
        sum += flattened.at<float>(0, i);
    }
    
    if (sum > 1e-10) {
        flattened /= sum;
    }
    
    return flattened;
}

} // namespace cbir
