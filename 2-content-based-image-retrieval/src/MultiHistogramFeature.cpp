////////////////////////////////////////////////////////////////////////////////
// MultiHistogramFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of multi-region histogram feature extraction.
//              Splits image into regions, computes histogram for each region,
//              and concatenates them into a single feature vector.
//
// Process:
//   1. Split image into regions (top/bottom, left/right, or grid)
//   2. Compute histogram for each region using HistogramFeature
//   3. Concatenate all region histograms into single feature vector
//   4. Return combined feature vector
//
// Example with 2 horizontal regions (top/bottom):
//   - Top half histogram:    512 values
//   - Bottom half histogram: 512 values
//   - Combined:              1024 values
//
// During comparison, each region can be weighted differently.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "MultiHistogramFeature.h"
#include <iostream>

namespace cbir {

/**
 * Default constructor
 * Creates 2 horizontal regions (top/bottom) with RGB histograms
 */
MultiHistogramFeature::MultiHistogramFeature()
    : splitType_(SplitType::HORIZONTAL),
      numRegions_(2),
      histType_(HistogramFeature::HistogramType::RGB),
      binsPerChannel_(8),
      normalize_(true) {
    
    // Initialize equal weights for all regions
    initializeWeights();
}

/**
 * Constructor with custom configuration
 */
MultiHistogramFeature::MultiHistogramFeature(SplitType splitType,
                                           int numRegions,
                                           HistogramFeature::HistogramType histType,
                                           int binsPerChannel,
                                           bool normalize)
    : splitType_(splitType),
      numRegions_(numRegions),
      histType_(histType),
      binsPerChannel_(binsPerChannel),
      normalize_(normalize) {
    
    // Validate number of regions
    if (numRegions_ < 1) {
        std::cerr << "Warning: Number of regions must be at least 1. Using 2." << std::endl;
        numRegions_ = 2;
    }
    
    // Initialize equal weights
    initializeWeights();
}

/**
 * Extract multi-region histogram features
 * 
 * Process:
 *   1. Split image into regions based on configuration
 *   2. Create histogram extractor for region histograms
 *   3. Extract histogram from each region
 *   4. Concatenate all histograms into single feature vector
 * 
 * @param image Input image
 * @return Concatenated feature vector from all regions
 */
cv::Mat MultiHistogramFeature::extractFeatures(const cv::Mat& image) {
    // Validate input
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for multi-histogram extraction" << std::endl;
        return cv::Mat();
    }
    
    // Split image into regions
    std::vector<cv::Mat> regions = splitImage(image);
    
    if (regions.empty()) {
        std::cerr << "Error: Failed to split image into regions" << std::endl;
        return cv::Mat();
    }
    
    // Create histogram extractor for computing region histograms
    HistogramFeature histExtractor(histType_, binsPerChannel_, normalize_);
    
    // Extract histogram from each region
    std::vector<cv::Mat> regionHistograms;
    for (size_t i = 0; i < regions.size(); i++) {
        cv::Mat regionHist = histExtractor.extractFeatures(regions[i]);
        
        if (regionHist.empty()) {
            std::cerr << "Error: Failed to extract histogram from region " << i << std::endl;
            return cv::Mat();
        }
        
        regionHistograms.push_back(regionHist);
    }
    
    // Concatenate all region histograms into single feature vector
    // Result: [hist1, hist2, hist3, ...]
    int totalDimension = 0;
    for (const auto& hist : regionHistograms) {
        totalDimension += hist.cols;
    }
    
    cv::Mat combinedFeatures(1, totalDimension, CV_32F);
    
    int offset = 0;
    for (const auto& hist : regionHistograms) {
        // Copy this region's histogram into combined feature vector
        for (int j = 0; j < hist.cols; j++) {
            combinedFeatures.at<float>(0, offset + j) = hist.at<float>(0, j);
        }
        offset += hist.cols;
    }
    
    return combinedFeatures;
}

/**
 * Get feature name
 */
std::string MultiHistogramFeature::getFeatureName() const {
    std::string splitName;
    switch (splitType_) {
        case SplitType::HORIZONTAL: splitName = "Horizontal"; break;
        case SplitType::VERTICAL:   splitName = "Vertical"; break;
        case SplitType::GRID:       splitName = "Grid"; break;
    }
    
    std::string histName = (histType_ == HistogramFeature::HistogramType::RGB) 
                          ? "RGB" : "RGChromaticity";
    
    return "Multi" + histName + "Histogram_" + splitName + "_" 
           + std::to_string(numRegions_) + "regions_" 
           + std::to_string(binsPerChannel_) + "bins";
}

/**
 * Get total feature dimension
 * 
 * Total dimension = (histogram dimension) × (number of regions)
 * Example: RGB 8-bin histogram with 2 regions = 512 × 2 = 1024
 */
int MultiHistogramFeature::getFeatureDimension() const {
    int histDim;
    if (histType_ == HistogramFeature::HistogramType::RGB) {
        histDim = binsPerChannel_ * binsPerChannel_ * binsPerChannel_;
    } else {
        histDim = binsPerChannel_ * binsPerChannel_;
    }
    
    return histDim * numRegions_;
}

/**
 * Set split type
 */
void MultiHistogramFeature::setSplitType(SplitType type) {
    splitType_ = type;
}

/**
 * Set number of regions
 */
void MultiHistogramFeature::setNumRegions(int num) {
    if (num > 0) {
        numRegions_ = num;
        initializeWeights();  // Reinitialize weights for new number of regions
    }
}

/**
 * Get region weights for distance combination
 */
std::vector<double> MultiHistogramFeature::getRegionWeights() const {
    return regionWeights_;
}

/**
 * Set custom region weights
 */
void MultiHistogramFeature::setRegionWeights(const std::vector<double>& weights) {
    if (weights.size() != static_cast<size_t>(numRegions_)) {
        std::cerr << "Warning: Number of weights must match number of regions" << std::endl;
        return;
    }
    
    // Verify weights sum to approximately 1.0
    double sum = 0.0;
    for (double w : weights) {
        sum += w;
    }
    
    if (std::abs(sum - 1.0) > 0.01) {
        std::cerr << "Warning: Weights should sum to 1.0 (current sum: " << sum << ")" << std::endl;
    }
    
    regionWeights_ = weights;
}

/**
 * Split image into regions based on split type
 */
std::vector<cv::Mat> MultiHistogramFeature::splitImage(const cv::Mat& image) const {
    switch (splitType_) {
        case SplitType::HORIZONTAL:
            return splitHorizontal(image);
        case SplitType::VERTICAL:
            return splitVertical(image);
        case SplitType::GRID:
            return splitGrid(image);
        default:
            return splitHorizontal(image);
    }
}

/**
 * Split horizontally into regions (top to bottom)
 * 
 * Example with 2 regions:
 *   ┌─────────────┐
 *   │   Region 0  │  ← Top half
 *   ├─────────────┤
 *   │   Region 1  │  ← Bottom half
 *   └─────────────┘
 * 
 * Example with 3 regions:
 *   ┌─────────────┐
 *   │   Region 0  │  ← Top third
 *   ├─────────────┤
 *   │   Region 1  │  ← Middle third
 *   ├─────────────┤
 *   │   Region 2  │  ← Bottom third
 *   └─────────────┘
 */
std::vector<cv::Mat> MultiHistogramFeature::splitHorizontal(const cv::Mat& image) const {
    std::vector<cv::Mat> regions;
    
    int regionHeight = image.rows / numRegions_;
    
    for (int i = 0; i < numRegions_; i++) {
        int startRow = i * regionHeight;
        int endRow = (i == numRegions_ - 1) ? image.rows : (i + 1) * regionHeight;
        
        // Extract region using ROI (Region of Interest)
        cv::Rect roi(0, startRow, image.cols, endRow - startRow);
        regions.push_back(image(roi).clone());
    }
    
    return regions;
}

/**
 * Split vertically into regions (left to right)
 * 
 * Example with 2 regions:
 *   ┌──────┬──────┐
 *   │      │      │
 *   │Reg 0 │Reg 1 │
 *   │      │      │
 *   └──────┴──────┘
 *   Left    Right
 */
std::vector<cv::Mat> MultiHistogramFeature::splitVertical(const cv::Mat& image) const {
    std::vector<cv::Mat> regions;
    
    int regionWidth = image.cols / numRegions_;
    
    for (int i = 0; i < numRegions_; i++) {
        int startCol = i * regionWidth;
        int endCol = (i == numRegions_ - 1) ? image.cols : (i + 1) * regionWidth;
        
        cv::Rect roi(startCol, 0, endCol - startCol, image.rows);
        regions.push_back(image(roi).clone());
    }
    
    return regions;
}

/**
 * Split into grid (e.g., 2×2 = 4 regions, 3×3 = 9 regions)
 * 
 * Example with 4 regions (2×2):
 *   ┌──────┬──────┐
 *   │Reg 0 │Reg 1 │
 *   ├──────┼──────┤
 *   │Reg 2 │Reg 3 │
 *   └──────┴──────┘
 * 
 * numRegions must be a perfect square (4, 9, 16, etc.)
 */
std::vector<cv::Mat> MultiHistogramFeature::splitGrid(const cv::Mat& image) const {
    std::vector<cv::Mat> regions;
    
    // Calculate grid dimensions (assume square grid)
    int gridSize = static_cast<int>(std::sqrt(numRegions_));
    
    // If not perfect square, adjust to nearest square
    if (gridSize * gridSize != numRegions_) {
        gridSize = static_cast<int>(std::sqrt(numRegions_)) + 1;
        std::cerr << "Warning: Grid size adjusted to " << gridSize << "×" << gridSize 
                  << " = " << (gridSize * gridSize) << " regions" << std::endl;
    }
    
    int regionHeight = image.rows / gridSize;
    int regionWidth = image.cols / gridSize;
    
    // Extract each grid cell as a region
    for (int row = 0; row < gridSize; row++) {
        for (int col = 0; col < gridSize; col++) {
            int startRow = row * regionHeight;
            int startCol = col * regionWidth;
            
            int endRow = (row == gridSize - 1) ? image.rows : (row + 1) * regionHeight;
            int endCol = (col == gridSize - 1) ? image.cols : (col + 1) * regionWidth;
            
            cv::Rect roi(startCol, startRow, endCol - startCol, endRow - startRow);
            regions.push_back(image(roi).clone());
        }
    }
    
    return regions;
}

/**
 * Initialize equal weights for all regions
 * Each region gets weight = 1.0 / numRegions
 */
void MultiHistogramFeature::initializeWeights() {
    regionWeights_.clear();
    double weight = 1.0 / numRegions_;
    
    for (int i = 0; i < numRegions_; i++) {
        regionWeights_.push_back(weight);
    }
}

} // namespace cbir
