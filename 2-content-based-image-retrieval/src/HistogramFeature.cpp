////////////////////////////////////////////////////////////////////////////////
// HistogramFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of color histogram feature extraction for CBIR.
//              Manually computes RGB and RG chromaticity histograms without
//              using OpenCV's calcHist function as required by project specs.
//
// Supports two histogram types:
//   1. RGB Histogram: 3D histogram of R, G, B color channels
//      - Bins entire image into color "buckets"
//      - Default: 8 bins per channel = 8×8×8 = 512 bins
//      - Captures overall color distribution
//
//   2. RG Chromaticity Histogram: 2D histogram of normalized r, g values
//      - r = R/(R+G+B), g = G/(R+G+B)
//      - Default: 16 bins per channel = 16×16 = 256 bins
//      - More robust to lighting variations
//
// Features are normalized (sum=1.0) and flattened to 1D vectors for comparison.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "HistogramFeature.h"
#include <iostream>
#include <cmath>

namespace cbir {

/**
 * Constructor with histogram configuration
 * 
 * @param type RGB or RG_CHROMATICITY
 * @param binsPerChannel Number of bins per channel (8 or 16 typical)
 * @param normalize If true, normalize histogram to sum=1.0
 */
HistogramFeature::HistogramFeature(HistogramType type, int binsPerChannel, bool normalize)
    : type_(type), binsPerChannel_(binsPerChannel), normalize_(normalize) {
    
    // Validate bins per channel
    if (binsPerChannel_ <= 0) {
        std::cerr << "Warning: Bins per channel must be positive. Using 8." << std::endl;
        binsPerChannel_ = 8;
    }
}

/**
 * Extract histogram features from image
 * 
 * Process:
 *   1. Validate and convert image to color if needed
 *   2. Compute histogram based on type (RGB or RG chromaticity)
 *   3. Normalize histogram if requested (sum=1.0)
 *   4. Flatten multi-dimensional histogram to 1D vector
 * 
 * @param image Input image (color or grayscale)
 * @return 1D feature vector (512 for RGB, 256 for RG chromaticity)
 */
cv::Mat HistogramFeature::extractFeatures(const cv::Mat& image) {
    // Validate input image
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for histogram extraction" << std::endl;
        return cv::Mat();
    }
    
    // Convert grayscale to color if needed (histograms need color)
    cv::Mat colorImage = image;
    if (image.channels() == 1) {
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
    }
    
    // Compute histogram based on configured type
    cv::Mat histogram;
    if (type_ == HistogramType::RGB) {
        histogram = computeRGBHistogram(colorImage);
    } else {
        histogram = computeRGChromaticityHistogram(colorImage);
    }
    
    if (histogram.empty()) {
        return cv::Mat();
    }
    
    // Normalize histogram if configured (divide all bins by total count)
    if (normalize_) {
        histogram = normalizeHistogram(histogram);
    }
    
    // Flatten multi-dimensional histogram to 1D vector for distance computation
    return flattenHistogram(histogram);
}

/**
 * Get descriptive name of this feature type
 * 
 * @return Name with type and bin count (e.g., "RGBHistogram_8bins")
 */
std::string HistogramFeature::getFeatureName() const {
    std::string name = (type_ == HistogramType::RGB) ? "RGB" : "RGChromaticity";
    name += "Histogram_" + std::to_string(binsPerChannel_) + "bins";
    return name;
}

/**
 * Get dimension of feature vector
 * 
 * @return Total number of bins (8³=512 for RGB, 16²=256 for RG)
 */
int HistogramFeature::getFeatureDimension() const {
    if (type_ == HistogramType::RGB) {
        return binsPerChannel_ * binsPerChannel_ * binsPerChannel_; // 3D histogram
    } else {
        return binsPerChannel_ * binsPerChannel_; // 2D histogram
    }
}

/**
 * Set histogram type
 */
void HistogramFeature::setHistogramType(HistogramType type) {
    type_ = type;
}

/**
 * Set number of bins per channel
 */
void HistogramFeature::setBinsPerChannel(int bins) {
    if (bins > 0) {
        binsPerChannel_ = bins;
    }
}

/**
 * Set normalization flag
 */
void HistogramFeature::setNormalize(bool normalize) {
    normalize_ = normalize;
}

/**
 * Compute RGB color histogram
 * 
 * Creates 3D histogram by binning each pixel into one of 8×8×8 bins
 * based on its RGB color values.
 * 
 * Algorithm:
 *   1. Create empty 3D histogram (8×8×8 for default)
 *   2. For each pixel in image:
 *      - Get R, G, B values (0-255)
 *      - Compute bin indices (divide by bin size: 256/8 = 32)
 *      - Increment corresponding histogram bin
 *   3. Return 3D histogram
 * 
 * Example:
 *   Pixel (R=200, G=100, B=50)
 *   → binR=6, binG=3, binB=1
 *   → histogram[6][3][1] += 1
 * 
 * @param image Color image (BGR format)
 * @return 3D histogram (8×8×8)
 */
cv::Mat HistogramFeature::computeRGBHistogram(const cv::Mat& image) {
    // Create 3D histogram: bins for R, G, B channels
    int dims[3] = {binsPerChannel_, binsPerChannel_, binsPerChannel_};
    cv::Mat histogram = cv::Mat::zeros(3, dims, CV_32F);
    
    // Compute bin size for each channel (256 values → N bins)
    // Example: 8 bins → bin size = 256/8 = 32
    // Values 0-31 go to bin 0, 32-63 to bin 1, etc.
    float binSize = 256.0f / binsPerChannel_;
    
    // Manually compute histogram by counting pixels in each bin
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            // Get BGR values (OpenCV uses BGR order, not RGB!)
            int b = pixel[0];  // Blue:  0-255
            int g = pixel[1];  // Green: 0-255
            int r = pixel[2];  // Red:   0-255
            
            // Compute bin indices by dividing by bin size
            int binB = static_cast<int>(b / binSize);  // 0 to 7
            int binG = static_cast<int>(g / binSize);  // 0 to 7
            int binR = static_cast<int>(r / binSize);  // 0 to 7
            
            // Clamp to valid range (handle edge case where value=255)
            binB = std::min(binB, binsPerChannel_ - 1);
            binG = std::min(binG, binsPerChannel_ - 1);
            binR = std::min(binR, binsPerChannel_ - 1);
            
            // Increment count for this color bin
            histogram.at<float>(binR, binG, binB) += 1.0f;
        }
    }
    
    return histogram;
}

/**
 * Compute RG chromaticity histogram
 * 
 * Creates 2D histogram using normalized r, g color values.
 * Chromaticity is lighting-invariant (same color under different brightness).
 * 
 * Algorithm:
 *   1. For each pixel:
 *      - Get R, G, B values
 *      - Compute sum = R + G + B
 *      - Normalize: r = R/sum, g = G/sum (range [0,1])
 *      - Compute bin indices
 *      - Increment histogram bin
 *   2. Return 2D histogram
 * 
 * Example:
 *   Pixel (R=200, G=100, B=50)
 *   → sum=350
 *   → r=0.571, g=0.286
 *   → binR=9, binG=4 (for 16 bins)
 *   → histogram[9][4] += 1
 * 
 * @param image Color image (BGR format)
 * @return 2D histogram (16×16)
 */
cv::Mat HistogramFeature::computeRGChromaticityHistogram(const cv::Mat& image) {
    // Create 2D histogram: bins for r, g chromaticity
    cv::Mat histogram = cv::Mat::zeros(binsPerChannel_, binsPerChannel_, CV_32F);
    
    // Compute histogram by iterating through all pixels
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            // Get BGR values and convert to float
            float b = static_cast<float>(pixel[0]);
            float g = static_cast<float>(pixel[1]);
            float r = static_cast<float>(pixel[2]);
            
            // Compute sum for normalization
            float sum = r + g + b;
            
            // Avoid division by zero for black pixels
            if (sum < 1.0f) {
                sum = 1.0f;
            }
            
            // Compute chromaticity values (normalized to [0, 1])
            // r_chrom = R/(R+G+B), g_chrom = G/(R+G+B)
            float r_chrom = r / sum;
            float g_chrom = g / sum;
            
            // Compute bin indices (chromaticity already in [0, 1])
            // Multiply by bins to get index: 0.5 * 16 = 8
            int binR = static_cast<int>(r_chrom * binsPerChannel_);
            int binG = static_cast<int>(g_chrom * binsPerChannel_);
            
            // Clamp to valid range (handle edge case where chrom=1.0)
            binR = std::min(binR, binsPerChannel_ - 1);
            binG = std::min(binG, binsPerChannel_ - 1);
            
            // Increment count for this chromaticity bin
            histogram.at<float>(binR, binG) += 1.0f;
        }
    }
    
    return histogram;
}

/**
 * Flatten multi-dimensional histogram to 1D vector
 * 
 * Converts 3D (8×8×8) or 2D (16×16) histogram into 1D row vector
 * for distance computation.
 * 
 * Reading order:
 *   - 2D: Row-major (row 0, row 1, ...)
 *   - 3D: [R][G][B] order (R changes slowest, B changes fastest)
 * 
 * @param histogram Multi-dimensional histogram
 * @return 1D row vector (1×N)
 */
cv::Mat HistogramFeature::flattenHistogram(const cv::Mat& histogram) {
    // Calculate total number of bins
    int totalBins = 1;
    for (int i = 0; i < histogram.dims; i++) {
        totalBins *= histogram.size[i];
    }
    
    // Create 1D row vector
    cv::Mat flattened(1, totalBins, CV_32F);
    
    // Copy all values from multi-dimensional histogram to 1D vector
    int idx = 0;
    if (histogram.dims == 2) {
        // 2D histogram (RG chromaticity): 16×16 = 256 values
        for (int i = 0; i < histogram.rows; i++) {
            for (int j = 0; j < histogram.cols; j++) {
                flattened.at<float>(0, idx++) = histogram.at<float>(i, j);
            }
        }
    } else if (histogram.dims == 3) {
        // 3D histogram (RGB): 8×8×8 = 512 values
        const int* sizes = histogram.size.p;
        for (int i = 0; i < sizes[0]; i++) {
            for (int j = 0; j < sizes[1]; j++) {
                for (int k = 0; k < sizes[2]; k++) {
                    flattened.at<float>(0, idx++) = histogram.at<float>(i, j, k);
                }
            }
        }
    }
    
    return flattened;
}

/**
 * Normalize histogram to sum=1.0
 * 
 * Converts raw pixel counts to probabilities by dividing each bin
 * by the total number of pixels.
 * 
 * Example:
 *   Before: [100, 50, 25, ...] (raw counts)
 *   Sum: 10000 pixels
 *   After: [0.01, 0.005, 0.0025, ...] (probabilities)
 * 
 * @param histogram Input histogram (any dimension)
 * @return Normalized histogram (sum=1.0)
 */
cv::Mat HistogramFeature::normalizeHistogram(const cv::Mat& histogram) {
    // Compute sum of all histogram values
    double sum = 0.0;
    
    if (histogram.dims == 2) {
        // Sum all bins in 2D histogram
        for (int i = 0; i < histogram.rows; i++) {
            for (int j = 0; j < histogram.cols; j++) {
                sum += histogram.at<float>(i, j);
            }
        }
    } else if (histogram.dims == 3) {
        // Sum all bins in 3D histogram
        const int* sizes = histogram.size.p;
        for (int i = 0; i < sizes[0]; i++) {
            for (int j = 0; j < sizes[1]; j++) {
                for (int k = 0; k < sizes[2]; k++) {
                    sum += histogram.at<float>(i, j, k);
                }
            }
        }
    }
    
    // Avoid division by zero (empty histogram)
    if (sum < 1e-10) {
        std::cerr << "Warning: Histogram sum is zero or near-zero" << std::endl;
        return histogram.clone();
    }
    
    // Normalize: divide each bin by total sum
    // Result: all bins sum to 1.0 (probabilities)
    cv::Mat normalized = histogram.clone();
    normalized /= sum;
    
    return normalized;
}

} // namespace cbir
