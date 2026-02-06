////////////////////////////////////////////////////////////////////////////////
// HistogramFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of color histogram feature extraction for CBIR.
//              Manually computes RGB and RG chromaticity histograms without
//              using OpenCV's calcHist function.
////////////////////////////////////////////////////////////////////////////////

#include "HistogramFeature.h"
#include <iostream>
#include <cmath>

namespace cbir {

HistogramFeature::HistogramFeature(HistogramType type, int binsPerChannel, bool normalize)
    : type_(type), binsPerChannel_(binsPerChannel), normalize_(normalize) {
    
    if (binsPerChannel_ <= 0) {
        std::cerr << "Warning: Bins per channel must be positive. Using 8." << std::endl;
        binsPerChannel_ = 8;
    }
}

cv::Mat HistogramFeature::extractFeatures(const cv::Mat& image) {
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for histogram extraction" << std::endl;
        return cv::Mat();
    }
    
    // Convert to BGR if grayscale
    cv::Mat colorImage = image;
    if (image.channels() == 1) {
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
    }
    
    // Compute histogram based on type
    cv::Mat histogram;
    if (type_ == HistogramType::RGB) {
        histogram = computeRGBHistogram(colorImage);
    } else {
        histogram = computeRGChromaticityHistogram(colorImage);
    }
    
    if (histogram.empty()) {
        return cv::Mat();
    }
    
    // Normalize if requested
    if (normalize_) {
        histogram = normalizeHistogram(histogram);
    }
    
    // Flatten to 1D vector
    return flattenHistogram(histogram);
}

std::string HistogramFeature::getFeatureName() const {
    std::string name = (type_ == HistogramType::RGB) ? "RGB" : "RGChromaticity";
    name += "Histogram_" + std::to_string(binsPerChannel_) + "bins";
    return name;
}

int HistogramFeature::getFeatureDimension() const {
    if (type_ == HistogramType::RGB) {
        return binsPerChannel_ * binsPerChannel_ * binsPerChannel_; // 3D histogram
    } else {
        return binsPerChannel_ * binsPerChannel_; // 2D histogram
    }
}

void HistogramFeature::setHistogramType(HistogramType type) {
    type_ = type;
}

void HistogramFeature::setBinsPerChannel(int bins) {
    if (bins > 0) {
        binsPerChannel_ = bins;
    }
}

void HistogramFeature::setNormalize(bool normalize) {
    normalize_ = normalize;
}

cv::Mat HistogramFeature::computeRGBHistogram(const cv::Mat& image) {
    // Create 3D histogram: bins for R, G, B
    int dims[3] = {binsPerChannel_, binsPerChannel_, binsPerChannel_};
    cv::Mat histogram = cv::Mat::zeros(3, dims, CV_32F);
    
    // Compute bin size for each channel (0-255 -> 0 to binsPerChannel-1)
    float binSize = 256.0f / binsPerChannel_;
    
    // Manually compute histogram by iterating through all pixels
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            // Get BGR values (OpenCV uses BGR order)
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            
            // Compute bin indices
            int binB = static_cast<int>(b / binSize);
            int binG = static_cast<int>(g / binSize);
            int binR = static_cast<int>(r / binSize);
            
            // Clamp to valid range (handle edge case of 255)
            binB = std::min(binB, binsPerChannel_ - 1);
            binG = std::min(binG, binsPerChannel_ - 1);
            binR = std::min(binR, binsPerChannel_ - 1);
            
            // Increment histogram bin
            histogram.at<float>(binR, binG, binB) += 1.0f;
        }
    }
    
    return histogram;
}

cv::Mat HistogramFeature::computeRGChromaticityHistogram(const cv::Mat& image) {
    // Create 2D histogram: bins for r, g chromaticity
    cv::Mat histogram = cv::Mat::zeros(binsPerChannel_, binsPerChannel_, CV_32F);
    
    // Compute histogram by iterating through all pixels
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            // Get BGR values
            float b = static_cast<float>(pixel[0]);
            float g = static_cast<float>(pixel[1]);
            float r = static_cast<float>(pixel[2]);
            
            // Compute sum
            float sum = r + g + b;
            
            // Avoid division by zero
            if (sum < 1.0f) {
                sum = 1.0f;
            }
            
            // Compute chromaticity values (normalized to [0, 1])
            float r_chrom = r / sum;
            float g_chrom = g / sum;
            
            // Compute bin indices (chromaticity is in [0, 1])
            int binR = static_cast<int>(r_chrom * binsPerChannel_);
            int binG = static_cast<int>(g_chrom * binsPerChannel_);
            
            // Clamp to valid range
            binR = std::min(binR, binsPerChannel_ - 1);
            binG = std::min(binG, binsPerChannel_ - 1);
            
            // Increment histogram bin
            histogram.at<float>(binR, binG) += 1.0f;
        }
    }
    
    return histogram;
}

cv::Mat HistogramFeature::flattenHistogram(const cv::Mat& histogram) {
    // Reshape multi-dimensional histogram to 1D row vector
    int totalBins = 1;
    for (int i = 0; i < histogram.dims; i++) {
        totalBins *= histogram.size[i];
    }
    
    cv::Mat flattened(1, totalBins, CV_32F);
    
    // Copy all values
    int idx = 0;
    if (histogram.dims == 2) {
        // 2D histogram (RG chromaticity)
        for (int i = 0; i < histogram.rows; i++) {
            for (int j = 0; j < histogram.cols; j++) {
                flattened.at<float>(0, idx++) = histogram.at<float>(i, j);
            }
        }
    } else if (histogram.dims == 3) {
        // 3D histogram (RGB)
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

cv::Mat HistogramFeature::normalizeHistogram(const cv::Mat& histogram) {
    // Compute sum of all histogram values
    double sum = 0.0;
    
    if (histogram.dims == 2) {
        for (int i = 0; i < histogram.rows; i++) {
            for (int j = 0; j < histogram.cols; j++) {
                sum += histogram.at<float>(i, j);
            }
        }
    } else if (histogram.dims == 3) {
        const int* sizes = histogram.size.p;
        for (int i = 0; i < sizes[0]; i++) {
            for (int j = 0; j < sizes[1]; j++) {
                for (int k = 0; k < sizes[2]; k++) {
                    sum += histogram.at<float>(i, j, k);
                }
            }
        }
    }
    
    // Avoid division by zero
    if (sum < 1e-10) {
        std::cerr << "Warning: Histogram sum is zero or near-zero" << std::endl;
        return histogram.clone();
    }
    
    // Normalize by dividing by sum
    cv::Mat normalized = histogram.clone();
    normalized /= sum;
    
    return normalized;
}

} // namespace cbir
