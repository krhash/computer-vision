////////////////////////////////////////////////////////////////////////////////
// BaselineFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of baseline feature extractor that extracts a
//              7x7 pixel square from the center of images for CBIR Task 1.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "BaselineFeature.h"
#include <iostream>

namespace cbir {

/**
 * @brief Default constructor - creates 7x7 baseline extractor
 * 
 * @author Krushna Sanjay Sharma
 */
BaselineFeature::BaselineFeature() : squareSize_(7) {
    // Default 7x7 square size
}

/**
 * @brief Constructor with custom square size
 * 
 * @author Krushna Sanjay Sharma
 */
BaselineFeature::BaselineFeature(int squareSize) : squareSize_(squareSize) {
    if (squareSize_ <= 0 || squareSize_ % 2 == 0) {
        std::cerr << "Warning: Square size should be positive and odd. Using 7." 
                  << std::endl;
        squareSize_ = 7;
    }
}

/**
 * @brief Extract 7x7 center square as feature vector
 * 
 * This function extracts a square region from the center of the image
 * and flattens it into a 1D feature vector. The pixels are read row by row.
 * 
 * For color images: RGB values are stored sequentially (R1,G1,B1,R2,G2,B2,...)
 * For grayscale: pixel values are stored directly (P1,P2,P3,...)
 * 
 * @author Krushna Sanjay Sharma
 */
cv::Mat BaselineFeature::extractFeatures(const cv::Mat& image) {
    // Validate input image
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for feature extraction" << std::endl;
        return cv::Mat();
    }
    
    // Check if image is large enough
    if (image.rows < squareSize_ || image.cols < squareSize_) {
        std::cerr << "Error: Image too small. Need at least " << squareSize_ 
                  << "x" << squareSize_ << " pixels" << std::endl;
        return cv::Mat();
    }
    
    // Extract center square
    cv::Mat centerSquare = extractCenterSquare(image);
    
    if (centerSquare.empty()) {
        return cv::Mat();
    }
    
    // Flatten to 1D feature vector
    int numChannels = centerSquare.channels();
    int totalElements = squareSize_ * squareSize_ * numChannels;
    
    cv::Mat features(1, totalElements, CV_32F);
    
    int idx = 0;
    for (int row = 0; row < centerSquare.rows; row++) {
        for (int col = 0; col < centerSquare.cols; col++) {
            if (numChannels == 1) {
                // Grayscale image
                features.at<float>(0, idx++) = 
                    static_cast<float>(centerSquare.at<uchar>(row, col));
            } else if (numChannels == 3) {
                // Color image (BGR in OpenCV)
                cv::Vec3b pixel = centerSquare.at<cv::Vec3b>(row, col);
                features.at<float>(0, idx++) = static_cast<float>(pixel[0]); // B
                features.at<float>(0, idx++) = static_cast<float>(pixel[1]); // G
                features.at<float>(0, idx++) = static_cast<float>(pixel[2]); // R
            }
        }
    }
    
    return features;
}

/**
 * @brief Get feature extractor name
 * 
 * @author Krushna Sanjay Sharma
 */
std::string BaselineFeature::getFeatureName() const {
    return "Baseline" + std::to_string(squareSize_) + "x" + std::to_string(squareSize_);
}

/**
 * @brief Get feature dimension
 * 
 * @author Krushna Sanjay Sharma
 */
int BaselineFeature::getFeatureDimension() const {
    // 7x7 = 49 for grayscale
    // 7x7x3 = 147 for color
    // We return the grayscale dimension; actual dimension depends on image
    return squareSize_ * squareSize_;
}

/**
 * @brief Set square size
 * 
 * @author Krushna Sanjay Sharma
 */
void BaselineFeature::setSquareSize(int size) {
    if (size <= 0 || size % 2 == 0) {
        std::cerr << "Warning: Square size should be positive and odd" << std::endl;
        return;
    }
    squareSize_ = size;
}

/**
 * @brief Extract center square from image
 * 
 * Calculates the center position and extracts a square region of size
 * squareSize_ x squareSize_ from the image center.
 * 
 * @author Krushna Sanjay Sharma
 */
cv::Mat BaselineFeature::extractCenterSquare(const cv::Mat& image) const {
    // Calculate center position
    int centerRow = image.rows / 2;
    int centerCol = image.cols / 2;
    
    // Calculate top-left corner of the square
    int halfSize = squareSize_ / 2;
    int startRow = centerRow - halfSize;
    int startCol = centerCol - halfSize;
    
    // Define region of interest (ROI)
    cv::Rect roi(startCol, startRow, squareSize_, squareSize_);
    
    // Extract and return the square region
    cv::Mat square = image(roi).clone();
    
    return square;
}

} // namespace cbir
