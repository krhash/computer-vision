////////////////////////////////////////////////////////////////////////////////
// FeatureExtractor.h
// Author: Krushna Sanjay Sharma
// Description: Abstract base class for all feature extraction methods in the
//              Content-Based Image Retrieval (CBIR) system. This provides a
//              unified interface for different feature extraction techniques.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace cbir {

/**
 * @class FeatureExtractor
 * @brief Abstract base class for feature extraction methods
 * 
 * This class defines the interface that all feature extraction methods must
 * implement. Feature extractors convert images into numerical feature vectors
 * that can be compared using distance metrics.
 * 
 * Derived classes must implement:
 * - extractFeatures(): Extract features from an image
 * - getFeatureName(): Return a descriptive name for the feature type
 * - getFeatureDimension(): Return the dimension of the feature vector
 * 
 * @author Krushna Sanjay Sharma
 */
class FeatureExtractor {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~FeatureExtractor() = default;

    /**
     * @brief Extract features from an input image
     * 
     * This pure virtual function must be implemented by all derived classes.
     * It takes an input image and returns a feature vector as a cv::Mat.
     * 
     * @param image Input image (can be grayscale or color)
     * @return cv::Mat Feature vector as a single row matrix (1 x N)
     *         or multi-dimensional matrix for histogram-based features
     * 
     * @note The returned cv::Mat should be of type CV_32F or CV_64F for
     *       compatibility with distance metrics
     */
    virtual cv::Mat extractFeatures(const cv::Mat& image) = 0;

    /**
     * @brief Get the name/type of this feature extractor
     * 
     * @return std::string Descriptive name (e.g., "Baseline", "ColorHistogram")
     */
    virtual std::string getFeatureName() const = 0;

    /**
     * @brief Get the dimension of the feature vector
     * 
     * For 1D vectors, this is the length. For multi-dimensional features
     * (e.g., 2D histograms), this represents the total number of elements.
     * 
     * @return int Total dimension/size of the feature vector
     */
    virtual int getFeatureDimension() const = 0;

    /**
     * @brief Check if an image is valid for feature extraction
     * 
     * @param image Input image to validate
     * @return bool True if image is valid, false otherwise
     */
    virtual bool isValidImage(const cv::Mat& image) const {
        return !image.empty() && (image.channels() == 1 || image.channels() == 3);
    }

protected:
    /**
     * @brief Protected constructor - only derived classes can instantiate
     */
    FeatureExtractor() = default;

    /**
     * @brief Normalize feature vector to [0, 1] range
     * 
     * Utility function for derived classes to normalize their features.
     * 
     * @param features Input feature vector
     * @return cv::Mat Normalized feature vector
     */
    cv::Mat normalizeFeatures(const cv::Mat& features) const;
};

// Type alias for smart pointer to FeatureExtractor
using FeatureExtractorPtr = std::shared_ptr<FeatureExtractor>;

} // namespace cbir

#endif // FEATURE_EXTRACTOR_H
