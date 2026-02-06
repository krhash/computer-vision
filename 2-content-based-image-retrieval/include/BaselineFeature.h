////////////////////////////////////////////////////////////////////////////////
// BaselineFeature.h
// Author: Krushna Sanjay Sharma
// Description: Baseline feature extractor that uses a 7x7 square from the
//              center of the image as the feature vector. This is the simplest
//              feature extraction method for Task 1 of the CBIR system.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef BASELINE_FEATURE_H
#define BASELINE_FEATURE_H

#include "FeatureExtractor.h"

namespace cbir {

/**
 * @class BaselineFeature
 * @brief Extracts a 7x7 pixel square from the center of an image
 * 
 * This is the baseline feature extraction method that uses the 7x7 pixel
 * region at the center of an image as the feature vector. The resulting
 * feature vector has dimension 49 (7x7) for grayscale images or 147 (7x7x3)
 * for color images.
 * 
 * The feature vector is created by reading the pixel values row by row,
 * flattened into a 1D vector. For color images, RGB values are interleaved.
 * 
 * Usage example:
 * @code
 *   BaselineFeature extractor;
 *   cv::Mat image = cv::imread("image.jpg");
 *   cv::Mat features = extractor.extractFeatures(image);
 * @endcode
 * 
 * @author Krushna Sanjay Sharma
 */
class BaselineFeature : public FeatureExtractor {
public:
    /**
     * @brief Default constructor
     * 
     * Creates a baseline feature extractor with default 7x7 square size.
     */
    BaselineFeature();

    /**
     * @brief Constructor with custom square size
     * 
     * @param squareSize Size of the square region to extract (default: 7)
     */
    explicit BaselineFeature(int squareSize);

    /**
     * @brief Destructor
     */
    virtual ~BaselineFeature() = default;

    /**
     * @brief Extract 7x7 center square as feature vector
     * 
     * Extracts a square region from the center of the image and flattens it
     * into a 1D feature vector. If the image is too small (< 7x7), returns
     * an empty cv::Mat.
     * 
     * @param image Input image (grayscale or color)
     * @return cv::Mat Feature vector as 1xN matrix (N=49 for grayscale, 
     *                 N=147 for color), type CV_32F
     * 
     * @note Returns empty Mat if image is too small or invalid
     */
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;

    /**
     * @brief Get the name of this feature extractor
     * 
     * @return std::string "Baseline7x7"
     */
    virtual std::string getFeatureName() const override;

    /**
     * @brief Get the dimension of the feature vector
     * 
     * @return int 49 for grayscale, 147 for color (7x7x3)
     */
    virtual int getFeatureDimension() const override;

    /**
     * @brief Set the size of the square region
     * 
     * @param size New square size (must be odd and positive)
     */
    void setSquareSize(int size);

    /**
     * @brief Get the current square size
     * 
     * @return int Current square size
     */
    int getSquareSize() const { return squareSize_; }

private:
    int squareSize_;  ///< Size of the square region to extract (default: 7)

    /**
     * @brief Extract square region from center of image
     * 
     * @param image Input image
     * @return cv::Mat Extracted square region
     */
    cv::Mat extractCenterSquare(const cv::Mat& image) const;
};

} // namespace cbir

#endif // BASELINE_FEATURE_H
