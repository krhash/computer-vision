////////////////////////////////////////////////////////////////////////////////
// FaceAwareFeature.h
// Author: Krushna Sanjay Sharma
// Description: Adaptive feature extractor that automatically selects features
//              based on image content (Extension for Task 7).
//
// Design Rationale:
//   Different image types need different features:
//   - Photos with people → Face-based features
//   - Product/object photos → ProductMatcher features
//
// Algorithm:
//   1. Detect faces using Haar cascade
//   2. If faces found:
//      - Extract face-specific features (count, positions, colors)
//      - Combine with DNN embeddings
//   3. If no faces:
//      - Use ProductMatcher (DNN + center-color)
//
// Feature Structure (Face Mode):
//   [dnn(512), face_count(1), face_region_color(512), spatial_layout(4)] = 1029 values
//
// Feature Structure (No Face Mode):
//   [dnn(512), center_color(512)] = 1024 values
//   (Same as ProductMatcher)
//
// Use Cases:
//   - Photo album organization (people photos)
//   - Social media image search
//   - Automatic feature selection
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef FACE_AWARE_FEATURE_H
#define FACE_AWARE_FEATURE_H

#include "FeatureExtractor.h"
#include "DNNFeature.h"
#include "ProductMatcherFeature.h"
#include <opencv2/objdetect.hpp>

namespace cbir {

/**
 * @class FaceAwareFeature
 * @brief Adaptive feature that changes based on face detection
 * 
 * This extension demonstrates intelligent, content-aware feature selection.
 * The feature extraction strategy adapts based on whether the image
 * contains faces (people photos) or not (object photos).
 * 
 * Innovation:
 *   - Automatic feature selection (no manual user input)
 *   - Optimized features for each image type
 *   - Combines multiple detection and extraction strategies
 * 
 * For images WITH faces:
 *   - Prioritizes face-based similarity
 *   - Captures: number of people, face positions, clothing colors
 *   
 * For images WITHOUT faces:
 *   - Falls back to ProductMatcher
 *   - Focuses on object matching
 * 
 * @author Krushna Sanjay Sharma
 */
class FaceAwareFeature : public FeatureExtractor {
public:
    /**
     * Constructor
     * 
     * @param dnnCsvPath Path to pre-computed DNN features
     * @param cascadePath Path to Haar cascade XML file for face detection
     */
    FaceAwareFeature(const std::string& dnnCsvPath,
                    const std::string& cascadePath = "haarcascade_frontalface_default.xml");
    
    virtual ~FaceAwareFeature() = default;
    
    /**
     * Extract features - requires filename for DNN lookup
     */
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    
    /**
     * Extract adaptive features with filename
     * 
     * @param image Input image
     * @param filename Image filename for DNN lookup
     * @return Feature vector (size depends on face detection)
     */
    cv::Mat extractFeaturesWithFilename(const cv::Mat& image, 
                                       const std::string& filename);
    
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    /**
     * Check if last extraction used face features
     */
    bool lastImageHadFaces() const { return lastHadFaces_; }
    
    /**
     * Get number of faces detected in last image
     */
    int getLastFaceCount() const { return lastFaceCount_; }

private:
    DNNFeature dnnExtractor_;              ///< DNN feature loader
    ProductMatcherFeature productMatcher_; ///< Fallback for non-face images
    cv::CascadeClassifier faceCascade_;   ///< Haar cascade face detector
    
    bool lastHadFaces_;   ///< Flag: did last image have faces
    int lastFaceCount_;   ///< Number of faces in last image
    
    /**
     * Detect faces in image
     * 
     * @param image Input image
     * @return Vector of face bounding boxes
     */
    std::vector<cv::Rect> detectFaces(const cv::Mat& image);
    
    /**
     * Extract face-based features when faces are present
     * 
     * @param image Input image
     * @param faces Detected face bounding boxes
     * @param filename Filename for DNN lookup
     * @return Face-aware feature vector
     */
    cv::Mat extractFaceFeatures(const cv::Mat& image, 
                               const std::vector<cv::Rect>& faces,
                               const std::string& filename);
    
    /**
     * Compute color histogram of face regions
     * 
     * @param image Input image
     * @param faces Face bounding boxes
     * @return RGB histogram of face regions (clothing/context)
     */
    cv::Mat computeFaceRegionColor(const cv::Mat& image, 
                                  const std::vector<cv::Rect>& faces);
    
    /**
     * Compute spatial layout of faces
     * 
     * Returns 4D vector: [avg_x, avg_y, spread_x, spread_y]
     * Captures where faces are and how spread out they are
     * 
     * @param faces Face bounding boxes
     * @param imageWidth Image width for normalization
     * @param imageHeight Image height for normalization
     * @return 4D spatial layout vector
     */
    cv::Mat computeFaceSpatialLayout(const std::vector<cv::Rect>& faces,
                                    int imageWidth, int imageHeight);
};

} // namespace cbir

#endif // FACE_AWARE_FEATURE_H
