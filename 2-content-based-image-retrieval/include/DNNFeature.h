////////////////////////////////////////////////////////////////////////////////
// DNNFeature.h
// Author: Krushna Sanjay Sharma
// Description: Deep Neural Network feature extractor for Task 5. Loads
//              pre-computed ResNet18 embeddings from CSV file instead of
//              computing features from images.
//
// ResNet18 Embeddings:
//   - Pre-trained on ImageNet (1M images, 1k categories)
//   - 512-dimensional feature vectors
//   - Output of final global average pooling layer
//   - Captures high-level semantic image content
//
// Key Difference from Other Features:
//   - Does NOT compute features from images
//   - Loads pre-computed features from CSV file
//   - extractFeatures() looks up features by filename
//
// CSV Format:
//   filename,feature1,feature2,...,feature512
//   pic.0001.jpg,0.123,0.456,...,0.789
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef DNN_FEATURE_H
#define DNN_FEATURE_H

#include "FeatureExtractor.h"
#include <map>
#include <string>

namespace cbir {

/**
 * @class DNNFeature
 * @brief Loads pre-computed deep learning features from CSV
 * 
 * Unlike other feature extractors that compute features from images,
 * this class loads pre-computed ResNet18 embeddings from a CSV file.
 * 
 * The CSV must be loaded before extractFeatures() is called.
 * extractFeatures() then looks up features by filename.
 * 
 * Usage:
 * @code
 *   DNNFeature extractor;
 *   extractor.loadFeaturesFromCSV("ResNet18_olym.csv");
 *   
 *   // Extract by filename (not by cv::Mat image!)
 *   cv::Mat features = extractor.getFeaturesByFilename("pic.0001.jpg");
 * @endcode
 * 
 * @author Krushna Sanjay Sharma
 */
class DNNFeature : public FeatureExtractor {
public:
    /**
     * Constructor
     * 
     * @param csvPath Optional path to DNN features CSV file
     */
    DNNFeature(const std::string& csvPath = "");
    
    virtual ~DNNFeature() = default;
    
    /**
     * Extract features from image
     * 
     * NOTE: For DNN features, this attempts to find features by
     * extracting filename from image path. Not recommended.
     * Use getFeaturesByFilename() instead.
     * 
     * @param image Not used (features are pre-computed)
     * @return Empty Mat (use getFeaturesByFilename instead)
     */
    virtual cv::Mat extractFeatures(const cv::Mat& image) override;
    
    /**
     * Get features by filename (recommended for DNN features)
     * 
     * @param filename Image filename (e.g., "pic.0001.jpg")
     * @return 512-dimensional feature vector
     */
    cv::Mat getFeaturesByFilename(const std::string& filename) const;
    
    virtual std::string getFeatureName() const override;
    virtual int getFeatureDimension() const override;
    
    /**
     * Load DNN features from CSV file
     * 
     * CSV format: filename,f1,f2,...,f512
     * 
     * @param csvPath Path to CSV file
     * @return True if successful
     */
    bool loadFeaturesFromCSV(const std::string& csvPath);
    
    /**
     * Check if features are loaded
     */
    bool isFeaturesLoaded() const { return !features_.empty(); }
    
    /**
     * Get number of loaded features
     */
    size_t getNumFeatures() const { return features_.size(); }

private:
    std::string csvPath_;                        ///< Path to DNN features CSV
    std::map<std::string, cv::Mat> features_;    ///< Map: filename → features
    
    /**
     * Normalize filename for consistent lookup
     */
    std::string normalizeFilename(const std::string& filename) const;
};

} // namespace cbir

#endif // DNN_FEATURE_H
