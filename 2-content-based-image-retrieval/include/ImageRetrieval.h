////////////////////////////////////////////////////////////////////////////////
// ImageRetrieval.h
// Author: Krushna Sanjay Sharma
// Description: Main CBIR engine that performs image queries against a feature
//              database using specified feature extractors and distance metrics.
//              Returns top N most similar images to a query.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_RETRIEVAL_H
#define IMAGE_RETRIEVAL_H

#include "FeatureExtractor.h"
#include "DistanceMetric.h"
#include "FeatureDatabase.h"
#include "Utils.h"
#include <vector>
#include <string>

namespace cbir {

/**
 * @class ImageRetrieval
 * @brief Main content-based image retrieval engine
 * 
 * This class implements the core CBIR functionality:
 * 1. Extract features from query image
 * 2. Compare against all images in database
 * 3. Rank results by similarity
 * 4. Return top N matches
 * 
 * Usage example:
 * @code
 *   // Setup
 *   ImageRetrieval retrieval;
 *   retrieval.setFeatureDatabase(database);
 *   retrieval.setFeatureExtractor(new BaselineFeature());
 *   retrieval.setDistanceMetric(new SSDMetric());
 *   
 *   // Query
 *   cv::Mat queryImage = cv::imread("query.jpg");
 *   std::vector<ImageMatch> results = retrieval.query(queryImage, 5);
 *   
 *   // Display results
 *   for (const auto& match : results) {
 *       std::cout << match.filename << " : " << match.distance << std::endl;
 *   }
 * @endcode
 * 
 * @author Krushna Sanjay Sharma
 */
class ImageRetrieval {
public:
    /**
     * @brief Default constructor
     */
    ImageRetrieval();

    /**
     * @brief Destructor
     */
    ~ImageRetrieval() = default;

    /**
     * @brief Query database with an image
     * 
     * Extracts features from the query image, compares against all images
     * in the database, and returns the top N most similar matches.
     * 
     * @param queryImage Query image
     * @param topN Number of top matches to return
     * @return std::vector<ImageMatch> Top N matches sorted by distance (ascending)
     */
    std::vector<ImageMatch> query(const cv::Mat& queryImage, int topN);

    /**
     * @brief Query database with pre-computed features
     * 
     * Useful when features are already extracted or loaded from file.
     * 
     * @param queryFeatures Pre-computed feature vector
     * @param topN Number of top matches to return
     * @return std::vector<ImageMatch> Top N matches sorted by distance (ascending)
     */
    std::vector<ImageMatch> queryWithFeatures(const cv::Mat& queryFeatures, 
                                              int topN);

    /**
     * @brief Set the feature database to query against
     * 
     * @param database Pointer to feature database
     */
    void setFeatureDatabase(FeatureDatabase* database);

    /**
     * @brief Set the feature extractor to use
     * 
     * @param extractor Pointer to feature extractor (takes ownership)
     */
    void setFeatureExtractor(FeatureExtractor* extractor);

    /**
     * @brief Set the distance metric to use
     * 
     * @param metric Pointer to distance metric (takes ownership)
     */
    void setDistanceMetric(DistanceMetric* metric);

    /**
     * @brief Get current feature extractor name
     * 
     * @return std::string Name of current feature extractor
     */
    std::string getFeatureExtractorName() const;

    /**
     * @brief Get current distance metric name
     * 
     * @return std::string Name of current distance metric
     */
    std::string getDistanceMetricName() const;

private:
    FeatureDatabase* database_;              ///< Pointer to feature database
    FeatureExtractorPtr featureExtractor_;   ///< Feature extraction method
    DistanceMetricPtr distanceMetric_;       ///< Distance computation method

    /**
     * @brief Validate that all required components are set
     * 
     * @return bool True if ready for querying
     */
    bool isReady() const;

    /**
     * @brief Compute distances from query features to all database images
     * 
     * @param queryFeatures Query feature vector
     * @return std::vector<ImageMatch> All matches with computed distances
     */
    std::vector<ImageMatch> computeAllDistances(const cv::Mat& queryFeatures);
};

} // namespace cbir

#endif // IMAGE_RETRIEVAL_H
