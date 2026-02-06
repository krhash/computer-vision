////////////////////////////////////////////////////////////////////////////////
// ImageRetrieval.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of the main CBIR engine that performs image
//              queries by comparing features and ranking results by similarity.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "ImageRetrieval.h"
#include <iostream>
#include <algorithm>

namespace cbir {

/**
 * @brief Constructor
 * 
 * @author Krushna Sanjay Sharma
 */
ImageRetrieval::ImageRetrieval() 
    : database_(nullptr), featureExtractor_(nullptr), distanceMetric_(nullptr) {
    // Initialize with null pointers
}

/**
 * @brief Query database with an image
 * 
 * @author Krushna Sanjay Sharma
 */
std::vector<ImageMatch> ImageRetrieval::query(const cv::Mat& queryImage, int topN) {
    // Check if system is ready
    if (!isReady()) {
        std::cerr << "Error: ImageRetrieval system not properly configured" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    // Extract features from query image
    std::cout << "Extracting features from query image..." << std::endl;
    cv::Mat queryFeatures = featureExtractor_->extractFeatures(queryImage);
    
    if (queryFeatures.empty()) {
        std::cerr << "Error: Failed to extract features from query image" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    // Query with extracted features
    return queryWithFeatures(queryFeatures, topN);
}

/**
 * @brief Query database with pre-computed features
 * 
 * @author Krushna Sanjay Sharma
 */
std::vector<ImageMatch> ImageRetrieval::queryWithFeatures(const cv::Mat& queryFeatures, 
                                                          int topN) {
    // Check if system is ready
    if (!isReady()) {
        std::cerr << "Error: ImageRetrieval system not properly configured" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    std::cout << "Computing distances to all database images..." << std::endl;
    
    // Compute distances to all images
    std::vector<ImageMatch> allMatches = computeAllDistances(queryFeatures);
    
    if (allMatches.empty()) {
        std::cerr << "Error: No matches found" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    // Sort by distance (ascending - lower distance = more similar)
    std::sort(allMatches.begin(), allMatches.end());
    
    // Return top N matches
    int numToReturn = std::min(topN, static_cast<int>(allMatches.size()));
    std::vector<ImageMatch> topMatches(allMatches.begin(), 
                                       allMatches.begin() + numToReturn);
    
    std::cout << "Found top " << numToReturn << " matches" << std::endl;
    
    return topMatches;
}

/**
 * @brief Set feature database
 * 
 * @author Krushna Sanjay Sharma
 */
void ImageRetrieval::setFeatureDatabase(FeatureDatabase* database) {
    database_ = database;
}

/**
 * @brief Set feature extractor
 * 
 * @author Krushna Sanjay Sharma
 */
void ImageRetrieval::setFeatureExtractor(FeatureExtractor* extractor) {
    featureExtractor_ = FeatureExtractorPtr(extractor);
}

/**
 * @brief Set distance metric
 * 
 * @author Krushna Sanjay Sharma
 */
void ImageRetrieval::setDistanceMetric(DistanceMetric* metric) {
    distanceMetric_ = DistanceMetricPtr(metric);
}

/**
 * @brief Get feature extractor name
 * 
 * @author Krushna Sanjay Sharma
 */
std::string ImageRetrieval::getFeatureExtractorName() const {
    if (featureExtractor_) {
        return featureExtractor_->getFeatureName();
    }
    return "None";
}

/**
 * @brief Get distance metric name
 * 
 * @author Krushna Sanjay Sharma
 */
std::string ImageRetrieval::getDistanceMetricName() const {
    if (distanceMetric_) {
        return distanceMetric_->getMetricName();
    }
    return "None";
}

/**
 * @brief Check if system is ready for queries
 * 
 * @author Krushna Sanjay Sharma
 */
bool ImageRetrieval::isReady() const {
    if (database_ == nullptr) {
        std::cerr << "Error: Feature database not set" << std::endl;
        return false;
    }
    
    if (featureExtractor_ == nullptr) {
        std::cerr << "Error: Feature extractor not set" << std::endl;
        return false;
    }
    
    if (distanceMetric_ == nullptr) {
        std::cerr << "Error: Distance metric not set" << std::endl;
        return false;
    }
    
    if (database_->empty()) {
        std::cerr << "Error: Feature database is empty" << std::endl;
        return false;
    }
    
    return true;
}

/**
 * @brief Compute distances from query to all database images
 * 
 * @author Krushna Sanjay Sharma
 */
std::vector<ImageMatch> ImageRetrieval::computeAllDistances(const cv::Mat& queryFeatures) {
    std::vector<ImageMatch> matches;
    
    // Get all image names from database
    std::vector<std::string> imageNames = database_->getImageNames();
    
    std::cout << "Comparing against " << imageNames.size() << " database images..." 
              << std::endl;
    
    // Compute distance to each database image
    for (const auto& imageName : imageNames) {
        // Get features for this database image
        cv::Mat dbFeatures = database_->getFeatures(imageName);
        
        if (dbFeatures.empty()) {
            std::cerr << "Warning: No features found for " << imageName << std::endl;
            continue;
        }
        
        // Compute distance
        double distance = distanceMetric_->compute(queryFeatures, dbFeatures);
        
        if (distance < 0) {
            std::cerr << "Warning: Invalid distance for " << imageName << std::endl;
            continue;
        }
        
        // Create match and add to results
        ImageMatch match(imageName, distance);
        matches.push_back(match);
    }
    
    return matches;
}

} // namespace cbir
