////////////////////////////////////////////////////////////////////////////////
// ImageRetrieval.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of the main CBIR engine that performs image
//              queries by comparing features and ranking results by similarity.
//
// This class orchestrates the query process:
//   1. Extracts features from query image
//   2. Compares query features to all database features
//   3. Computes distances using specified metric
//   4. Sorts results by distance (ascending)
//   5. Returns top N most similar images
//
// The class uses polymorphism to support different feature extractors
// and distance metrics without code changes.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "ImageRetrieval.h"
#include <iostream>
#include <algorithm>

namespace cbir {

/**
 * Constructor - Initialize with null pointers
 * Components must be set using setter methods before querying
 */
ImageRetrieval::ImageRetrieval() 
    : database_(nullptr), featureExtractor_(nullptr), distanceMetric_(nullptr) {
    // Initialize all components to null
    // User must call setters before performing queries
}

/**
 * Query database with an image
 * 
 * Main query interface - extracts features from query image and finds
 * top N most similar images from database.
 * 
 * Process:
 *   1. Validate system is ready (all components set)
 *   2. Extract features from query image
 *   3. Delegate to queryWithFeatures() for comparison
 * 
 * @param queryImage Input image to search for
 * @param topN Number of top matches to return
 * @return Vector of top N matches sorted by distance (ascending)
 */
std::vector<ImageMatch> ImageRetrieval::query(const cv::Mat& queryImage, int topN) {
    // Validate system is properly configured
    if (!isReady()) {
        std::cerr << "Error: ImageRetrieval system not properly configured" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    // Extract features from query image using configured extractor
    std::cout << "Extracting features from query image..." << std::endl;
    cv::Mat queryFeatures = featureExtractor_->extractFeatures(queryImage);
    
    if (queryFeatures.empty()) {
        std::cerr << "Error: Failed to extract features from query image" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    // Delegate to feature-based query method
    return queryWithFeatures(queryFeatures, topN);
}

/**
 * Query database with pre-computed features
 * 
 * Useful when features are already extracted or loaded from file.
 * Performs the actual comparison against database.
 * 
 * Process:
 *   1. Validate system is ready
 *   2. Compute distance from query to all database images
 *   3. Sort by distance (ascending - lower is more similar)
 *   4. Return top N matches
 * 
 * @param queryFeatures Pre-computed feature vector
 * @param topN Number of top matches to return
 * @return Vector of top N matches sorted by distance (ascending)
 */
std::vector<ImageMatch> ImageRetrieval::queryWithFeatures(const cv::Mat& queryFeatures, 
                                                          int topN) {
    // Validate system is properly configured
    if (!isReady()) {
        std::cerr << "Error: ImageRetrieval system not properly configured" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    std::cout << "Computing distances to all database images..." << std::endl;
    
    // Compute distance to every image in database
    std::vector<ImageMatch> allMatches = computeAllDistances(queryFeatures);
    
    if (allMatches.empty()) {
        std::cerr << "Error: No matches found" << std::endl;
        return std::vector<ImageMatch>();
    }
    
    // Sort matches by distance (ascending: 0.0 = identical, higher = less similar)
    std::sort(allMatches.begin(), allMatches.end());
    
    // Extract top N matches
    int numToReturn = std::min(topN, static_cast<int>(allMatches.size()));
    std::vector<ImageMatch> topMatches(allMatches.begin(), 
                                       allMatches.begin() + numToReturn);
    
    std::cout << "Found top " << numToReturn << " matches" << std::endl;
    
    return topMatches;
}

/**
 * Set the feature database to query against
 * 
 * @param database Pointer to loaded feature database
 */
void ImageRetrieval::setFeatureDatabase(FeatureDatabase* database) {
    database_ = database;
}

/**
 * Set the feature extractor to use for query images
 * 
 * Takes ownership via smart pointer to manage lifetime.
 * 
 * @param extractor Pointer to feature extractor
 */
void ImageRetrieval::setFeatureExtractor(FeatureExtractor* extractor) {
    featureExtractor_ = FeatureExtractorPtr(extractor);
}

/**
 * Set the distance metric to use for comparisons
 * 
 * Takes ownership via smart pointer to manage lifetime.
 * 
 * @param metric Pointer to distance metric
 */
void ImageRetrieval::setDistanceMetric(DistanceMetric* metric) {
    distanceMetric_ = DistanceMetricPtr(metric);
}

/**
 * Get name of currently configured feature extractor
 * 
 * @return Feature extractor name or "None" if not set
 */
std::string ImageRetrieval::getFeatureExtractorName() const {
    if (featureExtractor_) {
        return featureExtractor_->getFeatureName();
    }
    return "None";
}

/**
 * Get name of currently configured distance metric
 * 
 * @return Distance metric name or "None" if not set
 */
std::string ImageRetrieval::getDistanceMetricName() const {
    if (distanceMetric_) {
        return distanceMetric_->getMetricName();
    }
    return "None";
}

/**
 * Check if system is ready to perform queries
 * 
 * Validates that all required components are set:
 *   - Feature database is set and not empty
 *   - Feature extractor is set
 *   - Distance metric is set
 * 
 * @return True if ready, false otherwise
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
 * Compute distances from query features to all database images
 * 
 * Iterates through entire database, computing distance between
 * query features and each database image's features.
 * 
 * Process:
 *   1. Get list of all image names in database
 *   2. For each image:
 *      - Retrieve its pre-computed features
 *      - Compute distance to query features
 *      - Store in results vector
 *   3. Return unsorted results
 * 
 * @param queryFeatures Query feature vector
 * @return Vector of all matches with computed distances (unsorted)
 */
std::vector<ImageMatch> ImageRetrieval::computeAllDistances(const cv::Mat& queryFeatures) {
    std::vector<ImageMatch> matches;
    
    // Get all image names from database
    std::vector<std::string> imageNames = database_->getImageNames();
    
    std::cout << "Comparing against " << imageNames.size() << " database images..." 
              << std::endl;
    
    // Compute distance to each database image
    for (const auto& imageName : imageNames) {
        // Retrieve pre-computed features for this database image
        cv::Mat dbFeatures = database_->getFeatures(imageName);
        
        if (dbFeatures.empty()) {
            std::cerr << "Warning: No features found for " << imageName << std::endl;
            continue;
        }
        
        // Compute distance using configured metric
        // Lower distance = more similar
        double distance = distanceMetric_->compute(queryFeatures, dbFeatures);
        
        if (distance < 0) {
            // Negative distance indicates error in metric computation
            std::cerr << "Warning: Invalid distance for " << imageName << std::endl;
            continue;
        }
        
        // Create match record and add to results
        ImageMatch match(imageName, distance);
        matches.push_back(match);
    }
    
    return matches;
}

} // namespace cbir
