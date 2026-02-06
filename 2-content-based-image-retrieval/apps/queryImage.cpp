////////////////////////////////////////////////////////////////////////////////
// queryImage.cpp
// Author: Krushna Sanjay Sharma
// Description: Application to query the CBIR system with a target image.
//              Loads pre-computed features and returns the top N most similar
//              images from the database.
//
// Usage: queryImage <target_image> <feature_csv> <feature_type> <metric> <topN>
//
// Example: queryImage data/images/pic.1016.jpg baseline_features.csv baseline ssd 3
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "ImageRetrieval.h"
#include "BaselineFeature.h"
#include "SSDMetric.h"
#include <iostream>
#include <iomanip>
#include <string>

/**
 * @brief Print usage information
 * 
 * @author Krushna Sanjay Sharma
 */
void printUsage(const char* programName) {
    std::cout << "========================================" << std::endl;
    std::cout << "CBIR Image Query System" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << programName 
              << " <target_image> <feature_csv> <feature_type> <metric> <topN>" 
              << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  target_image : Path to query image" << std::endl;
    std::cout << "  feature_csv  : CSV file with pre-computed features" << std::endl;
    std::cout << "  feature_type : Type of features (must match CSV)" << std::endl;
    std::cout << "                 Options: baseline" << std::endl;
    std::cout << "  metric       : Distance metric to use" << std::endl;
    std::cout << "                 Options: ssd" << std::endl;
    std::cout << "  topN         : Number of top matches to return" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << programName 
              << " data/images/pic.1016.jpg baseline_features.csv baseline ssd 3" 
              << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Create feature extractor based on type
 * 
 * @author Krushna Sanjay Sharma
 */
cbir::FeatureExtractor* createFeatureExtractor(const std::string& featureType) {
    std::string type = cbir::Utils::toLower(featureType);
    
    if (type == "baseline") {
        return new cbir::BaselineFeature();
    }
    
    std::cerr << "Error: Unknown feature type '" << featureType << "'" << std::endl;
    return nullptr;
}

/**
 * @brief Create distance metric based on type
 * 
 * @author Krushna Sanjay Sharma
 */
cbir::DistanceMetric* createDistanceMetric(const std::string& metricType) {
    std::string type = cbir::Utils::toLower(metricType);
    
    if (type == "ssd") {
        return new cbir::SSDMetric();
    }
    
    std::cerr << "Error: Unknown metric type '" << metricType << "'" << std::endl;
    return nullptr;
}

/**
 * @brief Display query results
 * 
 * @author Krushna Sanjay Sharma
 */
void displayResults(const std::string& queryImage,
                   const std::vector<cbir::ImageMatch>& results) {
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Query Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Query image: " << queryImage << std::endl;
    std::cout << std::endl;
    std::cout << std::left << std::setw(5) << "Rank" 
              << std::setw(30) << "Filename" 
              << std::setw(15) << "Distance" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    for (size_t i = 0; i < results.size(); i++) {
        std::cout << std::left << std::setw(5) << (i + 1)
                  << std::setw(30) << results[i].filename
                  << std::fixed << std::setprecision(4) << results[i].distance 
                  << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Main function
 * 
 * @author Krushna Sanjay Sharma
 */
int main(int argc, char* argv[]) {
    // Check arguments
    if (argc != 6) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse arguments
    std::string targetImage = argv[1];
    std::string featureCSV = argv[2];
    std::string featureType = argv[3];
    std::string metricType = argv[4];
    int topN = std::stoi(argv[5]);
    
    std::cout << "========================================" << std::endl;
    std::cout << "CBIR Query System" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Target image    : " << targetImage << std::endl;
    std::cout << "Feature CSV     : " << featureCSV << std::endl;
    std::cout << "Feature type    : " << featureType << std::endl;
    std::cout << "Distance metric : " << metricType << std::endl;
    std::cout << "Top N matches   : " << topN << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Check if target image exists
    if (!cbir::Utils::fileExists(targetImage)) {
        std::cerr << "Error: Target image does not exist: " << targetImage << std::endl;
        return 1;
    }
    
    // Check if feature CSV exists
    if (!cbir::Utils::fileExists(featureCSV)) {
        std::cerr << "Error: Feature CSV does not exist: " << featureCSV << std::endl;
        return 1;
    }
    
    // Load target image
    std::cout << "Loading target image..." << std::endl;
    cv::Mat queryImage = cbir::Utils::loadImage(targetImage);
    if (queryImage.empty()) {
        std::cerr << "Error: Failed to load target image" << std::endl;
        return 1;
    }
    
    // Create feature extractor
    cbir::FeatureExtractor* extractor = createFeatureExtractor(featureType);
    if (extractor == nullptr) {
        return 1;
    }
    
    // Create distance metric
    cbir::DistanceMetric* metric = createDistanceMetric(metricType);
    if (metric == nullptr) {
        delete extractor;
        return 1;
    }
    
    // Load feature database
    std::cout << "Loading feature database..." << std::endl;
    cbir::FeatureDatabase database;
    if (!database.loadFromCSV(featureCSV)) {
        std::cerr << "Error: Failed to load feature database" << std::endl;
        delete extractor;
        delete metric;
        return 1;
    }
    
    // Setup retrieval system
    std::cout << "Setting up retrieval system..." << std::endl;
    cbir::ImageRetrieval retrieval;
    retrieval.setFeatureDatabase(&database);
    retrieval.setFeatureExtractor(extractor);
    retrieval.setDistanceMetric(metric);
    
    // Perform query
    std::cout << std::endl;
    std::cout << "Querying database..." << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    std::vector<cbir::ImageMatch> results = retrieval.query(queryImage, topN);
    
    if (results.empty()) {
        std::cerr << "Error: No results found" << std::endl;
        return 1;
    }
    
    // Display results
    displayResults(targetImage, results);
    
    return 0;
}
