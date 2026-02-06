////////////////////////////////////////////////////////////////////////////////
// buildFeatureDB.cpp
// Author: Krushna Sanjay Sharma
// Description: Application to build and save feature database for CBIR system.
//              Pre-computes features for all images in a directory and saves
//              them to a CSV file for fast querying.
//
// Usage: buildFeatureDB <image_dir> <feature_type> <output_csv>
//
// Example: buildFeatureDB data/images baseline baseline_features.csv
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "FeatureDatabase.h"
#include "BaselineFeature.h"
#include <iostream>
#include <string>

/**
 * @brief Print usage information
 * 
 * @author Krushna Sanjay Sharma
 */
void printUsage(const char* programName) {
    std::cout << "========================================" << std::endl;
    std::cout << "CBIR Feature Database Builder" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << programName << " <image_dir> <feature_type> <output_csv>" 
              << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  image_dir    : Directory containing images" << std::endl;
    std::cout << "  feature_type : Type of features to extract" << std::endl;
    std::cout << "                 Options: baseline" << std::endl;
    std::cout << "  output_csv   : Output CSV file for features" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << programName << " data/images baseline baseline_features.csv" 
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
    std::cerr << "Available types: baseline" << std::endl;
    return nullptr;
}

/**
 * @brief Main function
 * 
 * @author Krushna Sanjay Sharma
 */
int main(int argc, char* argv[]) {
    // Check arguments
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse arguments
    std::string imageDir = argv[1];
    std::string featureType = argv[2];
    std::string outputCSV = argv[3];
    
    std::cout << "========================================" << std::endl;
    std::cout << "Building Feature Database" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Image directory : " << imageDir << std::endl;
    std::cout << "Feature type    : " << featureType << std::endl;
    std::cout << "Output CSV      : " << outputCSV << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Check if image directory exists
    if (!cbir::Utils::directoryExists(imageDir)) {
        std::cerr << "Error: Image directory does not exist: " << imageDir << std::endl;
        return 1;
    }
    
    // Create feature extractor
    cbir::FeatureExtractor* extractor = createFeatureExtractor(featureType);
    if (extractor == nullptr) {
        return 1;
    }
    
    // Create feature database
    cbir::FeatureDatabase database;
    
    // Build database
    std::cout << "Step 1: Extracting features from images..." << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    bool success = database.buildDatabase(imageDir, extractor, false);
    
    if (!success) {
        std::cerr << "Error: Failed to build feature database" << std::endl;
        delete extractor;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "Step 2: Saving features to CSV..." << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    // Save to CSV
    success = database.saveToCSV(outputCSV);
    
    if (!success) {
        std::cerr << "Error: Failed to save features to CSV" << std::endl;
        delete extractor;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "SUCCESS!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Feature database saved to: " << outputCSV << std::endl;
    std::cout << "Total images processed: " << database.size() << std::endl;
    std::cout << std::endl;
    std::cout << "Next step: Query images using queryImage" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Cleanup
    delete extractor;
    
    return 0;
}
