////////////////////////////////////////////////////////////////////////////////
// buildFeatureDB.cpp
// Author: Krushna Sanjay Sharma
// Description: Application to build and save feature database for CBIR system.
//              Pre-computes features for all images in a directory and saves
//              them to a CSV file for fast querying.
//
// Usage: buildFeatureDB <image_dir> <feature_type> <output_csv>
// Example: buildFeatureDB data/images histogram histogram_features.csv
//
// Workflow:
//   1. Scan image directory for all image files
//   2. Extract features from each image using specified feature type
//   3. Store features in memory (FeatureDatabase map)
//   4. Save all features to CSV file for persistence
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "FeatureDatabase.h"
#include "BaselineFeature.h"
#include "HistogramFeature.h"
#include <iostream>
#include <string>

using namespace std;
using namespace cbir;

/**
 * Print usage information and examples
 */
void printUsage(const char* programName) {
    cout << "========================================" << endl;
    cout << "CBIR Feature Database Builder" << endl;
    cout << "========================================" << endl;
    cout << endl;
    cout << "Usage: " << programName << " <image_dir> <feature_type> <output_csv>" << endl;
    cout << endl;
    cout << "Arguments:" << endl;
    cout << "  image_dir    : Directory containing images" << endl;
    cout << "  feature_type : Type of features to extract" << endl;
    cout << "                 Options: baseline, histogram, chromaticity" << endl;
    cout << "  output_csv   : Output CSV file for features" << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  " << programName << " data/images baseline baseline_features.csv" << endl;
    cout << "  " << programName << " data/images histogram histogram_features.csv" << endl;
    cout << endl;
}

/**
 * Factory function to create appropriate feature extractor based on type
 * 
 * Supports:
 *   - "baseline": 7x7 center square feature (49 or 147 values)
 *   - "histogram" or "rgb": RGB histogram (8 bins per channel = 512 values)
 *   - "chromaticity" or "rg": RG chromaticity histogram (16x16 = 256 values)
 * 
 * @param featureType Type of feature extractor to create
 * @return Pointer to created feature extractor, nullptr if type unknown
 */
cbir::FeatureExtractor* createFeatureExtractor(const string& featureType) {
    // Convert to lowercase for case-insensitive comparison
    string type = cbir::Utils::toLower(featureType);
    
    if (type == "baseline") {
        // Baseline: Extract 7x7 pixel square from center of image
        return new cbir::BaselineFeature();
    } else if (type == "histogram" || type == "rgb") {
        // RGB Histogram: 8 bins per channel (8x8x8 = 512 bins), normalized
        return new cbir::HistogramFeature(
            cbir::HistogramFeature::HistogramType::RGB, 
            8,      // bins per channel
            true    // normalize to sum=1.0
        );
    } else if (type == "chromaticity" || type == "rg") {
        // RG Chromaticity: 16 bins per channel (16x16 = 256 bins), normalized
        // More robust to lighting changes
        return new cbir::HistogramFeature(
            cbir::HistogramFeature::HistogramType::RG_CHROMATICITY, 
            16,     // bins per channel
            true    // normalize to sum=1.0
        );
    }
    
    // Unknown feature type
    cerr << "Error: Unknown feature type '" << featureType << "'" << endl;
    cerr << "Available types: baseline, histogram, chromaticity" << endl;
    return nullptr;
}

/**
 * Main function - Build feature database from image directory
 * 
 * Process:
 *   1. Validate command line arguments
 *   2. Check if image directory exists
 *   3. Create appropriate feature extractor
 *   4. Build feature database (extract features from all images)
 *   5. Save features to CSV file
 * 
 * @param argc Argument count
 * @param argv Argument values
 * @return 0 on success, 1 on error
 */
int main(int argc, char* argv[]) {
    // Validate command line arguments
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    string imageDir = argv[1];      // Directory containing images
    string featureType = argv[2];   // Type of features to extract
    string outputCSV = argv[3];     // Output CSV file path
    
    // Display configuration
    cout << "========================================" << endl;
    cout << "Building Feature Database" << endl;
    cout << "========================================" << endl;
    cout << "Image directory : " << imageDir << endl;
    cout << "Feature type    : " << featureType << endl;
    cout << "Output CSV      : " << outputCSV << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Validate image directory exists
    if (!Utils::directoryExists(imageDir)) {
        cerr << "Error: Image directory does not exist: " << imageDir << endl;
        return 1;
    }
    
    // Create feature extractor for specified type
    FeatureExtractor* extractor = createFeatureExtractor(featureType);
    if (extractor == nullptr) {
        return 1;  // Error message already printed by createFeatureExtractor
    }
    
    // Create feature database (in-memory map)
    FeatureDatabase database;
    
    // Step 1: Extract features from all images in directory
    cout << "Step 1: Extracting features from images..." << endl;
    cout << "-------------------------------------------" << endl;
    
    bool success = database.buildDatabase(imageDir, extractor, false);
    
    if (!success) {
        cerr << "Error: Failed to build feature database" << endl;
        delete extractor;
        return 1;
    }
    
    cout << endl;
    
    // Step 2: Save features to CSV file for persistence
    cout << "Step 2: Saving features to CSV..." << endl;
    cout << "-------------------------------------------" << endl;
    
    success = database.saveToCSV(outputCSV);
    
    if (!success) {
        cerr << "Error: Failed to save features to CSV" << endl;
        delete extractor;
        return 1;
    }
    
    // Success! Display summary
    cout << endl;
    cout << "========================================" << endl;
    cout << "SUCCESS!" << endl;
    cout << "========================================" << endl;
    cout << "Feature database saved to: " << outputCSV << endl;
    cout << "Total images processed: " << database.size() << endl;
    cout << endl;
    cout << "Next step: Query images using queryImage" << endl;
    cout << "========================================" << endl;
    
    // Cleanup
    delete extractor;
    
    return 0;
}
