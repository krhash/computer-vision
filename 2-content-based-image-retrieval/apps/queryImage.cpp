////////////////////////////////////////////////////////////////////////////////
// queryImage.cpp
// Author: Krushna Sanjay Sharma
// Description: Application to query the CBIR system with a target image.
//              Loads pre-computed features and returns the top N most similar
//              images from the database based on specified distance metric.
//
// Usage: queryImage <target_image> <feature_csv> <feature_type> <metric> <topN>
// Example: queryImage data/images/pic.0164.jpg histogram_features.csv histogram histogram 3
//
// Workflow:
//   1. Load pre-computed features from CSV into memory
//   2. Extract features from query image
//   3. Compare query features to all database features
//   4. Sort by distance and return top N matches
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "ImageRetrieval.h"
#include "BaselineFeature.h"
#include "SSDMetric.h"
#include "HistogramFeature.h"
#include "HistogramIntersection.h"
#include "MultiHistogramFeature.h"
#include "TextureColorFeature.h"
#include "WeightedHistogramIntersection.h"
#include <iostream>
#include <iomanip>
#include <string>

using namespace std;
using namespace cbir;

/**
 * Print usage information and examples
 */
void printUsage(const char* programName) {
    cout << "========================================" << endl;
    cout << "CBIR Image Query System" << endl;
    cout << "========================================" << endl;
    cout << endl;
    cout << "Usage: " << programName << " <target_image> <feature_csv> <feature_type> <metric> <topN>" << endl;
    cout << endl;
    cout << "Arguments:" << endl;
    cout << "  target_image : Path to query image" << endl;
    cout << "  feature_csv  : CSV file with pre-computed features" << endl;
    cout << "  feature_type : Type of features (must match CSV)" << endl;
    cout << "                 Options: baseline, histogram, chromaticity" << endl;
    cout << "  metric       : Distance metric to use" << endl;
    cout << "                 Options: ssd, histogram" << endl;
    cout << "  topN         : Number of top matches to return" << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  " << programName << " data/images/pic.1016.jpg baseline_features.csv baseline ssd 3" << endl;
    cout << "  " << programName << " data/images/pic.0164.jpg histogram_features.csv histogram histogram 3" << endl;
    cout << endl;
}

/**
 * Factory function to create appropriate feature extractor based on type
 * 
 * Must match the feature type used when building the database!
 * 
 * Supports:
 *   - "baseline": 7x7 center square feature
 *   - "histogram" or "rgb": RGB histogram (8 bins per channel)
 *   - "chromaticity" or "rg": RG chromaticity histogram (16 bins per channel)
 * 
 * @param featureType Type of feature extractor to create
 * @return Pointer to created feature extractor, nullptr if type unknown
 */
cbir::FeatureExtractor* createFeatureExtractor(const string& featureType) {
    string type = cbir::Utils::toLower(featureType);
    
    if (type == "baseline") {
        return new cbir::BaselineFeature();
    } else if (type == "histogram" || type == "rgb") {
        return new cbir::HistogramFeature(
            cbir::HistogramFeature::HistogramType::RGB, 
            8, true
        );
    } else if (type == "chromaticity" || type == "rg") {
        return new cbir::HistogramFeature(
            cbir::HistogramFeature::HistogramType::RG_CHROMATICITY, 
            16, true
        );
    } else if (type == "multihistogram" || type == "multi") {
        // MUST match parameters used in buildFeatureDB!
        return new cbir::MultiHistogramFeature(
            cbir::MultiHistogramFeature::SplitType::GRID,
            4,
            cbir::HistogramFeature::HistogramType::RGB,
            8,
            true
        );
    } else if (type == "texturecolor" || type == "texture") {
        // Texture + Color: Sobel gradient histogram + RGB histogram
        // Texture: 16 bins (gradient magnitudes 0-255)
        // Color: 8 bins per channel (8×8×8 = 512 bins)
        // Total: 16 + 512 = 528 values
        return new cbir::TextureColorFeature(
            16,   // texture bins
            8,    // color bins per channel
            true  // normalize
        );
    }
    
    cerr << "Error: Unknown feature type '" << featureType << "'" << endl;
    return nullptr;
}

/**
 * Factory function to create appropriate distance metric based on type
 * 
 * Supports:
 *   - "ssd": Sum of Squared Differences (for baseline features)
 *   - "histogram" or "intersection": Histogram Intersection (for histogram features)
 * 
 * @param metricType Type of distance metric to create
 * @return Pointer to created distance metric, nullptr if type unknown
 */
cbir::DistanceMetric* createDistanceMetric(const string& metricType) {
    string type = cbir::Utils::toLower(metricType);
    
    if (type == "ssd") {
        return new cbir::SSDMetric();
    } else if (type == "histogram" || type == "intersection") {
        return new cbir::HistogramIntersection();
    } else if (type == "weighted" || type == "texturecolor") {
        // Weighted intersection: 50% texture, 50% color
        return new cbir::WeightedHistogramIntersection(
            16,   // texture dimension
            512,  // color dimension
            0.5,  // 50% weight for texture
            0.5   // 50% weight for color
        );
    }
    
    cerr << "Error: Unknown metric type '" << metricType << "'" << endl;
    cerr << "Available: ssd, histogram, weighted" << endl;
    return nullptr;
}

/**
 * Display query results in formatted table
 * 
 * @param queryImage Path to query image
 * @param results Vector of ImageMatch results sorted by distance
 */
void displayResults(const string& queryImage, const vector<ImageMatch>& results) {
    cout << endl;
    cout << "========================================" << endl;
    cout << "Query Results" << endl;
    cout << "========================================" << endl;
    cout << "Query image: " << queryImage << endl;
    cout << endl;
    
    // Print table header
    cout << left << setw(5) << "Rank" 
         << setw(30) << "Filename" 
         << setw(15) << "Distance" << endl;
    cout << "----------------------------------------" << endl;
    
    // Print each result
    for (size_t i = 0; i < results.size(); i++) {
        cout << left << setw(5) << (i + 1)
             << setw(30) << results[i].filename
             << fixed << setprecision(4) << results[i].distance << endl;
    }
    
    cout << "========================================" << endl;
    cout << endl;
}

/**
 * Main function - Query database with target image
 * 
 * Process:
 *   1. Validate command line arguments
 *   2. Load target image
 *   3. Create feature extractor and distance metric
 *   4. Load pre-computed features from CSV
 *   5. Extract features from query image
 *   6. Compare query to all database images
 *   7. Sort by distance and return top N matches
 * 
 * @param argc Argument count
 * @param argv Argument values
 * @return 0 on success, 1 on error
 */
int main(int argc, char* argv[]) {
    // Validate command line arguments
    if (argc != 6) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    string targetImage = argv[1];   // Path to query image
    string featureCSV = argv[2];    // Pre-computed features CSV file
    string featureType = argv[3];   // Type of features (must match CSV)
    string metricType = argv[4];    // Distance metric to use
    int topN = stoi(argv[5]);       // Number of top matches to return
    
    // Display configuration
    cout << "========================================" << endl;
    cout << "CBIR Query System" << endl;
    cout << "========================================" << endl;
    cout << "Target image    : " << targetImage << endl;
    cout << "Feature CSV     : " << featureCSV << endl;
    cout << "Feature type    : " << featureType << endl;
    cout << "Distance metric : " << metricType << endl;
    cout << "Top N matches   : " << topN << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    // Validate target image exists
    if (!Utils::fileExists(targetImage)) {
        cerr << "Error: Target image does not exist: " << targetImage << endl;
        return 1;
    }
    
    // Validate feature CSV exists
    if (!Utils::fileExists(featureCSV)) {
        cerr << "Error: Feature CSV does not exist: " << featureCSV << endl;
        return 1;
    }
    
    // Load target image
    cout << "Loading target image..." << endl;
    cv::Mat queryImage = Utils::loadImage(targetImage);
    if (queryImage.empty()) {
        cerr << "Error: Failed to load target image" << endl;
        return 1;
    }
    
    // Create feature extractor (must match type used to build database)
    FeatureExtractor* extractor = createFeatureExtractor(featureType);
    if (extractor == nullptr) {
        return 1;  // Error message already printed
    }
    
    // Create distance metric
    DistanceMetric* metric = createDistanceMetric(metricType);
    if (metric == nullptr) {
        delete extractor;
        return 1;  // Error message already printed
    }
    
    // Load pre-computed features from CSV into memory
    // This loads the entire database (all feature vectors) into RAM
    cout << "Loading feature database..." << endl;
    FeatureDatabase database;
    if (!database.loadFromCSV(featureCSV)) {
        cerr << "Error: Failed to load feature database" << endl;
        delete extractor;
        delete metric;
        return 1;
    }
    
    // Setup retrieval system with loaded components
    cout << "Setting up retrieval system..." << endl;
    ImageRetrieval retrieval;
    retrieval.setFeatureDatabase(&database);      // Database with all pre-computed features
    retrieval.setFeatureExtractor(extractor);     // Extractor for query image
    retrieval.setDistanceMetric(metric);          // Metric for comparison
    
    // Perform query
    // This will:
    //   1. Extract features from query image
    //   2. Compare to all features in database
    //   3. Sort by distance
    //   4. Return top N matches
    cout << endl;
    cout << "Querying database..." << endl;
    cout << "-------------------------------------------" << endl;
    
    vector<ImageMatch> results = retrieval.query(queryImage, topN);
    
    // Check if query succeeded
    if (results.empty()) {
        cerr << "Error: No results found" << endl;
        return 1;
    }
    
    // Display results
    displayResults(targetImage, results);
    
    // Cleanup (not strictly necessary as program exits, but good practice)
    // Note: extractor and metric are managed by smart pointers in retrieval
    
    return 0;
}
