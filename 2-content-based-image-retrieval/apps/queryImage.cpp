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
#include "MultiRegionHistogramIntersection.h"
#include "TextureColorFeature.h"
#include "WeightedHistogramIntersection.h"
#include "GaborTextureColorFeature.h"
#include "DNNFeature.h"
#include "CosineDistance.h"
#include "ProductMatcherFeature.h"
#include "ProductMatcherDistance.h"
#include "FaceAwareFeature.h"
#include "FaceAwareDistance.h"
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
    cout << "                 Options: baseline, histogram, chromaticity," << endl;
    cout << "                          multihistogram, texturecolor, gabor," << endl;
    cout << "                          dnn, productmatcher" << endl;
    cout << "  metric       : Distance metric to use" << endl;
    cout << "                 Options: ssd, histogram, multiregion," << endl;
    cout << "                          weighted, gabor, cosine, productmatcher" << endl;
    cout << "  topN         : Number of top matches to return" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  " << programName << " pic.1016.jpg baseline_features.csv baseline ssd 3" << endl;
    cout << "  " << programName << " pic.0164.jpg histogram_features.csv histogram histogram 3" << endl;
    cout << "  " << programName << " pic.0274.jpg multi_features.csv multihistogram multiregion 3" << endl;
    cout << "  " << programName << " pic.1072.jpg product_features.csv productmatcher productmatcher 5" << endl;
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
        // Task 3: MUST match buildFeatureDB parameters!
        // GRID split, 4 regions (2×2 quadrants)
        // Total: 4 × 512 = 2048 values
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
    } else if (type == "gabor" || type == "gaborcolor") {
        // Task 4 Extension: Gabor texture + Color
        // 4 orientations × 2 scales × 8 bins = 64 texture bins
        // 8×8×8 = 512 color bins
        // Total: 576 values
        return new cbir::GaborTextureColorFeature(
            4,    // orientations
            2,    // scales
            8,    // bins per Gabor histogram
            8,    // color bins per channel
            true  // normalize
        );
    } else if (type == "dnn" || type == "resnet") {
        // Task 5: DNN features (pre-computed ResNet18 embeddings)
        // Features are loaded from CSV, not computed from images
        return new cbir::DNNFeature("../data/features/ResNet18_olym.csv");
    } else if (type == "productmatcher" || type == "product") {
        return new cbir::ProductMatcherFeature(
            "../data/features/ResNet18_olym.csv", 0.3, 8
        );
    }  else if (type == "faceaware" || type == "adaptive") {
        return new cbir::FaceAwareFeature(
            "../data/features/ResNet18_olym.csv",
            "haarcascade_frontalface_alt2.xml"
        );
    }
    
    cerr << "Error: Unknown feature type '" << featureType << "'" << endl;
    cerr << "Available: baseline, histogram, chromaticity, multihorizontal, texturecolor, gabor, dnn, productmatcher, faceaware" << endl;
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

    } else if (type == "multiregion") {

        // Task 3: Custom multi-region histogram intersection
        // Computes intersection per region, combines with equal weights
        // For 2×2 grid: 4 regions, 512 bins each
        std::vector<double> equalWeights = {0.25, 0.25, 0.25, 0.25};
        return new cbir::MultiRegionHistogramIntersection(
            4,              // 4 regions
            512,            // 512 bins per region
            equalWeights    // Equal weights for all regions
        );
        
    } else if (type == "weighted" || type == "texturecolor") {

        // Weighted intersection: 50% texture, 50% color
        return new cbir::WeightedHistogramIntersection(
            16,   // texture dimension
            512,  // color dimension
            0.5,  // 50% weight for texture
            0.5   // 50% weight for color
        );

    } else if (type == "gabor" || type == "gaborweighted") {

        // Gabor texture + color: 64 + 512
        return new cbir::WeightedHistogramIntersection(64, 512, 0.5, 0.5);

    } else if (type == "cosine") {
        // Task 5: Cosine distance
        // Measures angle between vectors (scale-invariant)
        return new cbir::CosineDistance();
    } else if (type == "productmatcher" || type == "product") {
        return new cbir::ProductMatcherDistance(0.6, 0.4);
    } else if (type == "faceaware" || type == "adaptive") {
        return new cbir::FaceAwareDistance();
    }
    
    cerr << "Error: Unknown metric type '" << metricType << "'" << endl;
    cerr << "Available: ssd, histogram, multiregion, weighted, gabor, cosine, productmatcher, faceaware" << endl;
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
    string targetImage = argv[1];
    string featureCSV = argv[2];
    string featureType = argv[3];
    string metricType = argv[4];
    int topN = stoi(argv[5]);
    
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
    
    // Validate files exist
    if (!Utils::fileExists(targetImage)) {
        cerr << "Error: Target image does not exist: " << targetImage << endl;
        return 1;
    }
    
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
    
    // Create feature extractor
    FeatureExtractor* extractor = createFeatureExtractor(featureType);
    if (extractor == nullptr) {
        return 1;
    }
    
    // Create distance metric
    DistanceMetric* metric = createDistanceMetric(metricType);
    if (metric == nullptr) {
        delete extractor;
        return 1;
    }
    
    // Load feature database
    cout << "Loading feature database..." << endl;
    FeatureDatabase database;
    if (!database.loadFromCSV(featureCSV)) {
        cerr << "Error: Failed to load feature database" << endl;
        delete extractor;
        delete metric;
        return 1;
    }
    
    // Setup retrieval system
    cout << "Setting up retrieval system..." << endl;
    ImageRetrieval retrieval;
    retrieval.setFeatureDatabase(&database);
    retrieval.setFeatureExtractor(extractor);
    retrieval.setDistanceMetric(metric);
    
    // Perform query
    cout << endl;
    cout << "Querying database..." << endl;
    cout << "-------------------------------------------" << endl;
    
    // ⭐ SINGLE results variable declaration
    vector<ImageMatch> results;
    
    // Check for special feature types that need filename-based extraction
    ProductMatcherFeature* productMatcher = dynamic_cast<ProductMatcherFeature*>(extractor);
    FaceAwareFeature* faceAware = dynamic_cast<FaceAwareFeature*>(extractor);
    DNNFeature* dnnExtractor = dynamic_cast<DNNFeature*>(extractor);
    
    if (productMatcher) {
        // ProductMatcher: Needs filename for DNN lookup
        string queryFilename = Utils::getFilename(targetImage);
        cv::Mat queryFeatures = productMatcher->extractFeaturesWithFilename(queryImage, queryFilename);
        
        if (queryFeatures.empty()) {
            cerr << "Error: Failed to extract ProductMatcher features" << endl;
            return 1;
        }
        
        results = retrieval.queryWithFeatures(queryFeatures, topN);
        
    } else if (dnnExtractor) {
        // Pure DNN: Needs filename for feature lookup
        string queryFilename = Utils::getFilename(targetImage);
        cv::Mat queryFeatures = dnnExtractor->getFeaturesByFilename(queryFilename);
        
        if (queryFeatures.empty()) {
            cerr << "Error: No DNN features found for " << queryFilename << endl;
            return 1;
        }
        
        results = retrieval.queryWithFeatures(queryFeatures, topN);
        
    } else if (faceAware) {
        string queryFilename = Utils::getFilename(targetImage);
        cv::Mat queryFeatures = faceAware->extractFeaturesWithFilename(queryImage, queryFilename);
        
        if (queryFeatures.empty()) {
            cerr << "Error: Failed to extract FaceAware features" << endl;
            return 1;
        }
        
        cout << "Face detection: " << (faceAware->lastImageHadFaces() ? "YES" : "NO") 
            << " (" << faceAware->getLastFaceCount() << " faces)" << endl;
        
        results = retrieval.queryWithFeatures(queryFeatures, topN);
    } else {
        // Normal query: Extract features from image
        results = retrieval.query(queryImage, topN);
    }
    
    // Check if query succeeded
    if (results.empty()) {
        cerr << "Error: No results found" << endl;
        return 1;
    }
    
    // Display results
    displayResults(targetImage, results);
    
    return 0;
}
