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
#include "MultiHistogramFeature.h"
#include "TextureColorFeature.h"
#include "GaborTextureColorFeature.h"
#include "DNNFeature.h"
#include "ProductMatcherFeature.h"
#include "FaceAwareFeature.h"
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
    cout << "                 Options: baseline, histogram, chromaticity," << endl;
    cout << "                          multihistogram, texturecolor, gabor," << endl;
    cout << "                          dnn, productmatcher" << endl;  // ⭐ UPDATED
    cout << "  output_csv   : Output CSV file for features" << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  " << programName << " data/images baseline baseline_features.csv" << endl;
    cout << "  " << programName << " data/images histogram histogram_features.csv" << endl;
    cout << "  " << programName << " data/images productmatcher product_features.csv" << endl;
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
        return new cbir::HistogramFeature(
            cbir::HistogramFeature::HistogramType::RG_CHROMATICITY, 
            16,     // bins per channel
            true    // normalize to sum=1.0
        );
    } else if (type == "multihistogram" || type == "multi") {
        // Task 3: Multi-region RGB histogram
        // Split: GRID (2×2 quadrants)
        // Regions: 4 (top-left, top-right, bottom-left, bottom-right)
        // Each region: 8×8×8 = 512 bins
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
        // Task 5: DNN features (pre-computed, loaded from CSV)
        // NOTE: For DNN, buildFeatureDB just copies the existing CSV
        // No feature extraction is performed
        return new cbir::DNNFeature();
    } else if (type == "productmatcher" || type == "product") {
        // Task 7: Custom ProductMatcher feature
        // DNN (85%) + Center-region color (15%)
        return new cbir::ProductMatcherFeature(
            "../data/features/ResNet18_olym.csv",  // DNN features path
            0.3,  // Center 50% of image
            8     // 8 bins per color channel
        );
    } else if (type == "faceaware" || type == "adaptive") {
        // Extension: Adaptive face-aware feature
        return new cbir::FaceAwareFeature(
            "../data/features/ResNet18_olym.csv",
            "haarcascade_frontalface_alt2.xml"
        );
    }

    cerr << "Error: Unknown feature type '" << featureType << "'" << endl;
    cerr << "Available: baseline, histogram, chromaticity, multihistogram, texturecolor, gabor, dnn, productmatcher, faceaware" << endl;
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
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    string imageDir = argv[1];
    string featureType = argv[2];
    string outputCSV = argv[3];

    // Special case for pure DNN features
    if (Utils::toLower(featureType) == "dnn" || Utils::toLower(featureType) == "resnet") {
        cout << "========================================" << endl;
        cout << "DNN Features (Pre-computed)" << endl;
        cout << "========================================" << endl;
        cout << "DNN features are pre-computed in ResNet18_olym.csv" << endl;
        cout << "No feature extraction needed." << endl;
        cout << endl;
        cout << "For querying, use the existing CSV file:" << endl;
        cout << "  ../data/features/ResNet18_olym.csv" << endl;
        cout << "========================================" << endl;
        return 0;
    }
    
    cout << "========================================" << endl;
    cout << "Building Feature Database" << endl;
    cout << "========================================" << endl;
    cout << "Image directory : " << imageDir << endl;
    cout << "Feature type    : " << featureType << endl;
    cout << "Output CSV      : " << outputCSV << endl;
    cout << "========================================" << endl;
    cout << endl;
    
    if (!Utils::directoryExists(imageDir)) {
        cerr << "Error: Image directory does not exist: " << imageDir << endl;
        return 1;
    }
    
    FeatureExtractor* extractor = createFeatureExtractor(featureType);
    if (extractor == nullptr) {
        return 1;
    }
    
    // Check for special feature types that need filename-based extraction
    ProductMatcherFeature* productMatcher = dynamic_cast<ProductMatcherFeature*>(extractor);
    FaceAwareFeature* faceAware = dynamic_cast<FaceAwareFeature*>(extractor);
    
    if (productMatcher) {
        cout << "ProductMatcher mode: Combining DNN + Center-region color" << endl;
        cout << endl;
    }
    
    if (faceAware) {
        cout << "FaceAware mode: Adaptive feature selection based on face detection" << endl;
        cout << endl;
    }
    
    // Get list of image files
    cout << "Scanning image directory..." << endl;
    vector<string> imageFiles = Utils::getImageFiles(imageDir, false);
    
    if (imageFiles.empty()) {
        cerr << "Error: No image files found in " << imageDir << endl;
        delete extractor;
        return 1;
    }
    
    cout << "Found " << imageFiles.size() << " images" << endl;
    cout << endl;
    
    // Step 1: Extract features from all images
    cout << "Step 1: Extracting features from images..." << endl;
    cout << "-------------------------------------------" << endl;
    
    // Create feature database
    FeatureDatabase database;
    int successCount = 0;
    int failCount = 0;
    
    for (const auto& imagePath : imageFiles) {
        // Load image
        cv::Mat image = Utils::loadImage(imagePath);
        
        if (image.empty()) {
            cerr << "Warning: Failed to load " << imagePath << endl;
            failCount++;
            continue;
        }
        
        // Extract features
        string filename = Utils::getFilename(imagePath);
        cv::Mat featureVec;  // ⭐ Single feature vector (cv::Mat)
        
        if (productMatcher) {
            // Special: ProductMatcher needs filename
            featureVec = productMatcher->extractFeaturesWithFilename(image, filename);
        } else if (faceAware) {
            // Special: FaceAware needs filename
            featureVec = faceAware->extractFeaturesWithFilename(image, filename);
        } else {
            // Normal: Extract from image only
            featureVec = extractor->extractFeatures(image);
        }
        
        if (featureVec.empty()) {
            cerr << "Warning: Failed to extract features from " << filename << endl;
            failCount++;
            continue;
        }
        
        // Store features in database
        database.addFeatures(filename, featureVec);
        successCount++;
        
        // Progress indicator
        if ((successCount + failCount) % 100 == 0) {
            cout << "  Processed " << (successCount + failCount) 
                 << "/" << imageFiles.size() << " images..." << endl;
        }
    }
    
    cout << "Feature extraction complete!" << endl;
    cout << "  Success: " << successCount << endl;
    cout << "  Failed:  " << failCount << endl;
    cout << endl;
    
    if (successCount == 0) {
        cerr << "Error: No features extracted" << endl;
        delete extractor;
        return 1;
    }
    
    // Step 2: Save features to CSV
    cout << "Step 2: Saving features to CSV..." << endl;
    cout << "-------------------------------------------" << endl;
    
    bool saveSuccess = database.saveToCSV(outputCSV);
    
    if (!saveSuccess) {
        cerr << "Error: Failed to save features to CSV" << endl;
        delete extractor;
        return 1;
    }
    
    cout << endl;
    cout << "========================================" << endl;
    cout << "SUCCESS!" << endl;
    cout << "========================================" << endl;
    cout << "Feature database saved to: " << outputCSV << endl;
    cout << "Total images processed: " << successCount << endl;
    cout << endl;
    cout << "Next step: Query images using queryImage" << endl;
    cout << "========================================" << endl;
    
    delete extractor;
    
    return 0;
}
