////////////////////////////////////////////////////////////////////////////////
// DNNFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of DNN feature loader for pre-computed ResNet18
//              embeddings. Loads features from CSV instead of computing them.
//
// This is different from other feature extractors because:
//   - Features are pre-computed (no image processing needed)
//   - Features are loaded from CSV file at initialization
//   - extractFeatures() performs lookup, not computation
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "DNNFeature.h"
#include "Utils.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace cbir {

/**
 * Constructor
 * 
 * @param csvPath Optional path to CSV file (can load later)
 */
DNNFeature::DNNFeature(const std::string& csvPath)
    : csvPath_(csvPath) {
    
    // Load features if path provided
    if (!csvPath_.empty()) {
        loadFeaturesFromCSV(csvPath_);
    }
}

/**
 * Extract features from image
 * 
 * For DNN features, we don't compute from image.
 * This returns empty Mat. Use getFeaturesByFilename() instead.
 */
cv::Mat DNNFeature::extractFeatures(const cv::Mat& image) {
    std::cerr << "Warning: DNNFeature::extractFeatures() not supported." << std::endl;
    std::cerr << "DNN features are pre-computed. Use getFeaturesByFilename() instead." << std::endl;
    return cv::Mat();
}

/**
 * Get pre-computed features by filename
 * 
 * @param filename Image filename (e.g., "pic.0001.jpg")
 * @return 512-dimensional DNN feature vector, empty if not found
 */
cv::Mat DNNFeature::getFeaturesByFilename(const std::string& filename) const {
    // Normalize filename for lookup
    std::string normalizedName = normalizeFilename(filename);
    
    // Search in loaded features
    auto it = features_.find(normalizedName);
    
    if (it != features_.end()) {
        return it->second;
    }
    
    // Not found
    std::cerr << "Warning: No DNN features found for " << filename << std::endl;
    return cv::Mat();
}

std::string DNNFeature::getFeatureName() const {
    return "DNNEmbedding_ResNet18";
}

int DNNFeature::getFeatureDimension() const {
    return 512;  // ResNet18 embeddings are 512-dimensional
}

/**
 * Load DNN features from CSV file
 * 
 * CSV format:
 *   filename,feature1,feature2,...,feature512
 *   pic.0001.jpg,0.123,0.456,...,0.789
 * 
 * @param csvPath Path to CSV file
 * @return True if successful
 */
bool DNNFeature::loadFeaturesFromCSV(const std::string& csvPath) {
    std::cout << "Loading DNN features from: " << csvPath << std::endl;
    
    // Clear existing features
    features_.clear();
    csvPath_ = csvPath;
    
    // Open CSV file
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open DNN features CSV: " << csvPath << std::endl;
        return false;
    }
    
    std::string line;
    int lineNum = 0;
    
    // Read file line by line
    while (std::getline(file, line)) {
        lineNum++;
        
        // Skip header (first line)
        if (lineNum == 1) {
            continue;
        }
        
        // Skip empty lines
        if (line.empty() || Utils::trim(line).empty()) {
            continue;
        }
        
        // Parse line
        std::vector<std::string> tokens = Utils::split(line, ',');
        
        // Expect: filename + 512 feature values = 513 tokens
        if (tokens.size() < 513) {
            std::cerr << "Warning: Line " << lineNum << " has only " 
                      << tokens.size() << " tokens (expected 513)" << std::endl;
            continue;
        }
        
        // First token is filename
        std::string filename = Utils::trim(tokens[0]);
        
        // Remaining 512 tokens are feature values
        cv::Mat featureVec(1, 512, CV_32F);
        
        for (int i = 0; i < 512; i++) {
            try {
                featureVec.at<float>(0, i) = std::stof(Utils::trim(tokens[i + 1]));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid feature value at line " << lineNum 
                          << ", column " << (i + 1) << std::endl;
                continue;
            }
        }
        
        // Store in map
        features_[normalizeFilename(filename)] = featureVec;
    }
    
    file.close();
    
    std::cout << "Loaded " << features_.size() << " DNN feature vectors" << std::endl;
    
    return !features_.empty();
}

/**
 * Normalize filename for consistent lookup
 * 
 * Extracts just the filename from full path
 */
std::string DNNFeature::normalizeFilename(const std::string& filename) const {
    return Utils::getFilename(filename);
}

} // namespace cbir
