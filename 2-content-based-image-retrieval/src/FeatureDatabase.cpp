////////////////////////////////////////////////////////////////////////////////
// FeatureDatabase.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of feature database management for storing and
//              retrieving pre-computed image features for efficient CBIR queries.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "FeatureDatabase.h"
#include <iostream>
#include <algorithm>

namespace cbir {

/**
 * @brief Constructor
 * 
 * @author Krushna Sanjay Sharma
 */
FeatureDatabase::FeatureDatabase() {
    // Initialize empty database
}

/**
 * @brief Build feature database from image directory
 * 
 * Scans directory for images, extracts features, and stores in memory.
 * 
 * @author Krushna Sanjay Sharma
 */
bool FeatureDatabase::buildDatabase(const std::string& imageDirectory,
                                    FeatureExtractor* extractor,
                                    bool recursive) {
    if (extractor == nullptr) {
        std::cerr << "Error: Feature extractor is null" << std::endl;
        return false;
    }
    
    // Clear existing features
    clear();
    
    std::cout << "Building feature database from: " << imageDirectory << std::endl;
    std::cout << "Using feature extractor: " << extractor->getFeatureName() << std::endl;
    
    // Get list of image files
    std::vector<std::string> imageFiles = Utils::getImageFiles(imageDirectory, recursive);
    
    if (imageFiles.empty()) {
        std::cerr << "Error: No image files found in " << imageDirectory << std::endl;
        return false;
    }
    
    std::cout << "Processing " << imageFiles.size() << " images..." << std::endl;
    
    // Extract features for each image
    int successCount = 0;
    int failCount = 0;
    
    for (const auto& imagePath : imageFiles) {
        // Load image
        cv::Mat image = Utils::loadImage(imagePath);
        
        if (image.empty()) {
            std::cerr << "Warning: Failed to load " << imagePath << std::endl;
            failCount++;
            continue;
        }
        
        // Extract features
        cv::Mat features = extractor->extractFeatures(image);
        
        if (features.empty()) {
            std::cerr << "Warning: Failed to extract features from " 
                      << imagePath << std::endl;
            failCount++;
            continue;
        }
        
        // Store features with normalized filename
        std::string filename = normalizeImageName(imagePath);
        features_[filename] = features;
        successCount++;
        
        // Progress indicator
        if ((successCount + failCount) % 100 == 0) {
            std::cout << "  Processed " << (successCount + failCount) 
                      << "/" << imageFiles.size() << " images..." << std::endl;
        }
    }
    
    std::cout << "Feature extraction complete!" << std::endl;
    std::cout << "  Success: " << successCount << std::endl;
    std::cout << "  Failed:  " << failCount << std::endl;
    
    return successCount > 0;
}

/**
 * @brief Save features to CSV file
 * 
 * @author Krushna Sanjay Sharma
 */
bool FeatureDatabase::saveToCSV(const std::string& filename) const {
    if (features_.empty()) {
        std::cerr << "Error: No features to save" << std::endl;
        return false;
    }
    
    std::cout << "Saving feature database to: " << filename << std::endl;
    
    // Create header (optional - can be empty for this project)
    std::string header = "filename";
    
    // Use Utils function to write CSV
    bool success = Utils::writeFeaturesCSV(filename, features_, header);
    
    if (success) {
        std::cout << "Successfully saved " << features_.size() 
                  << " feature vectors" << std::endl;
    }
    
    return success;
}

/**
 * @brief Load features from CSV file
 * 
 * @author Krushna Sanjay Sharma
 */
bool FeatureDatabase::loadFromCSV(const std::string& filename) {
    std::cout << "Loading feature database from: " << filename << std::endl;
    
    // Clear existing features
    clear();
    
    // Use Utils function to read CSV
    bool success = Utils::readFeaturesCSV(filename, features_, true);
    
    if (success) {
        std::cout << "Successfully loaded " << features_.size() 
                  << " feature vectors" << std::endl;
    } else {
        std::cerr << "Error: Failed to load features from " << filename << std::endl;
    }
    
    return success;
}

/**
 * @brief Get features for a specific image
 * 
 * @author Krushna Sanjay Sharma
 */
cv::Mat FeatureDatabase::getFeatures(const std::string& imageName) const {
    // Normalize the image name
    std::string normalizedName = normalizeImageName(imageName);
    
    // Search in database
    auto it = features_.find(normalizedName);
    
    if (it != features_.end()) {
        return it->second;
    }
    
    // Not found - return empty Mat
    return cv::Mat();
}

/**
 * @brief Check if database has features for an image
 * 
 * @author Krushna Sanjay Sharma
 */
bool FeatureDatabase::hasFeatures(const std::string& imageName) const {
    std::string normalizedName = normalizeImageName(imageName);
    return features_.find(normalizedName) != features_.end();
}

/**
 * @brief Get all image names in database
 * 
 * @author Krushna Sanjay Sharma
 */
std::vector<std::string> FeatureDatabase::getImageNames() const {
    std::vector<std::string> names;
    names.reserve(features_.size());
    
    for (const auto& pair : features_) {
        names.push_back(pair.first);
    }
    
    return names;
}

/**
 * @brief Clear all features
 * 
 * @author Krushna Sanjay Sharma
 */
void FeatureDatabase::clear() {
    features_.clear();
}

/**
 * @brief Add or update features for an image
 * 
 * @author Krushna Sanjay Sharma
 */
void FeatureDatabase::addFeatures(const std::string& imageName, 
                                  const cv::Mat& features) {
    std::string normalizedName = normalizeImageName(imageName);
    features_[normalizedName] = features;
}

/**
 * @brief Normalize image name (extract filename from path)
 * 
 * Ensures consistent naming regardless of whether full paths or just
 * filenames are provided.
 * 
 * @author Krushna Sanjay Sharma
 */
std::string FeatureDatabase::normalizeImageName(const std::string& imagePath) const {
    // Extract just the filename from the path
    return Utils::getFilename(imagePath);
}

} // namespace cbir
