////////////////////////////////////////////////////////////////////////////////
// Utils.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of utility functions for CSV I/O, file operations,
//              image loading, and string processing for the CBIR system.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "Utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace cbir {
namespace Utils {

//==============================================================================
// CSV File I/O Operations
//==============================================================================

/**
 * @brief Read feature vectors from CSV file
 * 
 * File format: filename,feature1,feature2,...,featureN
 * 
 * @author Krushna Sanjay Sharma
 */
bool readFeaturesCSV(const std::string& filename,
                    std::map<std::string, cv::Mat>& features,
                    bool hasHeader) {
    // Open file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    // Clear output map
    features.clear();
    
    std::string line;
    int lineNum = 0;
    
    // Read file line by line
    while (std::getline(file, line)) {
        lineNum++;
        
        // Skip header if present
        if (lineNum == 1 && hasHeader) {
            continue;
        }
        
        // Skip empty lines
        if (line.empty() || trim(line).empty()) {
            continue;
        }
        
        // Parse line
        std::vector<std::string> tokens = split(line, ',');
        
        if (tokens.size() < 2) {
            std::cerr << "Warning: Line " << lineNum << " has insufficient data" << std::endl;
            continue;
        }
        
        // First token is filename
        std::string imageName = trim(tokens[0]);
        
        // Remaining tokens are feature values
        std::vector<float> featureValues;
        for (size_t i = 1; i < tokens.size(); i++) {
            try {
                float value = std::stof(trim(tokens[i]));
                featureValues.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid feature value at line " << lineNum 
                          << ", column " << i << std::endl;
                continue;
            }
        }
        
        // Create cv::Mat from feature values (1 row, N columns)
        cv::Mat featureMat(1, static_cast<int>(featureValues.size()), CV_32F);
        for (size_t i = 0; i < featureValues.size(); i++) {
            featureMat.at<float>(0, i) = featureValues[i];
        }
        
        // Store in map
        features[imageName] = featureMat;
    }
    
    file.close();
    
    std::cout << "Read " << features.size() << " feature vectors from " 
              << filename << std::endl;
    
    return !features.empty();
}

/**
 * @brief Write feature vectors to CSV file
 * 
 * @author Krushna Sanjay Sharma
 */
bool writeFeaturesCSV(const std::string& filename,
                     const std::map<std::string, cv::Mat>& features,
                     const std::string& header) {
    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return false;
    }
    
    // Write header if provided
    if (!header.empty()) {
        file << header << std::endl;
    }
    
    // Write each feature vector
    for (const auto& pair : features) {
        const std::string& imageName = pair.first;
        const cv::Mat& featureMat = pair.second;
        
        // Write filename
        file << imageName;
        
        // Write feature values
        for (int i = 0; i < featureMat.total(); i++) {
            file << "," << featureMat.at<float>(i);
        }
        
        file << std::endl;
    }
    
    file.close();
    
    std::cout << "Wrote " << features.size() << " feature vectors to " 
              << filename << std::endl;
    
    return true;
}

//==============================================================================
// Directory Operations
//==============================================================================

/**
 * @brief Get list of image files in directory
 * 
 * @author Krushna Sanjay Sharma
 */
std::vector<std::string> getImageFiles(const std::string& directory,
                                       bool recursive) {
    std::vector<std::string> imageFiles;
    
    // Check if directory exists
    if (!directoryExists(directory)) {
        std::cerr << "Error: Directory does not exist: " << directory << std::endl;
        return imageFiles;
    }
    
    // Supported image extensions
    std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    };
    
    try {
        // Iterate through directory
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string ext = toLower(entry.path().extension().string());
                    if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string ext = toLower(entry.path().extension().string());
                    if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                        imageFiles.push_back(entry.path().string());
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }
    
    // Sort alphabetically
    std::sort(imageFiles.begin(), imageFiles.end());
    
    std::cout << "Found " << imageFiles.size() << " image files in " 
              << directory << std::endl;
    
    return imageFiles;
}

/**
 * @brief Extract filename from full path
 * 
 * @author Krushna Sanjay Sharma
 */
std::string getFilename(const std::string& filepath) {
    fs::path p(filepath);
    return p.filename().string();
}

/**
 * @brief Check if file exists
 * 
 * @author Krushna Sanjay Sharma
 */
bool fileExists(const std::string& filepath) {
    return fs::exists(filepath) && fs::is_regular_file(filepath);
}

/**
 * @brief Check if directory exists
 * 
 * @author Krushna Sanjay Sharma
 */
bool directoryExists(const std::string& dirpath) {
    return fs::exists(dirpath) && fs::is_directory(dirpath);
}

//==============================================================================
// Image Operations
//==============================================================================

/**
 * @brief Load image with error handling
 * 
 * @author Krushna Sanjay Sharma
 */
cv::Mat loadImage(const std::string& filepath, int flags) {
    cv::Mat image = cv::imread(filepath, flags);
    
    if (image.empty()) {
        std::cerr << "Error: Cannot load image " << filepath << std::endl;
    }
    
    return image;
}

/**
 * @brief Display image in window
 * 
 * @author Krushna Sanjay Sharma
 */
void showImage(const std::string& windowName,
               const cv::Mat& image,
               int waitTime) {
    if (image.empty()) {
        std::cerr << "Error: Cannot display empty image" << std::endl;
        return;
    }
    
    cv::imshow(windowName, image);
    cv::waitKey(waitTime);
}

//==============================================================================
// String Operations
//==============================================================================

/**
 * @brief Split string by delimiter
 * 
 * @author Krushna Sanjay Sharma
 */
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

/**
 * @brief Trim whitespace from string
 * 
 * @author Krushna Sanjay Sharma
 */
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    size_t end = str.find_last_not_of(" \t\n\r");
    
    if (start == std::string::npos) {
        return "";
    }
    
    return str.substr(start, end - start + 1);
}

/**
 * @brief Convert string to lowercase
 * 
 * @author Krushna Sanjay Sharma
 */
std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

//==============================================================================
// Feature Operations
//==============================================================================

/**
 * @brief Convert feature vector to CSV string
 * 
 * @author Krushna Sanjay Sharma
 */
std::string featuresToString(const cv::Mat& features) {
    std::ostringstream oss;
    
    for (int i = 0; i < features.total(); i++) {
        if (i > 0) oss << ",";
        oss << features.at<float>(i);
    }
    
    return oss.str();
}

/**
 * @brief Parse feature vector from CSV string
 * 
 * @author Krushna Sanjay Sharma
 */
cv::Mat stringToFeatures(const std::string& str) {
    std::vector<std::string> tokens = split(str, ',');
    cv::Mat features(1, static_cast<int>(tokens.size()), CV_32F);
    
    for (size_t i = 0; i < tokens.size(); i++) {
        features.at<float>(0, i) = std::stof(trim(tokens[i]));
    }
    
    return features;
}

/**
 * @brief Print feature vector to console
 * 
 * @author Krushna Sanjay Sharma
 */
void printFeatures(const cv::Mat& features, int maxElements) {
    std::cout << "Feature vector [" << features.total() << " elements]: ";
    
    int numToPrint = std::min(maxElements, static_cast<int>(features.total()));
    
    for (int i = 0; i < numToPrint; i++) {
        std::cout << features.at<float>(i);
        if (i < numToPrint - 1) std::cout << ", ";
    }
    
    if (features.total() > maxElements) {
        std::cout << " ...";
    }
    
    std::cout << std::endl;
}

} // namespace Utils
} // namespace cbir
