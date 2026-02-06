////////////////////////////////////////////////////////////////////////////////
// Utils.h
// Author: Krushna Sanjay Sharma
// Description: Utility functions for the CBIR system including CSV file I/O,
//              image loading, directory operations, and helper functions for
//              feature processing and visualization.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

namespace cbir {

/**
 * @struct ImageMatch
 * @brief Structure to store image matching results
 */
struct ImageMatch {
    std::string filename;    ///< Name of the matched image file
    double distance;         ///< Distance from query image (lower = better match)
    cv::Mat image;          ///< Optional: loaded image data
    
    /**
     * @brief Constructor
     */
    ImageMatch(const std::string& fname = "", double dist = 0.0)
        : filename(fname), distance(dist) {}
    
    /**
     * @brief Comparison operator for sorting (ascending distance)
     */
    bool operator<(const ImageMatch& other) const {
        return distance < other.distance;
    }
};

/**
 * @namespace Utils
 * @brief Utility functions for CBIR system operations
 */
namespace Utils {

    //==========================================================================
    // File I/O Operations
    //==========================================================================

    /**
     * @brief Read CSV file containing feature vectors
     * 
     * Reads a CSV file where each row contains:
     * filename, feature1, feature2, ..., featureN
     * 
     * @param filename Path to CSV file
     * @param features Output map: filename -> feature vector
     * @param hasHeader If true, skip first row as header
     * @return bool True if successful, false on error
     */
    bool readFeaturesCSV(const std::string& filename,
                        std::map<std::string, cv::Mat>& features,
                        bool hasHeader = true);

    /**
     * @brief Write feature vectors to CSV file
     * 
     * Writes features in format:
     * filename, feature1, feature2, ..., featureN
     * 
     * @param filename Output CSV file path
     * @param features Map of filename -> feature vector
     * @param header Optional header row
     * @return bool True if successful, false on error
     */
    bool writeFeaturesCSV(const std::string& filename,
                         const std::map<std::string, cv::Mat>& features,
                         const std::string& header = "");

    //==========================================================================
    // Directory Operations
    //==========================================================================

    /**
     * @brief Get list of all image files in a directory
     * 
     * Supported formats: .jpg, .jpeg, .png, .bmp, .tif, .tiff
     * 
     * @param directory Path to directory
     * @param recursive If true, search subdirectories
     * @return std::vector<std::string> List of image file paths
     */
    std::vector<std::string> getImageFiles(const std::string& directory,
                                           bool recursive = false);

    /**
     * @brief Extract filename from full path
     * 
     * Example: "C:/data/images/pic.0123.jpg" -> "pic.0123.jpg"
     * 
     * @param filepath Full file path
     * @return std::string Filename only
     */
    std::string getFilename(const std::string& filepath);

    /**
     * @brief Check if file exists
     * 
     * @param filepath Path to file
     * @return bool True if file exists
     */
    bool fileExists(const std::string& filepath);

    /**
     * @brief Check if directory exists
     * 
     * @param dirpath Path to directory
     * @return bool True if directory exists
     */
    bool directoryExists(const std::string& dirpath);

    //==========================================================================
    // Image Operations
    //==========================================================================

    /**
     * @brief Load image with error handling
     * 
     * @param filepath Path to image file
     * @param flags OpenCV imread flags (default: cv::IMREAD_COLOR)
     * @return cv::Mat Loaded image, empty if failed
     */
    cv::Mat loadImage(const std::string& filepath, 
                      int flags = cv::IMREAD_COLOR);

    /**
     * @brief Display image in a window
     * 
     * @param windowName Name of display window
     * @param image Image to display
     * @param waitTime Wait time in ms (0 = wait for key press)
     */
    void showImage(const std::string& windowName,
                   const cv::Mat& image,
                   int waitTime = 0);

    //==========================================================================
    // String Operations
    //==========================================================================

    /**
     * @brief Split string by delimiter
     * 
     * @param str Input string
     * @param delimiter Delimiter character
     * @return std::vector<std::string> Split tokens
     */
    std::vector<std::string> split(const std::string& str, char delimiter);

    /**
     * @brief Trim whitespace from string
     * 
     * @param str Input string
     * @return std::string Trimmed string
     */
    std::string trim(const std::string& str);

    /**
     * @brief Convert string to lowercase
     * 
     * @param str Input string
     * @return std::string Lowercase string
     */
    std::string toLower(const std::string& str);

    //==========================================================================
    // Feature Operations
    //==========================================================================

    /**
     * @brief Convert feature vector to string for CSV output
     * 
     * @param features Feature vector
     * @return std::string Comma-separated values
     */
    std::string featuresToString(const cv::Mat& features);

    /**
     * @brief Parse feature vector from CSV string
     * 
     * @param str Comma-separated feature values
     * @return cv::Mat Feature vector as cv::Mat
     */
    cv::Mat stringToFeatures(const std::string& str);

    /**
     * @brief Print feature vector to console
     * 
     * @param features Feature vector
     * @param maxElements Maximum number of elements to print
     */
    void printFeatures(const cv::Mat& features, int maxElements = 10);

} // namespace Utils

} // namespace cbir

#endif // UTILS_H
