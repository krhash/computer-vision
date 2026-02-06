////////////////////////////////////////////////////////////////////////////////
// FeatureDatabase.h
// Author: Krushna Sanjay Sharma
// Description: Manages pre-computed feature vectors for image database. Handles
//              building, saving, and loading feature databases to/from CSV files
//              for efficient CBIR queries.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef FEATURE_DATABASE_H
#define FEATURE_DATABASE_H

#include "FeatureExtractor.h"
#include "Utils.h"
#include <map>
#include <string>
#include <vector>

namespace cbir {

/**
 * @class FeatureDatabase
 * @brief Manages feature vectors for an image database
 * 
 * This class handles the creation, storage, and retrieval of feature vectors
 * for a collection of images. Features can be pre-computed and saved to CSV
 * files for fast querying without recomputing features each time.
 * 
 * Workflow:
 * 1. Build database: extract features from all images in a directory
 * 2. Save to CSV: persist features for future use
 * 3. Load from CSV: quickly load pre-computed features
 * 4. Query: retrieve features for specific images
 * 
 * Usage example:
 * @code
 *   // Build and save
 *   FeatureDatabase db;
 *   BaselineFeature extractor;
 *   db.buildDatabase("images/", &extractor);
 *   db.saveToCSV("baseline_features.csv");
 *   
 *   // Later: load and query
 *   FeatureDatabase db2;
 *   db2.loadFromCSV("baseline_features.csv");
 *   cv::Mat features = db2.getFeatures("pic.0123.jpg");
 * @endcode
 * 
 * @author Krushna Sanjay Sharma
 */
class FeatureDatabase {
public:
    /**
     * @brief Default constructor
     */
    FeatureDatabase();

    /**
     * @brief Destructor
     */
    ~FeatureDatabase() = default;

    /**
     * @brief Build feature database from image directory
     * 
     * Scans the directory for image files, extracts features using the
     * provided extractor, and stores them in memory.
     * 
     * @param imageDirectory Path to directory containing images
     * @param extractor Feature extractor to use
     * @param recursive Search subdirectories if true
     * @return bool True if successful, false on error
     */
    bool buildDatabase(const std::string& imageDirectory,
                      FeatureExtractor* extractor,
                      bool recursive = false);

    /**
     * @brief Save feature database to CSV file
     * 
     * Format: filename,feature1,feature2,...,featureN
     * 
     * @param filename Output CSV file path
     * @return bool True if successful, false on error
     */
    bool saveToCSV(const std::string& filename) const;

    /**
     * @brief Load feature database from CSV file
     * 
     * @param filename Input CSV file path
     * @return bool True if successful, false on error
     */
    bool loadFromCSV(const std::string& filename);

    /**
     * @brief Get feature vector for a specific image
     * 
     * @param imageName Name of image file (can be full path or just filename)
     * @return cv::Mat Feature vector, empty if not found
     */
    cv::Mat getFeatures(const std::string& imageName) const;

    /**
     * @brief Check if database contains features for an image
     * 
     * @param imageName Name of image file
     * @return bool True if features exist for this image
     */
    bool hasFeatures(const std::string& imageName) const;

    /**
     * @brief Get all image names in the database
     * 
     * @return std::vector<std::string> List of all image filenames
     */
    std::vector<std::string> getImageNames() const;

    /**
     * @brief Get number of images in database
     * 
     * @return size_t Number of stored feature vectors
     */
    size_t size() const { return features_.size(); }

    /**
     * @brief Check if database is empty
     * 
     * @return bool True if no features are stored
     */
    bool empty() const { return features_.empty(); }

    /**
     * @brief Clear all stored features
     */
    void clear();

    /**
     * @brief Add or update features for a single image
     * 
     * @param imageName Image filename
     * @param features Feature vector
     */
    void addFeatures(const std::string& imageName, const cv::Mat& features);

    /**
     * @brief Get all features as a map
     * 
     * @return const std::map<std::string, cv::Mat>& Reference to features map
     */
    const std::map<std::string, cv::Mat>& getAllFeatures() const {
        return features_;
    }

private:
    /// Map: image filename -> feature vector
    std::map<std::string, cv::Mat> features_;

    /**
     * @brief Normalize image name (extract filename from full path)
     * 
     * @param imagePath Full or partial image path
     * @return std::string Just the filename
     */
    std::string normalizeImageName(const std::string& imagePath) const;
};

} // namespace cbir

#endif // FEATURE_DATABASE_H
