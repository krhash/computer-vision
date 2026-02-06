////////////////////////////////////////////////////////////////////////////////
// DistanceMetric.h
// Author: Krushna Sanjay Sharma
// Description: Abstract base class for distance/similarity metrics used to
//              compare feature vectors in the CBIR system. Lower distances
//              indicate higher similarity between images.
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef DISTANCE_METRIC_H
#define DISTANCE_METRIC_H

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace cbir {

/**
 * @class DistanceMetric
 * @brief Abstract base class for distance/similarity metrics
 * 
 * This class defines the interface for computing distances between feature
 * vectors. Distance metrics quantify the similarity between two images based
 * on their extracted features. Lower distances indicate higher similarity.
 * 
 * Derived classes must implement:
 * - compute(): Calculate distance between two feature vectors
 * - getMetricName(): Return the name of the metric
 * 
 * Common distance metrics include:
 * - Sum of Squared Differences (SSD / L2 distance)
 * - Histogram Intersection
 * - Cosine Distance
 * - Manhattan Distance (L1)
 * - Chi-Square Distance
 * 
 * @author Krushna Sanjay Sharma
 */
class DistanceMetric {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~DistanceMetric() = default;

    /**
     * @brief Compute distance between two feature vectors
     * 
     * This pure virtual function must be implemented by all derived classes.
     * It computes the distance/dissimilarity between two feature vectors.
     * 
     * @param features1 First feature vector
     * @param features2 Second feature vector
     * @return double Distance value (lower = more similar)
     * 
     * @note Feature vectors must have compatible dimensions
     * @note Return 0.0 when comparing identical features
     * @note Return positive values for dissimilar features
     */
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) = 0;

    /**
     * @brief Get the name of this distance metric
     * 
     * @return std::string Descriptive name (e.g., "SSD", "HistogramIntersection")
     */
    virtual std::string getMetricName() const = 0;

    /**
     * @brief Check if two feature vectors are compatible for comparison
     * 
     * Validates that the feature vectors have compatible dimensions and types
     * for distance computation.
     * 
     * @param features1 First feature vector
     * @param features2 Second feature vector
     * @return bool True if compatible, false otherwise
     */
    virtual bool areCompatible(const cv::Mat& features1, 
                               const cv::Mat& features2) const {
        // Check if matrices are empty
        if (features1.empty() || features2.empty()) {
            return false;
        }
        
        // Check if dimensions match
        if (features1.size() != features2.size()) {
            return false;
        }
        
        // Check if types match
        if (features1.type() != features2.type()) {
            return false;
        }
        
        return true;
    }

protected:
    /**
     * @brief Protected constructor - only derived classes can instantiate
     */
    DistanceMetric() = default;
};

// Type alias for smart pointer to DistanceMetric
using DistanceMetricPtr = std::shared_ptr<DistanceMetric>;

} // namespace cbir

#endif // DISTANCE_METRIC_H
