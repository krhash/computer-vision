////////////////////////////////////////////////////////////////////////////////
// CosineDistance.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of cosine distance metric for high-dimensional
//              feature vectors (Task 5).
//
// Cosine distance measures the angle between two vectors:
//   cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
//   distance = 1 - cos(θ)
//
// Algorithm:
//   1. Normalize both vectors by their L2-norm
//   2. Compute dot product of normalized vectors (= cosine)
//   3. distance = 1 - cosine
//
// Advantages for DNN embeddings:
//   - Scale-invariant (only cares about direction)
//   - Works well in high-dimensional spaces (512D)
//   - Captures semantic similarity better than Euclidean distance
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "CosineDistance.h"
#include <iostream>
#include <cmath>

namespace cbir {

/**
 * Compute cosine distance between two feature vectors
 * 
 * Formula: distance = 1 - cos(θ)
 * 
 * where cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
 * 
 * Process:
 *   1. Compute L2 norms of both vectors
 *   2. Compute dot product
 *   3. cosine = dot_product / (norm1 × norm2)
 *   4. distance = 1 - cosine
 * 
 * Range:
 *   - 0.0 = identical direction (parallel vectors)
 *   - 1.0 = orthogonal (90° angle)
 *   - 2.0 = opposite direction (180° angle)
 * 
 * @param features1 First feature vector
 * @param features2 Second feature vector
 * @return Cosine distance [0, 2]
 */
double CosineDistance::compute(const cv::Mat& features1, 
                              const cv::Mat& features2) {
    // Validate compatibility
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Features not compatible for cosine distance" << std::endl;
        return -1.0;
    }
    
    // Compute L2 norms (Euclidean lengths)
    double norm1 = computeL2Norm(features1);
    double norm2 = computeL2Norm(features2);
    
    // Check for zero vectors
    if (norm1 < 1e-10 || norm2 < 1e-10) {
        std::cerr << "Warning: Zero vector in cosine distance computation" << std::endl;
        return 1.0;  // Orthogonal (maximum distance for normalized case)
    }
    
    // Compute dot product
    double dotProduct = computeDotProduct(features1, features2);
    
    // Compute cosine: cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
    double cosine = dotProduct / (norm1 * norm2);
    
    // Clamp to valid range [-1, 1] to handle numerical errors
    cosine = std::max(-1.0, std::min(1.0, cosine));
    
    // Convert to distance: distance = 1 - cos(θ)
    // cos(θ) = 1  → distance = 0 (same direction)
    // cos(θ) = 0  → distance = 1 (orthogonal)
    // cos(θ) = -1 → distance = 2 (opposite)
    double distance = 1.0 - cosine;
    
    return distance;
}

std::string CosineDistance::getMetricName() const {
    return "CosineDistance";
}

/**
 * Compute L2 norm (Euclidean length) of vector
 * 
 * ||v|| = sqrt(v[0]² + v[1]² + ... + v[n-1]²)
 * 
 * @param features Feature vector
 * @return L2 norm
 */
double CosineDistance::computeL2Norm(const cv::Mat& features) const {
    double sumSquares = 0.0;
    
    int totalElements = features.total();
    for (int i = 0; i < totalElements; i++) {
        float val = features.at<float>(i);
        sumSquares += val * val;
    }
    
    return std::sqrt(sumSquares);
}

/**
 * Compute dot product of two vectors
 * 
 * v1 · v2 = v1[0]×v2[0] + v1[1]×v2[1] + ... + v1[n-1]×v2[n-1]
 * 
 * @param features1 First vector
 * @param features2 Second vector
 * @return Dot product
 */
double CosineDistance::computeDotProduct(const cv::Mat& features1, 
                                        const cv::Mat& features2) const {
    double dotProduct = 0.0;
    
    int totalElements = features1.total();
    for (int i = 0; i < totalElements; i++) {
        float val1 = features1.at<float>(i);
        float val2 = features2.at<float>(i);
        dotProduct += val1 * val2;
    }
    
    return dotProduct;
}

} // namespace cbir
