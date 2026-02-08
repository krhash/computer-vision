////////////////////////////////////////////////////////////////////////////////
// CosineDistance.h
// Author: Krushna Sanjay Sharma
// Description: Cosine distance metric for comparing high-dimensional feature
//              vectors. Often works better than Euclidean distance for deep
//              learning embeddings and normalized feature spaces.
//
// Cosine Distance Formula:
//   distance = 1 - cos(θ)
//   
//   where cos(θ) is the cosine of angle between two vectors:
//   cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
//          = dot(v1, v2) / (norm(v1) × norm(v2))
//
// Simplified computation:
//   1. Normalize both vectors by their L2-norm
//   2. Compute dot product of normalized vectors
//   3. distance = 1 - dot_product
//
// Properties:
//   - Range: [0, 2] (but typically [0, 1] for similar vectors)
//   - 0 = identical direction (parallel vectors)
//   - 1 = orthogonal vectors (90° angle)
//   - 2 = opposite direction (180° angle)
//   - Scale-invariant (only cares about direction, not magnitude)
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#ifndef COSINE_DISTANCE_H
#define COSINE_DISTANCE_H

#include "DistanceMetric.h"

namespace cbir {

/**
 * @class CosineDistance
 * @brief Cosine distance metric for high-dimensional vectors
 * 
 * Computes the cosine distance between two feature vectors.
 * This metric is particularly effective for:
 *   - High-dimensional spaces (e.g., 512D deep learning features)
 *   - Normalized feature vectors
 *   - Semantic similarity (direction matters more than magnitude)
 * 
 * The cosine distance measures the angle between vectors rather than
 * their Euclidean distance, making it scale-invariant.
 * 
 * @author Krushna Sanjay Sharma
 */
class CosineDistance : public DistanceMetric {
public:
    CosineDistance() = default;
    virtual ~CosineDistance() = default;
    
    virtual double compute(const cv::Mat& features1, 
                          const cv::Mat& features2) override;
    
    virtual std::string getMetricName() const override;

private:
    /**
     * Compute L2 norm (Euclidean length) of vector
     * 
     * @param features Feature vector
     * @return L2 norm (||v|| = sqrt(Σ v[i]²))
     */
    double computeL2Norm(const cv::Mat& features) const;
    
    /**
     * Compute dot product of two vectors
     * 
     * @param features1 First vector
     * @param features2 Second vector
     * @return Dot product (Σ v1[i] × v2[i])
     */
    double computeDotProduct(const cv::Mat& features1, 
                            const cv::Mat& features2) const;
};

} // namespace cbir

#endif // COSINE_DISTANCE_H
