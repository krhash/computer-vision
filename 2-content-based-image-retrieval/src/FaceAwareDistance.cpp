////////////////////////////////////////////////////////////////////////////////
// FaceAwareDistance.cpp
// Author: Krushna Sanjay Sharma
////////////////////////////////////////////////////////////////////////////////

#include "FaceAwareDistance.h"
#include <iostream>
#include <cmath>

namespace cbir {

FaceAwareDistance::FaceAwareDistance() {
}

double FaceAwareDistance::compute(const cv::Mat& features1, 
                                 const cv::Mat& features2) {
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Features not compatible" << std::endl;
        return -1.0;
    }
    
    // Determine feature type by dimension
    if (features1.cols == 1029) {
        // Face mode
        return computeFaceDistance(features1, features2);
    } else if (features1.cols == 1024) {
        // Non-face mode (ProductMatcher)
        return computeNonFaceDistance(features1, features2);
    } else {
        std::cerr << "Error: Unknown feature dimension: " << features1.cols << std::endl;
        return -1.0;
    }
}

std::string FaceAwareDistance::getMetricName() const {
    return "FaceAwareDistance";
}

double FaceAwareDistance::computeFaceDistance(const cv::Mat& f1, const cv::Mat& f2) const {
    // Feature structure: [dnn(512), count(1), color(512), spatial(4)]
    
    // DNN distance (cosine) - 50% weight
    double dnnDist = computeCosineDistance(f1, f2, 0, 512);
    
    // Face count distance (absolute difference) - 10% weight
    double countDist = std::abs(f1.at<float>(0, 512) - f2.at<float>(0, 512));
    
    // Face color distance (histogram intersection) - 30% weight
    double colorDist = 1.0 - computeHistogramIntersection(f1, f2, 513, 1025);
    
    // Spatial layout distance (Euclidean) - 10% weight
    double spatialDist = computeEuclideanDistance(f1, f2, 1025, 1029);
    
    // Weighted combination
    double combined = 0.5 * dnnDist + 0.1 * countDist + 0.3 * colorDist + 0.1 * spatialDist;
    
    return combined;
}

double FaceAwareDistance::computeNonFaceDistance(const cv::Mat& f1, const cv::Mat& f2) const {
    // Feature structure: [dnn(512), color(512)]
    // Use ProductMatcher weighting: 85% DNN, 15% color
    
    double dnnDist = computeCosineDistance(f1, f2, 0, 512);
    double colorDist = 1.0 - computeHistogramIntersection(f1, f2, 512, 1024);
    
    return 0.85 * dnnDist + 0.15 * colorDist;
}

double FaceAwareDistance::computeCosineDistance(const cv::Mat& f1, const cv::Mat& f2,
                                               int start, int end) const {
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    
    for (int i = start; i < end; i++) {
        float v1 = f1.at<float>(0, i);
        float v2 = f2.at<float>(0, i);
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < 1e-10 || norm2 < 1e-10) {
        return 1.0;
    }
    
    double cosine = dot / (norm1 * norm2);
    cosine = std::max(-1.0, std::min(1.0, cosine));
    
    return 1.0 - cosine;
}

double FaceAwareDistance::computeHistogramIntersection(const cv::Mat& f1, const cv::Mat& f2,
                                                      int start, int end) const {
    double intersection = 0.0;
    
    for (int i = start; i < end; i++) {
        intersection += std::min(f1.at<float>(0, i), f2.at<float>(0, i));
    }
    
    return intersection;
}

double FaceAwareDistance::computeEuclideanDistance(const cv::Mat& f1, const cv::Mat& f2,
                                                  int start, int end) const {
    double sumSq = 0.0;
    
    for (int i = start; i < end; i++) {
        float diff = f1.at<float>(0, i) - f2.at<float>(0, i);
        sumSq += diff * diff;
    }
    
    return std::sqrt(sumSq);
}

} // namespace cbir
