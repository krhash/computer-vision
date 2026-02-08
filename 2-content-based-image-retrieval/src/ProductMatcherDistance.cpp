////////////////////////////////////////////////////////////////////////////////
// ProductMatcherDistance.cpp
// Author: Krushna Sanjay Sharma
////////////////////////////////////////////////////////////////////////////////

#include "ProductMatcherDistance.h"
#include <iostream>
#include <cmath>

namespace cbir {

ProductMatcherDistance::ProductMatcherDistance()
    : dnnWeight_(0.85), colorWeight_(0.15) {
}

ProductMatcherDistance::ProductMatcherDistance(double dnnWeight, double colorWeight)
    : dnnWeight_(dnnWeight), colorWeight_(colorWeight) {
    normalizeWeights();
}

double ProductMatcherDistance::compute(const cv::Mat& features1, 
                                      const cv::Mat& features2) {
    if (!areCompatible(features1, features2)) {
        std::cerr << "Error: Features not compatible" << std::endl;
        return -1.0;
    }
    
    if (features1.cols != TOTAL_DIM) {
        std::cerr << "Error: Expected " << TOTAL_DIM << "D features, got " 
                  << features1.cols << std::endl;
        return -1.0;
    }
    
    // Compute DNN distance (cosine)
    double dnnDist = computeDNNDistance(features1, features2);
    
    // Compute color distance (histogram intersection)
    double colorDist = computeColorDistance(features1, features2);
    
    // Weighted combination
    double combinedDist = dnnWeight_ * dnnDist + colorWeight_ * colorDist;
    
    return combinedDist;
}

std::string ProductMatcherDistance::getMetricName() const {
    return "ProductMatcherDistance";
}

double ProductMatcherDistance::computeDNNDistance(const cv::Mat& f1, const cv::Mat& f2) const {
    // Cosine distance for DNN embeddings (bins 0-511)
    double norm1 = computeL2Norm(f1, 0, DNN_DIM);
    double norm2 = computeL2Norm(f2, 0, DNN_DIM);
    
    if (norm1 < 1e-10 || norm2 < 1e-10) {
        return 1.0;
    }
    
    double dotProd = computeDotProduct(f1, f2, 0, DNN_DIM);
    double cosine = dotProd / (norm1 * norm2);
    cosine = std::max(-1.0, std::min(1.0, cosine));
    
    return 1.0 - cosine;
}

double ProductMatcherDistance::computeColorDistance(const cv::Mat& f1, const cv::Mat& f2) const {
    // Histogram intersection for color (bins 512-1023)
    double intersection = 0.0;
    
    for (int i = DNN_DIM; i < TOTAL_DIM; i++) {
        float val1 = f1.at<float>(0, i);
        float val2 = f2.at<float>(0, i);
        intersection += std::min(val1, val2);
    }
    
    return 1.0 - intersection;
}

void ProductMatcherDistance::normalizeWeights() {
    double sum = dnnWeight_ + colorWeight_;
    if (std::abs(sum - 1.0) > 0.01) {
        dnnWeight_ /= sum;
        colorWeight_ /= sum;
    }
}

double ProductMatcherDistance::computeL2Norm(const cv::Mat& features, 
                                            int startIdx, int endIdx) const {
    double sumSquares = 0.0;
    for (int i = startIdx; i < endIdx; i++) {
        float val = features.at<float>(0, i);
        sumSquares += val * val;
    }
    return std::sqrt(sumSquares);
}

double ProductMatcherDistance::computeDotProduct(const cv::Mat& f1, const cv::Mat& f2, 
                                                int startIdx, int endIdx) const {
    double dotProd = 0.0;
    for (int i = startIdx; i < endIdx; i++) {
        dotProd += f1.at<float>(0, i) * f2.at<float>(0, i);
    }
    return dotProd;
}

} // namespace cbir
