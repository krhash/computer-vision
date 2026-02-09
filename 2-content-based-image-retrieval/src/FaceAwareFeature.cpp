////////////////////////////////////////////////////////////////////////////////
// FaceAwareFeature.cpp
// Author: Krushna Sanjay Sharma
// Description: Implementation of adaptive face-aware feature extraction.
//              Automatically switches between face-based and object-based
//              features depending on image content.
//
// Face Detection:
//   Uses OpenCV Haar Cascade classifier (same as your previous project)
//
// Feature Extraction Strategy:
//   WITH faces: [DNN, face_count, face_colors, spatial_layout]
//   WITHOUT faces: [DNN, center_color] (ProductMatcher)
//
// This demonstrates intelligent, content-aware CBIR that adapts to
// different image types automatically.
//
// Date: February 2026
////////////////////////////////////////////////////////////////////////////////

#include "FaceAwareFeature.h"
#include "Utils.h"
#include <iostream>

namespace cbir {

/**
 * Constructor
 */
FaceAwareFeature::FaceAwareFeature(const std::string& dnnCsvPath,
                                 const std::string& cascadePath)
    : dnnExtractor_(dnnCsvPath),
      productMatcher_(dnnCsvPath, 0.5, 8),
      lastHadFaces_(false),
      lastFaceCount_(0) {
    
    // Load Haar cascade for face detection
    if (!faceCascade_.load(cascadePath)) {
        std::cerr << "Warning: Failed to load Haar cascade: " << cascadePath << std::endl;
        std::cerr << "Trying default location..." << std::endl;
        
        // Try common locations
        if (!faceCascade_.load("haarcascade_frontalface_alt2.xml")) {
            std::cerr << "Error: Could not load face detector!" << std::endl;
            std::cerr << "Face detection will be disabled." << std::endl;
        }
    } else {
        std::cout << "Face detector loaded successfully" << std::endl;
    }
}

/**
 * Extract features (base class requirement)
 */
cv::Mat FaceAwareFeature::extractFeatures(const cv::Mat& image) {
    (void)image;
    std::cerr << "Warning: FaceAwareFeature requires filename." << std::endl;
    std::cerr << "Use extractFeaturesWithFilename() instead." << std::endl;
    return cv::Mat();
}

/**
 * Extract adaptive features based on face detection
 * 
 * Main feature extraction logic:
 *   1. Detect faces
 *   2. If faces found → extract face-based features
 *   3. If no faces → use ProductMatcher features
 */
cv::Mat FaceAwareFeature::extractFeaturesWithFilename(const cv::Mat& image, 
                                                     const std::string& filename) {
    if (!isValidImage(image)) {
        std::cerr << "Error: Invalid image for FaceAware extraction" << std::endl;
        return cv::Mat();
    }
    
    // Detect faces in image
    std::vector<cv::Rect> faces = detectFaces(image);
    
    // Update state
    lastFaceCount_ = static_cast<int>(faces.size());
    lastHadFaces_ = (lastFaceCount_ > 0);
    
    cv::Mat features;
    
    if (lastHadFaces_) {
        // Image contains faces - use face-based features
        features = extractFaceFeatures(image, faces, filename);
    } else {
        // No faces - use ProductMatcher features
        features = productMatcher_.extractFeaturesWithFilename(image, filename);
    }
    
    return features;
}

std::string FaceAwareFeature::getFeatureName() const {
    return "FaceAware_Adaptive";
}

int FaceAwareFeature::getFeatureDimension() const {
    // Maximum dimension (face mode)
    return 512 + 1 + 512 + 4;  // DNN + count + color + spatial = 1029
}

/**
 * Detect faces using Haar cascade
 * 
 * Uses OpenCV's cascade classifier (same approach as your video project)
 */
std::vector<cv::Rect> FaceAwareFeature::detectFaces(const cv::Mat& image) {
    std::vector<cv::Rect> faces;
    
    // Check if cascade is loaded
    if (faceCascade_.empty()) {
        return faces;  // Return empty if detector not loaded
    }
    
    // Convert to grayscale for detection
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Equalize histogram for better detection
    cv::equalizeHist(gray, gray);
    
    // Detect faces
    faceCascade_.detectMultiScale(
        gray,
        faces,
        1.1,        // Scale factor
        3,          // Min neighbors
        0,          // Flags
        cv::Size(30, 30)  // Min size
    );
    
    return faces;
}

/**
 * Extract face-based features
 * 
 * Feature components:
 *   1. DNN embeddings (512) - Semantic content
 *   2. Face count (1) - Number of people
 *   3. Face region colors (512) - Clothing/context colors around faces
 *   4. Spatial layout (4) - Where faces are positioned
 * 
 * Total: 1029 values
 */
cv::Mat FaceAwareFeature::extractFaceFeatures(const cv::Mat& image, 
                                             const std::vector<cv::Rect>& faces,
                                             const std::string& filename) {
    // Component 1: DNN embeddings
    cv::Mat dnnFeatures = dnnExtractor_.getFeaturesByFilename(filename);
    
    if (dnnFeatures.empty()) {
        std::cerr << "Error: No DNN features for " << filename << std::endl;
        return cv::Mat();
    }
    
    // Component 2: Face count (normalized to [0, 1])
    // Assume max 10 faces in an image for normalization
    float faceCountNorm = std::min(static_cast<float>(faces.size()), 10.0f) / 10.0f;
    cv::Mat faceCountFeature(1, 1, CV_32F);
    faceCountFeature.at<float>(0, 0) = faceCountNorm;
    
    // Component 3: Color histogram of face regions (context/clothing)
    cv::Mat faceRegionColor = computeFaceRegionColor(image, faces);
    
    // Component 4: Spatial layout of faces
    cv::Mat spatialLayout = computeFaceSpatialLayout(faces, image.cols, image.rows);
    
    // Concatenate all components
    int totalDim = dnnFeatures.cols + faceCountFeature.cols + 
                   faceRegionColor.cols + spatialLayout.cols;
    cv::Mat combinedFeatures(1, totalDim, CV_32F);
    
    int offset = 0;
    
    // Copy DNN features
    for (int i = 0; i < dnnFeatures.cols; i++) {
        combinedFeatures.at<float>(0, offset++) = dnnFeatures.at<float>(0, i);
    }
    
    // Copy face count
    combinedFeatures.at<float>(0, offset++) = faceCountFeature.at<float>(0, 0);
    
    // Copy face region colors
    for (int i = 0; i < faceRegionColor.cols; i++) {
        combinedFeatures.at<float>(0, offset++) = faceRegionColor.at<float>(0, i);
    }
    
    // Copy spatial layout
    for (int i = 0; i < spatialLayout.cols; i++) {
        combinedFeatures.at<float>(0, offset++) = spatialLayout.at<float>(0, i);
    }
    
    return combinedFeatures;
}

/**
 * Compute color histogram of face regions
 * 
 * Extracts expanded regions around detected faces (1.5× face size)
 * to capture clothing and context colors, not just skin tones.
 */
cv::Mat FaceAwareFeature::computeFaceRegionColor(const cv::Mat& image, 
                                                const std::vector<cv::Rect>& faces) {
    // Create mask for face regions (expanded)
    cv::Mat faceMask = cv::Mat::zeros(image.size(), CV_8U);
    
    for (const auto& face : faces) {
        // Expand face region by 50% to capture context
        int expandW = face.width / 4;
        int expandH = face.height / 4;
        
        cv::Rect expandedFace(
            std::max(0, face.x - expandW),
            std::max(0, face.y - expandH),
            std::min(image.cols - face.x + expandW, face.width + 2 * expandW),
            std::min(image.rows - face.y + expandH, face.height + 2 * expandH)
        );
        
        // Mark region in mask
        faceMask(expandedFace) = 255;
    }
    
    // Compute histogram only for face regions
    int dims[3] = {8, 8, 8};
    cv::Mat histogram = cv::Mat::zeros(3, dims, CV_32F);
    float binSize = 256.0f / 8;
    
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            // Only process pixels in face regions
            if (faceMask.at<uchar>(row, col) == 0) {
                continue;
            }
            
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            
            int binB = std::min(static_cast<int>(b / binSize), 7);
            int binG = std::min(static_cast<int>(g / binSize), 7);
            int binR = std::min(static_cast<int>(r / binSize), 7);
            
            histogram.at<float>(binR, binG, binB) += 1.0f;
        }
    }
    
    // Flatten and normalize
    cv::Mat flattened(1, 512, CV_32F);
    int idx = 0;
    const int* sizes = histogram.size.p;
    
    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                flattened.at<float>(0, idx++) = histogram.at<float>(i, j, k);
            }
        }
    }
    
    // Normalize
    double sum = 0.0;
    for (int i = 0; i < 512; i++) {
        sum += flattened.at<float>(0, i);
    }
    if (sum > 1e-10) {
        flattened /= sum;
    }
    
    return flattened;
}

/**
 * Compute spatial layout of detected faces
 * 
 * Returns 4D vector:
 *   [avg_x, avg_y, spread_x, spread_y]
 * 
 * Where:
 *   - avg_x, avg_y: Average position of faces (normalized [0, 1])
 *   - spread_x, spread_y: Spatial variance of faces (how spread out)
 * 
 * Use case: Matches group photos with similar layouts
 */
cv::Mat FaceAwareFeature::computeFaceSpatialLayout(const std::vector<cv::Rect>& faces,
                                                  int imageWidth, 
                                                  int imageHeight) {
    cv::Mat spatialFeature(1, 4, CV_32F);
    
    if (faces.empty()) {
        spatialFeature = cv::Mat::zeros(1, 4, CV_32F);
        return spatialFeature;
    }
    
    // Compute average face position (normalized)
    float avgX = 0.0f;
    float avgY = 0.0f;
    
    for (const auto& face : faces) {
        float centerX = (face.x + face.width / 2.0f) / imageWidth;
        float centerY = (face.y + face.height / 2.0f) / imageHeight;
        avgX += centerX;
        avgY += centerY;
    }
    
    avgX /= faces.size();
    avgY /= faces.size();
    
    // Compute spatial variance (spread)
    float varX = 0.0f;
    float varY = 0.0f;
    
    for (const auto& face : faces) {
        float centerX = (face.x + face.width / 2.0f) / imageWidth;
        float centerY = (face.y + face.height / 2.0f) / imageHeight;
        varX += (centerX - avgX) * (centerX - avgX);
        varY += (centerY - avgY) * (centerY - avgY);
    }
    
    varX = std::sqrt(varX / faces.size());
    varY = std::sqrt(varY / faces.size());
    
    // Store in feature vector
    spatialFeature.at<float>(0, 0) = avgX;
    spatialFeature.at<float>(0, 1) = avgY;
    spatialFeature.at<float>(0, 2) = varX;
    spatialFeature.at<float>(0, 3) = varY;
    
    return spatialFeature;
}

} // namespace cbir
