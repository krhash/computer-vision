/*
  Author: Krushna Sanjay Sharma
  Date: January 25, 2026
  Purpose: Video cartoonization using bilateral filtering and DoG edge detection
           Based on Winnemöller et al. (2006) "Real-time video abstraction"
*/

#ifndef CARTOON_VIDEO_HPP
#define CARTOON_VIDEO_HPP

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class CartoonVideo
 * @brief Implements real-time video abstraction/cartoonization
 * 
 * Based on the paper:
 * Winnemöller, H., Olsen, S. C., & Gooch, B. (2006). 
 * "Real-time video abstraction." ACM Transactions on Graphics, 25(3), 1221-1226.
 * 
 * The algorithm combines:
 * 1. Bilateral filtering for edge-preserving smoothing
 * 2. Difference-of-Gaussians (DoG) for edge detection
 * 3. Color quantization for cartoon appearance
 * 4. Temporal coherence for stable video output
 */
class CartoonVideo {
public:
    /**
     * @brief Constructor with default parameters
     */
    CartoonVideo();
    
    /**
     * @brief Constructor with custom parameters
     * 
     * @param bilateralD Diameter of bilateral filter neighborhood
     * @param bilateralSigmaColor Filter sigma in color space
     * @param bilateralSigmaSpace Filter sigma in coordinate space
     * @param dogSigma1 Smaller Gaussian sigma for DoG
     * @param dogSigma2 Larger Gaussian sigma for DoG
     * @param dogThreshold Edge detection threshold
     * @param quantizeLevels Number of color quantization levels
     */
    CartoonVideo(int bilateralD, double bilateralSigmaColor, 
                 double bilateralSigmaSpace, double dogSigma1, 
                 double dogSigma2, double dogThreshold, int quantizeLevels);
    
    /**
     * @brief Process a single frame to create cartoon effect
     * 
     * Algorithm steps:
     * 1. Apply bilateral filter for edge-preserving smoothing
     * 2. Compute Difference-of-Gaussians for edge detection
     * 3. Quantize colors into discrete levels
     * 4. Combine edges with quantized colors
     * 5. Apply temporal smoothing (optional)
     * 
     * @param src Input color frame (CV_8UC3)
     * @param dst Output cartoonized frame (CV_8UC3)
     * @return 0 on success, -1 on error
     */
    int processFrame(const cv::Mat &src, cv::Mat &dst);
    
    /**
     * @brief Apply bilateral filter for edge-preserving smoothing
     * 
     * Bilateral filter smooths flat regions while preserving edges.
     * Uses both spatial and color similarity for filtering.
     * 
     * @param src Input image
     * @param dst Output smoothed image
     */
    void applyBilateralFilter(const cv::Mat &src, cv::Mat &dst);
    
    /**
     * @brief Detect edges using Difference-of-Gaussians (DoG)
     * 
     * DoG approximates Laplacian-of-Gaussian for edge detection.
     * Computed as: DoG = Gaussian(σ1) - Gaussian(σ2)
     * where σ2 > σ1
     * 
     * @param src Input image
     * @param edges Output edge map (binary)
     */
    void detectEdgesDoG(const cv::Mat &src, cv::Mat &edges);
    
    /**
     * @brief Quantize colors into discrete levels
     * 
     * Reduces number of colors to create posterization effect.
     * Formula: quantized = (value / bucketSize) * bucketSize
     * 
     * @param src Input image
     * @param dst Output quantized image
     */
    void quantizeColors(const cv::Mat &src, cv::Mat &dst);
    
    /**
     * @brief Combine edges with quantized image
     * 
     * Darkens pixels at edge locations to create cartoon outlines.
     * 
     * @param quantized Quantized color image
     * @param edges Binary edge map
     * @param dst Output cartoon image
     */
    void combineEdgesAndColors(const cv::Mat &quantized, 
                               const cv::Mat &edges, cv::Mat &dst);
    
    /**
     * @brief Apply temporal smoothing for video coherence
     * 
     * Blends current frame with previous frame to reduce flickering.
     * Uses exponential moving average.
     * 
     * @param current Current frame
     * @param dst Output temporally smoothed frame
     */
    void applyTemporalSmoothing(const cv::Mat &current, cv::Mat &dst);
    
    /**
     * @brief Reset temporal buffer (call when video restarts)
     */
    void resetTemporalBuffer();
    
    /**
     * @brief Set bilateral filter parameters
     */
    void setBilateralParams(int d, double sigmaColor, double sigmaSpace);
    
    /**
     * @brief Set DoG edge detection parameters
     */
    void setDoGParams(double sigma1, double sigma2, double threshold);
    
    /**
     * @brief Set color quantization levels
     */
    void setQuantizeLevels(int levels);
    
    /**
     * @brief Enable/disable temporal smoothing
     */
    void setTemporalSmoothing(bool enable);
    
    /**
     * @brief Set temporal smoothing strength
     * 
     * @param alpha Blending factor [0.0-1.0], higher = more smoothing
     */
    void setTemporalAlpha(double alpha);

private:
    // Bilateral filter parameters
    int bilateralD_;              ///< Diameter of pixel neighborhood
    double bilateralSigmaColor_;  ///< Filter sigma in color space
    double bilateralSigmaSpace_;  ///< Filter sigma in coordinate space
    
    // DoG edge detection parameters
    double dogSigma1_;            ///< Smaller Gaussian sigma
    double dogSigma2_;            ///< Larger Gaussian sigma
    double dogThreshold_;         ///< Edge detection threshold
    
    // Color quantization
    int quantizeLevels_;          ///< Number of color levels
    
    // Temporal coherence
    bool useTemporalSmoothing_;   ///< Enable temporal smoothing
    double temporalAlpha_;        ///< Temporal blending factor
    cv::Mat previousFrame_;       ///< Previous frame buffer
    bool hasFirstFrame_;          ///< First frame flag
};

#endif // CARTOON_VIDEO_HPP