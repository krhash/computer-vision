/*
  Author: Krushna Sanjay Sharma
  Date: January 24, 2026
  Purpose: Implementation of image filtering and manipulation functions.
*/

#define _USE_MATH_DEFINES
#include "filters.hpp"
#include <iostream>
#include <cmath>
#include <ctime>

// Task 4: Custom greyscale conversion
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Check if source image is valid
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Invalid source image for greyscale conversion" << std::endl;
        return -1;
    }
    
    // Create destination image with same size and type as source
    dst.create(src.size(), src.type());
    
    // Custom greyscale algorithm: Channel Difference Method
    // This creates a unique greyscale based on color contrast
    // Formula: grey = |R - B| + G/2
    // This emphasizes color differences and gives a distinct look
    
    // Iterate through each row using row pointers (efficient)
    for (int row = 0; row < src.rows; row++) {
        // Get pointers to current row in source and destination
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        // Iterate through each pixel in the row
        for (int col = 0; col < src.cols; col++) {
            // Get BGR values from source pixel
            int blue = srcRow[col][0];
            int green = srcRow[col][1];
            int red = srcRow[col][2];
            
            // Custom greyscale calculation
            // Use channel difference
            // Emphasizes edges and color boundaries
            int diff = std::abs(red - blue);
            int grey = diff + (green / 2);
            
            // Clip to valid range [0, 255]
            if (grey > 255) grey = 255;
            
            // Set all three channels to the same value (greyscale)
            uchar greyValue = static_cast<uchar>(grey);
            dstRow[col][0] = greyValue;  // Blue
            dstRow[col][1] = greyValue;  // Green
            dstRow[col][2] = greyValue;  // Red
        }
    }
    
    return 0;
}

// Task 5: Sepia tone filter
int sepiaTone(cv::Mat &src, cv::Mat &dst, bool applyVignetting) {
    // Create destination image if needed
    dst.create(src.size(), src.type());
    
    // Check if source is valid 3-channel color image
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Pre-calculate vignetting parameters if enabled
    float centerX = src.cols / 2.0f;
    float centerY = src.rows / 2.0f;
    float maxDist = sqrt(centerX * centerX + centerY * centerY);
    float vignetteStrength = 1.2f;  // Adjustable strength
    
    // Iterate through each row using row pointers
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        // Pre-calculate y distance for vignetting (optimization)
        float dy = row - centerY;
        
        for (int col = 0; col < src.cols; col++) {
            // Get original BGR values (use float for precision)
            float blue = srcRow[col][0];
            float green = srcRow[col][1];
            float red = srcRow[col][2];
            
            // Apply sepia tone transformation using original values
            // IMPORTANT: Use original R, G, B for all calculations
            float newRed   = 0.393f * red + 0.769f * green + 0.189f * blue;
            float newGreen = 0.349f * red + 0.686f * green + 0.168f * blue;
            float newBlue  = 0.272f * red + 0.534f * green + 0.131f * blue;
            
            // Clamp sepia values to [0, 255]
            newRed   = std::min(newRed, 255.0f);
            newGreen = std::min(newGreen, 255.0f);
            newBlue  = std::min(newBlue, 255.0f);
            
            // Apply vignetting if enabled
            if (applyVignetting) {
                float dx = col - centerX;
                
                // Calculate distance from center
                float dist = sqrt(dx * dx + dy * dy);
                
                // Normalize distance to [0, 1]
                float d = dist / maxDist;
                
                // Quadratic falloff: vignette factor from 1.0 (center) to darker (edges)
                float vignette = 1.0f - vignetteStrength * d * d;
                vignette = std::max(vignette, 0.0f);
                
                // Apply vignetting
                newRed   *= vignette;
                newGreen *= vignette;
                newBlue  *= vignette;
            }
            
            // Write to destination (no need to clamp again, vignetting only darkens)
            dstRow[col][0] = static_cast<uchar>(newBlue);
            dstRow[col][1] = static_cast<uchar>(newGreen);
            dstRow[col][2] = static_cast<uchar>(newRed);
        }
    }
    
    return 0;
}

// Task 6: 5x5 Gaussian blur (naive implementation)
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // Check if source is valid 3-channel color image
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Create destination image (don't copy source - we'll handle borders differently)
    dst.create(src.size(), src.type());
    
    // 5x5 Gaussian kernel (integer approximation)
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    
    // Process ALL pixels, handling borders by clamping coordinates
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            int weightSum = 0;
            
            // Apply 5x5 kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int ny = row + ky;
                    int nx = col + kx;
                    
                    // Clamp to image boundaries
                    ny = std::max(0, std::min(ny, src.rows - 1));
                    nx = std::max(0, std::min(nx, src.cols - 1));
                    
                    cv::Vec3b pixel = src.at<cv::Vec3b>(ny, nx);
                    int weight = kernel[ky + 2][kx + 2];
                    
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                    weightSum += weight;
                }
            }
            
            // Divide by actual sum of weights used
            dst.at<cv::Vec3b>(row, col)[0] = static_cast<uchar>(sumB / weightSum);
            dst.at<cv::Vec3b>(row, col)[1] = static_cast<uchar>(sumG / weightSum);
            dst.at<cv::Vec3b>(row, col)[2] = static_cast<uchar>(sumR / weightSum);
        }
    }
    
    return 0;
}

// Task 6: 5x5 Gaussian blur (optimized with separable filters)
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // 1D Gaussian kernel [1 2 4 2 1], sum = 10
    int kernel1D[5] = {1, 2, 4, 2, 1};
    
    // Create temporary image with 16-bit signed values
    cv::Mat temp(src.size(), CV_16SC3);
    
    // First pass: Horizontal blur
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3s* tempRow = temp.ptr<cv::Vec3s>(row);
        
        for (int col = 0; col < src.cols; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            int weightSum = 0;
            
            // Apply 1x5 horizontal kernel with clamping
            for (int k = -2; k <= 2; k++) {
                int nx = col + k;
                // Clamp to boundaries
                nx = std::max(0, std::min(nx, src.cols - 1));
                
                cv::Vec3b pixel = srcRow[nx];
                int weight = kernel1D[k + 2];
                
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
                weightSum += weight;
            }
            
            // Store normalized values (divide to keep in reasonable range)
            tempRow[col][0] = static_cast<short>(sumB / weightSum);
            tempRow[col][1] = static_cast<short>(sumG / weightSum);
            tempRow[col][2] = static_cast<short>(sumR / weightSum);
        }
    }
    
    // Create destination
    dst.create(src.size(), src.type());
    
    // Second pass: Vertical blur
    for (int row = 0; row < temp.rows; row++) {
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < temp.cols; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            int weightSum = 0;
            
            // Apply 5x1 vertical kernel with clamping
            for (int k = -2; k <= 2; k++) {
                int ny = row + k;
                // Clamp to boundaries
                ny = std::max(0, std::min(ny, temp.rows - 1));
                
                const cv::Vec3s* tempRowK = temp.ptr<cv::Vec3s>(ny);
                int weight = kernel1D[k + 2];
                
                sumB += tempRowK[col][0] * weight;
                sumG += tempRowK[col][1] * weight;
                sumR += tempRowK[col][2] * weight;
                weightSum += weight;
            }
            
            // Normalize by weight sum
            dstRow[col][0] = static_cast<uchar>(sumB / weightSum);
            dstRow[col][1] = static_cast<uchar>(sumG / weightSum);
            dstRow[col][2] = static_cast<uchar>(sumR / weightSum);
        }
    }
    
    return 0;
}

// Task 7: Sobel X filter (detects vertical edges)
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    // Check if source is valid 3-channel color image
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Create destination as 16-bit signed 3-channel image
    dst.create(src.size(), CV_16SC3);
    dst.setTo(0);  // Initialize to zero
    
    // Sobel X is separable:
    // Horizontal derivative: [-1, 0, 1]
    // Vertical smoothing: [1, 2, 1]
    
    int horizKernel[3] = {-1, 0, 1};   // Derivative
    int vertKernel[3] = {1, 2, 1};     // Smoothing, sum = 4
    
    // Create temporary image for intermediate result (after horizontal pass)
    cv::Mat temp(src.size(), CV_16SC3);
    temp.setTo(0);
    
    // First pass: Apply horizontal derivative [-1, 0, 1]
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3s* tempRow = temp.ptr<cv::Vec3s>(row);
        
        for (int col = 1; col < src.cols - 1; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            
            for (int k = -1; k <= 1; k++) {
                cv::Vec3b pixel = srcRow[col + k];
                int weight = horizKernel[k + 1];
                
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }
            
            // Store signed values (can be negative)
            tempRow[col][0] = static_cast<short>(sumB);
            tempRow[col][1] = static_cast<short>(sumG);
            tempRow[col][2] = static_cast<short>(sumR);
        }
    }
    
    // Second pass: Apply vertical smoothing [1, 2, 1]
    for (int row = 1; row < temp.rows - 1; row++) {
        cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(row);
        
        for (int col = 1; col < temp.cols - 1; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            
            for (int k = -1; k <= 1; k++) {
                const cv::Vec3s* tempRowK = temp.ptr<cv::Vec3s>(row + k);
                int weight = vertKernel[k + 1];
                
                sumB += tempRowK[col][0] * weight;
                sumG += tempRowK[col][1] * weight;
                sumR += tempRowK[col][2] * weight;
            }
            
            // Divide by vertical kernel sum (4)
            dstRow[col][0] = static_cast<short>(sumB / 4);
            dstRow[col][1] = static_cast<short>(sumG / 4);
            dstRow[col][2] = static_cast<short>(sumR / 4);
        }
    }
    
    return 0;
}

// Task 7: Sobel Y filter (detects horizontal edges)
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    // Check if source is valid 3-channel color image
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Create destination as 16-bit signed 3-channel image
    dst.create(src.size(), CV_16SC3);
    dst.setTo(0);  // Initialize to zero
    
    // Sobel Y is separable:
    // Horizontal smoothing: [1, 2, 1]
    // Vertical derivative: [1, 0, -1] (positive up means [1, 0, -1])
    
    int horizKernel[3] = {1, 2, 1};    // Smoothing, sum = 4
    int vertKernel[3] = {1, 0, -1};    // Derivative (positive up)
    
    // Create temporary image for intermediate result
    cv::Mat temp(src.size(), CV_16SC3);
    temp.setTo(0);
    
    // First pass: Apply horizontal smoothing [1, 2, 1]
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3s* tempRow = temp.ptr<cv::Vec3s>(row);
        
        for (int col = 1; col < src.cols - 1; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            
            for (int k = -1; k <= 1; k++) {
                cv::Vec3b pixel = srcRow[col + k];
                int weight = horizKernel[k + 1];
                
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }
            
            // Divide by horizontal kernel sum (4)
            tempRow[col][0] = static_cast<short>(sumB / 4);
            tempRow[col][1] = static_cast<short>(sumG / 4);
            tempRow[col][2] = static_cast<short>(sumR / 4);
        }
    }
    
    // Second pass: Apply vertical derivative [1, 0, -1]
    for (int row = 1; row < temp.rows - 1; row++) {
        cv::Vec3s* dstRow = dst.ptr<cv::Vec3s>(row);
        
        for (int col = 1; col < temp.cols - 1; col++) {
            int sumB = 0, sumG = 0, sumR = 0;
            
            for (int k = -1; k <= 1; k++) {
                const cv::Vec3s* tempRowK = temp.ptr<cv::Vec3s>(row + k);
                int weight = vertKernel[k + 1];
                
                sumB += tempRowK[col][0] * weight;
                sumG += tempRowK[col][1] * weight;
                sumR += tempRowK[col][2] * weight;
            }
            
            // Store signed values (no division for derivative)
            dstRow[col][0] = static_cast<short>(sumB);
            dstRow[col][1] = static_cast<short>(sumG);
            dstRow[col][2] = static_cast<short>(sumR);
        }
    }
    
    return 0;
}

// Task 8: Gradient magnitude from Sobel X and Y
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Check if inputs are valid 3-channel signed short images
    if (sx.empty() || sy.empty() || sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        std::cerr << "Error: Inputs must be 3-channel signed short (CV_16SC3) images" << std::endl;
        return -1;
    }
    
    // Check if dimensions match
    if (sx.size() != sy.size()) {
        std::cerr << "Error: Sobel X and Y images must have same dimensions" << std::endl;
        return -1;
    }
    
    // Create destination as 8-bit unsigned 3-channel image
    dst.create(sx.size(), CV_8UC3);
    
    // Calculate magnitude for each pixel: sqrt(sx^2 + sy^2)
    for (int row = 0; row < sx.rows; row++) {
        const cv::Vec3s* sxRow = sx.ptr<cv::Vec3s>(row);
        const cv::Vec3s* syRow = sy.ptr<cv::Vec3s>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < sx.cols; col++) {
            // Calculate magnitude for each color channel separately
            for (int c = 0; c < 3; c++) {
                // Get signed short values
                short sxVal = sxRow[col][c];
                short syVal = syRow[col][c];
                
                // Calculate magnitude: sqrt(sx^2 + sy^2)
                double mag = sqrt(static_cast<double>(sxVal * sxVal + syVal * syVal));
                
                // Clamp to [0, 255] and convert to uchar
                mag = std::min(255.0, std::max(0.0, mag));
                dstRow[col][c] = static_cast<uchar>(mag);
            }
        }
    }
    
    return 0;
}

// Task 9: Blur and quantize
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    // Check if source is valid 3-channel color image
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Check if levels is valid
    if (levels <= 0 || levels > 255) {
        std::cerr << "Error: levels must be between 1 and 255" << std::endl;
        return -1;
    }
    
    // Step 1: Blur the image using separable blur filter
    cv::Mat blurred;
    if (blur5x5_2(src, blurred) != 0) {
        std::cerr << "Error: Blur operation failed" << std::endl;
        return -1;
    }
    
    // Step 2: Quantize the blurred image
    dst.create(src.size(), src.type());
    
    // Calculate bucket size
    int bucketSize = 255 / levels;
    
    // Quantize each pixel
    for (int row = 0; row < blurred.rows; row++) {
        const cv::Vec3b* blurredRow = blurred.ptr<cv::Vec3b>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < blurred.cols; col++) {
            // Quantize each color channel
            for (int c = 0; c < 3; c++) {
                int value = blurredRow[col][c];
                
                // Quantize: xt = x / bucketSize, then xf = xt * bucketSize
                int xt = value / bucketSize;
                int xf = xt * bucketSize;
                
                dstRow[col][c] = static_cast<uchar>(xf);
            }
        }
    }
    
    return 0;
}

// Task 11/12: Depth-based fog effect
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, float intensity) {
    if (src.empty() || depth.empty() || src.size() != depth.size()) {
        std::cerr << "Error: Invalid input for depth fog" << std::endl;
        return -1;
    }
    
    dst.create(src.size(), src.type());
    
    // Fog color (light bluish-gray for atmospheric effect)
    cv::Vec3b fogColor(220, 220, 200);
    
    // Apply fog based on depth using row pointers
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        const uchar* depthRow = depth.ptr<uchar>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < src.cols; col++) {
            // Depth value: 0=close, 255=far
            float depthNorm = depthRow[col] / 255.0f;
            
            // Exponential fog: distant objects get more fog
            // fogFactor increases with distance
            float fogFactor = 1.0f - exp(-intensity * depthNorm * depthNorm);
            fogFactor = std::max(0.0f, std::min(1.0f, fogFactor));
            
            // Blend original pixel with fog color
            for (int c = 0; c < 3; c++) {
                dstRow[col][c] = static_cast<uchar>(
                    srcRow[col][c] * (1.0f - fogFactor) + 
                    fogColor[c] * fogFactor
                );
            }
        }
    }
    
    return 0;
}

// Task 12: Emboss Effect using Sobel gradients
// Creates a 3D raised appearance by taking dot product of gradients with light direction
int embossEffect(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Get Sobel X and Y gradients
    cv::Mat sobelX, sobelY;
    if (sobelX3x3(src, sobelX) != 0 || sobelY3x3(src, sobelY) != 0) {
        std::cerr << "Error: Sobel filters failed in emboss" << std::endl;
        return -1;
    }
    
    dst.create(src.size(), src.type());
    
    // Light direction vector (normalized 45-degree angle from top-left)
    // (0.7071, 0.7071) = (1/sqrt(2), 1/sqrt(2))
    const float lightX = 0.7071f;
    const float lightY = 0.7071f;
    
    // Emboss strength
    const float strength = 1.5f;
    
    // Process each pixel
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3s* sxRow = sobelX.ptr<cv::Vec3s>(row);
        const cv::Vec3s* syRow = sobelY.ptr<cv::Vec3s>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < src.cols; col++) {
            for (int c = 0; c < 3; c++) {
                // Get gradient values
                float gx = static_cast<float>(sxRow[col][c]);
                float gy = static_cast<float>(syRow[col][c]);
                
                // Dot product with light direction
                // This gives us how much the surface faces the light
                float dot = (gx * lightX + gy * lightY) * strength;
                
                // Add to neutral gray (128) to create emboss effect
                float value = 128.0f + dot;
                
                // Clamp to valid range
                value = std::max(0.0f, std::min(255.0f, value));
                dstRow[col][c] = static_cast<uchar>(value);
            }
        }
    }
    
    return 0;
}

// Task 12: Negative/Inverse Effect
// Simple pixel-wise inversion: output = 255 - input
int negativeEffect(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        std::cerr << "Error: Source image is empty" << std::endl;
        return -1;
    }
    
    dst.create(src.size(), src.type());
    
    // Process each pixel using row pointers for efficiency
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < src.cols; col++) {
            // Invert each channel: 255 - value
            dstRow[col][0] = 255 - srcRow[col][0];  // Blue
            dstRow[col][1] = 255 - srcRow[col][1];  // Green
            dstRow[col][2] = 255 - srcRow[col][2];  // Red
        }
    }
    
    return 0;
}

// Task 12: Face Highlight Effect
// Keeps detected faces in color while rest of image is grayscale
// faces: vector of face rectangles from face detection
int faceHighlight(cv::Mat &src, cv::Mat &dst, const std::vector<cv::Rect> &faces) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // First convert entire image to grayscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    
    // Convert grayscale back to 3-channel for consistent output
    cv::cvtColor(grey, dst, cv::COLOR_GRAY2BGR);
    
    // If no faces detected, return grayscale image
    if (faces.empty()) {
        return 0;
    }
    
    // Create a mask for face regions with smooth edges
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    
    for (const cv::Rect& face : faces) {
        // Expand face region slightly for better coverage
        int expandX = face.width / 8;
        int expandY = face.height / 8;
        
        cv::Rect expanded(
            std::max(0, face.x - expandX),
            std::max(0, face.y - expandY),
            std::min(src.cols - face.x + expandX, face.width + 2 * expandX),
            std::min(src.rows - face.y + expandY, face.height + 2 * expandY)
        );
        
        // Draw filled ellipse for face region (more natural than rectangle)
        cv::Point center(face.x + face.width / 2, face.y + face.height / 2);
        cv::Size axes(face.width / 2 + expandX, face.height / 2 + expandY);
        cv::ellipse(mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
    }
    
    // Blur the mask for smooth transition
    cv::GaussianBlur(mask, mask, cv::Size(31, 31), 15);
    
    // Blend color face with grayscale background using the mask
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        const uchar* maskRow = mask.ptr<uchar>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < src.cols; col++) {
            float alpha = maskRow[col] / 255.0f;
            
            // Blend: result = alpha * color + (1 - alpha) * grayscale
            for (int c = 0; c < 3; c++) {
                dstRow[col][c] = static_cast<uchar>(
                    alpha * srcRow[col][c] + (1.0f - alpha) * dstRow[col][c]
                );
            }
        }
    }
    
    return 0;
}

// Task 12: Cartoon Effect (bonus)
// Combines edge detection with color quantization for cartoon look
int cartoonEffect(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: Source must be a 3-channel color image" << std::endl;
        return -1;
    }
    
    // Step 1: Get edges using gradient magnitude
    cv::Mat sobelX, sobelY, edges;
    if (sobelX3x3(src, sobelX) != 0 || sobelY3x3(src, sobelY) != 0) {
        return -1;
    }
    if (magnitude(sobelX, sobelY, edges) != 0) {
        return -1;
    }
    
    // Convert edges to grayscale
    cv::Mat edgesGray;
    cv::cvtColor(edges, edgesGray, cv::COLOR_BGR2GRAY);
    
    // Threshold edges to get strong edges only
    cv::Mat edgeMask;
    cv::threshold(edgesGray, edgeMask, 30, 255, cv::THRESH_BINARY);
    
    // Dilate edges slightly
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(edgeMask, edgeMask, kernel);
    
    // Step 2: Quantize colors
    cv::Mat quantized;
    if (blurQuantize(src, quantized, 8) != 0) {
        return -1;
    }
    
    // Step 3: Combine - darken edges on quantized image
    dst = quantized.clone();
    
    for (int row = 0; row < dst.rows; row++) {
        const uchar* edgeRow = edgeMask.ptr<uchar>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < dst.cols; col++) {
            if (edgeRow[col] > 0) {
                // Make edges dark (cartoon outline)
                dstRow[col][0] = static_cast<uchar>(dstRow[col][0] * 0.3);
                dstRow[col][1] = static_cast<uchar>(dstRow[col][1] * 0.3);
                dstRow[col][2] = static_cast<uchar>(dstRow[col][2] * 0.3);
            }
        }
    }
    
    return 0;
}

// Extension: Depth-based focus effect (portrait mode)
// Blurs areas that are far from the focus depth
int depthFocus(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, int focusDepth, int focusRange) {
    if (src.empty() || depth.empty() || src.size() != depth.size()) {
        std::cerr << "Error: Invalid input for depth focus" << std::endl;
        return -1;
    }
    
    // Create heavily blurred version for background
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(31, 31), 15);
    
    dst.create(src.size(), src.type());
    
    // Blend based on depth distance from focus point
    for (int row = 0; row < src.rows; row++) {
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        const cv::Vec3b* blurRow = blurred.ptr<cv::Vec3b>(row);
        const uchar* depthRow = depth.ptr<uchar>(row);
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 0; col < src.cols; col++) {
            // Calculate distance from focus depth
            int depthDiff = std::abs(static_cast<int>(depthRow[col]) - focusDepth);
            
            // Calculate blur amount (0 = sharp, 1 = full blur)
            float blurAmount = 0.0f;
            if (depthDiff > focusRange) {
                // Smooth transition beyond focus range
                blurAmount = std::min(1.0f, (depthDiff - focusRange) / 80.0f);
            }
            
            // Blend sharp and blurred
            for (int c = 0; c < 3; c++) {
                dstRow[col][c] = static_cast<uchar>(
                    srcRow[col][c] * (1.0f - blurAmount) + 
                    blurRow[col][c] * blurAmount
                );
            }
        }
    }
    
    return 0;
}

// Helper function for bilinear interpolation
// Provides smooth pixel values for non-integer coordinates
static cv::Vec3b bilinearInterpolate(const cv::Mat &src, float x, float y) {
    // Bounds check
    if (x < 0 || x >= src.cols - 1 || y < 0 || y >= src.rows - 1) {
        return cv::Vec3b(0, 0, 0);
    }
    
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    float fx = x - x0;
    float fy = y - y0;
    
    // Get four neighboring pixels
    const cv::Vec3b& p00 = src.at<cv::Vec3b>(y0, x0);
    const cv::Vec3b& p01 = src.at<cv::Vec3b>(y0, x0 + 1);
    const cv::Vec3b& p10 = src.at<cv::Vec3b>(y0 + 1, x0);
    const cv::Vec3b& p11 = src.at<cv::Vec3b>(y0 + 1, x0 + 1);
    
    cv::Vec3b result;
    for (int c = 0; c < 3; c++) {
        float val = (1 - fx) * (1 - fy) * p00[c] +
                    fx * (1 - fy) * p01[c] +
                    (1 - fx) * fy * p10[c] +
                    fx * fy * p11[c];
        result[c] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, val)));
    }
    
    return result;
}

/*
  Bulge Effect Implementation
  
  Mathematical basis:
  - Convert each pixel to polar coordinates relative to center
  - Apply power function to radius: r' = r^strength
  - Convert back to Cartesian coordinates
  - For strength < 1: pixels move outward (bulge)
  - For strength > 1: pixels move inward (pinch)
*/
int bulgeEffect(cv::Mat &src, cv::Mat &dst, float strength) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }
    
    dst.create(src.size(), src.type());
    
    float cx = src.cols / 2.0f;
    float cy = src.rows / 2.0f;
    float radius = std::min(cx, cy) * 0.9f;
    
    // Invert strength: lower value = more bulge
    // Map 0.1-1.0 to exponent that creates bulge
    float exponent = 1.0f / (strength + 0.5f);
    
    for (int y = 0; y < src.rows; y++) {
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
        
        for (int x = 0; x < src.cols; x++) {
            float dx = x - cx;
            float dy = y - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            
            float srcX, srcY;
            
            if (dist < radius && dist > 0.001f) {
                float normalizedDist = dist / radius;
                
                // Bulge: sample from inner radius to spread outward
                float warpedDist = std::pow(normalizedDist, exponent);
                float scale = warpedDist / normalizedDist;
                
                srcX = cx + dx * scale;
                srcY = cy + dy * scale;
            } else {
                srcX = static_cast<float>(x);
                srcY = static_cast<float>(y);
            }
            
            // Bilinear interpolation
            if (srcX >= 0 && srcX < src.cols - 1 && srcY >= 0 && srcY < src.rows - 1) {
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                float fx = srcX - x0;
                float fy = srcY - y0;
                
                const cv::Vec3b& p00 = src.at<cv::Vec3b>(y0, x0);
                const cv::Vec3b& p01 = src.at<cv::Vec3b>(y0, x0 + 1);
                const cv::Vec3b& p10 = src.at<cv::Vec3b>(y0 + 1, x0);
                const cv::Vec3b& p11 = src.at<cv::Vec3b>(y0 + 1, x0 + 1);
                
                for (int c = 0; c < 3; c++) {
                    float val = (1-fx)*(1-fy)*p00[c] + fx*(1-fy)*p01[c] +
                               (1-fx)*fy*p10[c] + fx*fy*p11[c];
                    dstRow[x][c] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, val)));
                }
            } else {
                dstRow[x] = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    return 0;
}

/*
  Wave Effect Implementation
  
  Mathematical basis:
  - Apply sinusoidal displacement to each pixel
  - Horizontal wave: x' = x + amplitude * sin(frequency * y * 2π)
  - Vertical wave: y' = y + amplitude * sin(frequency * x * 2π)
  - Creates ripple pattern across image
*/
int waveEffect(cv::Mat &src, cv::Mat &dst, float amplitude, float frequency) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: waveEffect requires 3-channel image" << std::endl;
        return -1;
    }
    
    dst.create(src.size(), src.type());
    
    const float PI = 3.14159265358979f;
    
    for (int y = 0; y < src.rows; y++) {
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
        
        for (int x = 0; x < src.cols; x++) {
            // Sinusoidal displacement
            float offsetX = amplitude * std::sin(frequency * y * 2.0f * PI);
            float offsetY = amplitude * std::sin(frequency * x * 2.0f * PI);
            
            float srcX = x + offsetX;
            float srcY = y + offsetY;
            
            // Bounds check and sample
            if (srcX >= 0 && srcX < src.cols - 1 && srcY >= 0 && srcY < src.rows - 1) {
                dstRow[x] = bilinearInterpolate(src, srcX, srcY);
            } else {
                // Mirror at boundaries for smoother edges
                srcX = std::max(0.0f, std::min(static_cast<float>(src.cols - 1), srcX));
                srcY = std::max(0.0f, std::min(static_cast<float>(src.rows - 1), srcY));
                dstRow[x] = src.at<cv::Vec3b>(static_cast<int>(srcY), static_cast<int>(srcX));
            }
        }
    }
    
    return 0;
}

/*
  Swirl Effect Implementation
  
  Mathematical basis:
  - Convert to polar coordinates: (r, θ)
  - Add rotation that decreases with distance: θ' = θ + angle * (1 - r/maxRadius)
  - Convert back to Cartesian
  - Center pixels rotate most, edge pixels rotate least
  
  Reference: This is a classic image warping technique used in 
  photo editing software like Photoshop's Twirl filter
*/
int swirlEffect(cv::Mat &src, cv::Mat &dst, float angle) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: swirlEffect requires 3-channel image" << std::endl;
        return -1;
    }
    
    dst.create(src.size(), src.type());
    
    float cx = src.cols / 2.0f;
    float cy = src.rows / 2.0f;
    float maxRadius = std::sqrt(cx * cx + cy * cy);
    
    for (int y = 0; y < src.rows; y++) {
        cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
        
        for (int x = 0; x < src.cols; x++) {
            float dx = x - cx;
            float dy = y - cy;
            float dist = std::sqrt(dx * dx + dy * dy);
            
            // Rotation angle decreases with distance from center
            // This creates the swirl effect
            float theta = angle * (1.0f - dist / maxRadius);
            
            // Apply rotation transformation
            // x' = x*cos(θ) - y*sin(θ)
            // y' = x*sin(θ) + y*cos(θ)
            float cosTheta = std::cos(theta);
            float sinTheta = std::sin(theta);
            
            float srcX = cx + dx * cosTheta - dy * sinTheta;
            float srcY = cy + dx * sinTheta + dy * cosTheta;
            
            // Sample with bilinear interpolation
            if (srcX >= 0 && srcX < src.cols - 1 && srcY >= 0 && srcY < src.rows - 1) {
                dstRow[x] = bilinearInterpolate(src, srcX, srcY);
            } else {
                dstRow[x] = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    return 0;
}

/*
  Face Bulge Effect Implementation
  
  Creates a "big head" caricature effect on detected faces.
  - strength < 1.0: Enlarges/bulges the face outward
  - strength = 1.0: No effect
  - strength > 1.0: Shrinks/pinches the face inward
  
  The key fix: We need to INVERT the mapping direction.
  Instead of asking "where does this destination pixel come from?"
  with a shrinking factor, we use an EXPANDING factor.
*/
int faceBulgeEffect(cv::Mat &src, cv::Mat &dst, const std::vector<cv::Rect> &faces, float strength) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: faceBulgeEffect requires 3-channel image" << std::endl;
        return -1;
    }
    
    // Start with original image
    dst = src.clone();
    
    if (faces.empty()) {
        return 0;
    }
    
    // Invert strength for intuitive control: lower = more bulge
    // Map strength 0.1-1.0 to exponent 2.0-1.0 (inverted power)
    float exponent = 1.0f / (strength + 0.5f);  // strength 0.5 -> exp 1.33, strength 0.1 -> exp 1.67
    
    for (const cv::Rect& face : faces) {
        // Face center
        float cx = face.x + face.width / 2.0f;
        float cy = face.y + face.height / 2.0f;
        
        // Effect radius - covers the face
        float radius = std::max(face.width, face.height) * 0.75f;
        
        // Process region around face
        int pad = static_cast<int>(radius * 1.2f);
        int x1 = std::max(0, static_cast<int>(cx - radius - pad));
        int y1 = std::max(0, static_cast<int>(cy - radius - pad));
        int x2 = std::min(src.cols, static_cast<int>(cx + radius + pad));
        int y2 = std::min(src.rows, static_cast<int>(cy + radius + pad));
        
        for (int y = y1; y < y2; y++) {
            cv::Vec3b* dstRow = dst.ptr<cv::Vec3b>(y);
            
            for (int x = x1; x < x2; x++) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = std::sqrt(dx * dx + dy * dy);
                
                if (dist < radius && dist > 0.001f) {
                    // Normalize distance to [0, 1]
                    float normalizedDist = dist / radius;
                    
                    // Apply inverse power for bulge effect
                    // When exponent > 1: pixels from center spread outward (bulge)
                    // This makes the face appear larger
                    float warpedDist = std::pow(normalizedDist, exponent);
                    
                    // Calculate source coordinates (where to sample FROM)
                    // We sample from a SMALLER radius to create enlargement
                    float scale = warpedDist / normalizedDist;
                    
                    float srcX = cx + dx * scale;
                    float srcY = cy + dy * scale;
                    
                    // Bilinear interpolation
                    if (srcX >= 0 && srcX < src.cols - 1 && srcY >= 0 && srcY < src.rows - 1) {
                        int x0 = static_cast<int>(srcX);
                        int y0 = static_cast<int>(srcY);
                        float fx = srcX - x0;
                        float fy = srcY - y0;
                        
                        const cv::Vec3b& p00 = src.at<cv::Vec3b>(y0, x0);
                        const cv::Vec3b& p01 = src.at<cv::Vec3b>(y0, x0 + 1);
                        const cv::Vec3b& p10 = src.at<cv::Vec3b>(y0 + 1, x0);
                        const cv::Vec3b& p11 = src.at<cv::Vec3b>(y0 + 1, x0 + 1);
                        
                        for (int c = 0; c < 3; c++) {
                            float val = (1-fx)*(1-fy)*p00[c] + fx*(1-fy)*p01[c] +
                                       (1-fx)*fy*p10[c] + fx*fy*p11[c];
                            dstRow[x][c] = static_cast<uchar>(std::min(255.0f, std::max(0.0f, val)));
                        }
                    }
                }
            }
        }
    }
    
    return 0;
}

// Add this to filters.cpp

/*
  Sparkle Effect Implementation
  
  Creates animated sparkles that orbit around detected faces in 3D space.
  Uses depth sorting to draw sparkles in front of or behind the face.
  
  Mathematical basis:
  - Sparkles orbit in 3D: x = r*cos(θ), y = r*sin(θ)*cos(φ), z = r*sin(θ)*sin(φ)
  - z-depth determines drawing order (behind face vs in front)
  - Time-based animation updates angle θ
*/

/**
 * @brief Initializes sparkle particles with randomized properties for each face.
 *
 * This function creates a new set of sparkles for each detected face. Each sparkle
 * is given random properties to ensure varied motion and appearance. The sparkles
 * are initially positioned evenly around a circle, with randomization applied to
 * radius, phase (orbit tilt), size, speed, and color.
 *
 * The base radius is calculated from the face dimensions to ensure sparkles orbit
 * at an appropriate distance. Additional random variation (±30%) prevents sparkles
 * from appearing too uniform.
 *
 * @param sparkles Output vector, resized to match number of faces and filled with initialized sparkles
 * @param faces Input vector of face rectangles to create sparkles around
 * @param numSparkles Number of sparkles to create per face (default: 12)
 *
 * @implementation
 * - Resizes output vector to match number of faces
 * - For each face:
 *   - Calculates base orbital radius (60% of max face dimension)
 *   - Creates numSparkles evenly distributed around circle
 *   - Randomizes: radius (±30%), phase (0-2π), size (3-8px), speed (0.8-1.2x), color
 * - Color palette: Gold, White, Light Blue, Pink
 */
void initializeSparkles(std::vector<std::vector<Sparkle>> &sparkles, 
                       const std::vector<cv::Rect> &faces,
                       int numSparkles) {
    sparkles.resize(faces.size());
    
    for (size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++) {
        sparkles[faceIdx].clear();
        
        float faceRadius = std::max(faces[faceIdx].width, faces[faceIdx].height) * 0.6f;
        
        // Create sparkles evenly distributed around circle
        for (int i = 0; i < numSparkles; i++) {
            Sparkle s;
            s.angle = (2.0f * M_PI * i) / numSparkles;
            s.radius = faceRadius * (0.9f + 0.3f * (rand() % 100) / 100.0f);
            s.phase = (rand() % 100) / 100.0f * 2.0f * M_PI;
            s.size = 3.0f + (rand() % 5);
            s.speed = 0.8f + 0.4f * (rand() % 100) / 100.0f;
            
            // Random sparkle colors (gold, white, light blue, pink)
            int colorChoice = rand() % 4;
            switch (colorChoice) {
                case 0: s.color = cv::Vec3b(102, 204, 255); break; // Gold
                case 1: s.color = cv::Vec3b(255, 255, 255); break; // White
                case 2: s.color = cv::Vec3b(255, 200, 150); break; // Light blue
                case 3: s.color = cv::Vec3b(203, 192, 255); break; // Pink
            }
            
            sparkles[faceIdx].push_back(s);
        }
    }
}

/**
 * @brief Draws a single sparkle with glow effect at specified location.
 *
 * Creates a multi-layered glowing sparkle by drawing concentric circles with
 * decreasing alpha (transparency). The innermost layers are brightest, creating
 * a soft glow effect. Additionally draws a cross pattern at the center for
 * bright sparkles to enhance the "twinkling" appearance.
 *
 * The glow is achieved through alpha blending with the existing image pixels:
 * result = background * (1 - alpha) + sparkleColor * alpha
 *
 * @param img Image to draw sparkle on (modified in-place)
 * @param center Center position of sparkle in pixel coordinates
 * @param size Base radius of sparkle in pixels
 * @param color RGB color of sparkle (BGR format)
 * @param brightness Overall brightness multiplier [0.0-1.0]
 *
 * @implementation
 * - Draws 4 concentric layers with increasing radius and decreasing alpha
 * - Each layer uses distance-based falloff for smooth gradient
 * - For bright sparkles (brightness > 0.5), adds cross pattern at center
 * - All drawing operations include bounds checking
 */
// Helper function to draw a sparkle with glow effect
void drawSparkle(cv::Mat &img, cv::Point center, float size, cv::Vec3b color, float brightness) {
    if (center.x < 0 || center.x >= img.cols || center.y < 0 || center.y >= img.rows) {
        return;
    }
    
    // Draw glow layers
    for (int layer = 3; layer >= 0; layer--) {
        float layerSize = size + layer * 1.5f;
        float layerAlpha = brightness * (1.0f - layer * 0.25f);
        
        for (int dy = -layerSize; dy <= layerSize; dy++) {
            for (int dx = -layerSize; dx <= layerSize; dx++) {
                int px = center.x + dx;
                int py = center.y + dy;
                
                if (px >= 0 && px < img.cols && py >= 0 && py < img.rows) {
                    float dist = std::sqrt(dx*dx + dy*dy);
                    if (dist <= layerSize) {
                        float alpha = layerAlpha * (1.0f - dist / layerSize);
                        cv::Vec3b &pixel = img.at<cv::Vec3b>(py, px);
                        
                        for (int c = 0; c < 3; c++) {
                            float blended = pixel[c] * (1.0f - alpha) + color[c] * alpha;
                            pixel[c] = static_cast<uchar>(std::min(255.0f, blended));
                        }
                    }
                }
            }
        }
    }
    
    // Draw bright center cross pattern
    if (brightness > 0.5f) {
        int crossLen = static_cast<int>(size * 1.5f);
        cv::line(img, 
                cv::Point(center.x - crossLen, center.y),
                cv::Point(center.x + crossLen, center.y),
                cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
        cv::line(img, 
                cv::Point(center.x, center.y - crossLen),
                cv::Point(center.x, center.y + crossLen),
                cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
    }
}

/**
 * @brief Applies animated sparkle effect around detected faces with 3D depth sorting.
 *
 * This is the main sparkle effect function that creates magical animated particles
 * orbiting around faces. The effect achieves realistic 3D appearance by:
 * 1. Computing 3D positions using elliptical orbit mathematics
 * 2. Sorting sparkles by z-depth into behind/front groups
 * 3. Drawing back sparkles first (smaller, darker) then front sparkles (larger, brighter)
 * 4. Applying time-based animation and pulsing effects
 *
 * The 3D illusion is created using parallax and depth cues - sparkles further from
 * the viewer (negative z) appear smaller and dimmer, while closer sparkles (positive z)
 * appear larger and brighter.
 *
 * @param src Source image (must be 3-channel BGR color image)
 * @param dst Destination image with sparkle effect (cloned from source then modified)
 * @param faces Vector of detected face rectangles (from face detection)
 * @param sparkles Persistent sparkle state data (auto-initialized if size mismatches faces)
 * @param time Current animation time in seconds (continuously increasing for smooth animation)
 * @return 0 on success, -1 on error (invalid image format)
 *
 * @implementation
 * Algorithm steps:
 * 1. Validate input and clone source to destination
 * 2. Initialize sparkles if needed (faces count changed)
 * 3. For each face:
 *    a. Calculate face center point
 *    b. Update sparkle angles based on time and individual speeds
 *    c. Compute 3D positions: x = r*cos(θ), y = r*sin(θ)*cos(φ), z = r*sin(θ)*sin(φ)*0.7
 *    d. Separate sparkles into behind (z < 0) and front (z ≥ 0) groups
 *    e. Sort each group by z-depth
 *    f. Draw behind sparkles with reduced size/brightness (depth factor 0.3-0.7)
 *    g. Draw front sparkles with enhanced size/brightness (depth factor 0.7-1.0)
 * 4. Apply pulsing effect: brightness *= 0.8 + 0.2*sin(time*3 + phase)
 *
 * @note Ellipse ratio of 0.7 creates the illusion of tilted circular orbits in 3D space
 * @note Each sparkle maintains its own phase offset for orbit tilt variation
 */
int sparkleEffect(cv::Mat &src, cv::Mat &dst, 
                  const std::vector<cv::Rect> &faces,
                  std::vector<std::vector<Sparkle>> &sparkles,
                  float time) {
    if (src.empty() || src.channels() != 3) {
        std::cerr << "Error: sparkleEffect requires 3-channel image" << std::endl;
        return -1;
    }
    
    // Start with original image
    dst = src.clone();
    
    if (faces.empty()) {
        return 0;
    }
    
    // Initialize sparkles if needed
    if (sparkles.size() != faces.size()) {
        initializeSparkles(sparkles, faces);
    }
    
    // Process each face
    for (size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++) {
        const cv::Rect &face = faces[faceIdx];
        std::vector<Sparkle> &faceSparkles = sparkles[faceIdx];
        
        cv::Point faceCenter(face.x + face.width / 2, face.y + face.height / 2);
        
        // Separate sparkles into behind and in-front groups
        std::vector<std::pair<float, int>> behindSparkles;  // (z-depth, index)
        std::vector<std::pair<float, int>> frontSparkles;
        
        for (size_t i = 0; i < faceSparkles.size(); i++) {
            Sparkle &s = faceSparkles[i];
            
            // Update angle based on time
            s.angle += s.speed * 0.02f;
            if (s.angle > 2.0f * M_PI) {
                s.angle -= 2.0f * M_PI;
            }
            
            // Calculate 3D position
            // Use elliptical orbit with depth component
            float ellipseRatio = 0.7f;  // Makes orbit appear elliptical (3D effect)
            float tiltAngle = s.phase;  // Each sparkle has different orbit tilt
            
            // Position in 3D space
            float x = s.radius * std::cos(s.angle);
            float y = s.radius * std::sin(s.angle) * std::cos(tiltAngle);
            float z = s.radius * std::sin(s.angle) * std::sin(tiltAngle) * ellipseRatio;
            
            // Sort by z-depth
            if (z < 0) {
                behindSparkles.push_back(std::make_pair(z, i));
            } else {
                frontSparkles.push_back(std::make_pair(z, i));
            }
        }
        
        // Sort by z-depth (furthest first)
        std::sort(behindSparkles.begin(), behindSparkles.end());
        std::sort(frontSparkles.begin(), frontSparkles.end());
        
        // Draw sparkles behind face first (darker/smaller)
        for (const auto &pair : behindSparkles) {
            Sparkle &s = faceSparkles[pair.second];
            
            float x = s.radius * std::cos(s.angle);
            float y = s.radius * std::sin(s.angle) * std::cos(s.phase);
            float z = s.radius * std::sin(s.angle) * std::sin(s.phase) * 0.7f;
            
            cv::Point sparklePos(
                faceCenter.x + static_cast<int>(x),
                faceCenter.y + static_cast<int>(y)
            );
            
            // Scale and fade based on depth
            float depthFactor = 0.3f + 0.4f * (z + s.radius) / (2.0f * s.radius);
            float adjustedSize = s.size * depthFactor;
            float brightness = 0.4f + 0.3f * depthFactor;
            
            // Pulsing effect
            brightness *= 0.8f + 0.2f * std::sin(time * 3.0f + s.phase);
            
            drawSparkle(dst, sparklePos, adjustedSize, s.color, brightness);
        }
        
        // Draw face region outline (optional - helps visualize depth)
        // You can comment this out if not needed
        // cv::ellipse(dst, faceCenter, 
        //            cv::Size(face.width/2, face.height/2),
        //            0, 0, 360, cv::Scalar(100, 100, 100), 1);
        
        // Draw sparkles in front of face (brighter/larger)
        for (const auto &pair : frontSparkles) {
            Sparkle &s = faceSparkles[pair.second];
            
            float x = s.radius * std::cos(s.angle);
            float y = s.radius * std::sin(s.angle) * std::cos(s.phase);
            float z = s.radius * std::sin(s.angle) * std::sin(s.phase) * 0.7f;
            
            cv::Point sparklePos(
                faceCenter.x + static_cast<int>(x),
                faceCenter.y + static_cast<int>(y)
            );
            
            // Scale and brighten based on depth
            float depthFactor = 0.7f + 0.3f * (z + s.radius) / (2.0f * s.radius);
            float adjustedSize = s.size * depthFactor;
            float brightness = 0.7f + 0.3f * depthFactor;
            
            // Pulsing effect
            brightness *= 0.8f + 0.2f * std::sin(time * 3.0f + s.phase);
            
            drawSparkle(dst, sparklePos, adjustedSize, s.color, brightness);
        }
    }
    
    return 0;
}

// Utility functions

std::string generateTimestampFilename(const std::string &prefix, 
                                      const std::string &extension) {
    time_t now = time(0);
    tm* ltm = localtime(&now);
    
    char buffer[100];
    sprintf(buffer, "%s_%04d%02d%02d_%02d%02d%02d%s",
            prefix.c_str(),
            1900 + ltm->tm_year,
            1 + ltm->tm_mon,
            ltm->tm_mday,
            ltm->tm_hour,
            ltm->tm_min,
            ltm->tm_sec,
            extension.c_str());
    
    return std::string(buffer);
}