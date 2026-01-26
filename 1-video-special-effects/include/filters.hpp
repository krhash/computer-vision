/*
  Author: Krushna Sanjay Sharma
  Date: January 24, 2026
  Purpose: Header file for image filtering and manipulation functions.
           Contains declarations for greyscale, blur, Sobel, and special effects filters.
           Includes sparkle animation effect for face detection.
*/

#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @struct Sparkle
 * @brief Represents a single animated sparkle particle that orbits in 3D space.
 *
 * Each sparkle maintains its own position, appearance, and animation parameters
 * to create varied motion around a detected face. Sparkles orbit in elliptical
 * paths with depth (z-axis) variation to create the illusion of moving in front
 * of and behind the face.
 */
struct Sparkle {
    float angle;        ///< Current angular position in radians (0 to 2π) around orbit
    float radius;       ///< Orbital radius distance from face center in pixels
    float phase;        ///< Phase offset for orbit tilt angle, creates 3D depth variation
    float size;         ///< Base size of sparkle in pixels (before depth scaling)
    cv::Vec3b color;    ///< RGB color of the sparkle (BGR format for OpenCV)
    float speed;        ///< Angular velocity multiplier for rotation speed
};

// ============================================================================
// TASK 4: CUSTOM GREYSCALE CONVERSION
// ============================================================================

/**
 * @brief Convert a color image to greyscale using a custom algorithm
 * 
 * Uses a custom greyscale conversion that emphasizes color differences:
 * grey = |R - B| + G/2
 * This creates a unique look that highlights edges and color boundaries.
 * 
 * @param src Input color image (CV_8UC3, BGR format)
 * @param dst Output greyscale image (CV_8UC3, all channels identical)
 * @return 0 on success, -1 on error
 */
int greyscale(cv::Mat &src, cv::Mat &dst);

// ============================================================================
// TASK 5: SEPIA TONE FILTER
// ============================================================================

/**
 * @brief Apply a sepia tone filter to create an antique photo effect
 * 
 * Applies standard sepia tone transformation matrix with optional vignetting.
 * Sepia formula:
 *   R' = 0.393*R + 0.769*G + 0.189*B
 *   G' = 0.349*R + 0.686*G + 0.168*B
 *   B' = 0.272*R + 0.534*G + 0.131*B
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output sepia-toned image (CV_8UC3)
 * @param applyVignetting If true, adds darkening at edges (default: true)
 * @return 0 on success, -1 on error
 */
int sepiaTone(cv::Mat &src, cv::Mat &dst, bool applyVignetting = true);

// ============================================================================
// TASK 6: GAUSSIAN BLUR FILTERS
// ============================================================================

/**
 * @brief Apply 5x5 Gaussian blur using single nested loop (naive approach)
 * 
 * Uses 2D convolution with full 5x5 kernel. Slower but straightforward.
 * Gaussian kernel (integer approximation):
 *   [1  2  4  2  1]
 *   [2  4  8  4  2]
 *   [4  8 16  8  4]
 *   [2  4  8  4  2]
 *   [1  2  4  2  1]
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output blurred image (CV_8UC3)
 * @return 0 on success, -1 on error
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply 5x5 Gaussian blur using separable 1D filters (optimized)
 * 
 * Separates 2D convolution into two 1D passes for efficiency.
 * 1D kernel: [1 2 4 2 1]
 * First pass: horizontal blur
 * Second pass: vertical blur on result
 * Approximately 2-3x faster than naive approach.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output blurred image (CV_8UC3)
 * @return 0 on success, -1 on error
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

// ============================================================================
// TASK 7: SOBEL EDGE DETECTION FILTERS
// ============================================================================

/**
 * @brief Apply 3x3 Sobel X filter for vertical edge detection
 * 
 * Detects vertical edges (brightness changes in horizontal direction).
 * Implemented as separable filter:
 *   Horizontal: [-1, 0, 1] (derivative)
 *   Vertical:   [1, 2, 1] (smoothing)
 * 
 * Combined kernel:
 *   [-1  0  +1]
 *   [-2  0  +2]
 *   [-1  0  +1]
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output Sobel X gradient (CV_16SC3, signed short for negative values)
 * @return 0 on success, -1 on error
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Apply 3x3 Sobel Y filter for horizontal edge detection
 * 
 * Detects horizontal edges (brightness changes in vertical direction).
 * Implemented as separable filter:
 *   Horizontal: [1, 2, 1] (smoothing)
 *   Vertical:   [1, 0, -1] (derivative)
 * 
 * Combined kernel:
 *   [+1  +2  +1]
 *   [ 0   0   0]
 *   [-1  -2  -1]
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output Sobel Y gradient (CV_16SC3, signed short for negative values)
 * @return 0 on success, -1 on error
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// ============================================================================
// TASK 8: GRADIENT MAGNITUDE
// ============================================================================

/**
 * @brief Calculate gradient magnitude from Sobel X and Y images
 * 
 * Combines horizontal and vertical gradients using Pythagorean theorem:
 *   magnitude = sqrt(sobelX² + sobelY²)
 * 
 * This produces edge detection independent of edge orientation.
 * Edges in any direction (vertical, horizontal, diagonal) are detected.
 * 
 * @param sx Sobel X gradient image (CV_16SC3)
 * @param sy Sobel Y gradient image (CV_16SC3)
 * @param dst Output magnitude image (CV_8UC3)
 * @return 0 on success, -1 on error
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// ============================================================================
// TASK 9: BLUR AND QUANTIZE
// ============================================================================

/**
 * @brief Blur image and quantize colors into fixed number of levels
 * 
 * Two-step process:
 * 1. Blur using 5x5 separable Gaussian filter
 * 2. Quantize color values into discrete levels
 * 
 * Quantization formula:
 *   bucketSize = 255 / levels
 *   quantized = (value / bucketSize) * bucketSize
 * 
 * Creates posterization effect with smooth color transitions.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output blurred and quantized image (CV_8UC3)
 * @param levels Number of quantization levels per channel (1-255)
 * @return 0 on success, -1 on error
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

// ============================================================================
// TASK 11/12: DEPTH-BASED EFFECTS
// ============================================================================

/**
 * @brief Apply fog effect based on depth map
 * 
 * Simulates atmospheric fog where distant objects become obscured.
 * Uses exponential fog model:
 *   fogFactor = 1 - exp(-intensity * depth²)
 *   output = (1-fogFactor)*original + fogFactor*fogColor
 * 
 * Depth map interpretation: 0=close, 255=far
 * 
 * @param src Input color image (CV_8UC3)
 * @param depth Depth map (CV_8UC1, 0=close, 255=far)
 * @param dst Output image with fog effect (CV_8UC3)
 * @param intensity Fog intensity factor, higher = more fog (default: 3.0)
 * @return 0 on success, -1 on error
 */
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, float intensity = 3.0f);

/**
 * @brief Depth-based focus effect simulating portrait mode
 * 
 * Keeps objects at specified depth in focus while blurring background/foreground.
 * Simulates shallow depth of field effect from DSLR cameras.
 * 
 * Algorithm:
 * 1. Create heavily blurred version of image
 * 2. Calculate blur amount based on distance from focus depth
 * 3. Blend sharp and blurred versions using smooth transition
 * 
 * @param src Input color image (CV_8UC3)
 * @param depth Depth map (CV_8UC1)
 * @param dst Output image with focus effect (CV_8UC3)
 * @param focusDepth Target depth to keep in focus (0-255, default: 200)
 * @param focusRange Width of focus area in depth units (default: 40)
 * @return 0 on success, -1 on error
 */
int depthFocus(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, int focusDepth = 200, int focusRange = 40);

// ============================================================================
// TASK 12: SPECIAL EFFECTS
// ============================================================================

/**
 * @brief Create 3D embossed appearance using Sobel gradients
 * 
 * Simulates relief/embossing by computing surface normals from gradients
 * and shading based on virtual light direction.
 * 
 * Algorithm:
 * 1. Compute Sobel X and Y gradients
 * 2. Calculate dot product with light direction (45° from top-left)
 * 3. Add to neutral gray (128) for emboss effect
 * 
 * Light direction: (0.7071, 0.7071) = 45° diagonal
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output embossed image (CV_8UC3)
 * @return 0 on success, -1 on error
 */
int embossEffect(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Create negative/inverse of the image
 * 
 * Simple pixel-wise inversion:
 *   output = 255 - input for each channel
 * 
 * Creates photographic negative effect.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output negative image (CV_8UC3)
 * @return 0 on success, -1 on error
 */
int negativeEffect(cv::Mat &src, cv::Mat &dst);

/**
 * @brief Keep detected faces in color while rest of image is grayscale
 * 
 * Creates spotlight effect on faces with smooth alpha blending.
 * 
 * Algorithm:
 * 1. Convert entire image to grayscale
 * 2. Create soft mask for face regions (ellipses)
 * 3. Blur mask for smooth transition
 * 4. Blend color faces with grayscale background
 * 
 * Face regions expanded 12.5% for better coverage.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output image with highlighted faces (CV_8UC3)
 * @param faces Vector of detected face rectangles
 * @return 0 on success, -1 on error
 */
int faceHighlight(cv::Mat &src, cv::Mat &dst, const std::vector<cv::Rect> &faces);

/**
 * @brief Create cartoon-style rendering
 * 
 * Combines edge detection with color quantization for cartoon appearance.
 * 
 * Algorithm:
 * 1. Extract edges using gradient magnitude
 * 2. Threshold and dilate edges
 * 3. Quantize colors (8 levels)
 * 4. Darken edge pixels to create outlines
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output cartoon image (CV_8UC3)
 * @return 0 on success, -1 on error
 */
int cartoonEffect(cv::Mat &src, cv::Mat &dst);

// ============================================================================
// EXTENSION: WARP EFFECTS
// ============================================================================

/**
 * @brief Creates bulge/pinch distortion centered on image
 * 
 * Applies radial distortion using power function in polar coordinates.
 * 
 * Mathematical basis:
 *   r' = r^exponent
 *   where exponent = 1/(strength + 0.5)
 * 
 * strength < 1.0: bulge outward (fisheye effect)
 * strength > 1.0: pinch inward
 * 
 * Uses bilinear interpolation for smooth result.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output warped image (CV_8UC3)
 * @param strength Distortion strength (0.1-1.0 = bulge, >1.0 = pinch)
 * @return 0 on success, -1 on error
 * @see Digital Image Warping, George Wolberg (1990)
 */
int bulgeEffect(cv::Mat &src, cv::Mat &dst, float strength = 0.5f);

/**
 * @brief Applies sinusoidal wave distortion to image
 * 
 * Creates ripple pattern using sine function displacement.
 * 
 * Displacement:
 *   x' = x + amplitude * sin(frequency * y * 2π)
 *   y' = y + amplitude * sin(frequency * x * 2π)
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output warped image (CV_8UC3)
 * @param amplitude Wave height in pixels (default: 10.0)
 * @param frequency Wave frequency, smaller = wider waves (default: 0.05)
 * @return 0 on success, -1 on error
 */
int waveEffect(cv::Mat &src, cv::Mat &dst, float amplitude = 10.0f, float frequency = 0.05f);

/**
 * @brief Creates swirl/twirl distortion rotating around center
 * 
 * Rotation angle decreases with distance from center.
 * 
 * Transformation:
 *   θ' = θ + angle * (1 - r/maxRadius)
 * 
 * Center pixels rotate most, edge pixels least.
 * Similar to Photoshop's Twirl filter.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output warped image (CV_8UC3)
 * @param angle Maximum rotation at center in radians (default: 2.0)
 * @return 0 on success, -1 on error
 */
int swirlEffect(cv::Mat &src, cv::Mat &dst, float angle = 2.0f);

/**
 * @brief Applies bulge distortion centered on each detected face
 * 
 * Creates "big head" caricature effect on faces.
 * Combines face detection with local coordinate warping.
 * 
 * Each face gets independent bulge transformation.
 * Useful for humorous photo effects.
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output warped image (CV_8UC3)
 * @param faces Vector of detected face rectangles
 * @param strength Bulge strength (0.1-1.0 = bulge, >1.0 = pinch)
 * @return 0 on success, -1 on error
 */
int faceBulgeEffect(cv::Mat &src, cv::Mat &dst, const std::vector<cv::Rect> &faces, float strength = 0.4f);

// ============================================================================
// EXTENSION: SPARKLE ANIMATION EFFECT
// ============================================================================

/**
 * @brief Creates animated sparkles orbiting detected faces in 3D space
 * 
 * Generates magical particle effect with proper depth sorting.
 * Sparkles appear to move in front of and behind faces.
 * 
 * Algorithm:
 * 1. Update sparkle positions using time-based animation
 * 2. Compute 3D coordinates: x = r*cos(θ), y = r*sin(θ)*cos(φ), z = r*sin(θ)*sin(φ)
 * 3. Separate into behind (z<0) and front (z≥0) groups
 * 4. Sort by z-depth
 * 5. Draw back sparkles (darker, smaller)
 * 6. Draw front sparkles (brighter, larger)
 * 7. Apply pulsing animation
 * 
 * Depth cues:
 * - Behind: 30-70% size/brightness
 * - Front: 70-100% size/brightness
 * 
 * @param src Input color image (CV_8UC3)
 * @param dst Output image with sparkle effect (CV_8UC3)
 * @param faces Vector of detected face rectangles
 * @param sparkles Persistent state (auto-initialized on face count change)
 * @param time Animation time in seconds (continuously increasing)
 * @return 0 on success, -1 on error
 * 
 * @note Maintains state between frames for smooth animation
 * 
 * Example usage:
 * @code
 *   vector<vector<Sparkle>> sparkles;
 *   float time = 0.0f;
 *   while (capturing) {
 *       detectFaces(grey, faces);
 *       sparkleEffect(frame, output, faces, sparkles, time);
 *       time += 0.033f; // ~30 FPS
 *   }
 * @endcode
 */
int sparkleEffect(cv::Mat &src, cv::Mat &dst, 
                  const std::vector<cv::Rect> &faces,
                  std::vector<std::vector<Sparkle>> &sparkles,
                  float time);

/**
 * @brief Initialize sparkle particles for detected faces
 * 
 * Creates sparkles with randomized properties for natural motion.
 * 
 * Randomized properties per sparkle:
 * - Angle: Evenly distributed 0 to 2π
 * - Radius: 90-120% of base face size
 * - Phase: Random orbit tilt (0 to 2π)
 * - Size: 3-8 pixels
 * - Speed: 0.8-1.2x base speed
 * - Color: Gold, white, light blue, or pink
 * 
 * Base radius = 60% of max(face width, face height)
 * 
 * @param sparkles Output vector, resized to match face count
 * @param faces Vector of face rectangles
 * @param numSparkles Sparkles per face (default: 12)
 * 
 * @note Called automatically by sparkleEffect() when face count changes
 */
void initializeSparkles(std::vector<std::vector<Sparkle>> &sparkles, 
                       const std::vector<cv::Rect> &faces,
                       int numSparkles = 12);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Generate unique filename with timestamp
 * 
 * Creates filename in format: prefix_YYYYMMDD_HHMMSS.extension
 * Example: "frame_20260124_153045.jpg"
 * 
 * Uses local time for timestamp.
 * 
 * @param prefix Filename prefix (e.g., "frame", "image")
 * @param extension File extension with dot (e.g., ".jpg", ".png")
 * @return Complete filename string with timestamp
 */
std::string generateTimestampFilename(const std::string &prefix, 
                                      const std::string &extension);

#endif // FILTERS_HPP