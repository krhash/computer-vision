/*
  Author: [Your Name]
  Date: January 24, 2026
  Purpose: Header file for image filtering and manipulation functions.
           Contains declarations for greyscale, blur, Sobel, and special effects filters.
*/

#ifndef FILTERS_HPP
#define FILTERS_HPP

#include <opencv2/opencv.hpp>
#include <string>

// Task 4: Alternative greyscale conversion
/*
  Function: greyscale
  Purpose: Convert a color image to greyscale using a custom algorithm
  Arguments:
    src - input color image (CV_8UC3)
    dst - output greyscale image (CV_8UC3, all channels identical)
  Return value: 0 on success, -1 on error
*/
int greyscale(cv::Mat &src, cv::Mat &dst);

// Task 5: Sepia tone filter
/*
  Function: sepiaTone
  Purpose: Apply a sepia tone filter to create an antique photo effect
  Arguments:
    src - input color image (CV_8UC3)
    dst - output sepia-toned image (CV_8UC3)
    applyVignetting - whether to apply vignetting effect (default: true)
  Return value: 0 on success, -1 on error
*/
int sepiaTone(cv::Mat &src, cv::Mat &dst, bool applyVignetting = true);

// Task 6: 5x5 Gaussian blur (naive implementation)
/*
  Function: blur5x5_naive
  Purpose: Apply 5x5 Gaussian blur using single nested loop (naive)
  Arguments:
    src - input color image (CV_8UC3)
    dst - output blurred image (CV_8UC3)
  Return value: 0 on success, -1 on error
*/
int blur5x5_naive(cv::Mat &src, cv::Mat &dst);

// Task 6: 5x5 Gaussian blur (optimized with separable filters)
/*
  Function: blur5x5_2
  Purpose: Apply 5x5 Gaussian blur using separable 1x5 filters (optimized)
  Arguments:
    src - input color image (CV_8UC3)
    dst - output blurred image (CV_8UC3)
  Return value: 0 on success, -1 on error
*/
int blur5x5_sep(cv::Mat &src, cv::Mat &dst);

// Task 7: Sobel X filter
/*
  Function: sobelX3x3
  Purpose: Apply 3x3 Sobel X filter for vertical edge detection
  Arguments:
    src - input color image (CV_8UC3)
    dst - output Sobel X image (CV_16SC3, signed short)
  Return value: 0 on success, -1 on error
*/
int sobelX3x3(cv::Mat &src, cv::Mat &dst);

// Task 7: Sobel Y filter
/*
  Function: sobelY3x3
  Purpose: Apply 3x3 Sobel Y filter for horizontal edge detection
  Arguments:
    src - input color image (CV_8UC3)
    dst - output Sobel Y image (CV_16SC3, signed short)
  Return value: 0 on success, -1 on error
*/
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

// Task 8: Gradient magnitude
/*
  Function: magnitude
  Purpose: Calculate gradient magnitude from Sobel X and Y images
  Arguments:
    sx - Sobel X image (CV_16SC3)
    sy - Sobel Y image (CV_16SC3)
    dst - output magnitude image (CV_8UC3)
  Return value: 0 on success, -1 on error
*/
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// Task 9: Blur and quantize
/*
  Function: blurQuantize
  Purpose: Blur image and quantize colors into fixed number of levels
  Arguments:
    src - input color image (CV_8UC3)
    dst - output blurred and quantized image (CV_8UC3)
    levels - number of quantization levels per channel
  Return value: 0 on success, -1 on error
*/
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

/*
  Function: depthFog
  Purpose: Apply fog effect based on depth (distant objects become foggy)
  Arguments:
    src - input color image (CV_8UC3)
    depth - depth map (CV_8UC1, 0=close, 255=far)
    dst - output image with fog effect (CV_8UC3)
    intensity - fog intensity factor (higher = more fog, default 3.0)
  Return value: 0 on success, -1 on error
*/
int depthFog(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, float intensity = 3.0f);

/*
  Function: embossEffect
  Purpose: Create 3D embossed appearance using Sobel gradients
           Uses dot product with light direction for shading
  Arguments:
    src - input color image (CV_8UC3)
    dst - output embossed image (CV_8UC3)
  Return value: 0 on success, -1 on error
*/
int embossEffect(cv::Mat &src, cv::Mat &dst);

/*
  Function: negativeEffect
  Purpose: Create negative/inverse of the image
           Formula: output = 255 - input for each channel
  Arguments:
    src - input color image (CV_8UC3)
    dst - output negative image (CV_8UC3)
  Return value: 0 on success, -1 on error
*/
int negativeEffect(cv::Mat &src, cv::Mat &dst);

/*
  Function: faceHighlight
  Purpose: Keep detected faces in color while rest of image is grayscale
           Creates spotlight effect on faces with smooth blending
  Arguments:
    src - input color image (CV_8UC3)
    dst - output image with highlighted faces (CV_8UC3)
    faces - vector of detected face rectangles
  Return value: 0 on success, -1 on error
*/
int faceHighlight(cv::Mat &src, cv::Mat &dst, const std::vector<cv::Rect> &faces);

/*
  Function: cartoonEffect
  Purpose: Create cartoon-style rendering combining edges with quantized colors
  Arguments:
    src - input color image (CV_8UC3)
    dst - output cartoon image (CV_8UC3)
  Return value: 0 on success, -1 on error
*/
int cartoonEffect(cv::Mat &src, cv::Mat &dst);

/*
  Function: depthFocus
  Purpose: Depth-based focus effect (portrait mode) - blurs background
  Arguments:
    src - input color image (CV_8UC3)
    depth - depth map (CV_8UC1, 0=close, 255=far after inversion)
    dst - output image with focus effect (CV_8UC3)
    focusDepth - depth value to keep in focus (0-255), default 200 (close objects)
    focusRange - width of focus area, default 40
  Return value: 0 on success, -1 on error
*/
int depthFocus(cv::Mat &src, cv::Mat &depth, cv::Mat &dst, int focusDepth = 200, int focusRange = 40);

/*
  Function: bulgeEffect
  Purpose: Creates a bulge/pinch distortion centered on the image
           Bulge makes the center appear to push outward (like a bubble)
           Based on polar coordinate transformation with power function
  Arguments:
    src - input color image (CV_8UC3)
    dst - output warped image (CV_8UC3)
    strength - distortion strength (0.0-1.0 = bulge out, >1.0 = pinch in)
  Return value: 0 on success, -1 on error
  Reference: Digital Image Warping, George Wolberg, IEEE Computer Society Press, 1990
*/
int bulgeEffect(cv::Mat &src, cv::Mat &dst, float strength = 0.5f);

/*
  Function: waveEffect
  Purpose: Applies sinusoidal wave distortion to the image
           Creates a ripple/wave pattern across the image
           Uses sine function for horizontal and vertical displacement
  Arguments:
    src - input color image (CV_8UC3)
    dst - output warped image (CV_8UC3)
    amplitude - wave height in pixels (default 10.0)
    frequency - wave frequency, smaller = wider waves (default 0.05)
  Return value: 0 on success, -1 on error
  Reference: https://en.wikipedia.org/wiki/Image_distortion
*/
int waveEffect(cv::Mat &src, cv::Mat &dst, float amplitude = 10.0f, float frequency = 0.05f);

/*
  Function: swirlEffect
  Purpose: Creates a swirl/twirl distortion rotating pixels around center
           Rotation angle decreases with distance from center
           Uses polar coordinate rotation transformation
  Arguments:
    src - input color image (CV_8UC3)
    dst - output warped image (CV_8UC3)
    angle - maximum rotation in radians at center (default 2.0)
  Return value: 0 on success, -1 on error
  Reference: "A Parametric Texture Model Based on Joint Statistics of Complex Wavelet Coefficients"
             Portilla & Simoncelli, Int'l Journal of Computer Vision, 2000
*/
int swirlEffect(cv::Mat &src, cv::Mat &dst, float angle = 2.0f);

/*
  Function: faceBulgeEffect
  Purpose: Applies bulge distortion centered on each detected face
           Creates a "big head" or "funhouse mirror" effect on faces
           Combines face detection with local coordinate warping
  Arguments:
    src - input color image (CV_8UC3)
    dst - output warped image (CV_8UC3)
    faces - vector of detected face rectangles from face detection
    strength - bulge strength per face (0.0-1.0 = bulge, >1.0 = pinch)
  Return value: 0 on success, -1 on error
*/
int faceBulgeEffect(cv::Mat &src, cv::Mat &dst, const std::vector<cv::Rect> &faces, float strength = 0.4f);

// Utility functions
/*
  Function: generateTimestampFilename
  Purpose: Generate a unique filename with timestamp
  Arguments:
    prefix - filename prefix
    extension - file extension (e.g., ".jpg")
  Return value: generated filename string
*/
std::string generateTimestampFilename(const std::string &prefix, 
                                      const std::string &extension);

#endif // FILTERS_HPP
