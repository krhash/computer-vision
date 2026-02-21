/**
 * @file    Morphology.h
 * @brief   Morphological filtering to clean binary images — written from scratch.
 *
 *          Implements erosion and dilation using a flat rectangular structuring
 *          element, then composes them into open/close operations.
 *
 *          Why from scratch:
 *            Satisfies the solo-developer requirement to implement at least
 *            one of the first four pipeline stages without OpenCV morphology
 *            functions (cv::erode / cv::dilate / cv::morphologyEx).
 *
 *          Operations:
 *            Erode  — shrinks foreground; removes small noise blobs
 *            Dilate — grows foreground; fills small holes
 *            Open   — erode then dilate; removes noise, preserves shape
 *            Close  — dilate then erode; fills holes, preserves shape
 *
 *          Input/output: CV_8UC1 binary images (0 = background, 255 = object).
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include "AppState.h"

/**
 * @brief Erode a binary image using a rectangular structuring element.
 *
 *        A foreground pixel (255) survives only if ALL pixels under the
 *        kernel are also foreground.  Shrinks object boundaries and
 *        removes isolated noise pixels smaller than the kernel.
 *
 * @param src      Input binary image  (CV_8UC1, 0/255).
 * @param dst      Output binary image (CV_8UC1, 0/255).
 * @param kSize    Side length of the square structuring element (odd, >= 1).
 * @param iters    Number of times to apply erosion.
 */
void erodeCustom(const cv::Mat& src, cv::Mat& dst, int kSize, int iters = 1);

/**
 * @brief Dilate a binary image using a rectangular structuring element.
 *
 *        A background pixel (0) becomes foreground (255) if ANY pixel
 *        under the kernel is foreground.  Grows object boundaries and
 *        fills small holes inside objects.
 *
 * @param src      Input binary image  (CV_8UC1, 0/255).
 * @param dst      Output binary image (CV_8UC1, 0/255).
 * @param kSize    Side length of the square structuring element (odd, >= 1).
 * @param iters    Number of times to apply dilation.
 */
void dilateCustom(const cv::Mat& src, cv::Mat& dst, int kSize, int iters = 1);

/**
 * @brief Apply morphological filtering to a binary image.
 *
 *        Mode is selected via params.morphMode:
 *          0 = Open  (erode → dilate) — best for removing speckle noise
 *          1 = Close (dilate → erode) — best for filling holes in objects
 *          2 = Erode only
 *          3 = Dilate only
 *
 * @param src      Input binary image  (CV_8UC1).
 * @param dst      Output binary image (CV_8UC1).
 * @param params   Pipeline parameters (morphMode, morphKernelSize, morphIterations).
 */
void applyMorphology(const cv::Mat& src, cv::Mat& dst, const PipelineParams& params);
