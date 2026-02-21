/**
 * @file    Threshold.h
 * @brief   Thresholding algorithms to separate objects from background.
 *
 *          Supports three modes controlled by PipelineParams:
 *            1. Global fixed threshold
 *            2. Adaptive (local) threshold
 *            3. ISODATA / k-means dynamic threshold (k=2)
 *
 *          Objects are assumed darker than a light background.
 *          Output is always a binary image (0 = background, 255 = object).
 *
 * @author  Krushna Sanjay Sharma
 * @date    February 2026
 */

#pragma once

#include <opencv2/opencv.hpp>
#include "AppState.h"

/**
 * @brief Apply pre-processing blur to reduce noise before thresholding.
 *
 * @param src    Input colour frame.
 * @param dst    Output blurred frame (same type as src).
 * @param params Pipeline parameters (blurKernelSize used).
 */
void applyBlur(const cv::Mat& src, cv::Mat& dst, const PipelineParams& params);

/**
 * @brief Convert frame to grayscale.
 *
 * @param src  Input colour frame (BGR).
 * @param dst  Output single-channel grayscale frame.
 */
void toGrayscale(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Compute a dynamic threshold value using ISODATA (k-means, k=2).
 *
 *        Samples 1/16 of pixels at random, runs two-cluster k-means,
 *        and returns the midpoint between the two cluster means.
 *        Useful when lighting changes between sessions.
 *
 * @param gray   Single-channel grayscale image.
 * @return       Suggested threshold value [0..255].
 */
int computeISODATAThreshold(const cv::Mat& gray);

/**
 * @brief Custom threshold using saturation and intensity channels (HSV).
 *
 *        Strongly coloured pixels (high saturation) are treated as darker,
 *        moving them away from the unsaturated white background.
 *        Combined score per pixel:  score = (1 - S) * V
 *        where S = saturation [0..1], V = value/intensity [0..1].
 *        White background → low S, high V → high score → background.
 *        Dark/coloured object → high S or low V → low score → foreground.
 *        Pixels with score below threshold → foreground (255).
 *
 * @param src      Input colour frame (BGR).
 * @param dst      Output binary mask (CV_8UC1).
 * @param params   Pipeline parameters (thresholdValue used as threshold).
 */
void applyCustomSatIntensityThreshold(const cv::Mat& src, cv::Mat& dst,
                                      const PipelineParams& params);

/**
 * @brief Apply thresholding to produce a binary object mask.
 *
 *        Mode selected via params:
 *          - useSatIntensity=true → custom saturation+intensity threshold
 *          - useKMeans=true       → ISODATA dynamic threshold
 *          - useAdaptive=true     → cv::adaptiveThreshold
 *          - default              → global cv::threshold
 *
 *        Pixels below threshold → 255 (object), above → 0 (background).
 *
 * @param src    Input colour frame (BGR).
 * @param dst    Output binary mask (CV_8UC1).
 * @param params Pipeline parameters.
 */
void applyThreshold(const cv::Mat& src, cv::Mat& dst,
                    const PipelineParams& params);